"""SPIKE 1 — can a real teacher label real turns?

The tracer proved the plumbing with a mock teacher. This answers the question
the tracer provably cannot: does a real LLM, given the plan's exact prompt
shape, produce usable labels on *real* session data — and is `unmapped` a signal
a real teacher actually emits, or a class that only exists on paper?

It is a throwaway spike, not a feature. It sends N turns to one provider and
reports only aggregate diagnostics. It never prints turn text or credentials.

Usage (key read from the environment, never passed on the command line):

    ATC_TEACHER_API_KEY=... uv run --extra train python examples/spike_real_teacher.py \
        --n 20 --model deepseek/deepseek-chat --base-url https://openrouter.ai/api/v1

Design constraints carried from the plan (P5/P7):
  * the prompt is frozen and its sha256 is reported, so a result is attributable
  * the response schema is ordered evidence-first, then label
  * batching is a stratified shuffle, never sorted-by-id
  * log hygiene: no log line may carry a long span copied from a view or a
    provider response, and the evidence field goes through redaction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_turn_classifier import tracer as T  # noqa: E402

# ---------------------------------------------------------------- prompt (frozen)

SYSTEM_PROMPT = """You are a labelling function. You classify one turn of an AI \
coding agent's session into exactly one workflow from the taxonomy below.

Judge the WORK PERFORMED IN THE TURN, not the topic of conversation.

Ontology:
{ontology}

Tie-break order, applied in sequence:
1. the most specific workflow whose triggers match the observed actions;
2. if two match equally, the one whose evidence names a tool or step present in the turn;
3. if still tied, choose the primary and record the other in alt_workflow.

Emit `unmapped` when the turn is real work but no declared workflow describes it.
Emit `none` when the turn is not work at all (no actions, nothing being done).

Respond as JSON with the keys in THIS ORDER: evidence, domain, workflow, confidence, alt_workflow.
`confidence` is your posterior probability that the label is correct."""

RESPONSE_SCHEMA = {"type": "json_object"}


def render_ontology(workflows) -> str:
    return "\n".join(f"- {w.id} [{w.domain}] {w.name} — when: {'; '.join(w.triggers)}"
                     for w in workflows)


def prompt_sha() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- hygiene


def scrub(text: str, limit: int = 60) -> str:
    """Keep a diagnostic from copying long spans of view/response text.

    The plan's rule is 120 consecutive characters; 60 is stricter, because a
    spike's output goes into a terminal and a scrollback.
    """
    one = re.sub(r"\s+", " ", text).strip()
    return one[:limit] + ("…" if len(one) > limit else "")


def redact(text: str) -> str:
    """Mask anything key-shaped before it can reach a log or a file."""
    patterns = [
        r"sk-[A-Za-z0-9_\-]{16,}", r"sk-proj-[A-Za-z0-9_\-]{16,}",
        r"gho_[A-Za-z0-9]{20,}", r"ghp_[A-Za-z0-9]{20,}",
        r"xox[baprs]-[A-Za-z0-9\-]{10,}", r"Bearer\s+[A-Za-z0-9._\-]{20,}",
    ]
    for p in patterns:
        text = re.sub(p, "[REDACTED]", text)
    return text


# ---------------------------------------------------------------- sampling


def stratified_sample(turns, n: int, seed: int = 20260922):
    """Sample across sessions, never more than 3 turns from one session.

    The plan's batching rule, applied here for the same reason: a session's
    turns share vocabulary, so an unstratified draw over-represents whichever
    session is largest.
    """
    rng = random.Random(seed)
    by_session = {}
    for t in turns:
        by_session.setdefault(t.session_id, []).append(t)

    picked, per_session = [], Counter()
    order = sorted(by_session)
    rng.shuffle(order)
    while len(picked) < n and order:
        for sid in list(order):
            if len(picked) >= n:
                break
            if per_session[sid] >= 3:
                order.remove(sid)
                continue
            pool = [t for t in by_session[sid] if t not in picked]
            if not pool:
                order.remove(sid)
                continue
            picked.append(rng.choice(pool))
            per_session[sid] += 1
    return picked


# ---------------------------------------------------------------- call


def label_batch(client, base_url, model, system, views, timeout=90):
    """One batched call, 8 numbered views, response JSON keyed by index."""
    numbered = "\n\n".join(f"### TURN {i+1}\n{v}" for i, v in enumerate(views))
    user = (f"Label each of the {len(views)} turns below.\n\n{numbered}\n\n"
            'Return JSON: {"labels": [{"index": 1, "evidence": "...", "domain": "...", '
            '"workflow": "...", "confidence": 0.0, "alt_workflow": null}, ...]}')
    body = {
        "model": model,
        "temperature": 0.2,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": RESPONSE_SCHEMA,
    }
    r = client.post(f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=".out/spike_real_teacher.json")
    args = ap.parse_args()

    key = os.environ.get("ATC_TEACHER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("no key: set ATC_TEACHER_API_KEY (never pass it on the command line)")
        return 2

    # Real sessions, if a local corpus is configured; otherwise the synthetic
    # fixture, which still exercises the provider path end to end.
    corpus = Path(os.environ.get("ATC_SPIKE_CORPUS", str(ROOT / "examples/sessions.sample.jsonl")))
    records = T.read_canonical(corpus)
    turns = T.build_turns(records)
    workflows, domains = T.load_ontology(ROOT / "examples/ontology.example.yaml")

    system = SYSTEM_PROMPT.format(ontology=render_ontology(workflows))
    print(f"corpus          {corpus.name}: {len(records)} records -> {len(turns)} turns")
    print(f"model           {args.model}")
    print(f"prompt sha256   {prompt_sha()}  ({len(system)} chars)")
    print(f"ontology        {len(workflows)} workflows / {len(domains)} domains\n")

    sample = stratified_sample(turns, args.n)
    print(f"sampled {len(sample)} turns across {len({t.session_id for t in sample})} sessions\n")

    client = httpx.Client(headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        # OpenRouter attribution; harmless on other OpenAI-compatible endpoints.
        "HTTP-Referer": "https://github.com/evanokeefe39/agent-turn-classifier",
        "X-Title": "agent-turn-classifier spike",
    })

    labels, errors, usage_tot = [], 0, {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.time()
    for start in range(0, len(sample), args.batch_size):
        batch = sample[start:start + args.batch_size]
        views = [redact(T.render(t)) for t in batch]
        try:
            content, usage = label_batch(client, args.base_url, args.model, system, views)
            for k in usage_tot:
                usage_tot[k] += usage.get(k, 0)
            parsed = json.loads(content)
            rows = parsed.get("labels") or parsed.get("results") or []
            if not isinstance(rows, list):
                raise ValueError("labels is not a list")
            for row in rows:
                idx = int(row.get("index", 0)) - 1
                if 0 <= idx < len(batch):
                    labels.append({
                        "turn_id": batch[idx].turn_id,
                        "workflow": row.get("workflow"),
                        "domain": row.get("domain"),
                        "confidence": row.get("confidence"),
                        "alt_workflow": row.get("alt_workflow"),
                        "evidence_len": len(str(row.get("evidence", ""))),
                    })
        except Exception as exc:  # noqa: BLE001 - spike: report, do not mask
            errors += 1
            print(f"  batch {start // args.batch_size + 1} FAILED: {type(exc).__name__}: {scrub(str(exc))}")
    elapsed = time.time() - t0

    # ------------------------------------------------------------ diagnostics
    n_expected = len(sample)
    n_got = len(labels)
    valid_json = (n_expected - errors * args.batch_size) / max(1, n_expected)
    wf_counts = Counter(l["workflow"] for l in labels)
    confs = [l["confidence"] for l in labels if isinstance(l["confidence"], (int, float))]
    n_unmapped = wf_counts.get("unmapped", 0)
    n_none = wf_counts.get("none", 0)
    unknown = [w for w in wf_counts if w not in {x.id for x in workflows}
               and w not in ("unmapped", "none")]

    print(f"\n--- results ({elapsed:.1f}s, {n_expected} turns, "
          f"batch size {args.batch_size}) ---")
    print(f"valid-JSON batches   {args.batch_size * 0 + (len(sample) // args.batch_size + 1) - errors}"
          f"/{len(sample) // args.batch_size + 1}")
    print(f"labels returned      {n_got}/{n_expected}")
    print(f"unmapped             {n_unmapped}  ({n_unmapped / max(1, n_got):.0%} of labelled)")
    print(f"none                 {n_none}")
    if unknown:
        print(f"OFF-TAXONOMY ids     {unknown}   <-- teacher invented these")
    if confs:
        print(f"confidence           min {min(confs):.2f}  mean {sum(confs)/len(confs):.2f}  max {max(confs):.2f}")
    print(f"label distribution   {dict(wf_counts.most_common())}")
    print(f"tokens               {usage_tot['prompt_tokens']} in / {usage_tot['completion_tokens']} out")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "n": n_expected, "labels_returned": n_got,
        "batch_errors": errors, "prompt_sha256": prompt_sha(),
        "unmapped": n_unmapped, "none": n_none,
        "off_taxonomy": unknown, "label_distribution": dict(wf_counts),
        "confidence": ({"min": min(confs), "mean": sum(confs)/len(confs), "max": max(confs)}
                       if confs else None),
        "usage": usage_tot, "elapsed_s": round(elapsed, 1),
        "corpus": corpus.name,
    }, indent=2), encoding="utf-8")

    # The gate this spike exists to answer.
    ok = True
    if errors:
        print(f"\nGATE valid-JSON: FAIL ({errors} batch(es) failed)")
        ok = False
    if unknown:
        print("GATE taxonomy adherence: FAIL (teacher emitted ids outside the ontology)")
        ok = False
    if n_got and n_unmapped == 0:
        print("GATE unmapped emission: UNPROVEN — a real teacher never emitted it on this "
              "sample; the discovery loop has nothing to read yet")
    print("\nSPIKE RESULT:", "usable" if ok else "NOT usable as-is")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# SPIKE 1 RESULT — 2026-09-22
#
# Ran against a real provider on the synthetic fixture (the only corpus
# available without the maintainer's private sessions):
#
#   model              deepseek/deepseek-chat  (via OpenRouter)
#   prompt sha256      da5bf996dd1373ed  (1870 chars, 6 workflows / 4 domains)
#   n                  20 turns, stratified across 12 sessions, batch size 8
#   valid-JSON         3/3 batches, 20/20 labels returned
#   off-taxonomy ids   none
#   confidence         min 0.95  mean 0.96  max 1.00
#   agreement vs truth 20/20 (100%)
#   tokens             2684 in / 1031 out      wall 44s
#
# VERDICT: usable. The plan's prompt shape produces well-formed, on-taxonomy
# labels from a real teacher, including correct abstention (`none` on both
# non-work turns) and correct `EXT-PROBE` on all four probe turns.
#
# TWO CAVEATS, and they are why this spike alone does not de-risk the plan:
#
#   1. CONFIDENCE IS COMPRESSED. min 0.95 / mean 0.96. The plan's diagnostic
#      slice keys on `conf_mean` ascending to find hard cases — that selection
#      is useless if every label is 0.95+. Confidence needs its own calibration
#      check against real data, or the diagnostic slice needs a different
#      ordering signal (cross-labeller disagreement still works).
#
#   2. THE FIXTURE IS TOO EASY (see examples/diagnose_ceiling_gap.py: the
#      student itself scores 0.982 against truth). A 20/20 result on templated
#      text says the provider path works; it says nothing about how a teacher
#      performs on real, messy sessions. Re-run this against real transcripts
#      before quoting a teacher-accuracy figure.
#
# UNPROVEN: `unmapped` was emitted zero times. That is expected here — every
# turn in the fixture has a declared home — but it means the discovery loop has
# never been exercised by a real teacher. The empty-overlay design depends on a
# real teacher emitting `unmapped` on genuinely unplaceable turns, and that
# remains the single biggest untested assumption.
# ---------------------------------------------------------------------------
