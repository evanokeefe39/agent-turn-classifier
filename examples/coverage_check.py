"""Measure whether the declared workflow taxonomy describes real agent work.

Reads real OMP sessions, groups them into spans of agent work, and reports what
fraction of spans can be placed into the 15 declared domains — so the question
"is our work standardized, or is it chaos?" has a number instead of an
impression.

Sampling is SEEDED and the file list is deterministic, so a reader's labels stay
attached to a fixed sample. Corpus drift between runs is expected and is stamped
in the output rather than silently changing the population.

Usage:
    python examples/coverage_check.py --sample 120 --out .out/coverage_sample.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spans import classify_span, load_rows, span_view, spans_of  # noqa: E402

# Harness-probe / scratch directories, excluded by exact name. These are the
# agent's own test scratch rather than real work, and they are generic across
# installs. User-specific corpus roots and project exclusions come from config
# (see load_corpus_config) — never hardcoded here, because this repo is public
# and its fixtures are deliberately fictional.
NOISE_DIRS = {
    "-", "--C--tmp--", "--C--tmp-hookprobe--", "--C--tmp-probe-main--",
    "-.omp", "-.cache-otel-spike",
    # Parent/scratch cwd encodings, not projects: `-repos` is the bare parent of
    # every repo and `-tmp` is scratch. They contribute spans but carry no
    # project identity, so counting them overstates the corpus with sessions
    # that belong to no project. Extend with corpus.exclude_projects if your
    # layout differs.
    "-repos", "-tmp",
}

CONFIG_CANDIDATES = (
    "user_data/config.local.yaml",   # the user's own, gitignored
    "user_data/config.example.yaml", # tracked fallback, points at examples/
)


def load_corpus_config(path: str | None = None) -> dict:
    """Resolve the sessions root and project exclusions from config.

    Fails loudly rather than defaulting to a path that may not exist: a silent
    fallback would either find nothing (and report an empty corpus as a result)
    or leak a personal path into a public repo.

    Accepts either a `sessions.<adapter>.root` mapping or a flat
    `sessions_root`, plus optional `corpus.exclude_projects`.
    """
    import yaml

    candidates = [Path(path)] if path else [Path(c) for c in CONFIG_CANDIDATES]
    chosen = next((c for c in candidates if c.is_file()), None)
    if chosen is None:
        raise SystemExit(
            "no config found. Copy user_data/config.example.yaml to "
            "user_data/config.local.yaml and set sessions.omp.root, or pass "
            "--config. Refusing to guess a sessions path."
        )
    raw = yaml.safe_load(chosen.read_text(encoding="utf-8")) or {}

    root = raw.get("sessions_root")
    if not root:
        for spec in (raw.get("sessions") or {}).values():
            if isinstance(spec, dict) and spec.get("root"):
                root = spec["root"]
                break
    if not root:
        raise SystemExit(f"{chosen}: no sessions root found (set sessions.omp.root)")
    root = Path(str(root).replace("<you>", Path.home().name))

    return {
        "config_path": str(chosen),
        "sessions_root": root,
        "exclude_projects": set((raw.get("corpus") or {}).get("exclude_projects") or []),
    }


def discover(sessions_root: Path, exclude_projects: set[str]) -> list[Path]:
    """Deterministic file list: sorted, so the sample is reproducible."""
    paths: list[Path] = []
    for directory in sorted(p for p in sessions_root.iterdir() if p.is_dir()):
        if directory.name in NOISE_DIRS or directory.name in exclude_projects:
            continue
        paths.extend(sorted(directory.glob("*.jsonl")))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", default=".out/coverage_sample.json")
    ap.add_argument("--config", help="config file; defaults to user_data/config.local.yaml, else the example")
    ap.add_argument("--per-session", type=int, default=3,
                    help="cap spans drawn from any one session, so one long session cannot dominate")
    args = ap.parse_args()

    cfg = load_corpus_config(args.config)
    root = cfg["sessions_root"]
    if not root.is_dir():
        raise SystemExit(f"{cfg['config_path']}: sessions root not a directory: {root}")
    # The tracked example config points at `examples/`, so a clone with no local
    # config would otherwise "succeed" against three fixture files and print a
    # coverage percentage from a handful of spans — the finds-nothing-and-reports
    # -it-as-a-result mode this script exists to avoid.
    if root.resolve() == Path("examples").resolve():
        print("WARNING: running against the tracked `examples/` fixtures, not a "
              "real corpus. The percentages below describe sample data. Point "
              "sessions.omp.root at your own sessions root in "
              "user_data/config.local.yaml.\n")
    files = discover(root, cfg["exclude_projects"])
    spans: list[dict] = []
    excluded = Counter()
    excluded_total = 0
    for path in files:
        for span in spans_of(load_rows(path)):
            ok, why = classify_span(span)
            if not ok:
                excluded[why] += 1
                excluded_total += 1
                continue
            span["_session"] = path.name
            span["_project"] = path.parent.name
            # Positional within the file: a (session, ts) key can repeat when
            # two user messages share a millisecond, so identity is positional.
            span["_id"] = f"{path.name}:{len(spans) + excluded_total}"
            spans.append(span)

    # Corpus stamp: totals drift as the maintainer works, so record the cutoff.
    stamped = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sessions_root": str(root),
        "config_path": cfg["config_path"],
        "sessions_scanned": len(files),
        "work_spans_found": len(spans),
        "excluded": dict(excluded),
        "seed": args.seed,
    }

    rng = random.Random(args.seed)
    by_session: dict[str, list[dict]] = {}
    for s in spans:
        by_session.setdefault(s["_session"], []).append(s)
    # Stratify by project first so no single repo dominates the sample.
    by_project: dict[str, list[dict]] = {}
    for s in spans:
        by_project.setdefault(s["_project"], []).append(s)

    picked: list[dict] = []
    taken: set[str] = set()
    project_names = sorted(by_project)
    while len(picked) < args.sample and project_names:
        progressed = False
        for name in project_names:
            if len(picked) >= args.sample:
                break
            # BUG FIXED: the pool previously filtered only on the per-session
            # cap, never on spans already drawn, so a session holding fewer
            # spans than the cap re-offered the same span and `rng.choice`
            # returned it again — the same span could enter the sample up to
            # `--per-session` times. That broke independence between sampled
            # units and inflated the corpus counts. Identity is now tracked.
            pool = [s for s in by_project[name]
                    if s["_id"] not in taken
                    and sum(1 for p in picked if p["_session"] == s["_session"]) < args.per_session]
            if not pool:
                continue
            choice = rng.choice(pool)
            picked.append(choice)
            taken.add(choice["_id"])
            progressed = True
        if not progressed:
            break

    # Size-stratify the printout so the summary covers small and large spans.
    def band(s: dict) -> str:
        n = len(s["tools"])
        if n == 0:
            return "reply_only"
        if n <= 3:
            return "small_1_3"
        if n <= 10:
            return "medium_4_10"
        if n <= 25:
            return "large_11_25"
        return "very_large_26plus"

    stamped["sample_size"] = len(picked)
    stamped["sample_by_band"] = dict(Counter(band(s) for s in picked))
    stamped["sample_by_project"] = dict(Counter(s["_project"] for s in picked))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"stamp": stamped, "spans": [
            {"project": s["_project"], "session": s["_session"], "ts": s["ts"],
             "id": s["_id"], "reason": classify_span(s)[1],
             "band": band(s), "n_actions": len(s["tools"]),
             "user": s["user"][:400],
             "intents": [t["intent"] for t in s["tools"] if t["intent"]],
             "reply": s["reply"][:800],
             "view": span_view(s)[:1200]}
            for s in picked]}, indent=2, ensure_ascii=False),
        encoding="utf-8", newline="\n")

    print(json.dumps(stamped, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
