"""Tracer: the thinnest end-to-end slice that proves the architecture.

This is a *tracer bullet*, not the product. It exists to answer one question
before the v0.1.0 build starts: does the chain

    canonical records -> turn -> view -> teacher label -> majority vote
                      -> embedding -> logistic head -> session-disjoint eval

hold together, and does it run on a free Colab CPU runtime with no API key?

What it deliberately does NOT do (all of these are v0.1.0 plan items):
  - no adapters (it reads the canonical shape directly, not OMP/Claude Code JSONL)
  - no parquet, no DuckDB, no write-audit-publish
  - no real teacher (a mock replays canned labels; no network, no key)
  - no sampling arms (agreement across repeats is meaningless against a
    deterministic mock — see run_tracer's note)
  - no calibration, no abstention, no ONNX export, no ship rule
  - no unmapped/discovery loop (the residue needs a real teacher to be meaningful)

Every function name here matches the seam the plan names, so the v0.1.0
implementation is a substitution rather than a redesign.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

VIEW_CAP = 4000
USER_CHARS = 1200
ASSISTANT_CHARS = 800
ACTION_LINES = 40

# ---------------------------------------------------------------- ontology


@dataclass(frozen=True)
class Workflow:
    id: str
    name: str
    domain: str
    triggers: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()


def load_ontology(path: str | Path) -> tuple[list[Workflow], set[str]]:
    """Load workflows and the set of declared domains."""
    import yaml

    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    workflows = [
        Workflow(
            id=w["id"],
            name=w["name"],
            domain=w["domain"],
            triggers=tuple(w.get("triggers") or ()),
            excludes=tuple(w.get("excludes") or ()),
        )
        for w in doc["workflows"]
    ]
    domains = {d["id"] for d in doc["domains"]}
    return workflows, domains


def validate(workflows: list[Workflow], domains: set[str]) -> list[str]:
    """Return a list of validator errors. Empty means valid.

    Mirrors the plan's P3 rules: every workflow's domain exists, >= 1 trigger,
    and a domain holding >= 2 workflows requires a discriminating `excludes`
    reason on each.
    """
    errors: list[str] = []
    seen: set[str] = set()
    for w in workflows:
        if w.id in seen:
            errors.append(f"duplicate workflow id {w.id!r}")
        seen.add(w.id)
        if w.domain not in domains:
            errors.append(f"{w.id}: unknown domain {w.domain!r}")
        if not w.triggers:
            errors.append(f"{w.id}: >= 1 non-empty trigger required")

    by_domain: dict[str, list[Workflow]] = {}
    for w in workflows:
        by_domain.setdefault(w.domain, []).append(w)
    for domain, members in by_domain.items():
        if len(members) < 2:
            continue
        for w in members:
            if not w.excludes:
                errors.append(f"{w.id}: domain {domain!r} holds >= 2 workflows, excludes required")
            for rule in w.excludes:
                if ": " not in rule:
                    errors.append(f"{w.id}: malformed exclude {rule!r}")
                    continue
                target, reason = rule.split(": ", 1)
                if target not in seen:
                    errors.append(f"{w.id}: exclude names unknown workflow {target!r}")
                # The plan's anti-vacuity rule: a reason must carry real content,
                # not a placeholder that validates while disambiguating nothing.
                if len(reason.split()) < 4:
                    errors.append(f"{w.id}: exclude reason too short to discriminate {reason!r}")
    return errors


# ---------------------------------------------------------------- records/turns


@dataclass
class Turn:
    turn_id: str
    session_id: str
    seq: int
    ts_start: str
    session_title: str = ""
    user_text: str = ""
    assistant_text: str = ""
    thinking_text: str = ""
    actions: list[dict] = field(default_factory=list)
    model: str = ""
    turn_kind: str = "other"

    @property
    def path_label(self) -> str:
        return f"{self.domain}/{self.workflow}" if getattr(self, "domain", None) else ""

    domain: str | None = None
    workflow: str | None = None


def read_canonical(path: str | Path) -> list[dict]:
    """Read the canonical JSONL shape, skipping blank lines.

    Blank lines are skipped here for tracer brevity; the real extractor treats
    them as records so the byte-identity claim needs no scoping caveat.
    """
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def build_turns(records: list[dict]) -> list[Turn]:
    """Segment records into turns.

    Tracer simplification: one user record starts one turn, and every following
    record until the next user record belongs to it. The real segmenter keys on
    `(source, session_id, seq)` and orders by `(file, ts, src_line)`; the
    `src_line` tie-break matters because batched records share timestamps.
    """
    turns: list[Turn] = []
    current: Turn | None = None
    titles: dict[str, str] = {}

    for rec in records:
        sid = rec.get("sessionId", "")
        if rec.get("type") == "title":
            titles[sid] = rec.get("title", "")
            continue

        if rec.get("role") == "user" or (rec.get("type") == "message" and rec.get("role") == "user"):
            current = Turn(
                turn_id=f"{sid}:{rec.get('seq', len(turns) + 1)}",
                session_id=sid,
                seq=int(rec.get("seq", len(turns) + 1)),
                ts_start=rec.get("ts", ""),
                session_title=titles.get(sid, ""),
            )
            for block in rec.get("content") or []:
                if block.get("type") == "text":
                    current.user_text += block.get("text", "")
            turns.append(current)
            continue

        if current is None:
            continue
        if rec.get("type") == "message":
            for block in rec.get("content") or []:
                if block.get("type") == "text":
                    current.assistant_text += block.get("text", "")
                elif block.get("type") == "thinking":
                    current.thinking_text += block.get("thinking", "")
            current.model = current.model or rec.get("model", "")
        elif rec.get("customType") == "tool_execution_start":
            data = rec.get("data") or {}
            args = data.get("args") or {}
            arg = args.get("path") or args.get("command") or ""
            current.actions.append({"name": data.get("toolName", ""), "arg": arg})

    return turns


# ---------------------------------------------------------------- view


def render(turn: Turn, redaction: dict | None = None) -> str:
    """Render the single text both teacher and student see.

    Redaction is applied BEFORE the caller hashes the result, so the hash
    attests the redacted payload — the plan's egress-seam contract.
    """
    parts = [
        f"TITLE: {turn.session_title}",
        f"KIND: {turn.turn_kind}",
        f"USER: {turn.user_text[:USER_CHARS]}",
    ]
    if turn.actions:
        # A path argument is reduced to its basename to keep the view short; a
        # command is not a path, so basename-ing it would corrupt the signal
        # ("uv run pytest tests/sources -q" -> "sources -q").
        def _arg(a: dict) -> str:
            raw = a.get("arg") or ""
            if not raw:
                return ""
            return Path(raw).name if ("/" in raw or "\\" in raw) else raw

        lines = [f"- {a['name']} {_arg(a)}" for a in turn.actions[:ACTION_LINES]]
        parts.append("ACTIONS:\n" + "\n".join(lines))
    parts.append(f"ASSISTANT: {turn.assistant_text[-ASSISTANT_CHARS:]}")
    text = "\n".join(parts)

    if redaction:
        for pat in redaction.get("patterns", []):
            text = re.sub(pat, "[REDACTED]", text)

    return text[:VIEW_CAP]


def view_sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- teacher


class MockTeacher:
    """Replays canned labels. Deterministic, offline, no key.

    A real teacher is an OpenAI-compatible chat-completions client. The point of
    the tracer is that everything downstream of `label()` is identical either way.
    """

    def __init__(self, labels: dict[str, str], confidence: float = 0.9):
        self._labels = labels
        self._confidence = confidence

    def label(self, turn: Turn, view: str) -> dict:
        # Keyed on the rendered view so the mock cannot cheat by reading the
        # turn's true label — it sees exactly what a real teacher would see.
        wf = self._labels.get(view)
        if wf is None:
            return {"workflow": "unmapped", "confidence": 0.0, "evidence": "no match"}
        return {"workflow": wf, "confidence": self._confidence, "evidence": f"mock: {wf}"}


def majority(votes: list[dict]) -> dict:
    """Collapse per-pass teacher votes into one majority row.

    `vote_split` is what the plan's diagnostic slice keys on for
    self-disagreement, so it is computed here even though the tracer's mock
    never actually splits.
    """
    counts = Counter(v["workflow"] for v in votes)
    top, n = counts.most_common(1)[0]
    mean_conf = sum(v["confidence"] for v in votes) / len(votes)
    return {
        "workflow": top,
        "vote_split": f"{n}/{len(votes)}",
        "conf_mean": mean_conf,
        "n_valid": len(votes),
    }


# ---------------------------------------------------------------- dataset


def build_frame(turns: list[Turn], views: dict[str, str], labels: dict[str, str]):
    """Build a pandas frame of (session, view, label) for turns that have labels."""
    import pandas as pd

    rows = []
    for t in turns:
        lab = labels.get(t.turn_id)
        if not lab or lab == "unmapped":
            continue  # unmapped is the discovery residue: never trained on (P3)
        rows.append({"turn_id": t.turn_id, "session_id": t.session_id,
                     "view": views[t.turn_id], "label": lab})
    return pd.DataFrame(rows)


def session_disjoint_folds(frame, n_splits: int = 3):
    """Yield (train_idx, test_idx) with no session appearing in both.

    Turns inside a session are correlated; a random split leaks. The plan
    asserts this on the fitted folds rather than by intent.
    """
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(frame, frame["label"], groups=frame["session_id"]))


# ---------------------------------------------------------------- students


def fit_linear(train_texts, train_labels, encoder_name: str = "BAAI/bge-small-en-v1.5"):
    """Encoder + multinomial logistic head. The plan's default student."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    encoder = SentenceTransformer(encoder_name)
    x = encoder.encode(list(train_texts), normalize_embeddings=True)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x, list(train_labels))
    return {"encoder": encoder, "clf": clf}


def predict_linear(model, texts):
    x = model["encoder"].encode(list(texts), normalize_embeddings=True)
    return model["clf"].predict(x)


# ---------------------------------------------------------------- evaluate


def path_macro_f1(y_true, y_pred) -> float:
    """Primary metric: macro-F1 over the flat domain/workflow path label.

    Computed over the promoted label space only — `unmapped` rows are excluded
    from the denominator rather than scored, because `unmapped` is a
    teacher-only discovery class, not a prediction the student makes.
    """
    from sklearn.metrics import f1_score

    return float(f1_score(list(y_true), list(y_pred), average="macro", zero_division=0))


# ---------------------------------------------------------------- entry point


def run_tracer(
    sessions: str | Path = "examples/sessions.sample.jsonl",
    ontology: str | Path = "examples/ontology.example.yaml",
    labels: str | Path = "examples/sessions.sample.labels.jsonl",
    encoder: str = "BAAI/bge-small-en-v1.5",
    folds: int = 3,
) -> dict:
    """Run the whole chain and return a metrics dict.

    The one thing this cannot measure is agreement across sampling arms: the
    teacher here is a deterministic mock, so repeated passes are identical by
    construction and alpha would be a meaningless 1.0. Arm selection needs the
    real teacher and stays a v0.1.0 item.
    """
    import pandas as pd

    workflows, domains = load_ontology(ontology)
    errors = validate(workflows, domains)
    if errors:
        raise ValueError("ontology invalid: " + "; ".join(errors))

    turns = build_turns(read_canonical(sessions))
    views = {t.turn_id: render(t, redaction={"patterns": []}) for t in turns}

    # Mock teacher: reads the canned labels keyed on the rendered-view hash. It
    # NEVER sees the truth file, so the eval below is a genuine held-out
    # measurement rather than the mock reading back the answer.
    canned_by_sha = {}
    for line in Path("examples/canned_labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            canned_by_sha[row["view_sha256"]] = row["workflow"]

    teacher = MockTeacher({v: canned_by_sha.get(view_sha(v), "unmapped") for v in views.values()})

    teacher_labels: dict[str, str] = {}
    majority_rows: dict[str, dict] = {}
    for t in turns:
        v = views[t.turn_id]
        # Three passes, as the plan's default arm specifies. Against a mock they
        # are identical by construction — that is the whole caveat above.
        votes = [teacher.label(t, v) for _ in range(3)]
        maj = majority(votes)
        teacher_labels[t.turn_id] = maj["workflow"]
        majority_rows[t.turn_id] = maj

    # Resolve the flat path label the primary metric scores.
    wf_domain = {w.id: w.domain for w in workflows}
    def path(wf: str) -> str:
        return f"{wf_domain[wf]}/{wf}" if wf in wf_domain else "unmapped"

    teacher_labels = {k: path(v) for k, v in teacher_labels.items()}

    truth = {}
    for line in Path(labels).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            truth[row["turn_id"]] = path(row["workflow"])

    frame = build_frame(turns, views, teacher_labels)
    if frame.empty:
        raise ValueError("no labelled turns — nothing to train on")

    n_splits = min(folds, frame["session_id"].nunique())
    splits = session_disjoint_folds(frame, n_splits=n_splits)

    # Session-disjointness is asserted on the fitted folds, not assumed (P9).
    for tr, te in splits:
        overlap = set(frame.iloc[tr]["session_id"]) & set(frame.iloc[te]["session_id"])
        assert not overlap, f"session leaked across folds: {overlap}"

    y_true, y_pred = [], []
    for tr, te in splits:
        model = fit_linear(frame.iloc[tr]["view"], frame.iloc[tr]["label"], encoder)
        preds = predict_linear(model, frame.iloc[te]["view"])
        y_true.extend(frame.iloc[te]["label"])
        y_pred.extend(preds)

    teacher_paths = [teacher_labels[t.turn_id] for t in turns if t.turn_id in teacher_labels]
    truth_paths = [truth[t.turn_id] for t in turns if t.turn_id in teacher_labels and t.turn_id in truth]

    return {
        "n_turns": len(turns),
        "n_sessions": frame["session_id"].nunique(),
        "n_labelled": len(frame),
        "n_classes": frame["label"].nunique(),
        "unmapped_rate": round(
            1 - len(frame) / max(1, sum(1 for t in turns if t.turn_id in teacher_labels)), 3
        ),
        "student_path_macro_f1": round(path_macro_f1(y_true, y_pred), 3),
        "teacher_path_macro_f1": round(
            path_macro_f1(truth_paths, [teacher_labels[t.turn_id] for t in turns
                                        if t.turn_id in teacher_labels and t.turn_id in truth]), 3
        ),
        "ceiling_gap": round(
            path_macro_f1(truth_paths, [teacher_labels[t.turn_id] for t in turns
                                        if t.turn_id in teacher_labels and t.turn_id in truth])
            - path_macro_f1(y_true, y_pred), 3
        ),
        "n_folds": n_splits,
        "encoder": encoder,
    }
