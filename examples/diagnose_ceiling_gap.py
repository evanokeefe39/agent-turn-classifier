"""Diagnose what the tracer's ceiling gap does and does not measure.

Run:  uv run --extra train python examples/diagnose_ceiling_gap.py

This is a permanent, documented diagnostic rather than a scratch script, because
its result is load-bearing for the plan: it shows that `ceiling_gap` cannot
detect distillation failure when a student can exceed its teacher — and the
plan's ship rule keys on exactly that quantity.

It cross-tabulates the student's macro-F1 over

    {GroupKFold(3), LeaveOneOut} x {target = teacher labels, target = truth}

which separates three things that were previously conflated:

  * the FOLD SCHEME  (GroupKFold vs LOO) — if these differ, there is leakage
  * the SCORING TARGET (teacher labels vs ground truth) — if these differ, the
    gap is measuring the teacher's error rate, not the student's quality
  * class separability — a nearest-neighbour split by class shows whether high
    cosine similarity is class signal (same-workflow neighbours) or duplicated
    text (cross-workflow neighbours)

Observed on the synthetic fixture (2026-09-22)::

                        target=teacher   target=truth
      GroupKFold(3)        0.705          0.982
      LeaveOneOut          0.691          0.966

    teacher macro-F1 vs truth: 0.708
    mean NN cosine:            0.932
    NN same-workflow:          72/72 (100%)
    cross-workflow NN:         n=0

Reading: no leakage (LOO trains on MORE turns, so leakage would RAISE it and it
did not move); the ~0.27 difference is entirely the target column; and every
nearest neighbour is same-workflow, so the similarity is class signal. The
fixture is simply too easy — six separable workflows at 12 examples per class —
so the student legitimately outscores a teacher whose 0.708 is just its
injected error rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EX = Path(__file__).resolve().parent
sys.path.insert(0, str(EX.parent / "src"))

from agent_turn_classifier import tracer as T  # noqa: E402


def build_scored_frame():
    """Turns, views, teacher labels and truth, in one label space."""
    turns = T.build_turns(T.read_canonical(EX / "sessions.sample.jsonl"))
    views = {t.turn_id: T.render(t) for t in turns}
    canned = {
        json.loads(line)["view_sha256"]: json.loads(line)["workflow"]
        for line in (EX / "canned_labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    teacher = {
        t.turn_id: canned.get(T.view_sha(views[t.turn_id]), "unmapped") for t in turns
    }
    truth_raw = {
        json.loads(line)["turn_id"]: json.loads(line)["workflow"]
        for line in (EX / "sessions.sample.labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    workflows, _ = T.load_ontology(EX / "ontology.example.yaml")
    wf_domain = {w.id: w.domain for w in workflows}

    def path(wf: str) -> str:
        if wf in T.RESERVED_NON_TRAINABLE:
            return wf
        return f"{wf_domain[wf]}/{wf}" if wf in wf_domain else "unmapped"

    frame = T.build_frame(turns, views, {k: path(v) for k, v in teacher.items()})
    frame["truth"] = [path(truth_raw[t]) for t in frame["turn_id"]]
    return frame, turns


def main() -> int:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold, LeaveOneOut

    frame, _ = build_scored_frame()
    ids = list(frame["turn_id"])
    n_classes = frame["label"].nunique()
    print(f"scored turns {len(ids)}  classes {n_classes}  "
          f"turns/class {len(ids) // max(1, n_classes)}\n")

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    x = encoder.encode(list(frame["view"]), normalize_embeddings=True)
    y_teacher = np.array(frame["label"])
    y_truth = np.array(frame["truth"])
    groups = np.array(frame["session_id"])

    def score(splits, target) -> float:
        y_true, y_pred = [], []
        for tr, te in splits:
            model = LogisticRegression(max_iter=2000, class_weight="balanced")
            model.fit(x[tr], y_teacher[tr])
            y_true.extend(target[te])
            y_pred.extend(model.predict(x[te]))
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    gkf = list(GroupKFold(n_splits=3).split(x, y_teacher, groups))
    loo = list(LeaveOneOut().split(x))

    print("                      target=teacher   target=truth")
    print(f"  GroupKFold(3)       {score(gkf, y_teacher):.3f}            {score(gkf, y_truth):.3f}")
    print(f"  LeaveOneOut         {score(loo, y_teacher):.3f}            {score(loo, y_truth):.3f}")

    print(f"\n  teacher macro-F1 vs truth: "
          f"{f1_score(y_truth, y_teacher, average='macro', zero_division=0):.3f}")

    # Class-conditioned nearest neighbours: class signal, or duplicated text?
    sim = x @ x.T
    np.fill_diagonal(sim, -1)
    nn = sim.argmax(1)
    nn_sim = sim.max(1)
    same = sum(1 for a, b in zip(y_truth, y_truth[nn]) if a == b)
    cross = [s for s, a, b in zip(nn_sim, y_truth, y_truth[nn]) if a != b]
    print(f"\n  mean NN cosine:        {nn_sim.mean():.3f}")
    print(f"  NN same-workflow:      {same}/{len(ids)} ({same / len(ids):.0%})")
    print(f"  cross-workflow NN:     n={len(cross)}"
          + (f" mean cos {np.mean(cross):.3f}" if cross else " (none)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
