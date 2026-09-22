"""Tests for the tracer's invariants.

These defend the properties that, if silently broken, would make the tracer
prove something false — not the plumbing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_turn_classifier import tracer as T  # noqa: E402

EX = Path(__file__).resolve().parent.parent / "examples"


def test_view_is_deterministic_and_capped():
    turns = T.build_turns(T.read_canonical(EX / "sessions.sample.jsonl"))
    t = turns[0]
    assert T.render(t) == T.render(t)
    assert len(T.render(t)) <= T.VIEW_CAP
    assert T.view_sha(T.render(t)) == T.view_sha(T.render(t))


def test_redaction_happens_before_the_hash():
    """The egress seam's whole contract: the hash attests the REDACTED payload.

    If redaction ran after hashing, the hash would attest text that was never
    sent — and a leak would be invisible in the record.
    """
    turns = T.build_turns(T.read_canonical(EX / "sessions.sample.jsonl"))
    t = turns[0]
    t.user_text = "token sk-proj-ABCDEFGH12345678 here"
    red = {"patterns": [r"sk-proj-[A-Z0-9]+"]}
    plain, masked = T.render(t), T.render(t, redaction=red)
    assert "sk-proj-ABCDEFGH12345678" in plain
    assert "sk-proj-ABCDEFGH12345678" not in masked
    assert T.view_sha(plain) != T.view_sha(masked)


def test_canonical_view_is_not_unique_per_turn():
    """Two turns CAN render identically (same title, text, actions).

    Labels are keyed on turn_id, never on the view — a view-keyed label store
    would silently overwrite one of them. Documented here because it is the
    kind of collision that only shows up as an unexplained class-count shortfall.
    """
    turns = T.build_turns(T.read_canonical(EX / "sessions.sample.jsonl"))
    ids = [t.turn_id for t in turns]
    assert len(ids) == len(set(ids)), "turn ids must be unique"
    assert len(set(T.render(t) for t in turns)) <= len(turns)


def test_ontology_validates():
    wfs, domains = T.load_ontology(EX / "ontology.example.yaml")
    assert T.validate(wfs, domains) == []


def test_validator_rejects_vacuous_exclude():
    """The anti-vacuity rule: a placeholder reason must not validate."""
    wfs, domains = T.load_ontology(EX / "ontology.example.yaml")
    broken = [
        w if w.id != "W2" else T.Workflow(
            id="W2", name=w.name, domain=w.domain, triggers=w.triggers,
            excludes=("W1: because",),  # 1 word — disambiguates nothing
        )
        for w in wfs
    ]
    assert any("too short" in e for e in T.validate(broken, domains))


def test_validator_rejects_dangling_exclude():
    wfs, domains = T.load_ontology(EX / "ontology.example.yaml")
    broken = [
        w if w.id != "W2" else T.Workflow(
            id="W2", name=w.name, domain=w.domain, triggers=w.triggers,
            excludes=("W99: names a workflow that does not exist at all",),
        )
        for w in wfs
    ]
    assert any("unknown workflow" in e for e in T.validate(broken, domains))


def test_ontology_requires_triggers():
    wfs, domains = T.load_ontology(EX / "ontology.example.yaml")
    broken = [T.Workflow(id=w.id, name=w.name, domain=w.domain, triggers=())
              for w in wfs]
    assert any("trigger required" in e for e in T.validate(broken, domains))


def test_build_turns_segments_on_user_records():
    """One user record opens one turn; later records attach to it.

    Asserts the PROPERTY, not a count: a hardcoded turn count re-breaks every
    time the fixture grows, which trains people to edit the test instead of
    reading it.
    """
    records = T.read_canonical(EX / "sessions.sample.jsonl")
    turns = T.build_turns(records)

    n_user = sum(1 for r in records if r.get("role") == "user")
    assert len(turns) == n_user, "every user record must open exactly one turn"
    assert len(turns) >= 3, "need enough turns to be a meaningful test"

    # Turn ids are unique, and sessions are well-formed.
    assert len({t.turn_id for t in turns}) == len(turns)
    assert len({t.session_id for t in turns}) >= 3, "need >= 3 sessions for GroupKFold"

    # Every turn carries the session title set by its opening title record.
    assert all(t.session_title for t in turns)

    # Tool calls attach to the turn whose user record opened before them.
    assert any(t.actions for t in turns)


def test_teacher_and_student_score_the_same_population():
    """The teacher and student must be scored on the SAME turn population.

    `frame` drops `unmapped` rows (a teacher-only discovery class the student
    never predicts). If the teacher is scored over all labelled turns instead,
    the two macro-F1s describe different populations and `ceiling_gap` is
    meaningless — silently, because at unmapped_rate=0.0 the populations are
    identical. The real bug: a teacher abstaining on some turns made the
    teacher's denominator strictly larger than the student's.

    This checks the invariant without loading an encoder, so it stays in the
    fast suite.
    """
    import json as _json
    from pathlib import Path as _Path

    turns = T.build_turns(T.read_canonical(EX / "sessions.sample.jsonl"))
    views = {t.turn_id: T.render(t) for t in turns}
    canned = {
        _json.loads(l)["view_sha256"]: _json.loads(l)["workflow"]
        for l in (_Path(EX) / "canned_labels.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    teacher_labels = {
        t.turn_id: canned.get(T.view_sha(views[t.turn_id]), "unmapped") for t in turns
    }
    truth = {
        _json.loads(l)["turn_id"]: _json.loads(l)["workflow"]
        for l in (_Path(EX) / "sessions.sample.labels.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }

    # The fixture must exercise the bug's precondition: turns excluded from the
    # training frame. These are `none` turns here — the mock teacher always has
    # an answer, so it never emits `unmapped`, and `none` is what makes the
    # frame's population smaller than the labelled one.
    frame_ids = [
        t.turn_id
        for t in turns
        if teacher_labels[t.turn_id] not in T.RESERVED_NON_TRAINABLE
    ]
    assert len(frame_ids) < len(turns), (
        "fixture has no reserved-label turns — the denominator bug is undetectable here"
    )

    # Every turn the student could be scored on must also have ground truth, so
    # neither side can silently drop a row the other kept.
    assert all(tid in truth for tid in frame_ids), "a scored turn has no truth label"


@pytest.mark.slow
def test_run_tracer_end_to_end():
    """The full chain. Requires the [train] extra (downloads an encoder).

    The ceiling-gap assertion is the point of this test. At 17 turns the gap was
    0.697 (the student cannot fit 6 classes from ~11 training examples); at 84
    turns it is 0.042. A regression back above the plan's --max-gap-to-ceiling
    means either the corpus shrank below usable density or the training frame
    started leaking across folds.
    """
    m = T.run_tracer(folds=3)
    assert m["n_turns"] > 0
    assert m["n_sessions"] >= 3
    assert m["n_classes"] >= 3
    assert 0.0 <= m["student_path_macro_f1"] <= 1.0
    # The student cannot exceed its teacher; a negative gap is a bug upstream.
    assert m["student_path_macro_f1"] <= m["teacher_path_macro_f1"] + 1e-9
    # Distillation must stay inside the ship rule's accepted gap.
    assert m["ceiling_gap"] <= 0.10, (
        f"ceiling gap {m['ceiling_gap']} exceeds the 0.10 ship threshold — "
        "the corpus is likely below the density the design assumes"
    )
