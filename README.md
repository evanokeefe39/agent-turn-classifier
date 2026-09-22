# agent-turn-classifier

Work in progress. The current artifact is a **tracer bullet**: the thinnest
end-to-end slice of the design, used to prove the chain holds before the real
build starts.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanokeefe39/agent-turn-classifier/blob/main/notebooks/00_tracer.ipynb)

## What this is

A tool for labelling every turn of an AI coding agent's session with a workflow
from *your own* taxonomy, then distilling a teacher LLM's labelling into a small
classifier that runs locally on CPU — no API call at inference time.

The eventual pipeline:

```
your transcripts -> records -> turns -> view -> teacher labels -> majority vote
                 -> student (embedding + linear head) -> ONNX int8 -> local predict
```

with the gates that make weak labelling trustworthy: agreement measured across
repeated passes and an independent check model, a small human-adjudicated gold
set, calibration and an abstention path, and a pre-declared ship rule that is
allowed to fail.

## Status: tracer only

What exists now is `notebooks/00_tracer.ipynb` and the `tracer.py` module behind
it. It runs the full chain on a **synthetic** 84-turn corpus, on a free Colab
**CPU** runtime, with **no API key** and no GPU.

### What it measured — and what it does NOT measure

| corpus | turns | sessions | classes | student macro-F1 | teacher macro-F1 | gap |
|---|---|---|---|---|---|---|
| first cut | 17 | 5 | 6 | 0.056 | 0.752 | +0.697 |
| scaled | 84 | 12 | 6 | 0.982 | 0.708 | **−0.274** |

**Neither row measures distillation, and the second one looks impossible.**

The first row was corpus density: `GroupKFold(3)` over 5 sessions left ~11
training examples for 6 classes, which the design's own `--min-class-support`
(8) rejects.

The second row is a **negative gap** — the student appears to beat the teacher.
A diagnostic over the four combinations of {GroupKFold, LeaveOneOut} ×
{target=teacher, target=truth} settles what it means:

```
                    target=teacher   target=truth
  GroupKFold(3)        0.705          0.982
  LeaveOneOut          0.691          0.966
```

**There is no leakage.** Student-vs-truth is 0.982 under GroupKFold and 0.966
under LeaveOneOut — essentially identical, and LOO *trains on more* turns (71 vs
48), so leakage would raise it, not leave it flat. And of 72 nearest neighbours,
**72 are same-workflow and 0 are cross-workflow**: the high cosine similarity is
class signal, not duplicated text.

**The 0.26 difference was a target artefact.** Reading across the table, the
target column — not the split — accounts for nearly all of it. Scoring against
the teacher's labels is simply harder, because the mock's labels carry a
deliberately injected 26% error rate.

So the honest reading: **this fixture is too easy to measure distillation.**
Six synthetic workflows are separable by bge-small at 12 examples per class, and
the student reaches ~0.97 against truth under any split. The teacher's 0.708 is
just `1 − injected_error_rate` — it is not a ceiling and bounds nothing.

The plan-level consequence is the thing worth carrying forward:

> **`ceiling_gap` cannot detect distillation failure when a student can exceed
> its teacher — and the ship rule keys on exactly that quantity.** The ship rule
> needs a different quantity: gate on teacher-vs-truth on held-out turns, or on
> a corpus where the teacher's accuracy genuinely exceeds the student's.

What the tracer *does* prove, and this is not nothing: the chain runs end to end
offline on a free CPU runtime in ~30s; folds are session-disjoint (asserted on
the fitted folds); and teacher and student are scored on the same turn
population, with counts that partition the turn set. The units are right even
where the magnitudes are uninformative.


| Included in the tracer | Deferred to the real build |
|---|---|
| canonical records → turns | adapters (OMP, Claude Code, generic JSONL) |
| view render + redaction seam | Parquet / DuckDB, write-audit-publish |
| mock teacher + majority vote | the real teacher, batching, cost caps |
| session-disjoint folds (asserted) | preflight referential integrity |
| encoder + logistic head | ONNX export, int8 quantisation, serving |
| path macro-F1 + ceiling gap | calibration, abstention, the ship rule |
| ontology validation | the `unmapped` discovery/promotion loop |

Not yet present: a CLI, tests, CI, packaging beyond a stub `pyproject.toml`.

## Run it locally

```bash
uv sync --extra train
uv run jupyter nbconvert --to notebook --execute notebooks/00_tracer.ipynb
```

To run just the chain, without a notebook:

```python
import sys; sys.path.insert(0, "src")
from agent_turn_classifier import tracer
print(tracer.run_tracer())
```

## Privacy

This repository ships **synthetic examples only**. The fixtures in `examples/`
are invented — fictional repositories, no real transcript text, no client or
employer taxonomy. Real sessions, the maintainer's taxonomy, and all generated
artefacts are gitignored by construction (`user_data/`, `.out/`).

## Licence

MIT.
