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

### What it measured

| corpus | turns | sessions | classes | student macro-F1 | teacher (ceiling) | **ceiling gap** |
|---|---|---|---|---|---|---|
| first cut | 17 | 5 | 6 | 0.056 | 0.752 | **0.697** |
| after scaling | 84 | 12 | 6 | 0.708 | 0.708 | **0.000** |

Both runs are on the synthetic corpus with a mock teacher; `teacher_path_macro_f1`
is the ceiling a student cannot beat.

The first row is the finding worth keeping. At 17 turns the student scored
barely above chance while its teacher scored 0.752 — a 0.697 gap, far outside
the ship rule's `--max-gap-to-ceiling` (0.10). The cause was **corpus density,
not the method**: `GroupKFold(3)` over 5 sessions left each fold training on
~11 examples for 6 classes, which the plan's own `--min-class-support` (8) would
have rejected outright. Scaling the fixture to 84 turns across 12 sessions
closed the gap without changing the model.

The `0.000` gap is a **coincidence, not a perfect score** — the teacher is wrong
on 19 of 72 scored turns, and the student's different error pattern happens to
land on the same macro-F1. Do not read it as evidence the student matches the
teacher.

**Correctness caveat.** The corpus is synthetic and the teacher is a mock, so
none of these numbers say anything about real sessions. They calibrate one
thing: the corpus needs tens of examples per class before the student's score
means anything.


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
