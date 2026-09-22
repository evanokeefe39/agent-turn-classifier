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
| first cut | 17 | 5 | 5 | 0.056 | 0.752 | **0.697** |
| after scaling | 84 | 12 | 7 | 0.708 | 0.750 | **0.042** |

The first row is the finding worth keeping. At 17 turns the student scored
barely above chance while its teacher scored 0.752 — a 0.697 gap, far outside
the ship rule's `--max-gap-to-ceiling` (0.10). The cause was **corpus density,
not the method**: `GroupKFold(3)` over 5 sessions left each fold training on
~11 examples for 6 classes, which the plan's own `--min-class-support` (8) would
have rejected outright. Scaling the fixture to 84 turns across 12 sessions
closed the gap to 0.042 without changing the model.

Recorded because it calibrates the real run: the corpus needs tens of examples
per class before the student's score means anything.


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
