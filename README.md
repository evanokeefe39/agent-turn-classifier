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

The synthetic fixture is templated, so held-out turns are **near-duplicates of
training turns**: mean nearest-neighbour cosine 0.932, with 23 of 72 turns
having a >0.95 neighbour. Session-disjoint folds do not prevent this, because
the *text* repeats across sessions, not just within them.

| corpus | turns | sessions | classes | student macro-F1 | teacher macro-F1 | gap |
|---|---|---|---|---|---|---|
| first cut | 17 | 5 | 6 | 0.056 | 0.752 | +0.697 |
| scaled | 84 | 12 | 6 | 0.982 | 0.708 | **−0.274** |

**Both rows are misleading, for different reasons.**

The first was corpus density: `GroupKFold(3)` over 5 sessions left ~11 training
examples for 6 classes, which the design's own `--min-class-support` (8)
rejects. Real as a lesson, but it measured the fixture's poverty.

The second is worse, and it is a **negative gap** — the student appears to beat
its ceiling, which should be impossible. It isn't impossible here: it is
near-duplicate leakage. The student memorises the templated surface forms, so a
held-out turn is answered by matching a near-identical training turn. Measured
honestly with leave-one-out (train on 71, test on 1), the same student scores
**0.722**, not 0.982.

**There is no real ceiling measurement in this tracer.** The mock teacher reads
`canned_labels.jsonl`, so `teacher_path_macro_f1` (0.708) is just
`1 − injected_error_rate` (19/72 ≈ 26%) — it is a property of the fixture, not a
bound on what the student can achieve. A ceiling requires a real teacher and a
gold set, which is the real run.

Each metric is asserted to be computed over the *same* turn population as its
counterpart, which is the one property the tracer does prove: the units are
right even when the numbers are not informative.


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
