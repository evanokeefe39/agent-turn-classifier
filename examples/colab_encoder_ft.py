# PASTE THIS INTO YOUR COLAB L4 SESSION
#
# Purpose: the `encoder_ft` arm — the one thing a GPU actually accelerates
# (est. 20-60 min CPU vs ~2-4 min on an L4). It fine-tunes a ModernBERT-class
# encoder on the tracer's corpus and reports macro-F1 vs truth, so the number is
# directly comparable to the CPU tracer's logistic head.
#
# Before pasting: Runtime > Change runtime type > L4 GPU.
# Expect: ~2 min install, ~1-3 min fine-tune per fold.
#
# Rewritten for sentence-transformers v5 (5.x): the v4 recipes
# (`losses.ContrastiveLoss` + `model.fit`) are deprecated on 5.4.1 —
# `ContrastiveLoss` still imports (with a DeprecationWarning) but the documented
# v4 training path is superseded. This uses `SentenceTransformerTrainer` +
# `TripletLoss`, which is the current API.
#
# VERIFICATION STATUS — read before trusting a number from this cell:
#   confirmed  the API surface, by reading the installed 5.4.1 package source:
#              SentenceTransformerTrainer / SentenceTransformerTrainingArguments
#              are top-level exports; TripletLoss is at losses/triplet.py with
#              ctor (model, distance_metric, margin); InputExample is exported.
#   confirmed  the loop logic mirrors the tracer (same frame, same
#              session_disjoint_folds, same nearest-centroid decision rule), so
#              its macro-F1 is comparable to the CPU logistic head's.
#   NOT run    the training loop has never executed end to end — every local
#              attempt stalled on the torch/datasets install on this machine.
#              The first Colab run is the first execution. If it fails, the
#              failure is as likely to be the environment as this code.
#   NOT a ceiling  the fixture cannot measure distillation (see the README), so
#              compare this against the CPU head only as "does the arm run and
#              roughly match", never as "encoder_ft is better/worse".

import subprocess, sys, os, time, json, random

# Install Colab-side. Deliberately does NOT reinstall torch: Colab ships a
# CUDA-matched build, and a plain `pip install sentence-transformers` can pull a
# CPU-only torch wheel that silently makes the L4 useless (training then runs on
# CPU, ~20-60 min, and the wall-time print is the only clue). `--no-deps` for
# torch is not enough (the resolver still upgrades), so instead we let pip see
# what is already satisfied and re-assert the torch build afterwards.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers>=5", "scikit-learn", "pandas", "pyyaml",
                "datasets"], check=True)

import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
if not torch.cuda.is_available():
    print("\nCUDA IS NOT AVAILABLE. Either the runtime is not a GPU (set "
          "Runtime > L4 GPU), or the install replaced Colab's CUDA torch with a "
          "CPU wheel. Fix with:\n"
          "    !pip install -q --force-reinstall torch --index-url "
          "https://download.pytorch.org/whl/cu121\n"
          "then Runtime > Restart session and re-paste.")
    raise SystemExit(1)

REPO = "https://github.com/evanokeefe39/agent-turn-classifier.git"
if not os.path.isdir("/content/atc"):
    subprocess.run(["git", "clone", "-q", REPO, "/content/atc"], check=True)
os.chdir("/content/atc")
sys.path.insert(0, "src")

import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
assert torch.cuda.is_available(), "set Runtime > L4 GPU before pasting"

import numpy as np
from pathlib import Path
from agent_turn_classifier import tracer as T

turns = T.build_turns(T.read_canonical("examples/sessions.sample.jsonl"))
views = {t.turn_id: T.render(t) for t in turns}
canned = {json.loads(l)["view_sha256"]: json.loads(l)["workflow"]
          for l in Path("examples/canned_labels.jsonl").read_text().splitlines() if l.strip()}
tl = {t.turn_id: canned.get(T.view_sha(views[t.turn_id]), "unmapped") for t in turns}
truth = {json.loads(l)["turn_id"]: json.loads(l)["workflow"]
         for l in Path("examples/sessions.sample.labels.jsonl").read_text().splitlines() if l.strip()}

workflows, _ = T.load_ontology("examples/ontology.example.yaml")
wf_domain = {w.id: w.domain for w in workflows}
path = lambda w: w if w in T.RESERVED_NON_TRAINABLE else (
    f"{wf_domain[w]}/{w}" if w in wf_domain else "unmapped")

frame = T.build_frame(turns, views, {k: path(v) for k, v in tl.items()})
frame["truth"] = [path(truth[t]) for t in frame["turn_id"]]
print(f"{len(frame)} scored turns, {frame['session_id'].nunique()} sessions, "
      f"{frame['label'].nunique()} classes")

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import TripletLoss
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score

ENCODER = "answerdotai/ModernBERT-base"
splits = T.session_disjoint_folds(frame, n_splits=3)
y_true, y_pred, times = [], [], []

for i, (tr, te) in enumerate(splits, 1):
    t0 = time.time()
    tr_df, te_df = frame.iloc[tr], frame.iloc[te]
    model = SentenceTransformer(ENCODER)

    # Triplets: anchor, a same-class positive, a random different-class negative.
    #
    # The first draft picked the negative from the ADJACENT label only
    # (labels[(j+1) % n]) and always used pos_pool[0] as the positive. That
    # contrasts each class against exactly one cyclic neighbour — 6 of 15
    # possible pairs on a 6-class problem — and makes every positive for a class
    # the same fixed example, which would understate the arm and risk rejecting
    # encoder_ft for a sampling artefact. Seeded, so a re-run is reproducible.
    rng = random.Random(20260922)
    by_label = {}
    for _, r in tr_df.iterrows():
        by_label.setdefault(r["label"], []).append(r["view"])
    labels = list(by_label)
    rows = []
    for lab in labels:
        pos_pool = by_label[lab]
        other = [l for l in labels if l != lab]
        if not other:
            continue
        for a in pos_pool:
            pos = rng.choice([v for v in pos_pool if v != a] or pos_pool)
            neg = rng.choice(by_label[rng.choice(other)])
            rows.append({"anchor": a, "positive": pos, "negative": neg})
    if not rows:
        continue
    ds = Dataset.from_list(rows)
    loss = TripletLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=f"/tmp/ft_{i}", num_train_epochs=4, per_device_train_batch_size=16,
        warmup_ratio=0.1, logging_steps=999, save_strategy="no", report_to=[],
        # bf16, not fp16: the L4 supports bf16 and fp16 can produce NaN loss on
        # ModernBERT. Only enabled when CUDA is present.
        bf16=torch.cuda.is_available(), disable_tqdm=True)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()

    # Classify by nearest class-centroid in the fine-tuned space.
    emb_tr = model.encode(list(tr_df["view"]), normalize_embeddings=True, show_progress_bar=False)
    cent, labs = [], []
    for lab in sorted(set(tr_df["label"])):
        idx = [k for k, l in enumerate(tr_df["label"]) if l == lab]
        v = emb_tr[idx].mean(0); cent.append(v / np.linalg.norm(v)); labs.append(lab)
    C = np.stack(cent)
    emb_te = model.encode(list(te_df["view"]), normalize_embeddings=True, show_progress_bar=False)
    pred = [labs[j] for j in (emb_te @ C.T).argmax(1)]

    y_true.extend(te_df["truth"]); y_pred.extend(pred)
    times.append(time.time() - t0)
    print(f"  fold {i}: {times[-1]:.0f}s  n_test={len(te_df)}")

score = f1_score(y_true, y_pred, average="macro", zero_division=0)
print(f"\nencoder_ft  macro-F1 vs truth: {score:.3f}")
print(f"total wall {sum(times):.0f}s   per fold {[round(t) for t in times]}")
print("\nCPU reference (bge-small + logistic head): 0.982 — but see the README: "
      "this fixture cannot measure distillation, so compare only the plumbing, "
      "not the magnitudes.")
