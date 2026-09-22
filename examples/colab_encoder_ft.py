# PASTE THIS INTO YOUR COLAB L4 SESSION
#
# Purpose: run the `encoder_ft` arm (the one thing a GPU actually accelerates)
# and confirm the L4 sees a real speedup over the 20-60 min CPU estimate.
#
# Before you paste: Runtime > Change runtime type > L4 GPU.
# Time: ~3-5 min (install ~2 min, fine-tune ~1-3 min).
#
# What it does: fine-tunes a ModernBERT-class encoder on the tracer's synthetic
# corpus, 3 session-disjoint folds, and reports macro-F1 vs truth plus wall time.
# Same data and metric as the CPU tracer, so the numbers are comparable.

import subprocess, sys, os, time, json

# 1. Install. torch is already present on Colab; sentence-transformers is not.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers", "scikit-learn", "pandas", "pyyaml"],
               check=True)

# 2. Clone the repo at the merged commit.
REPO = "https://github.com/evanokeefe39/agent-turn-classifier.git"
if not os.path.isdir("/content/atc"):
    subprocess.run(["git", "clone", "-q", REPO, "/content/atc"], check=True)
os.chdir("/content/atc")
sys.path.insert(0, "src")

# 3. Confirm we got a GPU (a CPU run here would take 20-60 min, not 3).
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
assert torch.cuda.is_available(), "set Runtime > L4 GPU before pasting"

# 4. Load the corpus and build the frame exactly as the tracer does.
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

wfs, domains = T.load_ontology("examples/ontology.example.yaml")
wf_dom = {w.id: w.domain for w in wfs}
path = lambda w: w if w in T.RESERVED_NON_TRAINABLE else (
    f"{wf_dom[w]}/{w}" if w in wf_dom else "unmapped")

frame = T.build_frame(turns, views, {k: path(v) for k, v in tl.items()})
frame["truth"] = [path(truth[t]) for t in frame["turn_id"]]
print(f"{len(frame)} scored turns, {frame['session_id'].nunique()} sessions, "
      f"{frame['label'].nunique()} classes")

# 5. Fine-tune a ModernBERT-class encoder per fold: the `encoder_ft` arm.
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score

ENCODER = "answerdotai/ModernBERT-base"   # ~150M params, encoder-only
splits = T.session_disjoint_folds(frame, n_splits=3)
y_true, y_pred, fold_times = [], [], []

for i, (tr, te) in enumerate(splits, 1):
    t0 = time.time()
    m = SentenceTransformer(ENCODER)
    tr_df, te_df = frame.iloc[tr], frame.iloc[te]

    from sentence_transformers import InputExample
    from torch.utils.data import DataLoader

    # Contrastive pairs: same-class positive, different-class negative.
    examples = []
    by_label = {}
    for _, r in tr_df.iterrows():
        by_label.setdefault(r["label"], []).append(r["view"])
    labels = list(by_label)
    for lab, items in by_label.items():
        for v in items:
            others = [o for o in labels if o != lab]
            if not others:
                continue
            neg_lab = others[i % len(others)]
            examples.append(InputExample(texts=[v, by_label[neg_lab][0]]))

    loader = DataLoader(examples, shuffle=True, batch_size=16)
    m.fit(train_objectives=[(loader, __import__("sentence_transformers").losses.ContrastiveLoss(m))],
          epochs=4, warmup_steps=10, show_progress_bar=False)

    # Classify by nearest class-centroid in the fine-tuned space.
    emb_tr = m.encode(list(tr_df["view"]), normalize_embeddings=True)
    cent = {lab: emb_tr[[j for j, l in enumerate(tr_df["label"]) if l == lab]].mean(0)
            for lab in set(tr_df["label"])}
    C = np.stack([v / np.linalg.norm(v) for v in cent.values()])
    labs = list(cent)
    emb_te = m.encode(list(te_df["view"]), normalize_embeddings=True)
    pred = [labs[j] for j in (emb_te @ C.T).argmax(1)]

    y_true.extend(te_df["truth"]); y_pred.extend(pred)
    fold_times.append(time.time() - t0)
    print(f"  fold {i}: {time.time()-t0:.0f}s  n_test={len(te_df)}")

print(f"\nencoder_ft  macro-F1 vs truth: {f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
print(f"total wall: {sum(fold_times):.0f}s   per-fold: {[round(t) for t in fold_times]}")
print(f"\nCPU reference (bge-small + logistic): 0.982 — see README for what that does and does not mean")
