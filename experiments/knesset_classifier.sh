#!/usr/bin/env bash
# Knesset G2P Classifier — renikud (root)
#
# Run from repo root:
#   bash experiments/knesset_classifier.sh
set -euo pipefail

# ── Step 1: Split raw knesset data (all lines → train, no val split) ──────
uv run src/prepare_data.py \
  --input      dataset/knesset_phonemes_v1.txt \
  --output-dir dataset/knesset_split \
  --lines      2000000 \
  --val-ratio  0

# ── Step 2: Align train split (Hebrew → IPA chunks) ───────────────────────
uv run src/align_data.py \
  dataset/knesset_split/train.txt \
  dataset/knesset_split/train_alignment.jsonl

# ── Step 3: Align validation set (pred.tsv: strip 3-col header → 2-col) ───
mkdir -p dataset/.cache
awk -F'\t' 'NR>1 && NF>=2 {print $1 "\t" $2}' dataset/pred.tsv \
  > dataset/.cache/pred_2col.tsv

uv run src/align_data.py \
  dataset/.cache/pred_2col.tsv \
  dataset/knesset_split/val_alignment.jsonl

# ── Step 4: Tokenize train → Arrow ────────────────────────────────────────
uv run src/prepare_tokens.py \
  dataset/knesset_split/train_alignment.jsonl \
  dataset/.cache/knesset_train

# ── Step 5: Tokenize val → Arrow ──────────────────────────────────────────
uv run src/prepare_tokens.py \
  dataset/knesset_split/val_alignment.jsonl \
  dataset/.cache/knesset_val

# ── Step 6: Train ─────────────────────────────────────────────────────────
uv run src/train.py \
  --train-dataset  dataset/.cache/knesset_train \
  --eval-dataset   dataset/.cache/knesset_val \
  --output-dir     outputs/knesset-classifier \
  --train-batch-size 32 \
  --eval-batch-size  32 \
  --epochs         3 \
  --encoder-lr     2e-5 \
  --head-lr        1e-4 \
  --save-steps     500 \
  --max-steps      20000 \
  --wandb-mode     disabled
