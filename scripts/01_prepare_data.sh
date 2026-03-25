#!/usr/bin/env bash
# One-time data preparation for all experiments
# Run from repo root: bash scripts/01_prepare_data.sh
set -euo pipefail

# ── Step 1: Split raw knesset data (all lines → train) ────────────────────
uv run src/prepare_data.py \
  --input      dataset/knesset_phonemes_v1.txt \
  --output-dir dataset/knesset_split \
  --lines      5000000 \
  --val-ratio  0 \
  --seed       42

# ── Step 2: Align train split ──────────────────────────────────────────────
uv run src/align_data.py \
  dataset/knesset_split/train.txt \
  dataset/knesset_split/train_alignment.jsonl

# ── Step 3: Align pred.tsv (strip 3-col header) ───────────────────────────
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
