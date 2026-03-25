#!/usr/bin/env bash
# One-time data preparation for renikud-ctc experiments
# Run from renikud-ctc/: bash scripts/01_prepare_data.sh
set -euo pipefail

# ── Step 1: Split raw knesset data (all lines → train) ────────────────────
cd ..
uv run src/prepare_data.py \
  --input      dataset/knesset_phonemes_v1.txt \
  --output-dir dataset/knesset_split \
  --lines      5000000 \
  --val-ratio  0 \
  --seed       42
cd renikud-ctc

# ── Step 2: Tokenize train → Arrow ────────────────────────────────────────
uv run src/prepare_tokens.py \
  --input  ../dataset/knesset_split/train.txt \
  --output dataset/.cache/knesset_train

# ── Step 3: Tokenize val (pred.tsv: strip 3-col header) ───────────────────
mkdir -p dataset/.cache
awk -F'\t' 'NR>1 && NF>=2 {print $1 "\t" $2}' ../dataset/pred.tsv \
  > dataset/.cache/pred_2col.tsv

uv run src/prepare_tokens.py \
  --input  dataset/.cache/pred_2col.tsv \
  --output dataset/.cache/pred_val
