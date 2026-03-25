#!/usr/bin/env bash
# Knesset CTC G2P — renikud-ctc subproject
#
# Run from renikud-ctc/:
#   bash experiments/knesset_ctc.sh
#
# Stops after --max-steps optimizer steps.
# pred.tsv is used as val, evaluated every 500 steps.

set -euo pipefail

# ── Step 1: Split raw knesset data (all lines → train, no val split) ──────
cd ..
uv run src/prepare_data.py \
  --input      dataset/knesset_phonemes_v1.txt \
  --output-dir dataset/knesset_split \
  --lines      2000000 \
  --val-ratio  0 \
  --seed       42
cd renikud-ctc

# ── Step 2: Tokenize knesset train split → Arrow dataset ──────────────────
uv run src/prepare_tokens.py \
  --input  ../dataset/knesset_split/train.txt \
  --output dataset/.cache/knesset_train

# ── Step 3: Tokenize val set (pred.tsv: strip 3-col header → 2-col) ───────
mkdir -p dataset/.cache
awk -F'\t' 'NR>1 && NF>=2 {print $1 "\t" $2}' ../dataset/pred.tsv \
  > dataset/.cache/pred_2col.tsv

uv run src/prepare_tokens.py \
  --input  dataset/.cache/pred_2col.tsv \
  --output dataset/.cache/pred_val

# ── Step 2: Train ──────────────────────────────────────────────────────────
uv run src/train.py \
  --train-dataset              dataset/.cache/knesset_train \
  --eval-dataset               dataset/.cache/pred_val \
  --output-dir                 outputs/knesset-ctc \
  --train-batch-size           32 \
  --epochs                     3 \
  --save-steps                 500 \
  --max-steps                  20000 \
  --wandb-mode                 disabled
