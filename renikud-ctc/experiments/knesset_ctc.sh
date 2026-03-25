#!/usr/bin/env bash
# Knesset CTC G2P — renikud-ctc subproject
#
# Run from renikud-ctc/:
#   bash experiments/knesset_ctc.sh
#
# This subproject has no early stopping; training runs for --epochs epochs.
#
# Prerequisite: the knesset split must already exist at ../dataset/knesset_split/
# If not, run from the repo root first:
#   uv run src/prepare_data.py --input dataset/knesset_phonemes_v1.txt \
#     --output-dir dataset/knesset_split --lines 5000000 --max-val 0

set -euo pipefail

# ── Step 1a: Tokenize knesset train split → Arrow dataset ─────────────────
uv run src/prepare_tokens.py \
  --input  ../dataset/knesset_split/train.txt \
  --output dataset/.cache/knesset_train

# ── Step 1b: Tokenize val set (pred.tsv, 3-col → extract first 2 cols) ────
# renikud-ctc's prepare_tokens.py expects exactly 2-column TSV (Hebrew\tIPA).
PRED_TSV_2COL="dataset/.cache/pred_2col.tsv"
mkdir -p dataset/.cache
# Extract columns 1 and 2, skip the header line that starts with "Sentence"
awk -F'\t' 'NR>1 && NF>=2 {print $1 "\t" $2}' ../dataset/pred.tsv > "$PRED_TSV_2COL"

uv run src/prepare_tokens.py \
  --input  "$PRED_TSV_2COL" \
  --output dataset/.cache/pred_val

# ── Step 2: Train ──────────────────────────────────────────────────────────
uv run src/train.py \
  --train-dataset              dataset/.cache/knesset_train \
  --eval-dataset               dataset/.cache/pred_val \
  --output-dir                 outputs/knesset-ctc \
  --train-batch-size           32 \
  --epochs                     3 \
  --save-steps                 500 \
  --wandb-mode                 online
