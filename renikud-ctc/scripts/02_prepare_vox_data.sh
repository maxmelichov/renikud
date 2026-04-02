#!/usr/bin/env bash
# Data preparation for vox-knesset-ipa-v1 experiment
# Run from renikud-ctc/: bash scripts/02_prepare_vox_data.sh
set -euo pipefail

# ── Tokenize train → Arrow ────────────────────────────────────────────────
uv run src/prepare_tokens.py \
  --input  ../dataset/vox-knesset-ipa-v1.tsv \
  --output dataset/.cache/vox-knesset-train

# ── Tokenize val (pred.tsv: strip 3-col header) ───────────────────────────
mkdir -p dataset/.cache
awk -F'\t' 'NR>1 && NF>=2 {print $1 "\t" $2}' ../dataset/pred.tsv \
  > dataset/.cache/pred_2col.tsv

uv run src/prepare_tokens.py \
  --input  dataset/.cache/pred_2col.tsv \
  --output dataset/.cache/pred_val
