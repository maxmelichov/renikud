#!/usr/bin/env bash
# Knesset G2P Classifier — renikud (root)
# Prereq: run scripts/01_prepare_data.sh first
# Run from repo root: bash experiments/knesset_classifier.sh
set -euo pipefail

uv run src/train.py \
  --train-dataset  dataset/knesset_vox_new_asr_split.tsv \
  --eval-dataset   dataset/gt.tsv \
  --output-dir     outputs/knesset-classifier-split \
  --train-batch-size 64 \
  --eval-batch-size  64 \
  --epochs         3 \
  --encoder-lr     2e-5 \
  --head-lr        1e-4 \
  --save-steps     500 \
  --early-stopping-patience  40 \
  --wandb-mode               disabled \
  --device                   ${DEVICE:-cuda:0}
