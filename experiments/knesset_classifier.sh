#!/usr/bin/env bash
# Knesset G2P Classifier — renikud (root)
# Prereq: run scripts/01_prepare_data.sh first
# Run from repo root: bash experiments/knesset_classifier.sh
set -euo pipefail

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
  --max-steps                20000 \
  --early-stopping-patience  40 \
  --wandb-mode               disabled
