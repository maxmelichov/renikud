#!/usr/bin/env bash
# Knesset G2P Classifier — renikud (root)
# Prereq: run scripts/01_prepare_data.sh first
# Run from repo root: bash experiments/knesset_classifier.sh
set -euo pipefail

uv run src/train.py \
  --train-dataset  dataset/vox-knesset-ipa-v1.tsv \
  --eval-dataset   dataset/pred_alignment.jsonl \
  --output-dir     outputs/knesset-classifier-vox \
  --train-batch-size 56 \
  --eval-batch-size  56 \
  --epochs         3 \
  --encoder-lr     2e-5 \
  --head-lr        1e-4 \
  --save-steps     500 \
  --early-stopping-patience  40 \
  --wandb-mode               disabled \
  --device                   ${DEVICE:-cuda:0}
