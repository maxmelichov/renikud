#!/usr/bin/env bash
# Nikud classifier — predict Hebrew vowel diacritics from unvocalized text
# Training data: vocalized Hebrew text, one sentence per line (TSV col-1 is used)
# Run from repo root: bash experiments/nikud_classifier.sh
set -euo pipefail

uv run renikud_classifier_300M_model_nikud/src/train.py \
  --train-dataset  dataset/nikud_train.txt \
  --eval-dataset   dataset/nikud_eval.txt \
  --output-dir     outputs/nikud-classifier \
  --train-batch-size 64 \
  --eval-batch-size  64 \
  --epochs         1000 \
  --encoder-lr     2e-5 \
  --head-lr        1e-4 \
  --save-steps     500 \
  --early-stopping-patience  40 \
  --wandb-mode               disabled \
  --device                   ${DEVICE:-cuda:0}
