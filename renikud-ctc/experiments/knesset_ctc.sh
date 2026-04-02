#!/usr/bin/env bash
# Knesset CTC G2P — renikud-ctc subproject
# Prereq: run scripts/01_prepare_data.sh first
# Run from renikud-ctc/: bash experiments/knesset_ctc.sh
set -euo pipefail

uv run src/train.py \
  --train-dataset  dataset/.cache/knesset-metadata-train \
  --eval-dataset   dataset/.cache/gt-val \
  --output-dir     outputs/knesset-ctc-metadata \
  --train-batch-size 64 \
  --eval-batch-size  64 \
  --epochs         3 \
  --save-steps     500 \
  --early-stopping-patience  40 \
  --wandb-mode               disabled \
  --device                   ${DEVICE:-cuda:1}
