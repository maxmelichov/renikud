#!/usr/bin/env bash
# Knesset CTC G2P — renikud-ctc subproject
# Prereq: run scripts/01_prepare_data.sh first
# Run from renikud-ctc/: bash experiments/knesset_ctc.sh
set -euo pipefail

uv run src/train.py \
  --train-dataset  dataset/.cache/knesset_train \
  --eval-dataset   dataset/.cache/pred_val \
  --output-dir     outputs/knesset-ctc \
  --train-batch-size 32 \
  --eval-batch-size  32 \
  --epochs         3 \
  --save-steps     500 \
  --max-steps      20000 \
  --wandb-mode     disabled
