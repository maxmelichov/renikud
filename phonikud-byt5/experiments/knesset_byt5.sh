#!/usr/bin/env bash
# Knesset ByT5 G2P — phonikud-byt5 subproject
# Prereq: run ../scripts/01_prepare_data.sh first (needs dataset/knesset_split/)
# Run from phonikud-byt5/: bash experiments/knesset_byt5.sh
set -euo pipefail

uv run src/phonikud_byt5/run_train.py \
  --data-dir      ../dataset/knesset_split \
  --ckpt-dir      outputs/knesset-byt5 \
  --model-name    google/byt5-small \
  --batch-size    32 \
  --learning-rate 5e-5 \
  --val-split     0 \
  --split-seed    42 \
  --val-file      ../dataset/pred.tsv \
  --eval-steps    500 \
  --max-steps     20000 \
  --wandb-mode    disabled
