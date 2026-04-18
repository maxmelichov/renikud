#!/usr/bin/env bash
# Knesset CTC G2P — renikud-ctc subproject
# Prereq: run scripts/01_prepare_data.sh first
# Run from renikud-ctc/: bash experiments/knesset_ctc.sh
set -euo pipefail

uv run src/train.py \
  --train-dataset  /home/maxm/renikud/dataset/pairs_filtered.tsv \
  --eval-dataset   /home/maxm/renikud/dataset/gt.tsv \
  --output-dir     outputs/knesset-ctc-metadata \
  --train-batch-size 128 \
  --eval-batch-size  128 \
  --epochs         1000 \
  --save-steps     500 \
  --early-stopping-patience  40 \
  --wandb-mode               disabled \
  --device                   cuda:1 \
  --init-from-checkpoint     outputs/knesset-ctc-metadata/checkpoint-22000
