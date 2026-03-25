#!/usr/bin/env bash
# Knesset ByT5 G2P — phonikud-byt5 subproject
#
# Run from phonikud-byt5/:
#   bash experiments/knesset_byt5.sh
#
# Training reads all *.txt files in --data-dir; val split is carved out
# internally (1% by default via --val-split).
#
# Prerequisite: the knesset split must already exist at ../dataset/knesset_split/
# If not, run from the repo root first:
#   uv run src/prepare_data.py --input dataset/knesset_phonemes_v1.txt \
#     --output-dir dataset/knesset_split --lines 5000000 --max-val 0

set -euo pipefail

uv run src/phonikud_byt5/run_train.py \
  --data-dir    ../dataset/knesset_split \
  --ckpt-dir    outputs/knesset-byt5 \
  --model-name  google/byt5-small \
  --batch-size  32 \
  --learning-rate 5e-5 \
  --val-split   0.01 \
  --eval-steps  500 \
  --wandb-mode  online \
  --wandb-project phonikud \
  --wandb-entity  Phonikud
