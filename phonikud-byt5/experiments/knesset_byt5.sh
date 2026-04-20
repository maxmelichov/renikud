#!/usr/bin/env bash
# Knesset ByT5 G2P — phonikud-byt5 subproject
# Prereq: run ../scripts/01_prepare_data.sh first (needs dataset/knesset_split/)
# Safe to run from repo root or phonikud-byt5/: bash phonikud-byt5/experiments/knesset_byt5.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

uv run torchrun --nproc_per_node=2 src/phonikud_byt5/run_train.py \
  --data_dir      /home/maxm/renikud/dataset/pairs_filtered.tsv \
  --ckpt_dir      outputs/knesset-byt5-ASR \
  --model_name    google/byt5-small \
  --batch_size    8 \
  --learning_rate 5e-5 \
  --val_split     0 \
  --split_seed    42 \
  --val_file      /home/maxm/renikud/dataset/gt.tsv \
  --eval_steps    500 \
  --wandb_mode    disabled
