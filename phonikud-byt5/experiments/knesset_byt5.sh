#!/usr/bin/env bash
# Knesset ByT5 G2P — phonikud-byt5 subproject
# Prereq: run ../scripts/01_prepare_data.sh first (needs dataset/knesset_split/)
# Run from phonikud-byt5/: bash experiments/knesset_byt5.sh
set -euo pipefail

uv run torchrun --nproc_per_node=2 src/phonikud_byt5/run_train.py \
  --data_dir      ../dataset/vox-knesset-ipa-v1.tsv \
  --ckpt_dir      outputs/knesset-byt5-vox \
  --model_name    google/byt5-small \
  --batch_size    8 \
  --learning_rate 5e-5 \
  --val_split     0 \
  --split_seed    42 \
  --val_file      ../dataset/pred.tsv \
  --eval_steps    500 \
  --wandb_mode    disabled
