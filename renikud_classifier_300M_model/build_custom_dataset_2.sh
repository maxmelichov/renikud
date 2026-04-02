#!/bin/bash
set -e

echo "Combining datasets..."
cat dataset/vox-knesset-ipa-v1.tsv dataset/data_dj_j.tsv dataset/13k_manual.tsv | tr -d '\r' | awk -F'\t' '{if (NF==3) print $2"\t"$3; else if (NF==2) print $1"\t"$2}' > dataset/new_train_2.txt

echo "Aligning training data..."
uv run src/align_data.py dataset/new_train_2.txt dataset/new_train_2_alignment.jsonl

echo "Tokenizing training data..."
uv run src/prepare_tokens.py dataset/new_train_2_alignment.jsonl dataset/.cache/classifier-new-train-2

echo "Done!"
