#!/bin/bash
set -e

echo "Combining datasets..."
# For youtube-leomanut-v1.tsv, we need to remove the leading tab
awk -F'\t' '{if (NF==3) print $2"\t"$3; else if (NF==2) print $1"\t"$2}' dataset/youtube-leomanut-v1.tsv > dataset/new_train.txt
# For data_dj_j.tsv, just append
cat dataset/data_dj_j.tsv >> dataset/new_train.txt

echo "Aligning training data..."
uv run src/align_data.py dataset/new_train.txt dataset/new_train_alignment.jsonl

echo "Tokenizing training data..."
uv run src/prepare_tokens.py dataset/new_train_alignment.jsonl dataset/.cache/classifier-new-train

echo "Done!"
