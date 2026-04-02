#!/bin/bash
set -e

echo "Extracting 2,000,000 lines for training..."
head -n 2000000 dataset/train.txt > dataset/train_2m.txt

echo "Aligning training data (2M lines)..."
uv run src/align_data.py dataset/train_2m.txt dataset/train_2m_alignment.jsonl

echo "Tokenizing training data (2M lines)..."
uv run src/prepare_tokens.py dataset/train_2m_alignment.jsonl dataset/.cache/classifier-train-2m

echo "Done!"
