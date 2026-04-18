"""
Prepare tokenized Arrow dataset for nikud classifier training.

Reads JSONL produced by align_data.py (bare_hebrew -> [[char, nikud_label], ...])
and produces an Arrow dataset with per-token nikud and shin labels.

Usage:
    uv run src/prepare_tokens.py dataset/train_alignment.jsonl dataset/.cache/classifier-train
    uv run src/prepare_tokens.py dataset/val_alignment.jsonl   dataset/.cache/classifier-val
"""

from __future__ import annotations

import argparse
import json
import unicodedata

import datasets
from tqdm import tqdm

from constants import (
    NIKUD_TO_ID,
    SHIN_TO_ID,
    SHIN_LETTER,
    IGNORE_INDEX,
    is_hebrew_letter,
    strip_nikud,
)
from tokenization import load_encoder_tokenizer


def _extract_shin(vocalized: str) -> dict[int, str]:
    """
    Return a mapping from bare-string char position -> shin/sin dot char
    for each ש in the vocalized text.
    """
    nfd = unicodedata.normalize("NFD", vocalized)
    result: dict[int, str] = {}
    bare_pos = 0
    i = 0
    while i < len(nfd):
        ch = nfd[i]
        i += 1
        # collect combining marks
        combining: list[str] = []
        while i < len(nfd) and unicodedata.category(nfd[i]).startswith("M"):
            combining.append(nfd[i])
            i += 1
        if is_hebrew_letter(ch):
            if ch == SHIN_LETTER:
                for c in combining:
                    if c in SHIN_TO_ID:
                        result[bare_pos] = c
                        break
            bare_pos += 1
    return result


def build_token_labels(bare: str, pairs: list[list[str]], vocalized: str, tokenizer) -> dict | None:
    """
    Tokenize bare Hebrew and assign per-token nikud + shin labels.
    pairs: list of [char, nikud_label] from the JSONL.
    vocalized: original vocalized text (used to extract shin dot positions).
    """
    shin_map = _extract_shin(vocalized)

    encoding = tokenizer(
        bare,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"]
    seq_len = len(input_ids)

    nikud_labels = [IGNORE_INDEX] * seq_len
    shin_labels = [IGNORE_INDEX] * seq_len

    # Map bare char position -> (nikud_label, shin_label_or_None)
    char_to_nikud: dict[int, str] = {}
    pair_iter = iter(pairs)
    heb_pos = 0
    for char_pos, ch in enumerate(bare):
        if not is_hebrew_letter(ch):
            continue
        try:
            _, nikud_label = next(pair_iter)
        except StopIteration:
            break
        char_to_nikud[char_pos] = nikud_label
        heb_pos += 1

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if end - start != 1:
            continue
        if start not in char_to_nikud:
            continue
        nikud_label = char_to_nikud[start]
        nikud_labels[tok_idx] = NIKUD_TO_ID.get(nikud_label, IGNORE_INDEX)

        shin_dot = shin_map.get(start)
        if shin_dot is not None:
            shin_labels[tok_idx] = SHIN_TO_ID.get(shin_dot, IGNORE_INDEX)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "nikud_labels": nikud_labels,
        "shin_labels": shin_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare classifier training tokens from aligned JSONL")
    parser.add_argument("input", help="Input JSONL file (from align_data.py)")
    parser.add_argument("output", help="Output Arrow dataset directory")
    args = parser.parse_args()

    tokenizer = load_encoder_tokenizer()
    records = []
    skipped = 0

    with open(args.input, encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Tokenizing"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        bare, pairs = next(iter(obj.items()))
        # pairs is [[char, nikud_label], ...]
        # We don't have the original vocalized text here, so reconstruct shin from pairs
        # (shin info is not in pairs — re-derive from bare only if needed)
        record = build_token_labels(bare, pairs, bare, tokenizer)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    print(f"\nProcessed: {len(records):,}")
    print(f"Skipped:   {skipped:,}")

    dataset = datasets.Dataset.from_list(records)
    dataset.save_to_disk(args.output)
    print(f"Saved to:  {args.output}")


if __name__ == "__main__":
    main()
