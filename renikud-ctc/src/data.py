"""Dataset loading and collation utilities for Hebrew G2P."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk

from constants import MAX_LEN
from tokenization import encode_ipa, load_encoder_tokenizer


def _load_tsv(path: str) -> Dataset:
    """Load a raw TSV (hebrew<TAB>ipa[<TAB>field...]) and tokenize up-front."""
    tokenizer = load_encoder_tokenizer()
    rows = {"encoder_ids": [], "encoder_mask": [], "decoder_ids": []}
    skipped = 0
    for line in Path(path).read_text(encoding="utf-8").strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 2:
            skipped += 1
            continue
        hebrew, ipa = parts[0].strip(), parts[1].strip()
        if not hebrew or not ipa:
            skipped += 1
            continue
        # Skip header row and any line whose first column isn't Hebrew.
        if not any("\u0590" <= c <= "\u05FF" for c in hebrew):
            skipped += 1
            continue
        enc = tokenizer(hebrew, truncation=True, max_length=MAX_LEN, return_tensors="np")
        try:
            dec = encode_ipa(ipa)
        except ValueError:
            skipped += 1
            continue
        rows["encoder_ids"].append(enc["input_ids"][0].tolist())
        rows["encoder_mask"].append(enc["attention_mask"][0].tolist())
        rows["decoder_ids"].append(dec)
    if skipped:
        print(f"Skipped {skipped} rows from {path}")
    return Dataset.from_dict(rows)


def load_tokenized_dataset(path: str):
    """Load a dataset from either a raw TSV or a pretokenized Arrow directory."""
    if path.endswith(".tsv") or path.endswith(".txt"):
        return _load_tsv(path)
    return load_from_disk(path)


def load_dataset_splits(train_path: str, eval_path: str):
    """Load train/eval datasets from disk."""
    train_dataset = load_tokenized_dataset(train_path)
    eval_dataset = load_tokenized_dataset(eval_path)
    return train_dataset, eval_dataset


@dataclass
class G2PDataCollator:
    """Pad tokenized encoder inputs and decoder labels for CTC training."""

    encoder_pad_id: int = 0
    label_pad_id: int = -100

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_input_len = max(len(feature["encoder_ids"]) for feature in features)
        max_label_len = max(len(feature["decoder_ids"]) for feature in features)

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            encoder_ids = list(feature["encoder_ids"])
            encoder_mask = list(feature["encoder_mask"])
            decoder_ids = list(feature["decoder_ids"])

            input_pad_len = max_input_len - len(encoder_ids)
            label_pad_len = max_label_len - len(decoder_ids)

            input_ids.append(encoder_ids + [self.encoder_pad_id] * input_pad_len)
            attention_mask.append(encoder_mask + [0] * input_pad_len)
            labels.append(decoder_ids + [self.label_pad_id] * label_pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
