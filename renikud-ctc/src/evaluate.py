"""Evaluation helpers for Hebrew G2P training."""

from __future__ import annotations

import numpy as np
from jiwer import cer, wer

from tokenization import decode_ctc, decode_ipa


def compute_metrics(logits: np.ndarray, input_lengths: np.ndarray, labels: np.ndarray) -> dict:
    pred_texts = [
        decode_ctc(logits[i].argmax(-1)[:input_lengths[i]].tolist())
        for i in range(len(logits))
    ]
    label_texts = [
        decode_ipa([t for t in row if t != -100])
        for row in labels
    ]
    mean_wer = sum(wer(r, h) for r, h in zip(label_texts, pred_texts)) / len(label_texts)
    mean_cer = sum(cer(r, h) for r, h in zip(label_texts, pred_texts)) / len(label_texts)
    return {
        "cer": mean_cer,
        "wer": mean_wer,
        "acc": 1 - mean_wer,
        "refs": label_texts,
        "hyps": pred_texts,
    }
