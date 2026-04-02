"""Evaluation helpers for the Hebrew G2P classifier."""

from __future__ import annotations

import torch
from jiwer import cer, wer

from constants import IGNORE_INDEX
from decoder import decode


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Per-token accuracy ignoring IGNORE_INDEX positions."""
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).float().mean().item()


def evaluate(model, eval_loader, device, fp16: bool, tokenizer=None, max_len: int = 256) -> dict:
    model.eval()
    total_loss = 0.0
    consonant_acc_sum = vowel_acc_sum = stress_acc_sum = 0.0
    n = 0
    refs, hyps = [], []


    with torch.no_grad():
        for batch in eval_loader:
            texts = batch.pop("texts", None)
            ref_ipas = batch.pop("ref_ipas", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=fp16):
                out = model(**batch)
            total_loss += out["loss"].item()
            consonant_acc_sum += compute_accuracy(out["consonant_logits"], batch["consonant_labels"])
            vowel_acc_sum += compute_accuracy(out["vowel_logits"], batch["vowel_labels"])
            stress_acc_sum += compute_accuracy(out["stress_logits"], batch["stress_labels"])
            n += 1

            if tokenizer is not None and texts:
                for i, text in enumerate(texts):
                    enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
                    hyp = decode(
                        text=text,
                        offset_mapping=enc["offset_mapping"],
                        consonant_logits=out["consonant_logits"][i],
                        vowel_logits=out["vowel_logits"][i],
                        stress_logits=out["stress_logits"][i],
                    )
                    hyps.append(hyp)
                    refs.append(ref_ipas[i] if ref_ipas else "")

    model.train()
    metrics = {
        "eval_loss": total_loss / n,
        "consonant_acc": consonant_acc_sum / n,
        "vowel_acc": vowel_acc_sum / n,
        "stress_acc": stress_acc_sum / n,
        "mean_acc": (consonant_acc_sum + vowel_acc_sum + stress_acc_sum) / (3 * n),
    }

    if hyps:
        mean_wer = sum(wer(r, h) for r, h in zip(refs, hyps)) / len(refs)
        mean_cer = sum(cer(r, h) for r, h in zip(refs, hyps)) / len(refs)
        metrics["wer"] = mean_wer
        metrics["cer"] = mean_cer
        metrics["acc"] = 1 - mean_wer
        metrics["refs"] = refs
        metrics["hyps"] = hyps

    return metrics
