"""Hebrew nikud classifier model — per-character prediction of nikud class and shin/sin dot."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from constants import (
    ENCODER_MODEL,
    NUM_NIKUD_CLASSES,
    NUM_SHIN_CLASSES,
    IGNORE_INDEX,
)
from tokenization import unwrap_encoder_model


class HebrewNikudClassifier(nn.Module):
    """
    Per-character Hebrew nikud prediction model.

    For each Hebrew letter in the input, predicts:
      - nikud class  (one of NIKUD_CLASSES: empty / MAT_LECT / dagesh / vowel / dagesh+vowel)
      - shin/sin dot (SHIN_CLASSES; IGNORE_INDEX for all letters except ש)

    Non-Hebrew characters (spaces, punctuation, digits, Latin) are passed
    through unchanged at inference — the heads are never called for them.
    """

    def __init__(self, encoder_model: str = ENCODER_MODEL, dropout_rate: float = 0.1) -> None:
        super().__init__()

        encoder = AutoModel.from_pretrained(encoder_model, trust_remote_code=True)
        self.encoder = unwrap_encoder_model(encoder)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        self.nikud_head = nn.Linear(hidden_size, NUM_NIKUD_CLASSES)
        self.shin_head = nn.Linear(hidden_size, NUM_SHIN_CLASSES)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        nikud_labels: torch.Tensor | None = None,
        shin_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = self.dropout(encoder_outputs.last_hidden_state)  # [B, S, H]

        nikud_logits = self.nikud_head(hidden)  # [B, S, NUM_NIKUD_CLASSES]
        shin_logits = self.shin_head(hidden)    # [B, S, NUM_SHIN_CLASSES]

        output: dict[str, torch.Tensor] = {
            "nikud_logits": nikud_logits,
            "shin_logits": shin_logits,
        }

        if nikud_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            nikud_loss = loss_fct(nikud_logits.view(-1, NUM_NIKUD_CLASSES), nikud_labels.view(-1))
            shin_loss = loss_fct(shin_logits.view(-1, NUM_SHIN_CLASSES), shin_labels.view(-1))
            output["loss"] = nikud_loss + shin_loss

        return output

    def parameter_groups(self, encoder_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        """Discriminative LRs: lower for encoder, higher for classification heads."""
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

        def is_no_decay(name: str) -> bool:
            return any(term in name for term in no_decay)

        return [
            {
                "params": [p for n, p in self.encoder.named_parameters() if not is_no_decay(n)],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.encoder.named_parameters() if is_no_decay(n)],
                "lr": encoder_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not n.startswith("encoder.") and not is_no_decay(n)
                ],
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not n.startswith("encoder.") and is_no_decay(n)
                ],
                "lr": head_lr,
                "weight_decay": 0.0,
            },
        ]
