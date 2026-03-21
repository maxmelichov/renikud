"""Hebrew G2P classifier model — per-character prediction of consonant, vowel, and stress."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

from constants import (
    NUM_CONSONANT_CLASSES,
    NUM_VOWEL_CLASSES,
    NUM_STRESS_CLASSES,
    HEBREW_LETTER_TO_ALLOWED_CONSONANTS,
    IGNORE_INDEX,
    is_hebrew_letter,
)


# ---------------------------------------------------------------------------
# NeoBERT building blocks
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, x = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * x)


def precompute_freqs(dim: int, end: int, theta: float = 10000.0, device=None):
    h = dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, h, device=device, dtype=torch.float32) * 2.0 / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return angles.cos(), angles.sin()


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]):
    h = xq.shape[-1] // 2
    xq1, xq2 = xq[..., :h], xq[..., h:]
    xk1, xk2 = xk[..., :h], xk[..., h:]
    cos, sin = freqs
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return (
        torch.cat([xq1 * cos - xq2 * sin, xq1 * sin + xq2 * cos], dim=-1),
        torch.cat([xk1 * cos - xk2 * sin, xk1 * sin + xk2 * cos], dim=-1),
    )


class NeoBERTConfig(PretrainedConfig):
    model_type = "neobert"

    def __init__(
        self,
        vocab_size: int = 841,
        hidden_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        norm_eps: float = 1e-6,
        pad_token_id: int = 0,
        max_length: int = 1024,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dim_head = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.norm_eps = norm_eps
        self.max_length = max_length


class EncoderBlock(nn.Module):
    def __init__(self, config: NeoBERTConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size, config.hidden_size)
        self.attention_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, freqs: tuple):
        x = x + self._attn(self.attention_norm(x), attention_mask, freqs)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def _attn(self, x: torch.Tensor, attention_mask: torch.Tensor, freqs: tuple):
        B, L, _ = x.shape
        H, D = self.config.num_attention_heads, self.config.dim_head
        xq, xk, xv = self.qkv(x).view(B, L, H, D * 3).chunk(3, dim=-1)
        xq, xk = apply_rotary_emb(xq, xk, freqs)
        attn = scaled_dot_product_attention(
            xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2),
            attn_mask=attention_mask.bool(), dropout_p=0,
        ).transpose(1, 2)
        return self.wo(attn.reshape(B, L, H * D))


class NeoBERT(PreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)
        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        cos, sin = precompute_freqs(config.dim_head, config.max_length)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)
        self.transformer_encoder = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_dict: bool = True):
        L = input_ids.shape[1]
        freqs = (self.freqs_cos[:L].type_as(self.encoder.weight), self.freqs_sin[:L].type_as(self.encoder.weight))
        attn_mask = attention_mask[:, None, None, :]
        x = self.encoder(input_ids)
        for layer in self.transformer_encoder:
            x = layer(x, attn_mask, freqs)
        x = self.layer_norm(x)
        return BaseModelOutput(last_hidden_state=x)


# ---------------------------------------------------------------------------
# G2P Classifier
# ---------------------------------------------------------------------------

class HebrewG2PClassifier(nn.Module):
    """
    Per-character Hebrew G2P model.

    For each Hebrew letter in the input, predicts:
      - consonant class (from a per-letter constrained set)
      - vowel class     (a / e / i / o / u / ∅)
      - stress          (yes / no)
    """

    def __init__(self, dropout_rate: float = 0.1) -> None:
        super().__init__()

        config = NeoBERTConfig()
        self.encoder = NeoBERT(config)
        hidden_size = config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        self.consonant_head = nn.Linear(hidden_size, NUM_CONSONANT_CLASSES)
        self.vowel_head = nn.Linear(hidden_size, NUM_VOWEL_CLASSES)
        self.stress_head = nn.Linear(hidden_size, NUM_STRESS_CLASSES)

        self._consonant_mask: torch.Tensor | None = None
        self._build_consonant_mask()

    def _build_consonant_mask(self) -> None:
        from constants import ALEF_ORD, TAF_ORD
        n_letters = TAF_ORD - ALEF_ORD + 1
        mask = torch.ones(n_letters, NUM_CONSONANT_CLASSES, dtype=torch.bool)
        for char, allowed_ids in HEBREW_LETTER_TO_ALLOWED_CONSONANTS.items():
            idx = ord(char) - ALEF_ORD
            for cid in allowed_ids:
                mask[idx, cid] = False
        self._consonant_mask = mask

    def _apply_consonant_mask(self, consonant_logits, input_ids, tokenizer_vocab):
        from constants import ALEF_ORD
        mask = self._consonant_mask.to(consonant_logits.device)
        B, S, _ = consonant_logits.shape
        masked = consonant_logits.clone()
        for b in range(B):
            for s in range(S):
                char = tokenizer_vocab.get(input_ids[b, s].item(), "")
                if len(char) == 1 and is_hebrew_letter(char):
                    masked[b, s][mask[ord(char) - ALEF_ORD]] = -1e9
        return masked

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        consonant_labels: torch.Tensor | None = None,
        vowel_labels: torch.Tensor | None = None,
        stress_labels: torch.Tensor | None = None,
        tokenizer_vocab: dict[int, str] | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(encoder_outputs.last_hidden_state)

        consonant_logits = self.consonant_head(hidden)
        vowel_logits = self.vowel_head(hidden)
        stress_logits = self.stress_head(hidden)

        if tokenizer_vocab is not None:
            consonant_logits = self._apply_consonant_mask(consonant_logits, input_ids, tokenizer_vocab)

        output: dict[str, torch.Tensor] = {
            "consonant_logits": consonant_logits,
            "vowel_logits": vowel_logits,
            "stress_logits": stress_logits,
        }

        if consonant_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            output["loss"] = (
                loss_fct(consonant_logits.view(-1, NUM_CONSONANT_CLASSES), consonant_labels.view(-1))
                + loss_fct(vowel_logits.view(-1, NUM_VOWEL_CLASSES), vowel_labels.view(-1))
                + loss_fct(stress_logits.view(-1, NUM_STRESS_CLASSES), stress_labels.view(-1))
            )

        return output

    def parameter_groups(self, encoder_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        no_decay = {"bias", "norm.weight", "layer_norm.weight"}

        def is_no_decay(name: str) -> bool:
            return any(term in name for term in no_decay)

        return [
            {"params": [p for n, p in self.encoder.named_parameters() if not is_no_decay(n)], "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": [p for n, p in self.encoder.named_parameters() if is_no_decay(n)], "lr": encoder_lr, "weight_decay": 0.0},
            {"params": [p for n, p in self.named_parameters() if not n.startswith("encoder.") and not is_no_decay(n)], "lr": head_lr, "weight_decay": weight_decay},
            {"params": [p for n, p in self.named_parameters() if not n.startswith("encoder.") and is_no_decay(n)], "lr": head_lr, "weight_decay": 0.0},
        ]
