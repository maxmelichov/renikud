"""
Rotary Position Embeddings (RoPE) — rotate_half variant.

Precomputes cos/sin tables at init time and applies them via element-wise
multiplication, avoiding complex64 arithmetic and runtime trig calls.

Note: uses split-half pairing (first half / second half), not the interleaved
adjacent-pair scheme used in the original LLaMA implementation. The two are
mathematically equivalent rotations but produce different output tensors, so
checkpoints trained with the old implementation are incompatible.
"""

import torch
from typing import Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute cos/sin tables for RoPE.

    Returns a float32 tensor of shape (end, dim, 2) where
    [..., 0] = cos and [..., 1] = sin.  Compatible with the
    register_buffer / position_ids indexing in NeoBERT.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)           # (end, dim//2)
    freqs = torch.cat((freqs, freqs), dim=-1)  # (end, dim)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # (end, dim, 2)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.

    Args:
        xq: (batch, seq, heads, head_dim)
        xk: (batch, seq, heads, head_dim)
        freqs_cis: (seq, dim, 2) or (batch, seq, dim, 2) — from precompute_freqs_cis
    """
    cos = freqs_cis[..., 0].to(xq.dtype)  # (..., seq, dim)
    sin = freqs_cis[..., 1].to(xq.dtype)

    # Insert heads dim for broadcasting: (..., seq, 1, dim)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)

    return xq_out, xk_out
