"""Tokenization helpers for Hebrew G2P."""

from __future__ import annotations

from functools import lru_cache

from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

from constants import CHAR_TOKENIZER_MODEL


@lru_cache(maxsize=1)
def load_encoder_tokenizer() -> PreTrainedTokenizerFast:
    """Load the character-level Hebrew tokenizer."""
    tokenizer_file = hf_hub_download(repo_id=CHAR_TOKENIZER_MODEL, filename="tokenizer.json")
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
