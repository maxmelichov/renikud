"""Tokenization helpers for Hebrew G2P."""

from __future__ import annotations

from functools import lru_cache

from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

from constants import ENCODER_MODEL


@lru_cache(maxsize=1)
def load_tokenizer(model_name: str = ENCODER_MODEL) -> PreTrainedTokenizerFast:
    tokenizer_file = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
