"""Run inference with the Hebrew nikud classifier model.

Usage:
    uv run src/infer.py --checkpoint outputs/nikud-classifier/checkpoint-5000 --text "שלום עולם"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from constants import (
    ID_TO_NIKUD,
    ID_TO_SHIN,
    MAT_LECT_TOKEN,
    SHIN_LETTER,
    MAX_LEN,
    is_hebrew_letter,
)
from model import HebrewNikudClassifier
from tokenization import load_encoder_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Add nikud to unvocalized Hebrew text using the classifier model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    return parser.parse_args()


def load_checkpoint(model: HebrewNikudClassifier, checkpoint_dir: str) -> None:
    from safetensors.torch import load_file
    base = Path(checkpoint_dir)
    safetensors_path = base / "model.safetensors"
    bin_path = base / "pytorch_model.bin"
    if safetensors_path.exists():
        state = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No checkpoint weights found in {checkpoint_dir}")
    model.load_state_dict(state, strict=False)


def _decode(
    text: str,
    offset_mapping: list[tuple[int, int]],
    nikud_logits: torch.Tensor,
    shin_logits: torch.Tensor,
) -> str:
    """Reconstruct vocalized Hebrew by inserting predicted nikud marks after each letter."""
    nikud_preds = nikud_logits.argmax(dim=-1)  # [S]
    shin_preds = shin_logits.argmax(dim=-1)    # [S]

    result: list[str] = []
    prev_char_end = 0

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start > prev_char_end:
            result.append(text[prev_char_end:start])

        if end - start != 1:
            if end > start:
                prev_char_end = end
            continue

        char = text[start:end]
        prev_char_end = end

        if not is_hebrew_letter(char):
            result.append(char)
            continue

        # Shin/sin dot prediction (only for ש)
        if char == SHIN_LETTER:
            shin_mark = ID_TO_SHIN.get(int(shin_preds[tok_idx]), "")
            result.append(char + shin_mark)
        else:
            result.append(char)

        # Nikud prediction
        nikud = ID_TO_NIKUD.get(int(nikud_preds[tok_idx]), "")
        if nikud and nikud != MAT_LECT_TOKEN:
            result.append(nikud)
        # MAT_LECT means the letter is a vowel carrier — emit nothing extra

    if prev_char_end < len(text):
        result.append(text[prev_char_end:])

    return "".join(result)


def add_nikud(text: str, model: HebrewNikudClassifier, tokenizer, device: torch.device, max_len: int) -> str:
    """Add nikud diacritics to unvocalized Hebrew text using the classifier model."""
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    return _decode(
        text=text,
        offset_mapping=offset_mapping,
        nikud_logits=out["nikud_logits"][0],
        shin_logits=out["shin_logits"][0],
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_encoder_tokenizer()
    model = HebrewNikudClassifier()
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    print(add_nikud(args.text, model, tokenizer, device, args.max_len))


if __name__ == "__main__":
    main()
