"""renikud-onnx: Hebrew grapheme-to-phoneme inference via ONNX."""

from __future__ import annotations

import json
import unicodedata

import numpy as np
import onnxruntime as ort


class G2P:
    def __init__(self, model_path: str) -> None:
        self._session = ort.InferenceSession(model_path)
        meta = self._session.get_modelmeta().custom_metadata_map
        self._vocab: dict[str, int] = json.loads(meta["vocab"])
        self._ipa_vocab: dict[int, str] = {int(k): v for k, v in json.loads(meta["ipa_vocab"]).items()}
        self._cls_id = int(meta["cls_token_id"])
        self._sep_id = int(meta["sep_token_id"])
        self._blank_id = 0

    def _tokenize(self, text: str) -> tuple[list[int], list[int]]:
        text = unicodedata.normalize("NFD", text)
        unk_id = self._vocab.get("[UNK]", 0)
        ids = [self._cls_id] + [self._vocab.get(c, unk_id) for c in text] + [self._sep_id]
        mask = [1] * len(ids)
        return ids, mask

    def _decode(self, token_ids: list[int]) -> str:
        result = []
        prev = None
        for t in token_ids:
            if t == self._blank_id:
                prev = None
                continue
            if t != prev:
                token = self._ipa_vocab.get(t, "")
                if token not in ("<pad>", "<unk>", "<blank>"):
                    result.append(token)
            prev = t
        return "".join(result)

    def phonemize(self, text: str) -> str:
        ids, mask = self._tokenize(text)
        input_ids = np.array([ids], dtype=np.int64)
        attention_mask = np.array([mask], dtype=np.int64)
        logits, input_lengths = self._session.run(
            ["logits", "input_lengths"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        length = int(input_lengths[0])
        pred_ids = logits[0, :length].argmax(axis=-1).tolist()
        return self._decode(pred_ids)
