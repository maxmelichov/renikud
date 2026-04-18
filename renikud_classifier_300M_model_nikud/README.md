# renikud

Hebrew grapheme-to-phoneme (G2P) training project for converting unvocalized Hebrew text into IPA.

Model: [thewh1teagle/renikud](https://huggingface.co/thewh1teagle/renikud)

## Architecture

DictaBERT character-level encoder (300M params) → three linear heads (consonant, vowel, stress) per Hebrew letter.

Each Hebrew letter gets exactly one output slot predicting a (consonant, vowel, stress) triple. Trained on ~900K Hebrew sentences (vox-knesset corpus, silver IPA labels from Phonikud). Reaches **90.6% word accuracy** on [heb-g2p-benchmark](https://github.com/thewh1teagle/heb-g2p-benchmark), surpassing the Phonikud teacher (86.1%).

See `docs/ARCHITECTURE.md` for full design details.

## Main training data (nikud)

Primary diacritized-Hebrew corpora for this variant of the project:

| Source | Role | Upstream |
|--------|------|----------|
| **UNIKUD** | Large mixed web + public-domain nikud data ([`data/`](https://github.com/morrisalp/unikud/tree/main/data), DVC) | [morrisalp/unikud](https://github.com/morrisalp/unikud) |
| **Nakdimon** | `hebrew_diacritized` training set (git submodule) | [elazarg/nakdimon](https://github.com/elazarg/nakdimon) · [elazarg/hebrew_diacritized](https://github.com/elazarg/hebrew_diacritized) |

Vendor both under `third_party/` (ignored by git) and pull UNIKUD blobs with DVC:

```console
bash scripts/fetch_main_nikud_data.sh
```

Resolved paths are exposed in `src/constants.py` as `THIRD_PARTY_UNIKUD_DATA`, `THIRD_PARTY_NAKDIMON_HEBREW_DIACRITIZED`, etc.

## Data Preparation

```console
uv run src/align_data.py dataset/train.tsv dataset/train_alignment.jsonl
uv run src/align_data.py dataset/val.tsv dataset/val_alignment.jsonl
uv run src/prepare_tokens.py dataset/train_alignment.jsonl dataset/.cache/train
uv run src/prepare_tokens.py dataset/val_alignment.jsonl dataset/.cache/val
```

## Training

```console
uv run src/train.py \
  --train-dataset dataset/.cache/train \
  --eval-dataset dataset/.cache/val \
  --output-dir outputs/g2p-classifier \
  --encoder-lr 2e-6 \
  --head-lr 1e-5
```

See `docs/TRAINING.md` for full training instructions.

## Benchmark

```console
wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv
uv run scripts/benchmark.py --checkpoint outputs/g2p-classifier/checkpoint-1500
```

## Inference (ONNX)

Export a checkpoint to ONNX:

```console
cd renikud-onnx
uv run scripts/export.py --checkpoint ../outputs/g2p-classifier/checkpoint-1500 --output model.onnx
```

Then use the Python package:

```python
from renikud_onnx import G2P

g2p = G2P("model.onnx")
print(g2p.phonemize("שלום עולם"))
# → ʃalˈom ʔolˈam
```

Or the Rust crate — see `renikud-rs/`.

## Upload Checkpoint to HuggingFace

```console
uv run hf upload thewh1teagle/renikud outputs/g2p-classifier/checkpoint-1500 --include "model.safetensors" --commit-message "add weights"
```

## Download Checkpoint

```console
uv run hf download thewh1teagle/renikud model.safetensors --local-dir .
```
