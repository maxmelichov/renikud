# Training

## Commands

### 1. Align data

```console
uv run src/align_data.py dataset/train.tsv dataset/train_alignment.jsonl
uv run src/align_data.py dataset/val.tsv dataset/val_alignment.jsonl
```

### 2. Prepare tokenized dataset

```console
uv run src/prepare_tokens.py dataset/train_alignment.jsonl dataset/.cache/train
uv run src/prepare_tokens.py dataset/val_alignment.jsonl dataset/.cache/val
```

### 3. Train

```console
uv run src/train.py \
  --train-dataset dataset/.cache/train \
  --eval-dataset dataset/.cache/val \
  --output-dir outputs/my-run \
  --epochs 1 \
  --encoder-lr 1e-5 \
  --head-lr 5e-5 \
  --train-batch-size 128 \
  --gradient-accumulation-steps 1 \
  --warmup-steps 200 \
  --logging-steps 50 \
  --save-steps 200 \
  --mixed-precision bf16
```

### Fine-tuning from a checkpoint

Use `--init-from-checkpoint` to load weights only (resets optimizer state):

```console
uv run src/train.py \
  --train-dataset dataset/.cache/train \
  --eval-dataset dataset/.cache/val \
  --output-dir outputs/my-run \
  --init-from-checkpoint outputs/previous-run/checkpoint-1200 \
  --epochs 1
```

## Mixed Precision

`--mixed-precision` accepts `no`, `fp16`, or `bf16` (default: `bf16` if supported, else `fp16`). `GradScaler` is only active for `fp16`; `bf16` does not need loss scaling.

## Learning Rates

- `--encoder-lr 1e-5` — LR for the NeoBERT encoder layers
- `--head-lr 5e-5` — higher LR for the three classification heads

RMSNorm parameters (`attention_norm.weight`, `ffn_norm.weight`, `layer_norm.weight`) and biases receive no weight decay.

## Data Format

Input TSV: `hebrew_text<TAB>ipa_text` — one sentence per line, no header. Hebrew side may have nikud (diacritics are stripped automatically by the aligner).

The aligner outputs JSONL where each line is `{"hebrew sentence": [["char", "ipa_chunk"], ...]}`. Failed alignments are saved to `<output>_failures.txt`.
