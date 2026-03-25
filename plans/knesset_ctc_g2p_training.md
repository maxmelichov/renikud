# Knesset G2P Classifier Training Plan

## Branch & Model

- **Branch:** `thesis_experiments`
- **Model:** `HebrewG2PClassifier` (per-character classifier with consonant / vowel / stress heads)
- **Training script:** `src/train.py`

## Data

| Split      | Path                                        | Notes                                        |
|------------|---------------------------------------------|----------------------------------------------|
| Train      | `dataset/knesset_phonemes_v1.txt`           | ~5.3M lines of Knesset speech (nikud\tIPA)   |
| Validation | `dataset/pred.tsv`                          | 250 sentences (nikud\tIPA)                   |

## Step 1 — Prepare data

```bash
# 1a. Split knesset raw text into train/val txt files
uv run src/prepare_data.py \
  --input      dataset/knesset_phonemes_v1.txt \
  --output-dir dataset/knesset_split \
  --lines      5000000 \
  --max-val    0

# 1b. Align train split (Hebrew chars -> IPA chunks) -> JSONL
uv run src/align_data.py \
  dataset/knesset_split/train.txt \
  dataset/knesset_split/train_alignment.jsonl

# 1c. Align validation file
uv run src/align_data.py \
  dataset/pred.tsv \
  dataset/knesset_split/val_alignment.jsonl

# 1d. Tokenize train alignment to Arrow
uv run src/prepare_tokens.py \
  dataset/knesset_split/train_alignment.jsonl \
  dataset/.cache/knesset_train

# 1e. Tokenize val alignment to Arrow
uv run src/prepare_tokens.py \
  dataset/knesset_split/val_alignment.jsonl \
  dataset/.cache/knesset_val
```

> Alignment failures are saved to `<output>_failures.txt`.

## Step 2 — Train

```bash
uv run src/train.py \
  --train-dataset  dataset/.cache/knesset_train \
  --eval-dataset   dataset/.cache/knesset_val \
  --output-dir     outputs/knesset-g2p \
  --epochs         3 \
  --save-steps     500 \
  --wandb-mode     online
```

> To resume or fine-tune from a checkpoint, add `--init-from-checkpoint outputs/knesset-g2p/checkpoint-XXXX`.

## Output directory layout

```
outputs/knesset-g2p/
├── checkpoint-500/
│   ├── model.safetensors
│   └── train_state.json
├── checkpoint-1000/
│   ...
```

> Best checkpoint is the one with highest `mean_acc` in `train_state.json`.

## Training configuration (defaults)

| Parameter                    | Value                    |
|------------------------------|--------------------------|
| `epochs`                     | 3.0                      |
| `save_steps`                 | 500                      |
| `save_total_limit`           | 20                       |
| `train_batch_size`           | 32                       |
| `eval_batch_size`            | 32                       |
| `encoder_lr`                 | 2e-5                     |
| `head_lr`                    | 1e-4                     |
| `weight_decay`               | 0.01                     |
| `warmup_steps`               | 200                      |
| `logging_steps`              | 50                       |
| `gradient_accumulation_steps`| 1                        |
| `max_grad_norm`              | 1.0                      |
| `fp16`                       | true (if CUDA available) |
| `freeze_encoder_steps`       | 0                        |
| W&B project                  | `hebrew-g2p-classifier`  |

## Metrics tracked

| Metric          | Description                              |
|-----------------|------------------------------------------|
| `consonant_acc` | Per-token consonant prediction accuracy  |
| `vowel_acc`     | Per-token vowel prediction accuracy      |
| `stress_acc`    | Per-token stress prediction accuracy     |
| `mean_acc`      | Average of the three above               |
| `eval_loss`     | Combined cross-entropy loss              |

## Experiment results log

Copy key numbers here after each run:

| Run | Best Step | consonant_acc | vowel_acc | stress_acc | mean_acc | encoder_lr | head_lr | batch_size | Notes |
|-----|-----------|---------------|-----------|------------|----------|------------|---------|------------|-------|
|  1  |           |               |           |            |          | 2e-5       | 1e-4    | 32         | baseline knesset |

## Checklist

- [ ] Checkout branch `thesis_experiments`
- [ ] Run Step 1 (data preparation + alignment)
- [ ] Run Step 2 (training)
- [ ] Monitor metrics in W&B (project: `hebrew-g2p-classifier`)
- [ ] Record results in the table above
