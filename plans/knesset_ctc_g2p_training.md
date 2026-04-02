# Knesset G2P Classifier Training Plan

## Branch & Model

- **Branch:** `thesis_experiments`
- **Model:** `HebrewG2PClassifier` (per-character classifier with consonant / vowel / stress heads)
- **Training script:** `src/train.py`

## Data

| Split      | Path                              | Notes                                      |
|------------|-----------------------------------|--------------------------------------------|
| Train      | `dataset/knesset_phonemes_v1.txt` | ~5M lines of Knesset speech (nikud\tIPA)   |
| Validation | `dataset/pred.tsv`                | 250 sentences from `dataset/gt.tsv` (nikud\tIPA\tField), 247 aligned |

## Step 1 — Prepare data

```bash
# 1a. Align validation file (pred.tsv has a header row — stripped automatically)
awk -F'\t' 'NR>1 && NF>=2 {print $1 "\t" $2}' dataset/pred.tsv > dataset/pred_2col.txt
uv run src/align_data.py \
  dataset/pred_2col.txt \
  dataset/pred_alignment.jsonl
```

> Alignment failures are saved to `dataset/pred_alignment_failures.txt` (3 of 250 fail).
> The train dataset (`dataset/knesset_phonemes_v1.txt`) is loaded directly at training time — no pre-tokenization needed.

## Step 2 — Train

```bash
DEVICE=cuda:0 bash experiments/knesset_classifier.sh
```

Which runs:

```bash
uv run src/train.py \
  --train-dataset  dataset/knesset_phonemes_v1.txt \
  --eval-dataset   dataset/pred_alignment.jsonl \
  --output-dir     outputs/knesset-classifier \
  --train-batch-size 64 \
  --eval-batch-size  64 \
  --epochs         3 \
  --encoder-lr     2e-5 \
  --head-lr        1e-4 \
  --save-steps     500 \
  --early-stopping-patience 40 \
  --wandb-mode     disabled \
  --device         cuda:0
```

> To resume or fine-tune from a checkpoint, add `--init-from-checkpoint outputs/knesset-classifier/checkpoint-XXXX`.

## Output directory layout

```
outputs/knesset-classifier/
├── checkpoint-500/
│   ├── model.safetensors
│   └── train_state.json      # contains step, cer, wer, acc, eval_loss
├── checkpoint-1000/
│   ...
└── checkpoint-best/          # copy of best checkpoint by WER
    ├── model.safetensors
    └── train_state.json
```

> `checkpoint-best` is updated automatically whenever WER improves. Patience counter is printed after every eval.

## Training configuration (defaults)

| Parameter                     | Value                    |
|-------------------------------|--------------------------|
| `epochs`                      | 3.0                      |
| `save_steps`                  | 500                      |
| `save_total_limit`            | 20                       |
| `train_batch_size`            | 64                       |
| `eval_batch_size`             | 64                       |
| `encoder_lr`                  | 2e-5                     |
| `head_lr`                     | 1e-4                     |
| `weight_decay`                | 0.01                     |
| `warmup_steps`                | 200                      |
| `logging_steps`               | 50                       |
| `gradient_accumulation_steps` | 1                        |
| `max_grad_norm`               | 1.0                      |
| `early_stopping_patience`     | 40                       |
| `fp16`                        | true (if CUDA available) |
| `freeze_encoder_steps`        | 0                        |
| W&B project                   | `hebrew-g2p-classifier`  |

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

- [x] Checkout branch `thesis_experiments`
- [x] Run Step 1 (align `pred.tsv` → `pred_alignment.jsonl`)
- [x] Run Step 2 (training via `experiments/knesset_classifier.sh`)
- [ ] Record results in the table above
