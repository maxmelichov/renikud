# Knesset CTC G2P Training Plan

## Branch & Model

- **Branch:** `thesis_experiments`
- **Model folder:** `phonikud-byt5/`
- **Training script:** `src/train.py`

## Data

| Split      | Path                                        | Notes                         |
|------------|---------------------------------------------|-------------------------------|
| Train      | `dataset/knesset_phonemes_v1.txt`           | ~5.3M lines of Knesset speech |
| Validation | `dataset/pred.tsv`                          | 250 sentences, used for WER   |
| Test       | `dataset/ilspeech_speaker1_v1/`             | Evaluated after best val WER  |
|            | `dataset/ilspeech_speaker2_v1/`             | Evaluated after best val WER  |

## Step 1 — Prepare data

```bash
# 1a. Split knesset raw text into train/val txt files
uv run src/prepare_data.py \
  --input   dataset/knesset_phonemes_v1.txt \
  --output-dir dataset/knesset_split \
  --lines   5000000 \
  --max-val 0

# 1b. Tokenize knesset train split to Arrow
uv run src/prepare_tokens.py \
  --input  dataset/knesset_split/train.txt \
  --output dataset/.cache/knesset_train

# 1c. Tokenize pred.tsv as validation (3-column TSV; header is auto-skipped)
uv run src/prepare_tokens.py \
  --input  dataset/pred.tsv \
  --output dataset/.cache/pred_val
```

## Step 2 — Train

```bash
uv run src/train.py \
  --train-dataset           dataset/.cache/knesset_train \
  --eval-dataset            dataset/.cache/pred_val \
  --output-dir              outputs/knesset-ctc-g2p \
  --save-steps              500 \
  --early-stopping-patience 20000 \
  --wandb-mode              online
```

> `--epochs` defaults to 9999 — training runs until early stopping fires.
> No `--max-steps` argument exists; the loop is bounded solely by early stopping.

## Step 3 — Test on IL-Speech speakers

Run after training completes (uses `best/` checkpoint automatically):

```bash
uv run src/test_ilspeech.py \
  --checkpoint outputs/knesset-ctc-g2p/best \
  --datasets   dataset/ilspeech_speaker1_v1 dataset/ilspeech_speaker2_v1 \
  --output-dir outputs/knesset-ctc-g2p
```

Results are appended into `outputs/knesset-ctc-g2p/experiment_results.json`.

## Output directory layout

```
outputs/knesset-ctc-g2p/
├── best/                        # best checkpoint by val WER
│   ├── model.safetensors
│   └── train_state.json
├── checkpoint-500/
├── checkpoint-1000/
│   ...
├── experiment_results.json      # auto-written at end of training + test
├── test_ilspeech_speaker1_v1.tsv
└── test_ilspeech_speaker2_v1.tsv
```

## Training configuration (defaults)

| Parameter                    | Value                    |
|------------------------------|--------------------------|
| `epochs`                     | 9999 (unlimited)         |
| `save_steps`                 | 500                      |
| `early_stopping_patience`    | 20 000 steps             |
| `train_batch_size`           | 8                        |
| `eval_batch_size`            | 8                        |
| `encoder_lr`                 | 2e-5                     |
| `head_lr`                    | 1e-4                     |
| `weight_decay`               | 0.01                     |
| `warmup_steps`               | 200                      |
| `gradient_accumulation_steps`| 1                        |
| `max_grad_norm`              | 1.0                      |
| `upsample_factor`            | 2                        |
| `fp16`                       | true (if CUDA available) |
| `freeze_encoder_steps`       | 0                        |

## Experiment results log

`experiment_results.json` is written automatically by `train.py` and updated by `test_ilspeech.py`.
Copy the key numbers here after each run:

| Run | Best Step | Best Val WER | Test WER (spk1) | Test WER (spk2) | encoder_lr | head_lr | batch_size | Notes |
|-----|-----------|--------------|-----------------|-----------------|------------|---------|------------|-------|
|  1  |           |              |                 |                 | 2e-5       | 1e-4    | 8          | baseline knesset |

## Checklist

- [ ] Checkout branch `thesis_experiments`
- [ ] Run Step 1 (data preparation)
- [ ] Run Step 2 (training)
- [ ] Monitor val WER in W&B (project: `hebrew-g2p`)
- [ ] Run Step 3 (test on IL-Speech)
- [ ] Record results in the table above
