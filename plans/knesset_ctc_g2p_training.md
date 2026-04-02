# Knesset G2P Classifier Training Plan

## Branch & Model

- **Branch:** `thesis_experiments`
- **Model:** `G2PModel` (per-character classifier with coupled consonant / vowel / stress heads)
- **Training script:** `src/train.py`

## Data

| Split      | Path                                  | Notes                                      |
|------------|---------------------------------------|--------------------------------------------|
| Train      | `dataset/knesset_vox_new_asr.tsv`      | ~528K lines, Knesset speech (nikud\tIPA, no header) |
| Validation | `dataset/gt_alignment.jsonl`        | 247/250 aligned sentences from `dataset/gt.tsv` |

## Step 1 — Prepare validation data

```bash
# Strip header, keep nikud + IPA columns, align on the fly
awk -F'\t' 'NR>1 && NF>=2 {print $1 "\t" $2}' dataset/gt.tsv | \
  uv run src/data_align.py /dev/stdin dataset/gt_alignment.jsonl
```

> Alignment failures are saved to `dataset/gt_alignment_failures.txt` (3 of 250 fail).
> The train dataset (`dataset/knesset_vox_new_asr.tsv`) is loaded directly at training time — no pre-tokenization needed.

## Step 2 — Train

```bash
DEVICE=cuda:0 bash experiments/knesset_classifier.sh
```

Which runs:

```bash
uv run src/train.py \
  --train-dataset  dataset/knesset_vox_new_asr.tsv \
  --eval-dataset   dataset/gt_alignment.jsonl \
  --output-dir     outputs/knesset-classifier-metadata \
  --train-batch-size 56 \
  --eval-batch-size  56 \
  --epochs         3 \
  --encoder-lr     2e-5 \
  --head-lr        1e-4 \
  --save-steps     500 \
  --early-stopping-patience 40 \
  --wandb-mode     disabled \
  --device         ${DEVICE:-cuda:0}
```

> Device selection via `--device` is passed through to `ACCELERATOR`; alternatively set `CUDA_VISIBLE_DEVICES=0` before the command.
>
> To resume or fine-tune from a checkpoint, add `--init-from-checkpoint outputs/knesset-classifier-metadata/checkpoint-XXXX`.

## Output directory layout

```
outputs/knesset-classifier-metadata/
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
| `train_batch_size`            | 56                       |
| `eval_batch_size`             | 56                       |
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

| Metric       | Description                                        |
|--------------|----------------------------------------------------|
| `wer`        | Word Error Rate (IPA sequence comparison)          |
| `cer`        | Character Error Rate (IPA sequence comparison)     |
| `acc`        | `1 - wer`                                          |
| `eval_loss`  | Combined cross-entropy loss (consonant+vowel+stress)|

Early stopping is triggered when WER does not improve for `early_stopping_patience` eval intervals.

## Experiment results log

Copy key numbers here after each run:

| Run | Best Step | WER    | CER    | acc    | eval_loss | encoder_lr | head_lr | batch_size | Notes |
|-----|-----------|--------|--------|--------|-----------|------------|---------|------------|-------|
|  1  |           |        |        |        |           | 2e-5       | 1e-4    | 56         | metadata_ipa_clean baseline |

## Checklist

- [x] Checkout branch `thesis_experiments`
- [x] Run Step 1 (align `pred.tsv` → `gt_alignment.jsonl`)
- [ ] Run Step 2 (training via `experiments/knesset_classifier.sh`)
- [ ] Record results in the table above
