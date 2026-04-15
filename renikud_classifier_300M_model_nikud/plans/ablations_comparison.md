# Ablation & Comparison Plan

> Reviewer feedback (Morris): add comparison to other models and ablations covering architecture, initialization, and training data source.

---

## 1. Baselines & Comparisons

| Model | Type | Data | Status |
|-------|------|------|--------|
| **Phonikud** | rule-based + neural | formal text (nikud) | run inference on `gt.tsv` |
| **ReNikud (ours)** | classifier | audio-supervised (vox) | ✓ trained |
| ByT5 (seq2seq) | seq2seq | Knesset / vox | in progress |
| CTC | CTC | Knesset / vox | in progress |

**Action:** run Phonikud inference on `dataset/gt.tsv` and record CER / WER / Acc.

---

## 2. Architecture Ablations

### 2a. Sequence-to-sequence (ByT5)
- Model: `google/byt5-small`
- Training data variants:
  - [ ] `dataset/knesset_split/` (regular Knesset, nikud-free) — **missing, needs to be run**
  - [x] `dataset/vox-knesset-ipa-v1.tsv` (audio-supervised) — running
- Output dirs: `phonikud-byt5/outputs/knesset-byt5-v1` (knesset), `phonikud-byt5/outputs/knesset-byt5-vox` (vox)

### 2b. CTC
- Training data variants:
  - [x] `dataset/knesset_split/` (original knesset) → `renikud-ctc/outputs/knesset-ctc-v1`
  - [x] `dataset/vox-knesset-ipa-v1.tsv` → `renikud-ctc/outputs/knesset-ctc-vox` (running)

---

## 3. Initialization Ablations (Classifier)

All use the same architecture (`HebrewG2PClassifier`) and training data (`vox-knesset-ipa-v1.tsv`).

| Init | Encoder | Status | Output dir |
|------|---------|--------|------------|
| DictaBERT-large-char (default) | `dicta-il/dictabert-large-char` | [x] running | `outputs/knesset-classifier-vox` |
| **No pretraining** | random init | [ ] **missing** | `outputs/knesset-classifier-vox-nopretrain` |
| NeoBERT | `TBD` | [ ] todo | `outputs/knesset-classifier-vox-neobert` |

**Action for no-pretraining run:**
```bash
# Add --no-pretrain flag or pass a randomly initialized encoder
DEVICE=cuda:0 uv run src/train.py \
  --train-dataset  dataset/vox-knesset-ipa-v1.tsv \
  --eval-dataset   dataset/pred_alignment.jsonl \
  --output-dir     outputs/knesset-classifier-vox-nopretrain \
  --train-batch-size 64 --eval-batch-size 64 \
  --epochs 3 --encoder-lr 2e-5 --head-lr 1e-4 \
  --save-steps 500 --early-stopping-patience 40 \
  --wandb-mode disabled --no-pretrain
```
> Need to check if `src/train.py` supports `--no-pretrain` or equivalent — may need a flag added.

---

## 4. Training Data Ablations (Classifier)

| Training data | Notes | Status |
|---------------|-------|--------|
| `vox-knesset-ipa-v1.tsv` (audio-supervised) | main model | [x] running |
| `knesset_phonemes_v1.txt` (text-only, nikud) | no audio | [x] done → `outputs/knesset-classifier-v1` |

This directly measures the value of audio supervision.

---

## 5. Missing Experiments Checklist

- [ ] **ByT5 on regular Knesset** (`knesset_split/`) — run `phonikud-byt5/experiments/knesset_byt5.sh` with `--data_dir ../dataset/knesset_split --ckpt_dir outputs/knesset-byt5-knesset`
- [ ] **Classifier without pretraining weights** — add `--no-pretrain` support, run on vox data
- [ ] **Phonikud baseline** — run on `dataset/gt.tsv`, record metrics in same format
- [ ] **NeoBERT init** — identify model name, run classifier with it as encoder

---

## 6. Results Summary

> All metrics on `dataset/gt.tsv` (250 sentences). Best checkpoint by WER.

| Model | Init | Train data | Best step | CER | WER | Acc |
|-------|------|------------|-----------|-----|-----|-----|
| Phonikud | — | formal text | — | | | |
| **Classifier** | DictaBERT-char | vox (audio) | 🔄 running | | | |
| Classifier | no pretrain | vox (audio) | ❌ missing | | | |
| **ByT5** | — | vox (audio) | 120,500 | 4.01% | 21.5% | 78.5% |
| **CTC** | — | knesset (text) | 90,500 | 4.52% | 19.9% | 80.1% |
| CTC | — | vox (audio) | 🔄 running | | | |

### Observations so far

- **Classifier (knesset) vs CTC (knesset)**: near-identical CER/WER (~4.4% / ~19.5%), classifier slightly ahead in accuracy (80.6% vs 80.1%).
- **ByT5 (vox)**: best CER so far (4.01%) but weaker WER/Acc than classifier — seq2seq tends to get characters right but word boundaries wrong.
- **Audio vs text supervision**: still pending — waiting on classifier-vox and CTC-vox to complete.
- **ByT5 on knesset**: not yet run — needed to isolate architecture effect from data effect.
