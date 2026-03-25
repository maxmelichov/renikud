# ReNikud: Section-by-Section Plan

## Section Status

### 1. Abstract (~150 words) — EXISTS, needs polish
- Tighten: currently has placeholder bullets inside the body
- Must state the quantitative improvement clearly

### 2. Introduction (~300 words) — EXISTS, needs filling
- TODO: fill [Max] / [Yakov] comment stubs
- Must end with explicit bullet contributions (currently placeholders "XXX")
- Key examples to keep: וואלה (/waʔalla/→/wˈalla/), שווארמה, ובדרך

### 3. Dataset (~0.75 col) — EXISTS, solid
- §A Pretraining: HeDC4 + Phonikud formal data
- §B Audio-derived: ILSpeech + dual ASR pipeline — **needs exact sentence/hour counts**
- §C Benchmark: 13K sentences — **needs annotation process / quality control description**

### 4. Methodology (~1 col) — EXISTS, solid
- §A Architecture: DictaBERT-large-char + 3 heads
- §B Label alignment: DP aligner, offset mapping, ignore index
- §C Per-letter consonant masking
- §D Training: discriminative LR, cosine+warmup, optional encoder freeze, FP16
- §E Inference: assemble [consonant][ˈ][vowel] per character

### 5. Experiments & Results (~1 col) — EMPTY
Must fill:
- **Main table:** ReNikud vs. Phonikud vs. baselines (Acc, WER, CER)
- **Ablation:** audio vs. text-only, with/without masking, with/without stress head
- **Phenomenon breakdown:** stress accuracy, /w/ vs. /v/, loanwords, colloquialisms
- **Example table:** extend Table I with model predictions column

### 6. Related Work (~0.5 col) — EXISTS as bullets, needs prose
Group into 4 paragraphs (Morris):
1. Audio supervision for G2P/diacritization (Sun, CATT-Whisper, Shatnawi, Badr)
2. Hebrew G2P and diacritization (Phonikud, Silber-Varod)
3. Low-resource G2P (G2PU, G2PA, Ribeiro, Razavi)
4. Spoken vs. written / homographs (Alqahtani, POWSM)

### 7. Conclusion (~0.25 col) — STUB
- Weak audio supervision → captures spoken Hebrew better than text supervision
- Future: larger audio datasets, TTS integration, cross-lingual transfer

---

## Open Items

| Item | Owner | Status |
|------|-------|--------|
| Experiments section — run ablations | Yakov | TODO |
| Phenomenon analysis breakdown | Yakov | TODO |
| Exact training data counts (sentences, hours) | Yakov/Max | TODO |
| Benchmark annotation process description | Yakov/Max | TODO |
| Fill Introduction stubs ([Max], [Yakov] comments) | Max+Yakov | TODO |
| Related Work prose paragraphs | Morris | TODO |
| Conclusion | All | TODO |
| Replace all [?] citation placeholders | All | TODO |
| Replace XXX contribution placeholders | All | TODO |
| Confirm target EMNLP workshop | All | TODO |

---

## Key Numbers

- **Phonikud (teacher):** 86.1% acc, WER 0.14, CER 0.04
- **ReNikud (best):** 89.3% acc, WER 0.107, CER 0.026
- Benchmark: heb-g2p-benchmark (100 sentences, stress/homograph/gender)
- Our benchmark: 13K sentences (spoken Hebrew)

---

## Page Budget

**4 pages (tight):** Merge §3D+E, cut ablation to 1-row table, keep one example table
**6 pages (comfortable):** Full ablation, phenomenon breakdown, longer related work, error analysis figure

Recommendation: target 6 pages — dataset + benchmark contribution warrants the space.

---

## Immediate Next Steps

1. Run experiments and fill §IV — critical path blocker
2. Fill Introduction stubs (Max+Yakov)
3. Write Related Work prose (Morris)
4. Add data statistics to §II-B and §II-C
5. Write Conclusion
6. Confirm workshop CFP and formatting requirements
