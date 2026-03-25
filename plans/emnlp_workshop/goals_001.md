# ReNikud: EMNLP Workshop Paper — Goals

## What we're building

A 4–6 page workshop paper for EMNLP (target: MRL or SIGMorphon) presenting ReNikud — a weakly supervised Hebrew G2P model that learns from real speech instead of formal text annotations.

## Core claim

Using a dual ASR pipeline (text ASR + IPA ASR) on real Hebrew recordings produces a G2P model that outperforms text-supervised baselines on natural spoken Hebrew, capturing phenomena invisible in writing: /w/ vs. /v/, lexical stress, colloquialisms, loanwords.

## Contributions

1. **Weak supervision pipeline** — IPA labels extracted from audio via dual ASR → bypasses manual diacritization
2. **Per-character classifier** — DictaBERT-large-char (300M) + consonant/vowel/stress heads with linguistic masking
3. **Spoken Hebrew benchmark** — 13K sentences targeting written/spoken divergence
4. **SOTA results** — 89.3% acc vs. Phonikud's 86.1%, WER 0.107 vs. 0.14, CER 0.026 vs. 0.04

## Why it matters

Existing models (Phonikud) are trained on formal text — they predict /vaʔalla/ when speakers say /wˈalla/. Audio supervision captures how Hebrew is actually spoken.

## Current status

Draft exists (`ReNikud.pdf`). Methodology and dataset sections are written. Experiments section is empty — this is the critical path blocker.

## See also

- `paper_emnlp_002.md` — section-by-section plan and open items
