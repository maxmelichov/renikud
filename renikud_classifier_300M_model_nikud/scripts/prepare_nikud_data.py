"""
Prepare train/eval text files for the nikud classifier from the nakdimon corpus.

Reads all .txt files from data/nakdimon/hebrew_diacritized/, splits long lines
into sentence-sized chunks, filters noise, and writes:
    dataset/nikud_train.txt
    dataset/nikud_eval.txt

Usage:
    cd renikud_classifier_300M_model_nikud
    uv run scripts/prepare_nikud_data.py
"""

from __future__ import annotations

import glob
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "nakdimon" / "hebrew_diacritized"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"

# These directories are reserved for eval / test — excluded from training.
EVAL_DIRS = {"validation"}
SKIP_DIRS = {"garbage", "test_modern", "dictaTestCorpus"}

MAX_BARE_CHARS = 200     # max Hebrew-letter count per output line (fits in 256 tokens)
MIN_BARE_CHARS = 5       # discard lines shorter than this

# Sentence-boundary split pattern: split after .!?; optionally followed by a closing quote
SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

HEBREW_RE = re.compile(r'[\u05D0-\u05EA]')  # contains at least one Hebrew letter
SEPARATOR_RE = re.compile(r'^[=\-_*\s]+$')  # pure separator lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bare_len(text: str) -> int:
    """Count base Hebrew letters (strip nikud combining marks)."""
    nfd = unicodedata.normalize("NFD", text)
    return sum(1 for c in nfd if "\u05D0" <= c <= "\u05EA")


def split_line(line: str) -> list[str]:
    """
    Split a potentially long paragraph into sentence-sized chunks.
    If a single sentence is still too long, hard-split at word boundaries.
    """
    sentences = SPLIT_RE.split(line)
    chunks: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if bare_len(sent) <= MAX_BARE_CHARS:
            chunks.append(sent)
        else:
            # Hard-split on word boundaries
            words = sent.split()
            current: list[str] = []
            current_len = 0
            for w in words:
                wl = bare_len(w)
                if current_len + wl > MAX_BARE_CHARS and current:
                    chunks.append(" ".join(current))
                    current, current_len = [], 0
                current.append(w)
                current_len += wl
            if current:
                chunks.append(" ".join(current))
    return chunks


def is_valid(line: str) -> bool:
    if not HEBREW_RE.search(line):
        return False
    if SEPARATOR_RE.match(line):
        return False
    if bare_len(line) < MIN_BARE_CHARS:
        return False
    return True


def collect_lines(genre_dirs: list[Path]) -> list[str]:
    lines: list[str] = []
    for d in genre_dirs:
        for path in sorted(d.rglob("*.txt")):
            for raw in path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                for chunk in split_line(raw):
                    if is_valid(chunk):
                        lines.append(chunk)
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dirs = [p for p in DATA_ROOT.iterdir() if p.is_dir()]
    train_dirs = [d for d in all_dirs if d.name not in EVAL_DIRS and d.name not in SKIP_DIRS]
    eval_dirs  = [d for d in all_dirs if d.name in EVAL_DIRS]

    print("Collecting training lines…")
    train_lines = collect_lines(train_dirs)
    print(f"  {len(train_lines):,} train lines")

    print("Collecting eval lines…")
    eval_lines = collect_lines(eval_dirs)
    print(f"  {len(eval_lines):,} eval lines")

    train_path = OUT_DIR / "nikud_train.txt"
    eval_path  = OUT_DIR / "nikud_eval.txt"

    train_path.write_text("\n".join(train_lines), encoding="utf-8")
    eval_path.write_text("\n".join(eval_lines), encoding="utf-8")

    print(f"\nWrote {train_path}")
    print(f"Wrote {eval_path}")


if __name__ == "__main__":
    main()
