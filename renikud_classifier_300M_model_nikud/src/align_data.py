"""
Extract per-character nikud labels from vocalized Hebrew text.

Each Hebrew letter maps to a pair:
  (base_char, nikud_str)

where nikud_str is the full combining-mark sequence after the letter
(dagesh + vowel, or just vowel, or MAT_LECT, or '').
The shin/sin dot is kept as part of nikud_str so callers can split it out
if they need the separate shin head.

Input:  one vocalized Hebrew sentence per line (TSV uses column 1)
Output: JSONL, one object per line — key = bare Hebrew,
        value = list of [char, nikud_str] pairs.

Usage:
    uv run src/align_data.py input.tsv output.jsonl
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from tqdm import tqdm

from constants import (
    NIKUD_TO_ID,
    SHIN_TO_ID,
    SHIN_LETTER,
    MAT_LECT_TOKEN,
    is_hebrew_letter,
    strip_nikud,
)

# Letters that can act as matres lectionis (silent vowel markers) in Hebrew
MAT_LECT_LETTERS: frozenset[str] = frozenset("אהוי")


def _resolve_nikud(combining: list[str]) -> str:
    """
    Map a list of combining characters (excluding shin/sin dot) to a NIKUD_CLASSES label.
    Tries canonical order first, then reversed (some texts put vowel before dagesh).
    Falls back to '' on unknown combination.
    """
    nikud_str = "".join(combining)
    if nikud_str in NIKUD_TO_ID:
        return nikud_str
    rev = "".join(reversed(combining))
    if rev in NIKUD_TO_ID:
        return rev
    return ""


def extract_nikud(vocalized: str) -> list[tuple[str, str]] | None:
    """
    Decompose a vocalized Hebrew string into per-base-letter pairs:
        (base_char, nikud_str)

    nikud_str encodes the vowel/dagesh part (from NIKUD_CLASSES).
    Shin/sin dot is NOT included in nikud_str — it is stored separately.
    Use split_nikud_shin() to separate them when needed.

    Returns None on error (empty result).
    """
    nfd = unicodedata.normalize("NFD", vocalized)
    result: list[tuple[str, str]] = []

    i = 0
    while i < len(nfd):
        ch = nfd[i]
        i += 1

        if not is_hebrew_letter(ch):
            continue

        # Collect all combining marks that follow this base letter
        combining: list[str] = []
        while i < len(nfd) and unicodedata.category(nfd[i]).startswith("M"):
            combining.append(nfd[i])
            i += 1

        # Separate shin/sin dot — keep it aside, don't mix into nikud_str
        shin_dot: str | None = None
        if ch == SHIN_LETTER:
            shin_marks = [c for c in combining if c in SHIN_TO_ID]
            if shin_marks:
                shin_dot = shin_marks[0]
            combining = [c for c in combining if c not in SHIN_TO_ID]

        # Resolve the remaining marks to a NIKUD_CLASSES label
        if combining:
            nikud_label = _resolve_nikud(combining)
        elif ch in MAT_LECT_LETTERS:
            nikud_label = MAT_LECT_TOKEN
        else:
            nikud_label = ""

        result.append((ch, nikud_label))

    return result if result else None


def split_nikud_shin(char: str, nikud_str: str) -> tuple[str, str | None]:
    """
    Given a (char, nikud_str) pair, return (nikud_label, shin_label_or_None).
    Shin/sin dot information is not in nikud_str; it must be re-extracted from
    the original text by the caller (this helper is a no-op placeholder kept
    for API compatibility).
    """
    return nikud_str, None


def main():
    parser = argparse.ArgumentParser(description="Extract per-character nikud labels from vocalized Hebrew")
    parser.add_argument("input", help="Input file (vocalized Hebrew, one sentence per line; TSV uses column 1)")
    parser.add_argument("output", help="Output JSONL file")
    args = parser.parse_args()

    total = 0
    ok = 0
    failed = 0

    failures_path = args.output.replace(".jsonl", "_failures.txt")
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout, \
         open(failures_path, "w", encoding="utf-8") as ffail:

        for line in tqdm(fin, desc="Extracting"):
            line = line.strip()
            if not line:
                continue
            vocalized = line.split("\t", 1)[0]
            bare = strip_nikud(vocalized)
            total += 1

            pairs = extract_nikud(vocalized)
            if pairs is None:
                failed += 1
                ffail.write(f"{vocalized}\n")
                continue

            ok += 1
            fout.write(json.dumps({bare: pairs}, ensure_ascii=False) + "\n")

    print(f"\nTotal:   {total:,}")
    print(f"OK:      {ok:,} ({ok/total:.1%})" if total else "")
    print(f"Failed:  {failed:,} ({failed/total:.1%})" if total else "")


if __name__ == "__main__":
    main()
