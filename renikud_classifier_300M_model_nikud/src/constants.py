"""Constants for the Hebrew nikud classifier model."""

from pathlib import Path
from typing import Final

ENCODER_MODEL: Final[str] = "dicta-il/dictabert-large-char-menaked"
MAX_LEN: Final[int] = 256

# ---------------------------------------------------------------------------
# Main training corpora (diacritized Hebrew) — canonical upstream sources
# ---------------------------------------------------------------------------
# UNIKUD: Wikipedia / Wikisource / Ben-Yehuda–style nikud data (DVC on Dagshub).
UNIKUD_GITHUB_URL: Final[str] = "https://github.com/morrisalp/unikud"
UNIKUD_DATA_TREE_URL: Final[str] = "https://github.com/morrisalp/unikud/tree/main/data"
# Per upstream README: add remote and pull large files after cloning the repo.
UNIKUD_DVC_REMOTE_URL: Final[str] = "https://dagshub.com/morrisalp/unikud.dvc"

# Nakdimon: training text lives in the `hebrew_diacritized` git submodule.
NAKDIMON_REPO_URL: Final[str] = "https://github.com/elazarg/nakdimon"
NAKDIMON_HEBREW_DIACRITIZED_REPO_URL: Final[str] = (
    "https://github.com/elazarg/hebrew_diacritized"
)

# Default layout after `scripts/fetch_main_nikud_data.sh` (paths relative to this package root).
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
THIRD_PARTY_UNIKUD_ROOT: Final[str] = str(_PROJECT_ROOT / "data" / "unikud")
THIRD_PARTY_UNIKUD_DATA: Final[str] = str(_PROJECT_ROOT / "data" / "unikud" / "data")
THIRD_PARTY_NAKDIMON_ROOT: Final[str] = str(_PROJECT_ROOT / "data" / "nakdimon")
THIRD_PARTY_NAKDIMON_HEBREW_DIACRITIZED: Final[str] = str(
    _PROJECT_ROOT / "data" / "nakdimon" / "hebrew_diacritized"
)

# ---------------------------------------------------------------------------
# 1. Nikud vocabulary
#
# MAT_LECT (Em Kriaa) — a letter (ו/י/ה/א) that functions as a vowel marker
# (mater lectionis) rather than a consonant.  Represented by the special token
# '<MAT_LECT>' so that it is distinct from an ordinary unmarked letter ('').
#
# All other classes are Unicode combining-mark strings that are appended after
# the base Hebrew letter in composed text.
# ---------------------------------------------------------------------------
MAT_LECT_TOKEN: Final[str] = "<MAT_LECT>"

NIKUD_CLASSES: Final[tuple[str, ...]] = (
    "",                           # 0  — no mark (consonant with no vowel)
    MAT_LECT_TOKEN,               # 1  — mater lectionis (silent vowel letter)
    "\u05BC",                     # 2  — dagesh / mapiq only ּ
    "\u05B0",                     # 3  — sheva ְ
    "\u05B1",                     # 4  — hataf segol ֱ
    "\u05B2",                     # 5  — hataf patah ֲ
    "\u05B3",                     # 6  — hataf qamats ֳ
    "\u05B4",                     # 7  — hiriq ִ
    "\u05B5",                     # 8  — tsere ֵ
    "\u05B6",                     # 9  — segol ֶ
    "\u05B7",                     # 10 — patah ַ
    "\u05B8",                     # 11 — qamats ָ
    "\u05B9",                     # 12 — holam ֹ
    "\u05BA",                     # 13 — holam haser for vav ֺ
    "\u05BB",                     # 14 — qubuts ֻ
    "\u05BC\u05B0",               # 15 — dagesh + sheva
    "\u05BC\u05B1",               # 16 — dagesh + hataf segol
    "\u05BC\u05B2",               # 17 — dagesh + hataf patah
    "\u05BC\u05B3",               # 18 — dagesh + hataf qamats
    "\u05BC\u05B4",               # 19 — dagesh + hiriq
    "\u05BC\u05B5",               # 20 — dagesh + tsere
    "\u05BC\u05B6",               # 21 — dagesh + segol
    "\u05BC\u05B7",               # 22 — dagesh + patah
    "\u05BC\u05B8",               # 23 — dagesh + qamats
    "\u05BC\u05B9",               # 24 — dagesh + holam
    "\u05BC\u05BA",               # 25 — dagesh + holam haser for vav
    "\u05BC\u05BB",               # 26 — dagesh + qubuts
    "\u05C7",                     # 27 — qamats qatan ׇ
    "\u05BC\u05C7",               # 28 — dagesh + qamats qatan
)

NIKUD_TO_ID: Final[dict[str, int]] = {n: i for i, n in enumerate(NIKUD_CLASSES)}
ID_TO_NIKUD: Final[dict[int, str]] = {i: n for i, n in enumerate(NIKUD_CLASSES)}
NUM_NIKUD_CLASSES: Final[int] = len(NIKUD_CLASSES)

# ---------------------------------------------------------------------------
# 2. Shin / sin dot vocabulary
# Only meaningful for the letter ש.  All other letters always get IGNORE_INDEX
# for this head during training.
# ---------------------------------------------------------------------------
SHIN_CLASSES: Final[tuple[str, ...]] = (
    "\u05C1",   # 0 — shin dot ׁ
    "\u05C2",   # 1 — sin dot ׂ
)
SHIN_TO_ID: Final[dict[str, int]] = {s: i for i, s in enumerate(SHIN_CLASSES)}
ID_TO_SHIN: Final[dict[int, str]] = {i: s for i, s in enumerate(SHIN_CLASSES)}
NUM_SHIN_CLASSES: Final[int] = len(SHIN_CLASSES)

SHIN_LETTER: Final[str] = "ש"

# ---------------------------------------------------------------------------
# 3. Unicode helpers
# ---------------------------------------------------------------------------

# Hebrew Unicode range א–ת
ALEF_ORD: Final[int] = ord("א")
TAF_ORD: Final[int] = ord("ת")

# All nikud combining marks we handle (excluding MAT_LECT which is virtual)
NIKUD_COMBINING: Final[frozenset[str]] = frozenset(
    c for cls in NIKUD_CLASSES if cls != "" and cls != MAT_LECT_TOKEN for c in cls
) | frozenset(SHIN_CLASSES)

# Label ignore index — used for non-Hebrew positions (CLS, SEP, spaces, punct)
IGNORE_INDEX: Final[int] = -100


def is_hebrew_letter(char: str) -> bool:
    return ALEF_ORD <= ord(char) <= TAF_ORD


def strip_nikud(text: str) -> str:
    """Remove all nikud diacritics from Hebrew text."""
    import unicodedata
    import regex as re
    text = unicodedata.normalize("NFD", text)
    return re.sub(r"[\p{M}|]", "", text)
