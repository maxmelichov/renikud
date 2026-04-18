#!/usr/bin/env bash
# Vendor the main diacritized-Hebrew corpora used for nikud training:
#   - UNIKUD (GitHub + DVC large files via Dagshub)
#   - Nakdimon (GitHub repo + hebrew_diacritized submodule)
#
# Run from repo root:
#   bash scripts/fetch_main_nikud_data.sh
#
# References:
#   https://github.com/morrisalp/unikud/tree/main/data
#   https://github.com/elazarg/nakdimon

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT}/data"
mkdir -p "${TP}"

clone_or_update() {
  local url="$1"
  local dest="$2"
  local extra="${3:-}"
  if [[ -d "${dest}/.git" ]]; then
    echo "Updating existing clone: ${dest}"
    git -C "${dest}" fetch --all --prune
    git -C "${dest}" pull --ff-only
    if [[ -n "${extra}" ]]; then
      eval "${extra}"
    fi
  else
    echo "Cloning ${url} -> ${dest}"
    git clone "${url}" "${dest}"
    if [[ -n "${extra}" ]]; then
      eval "${extra}"
    fi
  fi
}

# --- UNIKUD (data under data/raw via DVC; see upstream README) ---
clone_or_update "https://github.com/morrisalp/unikud.git" "${TP}/unikud" ""

if command -v dvc >/dev/null 2>&1; then
  echo "Running DVC in ${TP}/unikud (pulls data/raw, etc.)..."
  (
    cd "${TP}/unikud"
    REMOTES="$(dvc remote list 2>/dev/null || true)"
    case "${REMOTES}" in
      *"origin "*) ;;
      *) dvc remote add -d origin "https://dagshub.com/morrisalp/unikud.dvc" || true ;;
    esac
    dvc pull -r origin
  ) || {
    echo "Note: dvc pull failed. Install DVC and ensure Dagshub access, then run:"
    echo "  cd ${TP}/unikud && dvc remote add -d origin https://dagshub.com/morrisalp/unikud.dvc && dvc pull -r origin"
  }
else
  echo "DVC not found in PATH; skipped large file pull for UNIKUD."
  echo "Install DVC (https://dvc.org/) then run:"
  echo "  cd ${TP}/unikud && dvc remote add -d origin https://dagshub.com/morrisalp/unikud.dvc && dvc pull -r origin"
fi

# --- Nakdimon + hebrew_diacritized training submodule ---
if [[ -d "${TP}/nakdimon/.git" ]]; then
  echo "Updating nakdimon submodules..."
  git -C "${TP}/nakdimon" submodule update --init --recursive
  git -C "${TP}/nakdimon" pull --ff-only
else
  echo "Cloning Nakdimon with submodules..."
  git clone --recurse-submodules "https://github.com/elazarg/nakdimon.git" "${TP}/nakdimon"
fi

echo
echo "Done. Local paths (see src/constants.py):"
echo "  UNIKUD:              ${TP}/unikud/data"
echo "  Nakdimon train text: ${TP}/nakdimon/hebrew_diacritized"
