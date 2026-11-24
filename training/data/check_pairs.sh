#!/usr/bin/env bash
# Every own photo needs its JSON sidecar (and vice versa). Lists any orphans and leftover .part files.
#   bash training/data/check_pairs.sh [filename-stamp, e.g. 20261121]
RAW="${RAW:-$HOME/bharatai-build/data/own/raw}"
STAMP="${1:-}"
jpg=$(find "$RAW" -name "*${STAMP}*.jpg" | wc -l)
json=$(find "$RAW" -name "*${STAMP}*.json" | wc -l)
echo "jpg=$jpg json=$json"
find "$RAW" -name "*${STAMP}*.jpg" | while read -r f; do [ -f "${f%.jpg}.json" ] || echo "missing sidecar: ${f#$RAW/}"; done
find "$RAW" -name "*${STAMP}*.json" | while read -r f; do [ -f "${f%.json}.jpg" ] || echo "missing photo:   ${f#$RAW/}"; done
find "$RAW" -name "*.part" | sed "s|^|leftover partial: |"
