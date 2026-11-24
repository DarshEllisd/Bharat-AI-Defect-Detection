#!/usr/bin/env bash
# Counts of own photos: used (per product/label) vs excluded.
RAW="${RAW:-$HOME/bharatai-build/data/own/raw}"
echo "== used photos"
find "$RAW" -path "$RAW/_excluded" -prune -o -name '*.jpg' -print | awk -F/ '{print $(NF-2)"/"$(NF-1)}' | sort | uniq -c
echo "== excluded photos: $(find "$RAW/_excluded" -name '*.jpg' 2>/dev/null | wc -l)"
