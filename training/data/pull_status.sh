#!/usr/bin/env bash
# Progress of a running/finished pull_captures.sh (log: ~/bharatai-build/pull.log).
#   bash training/data/pull_status.sh [date-stamp-in-filenames, e.g. 20261121]
STAMP="${1:-}"
LOG="$HOME/bharatai-build/pull.log"
grep -E '== attempt|PULL OK|failed|Verified|Copied|did not|size mismatch|no devices' "$LOG" 2>/dev/null | tail -n 12
if [ -n "$STAMP" ]; then
  echo "photos with $STAMP already on the laptop: $(find "$HOME/bharatai-build/data/own/raw" -name "*${STAMP}*.jpg" | wc -l)"
fi
echo "pull still running: $(pgrep -fc pull_captures.sh)"
echo "adb devices: $(adb.exe devices 2>/dev/null | tr -d '\r' | tail -n +2 | grep -v '^$' || echo none)"
date +%H:%M:%S
