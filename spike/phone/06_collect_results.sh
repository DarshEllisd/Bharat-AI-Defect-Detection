#!/data/data/com.termux/files/usr/bin/bash
# W1 spike - bundle all results into Download/bharatai/ so they can be copied to the laptop.
set -euo pipefail

B="$HOME/bharatai"
DEST="$HOME/storage/downloads/bharatai"
NAME="w1_results_$(date +%Y%m%d_%H%M%S).tar.gz"

mkdir -p "$DEST"
tar -czf "$DEST/$NAME" -C "$B" results
echo "Saved $DEST/$NAME"
echo "On the laptop:  adb pull /sdcard/Download/bharatai/$NAME spike/results/"
