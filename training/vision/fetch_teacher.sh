#!/usr/bin/env bash
# Download anomalib's pretrained EfficientAD teacher weights (GitHub release; slow over this
# connection's IPv4 and prone to resets, so: resume + retry until the zip tests OK).
#   bash training/vision/fetch_teacher.sh
set -uo pipefail
OUT="${OUT:-$HOME/bharatai-build/models/teachers}"
URL="https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
ZIP="$OUT/efficientad_pretrained_weights.zip"
mkdir -p "$OUT"
for attempt in $(seq 1 30); do
  echo "== attempt $attempt $(date +%H:%M:%S)"
  curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 -C - -o "$ZIP" "$URL" && unzip -tq "$ZIP" && break
  sleep 10
done
unzip -tq "$ZIP" || { echo "download failed"; exit 1; }
unzip -oq "$ZIP" -d "$OUT"
sha256sum "$ZIP"
find "$OUT" -name '*.pth' -exec ls -la {} \;
