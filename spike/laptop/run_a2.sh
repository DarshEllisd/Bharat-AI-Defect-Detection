#!/usr/bin/env bash
# A2 driver: install ultralytics, fetch yolo11n.pt (Hugging Face mirror first: GitHub is slow
# over this connection's IPv4), then export .pte files at 320 and 640 px.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-$HOME/bharatai-build}"
# shellcheck disable=SC1091
source "$WORK/et-venv/bin/activate"
cd "$WORK"   # ultralytics writes weights/exports to the CWD; keep them off /mnt/c

echo "== packages before"; pip list 2>/dev/null | grep -i -E '^(torch|torchvision|ultralytics|executorch) ' || true
pip install --quiet "ultralytics>=8.3.220"
echo "== packages after";  pip list 2>/dev/null | grep -i -E '^(torch|torchvision|ultralytics|executorch) '

if [ ! -s yolo11n.pt ]; then
  hf download Ultralytics/YOLO11 yolo11n.pt --local-dir . 2>/dev/null \
    || curl -fL -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
fi
ls -la yolo11n.pt

python "$REPO_ROOT/spike/laptop/export_yolo_pte.py" --weights "$WORK/yolo11n.pt" --sizes 320 640
ls -la "$REPO_ROOT/spike/artifacts/models/vision/"
