#!/usr/bin/env bash
# W3-C: train + evaluate YOLO11n on yolo_combined (log -> ~/bharatai-build/models/yolo/train.log)
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p $HOME/bharatai-build/models/yolo
$HOME/bharatai-build/train-venv/bin/python "$(dirname "${BASH_SOURCE[0]}")/../train_yolo.py" "$@" \
  > $HOME/bharatai-build/models/yolo/train.log 2>&1
echo "exit=$?"; tail -n 40 $HOME/bharatai-build/models/yolo/train.log
