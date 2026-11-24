#!/usr/bin/env bash
# W3-B: train + evaluate the four EfficientAD-S models (ADR-018/019), one after another on the GPU
# (one GPU-resident process at ~23 it/s beats two parallel lanes and keeps the CPU free - P-48).
#   bash training/vision/runs/train_all.sh          logs -> ~/bharatai-build/models/efficientad/logs/
set -uo pipefail
cd /mnt/c/Users/Darsh/Desktop/BharatAI-SOC
export PYTHONUNBUFFERED=1
PY=$HOME/bharatai-build/train-venv/bin/python
B=$HOME/bharatai-build
LOGS=$B/models/efficientad/logs; mkdir -p "$LOGS"
PEN="--penalty-dir $B/data/imagenette_flat --geom-aug"

run() {  # name train val test [extra eval args]
  local name=$1 train=$2 val=$3 test=$4; shift 4
  if [ -f "$B/models/efficientad/$name/model.pt" ] && [ -f "$B/models/efficientad/$name/eval.json" ]; then
    echo "[$name] already trained + evaluated - skipped"; return; fi   # resumable (2026-11-21 pause)
  $PY training/vision/train_efficientad.py --name "$name" --train "$train" --val "$val" $PEN \
    && $PY training/vision/evaluate_efficientad.py --name "$name" --val "$val" --test "$test" "$@"
  echo "[$name] exit=$?"
}
own() { run "own_$1" $B/datasets/anomaly_own/$1/train/good $B/datasets/anomaly_own/$1/val/good \
          $B/datasets/anomaly_own/$1/test --boxes $B/datasets/anomaly_own/$1/test_boxes.json; }
mvt() { run "mvtec_$1" $B/datasets/anomaly/$1/train90/good $B/datasets/anomaly/$1/val/good \
          $B/data/mvtec/$1/test --masks $B/data/mvtec/$1/ground_truth; }

{ own wood; own fabric; mvt wood; mvt carpet; } >> "$LOGS/train_all.log" 2>&1
echo ALL_DONE
