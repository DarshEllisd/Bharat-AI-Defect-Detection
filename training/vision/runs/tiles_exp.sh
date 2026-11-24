#!/usr/bin/env bash
# P-57 experiment: own_fabric with 2x2 tiles (2x resolution for thin defects such as thread).
# Starts after resume_w3b.sh (one GPU job at a time). Resumable (checkpoints, skip if evaluated).
set -uo pipefail
export PYTHONUNBUFFERED=1
cd /mnt/c/Users/Darsh/Desktop/BharatAI-SOC
PY=$HOME/bharatai-build/train-venv/bin/python
B=$HOME/bharatai-build; A=$B/datasets/anomaly_own/fabric; M=$B/models/efficientad
while pgrep -f "[r]esume_w3b.sh" >/dev/null; do sleep 30; done
if [ ! -f $M/own_fabric_t2/eval.json ]; then
  $PY training/vision/train_efficientad.py --name own_fabric_t2 --tiles 2 --train $A/train/good --val $A/val/good \
      --penalty-dir $B/data/imagenette_flat --geom-aug \
    && $PY training/vision/evaluate_efficientad.py --name own_fabric_t2 --val $A/val/good --test $A/test --boxes $A/test_boxes.json
fi
echo TILES_DONE
