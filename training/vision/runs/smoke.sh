#!/usr/bin/env bash
# W3-B smoke test: 600 iterations on MVTec wood, then evaluation (checks the whole pipeline end to end).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PY=$HOME/bharatai-build/train-venv/bin/python
B=$HOME/bharatai-build
$PY training/vision/train_efficientad.py --name smoke_wood --iters 600 "$@" \
  --train $B/datasets/anomaly/wood/train90/good --val $B/datasets/anomaly/wood/val/good \
  --penalty-dir $B/data/imagenette_flat --geom-aug
$PY training/vision/evaluate_efficientad.py --name smoke_wood \
  --val $B/datasets/anomaly/wood/val/good --test $B/data/mvtec/wood/test --masks $B/data/mvtec/wood/ground_truth
