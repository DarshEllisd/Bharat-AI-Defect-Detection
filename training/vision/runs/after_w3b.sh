#!/usr/bin/env bash
# Follow-up to train_all.sh (GPU, one job at a time):
#  1. re-evaluate the 4 models with the aligned heatmap (P-50); previous results kept as eval_nopad.json
#  2. own_fabric with scale augmentation (camera-distance mismatch, P-51)
set -uo pipefail
export PYTHONUNBUFFERED=1
cd /mnt/c/Users/Darsh/Desktop/BharatAI-SOC
PY=$HOME/bharatai-build/train-venv/bin/python
B=$HOME/bharatai-build
M=$B/models/efficientad
while pgrep -f "[t]rain_all.sh" >/dev/null; do sleep 30; done
ev() {  # name val test extra...
  local name=$1 val=$2 test=$3; shift 3
  [ -f $M/$name/eval.json ] && [ ! -f $M/$name/eval_nopad.json ] && cp $M/$name/eval.json $M/$name/eval_nopad.json
  $PY training/vision/evaluate_efficientad.py --name $name --val $val --test $test "$@"
  echo "[$name] eval exit=$?"
}
for p in wood fabric; do
  ev own_$p $B/datasets/anomaly_own/$p/val/good $B/datasets/anomaly_own/$p/test --boxes $B/datasets/anomaly_own/$p/test_boxes.json
done
for c in wood carpet; do
  ev mvtec_$c $B/datasets/anomaly/$c/val/good $B/data/mvtec/$c/test --masks $B/data/mvtec/$c/ground_truth
done
[ "${SKIP_S18:-0}" = 1 ] || [ -f $M/own_fabric_s18/eval.json ] || $PY training/vision/train_efficientad.py --name own_fabric_s18 --train $B/datasets/anomaly_own/fabric/train/good \
  --val $B/datasets/anomaly_own/fabric/val/good --penalty-dir $B/data/imagenette_flat --geom-aug --scale-aug 1.8 \
  && ev own_fabric_s18 $B/datasets/anomaly_own/fabric/val/good $B/datasets/anomaly_own/fabric/test \
       --boxes $B/datasets/anomaly_own/fabric/test_boxes.json
echo AFTER_DONE
