#!/usr/bin/env bash
# Re-evaluate the deployed anomaly models after the preprocessing fix (P-60). GPU, ~1 min each.
# Previous results are kept as eval_noaa.json (no antialias).
export PYTHONUNBUFFERED=1
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PY=$HOME/bharatai-build/train-venv/bin/python; B=$HOME/bharatai-build; M=$B/models/efficientad
ev() { local name=$1 val=$2 test=$3; shift 3
  [ -f $M/$name/eval.json ] && [ ! -f $M/$name/eval_noaa.json ] && cp $M/$name/eval.json $M/$name/eval_noaa.json
  $PY training/vision/evaluate_efficientad.py --name $name --val $val --test $test "$@" > /dev/null && echo "[$name] ok"; }
A=$B/datasets/anomaly_own
ev own_wood $A/wood/val/good $A/wood/test --boxes $A/wood/test_boxes.json
ev own_fabric_t2 $A/fabric/val/good $A/fabric/test --boxes $A/fabric/test_boxes.json
for c in wood carpet; do ev mvtec_$c $B/datasets/anomaly/$c/val/good $B/data/mvtec/$c/test --masks $B/data/mvtec/$c/ground_truth; done
for m in own_wood own_fabric_t2 mvtec_wood mvtec_carpet; do python3 -c "
import json; o=json.load(open('$M/$m/eval_noaa.json')); n=json.load(open('$M/$m/eval.json'))
ks=['image_auroc','pixel_auroc','false_alarm_rate','recall','box_hit_rate']
print('$m', {k:(o.get(k), n.get(k)) for k in ks if n.get(k) is not None}, n['recall_per_type'])"; done
