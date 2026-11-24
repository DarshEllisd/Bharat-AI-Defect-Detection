#!/usr/bin/env bash
# W3-E: quantised NPU exports (anomaly 16a8w, YOLO 8a8w) + int8-vs-fp32 accuracy for the deployed models, one at a time (CPU).
#   bash training/vision/runs/export_all.sh        log -> ~/bharatai-build/models/npu/export_all.log
# Skips a model whose export_report.json exists (resumable).
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N=$HOME/bharatai-build/models/npu; mkdir -p $N
Q=${Q:-16a8w}   # anomaly models: 16-bit activations (8a8w collapses the squared-difference maps, P-59)
for m in mvtec_wood own_wood own_fabric_t2 mvtec_carpet; do
  [ -f $N/${m}_$Q/export_report.json ] && { echo "[$m] done - skipped"; continue; }
  echo "== $m $(date +%H:%M)"
  bash $D/export_npu.sh efficientad $m --quant $Q 2>&1 | grep -v -E "^\[INFO\]|^INFO|[Ww]arn" | tail -n 25
done
[ -f $N/yolo11n_combined_8a8w/export_report.json ] || { echo "== yolo $(date +%H:%M)"; bash $D/export_npu.sh yolo 2>&1 | grep -v -E "^\[INFO\]|^INFO|[Ww]arn|Scanning|━" | tail -n 30; }
echo "EXPORT_ALL_DONE $(date +%H:%M)"
