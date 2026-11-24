#!/usr/bin/env bash
# W3-E: int8 NPU export in the et-venv with the Qualcomm SDK environment.
#   bash training/vision/runs/export_npu.sh efficientad <model name> [--quant 16a8w] [--calib N]
#   bash training/vision/runs/export_npu.sh yolo [args]
set -euo pipefail
WORK="$HOME/bharatai-build"
REPO=/mnt/c/Users/Darsh/Desktop/BharatAI-SOC
QNN_SDK_ROOT="$(dirname "$(dirname "$(find "$WORK/qairt" -type d -path '*/lib/aarch64-android' | head -n1)")")"
export QNN_SDK_ROOT PYTHONUNBUFFERED=1
# shellcheck disable=SC1091
source "$WORK/et-venv/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}"
set +u
# shellcheck disable=SC1091
source "$QNN_SDK_ROOT/bin/envsetup.sh" >/dev/null
set -u
kind=$1; shift
python "$REPO/training/vision/export_npu_$kind.py" "$@"
