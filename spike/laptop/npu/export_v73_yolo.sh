#!/usr/bin/env bash
# Export probe + YOLO11n int8 for the V73 / 2 MB-VTCM stand-in (SSG2115P) into artifacts/npu/models_v73.
# W1 showed this phone's HTP is V73 with at most 2 MB VTCM, so only SSG2115P-compiled contexts load.
set -euo pipefail
WORK="${WORK:-$HOME/bharatai-build}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
QNN_SDK_ROOT="$(dirname "$(dirname "$(find "$WORK/qairt" -type d -path '*/lib/aarch64-android' | head -n1)")")"
export QNN_SDK_ROOT
# shellcheck disable=SC1091
source "$WORK/et-venv/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}"
set +u
# shellcheck disable=SC1091
source "$QNN_SDK_ROOT/bin/envsetup.sh"
set -u
cd "$WORK"   # ultralytics looks for yolo11n.pt in the CWD
python "$REPO_ROOT/spike/laptop/npu/export_qnn_probe.py" \
  --out "$REPO_ROOT/spike/artifacts/npu/models_v73" --socs SSG2115P --yolo-int8 "$@"
cat "$REPO_ROOT/spike/artifacts/npu/models_v73/manifest.txt"
