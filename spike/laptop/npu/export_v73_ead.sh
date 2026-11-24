#!/usr/bin/env bash
# Run export_efficientad_probe.py inside the et-venv with the Qualcomm SDK environment loaded.
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
python "$REPO_ROOT/spike/laptop/npu/export_efficientad_probe.py" "$@"
