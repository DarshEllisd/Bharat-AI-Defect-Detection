#!/usr/bin/env bash
# W1 spike - run the NPU probe through `adb shell` (Qualcomm's documented developer route)
# and pull the results back. Compare with the Termux run from 07_npu_probe.sh.
#
#   bash spike/laptop/npu/run_npu_adb.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NPU="$REPO_ROOT/spike/artifacts/npu"
DEV="/data/local/tmp/bharatai_npu"
RESULTS="$REPO_ROOT/spike/results/npu_adb"

[ -d "$NPU/models" ] || { echo "Run spike/laptop/npu/build_qnn.sh first." >&2; exit 1; }
if command -v adb >/dev/null 2>&1; then ADB=adb
elif command -v adb.exe >/dev/null 2>&1; then ADB=adb.exe
else echo "adb not found." >&2; exit 1
fi
# Windows adb needs Windows paths for laptop files: convert explicitly (WSL: wslpath,
# Git Bash: cygpath), so MSYS_NO_PATHCONV=1 can protect the phone-side paths
host_path() {
  if [ "$ADB" = "adb.exe" ] && command -v wslpath >/dev/null 2>&1; then wslpath -w "$1"
  elif command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"
  else echo "$1"; fi
}
export MSYS_NO_PATHCONV=1

"$ADB" get-state >/dev/null
"$ADB" shell "rm -rf $DEV && mkdir -p $DEV"
for sub in bin lib dsp models; do "$ADB" push "$(host_path "$NPU/$sub")" "$DEV/"; done
"$ADB" push "$(host_path "$REPO_ROOT/spike/phone/07_npu_probe.sh")" "$DEV/"
"$ADB" shell "cd $DEV && sh ./07_npu_probe.sh $DEV $DEV/results"

rm -rf "$RESULTS"
mkdir -p "$(dirname "$RESULTS")"
"$ADB" pull "$DEV/results" "$(host_path "$RESULTS")"

# shellcheck disable=SC1091
[ -f "$HOME/bharatai-build/et-venv/bin/activate" ] && source "$HOME/bharatai-build/et-venv/bin/activate"
# Path conversion back on for Python (MSYS_NO_PATHCONV is only for adb's phone-side paths)
env -u MSYS_NO_PATHCONV python "$REPO_ROOT/spike/phone/npu_compare.py" "$RESULTS" "$NPU/models" | tee "$RESULTS/summary.csv"
echo "Full report: $RESULTS/npu_probe.txt"
