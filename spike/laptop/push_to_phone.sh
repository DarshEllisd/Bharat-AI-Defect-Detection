#!/usr/bin/env bash
# W1 spike - copy phone scripts + built artifacts to the phone's Download folder over USB (adb).
# Works from WSL2 (uses Windows adb.exe) or Git Bash with adb on PATH.
#
#   bash spike/laptop/push_to_phone.sh            # scripts + binaries + models
#   bash spike/laptop/push_to_phone.sh scripts    # scripts only (quick re-push after edits)
#
# No adb? Copy spike/phone, spike/artifacts/bin and spike/artifacts/models into
# the phone's Download/bharatai/ folder over a normal USB file transfer instead.
set -euo pipefail

WHAT="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="/sdcard/Download/bharatai"

if command -v adb >/dev/null 2>&1; then ADB=adb
elif command -v adb.exe >/dev/null 2>&1; then ADB=adb.exe
else echo "adb not found. Install Android platform-tools and enable USB debugging." >&2; exit 1
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
"$ADB" shell mkdir -p "$DEST"
"$ADB" push "$(host_path "$REPO_ROOT/spike/phone")" "$DEST/"
if [ "$WHAT" = "all" ]; then
  [ -d "$REPO_ROOT/spike/artifacts/bin" ] && "$ADB" push "$(host_path "$REPO_ROOT/spike/artifacts/bin")" "$DEST/"
  [ -d "$REPO_ROOT/spike/artifacts/models" ] && "$ADB" push "$(host_path "$REPO_ROOT/spike/artifacts/models")" "$DEST/"
  # NPU probe bundle (from npu/build_qnn.sh), for the Termux run of 07_npu_probe.sh
  [ -d "$REPO_ROOT/spike/artifacts/npu" ] && "$ADB" push "$(host_path "$REPO_ROOT/spike/artifacts/npu")" "$DEST/"
fi
"$ADB" shell ls -R "$DEST" | head -n 40
echo "Pushed to $DEST. On the phone (Termux): see spike/README.md, step B2."
