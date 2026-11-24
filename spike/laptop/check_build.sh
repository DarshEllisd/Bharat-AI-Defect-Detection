#!/usr/bin/env bash
# Quick status of the W1 laptop build (safe to run any time, read-only).
#   bash spike/laptop/check_build.sh
W="${WORK:-$HOME/bharatai-build}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "== Memory";        free -h | head -n 2
echo "== Work dir";      ls -la "$W" 2>&1
echo "== NDK";           ls -d "$W"/android-ndk-* 2>&1
echo "== ExecuTorch";    [ -d "$W/executorch" ] && git -C "$W/executorch" describe --tags --always 2>&1
echo "== Venv";          "$W/et-venv/bin/python" --version 2>&1
"$W/et-venv/bin/pip" list 2>/dev/null | grep -i -E '^(torch|executorch|cmake) '
echo "== Build dirs";    ls -d "$W"/executorch/cmake-out* "$W"/executorch/pip-out 2>&1
echo "== Artifacts";     ls -la "$REPO_ROOT/spike/artifacts/bin" 2>&1
echo "== Busiest processes"
ps -eo pid,etime,%cpu,%mem,args --sort=-%cpu | grep -v -E 'ps -eo|check_build' | head -n 12 | cut -c1-200
