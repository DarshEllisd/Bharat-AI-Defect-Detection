#!/usr/bin/env bash
# W1 spike - build ExecuTorch runners for Android arm64 (run in WSL2 Ubuntu).
#
# Produces (copied into spike/artifacts/bin/):
#   llama_main       - ExecuTorch Llama runner (advisory pipeline)
#   executor_runner  - generic .pte runner with XNNPACK (vision pipeline benchmark)
#
# Also leaves a Python venv with ExecuTorch installed from the same source tag,
# used by export_yolo_pte.py and get_llama_pte.sh so exported .pte files match
# the runtime version.
#
# Usage:  bash spike/laptop/build_android_runners.sh
# Env overrides: ET_VERSION, NDK_VERSION, WORK
set -euo pipefail

ET_VERSION="${ET_VERSION:-v1.5.0}"      # ExecuTorch release tag (latest stable when written, Nov 2026)
NDK_VERSION="${NDK_VERSION:-r28c}"      # NDK version the ExecuTorch llama README is tested with
PY_VERSION="${PY_VERSION:-3.12}"        # Ubuntu 22.04 ships 3.10; ExecuTorch 1.5 needs >= 3.11
JOBS="${JOBS:-4}"                       # parallel compile jobs; each can use ~1-2 GB RAM
WORK="${WORK:-$HOME/bharatai-build}"    # keep builds on the Linux filesystem, not /mnt/c (much faster)

mem_gb=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$mem_gb" -lt 8 ]; then
  echo "WARNING: WSL has only ${mem_gb} GB RAM. Raise it in %UserProfile%\\.wslconfig (see spike/README.md), or the build may be killed." >&2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ART_BIN="$REPO_ROOT/spike/artifacts/bin"
ET_DIR="$WORK/executorch"
VENV="$WORK/et-venv"
export ANDROID_NDK="$WORK/android-ndk-$NDK_VERSION"

mkdir -p "$WORK" "$ART_BIN"

echo "==> [1/6] System packages"
APT_PKGS=(git curl unzip file build-essential ninja-build libc++1 libc++abi1)
if dpkg -s "${APT_PKGS[@]}" >/dev/null 2>&1; then
  echo "already installed"
else
  sudo apt-get update
  sudo apt-get install -y "${APT_PKGS[@]}"
fi
# uv provides a standalone Python 3.12 without touching the system Python
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> [2/6] Android NDK $NDK_VERSION"
if [ ! -d "$ANDROID_NDK" ]; then
  curl -fL -o "$WORK/ndk.zip" "https://dl.google.com/android/repository/android-ndk-$NDK_VERSION-linux.zip"
  unzip -q "$WORK/ndk.zip" -d "$WORK"
  rm "$WORK/ndk.zip"
fi

echo "==> [3/6] ExecuTorch $ET_VERSION source"
if [ ! -d "$ET_DIR" ]; then
  git clone --branch "$ET_VERSION" --depth 1 https://github.com/pytorch/executorch.git "$ET_DIR"
fi
cd "$ET_DIR"
git submodule sync
git submodule update --init --recursive

echo "==> [4/6] Python $PY_VERSION venv + install_executorch.sh (editable)"
[ -d "$VENV" ] || uv venv --seed --python "$PY_VERSION" "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip "cmake>=3.29" ninja
# Editable install so backends built later in-tree (the Qualcomm/QNN AOT adaptor) are importable
CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" ./install_executorch.sh --editable

echo "==> [5/6] Build ExecuTorch libraries + executor_runner for arm64-v8a"
# Flags follow examples/models/llama/README.md (Android section), plus executor_runner.
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
    -Bcmake-out-android .
cmake --build cmake-out-android -j"$JOBS" --target install --config Release

echo "==> [6/6] Build llama_main for arm64-v8a"
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -DSUPPORT_REGEX_LOOKAHEAD=ON \
    -Bcmake-out-android/examples/models/llama \
    examples/models/llama
cmake --build cmake-out-android/examples/models/llama -j"$JOBS" --config Release

echo "==> Collecting binaries into $ART_BIN"
cp cmake-out-android/examples/models/llama/llama_main "$ART_BIN/"
RUNNER="$(find cmake-out-android -maxdepth 4 -type f \( -name executor_runner -o -name xnn_executor_runner \) | head -n1)"
if [ -n "$RUNNER" ]; then
  cp "$RUNNER" "$ART_BIN/executor_runner"
  # Debug info makes it ~155 MB; stripped it's ~7 MB (phone storage is tight)
  "$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" --strip-unneeded "$ART_BIN/executor_runner"
else
  echo "WARNING: executor_runner not found - vision benchmark (04) will be skipped; route A still works." >&2
fi
file "$ART_BIN"/* || true

cat > "$REPO_ROOT/spike/artifacts/BUILD_INFO.txt" <<EOF
executorch_tag=$ET_VERSION
ndk=$NDK_VERSION
built=$(date -Iseconds)
venv=$VENV
executorch_src=$ET_DIR
EOF
echo "Done. Next: python spike/laptop/export_yolo_pte.py  (inside: source $VENV/bin/activate)"
