#!/usr/bin/env bash
# W4-A: build vision_server for the phone (ADR-022). Run in WSL:
#   bash edge/vision_server/build_android.sh            # ExecuTorch (QNN + XNNPACK) + vision_server
#   SKIP_ET=1 bash edge/vision_server/build_android.sh  # only rebuild vision_server
# 1. ExecuTorch v1.5.0 for Android arm64 with BOTH the Qualcomm QNN (NPU) and XNNPACK (CPU) backends,
#    installed to $WORK/et-android-vision (the W1 trees each had only one of them).
# 2. vision_server (this folder) against that install -> $WORK/edge-bin/vision_server
set -euo pipefail
WORK="${WORK:-$HOME/bharatai-build}"
ET="$WORK/executorch"
PREFIX="$WORK/et-android-vision"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ANDROID_NDK_ROOT="$WORK/android-ndk-r28c"
export QNN_SDK_ROOT="$(dirname "$(dirname "$(find "$WORK/qairt" -type d -path '*/lib/aarch64-android' | head -n1)")")"
TC="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake"
JOBS="${JOBS:-6}"
# shellcheck disable=SC1091
source "$WORK/et-venv/bin/activate"
COMMON=(-DCMAKE_TOOLCHAIN_FILE="$TC" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 -DCMAKE_BUILD_TYPE=Release)

if [ "${SKIP_ET:-0}" != "1" ]; then
  echo "==> [1/2] ExecuTorch (QNN + XNNPACK) for Android -> $PREFIX"
  cmake -S "$ET" -B "$WORK/et-android-vision-build" -G Ninja "${COMMON[@]}" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DPYTHON_EXECUTABLE=python \
    -DQNN_SDK_ROOT="$QNN_SDK_ROOT" \
    -DEXECUTORCH_BUILD_QNN=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON
  cmake --build "$WORK/et-android-vision-build" -j"$JOBS" --target install
fi

echo "==> [2/2] vision_server"
cmake -S "$HERE" -B "$WORK/vision-server-build" -G Ninja "${COMMON[@]}" \
  -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_FIND_ROOT_PATH="$PREFIX" \
  -Dexecutorch_DIR="$PREFIX/lib/cmake/ExecuTorch"   # the NDK toolchain restricts find_package to its sysroot
cmake --build "$WORK/vision-server-build" -j"$JOBS"
OUT="$WORK/edge-bin"; mkdir -p "$OUT"   # binaries stay out of the repo
cp "$WORK/vision-server-build/vision_server" "$OUT/"
"$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" --strip-unneeded "$OUT/vision_server"
cp "$(find "$PREFIX" "$WORK/et-android-vision-build" -name libqnn_executorch_backend.so | head -n1)" "$OUT/"
ls -la "$OUT"
