#!/usr/bin/env bash
# W1 spike - NPU test, laptop side (run in WSL2 AFTER build_android_runners.sh).
#
# 1. Downloads the Qualcomm AI Runtime (QAIRT / QNN) SDK
# 2. Builds ExecuTorch's Qualcomm backend: x86 AOT compiler + Android qnn_executor_runner
# 3. Compiles probe models for every HTP architecture ExecuTorch knows (export_qnn_probe.py)
# 4. Bundles everything the phone needs into spike/artifacts/npu/
#
# The SoC isn't in ExecuTorch's chipset list, so the probe compiles for listed chipsets of
# each Hexagon generation (v68...v81). The phone then shows empirically which, if any, runs.
#
#   bash spike/laptop/npu/build_qnn.sh
#   QNN_ZIP=/path/to/v2.37.0.250724.zip bash spike/laptop/npu/build_qnn.sh   # if the download needs a login
set -euo pipefail

WORK="${WORK:-$HOME/bharatai-build}"
ET_DIR="$WORK/executorch"
VENV="$WORK/et-venv"
NDK_VERSION="${NDK_VERSION:-r28c}"
QNN_VERSION="${QNN_VERSION:-2.37.0.250724}"   # version ExecuTorch v1.5.0 docs/scripts pin (main uses newer)
QNN_ZIP="${QNN_ZIP:-$WORK/qairt-$QNN_VERSION.zip}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="$REPO_ROOT/spike/artifacts/npu"

[ -d "$ET_DIR" ] || { echo "Run spike/laptop/build_android_runners.sh first." >&2; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> [1/4] QAIRT SDK $QNN_VERSION"
QNN_BASE="$WORK/qairt/$QNN_VERSION"
if [ ! -d "$QNN_BASE" ]; then
  if [ ! -f "$QNN_ZIP" ]; then
    curl -fL -o "$QNN_ZIP" \
      "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/$QNN_VERSION/v$QNN_VERSION.zip" \
      || { echo "Download failed. Get the SDK from https://softwarecenter.qualcomm.com (Qualcomm AI Runtime) and re-run with QNN_ZIP=<zip>." >&2; exit 1; }
  fi
  mkdir -p "$QNN_BASE"
  unzip -q "$QNN_ZIP" -d "$QNN_BASE"
fi
ANDROID_LIB_DIR="$(find "$QNN_BASE" -type d -path '*/lib/aarch64-android' | head -n1)"
[ -n "$ANDROID_LIB_DIR" ] || { echo "lib/aarch64-android not found under $QNN_BASE" >&2; exit 1; }
export QNN_SDK_ROOT="$(dirname "$(dirname "$ANDROID_LIB_DIR")")"
export ANDROID_NDK_ROOT="$WORK/android-ndk-$NDK_VERSION"
export EXECUTORCH_ROOT="$ET_DIR"
echo "QNN_SDK_ROOT=$QNN_SDK_ROOT"

cd "$ET_DIR"
# ONLY_MODELS=1 re-runs just step 4 (probe models) against an existing build and bundle
if [ "${ONLY_MODELS:-0}" != "1" ]; then
echo "==> [2/4] Build ExecuTorch Qualcomm backend (x86 AOT + Android runtime)"
./backends/qualcomm/scripts/build.sh --release

echo "==> [3/4] Bundle runtime files into $OUT"
rm -rf "$OUT"
mkdir -p "$OUT/bin" "$OUT/lib" "$OUT/dsp" "$OUT/models"
cp "$(find build-android -type f -name qnn_executor_runner | head -n1)" "$OUT/bin/"
"$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" --strip-unneeded "$OUT/bin/qnn_executor_runner"
cp "$(find build-android -type f -name libqnn_executorch_backend.so | head -n1)" "$OUT/lib/"
VALIDATOR="$QNN_SDK_ROOT/bin/aarch64-android/qnn-platform-validator"
[ -f "$VALIDATOR" ] && cp "$VALIDATOR" "$OUT/bin/" || echo "NOTE: qnn-platform-validator not in this SDK; skipping that test."
cp "$ANDROID_LIB_DIR"/*.so "$OUT/lib/"
# Runner may be linked against the shared C++ runtime
cp "$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "$OUT/lib/" 2>/dev/null || true
# DSP-side skeleton libraries, one folder per Hexagon generation (unsigned = no Qualcomm signing needed)
for d in "$QNN_SDK_ROOT"/lib/hexagon-v*/unsigned; do
  [ -d "$d" ] || continue
  arch="$(basename "$(dirname "$d")" | sed 's/hexagon-//')"
  mkdir -p "$OUT/dsp/$arch"
  cp "$d"/*.so "$OUT/dsp/$arch/"
done
fi

echo "==> [4/4] Compile probe models for each HTP generation"
# envsetup.sh puts the x86 QNN libraries on LD_LIBRARY_PATH for the AOT compiler.
# It reads $PYTHONPATH unguarded, which `set -u` turns into a fatal error.
export PYTHONPATH="${PYTHONPATH:-}"
set +u
# shellcheck disable=SC1091
source "$QNN_SDK_ROOT/bin/envsetup.sh"
set -u
mkdir -p "$OUT/models"
python "$REPO_ROOT/spike/laptop/npu/export_qnn_probe.py" --out "$OUT/models" "$@"

cat > "$OUT/BUILD_INFO.txt" <<EOF
qnn_sdk=$QNN_VERSION
executorch_src=$ET_DIR ($(git -C "$ET_DIR" describe --tags --always 2>/dev/null))
built=$(date -Iseconds)
EOF
du -sh "$OUT"/*
echo "Done. Next: bash spike/laptop/npu/run_npu_adb.sh   (and push_to_phone.sh for the Termux run)"
