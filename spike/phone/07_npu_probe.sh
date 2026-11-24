#!/bin/sh
# W1 spike - empirical NPU (Hexagon HTP) test. POSIX sh so it runs both in Termux and in
# `adb shell`, because Android may allow DSP access from one context and not the other.
#
#   Termux:    sh ~/bharatai/spike/07_npu_probe.sh
#   adb shell: driven by spike/laptop/npu/run_npu_adb.sh
#   Args:      [NPU_DIR (bin/ lib/ dsp/ models/)] [OUT_DIR]
#
# Three levels of evidence, written to OUT_DIR/npu_probe.txt:
#   1. Device facts: SoC id, FastRPC device nodes, vendor DSP libraries (no SDK needed)
#   2. qnn-platform-validator: does Qualcomm's own runtime reach the HTP, and which Hexagon version
#   3. ExecuTorch: do QNN-delegated .pte files run on the HTP and match the CPU reference

NPU_DIR="${1:-$HOME/bharatai/npu}"
OUT="${2:-$HOME/bharatai/results/npu_termux}"
# MODELS_DIR overrides where manifest.txt and the .pte files live (default: NPU_DIR/models)
MODELS="${MODELS_DIR:-$NPU_DIR/models}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORT="$OUT/npu_probe.txt"
mkdir -p "$OUT/logs" "$OUT/outputs"
chmod +x "$NPU_DIR"/bin/* 2>/dev/null
: > "$REPORT"

say() { echo "$*" | tee -a "$REPORT"; }

VENDOR_DSP="/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp;/dsp"
ALL_DSP=""
for d in "$NPU_DIR"/dsp/v*; do [ -d "$d" ] && ALL_DSP="$ALL_DSP$d;"; done

say "== NPU probe $(date) as $(id -un 2>/dev/null) context=$(id -Z 2>/dev/null || echo '?')"

say ""; say "== 1. Device facts"
for p in ro.soc.manufacturer ro.soc.model ro.board.platform ro.hardware; do say "$p=$(getprop $p)"; done
for f in machine soc_id family revision; do
  [ -r /sys/devices/soc0/$f ] && say "soc0/$f=$(cat /sys/devices/soc0/$f)"
done
say "FastRPC device nodes:"
ls -l /dev/fastrpc* /dev/adsprpc-smd* 2>&1 | tee -a "$REPORT"
say "libcdsprpc (CPU-side DSP RPC library):"
ls -l /vendor/lib64/libcdsprpc.so 2>&1 | tee -a "$REPORT"
say "libcdsprpc exposed to apps (public.libraries):"
cat /vendor/etc/public.libraries.txt /system/etc/public.libraries.txt 2>/dev/null | grep -i cdsprpc | tee -a "$REPORT"
say "Vendor-shipped HTP/QNN/SNPE libraries (their names reveal the Hexagon version, e.g. V73):"
for d in /vendor/lib64 /vendor/dsp/cdsp /vendor/lib/rfsa/adsp /vendor/dsp /odm/lib64; do
  ls "$d" 2>/dev/null | grep -i -E 'htp|qnn|snpe|hexagon' | sed "s|^|  $d/|"
done | head -n 40 | tee -a "$REPORT"

say ""; say "== 2. qnn-platform-validator (one run per Hexagon generation's skeleton libraries)"
if [ -x "$NPU_DIR/bin/qnn-platform-validator" ]; then
  for d in "$NPU_DIR"/dsp/v*; do
    a="$(basename "$d")"
    log="$OUT/logs/validator_$a.log"
    ( cd "$OUT" && LD_LIBRARY_PATH="$NPU_DIR/lib" ADSP_LIBRARY_PATH="$d;$VENDOR_DSP" \
        "$NPU_DIR/bin/qnn-platform-validator" --backend dsp --coreVersion --libVersion --testBackend \
        < /dev/null > "$log" 2>&1 )
    rc=$?
    say "validator skel=$a rc=$rc :: $(grep -i -E 'pass|fail|support|version|error' "$log" | head -n 6 | tr '\n' ' ')"
  done
else
  say "skipped (validator binary not bundled)"
fi

say ""; say "== 3. ExecuTorch QNN-delegated models"
MANIFEST="$MODELS/manifest.txt"
if [ -x "$NPU_DIR/bin/qnn_executor_runner" ] && [ -f "$MANIFEST" ]; then
  while read -r name pte input ref arch soc variant; do
    case "$name" in ''|'#'*) continue ;; esac
    od="$OUT/outputs/$name"
    mkdir -p "$od"
    echo "$MODELS/$input" > "$od/input_list.txt"
    log="$OUT/logs/$name.log"
    # stdin from /dev/null so the runner can't swallow the rest of the manifest
    LD_LIBRARY_PATH="$NPU_DIR/lib" ADSP_LIBRARY_PATH="$ALL_DSP$VENDOR_DSP" \
      "$NPU_DIR/bin/qnn_executor_runner" --model_path "$MODELS/$pte" \
      --input_list_path "$od/input_list.txt" --output_folder_path "$od" \
      --iteration 20 --warm_up 3 \
      < /dev/null > "$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      # Retry with vendor/system lib dirs on the search path, in case the linker namespace hides libcdsprpc
      LD_LIBRARY_PATH="$NPU_DIR/lib:/vendor/lib64:/system/lib64" ADSP_LIBRARY_PATH="$ALL_DSP$VENDOR_DSP" \
        "$NPU_DIR/bin/qnn_executor_runner" --model_path "$MODELS/$pte" \
        --input_list_path "$od/input_list.txt" --output_folder_path "$od" \
        --iteration 20 --warm_up 3 \
        < /dev/null > "$log.retry" 2>&1
      rc2=$?
      [ "$rc2" -eq 0 ] && { rc=0; mv "$log.retry" "$log"; echo "needed /vendor/lib64 on LD_LIBRARY_PATH" > "$od/note"; }
    fi
    echo "$rc" > "$od/exit_code"
    # per-run time: the "<N> inference took X ms, avg Y ms" line, not the "Total" line (20-run sum)
    avg="$(grep 'inference took' "$log" | grep -v Total | grep -o 'avg [0-9.]* ms' | head -n1)"
    say "$name (as $soc, htp $arch): rc=$rc ${avg:-}  :: $(grep -i -E 'error|fail|unsupported|not supported' "$log" | head -n 2 | tr '\n' ' ')"
  done < "$MANIFEST"
else
  say "skipped (qnn_executor_runner or manifest missing in $NPU_DIR)"
fi

if command -v python >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/npu_compare.py" ]; then
  say ""; say "== Output check vs CPU reference"
  python "$SCRIPT_DIR/npu_compare.py" "$OUT" "$MODELS" | tee -a "$REPORT"
fi
say ""; say "Report: $REPORT"
