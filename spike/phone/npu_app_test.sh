#!/data/data/com.termux/files/usr/bin/sh
# NPU test from INSIDE the Termux app (SELinux domain untrusted_app), typed into Termux.
# 1. Full probe with the bundle's own libraries (as in the adb run)
# 2. The SSG2115P model again with copies of the vendor FastRPC libs (~/bharatai/npu/vendorlib),
#    since apps can't load /vendor/lib64/libcdsprpc.so without a manifest declaration
N="$HOME/bharatai/npu"
OUT="$HOME/bharatai/results/npu_termux_app"
rm -rf "$OUT"; mkdir -p "$OUT"
unset LD_PRELOAD
{
  echo "context: $(id -Z 2>/dev/null)"
  sh "$HOME/bharatai/spike/07_npu_probe.sh" "$N" "$OUT/probe"
  echo "== SSG2115P with vendorlib copies"
  O="$OUT/vendorlib"; mkdir -p "$O"
  echo "$N/models/probe.input.raw" > "$O/in.txt"
  LD_LIBRARY_PATH="$N/lib:$N/vendorlib" ADSP_LIBRARY_PATH="$N/dsp/v73;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/dsp" \
    "$N/bin/qnn_executor_runner" --model_path "$N/models/probe_int8_SSG2115P.pte" \
    --input_list_path "$O/in.txt" --output_folder_path "$O" --iteration 20 --warm_up 3 \
    < /dev/null > "$O/log.txt" 2>&1
  echo "rc=$?"
  grep -i -E "error|avg" "$O/log.txt" | head -5
} > "$OUT/report.txt" 2>&1
echo "NPU_APP_TEST_DONE" >> "$OUT/report.txt"
echo "done - results in $OUT/report.txt"
