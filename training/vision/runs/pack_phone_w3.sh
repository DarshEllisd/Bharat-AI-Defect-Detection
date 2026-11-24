#!/usr/bin/env bash
# W3-E: collect the NPU models + check inputs/reference outputs into one folder with one manifest for
# spike/phone/07_npu_probe.sh (MODELS_DIR=...). Files are prefixed with the model name (no clashes).
#   bash training/vision/runs/pack_phone_w3.sh   -> ~/bharatai-build/models/npu/phone_w3/
set -euo pipefail
N=$HOME/bharatai-build/models/npu; P=$N/phone_w3
rm -rf "$P"; mkdir -p "$P"
: > "$P/manifest.txt"
for d in "$N"/own_wood_16a8w "$N"/own_fabric_t2_16a8w "$N"/mvtec_wood_16a8w "$N"/mvtec_carpet_16a8w; do   # YOLO runs fp32 on the CPU (XNNPACK); its int8 class scores collapse (P-62)
  [ -f "$d/manifest.txt" ] || { echo "skip $(basename "$d") (no manifest)"; continue; }
  m=$(basename "$d")
  while read -r name pte input ref arch soc variant; do
    [ -n "$name" ] || continue
    [ -f "$P/$pte" ] || cp "$d/$pte" "$P/$pte"
    cp "$d/$input" "$P/${m}__$input"; cp "$d/$ref" "$P/${m}__$ref"
    fp=${ref%.ref.raw}.fp32.raw; [ -f "$d/$fp" ] && cp "$d/$fp" "$P/${m}__$fp"
    echo "$name $pte ${m}__$input ${m}__$ref $arch $soc $variant" >> "$P/manifest.txt"
  done < "$d/manifest.txt"
done
cat "$P/manifest.txt"; du -sh "$P"
