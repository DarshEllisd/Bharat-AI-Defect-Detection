#!/usr/bin/env bash
# Pre-flight for build_qnn.sh: does the local ExecuTorch source match what the scripts expect?
ET="$HOME/bharatai-build/executorch"
Q="$ET/backends/qualcomm"
echo "== QNN SDK version mentioned in v1.5.0 docs/scripts"
grep -rhoE "2\.[0-9]{2}\.[0-9]\.[0-9]{6}" "$ET/docs/source/backends-qualcomm.md" "$Q/scripts" 2>/dev/null | sort | uniq -c | sort -rn | head -5
echo "== build.sh options"
grep -oE -- "--[a-z_]+" "$Q/scripts/build.sh" | sort -u | tr '\n' ' '; echo
echo "== functions used by export_qnn_probe.py"
for f in generate_htp_compiler_spec generate_qnn_executorch_compiler_spec to_edge_transform_and_lower_to_qnn; do
  printf "%-42s %s\n" "$f" "$(grep -c "def $f" "$Q/utils/utils.py")"
done
echo "== QnnQuantizer __init__ signature"
grep -A4 "def __init__" "$Q/quantizer/quantizer.py" | head -6
echo "== chipsets (SM7xxx / SM85xx) in this version"
grep -oE "SM7[0-9]{3}|SM8[0-9]{3}" "$Q/serialization/qc_schema.py" | sort -u | tr '\n' ' '; echo
echo "== torchao pt2e import path"
"$HOME/bharatai-build/et-venv/bin/python" -c "from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e; print('ok')" 2>&1 | tail -1
