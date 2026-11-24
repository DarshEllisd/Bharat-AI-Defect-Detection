#!/usr/bin/env bash
# "Teach normal" retrain (P-69): GOOD photos taken on the user's stand with the app's "Teach good" button ->
# new anomaly model for that setup -> NPU export -> deployed to the phone.
#   ANDROID_SERIAL=<ip:port> bash training/vision/runs/teach_retrain.sh fabric [iters]     (WSL, WSLENV=ANDROID_SERIAL)
# Steps: pull ~/bharatai/teach/<product> -> split by 5-photo groups (every 5th group = val) -> train
# (fabric: 2x2 tiles) -> evaluate (val good of the new setup + the old own test defects, for reference) ->
# 16a8w QNN export with synthetic-defect calibration -> push .pte -> point categories.toml at it.
set -euo pipefail
P=${1:?product (wood|fabric)}; ITERS=${2:-20000}
REPO=/mnt/c/Users/Darsh/Desktop/BharatAI-SOC
B=$HOME/bharatai-build; PY=$B/train-venv/bin/python
NAME="${P}_teach"; [ "$P" = fabric ] && TILES=2 || TILES=1
export PYTHONUNBUFFERED=1
cd "$REPO"

echo "==> [1/6] pull taught photos"
bash spike/laptop/termux_pull.sh "~/bharatai/teach/$P" "$B/data/teach/$P"

# newest teach session only (earlier sessions may be at another distance/light); ALL_SESSIONS=1 uses all
if [ "${ALL_SESSIONS:-0}" = 1 ]; then SRC="$B/data/teach/$P"; GLOB='*/*.jpg'; else
  SRC="$B/data/teach/$P/$(ls -1 "$B/data/teach/$P" | grep -v '\.txt$' | sort | tail -n1)"; GLOB='*.jpg'; fi
echo "teach photos from: $SRC"
echo "==> [2/6] set the working distance from the taught photos, frame check + split"
# accepted texture-scale range = median +/- 35% of the taught photos (fabric only; P-70)
$PY - "$P" "$SRC" "$GLOB" <<'PY'
import glob, re, sys, tomllib, cv2, numpy as np
sys.path.insert(0, ".")
from edge.app.pipeline.quality import texture_period
p, d = sys.argv[1], sys.argv[2]
path = "edge/config/categories.toml"; s = open(path, encoding="utf-8").read()
if tomllib.loads(s)["products"][p].get("scale_px"):
    per = [texture_period(cv2.imread(f)) for f in sorted(glob.glob(f"{d}/{sys.argv[3]}"))]
    med = float(np.nanmedian(per)); lo, hi = int(med * 0.65), int(np.ceil(med * 1.35))
    lines, inside = s.split(chr(10)), False
    for i, line in enumerate(lines):          # rewrite scale_px inside [products.<p>] only
        if line.startswith("["):
            inside = line.strip() == f"[products.{p}]"
        elif inside and line.startswith("scale_px"):
            lines[i] = f"scale_px = [{lo}, {hi}]   # from taught photos: median {med:.1f} px +/- 35% (P-70)"
    open(path, "w", encoding="utf-8", newline="").write(chr(10).join(lines))
    print(f"scale_px for {p}: [{lo}, {hi}] (taught median {med:.1f} px)")
PY
D=$B/datasets/teach/$P; rm -rf "$D"; mkdir -p "$D/train/good" "$D/val/good"
# keep only photos the app itself would accept (same frame check, P-68), so training matches inference
$PY - "$P" "$SRC" "$GLOB" > "$B/data/teach/${P}_ok.txt" <<'PY'
import glob, sys, tomllib, cv2
sys.path.insert(0, ".")
from edge.app.pipeline.quality import check
p, d = sys.argv[1], sys.argv[2]
cfg = tomllib.load(open("edge/config/categories.toml", "rb"))["products"][p]
fs = sorted(glob.glob(f"{d}/{sys.argv[3]}"))
ok = [f for f in fs if check(cv2.imread(f), cfg)["ok"]]
for f in ok:
    print(f)
print(f"frame check: kept {len(ok)}/{len(fs)}", file=sys.stderr)
PY
i=0; cat "$B/data/teach/${P}_ok.txt" | while read -r f; do
  g=$(( i / 5 )); s=train; [ $(( g % 5 )) -eq 4 ] && s=val
  ln -s "$f" "$D/$s/good/$(basename "$(dirname "$f")")_$(basename "$f")"; i=$((i + 1))
done
echo "train $(ls "$D/train/good" | wc -l) / val $(ls "$D/val/good" | wc -l)"

echo "==> [3/6] train $NAME ($ITERS iterations, tiles $TILES)"
rm -rf "$B/models/efficientad/$NAME"
$PY training/vision/train_efficientad.py --name "$NAME" --tiles "$TILES" --iters "$ITERS" --geom-aug \
  --train "$D/train/good" --val "$D/val/good" --penalty-dir "$B/data/imagenette_flat" | grep -E "\] it|saved"

echo "==> [4/6] evaluate (val = new setup; test = old own photos, different conditions: reference only)"
A=$B/datasets/anomaly_own/$P
$PY training/vision/evaluate_efficientad.py --name "$NAME" --val "$D/val/good" --test "$A/test" --boxes "$A/test_boxes.json" \
  | grep -E "auroc|false_alarm|\"recall\"|threshold\"" || true

echo "==> [5/6] NPU export (16a8w, synthetic-defect calibration)"
bash training/vision/runs/export_npu.sh efficientad "$NAME" --quant 16a8w --skip-eval 2>&1 | grep -E "\"qnn\"|delegat|xnnpack_fp32" || true
PTE="efficientad_${NAME}_16a8w_SSG2115P.pte"
[ -f "$B/models/npu/${NAME}_16a8w/$PTE" ] || { echo "export failed"; exit 1; }

echo "==> [6/6] deploy"
S=$B/stage_teach_$P; rm -rf "$S"; mkdir -p "$S"; cp "$B/models/npu/${NAME}_16a8w/$PTE" "$S/"
bash spike/laptop/termux_sync.sh "$S" "~/bharatai/models_w3"
$B/train-venv/bin/python - "$P" "$PTE" <<'PY'
import re, sys
p, pte = sys.argv[1], sys.argv[2]
path = "edge/config/categories.toml"; s = open(path, encoding="utf-8").read()
block = re.search(rf"\[products\.{p}\]\n(.*?)(?=\n\[|\Z)", s, re.S)
new = re.sub(r'anomaly_pte = "[^"]*"', f'anomaly_pte = "{pte}"   # taught on the stand (P-69)', block.group(1), count=1)
s = s[:block.start(1)] + new + s[block.end(1):]
open(path, "w", encoding="utf-8", newline="\n").write(s)
print("categories.toml:", p, "->", pte)
PY
echo "DONE: restart the inspector on the phone, then Calibrate '$P' and inspect."
