#!/usr/bin/env bash
# W2 - download the 9 MVTec AD categories this project uses (official per-category archives).
#
# MVTec AD is licensed CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike):
# https://www.mvtec.com/company/research/datasets/mvtec-ad
#
#   bash training/data/download_mvtec.sh              # all 9 categories
#   bash training/data/download_mvtec.sh wood carpet  # a subset
#
# Output: $DATA/mvtec/<category>/{train,test,ground_truth}, plus
#         $DATA/mvtec/mvtec_manifest.json (archive name, bytes, sha256) and a copy in the repo.
set -euo pipefail

DATA="${DATA:-$HOME/bharatai-build/data}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DL="$DATA/downloads"
OUT="$DATA/mvtec"
mkdir -p "$DL" "$OUT"

# Official links from https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads (Nov 2026)
declare -A URL=(
  [cable]="https://www.mydrive.ch/shares/163641/97fd0295152453e533cfb3db1d253a35/download/420937413-1629958794/cable.tar.xz"
  [carpet]="https://www.mydrive.ch/shares/163643/70e8bebbe9f2d2c55043311a4e67eb1d/download/420937484-1629959013/carpet.tar.xz"
  [grid]="https://www.mydrive.ch/shares/163644/8c7fc7cba43ef5275c87ca750a6d0004/download/420937487-1629959044/grid.tar.xz"
  [leather]="https://www.mydrive.ch/shares/163646/7e0627b5718d2902bb85d24cff7a51ce/download/420937607-1629959262/leather.tar.xz"
  [metal_nut]="https://www.mydrive.ch/shares/163647/6d5a833cdee9035a45d5ce0921ecd24d/download/420937637-1629959294/metal_nut.tar.xz"
  [screw]="https://www.mydrive.ch/shares/163649/cfd394b4e3a49013745fab9e2084988a/download/420938130-1629960389/screw.tar.xz"
  [tile]="https://www.mydrive.ch/shares/163650/55ad2038770deff6fe6771bd784af8cc/download/420938133-1629960456/tile.tar.xz"
  [transistor]="https://www.mydrive.ch/shares/163652/619a88c3d5f3f57b46b00f44f04ad6e6/download/420938166-1629960554/transistor.tar.xz"
  [wood]="https://www.mydrive.ch/shares/163653/937e0d65bc9c4497d253cb7313868a60/download/420938383-1629960649/wood.tar.xz"
)
CATS=("$@")
[ ${#CATS[@]} -gt 0 ] || CATS=(cable carpet grid leather metal_nut screw tile transistor wood)

# 1. Check every link before downloading anything (a typo would otherwise surface 20 min in)
for c in "${CATS[@]}"; do
  [ -n "${URL[$c]:-}" ] || { echo "Unknown category: $c" >&2; exit 2; }
  code="$(curl -s -o /dev/null -r 0-0 -L -m 20 -w '%{http_code}' "${URL[$c]}")"
  [ "$code" = "206" ] || [ "$code" = "200" ] || { echo "Link check failed for $c (HTTP $code)" >&2; exit 1; }
done
echo "All ${#CATS[@]} links OK"

# 2. Download (resumable), verify, record checksum, extract atomically, then drop the archive.
#    A category is done only when $OUT/<cat>/.complete exists, so a crash mid-way is redone cleanly
#    (P-31: a full host disk once killed a download half-way through).
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"
for c in "${CATS[@]}"; do
  echo "==> $c"
  if [ -f "$OUT/$c/.complete" ]; then echo "already extracted"; continue; fi
  f="$DL/$c.tar.xz"
  if [ -f "$f" ] && xz -t "$f" 2>/dev/null; then
    echo "archive already downloaded and valid"
  else
    curl -fL --retry 5 --retry-delay 3 -C - -o "$f" "${URL[$c]}" || true   # verified just below
  fi
  if ! xz -t "$f" 2>/dev/null; then
    echo "archive corrupt - downloading $c again"
    rm -f "$f"
    curl -fL --retry 5 --retry-delay 3 -o "$f" "${URL[$c]}"
    xz -t "$f"
  fi
  # MVTec publishes no per-archive hashes, so record ours (before the archive is deleted)
  python3 - "$f" "$OUT/mvtec_manifest.json" "$c" <<'EOF'
import hashlib, json, sys
from pathlib import Path
p, out, c = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
h = hashlib.sha256()
with p.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
m = json.loads(out.read_text()) if out.exists() else {}
m[c] = {"archive": p.name, "bytes": p.stat().st_size, "sha256": h.hexdigest()}
out.write_text(json.dumps(m, indent=2, sort_keys=True))
print(f"{c:12s} {p.stat().st_size / 1e6:8.1f} MB  sha256 {h.hexdigest()[:16]}…")
EOF
  tmp="$OUT/.extract_$c"
  # MVTec's archives store directories as read-only (dr-xr-x---); make trees writable before
  # deleting a half-extracted copy, and right after extracting, so reruns and cleanup work
  for d in "$tmp" "$OUT/$c"; do [ -e "$d" ] && chmod -R u+w "$d"; done
  rm -rf "$tmp" "$OUT/$c"
  mkdir -p "$tmp"
  tar -xJf "$f" -C "$tmp"
  chmod -R u+w "$tmp"
  mv "$tmp/$c" "$OUT/$c"
  rm -rf "$tmp"
  touch "$OUT/$c/.complete"
  [ "$KEEP_ARCHIVES" = "1" ] || rm -f "$f"
done
mkdir -p "$REPO_ROOT/training/data/reports"
cp "$OUT/mvtec_manifest.json" "$REPO_ROOT/training/data/reports/mvtec_manifest.json"
echo "Done: $OUT"
