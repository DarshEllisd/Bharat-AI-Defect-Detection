#!/usr/bin/env bash
# W2 - copy captured photos from the phone to the laptop and verify every file.
#
#   bash training/data/pull_captures.sh           # copy + verify (phone copies kept)
#   bash training/data/pull_captures.sh --clear   # copy + verify, then delete them from the phone
#
# Phone: ~/bharatai/data/raw/<product>/<label>/*.{jpg,json}  (Termux, read via run-as)
# Laptop: $DATA/own/raw/... (default ~/bharatai-build/data/own/raw)
# Works from WSL or Git Bash. Files are copied one by one over `adb exec-out` (binary-safe) and
# size-checked; re-running resumes an interrupted pull.
set -euo pipefail

CLEAR=0; [ "${1:-}" = "--clear" ] && CLEAR=1
DATA="${DATA:-$HOME/bharatai-build/data}"
DEST="$DATA/own"
REMOTE=/data/data/com.termux/files/home/bharatai/data
SUBDIR="${SUBDIR:-raw}"   # override only for testing
export MSYS_NO_PATHCONV=1   # keep phone paths intact under Git Bash

if command -v adb >/dev/null 2>&1; then ADB=adb
elif command -v adb.exe >/dev/null 2>&1; then ADB=adb.exe
elif [ -x /mnt/c/Android/platform-tools/adb.exe ]; then ADB=/mnt/c/Android/platform-tools/adb.exe
elif [ -x /c/Android/platform-tools/adb.exe ]; then ADB=/c/Android/platform-tools/adb.exe
else echo "adb not found" >&2; exit 1
fi

"$ADB" get-state >/dev/null
listing() {  # "<bytes> <relative path>" for every file, sorted
  sed 's/\r$//' | awk '$2 != "total" {print $1, $2}' | sort   # plain sort: comm needs it
}
# adb exec-out exits 0 even when the phone-side command fails or the device drops, so an empty
# listing is ambiguous (P-40). The phone side prints an end marker; without it, the read failed.
raw_listing="$("$ADB" exec-out "run-as com.termux sh -c 'if cd $REMOTE 2>/dev/null && [ -d $SUBDIR ]; then find $SUBDIR -type f -exec wc -c {} +; else echo __NO_CAPTURES__; fi; echo __LISTING_OK__'" 2>&1 || true)"
if ! printf '%s\n' "$raw_listing" | grep -q '__LISTING_OK__'; then
  echo "ERROR: could not read the phone's capture folder (device dropped or adb failed):" >&2
  printf '%s\n' "$raw_listing" | tail -n 3 >&2
  exit 1
fi
if printf '%s\n' "$raw_listing" | grep -q '__NO_CAPTURES__'; then echo "No captures on the phone yet."; exit 0; fi
remote_list="$(printf '%s\n' "$raw_listing" | grep -v '__LISTING_OK__' | listing)"
n_remote=$(printf '%s\n' "$remote_list" | grep -c . || true)
[ "$n_remote" -gt 0 ] || { echo "No captures on the phone yet."; exit 0; }
echo "Phone: $n_remote files"

mkdir -p "$DEST"
# One file at a time (a single 530 MB tar stream over adb was cut off mid-way, P-38).
# Files already present with the right size are skipped, so an interrupted pull just resumes.
copied=0
while read -r size rel; do
  [ -n "$rel" ] || continue
  dst="$DEST/$rel"
  if [ -f "$dst" ] && [ "$(wc -c < "$dst")" -eq "$size" ]; then continue; fi
  # already curated out (curate_own.py moves files to raw/_excluded/...): present, don't re-copy
  exl="$DEST/$SUBDIR/_excluded/${rel#"$SUBDIR"/}"
  if [ -f "$exl" ] && [ "$(wc -c < "$exl")" -eq "$size" ]; then continue; fi
  mkdir -p "$(dirname "$dst")"
  "$ADB" exec-out "run-as com.termux cat $REMOTE/$rel" > "$dst.part" < /dev/null
  if [ "$(wc -c < "$dst.part")" -eq "$size" ]; then
    mv "$dst.part" "$dst"; copied=$((copied + 1))
  else
    echo "size mismatch on $rel - will be reported below" >&2; rm -f "$dst.part"
  fi
done <<< "$remote_list"
echo "Copied $copied new or changed files"

# Every phone file must exist locally with the same size
# (files in raw/_excluded/ count under their original path - P-53)
local_list="$(cd "$DEST" && find "$SUBDIR" -type f -exec wc -c {} + | sed "s| $SUBDIR/_excluded/| $SUBDIR/|" | listing)"
missing="$(comm -23 <(printf '%s\n' "$remote_list") <(printf '%s\n' "$local_list"))"
if [ -n "$missing" ]; then
  echo "VERIFY FAILED - missing or size mismatch:" >&2
  printf '%s\n' "$missing" | head -n 20 >&2
  exit 1
fi
echo "Verified $n_remote files in $DEST/$SUBDIR"
find "$DEST/$SUBDIR" -path '*/_excluded' -prune -o -name '*.jpg' -print | awk -F/ '{print $(NF-2)"/"$(NF-1)}' | sort | uniq -c

if [ "$CLEAR" = "1" ]; then
  "$ADB" exec-out "run-as com.termux rm -rf $REMOTE/$SUBDIR"
  echo "Cleared captures from the phone."
fi
