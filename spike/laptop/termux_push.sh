#!/usr/bin/env bash
# Copy laptop files straight into Termux's home over USB (no shared-storage step).
#
#   bash spike/laptop/termux_push.sh spike/phone ~/bharatai/spike      # directory contents
#   bash spike/laptop/termux_push.sh some/file.sh ~/bharatai/spike/    # single file
#
# Destination may use ~ for Termux's home (quote it in Git Bash: '~/bharatai/...').
# Small files stream through `adb exec-in`. Files over 32 MB go via `adb push` to
# /data/local/tmp and `run-as cp` (streaming ~1 GB crashed the adb server at ~150-230 MB).
# Every copy is size-checked on the phone.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <local file|dir> <termux dest>" >&2; exit 2; }
SRC="$1"
DEST="${2/#\~//data/data/com.termux/files/home}"

if command -v adb >/dev/null 2>&1; then ADB=adb
elif command -v adb.exe >/dev/null 2>&1; then ADB=adb.exe
elif [ -x /c/Android/platform-tools/adb.exe ]; then ADB=/c/Android/platform-tools/adb.exe
else echo "adb not found" >&2; exit 1
fi
export MSYS_NO_PATHCONV=1

LARGE=$((32 * 1024 * 1024))

push_one() {  # push_one <local file> <remote file>
  local dir; dir="$(dirname "$2")"
  if [ "$(wc -c < "$1")" -gt "$LARGE" ]; then
    local tmp="/data/local/tmp/termux_push.$$.$(basename "$2")"
    "$ADB" push "$1" "$tmp" < /dev/null | tr -d '\r' | tail -n 1
    "$ADB" shell "run-as com.termux mkdir -p $dir && run-as com.termux cp $tmp $2; rm -f $tmp" < /dev/null
  else
    # exec-in gives a binary-safe stdin (adb shell -T truncated binaries to ~1 KB)
    local remote="run-as com.termux sh -c 'mkdir -p \"$dir\" && cat > \"$2\"'"
    # Text files: strip Windows CRs here, on the laptop, so nothing gets mangled by remote quoting
    case "$1" in
      *.sh|*.py|*.txt|*.html|*.json|*.yaml) tr -d '\r' < "$1" | "$ADB" exec-in "$remote" ;;
      *) "$ADB" exec-in "$remote" < "$1" ;;
    esac
  fi
  # Verify the copy: byte count on the phone must match the local file (text: after CR stripping)
  local want got
  case "$1" in
    *.sh|*.py|*.txt|*.html|*.json|*.yaml) want=$(tr -d '\r' < "$1" | wc -c) ;;
    *) want=$(wc -c < "$1") ;;
  esac
  got=$("$ADB" exec-out "run-as com.termux wc -c $2" < /dev/null | tr -d '\r' | awk '{print $1}')
  if [ "$want" -eq "$got" ] 2>/dev/null; then echo "  $1 -> $2 ($got bytes)"
  else echo "  FAILED $1 -> $2 (expected $want bytes, got $got)" >&2; return 1
  fi
}

if [ -d "$SRC" ]; then
  (cd "$SRC" && find . -type f ! -path '*/__pycache__/*') | while read -r rel; do
    push_one "$SRC/${rel#./}" "$DEST/${rel#./}"
  done
else
  case "$DEST" in */) DEST="$DEST$(basename "$SRC")" ;; esac
  push_one "$SRC" "$DEST"
fi
