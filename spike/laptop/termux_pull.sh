#!/usr/bin/env bash
# Copy a Termux folder to the laptop, file by file, size-verified, resumable (wireless adb safe: P-38/P-66).
#   bash spike/laptop/termux_pull.sh '~/bharatai/teach/fabric' ~/bharatai-build/data/teach/fabric
set -uo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <termux dir> <local dir>" >&2; exit 2; }
REMOTE_REL="${1#\~/}"; REMOTE_REL="${REMOTE_REL%/}"; LOCAL="${2%/}"
ADB=adb; command -v adb >/dev/null 2>&1 || ADB=adb.exe
export MSYS_NO_PATHCONV=1
TH=/data/data/com.termux/files/home
for round in 1 2 3 4 5; do
  listing="$("$ADB" exec-out "run-as com.termux sh -c 'cd $TH/$REMOTE_REL 2>/dev/null && find . -type f -exec wc -c {} + | grep -v \" total\$\"; echo __END__'" | tr -d '\r')"
  echo "$listing" | grep -q __END__ || { echo "remote listing failed (round $round)"; sleep 3; continue; }
  todo=0
  while read -r size rel; do
    [ -n "${rel:-}" ] && [ "$size" != "__END__" ] || continue
    dst="$LOCAL/${rel#./}"
    [ -f "$dst" ] && [ "$(wc -c < "$dst")" = "$size" ] && continue
    todo=$((todo + 1)); mkdir -p "$(dirname "$dst")"
    "$ADB" exec-out "run-as com.termux cat $TH/$REMOTE_REL/${rel#./}" > "$dst.part" < /dev/null
    [ "$(wc -c < "$dst.part")" = "$size" ] && mv "$dst.part" "$dst" || rm -f "$dst.part"
  done <<< "$(echo "$listing" | grep -v __END__)"
  echo "round $round: fetched/retried $todo file(s)"
  [ "$todo" -eq 0 ] && { echo "PULL OK: $(find "$LOCAL" -type f | wc -l) files in $LOCAL"; exit 0; }
done
echo "PULL INCOMPLETE" >&2; exit 1
