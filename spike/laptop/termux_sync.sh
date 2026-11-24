#!/usr/bin/env bash
# Make a Termux directory match a laptop directory: every file present with the same MD5. Only missing or
# changed files are (re)sent, one by one via termux_push.sh, with retries. Safe to re-run.
# (termux_push.sh alone could stop half-way over flaky wireless adb - P-66. Comparing sizes only missed a
# retrained model of identical size - P-71.)
#
#   bash spike/laptop/termux_sync.sh <local dir> '~/bharatai/<remote dir>'
set -uo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <local dir> <termux dir>" >&2; exit 2; }
LOCAL="${1%/}"; REMOTE_REL="${2#\~/}"; REMOTE_REL="${REMOTE_REL%/}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADB=adb; command -v adb >/dev/null 2>&1 || ADB=adb.exe
export MSYS_NO_PATHCONV=1
TH=/data/data/com.termux/files/home

listing() {  # "<md5> <relative path>" of the remote dir, with an end marker (P-40)
  "$ADB" exec-out "run-as com.termux sh -c 'cd $TH/$REMOTE_REL 2>/dev/null && find . -type f -exec md5sum {} +; echo __END__'" | tr -d '\r'
}

for round in 1 2 3 4 5; do
  remote="$(listing)"
  echo "$remote" | grep -q __END__ || { echo "remote listing failed (round $round)"; sleep 3; continue; }
  todo=()
  while IFS= read -r -d '' f; do
    rel="${f#"$LOCAL"/}"; sum=$(md5sum < "$f" | cut -d' ' -f1)
    rs=$(echo "$remote" | awk -v p="./$rel" '$2==p{print $1}')
    [ "$rs" = "$sum" ] || todo+=("$rel")
  done < <(find "$LOCAL" -type f -print0)
  echo "round $round: ${#todo[@]} file(s) to send"
  [ ${#todo[@]} -eq 0 ] && { echo "SYNC OK: $(find "$LOCAL" -type f | wc -l) files match in ~/$REMOTE_REL"; exit 0; }
  for rel in "${todo[@]}"; do
    d="$(dirname "$rel")"; [ "$d" = "." ] && d="" || d="/$d"
    bash "$HERE/termux_push.sh" "$LOCAL/$rel" "~/$REMOTE_REL$d/" >/dev/null 2>&1 || true
  done
done
echo "SYNC INCOMPLETE after 5 rounds" >&2
exit 1
