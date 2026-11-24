#!/usr/bin/env bash
# Start the local Label Studio server in the background if it isn't running, and wait until healthy.
# The pgrep pattern uses "[l]abel" so it can't match this script's own command line (P-43).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HOME/bharatai-build/label-studio/server.log"
if pgrep -f "[l]abel-studio start" >/dev/null; then
  echo "Label Studio already running"
else
  setsid nohup bash "$HERE/start_label_studio.sh" > "$LOG" 2>&1 < /dev/null &
  echo "Label Studio starting (first start can take 2-3 min, P-36)"
fi
for _ in $(seq 1 60); do
  if curl -s -o /dev/null -m 3 http://127.0.0.1:8080/health; then echo "healthy: http://localhost:8080"; exit 0; fi
  sleep 5
done
echo "not healthy after 5 min - see $LOG" >&2
tail -n 5 "$LOG" >&2
exit 1
