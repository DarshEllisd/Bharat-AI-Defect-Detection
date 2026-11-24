#!/data/data/com.termux/files/usr/bin/sh
# One-tap start (week 6): start the inspector in the background if it isn't running, wait until it answers,
# then open the operator page in Chrome. Must run inside the Termux app (the NPU is only reachable from there).
#   sh ~/bharatai/edge/scripts/start_inspector.sh          (also run automatically by autostart.sh)
LOG=~/bharatai/results/server.log
mkdir -p ~/bharatai/results
if pgrep -f "[p]ython -m app.main" >/dev/null 2>&1; then
  echo "inspector already running"
else
  echo "starting inspector (models load in ~5 s)..."
  rm -f ~/bharatai/run/vision.sock
  cd ~/bharatai/edge && nohup python -m app.main > "$LOG" 2>&1 &
  i=0
  while [ $i -lt 60 ]; do
    grep -q "inspector: http" "$LOG" 2>/dev/null && break
    grep -q -E "Traceback|Error" "$LOG" 2>/dev/null && { echo "inspector failed to start:"; tail -n 5 "$LOG"; exit 1; }
    sleep 0.5; i=$((i + 1))
  done
  tail -n 1 "$LOG"
fi
termux-open-url http://localhost:8000 2>/dev/null || echo "open http://localhost:8000 in Chrome"
