#!/data/data/com.termux/files/usr/bin/bash
# W1 spike - Termux setup. Installs packages and tests which web stack installs cleanly.
# Run once:  bash ~/bharatai/spike/00_setup_termux.sh
# Writes:    ~/bharatai/results/setup_log.txt
set -uo pipefail

B="$HOME/bharatai"
LOG="$B/results/setup_log.txt"
mkdir -p "$B/results" "$B/bin" "$B/models"
: > "$LOG"

log() { echo "$*" | tee -a "$LOG"; }
try() {  # try <label> <command...> - records ok/FAILED without aborting the script
  local label="$1"; shift
  if "$@" >>"$LOG.verbose" 2>&1; then log "[ok]     $label"; else log "[FAILED] $label"; fi
}

termux-wake-lock 2>/dev/null || true
log "== Termux setup $(date -Iseconds)"

try "pkg update/upgrade" bash -c "yes | pkg update && yes | pkg upgrade"
try "core packages (python git clang time procps util-linux proot-distro)" \
  pkg install -y python git clang time procps util-linux proot-distro
# Prebuilt Termux packages are far faster than building wheels from source
for p in python-numpy python-pillow python-scipy; do try "pkg $p" pkg install -y "$p"; done
# OpenCV's Python bindings live in the x11 repo, not termux-main. It pulls in Qt, whose
# libQt6DBus needs dbus (not declared as a dependency as of 2026-11-01).
try "pkg x11-repo" pkg install -y x11-repo
try "pkg dbus" pkg install -y dbus
try "pkg python-opencv-python (x11-repo)" pkg install -y python-opencv-python
# Finish any package left half-configured by an interrupted or failed postinst
try "dpkg --configure -a" dpkg --configure -a

log "== Web stack: aiohttp"
# FastAPI was tested on 2026-11-01: pydantic-core needs a Rust build that pip can't bootstrap on
# Android, so the project uses aiohttp. Set WITH_FASTAPI=1 to try it anyway via Termux's rust package.
try "pip aiohttp" pip install aiohttp
if [ "${WITH_FASTAPI:-0}" = "1" ]; then
  try "pkg rust" pkg install -y rust
  try "pip fastapi + uvicorn + websockets" pip install fastapi uvicorn websockets
fi

log "== Import check"
for m in numpy PIL scipy cv2 aiohttp; do
  if python -c "import $m" 2>/dev/null; then
    log "[ok]     import $m ($(python -c "import $m; print(getattr($m, '__version__', '?'))" 2>/dev/null))"
  else
    log "[FAILED] import $m"
  fi
done
log "python: $(python --version 2>&1)"
log "Details of any failure: $LOG.verbose"
