#!/data/data/com.termux/files/usr/bin/bash
# W1 spike - route A: can the ExecuTorch *Python* runtime run on the phone?
# Termux uses Android's bionic libc, so manylinux wheels can't load there. A Debian
# userland under proot-distro provides glibc, where the PyPI aarch64 wheel should work.
#
#   bash ~/bharatai/spike/05_route_a_proot.sh
#   ET_PIP_VERSION=1.5.0 bash ~/bharatai/spike/05_route_a_proot.sh
# Writes: ~/bharatai/results/route_a.json and route_a.log
set -uo pipefail

B="$HOME/bharatai"
ET_PIP_VERSION="${ET_PIP_VERSION:-1.5.0}"   # keep equal to the laptop ExecuTorch tag (v1.5.0)
LOG="$B/results/route_a.log"
mkdir -p "$B/results"
termux-wake-lock 2>/dev/null || true

command -v proot-distro >/dev/null || pkg install -y proot-distro
# proot-distro 5.x (Python rewrite) pulls the official Debian image from Docker Hub
if ! proot-distro login debian -- true 2>/dev/null; then
  echo "Installing Debian container from Docker Hub (one-time)..."
  proot-distro install debian
fi

# ~/bharatai is bind-mounted at /mnt/bharatai inside Debian (options go before the container name)
proot-distro login --bind "$B:/mnt/bharatai" debian -- \
  bash /mnt/bharatai/spike/route_a_inner.sh "$ET_PIP_VERSION" 2>&1 | tee "$LOG"

echo "Results: $B/results/route_a.json"
