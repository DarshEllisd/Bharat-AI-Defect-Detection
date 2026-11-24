#!/usr/bin/env bash
# Runs INSIDE proot Debian (called by 05_route_a_proot.sh). Arg 1: executorch pip version.
# Termux's ~/bharatai is bind-mounted at /mnt/bharatai.
set -euo pipefail
ET_PIP_VERSION="$1"
# Inside the container's own filesystem: proot-distro shares Termux's home by default,
# and a venv there would clutter it
VENV=/opt/et-venv

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip >/dev/null
python3 --version

[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
# The executorch wheel does not depend on torch, but its Python runtime takes torch tensors.
# CPU-only index: keeps GPU libraries out; version matches the laptop's ExecuTorch 1.5.0 build.
TORCH_VERSION="${TORCH_VERSION:-2.14.0}"
"$VENV/bin/pip" install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install "executorch==$ET_PIP_VERSION"

"$VENV/bin/python" /mnt/bharatai/spike/bench_pte.py \
  --models /mnt/bharatai/models/vision \
  --out /mnt/bharatai/results/route_a.json
