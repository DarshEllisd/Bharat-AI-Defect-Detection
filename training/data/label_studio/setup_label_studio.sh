#!/usr/bin/env bash
# W2 task D - install Label Studio locally in WSL (its own venv) and create local-only credentials.
# Nothing is uploaded anywhere; the server listens on 127.0.0.1 only.
#
#   bash training/data/label_studio/setup_label_studio.sh
#   bash training/data/label_studio/start_label_studio.sh     # then open http://localhost:8080
set -euo pipefail

WORK="${WORK:-$HOME/bharatai-build}"
VENV="$WORK/ls-venv"
LS="$WORK/label-studio"
export PATH="$HOME/.local/bin:$PATH"

# Recreate if missing or half-created (an interrupted `uv venv` leaves a dir without pip)
if [ ! -x "$VENV/bin/pip" ]; then
  rm -rf "$VENV"
  uv venv --seed --python 3.12 "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet label-studio
# (not `label-studio --version`: it hung during initialisation, P-36)
"$VENV/bin/pip" show label-studio | grep -E '^(Name|Version)'

mkdir -p "$LS/data" "$WORK/data/own/raw"
# Local-only credentials, outside the repo (never commit or paste these anywhere)
CRED="$LS/credentials.env"
if [ ! -f "$CRED" ]; then
  umask 077
  {
    echo "LS_USER=labeler@local.test"
    echo "LS_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')"
    echo "LS_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(20))')"
  } > "$CRED"
fi
echo "Credentials (local only): $CRED"
