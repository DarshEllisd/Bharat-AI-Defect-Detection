#!/usr/bin/env bash
# Start the local Label Studio server (WSL, 127.0.0.1:8080). With WSL mirrored networking the
# Windows browser reaches it at http://localhost:8080. Log in with the user/password in
# ~/bharatai-build/label-studio/credentials.env. Stop with Ctrl+C.
set -euo pipefail

WORK="${WORK:-$HOME/bharatai-build}"
LS="$WORK/label-studio"
# shellcheck disable=SC1091
source "$LS/credentials.env"

export LABEL_STUDIO_BASE_DATA_DIR="$LS/data"
# Serve our own photos straight from disk (no upload/copy into Label Studio)
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$WORK/data/own"
export LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true

exec "$WORK/ls-venv/bin/label-studio" start \
  --internal-host 127.0.0.1 --port "${LS_PORT:-8080}" --no-browser \
  --username "$LS_USER" --password "$LS_PASSWORD" --user-token "$LS_TOKEN"
