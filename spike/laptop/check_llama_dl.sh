#!/usr/bin/env bash
# Status of the Llama download (get_llama_pte.sh): login, files, partial downloads, speed.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/spike/artifacts/models/llama"
VENV="$HOME/bharatai-build/et-venv"

echo "== HF login"
"$VENV/bin/hf" auth whoami 2>&1 | head -n 3
echo "== Output dir"
ls -la "$OUT" 2>&1
echo "== Partial downloads (.incomplete)"
find "$OUT" -name '*.incomplete' -exec ls -la {} \; 2>/dev/null
size() { find "$OUT" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
a=$(size); timeout 5 tail -f /dev/null; b=$(size)
echo "download rate: $(( (b - a) / 5 / 1024 )) KB/s   total so far: $(( b / 1024 / 1024 )) MB of ~1090 MB"
echo "== Processes"
pgrep -af 'hf |huggingface|get_llama' | grep -v pgrep | cut -c1-160
