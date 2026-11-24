#!/usr/bin/env bash
# Llama download: 20 s throughput sample + which remote IPs (v4/v6) the hf process uses.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/spike/artifacts/models/llama"
size() { find "$OUT" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
a=$(size); timeout 20 tail -f /dev/null; b=$(size)
echo "20 s average: $(( (b - a) / 20 / 1024 )) KB/s   total: $(( b / 1024 / 1024 )) MB"
pid="$(pgrep -f 'hf download' | head -n1)"
echo "== hf process $pid connections"
ss -tnp 2>/dev/null | grep "pid=$pid," | awk '{print $5}' | sort | uniq -c
echo "== xet / CDN hosts"
for h in huggingface.co cas-bridge.xethub.hf.co cdn-lfs.hf.co cdn-lfs-us-1.hf.co; do
  echo "$h -> $(getent ahosts "$h" | awk '{print $1}' | sort -u | tr '\n' ' ')"
done
"$HOME/bharatai-build/et-venv/bin/pip" list 2>/dev/null | grep -i -E '^(hf.xet|huggingface.hub) '
