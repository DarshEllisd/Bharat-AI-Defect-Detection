#!/usr/bin/env bash
# Does connecting to a CLOSED localhost port get refused at once (normal) or hang (WSL mirrored quirk)?
for p in "$@"; do
  s=$(date +%s%N)
  curl -s -m 5 -o /dev/null "http://127.0.0.1:$p"; rc=$?
  e=$(date +%s%N)
  echo "127.0.0.1:$p -> curl rc=$rc after $(( (e - s) / 1000000 )) ms   (7 = refused quickly, 28 = timed out)"
done
echo "tcp_syn_retries=$(cat /proc/sys/net/ipv4/tcp_syn_retries)"
