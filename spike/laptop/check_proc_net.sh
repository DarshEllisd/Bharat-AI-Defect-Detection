#!/usr/bin/env bash
# Which remote hosts is a process connected to (and are they IPv4 or IPv6)?  usage: check_proc_net.sh <pattern>
pat="${1:?process name pattern}"
for pid in $(pgrep -f "$pat"); do
  echo "== pid $pid: $(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-100)"
  ss -tnp 2>/dev/null | awk -v p="pid=$pid," 'index($0, p) {print $1, $4, "->", $5}'
done
echo "== reverse lookup of all established remote addresses"
ss -tn state established 2>/dev/null | awk 'NR>1 {print $4}' | sed -E 's/:[0-9]+$//; s/^\[//; s/\]$//' | sort -u |
  while read -r ip; do echo "$ip  $(getent hosts "$ip" | awk '{print $2}')"; done
