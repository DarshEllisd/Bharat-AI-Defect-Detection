#!/usr/bin/env bash
# Is the build downloading? Samples network bytes over 10 s and shows pip's temp/cache sizes.
rx() { awk '/eth0|ens|enp/ {s+=$2} END {print s+0}' /proc/net/dev; }
a=$(rx); timeout 10 tail -f /dev/null; b=$(rx)
echo "download rate: $(( (b - a) / 10 / 1024 )) KB/s"
echo "pip temp dirs:"; du -sh /tmp/pip-* 2>/dev/null | sort -h | tail -n 3
echo "pip cache:";     du -sh ~/.cache/pip 2>/dev/null
echo "venv size:";     du -sh ~/bharatai-build/et-venv 2>/dev/null
echo "pip children:";  pgrep -af pip | cut -c1-160
