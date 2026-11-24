#!/usr/bin/env bash
# Diagnose slow WSL networking: DNS, MTU, routes, and real download rate over IPv4 vs IPv6.
URL="https://files.pythonhosted.org/packages/source/n/numpy/numpy-1.26.4.tar.gz"
FMT="connect=%{time_connect}s ip=%{remote_ip} bytes=%{size_download} rate=%{speed_download}B/s\n"
echo "== resolv.conf";   grep -v '^#' /etc/resolv.conf
echo "== interfaces";    ip -brief addr | grep -v '^lo'
echo "== MTU";           ip link | awk '/mtu/ {print $2, $4, $5}'
echo "== default route"; ip route | head -n 3
echo "== DNS lookup";    getent ahosts files.pythonhosted.org | awk '{print $1}' | sort -u | head -n 4
echo "== HTTPS default (10 s max)"; curl -sL -o /dev/null -m 10 -w "$FMT" "$URL"
echo "== HTTPS IPv4 only";          curl -4 -sL -o /dev/null -m 10 -w "$FMT" "$URL"
echo "== HTTPS IPv6 only";          curl -6 -sL -o /dev/null -m 10 -w "$FMT" "$URL"
echo "== .wslconfig"; cat /mnt/c/Users/*/.wslconfig 2>/dev/null
