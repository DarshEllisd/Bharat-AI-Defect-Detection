#!/usr/bin/env bash
# Download speed from WSL to the hosts the W1 build uses, with the IP family actually used.
# On this connection IPv4 to some CDNs (Fastly, CloudFront) crawls while IPv6 is fast.
FMT="%-28s ip=%-40s bytes=%-10s rate=%s B/s\n"
for u in \
  "https://github.com/termux/termux-app/releases/download/v0.118.3/termux-app_v0.118.3+github-debug_arm64-v8a.apk" \
  "https://dl.google.com/android/repository/platform-tools-latest-linux.zip" \
  "https://files.pythonhosted.org/packages/source/n/numpy/numpy-1.26.4.tar.gz" \
  "https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp312-cp312-linux_x86_64.whl" \
  "https://softwarecenter.qualcomm.com/"; do
  host="$(echo "$u" | cut -d/ -f3)"
  read -r ip bytes rate < <(curl -sL -o /dev/null -m 10 -w "%{remote_ip} %{size_download} %{speed_download}" "$u")
  printf "$FMT" "$host" "$ip" "$bytes" "${rate%.*}"
done
