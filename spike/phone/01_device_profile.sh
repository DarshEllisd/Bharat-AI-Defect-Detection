#!/data/data/com.termux/files/usr/bin/bash
# W1 spike - record SoC, CPU cores, ISA features and memory.
# Writes: ~/bharatai/results/device_profile.txt
set -uo pipefail

OUT="$HOME/bharatai/results/device_profile.txt"
mkdir -p "$(dirname "$OUT")"

features="$(grep -m1 -i '^Features' /proc/cpuinfo | cut -d: -f2)"

{
  echo "== Device"
  for p in ro.product.manufacturer ro.product.model ro.soc.manufacturer ro.soc.model \
           ro.board.platform ro.build.version.release ro.build.version.sdk; do
    echo "$p=$(getprop "$p")"
  done

  echo; echo "== CPU cores (expected for Snapdragon 7 Gen 3: 1x A715 @~2.63GHz, 3x A715 @2.4GHz, 4x A510 @1.8GHz)"
  echo "   CPU part 0xd4d = Cortex-A715, 0xd46 = Cortex-A510"
  for c in /sys/devices/system/cpu/cpu[0-9]*; do
    n="${c##*cpu}"
    khz="$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo '?')"
    part="$(awk -v n="$n" '/^processor/{p=$3} /^CPU part/{if (p==n) print $4}' /proc/cpuinfo)"
    echo "cpu$n  max_khz=$khz  part=${part:-?}"
  done

  echo; echo "== ISA flags relevant to XNNPACK int8/int4 kernels"
  for f in asimddp i8mm bf16 sve sve2 asimdhp; do
    if echo " $features " | grep -qw "$f"; then echo "$f: yes"; else echo "$f: no"; fi
  done
  echo "Features:$features"

  echo; echo "== Memory"
  grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo

  echo; echo "== Storage (Termux home)"
  df -h "$HOME" | tail -n1

  echo; echo "== Thermal zones (current temps, millidegC)"
  for z in /sys/class/thermal/thermal_zone*; do
    t="$(cat "$z/temp" 2>/dev/null)" && echo "$(cat "$z/type" 2>/dev/null): $t"
  done 2>/dev/null | head -n 12

  echo; echo "== Software"
  echo "python: $(python --version 2>&1)"
  echo "kernel: $(uname -a)"
} | tee "$OUT"
