#!/usr/bin/env bash
# Run a command on the phone as the Termux app, over USB, with Termux's environment.
# Works because the GitHub build of Termux is debuggable, so `run-as com.termux` is allowed.
#
#   bash spike/laptop/termux_run.sh 'python --version'
#   bash spike/laptop/termux_run.sh 'bash ~/bharatai/spike/01_device_profile.sh'
#
# Caveats:
#   - The process belongs to the adb session: unplugging USB kills it.
#   - The SELinux context (runas_app) differs from the real Termux app. Use it for setup and
#     CPU benchmarks; run the NPU probe from inside the Termux app to test that context.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 '<command>'" >&2; exit 2; }

if command -v adb >/dev/null 2>&1; then ADB=adb
elif command -v adb.exe >/dev/null 2>&1; then ADB=adb.exe
elif [ -x /c/Android/platform-tools/adb.exe ]; then ADB=/c/Android/platform-tools/adb.exe
else echo "adb not found" >&2; exit 1
fi

PREFIX=/data/data/com.termux/files/usr
# The script is sent on stdin, so the command needs no extra quoting
{
  cat <<EOF
export PREFIX=$PREFIX HOME=/data/data/com.termux/files/home
export PATH=\$PREFIX/bin LD_LIBRARY_PATH=\$PREFIX/lib TMPDIR=\$PREFIX/tmp LANG=en_US.UTF-8 TERM=dumb
for lib in \$PREFIX/lib/libtermux-exec-ld-preload.so \$PREFIX/lib/libtermux-exec.so; do
  [ -f "\$lib" ] && { export LD_PRELOAD=\$lib; break; }
done
cd \$HOME
EOF
  printf '%s\n' "$*"
} | MSYS_NO_PATHCONV=1 "$ADB" shell -T run-as com.termux "$PREFIX/bin/bash" -s
