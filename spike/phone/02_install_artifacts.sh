#!/data/data/com.termux/files/usr/bin/bash
# W1 spike - copy binaries + models pushed from the laptop (Download/bharatai) into Termux home.
# Binaries cannot run from shared storage, so they go to ~/bharatai/bin and get +x.
set -euo pipefail

SRC="$HOME/storage/downloads/bharatai"
B="$HOME/bharatai"
[ -d "$SRC" ] || { echo "Missing $SRC - run termux-setup-storage and push files first." >&2; exit 1; }

mkdir -p "$B/bin" "$B/models"
if [ -d "$SRC/bin" ]; then
  cp -f "$SRC/bin/"* "$B/bin/"
  chmod +x "$B/bin/"*
fi
[ -d "$SRC/models" ] && cp -rf "$SRC/models/." "$B/models/"
# NPU probe bundle: binaries + QNN libraries must also live in Termux home to be executable/loadable
if [ -d "$SRC/npu" ]; then
  mkdir -p "$B/npu"
  cp -rf "$SRC/npu/." "$B/npu/"
  chmod +x "$B/npu/bin/"* 2>/dev/null || true
fi

# Refresh the spike scripts too (fixes CRLF line endings if a file was edited on Windows)
if [ -d "$SRC/phone" ]; then
  cp -rf "$SRC/phone/." "$B/spike/"
  find "$B/spike" -type f \( -name '*.sh' -o -name '*.py' -o -name '*.txt' \) -exec sed -i 's/\r$//' {} +
fi

echo "== Installed"
ls -lh "$B/bin"
find "$B/models" -type f -exec ls -lh {} \;
for exe in "$B/bin/"*; do
  # gflags --help exits non-zero; only 126/127 mean the binary cannot execute at all
  "$exe" --help >/dev/null 2>&1 && rc=0 || rc=$?
  if [ "$rc" -eq 126 ] || [ "$rc" -eq 127 ]; then echo "[FAILED] $exe cannot execute (rc=$rc)"; else echo "[ok]     $exe executes"; fi
done
