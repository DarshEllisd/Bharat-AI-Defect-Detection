#!/usr/bin/env bash
# Inspect the built Android binaries: architecture, linkage, size, debug info.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NDK="$(ls -d "$HOME"/bharatai-build/android-ndk-* | head -n1)"
READELF="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf"
cat "$REPO_ROOT/spike/artifacts/BUILD_INFO.txt" 2>/dev/null
for b in "$REPO_ROOT"/spike/artifacts/bin/*; do
  echo "== $(basename "$b")  $(du -h "$b" | cut -f1)"
  file -b "$b"
  echo "shared libs needed: $("$READELF" -d "$b" | awk '/NEEDED/ {print $5}' | tr -d '[]' | tr '\n' ' ')"
  echo "debug sections: $("$READELF" -S "$b" | grep -c '\.debug_')"
done
