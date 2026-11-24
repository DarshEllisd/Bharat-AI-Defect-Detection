#!/usr/bin/env bash
# After a WSL crash: is the Linux filesystem healthy, and did the project's files survive?
echo "== kernel log: ext4 / I/O errors (empty is good)"
dmesg 2>/dev/null | grep -i -E "ext4.*(error|warn)|I/O error|EXT4-fs" | tail -n 10
echo "== mount state of /"
grep " / " /proc/mounts
echo "== key project paths"
W="$HOME/bharatai-build"
for p in "$W/executorch/.git" "$W/et-venv/bin/python" "$W/android-ndk-r28c/ndk-build" \
         "$W/qairt" "$W/yolo11n.pt" "$W/data/mvtec" "$W/data/downloads"; do
  if [ -e "$p" ]; then echo "ok      $p"; else echo "MISSING $p"; fi
done
echo "== venv imports"
"$W/et-venv/bin/python" -c "import torch, executorch, ultralytics, cv2; print('torch', torch.__version__, '| cv2', cv2.__version__)" 2>&1 | tail -n 1
echo "== git integrity of executorch checkout (quick)"
git -C "$W/executorch" fsck --connectivity-only --no-progress 2>&1 | tail -n 3 || true
echo "== MVTec state"
ls -la "$W/data/downloads/" 2>/dev/null | tail -n +2
for d in "$W"/data/mvtec/*/; do [ -d "$d" ] && echo "$(basename "$d"): $(find "$d" -type f | wc -l) files, complete-marker: $([ -f "$d/.complete" ] && echo yes || echo no)"; done
echo "== disk"
df -h / | tail -n 1
