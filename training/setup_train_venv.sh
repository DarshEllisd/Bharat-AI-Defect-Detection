#!/usr/bin/env bash
# W3-A - GPU training environment (ADR-018). Separate from et-venv, which must keep CPU torch for
# ExecuTorch exports. Same torch version (2.14) so state_dicts load identically in both.
#   bash training/setup_train_venv.sh
set -euo pipefail
WORK="${WORK:-$HOME/bharatai-build}"
VENV="$WORK/train-venv"
TORCH="${TORCH:-2.14.0}"
TV="${TV:-0.29.0}"
export PATH="$HOME/.local/bin:$PATH"

if [ ! -x "$VENV/bin/pip" ]; then
  rm -rf "$VENV"
  uv venv --seed --python 3.12 "$VENV"
fi
# Pick the newest CUDA build of this torch version that exists (driver supports CUDA 13.x)
for cu in cu130 cu128 cu126; do
  if "$VENV/bin/pip" download --no-deps --dry-run "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$cu" \
       -d /tmp/_probe >/dev/null 2>&1 || \
     "$VENV/bin/pip" install --dry-run --no-deps "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$cu" >/dev/null 2>&1; then
    CU=$cu; break
  fi
done
echo "Using torch $TORCH+$CU"
"$VENV/bin/pip" install "torch==$TORCH" "torchvision==$TV" --index-url "https://download.pytorch.org/whl/$CU"
# Ultralytics etc. from PyPI; torch is already satisfied so it won't be replaced
"$VENV/bin/pip" install "ultralytics>=8.3.220" scikit-learn opencv-python-headless tqdm pyyaml
"$VENV/bin/python" - <<'EOF'
import time, torch, torchvision, ultralytics
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| ultralytics", ultralytics.__version__)
print("cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
if torch.cuda.is_available():
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50): y = x @ x
    torch.cuda.synchronize(); dt = time.time() - t
    print(f"fp16 matmul throughput: {50 * 2 * 4096**3 / dt / 1e12:.1f} TFLOPS")
    print(f"memory: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
EOF
