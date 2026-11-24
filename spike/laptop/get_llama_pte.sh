#!/usr/bin/env bash
# W1 spike - get a Llama 3.2 Instruct .pte + tokenizer for the phone.
#
#   bash spike/laptop/get_llama_pte.sh download        # pre-exported SpinQuant INT4 1B (fastest path)
#   bash spike/laptop/get_llama_pte.sh export          # export 1B ourselves (8da4w, XNNPACK)
#   SIZE=3B bash spike/laptop/get_llama_pte.sh export  # optional 3B comparison
#
# "download" pulls a .pte made by the ExecuTorch community with an older ExecuTorch.
# If llama_main on the phone fails to load it (version mismatch), use "export",
# which builds the .pte with the same ExecuTorch tag as the phone runner.
#
# Meta's Llama repos are gated: accept the Llama 3.2 licence on huggingface.co
# with your account and run `hf auth login` first if you get a 401/403.
set -euo pipefail

MODE="${1:-download}"
SIZE="${SIZE:-1B}"
WORK="${WORK:-$HOME/bharatai-build}"
ET_DIR="$WORK/executorch"
VENV="$WORK/et-venv"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/spike/artifacts/models/llama"
mkdir -p "$OUT"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade huggingface_hub
HF="$(command -v hf || command -v huggingface-cli)"

case "$MODE" in
  download)
    REPO_ID="executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET"
    "$HF" download "$REPO_ID" \
      Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte tokenizer.model \
      --local-dir "$OUT"
    ;;
  export)
    CKPT_DIR="$WORK/llama-3.2-$SIZE-instruct"
    "$HF" download "meta-llama/Llama-3.2-$SIZE-Instruct" --include "original/*" --local-dir "$CKPT_DIR"
    cd "$ET_DIR"
    bash examples/models/llama/install_requirements.sh
    marker="$(mktemp)"
    # llama_xnnpack.yaml = 8-bit dynamic activations, 4-bit grouped weights, KV cache, XNNPACK.
    # max_seq_length raised from the default so the ~400-token advisory prompt fits.
    python -m extension.llm.export.export_llm \
      --config examples/models/llama/config/llama_xnnpack.yaml \
      +base.model_class="llama3_2" \
      +base.checkpoint="$CKPT_DIR/original/consolidated.00.pth" \
      +base.params="$CKPT_DIR/original/params.json" \
      +export.max_seq_length=1024 \
      +export.max_context_length=1024
    PTE="$(find . -maxdepth 2 -name '*.pte' -newer "$marker" | head -n1)"
    rm -f "$marker"
    [ -n "$PTE" ] || { echo "Export finished but no new .pte found in $ET_DIR" >&2; exit 1; }
    mv "$PTE" "$OUT/llama3_2_${SIZE}_instruct_8da4w.pte"
    cp "$CKPT_DIR/original/tokenizer.model" "$OUT/tokenizer.model"
    ;;
  *)
    echo "usage: $0 [download|export]" >&2; exit 2 ;;
esac

ls -lh "$OUT"
