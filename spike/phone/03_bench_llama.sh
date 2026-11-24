#!/data/data/com.termux/files/usr/bin/bash
# W1 spike - advisory pipeline: run Llama via ExecuTorch llama_main natively in Termux,
# sweeping CPU thread counts with a realistic defect-advisory prompt.
#
#   bash ~/bharatai/spike/03_bench_llama.sh
#   MODEL=~/bharatai/models/llama/llama3_2_3B_instruct_8da4w.pte bash ~/bharatai/spike/03_bench_llama.sh
#
# Writes: ~/bharatai/results/llama/<model>_t<threads>.log and summary.csv
set -uo pipefail

B="$HOME/bharatai"
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$B/bin/llama_main"
MODEL="${MODEL:-$(find "$B/models/llama" -name '*.pte' | sort | head -n1)}"
TOK="${TOK:-$B/models/llama/tokenizer.model}"
THREADS="${THREADS:-4 2 6 8 -1}"     # 4 = the four A715 performance cores; -1 = runtime heuristic
MAX_NEW="${MAX_NEW:-128}"
COOLDOWN="${COOLDOWN:-20}"           # seconds between runs so thermals don't skew later runs
PROMPT_FILE="$HERE/prompts/advisory_prompt.txt"
OUTDIR="$B/results/llama"

[ -x "$BIN" ] || { echo "Missing $BIN - run 02_install_artifacts.sh" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "No .pte under $B/models/llama" >&2; exit 1; }
mkdir -p "$OUTDIR"
termux-wake-lock 2>/dev/null || true

# GNU time gives peak RSS ("Maximum resident set size")
TIMECMD=()
if [ -x "$PREFIX/bin/time" ]; then TIMECMD=("$PREFIX/bin/time" -v); fi

name="$(basename "$MODEL" .pte)"
echo "Model: $MODEL"
for t in $THREADS; do
  log="$OUTDIR/${name}_t${t}.log"
  echo "-- threads=$t -> $log"
  {
    echo "# model=$MODEL threads=$t max_new_tokens=$MAX_NEW date=$(date -Iseconds)"
    "${TIMECMD[@]}" "$BIN" \
      --model_path "$MODEL" \
      --tokenizer_path "$TOK" \
      --prompt_file "$PROMPT_FILE" \
      --temperature 0 \
      --max_new_tokens "$MAX_NEW" \
      --cpu_threads "$t" \
      --warmup
    echo "# exit_code=$?"
  } > "$log" 2>&1
  tail -n 3 "$log"
  sleep "$COOLDOWN"
done

python "$HERE/parse_llama_log.py" "$OUTDIR" | tee "$OUTDIR/summary.csv"
