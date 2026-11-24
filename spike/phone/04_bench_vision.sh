#!/data/data/com.termux/files/usr/bin/bash
# W1 spike - vision pipeline (route B): time YOLO .pte inference natively with executor_runner.
#
# Per-inference latency is measured by wall clock as (T_N - T_1) / (N - 1), which
# cancels out process start + model load and doesn't depend on the runner's log format.
#
#   bash ~/bharatai/spike/04_bench_vision.sh
# Writes: ~/bharatai/results/vision/summary.csv and raw logs
set -uo pipefail

B="$HOME/bharatai"
RUNNER="$B/bin/executor_runner"
N="${N:-50}"
THREADS="${THREADS:-4 2 -1}"
OUTDIR="$B/results/vision"

[ -x "$RUNNER" ] || { echo "Missing $RUNNER - run 02_install_artifacts.sh (or rely on route A)" >&2; exit 1; }
mkdir -p "$OUTDIR"
termux-wake-lock 2>/dev/null || true

# Only pass flags this runner build understands
HELP="$("$RUNNER" --help 2>&1)"
has_flag() { echo "$HELP" | grep -q -- "-$1"; }
EXTRA=()
has_flag print_output && EXTRA+=(--print_output none)

now_ms() { echo $(( $(date +%s%N) / 1000000 )); }

echo "model,threads,n,load_plus_1_ms,per_inference_ms,fps" > "$OUTDIR/summary.csv"
for model in "$B"/models/vision/*.pte; do
  name="$(basename "$model" .pte)"
  for t in $THREADS; do
    TFLAG=()
    has_flag cpu_threads && TFLAG=(--cpu_threads "$t")

    t0=$(now_ms)
    "$RUNNER" --model_path "$model" --num_executions 1 "${TFLAG[@]}" "${EXTRA[@]}" \
      > "$OUTDIR/${name}_t${t}_n1.log" 2>&1 || { echo "[FAILED] $name t=$t (see log)"; continue; }
    t1=$(now_ms)
    "$RUNNER" --model_path "$model" --num_executions "$N" "${TFLAG[@]}" "${EXTRA[@]}" \
      > "$OUTDIR/${name}_t${t}_n${N}.log" 2>&1 || { echo "[FAILED] $name t=$t N=$N"; continue; }
    t2=$(now_ms)

    one=$(( t1 - t0 ))
    per=$(python -c "print(f'{(($t2 - $t1) - $one) / ($N - 1):.1f}')")
    fps=$(python -c "print(f'{1000 / max($per, 0.001):.1f}')")
    echo "$name,$t,$N,$one,$per,$fps" | tee -a "$OUTDIR/summary.csv"
    sleep 5
  done
done
