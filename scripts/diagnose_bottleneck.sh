#!/usr/bin/env bash
# Runs ON THE JETSON. Answers "what is actually consuming the GPU?" by ablation.
#
#   ./scripts/diagnose_bottleneck.sh [streams] [seconds]
#
# The engine benchmarks say inference should cost ~7 ms of a batch at N=8, yet the measured
# batch time is ~190 ms and GR3D sits at 97%. Something outside nvinfer dominates. Rather than
# guess, this removes one component at a time and reports the throughput delta.
set -uo pipefail
cd "$(dirname "$0")/.."
N="${1:-8}"; SECS="${2:-30}"
OUT=bench/bottleneck_n${N}.csv
mkdir -p bench
echo "variant,measured_fps,gr3d_pct_peak,note" > "$OUT"

run() {
  local name="$1"; shift
  local log="bench/bottleneck_${name}_n${N}.log"
  local tg; tg=$(mktemp)
  sudo tegrastats --interval 500 > "$tg" 2>/dev/null & sleep 0.4
  timeout --signal=KILL $(( SECS + 12 )) \
    python3 app/safety_pipeline.py --streams "$N" --source file --no-display \
      --fps --duration "$SECS" "$@" > "$log" 2>&1
  sudo tegrastats --stop >/dev/null 2>&1 || true
  local fps gr3d
  fps=$(grep -aoE '\[fps\] [0-9.]+' "$log" | tail -1 | awk '{print $2}')
  gr3d=$(grep -oE 'GR3D_FREQ [0-9]+%' "$tg" | grep -oE '[0-9]+' | sort -n | tail -1)
  rm -f "$tg"
  printf "  %-28s %8s fps   GPU %s%%\n" "$name" "${fps:-FAIL}" "${gr3d:-?}"
  echo "${name},${fps:-},${gr3d:-},$*" >> "$OUT"
}

echo "==> ablation at N=${N}, ${SECS}s per variant"
run full
run no-probe            --no-probe
run no-tracker          --no-tracker
run no-tracker-no-probe --no-tracker --no-probe
run no-fire             --no-fire
run no-fire-no-tracker  --no-fire --no-tracker

echo
column -s, -t < "$OUT"
