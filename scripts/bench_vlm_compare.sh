#!/usr/bin/env bash
# Runs ON THE JETSON. Head-to-head: Cosmos Reason 2 **2B vs 8B** on the same incidents.
#
#   ./scripts/bench_vlm_compare.sh [--runs 3] [--models "2b 8b"]
#
# The question is NOT "which is faster" — the 8B is obviously slower. It is:
#
#   1. **Is it more STABLE?** Phase 2.4 found the same red tabard called "an apron" (confirmed) in
#      one run and "a hi-vis vest" (rejected) in the next — opposite verdicts on identical input.
#      An agent that cites these verdicts cannot be built on a coin flip, so run-to-run agreement
#      is the primary metric here.
#   2. **Is it more ACCURATE?** Measured separately, by rendering the crops and adjudicating them
#      by eye — this script produces the contact sheets, it does not score correctness itself.
#      There is no ground truth in this dataset; a human looking at the crop IS the ground truth.
#   3. **What does it cost?** Latency, memory, and the hot-path hit at 20 streams.
#
# Why stability can be measured automatically but accuracy cannot: running the same input N times
# and comparing answers needs no labels. Deciding whether "a red high-visibility vest" is true of a
# given crop needs eyes.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

RUNS=3
MODELS="2b 8b"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)   RUNS="$2";   shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

OUT=bench/vlm_compare
mkdir -p "$OUT"

# PRE-FLIGHT: nothing else may be reasoning.
#
# This is not hypothetical. The first run of this comparison was silently contaminated by the
# DAEMON reasoning service (started by run_services.sh) processing the same rows concurrently:
# each --once instance saw a different subset, the per-run incident sets did not even overlap,
# and the stability metric collapsed to "2/2 incidents". Same failure as the four concurrent
# pipelines in Phase 2.0 — a benchmark that cannot prove it had the machine to itself is not a
# measurement.
STRAY=$(ps -eo args | grep -c '[r]easoning_service.py' || true)
if [ "${STRAY:-0}" -gt 0 ]; then
  echo "ERROR: ${STRAY} reasoning_service.py already running — it would race this benchmark."
  echo "       ./scripts/run_services.sh stop"
  exit 1
fi
PIPE=$(ps -eo args | grep -c '[s]afety_pipeline.py' || true)
if [ "${PIPE:-0}" -gt 0 ]; then
  echo "NOTE: a pipeline is running; latency here will include contention with it."
fi

mem_mb() { awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print int((t-a)/1024)}' /proc/meminfo; }

for M in $MODELS; do
  echo "############ ${M} ############"

  ./scripts/setup_reasoning.sh stop >/dev/null 2>&1
  sleep 3
  BASE=$(mem_mb)

  ./scripts/setup_reasoning.sh llamacpp --serve --model "$M" 2>&1 | grep -E "^==>|^!!" || {
    echo "!! could not serve ${M}, skipping"; continue; }
  sleep 3
  LOADED=$(mem_mb)
  # Memory is measured as the server process's RSS, NOT as a /proc/meminfo delta.
  #
  # GGUF weights are mmap'd, so they land in page cache and MemAvailable barely moves — a 4.9GB
  # model measured as 303MB that way, which is nonsense. llama-server's own log does not report
  # buffer sizes at this verbosity either (checked), so RSS is the honest available figure.
  RSS=$(ps -o rss= -C llama-server 2>/dev/null | awk '{s+=$1} END{printf "%.0f", s/1024}')
  echo "    model memory: ${RSS:-?} MB RSS  (meminfo delta $(( LOADED - BASE )) MB is misleading — mmap)"
  echo "${RSS:-0}" > "${OUT}/${M}_memory_mb.txt"

  ALIAS="Cosmos-Reason2-$(echo "$M" | tr 'a-z' 'A-Z')"

  # Same incidents, N times, nothing else changing. `--redo` resets every verdict first so each
  # run starts from the same state.
  for i in $(seq 1 "$RUNS"); do
    echo "    run ${i}/${RUNS}"
    python3 services/reasoning_service.py --redo --once --model "$ALIAS" \
      > "${OUT}/${M}_run${i}.log" 2>&1
  done

  # Contact sheet from the LAST run, for eyeball adjudication of correctness.
  python3 tools/verify_verdicts.py --limit 12 --cols 4 \
    --out "${OUT}/${M}_sheet.jpg" > "${OUT}/${M}_sheet.txt" 2>&1

  echo "    latency: $(grep -hoE '[0-9]+\.[0-9]s \|' "${OUT}/${M}_run"*.log | tr -d 's |' \
        | sort -n | awk '{v[NR]=$1} END{if(NR)printf "median %.1fs over %d calls", v[int(NR/2)+1], NR}')"
done

echo
echo "############ stability + agreement ############"
python3 scripts/compare_verdicts.py --dir "$OUT" --runs "$RUNS" --models "$MODELS" \
  | tee "${OUT}/summary.txt"
echo
echo "==> ${OUT}/  (per-run logs, contact sheets, summary.txt)"
echo "    Adjudicate correctness by LOOKING at ${OUT}/*_sheet.jpg — this script does not score it."
