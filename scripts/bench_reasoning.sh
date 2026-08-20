#!/usr/bin/env bash
# Runs ON THE JETSON. Phase 2.0 gate, step 4: what does local reasoning cost the hot path?
#
#   ./scripts/bench_reasoning.sh --model <name> [--endpoint URL] [--streams N] [--repeat K]
#
# Everything else in the gate (does it load, does it answer, how fast) is measured by
# scripts/bench_reasoning.py against the endpoint alone. This script answers the only question
# that can actually kill the plan: **when the VLM runs, does the 20-stream pipeline still hold
# realtime?**
#
# Method — three phases against ONE continuously running pipeline, so nothing is compared across
# process starts (TensorRT deserialisation alone is ~20 s and would swamp the effect):
#
#   A  baseline    pipeline alone, VLM idle but LOADED. Loaded, not stopped: the weights are
#                  resident either way in production, so the honest baseline includes them
#                  sitting in memory. Measuring against a machine with no model loaded would
#                  flatter the result by counting memory pressure as a reasoning cost.
#   B  under load  K reasoning requests fired back to back — the worst realistic burst.
#   C  recovery    pipeline alone again. If fps does not return to A, the cost is not transient
#                  and something is leaking or thermally throttling.
#
# fps lines are attributed to phases by LINE OFFSET in the pipeline log rather than by timestamp,
# because the pipeline prints `[fps]` unstamped and adding a timestamp would mean touching the
# hot-path probe. Offsets are exact: the log is append-only and single-writer.
#
# The pass condition is on phase B, not on the average. An average over A+B+C hides exactly the
# dip we are looking for.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

ENDPOINT="http://127.0.0.1:8000/v1"
MODEL=""
STREAMS=20
REPEAT=8
FRAMES=6
SETTLE=60          # seconds of phase A/C. Must exceed the probe's 300-frame reporting window.
LABEL=""
DFI=0              # decoder drop-frame-interval; 0/1 = 30fps analytics, 3 = 10fps

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --model)    MODEL="$2";    shift 2 ;;
    --streams)  STREAMS="$2";  shift 2 ;;
    --repeat)   REPEAT="$2";   shift 2 ;;
    --frames)   FRAMES="$2";   shift 2 ;;
    --settle)   SETTLE="$2";   shift 2 ;;
    --dfi)      DFI="$2";      shift 2 ;;
    --label)    LABEL="$2";    shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done
[ -n "$MODEL" ] || { echo "ERROR: --model is required"; exit 1; }
LABEL="${LABEL:-$(basename "$MODEL")}"

OUT="bench/reasoning_${LABEL}_n${STREAMS}_dfi${DFI}"
mkdir -p "$OUT"
PIPE_LOG="${OUT}/pipeline.log"
VLM_LOG="${OUT}/vlm.log"
TG="${OUT}/tegrastats"

# Fail before spending three minutes on a pipeline if the endpoint is not actually up.
curl -sf --max-time 10 "${ENDPOINT%/}/models" >/dev/null 2>&1 \
  || { echo "ERROR: no OpenAI-compatible endpoint at ${ENDPOINT} — start the server first"; exit 1; }

# fps lines seen so far. The pipeline emits one per 300 frames per the probe's window.
#
# NOTE the `|| true`, not `|| echo 0`: on no match `grep -c` ALREADY prints "0" and then exits 1,
# so an `|| echo 0` fallback appends a SECOND zero and the caller compares the string "0\n0" as an
# integer ("integer expression expected"). Let grep's own output stand.
fps_lines() { grep -ac '^\[fps\]' "$PIPE_LOG" 2>/dev/null || true; }

# Median of the `[fps] X frames/s aggregate` values in a line range. Median, not mean: a single
# stall (a page fault, a log flush) skews a mean over only a handful of windows.
fps_median() {
  local from="$1" to="$2"
  grep -a '^\[fps\]' "$PIPE_LOG" | sed -n "${from},${to}p" \
    | grep -oE '^\[fps\] [0-9.]+' | awk '{print $2}' \
    | sort -n | awk '{v[NR]=$1} END{ if(NR==0){print "0"} else if(NR%2){printf "%.1f", v[(NR+1)/2]} else {printf "%.1f", (v[NR/2]+v[NR/2+1])/2} }'
}

echo "==> ${STREAMS} streams + ${MODEL} @ ${ENDPOINT}"

# LOCK THE CLOCKS, or this benchmark measures the governor instead of the workload.
#
# Measured on this device, dfi=3, before locking: the pipeline alone sits at 3% GPU and the
# memory controller idles at 45% EMC. Starting the VLM pushed EMC to 62% — and the PIPELINE got
# 25% FASTER, because it is memory-bound and the governor had been under-clocking it. The naive
# reading of that run is "reasoning improves throughput", which is nonsense; DVFS was the hidden
# variable in both directions. jetson_clocks pins CPU, GPU and EMC to their nvpmodel maxima so A,
# B and C are comparable.
#
# nvpmodel MAXN is asserted too: a benchmark run in a lower power mode is not comparable to one
# run in MAXN, and the mode survives reboots.
sudo nvpmodel -m 0 >/dev/null 2>&1 || true
sudo jetson_clocks 2>/dev/null || echo "!! jetson_clocks failed — numbers will carry DVFS noise"
echo "    clocks locked (nvpmodel MAXN + jetson_clocks)"

sudo tegrastats --stop >/dev/null 2>&1 || true
sudo tegrastats --interval 1000 > "$TG" 2>/dev/null &

# PRE-FLIGHT: refuse to run if a pipeline is already up.
#
# This is not paranoia — it happened. An earlier version launched the pipeline with
# `nohup setsid`, which moves the process into a NEW SESSION, so `$!` captured the short-lived
# setsid parent and the cleanup `kill` hit nothing. Four 20-stream pipelines ended up running
# concurrently and each successive benchmark measured the contention, not the workload: apparent
# throughput fell 871 -> 254 fps across runs and very nearly got written up as a DVFS finding.
# A benchmark that cannot prove it had the machine to itself is not a measurement.
STRAY=$(pgrep -fc 'safety_pipeline\.py' 2>/dev/null || true)
if [ "${STRAY:-0}" -gt 0 ]; then
  echo "ERROR: ${STRAY} safety_pipeline.py process(es) already running — refusing to benchmark."
  echo "       kill them first:  pkill -9 -f safety_pipeline.py"
  exit 1
fi

# The pipeline is bounded by --duration and killed with SIGKILL as the exit path: pipeline.wait()
# blocks in C++, so SIGINT/SIGTERM are ignored and a benchmark that outlives its window would
# otherwise have to be hunted down by hand. It has no state to flush.
#
# Plain `&`, NOT `nohup setsid`: this script is already launched detached by the caller, so its
# children need no session of their own — and keeping them in this session is what makes $! the
# real PID and the cleanup trap actually work.
DURATION=$(( SETTLE * 2 + REPEAT * 60 + 120 ))
timeout --signal=KILL "$DURATION" \
  python3 app/safety_pipeline.py --streams "$STREAMS" --source file \
          --drop-frame-interval "$DFI" \
          --no-display --fps --stats > "$PIPE_LOG" 2>&1 < /dev/null &
PIPE_PID=$!
# Belt and braces: kill the tracked PID, then sweep by name. `pkill -f` is safe here because this
# script's own command line does not contain the pattern (it is `bash scripts/bench_reasoning.sh`).
cleanup() {
  kill -9 "$PIPE_PID" 2>/dev/null
  pkill -9 -f 'safety_pipeline\.py' 2>/dev/null
  sudo tegrastats --stop >/dev/null 2>&1
}
trap cleanup EXIT

echo "--- waiting for the pipeline to reach steady state ---"
for _ in $(seq 1 90); do [ "$(fps_lines)" -ge 2 ] && break; sleep 2; done
[ "$(fps_lines)" -ge 2 ] || { echo "ERROR: pipeline never reported fps — see $PIPE_LOG"; exit 1; }

# Discard everything so far: the first windows still carry engine deserialisation and decoder
# ramp-up. Phase A starts from a pipeline that has already settled.
A_FROM=$(( $(fps_lines) + 1 ))
echo "--- A: baseline, ${SETTLE}s (VLM loaded, idle) ---"
sleep "$SETTLE"
A_TO=$(fps_lines)

echo "--- B: ${REPEAT} reasoning requests, back to back ---"
B_FROM=$(( A_TO + 1 ))
python3 scripts/bench_reasoning.py --endpoint "$ENDPOINT" --model "$MODEL" \
        --frames "$FRAMES" --repeat "$REPEAT" --warmup 1 \
        --json "${OUT}/vlm.json" > "$VLM_LOG" 2>&1
VLM_RC=$?
B_TO=$(fps_lines)

echo "--- C: recovery, ${SETTLE}s ---"
C_FROM=$(( B_TO + 1 ))
sleep "$SETTLE"
C_TO=$(fps_lines)

kill -9 "$PIPE_PID" 2>/dev/null
pkill -9 -f 'safety_pipeline\.py' 2>/dev/null
sudo tegrastats --stop >/dev/null 2>&1 || true
sleep 2
# Prove it is gone. A leftover pipeline would silently contaminate the NEXT run in a sweep.
LEFT=$(pgrep -fc 'safety_pipeline\.py' 2>/dev/null || true)
[ "${LEFT:-0}" -gt 0 ] && echo "!! WARNING: ${LEFT} pipeline process(es) survived cleanup"

# Same guard as scripts/sweep.sh, and for the same reason: in Phase 1 a broken tracker config
# emitted zero objects and became the FASTEST entry in the benchmark. An fps number from a
# pipeline that detected nothing is not a result.
DETS=$(grep -aoE 'detections: [^|]*' "$PIPE_LOG" | tail -1 | sed 's/detections: //;s/ *$//')
if [ -z "$DETS" ] || [ "$DETS" = "none" ]; then
  echo "!! ZERO DETECTIONS — the fps figures below are meaningless. Not a pass."
  exit 3
fi

A=$(fps_median "$A_FROM" "$A_TO")
B=$(fps_median "$B_FROM" "$B_TO")
C=$(fps_median "$C_FROM" "$C_TO")

# The realtime target MUST track the drop interval. With drop-frame-interval=3 the decoder emits
# 10 fps per stream, so 200 fps aggregate at 20 cameras IS realtime — judging that run against
# the 30 fps target of 600 would declare a healthy pipeline broken. Getting this wrong in either
# direction is how a benchmark lies: too high and the design looks unviable, too low and a real
# deficit is hidden.
ANALYTICS_FPS=$(( DFI > 1 ? 30 / DFI : 30 ))
TARGET=$(( STREAMS * ANALYTICS_FPS ))

pct() { awk -v a="$1" -v b="$2" 'BEGIN{ if(a<=0){print "n/a"} else {printf "%+.1f%%", (b-a)/a*100} }'; }
RAM=$(grep -oE 'RAM [0-9]+/' "$TG" | grep -oE '[0-9]+' | sort -n | tail -1)
GPU=$(grep -oE 'GR3D_FREQ [0-9]+%' "$TG" | grep -oE '[0-9]+' | sort -n | tail -1)
# EMC is reported because it is the variable that silently invalidated the first dfi=3 run. If a
# future run shows EMC moving between phases, the fps deltas are governor artefacts, not workload.
EMC_MED=$(grep -oE 'EMC_FREQ [0-9]+%' "$TG" | grep -oE '[0-9]+' | sort -n \
          | awk '{v[NR]=$1} END{ if(NR) print v[int(NR/2)+1] }')
EMC_MAX=$(grep -oE 'EMC_FREQ [0-9]+%' "$TG" | grep -oE '[0-9]+' | sort -n | tail -1)

{
  echo
  echo "streams              ${STREAMS}   dfi=${DFI} (${ANALYTICS_FPS} fps analytics/stream)"
  echo "realtime target      ${TARGET} fps aggregate"
  echo "A baseline           ${A} fps   [windows ${A_FROM}-${A_TO}]"
  echo "B under reasoning    ${B} fps   $(pct "$A" "$B")   [windows ${B_FROM}-${B_TO}]"
  echo "C recovery           ${C} fps   $(pct "$A" "$C")   [windows ${C_FROM}-${C_TO}]"
  echo "peak GPU ${GPU:-?}%   peak RAM ${RAM:-?} MB   EMC med ${EMC_MED:-?}% / max ${EMC_MAX:-?}%"
  echo "detections           ${DETS}"
  echo
  awk -v b="$B" -v t="$TARGET" 'BEGIN{
    if (b >= t) printf "VERDICT: hot path HOLDS realtime under reasoning (%.2fx margin)\n", b/t;
    else        printf "VERDICT: hot path DROPS BELOW realtime under reasoning (%.2fx) — fewer cameras or remote reasoning\n", b/t;
  }'
  [ $VLM_RC -ne 0 ] && echo "NOTE: bench_reasoning.py exited ${VLM_RC} — see ${VLM_LOG}"
} | tee "${OUT}/summary.txt"

echo
echo "==> ${OUT}/  (summary.txt, pipeline.log, vlm.log, vlm.json, tegrastats)"
