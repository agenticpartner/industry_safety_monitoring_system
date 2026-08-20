#!/usr/bin/env bash
# Runs ON THE JETSON. Builds GPU TensorRT engines and measures raw inference throughput.
#
#   ./scripts/build_engines.sh [ppe|fire|all] [batch] [precisions]
#
#   PRECISIONS defaults to "fp16" — the precision we can actually ship. Pass "fp16 int8" to also
#   measure INT8, but note that without a calibration cache INT8 accuracy is meaningless; it is
#   only ever an upper bound on what a properly calibrated engine could reach, and each build
#   costs several minutes of GPU time.
#
# This answers the question the whole 20-stream target hangs on: how many inferences per second
# can each model actually do? Decode is already proven (bench/decode_ceiling.md); inference is
# the remaining unknown.
#
# Budget to beat at 20 streams (600 frames/s into the muxer):
#   ppe  at interval=1 -> 300 inferences/s
#   fire at interval=5 -> 100 inferences/s
#
# FP16 is the headline number because it is the precision we can actually ship: INT8 is also
# built, but without a calibration cache its accuracy is meaningless — it is measured purely as
# an upper bound on what a properly calibrated INT8 engine could reach.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

WHICH="${1:-all}"
BATCH="${2:-$MAX_STREAMS}"
read -r -a PRECISIONS <<< "${3:-fp16}"
[ "$WHICH" = "all" ] && MODELS=(ppe fire) || MODELS=("$WHICH")

OUTDIR=bench/engines
mkdir -p "$OUTDIR"
CSV=bench/engine_throughput.csv
[ -f "$CSV" ] || echo "model,precision,batch,qps_batches_per_s,inferences_per_s,mean_latency_ms,build_s" > "$CSV"

# trtexec reports throughput in batches/s; inferences/s = qps * batch.
parse()     { grep -oE 'Throughput: [0-9.]+' "$1" | tail -1 | awk '{print $2}'; }
parse_lat() { grep -A2 'Latency:' "$1" | grep -oE 'mean = [0-9.]+' | head -1 | awk '{print $3}'; }

for M in "${MODELS[@]}"; do
  ONNX="models/${M}/model/${M}.onnx"
  [ -f "$ONNX" ] || { echo "!! $ONNX missing — run export_models.py first"; continue; }

  echo "================================================================"
  echo " ${M}  batch=${BATCH}"
  echo "================================================================"

  for PREC in "${PRECISIONS[@]}"; do
    ENG="models/${M}/model/${M}_gpu_b${BATCH}_${PREC}.engine"
    LOG="${OUTDIR}/${M}_gpu_b${BATCH}_${PREC}.log"

    FLAGS=(--onnx="$ONNX"
           --shapes="images:${BATCH}x3x640x640"
           --minShapes="images:1x3x640x640"
           --optShapes="images:${BATCH}x3x640x640"
           --maxShapes="images:${BATCH}x3x640x640"
           --saveEngine="$ENG"
           --timingCacheFile="${OUTDIR}/${M}.timing.cache"
           --noDataTransfers
           --useCudaGraph
           --avgRuns=50 --duration=15)
    if [ "$PREC" = fp16 ]; then FLAGS+=(--fp16); else FLAGS+=(--int8 --fp16); fi

    echo "==> building ${PREC}"
    s=$(date +%s)
    if ! trtexec "${FLAGS[@]}" > "$LOG" 2>&1; then
      echo "    BUILD FAILED — $(grep -m1 -iE 'error' "$LOG" | cut -c1-140)"
      echo "${M},${PREC},${BATCH},,,," >> "$CSV"
      continue
    fi
    e=$(date +%s); build=$(( e - s ))

    qps=$(parse "$LOG"); lat=$(parse_lat "$LOG")
    ips=$(echo "scale=1; ${qps:-0} * ${BATCH}" | bc)
    printf "    %-5s  %8s batches/s  ->  %8s inferences/s   mean %sms   (built in %ss)\n" \
      "$PREC" "${qps:-?}" "$ips" "${lat:-?}" "$build"
    echo "${M},${PREC},${BATCH},${qps},${ips},${lat},${build}" >> "$CSV"
  done
done

echo
echo "==> $CSV"
column -s, -t < "$CSV"
echo
echo "Budget at 20 streams:  ppe needs 300 inf/s (interval=1),  fire needs 100 inf/s (interval=5)"
