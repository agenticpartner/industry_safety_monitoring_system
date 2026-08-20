#!/usr/bin/env bash
# Runs ON THE JETSON. The all-or-nothing DLA admission gate (see project_skill.md, "DLA policy").
#
#   ./scripts/qualify_dla.sh [ppe|fire|all] [batch]
#
# A model is admitted to DLA only if TensorRT can place EVERY layer on DLA. We test that by
# building WITHOUT --allowGPUFallback: if any layer is unsupported, the build fails. That is the
# whole point — a build that only succeeds with fallback is a FAILED qualification, because a
# graph split across DLA and GPU copies tensors at every subgraph boundary and typically ends up
# slower than staying wholly on GPU, while still looking like "DLA is being used".
#
# Attempts, in order, stopping at the first pass:
#   A. plain               --useDLACore=N --fp16
#   B. adjusted            + --adjustForDLA   (TRT rewrites layers to be more DLA-amenable)
# On failure it then runs a DIAGNOSTIC build with --allowGPUFallback to report how far off the
# model is (how many subgraphs, which layers fell back), which tells us whether truncating the
# detect head is likely to close the gap.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

WHICH="${1:-all}"
BATCH="${2:-$MAX_STREAMS}"
[ "$WHICH" = "all" ] && MODELS=(ppe fire) || MODELS=("$WHICH")

OUTDIR=bench/dla
mkdir -p "$OUTDIR"
REPORT=bench/dla_qualification.md

{
  echo "# DLA qualification — all-or-nothing gate"
  echo
  echo "Device: Jetson AGX Orin 64GB, TensorRT $(dpkg -l | awk '/libnvinfer-bin/{print $3}')"
  echo "Batch: ${BATCH} · Precision: FP16 · Generated: $(date -Is)"
  echo
  echo "A PASS means TensorRT placed **every** layer on DLA with no GPU fallback."
  echo "A FAIL means the model stays 100% on GPU — partial placement is never shipped."
  echo
} > "$REPORT"

overall_any_pass=0

for M in "${MODELS[@]}"; do
  ONNX="models/${M}/model/${M}.onnx"
  [ -f "$ONNX" ] || { echo "!! $ONNX missing — run export_models.py first"; continue; }

  echo "================================================================"
  echo " ${M}  (batch=${BATCH})"
  echo "================================================================"

  # DLA cannot consume dynamic shapes: bake a static batch first.
  STATIC="models/${M}/model/${M}_b${BATCH}.onnx"
  if [ ! -f "$STATIC" ]; then
    echo "==> baking static batch=${BATCH}"
    ./build/venv-export/bin/python3 \
      .claude/skills/deepstream-import-vision-model/scripts/model/make-static-batch-onnx.py \
      "$ONNX" "$STATIC" "$BATCH" 2>&1 | tail -3
  fi
  [ -f "$STATIC" ] || { echo "!! static-batch conversion failed for ${M}"; continue; }

  SHAPES="images:${BATCH}x3x640x640"
  BASE=(--onnx="$STATIC" --fp16 --useDLACore=0 --reportCapabilityDLA
        --shapes="$SHAPES" --noDataTransfers --skipInference)

  verdict=FAIL; mode=none
  for attempt in plain adjusted; do
    LOG="${OUTDIR}/${M}_${attempt}.log"
    ARGS=("${BASE[@]}")
    [ "$attempt" = adjusted ] && ARGS+=(--adjustForDLA)

    echo "==> attempt: ${attempt} (no --allowGPUFallback)"
    if trtexec "${ARGS[@]}" > "$LOG" 2>&1; then
      verdict=PASS; mode="$attempt"
      echo "    PASS — every layer placed on DLA"
      break
    fi
    echo "    fail — $(grep -m1 -iE 'error|not supported|cannot|unsupported' "$LOG" | cut -c1-140)"
  done

  # ---- diagnostic: how far off is it? ----
  DIAG="${OUTDIR}/${M}_diagnostic.log"
  trtexec --onnx="$STATIC" --fp16 --useDLACore=0 --allowGPUFallback \
          --shapes="$SHAPES" --noDataTransfers --skipInference --verbose \
          > "$DIAG" 2>&1
  dla_sub=$(grep -ciE 'running on DLA|\[DLA\]|DLA node' "$DIAG" 2>/dev/null || echo 0)
  gpu_fb=$(grep -ciE 'running on GPU|falling back to GPU' "$DIAG" 2>/dev/null || echo 0)
  unsupported=$(grep -oiE '[A-Za-z_]+ is not supported on DLA' "$DIAG" | sort -u | head -8)

  echo "    diagnostic: DLA-placement mentions=${dla_sub}  GPU-fallback mentions=${gpu_fb}"
  [ -n "$unsupported" ] && { echo "    unsupported ops:"; echo "$unsupported" | sed 's/^/      /'; }

  [ "$verdict" = PASS ] && overall_any_pass=1

  {
    echo "## ${M}"
    echo
    echo "| | |"
    echo "|---|---|"
    echo "| ONNX | \`${STATIC}\` |"
    echo "| Verdict | **${verdict}** |"
    echo "| Passing mode | ${mode} |"
    echo "| DLA-placement mentions (diagnostic) | ${dla_sub} |"
    echo "| GPU-fallback mentions (diagnostic) | ${gpu_fb} |"
    echo
    if [ "$verdict" = PASS ]; then
      echo "Admitted to DLA. Set \`enable-dla: 1\` and \`use-dla-core\` in \`configs/pgie_${M}.yml\`."
    else
      echo "**Not admitted — stays 100% on GPU.**"
      if [ -n "$unsupported" ]; then
        echo
        echo "Ops TensorRT reported as unsupported on DLA:"
        echo '```'
        echo "$unsupported"
        echo '```'
      fi
      echo
      echo "Next lever if GPU throughput proves insufficient: truncate the ONNX after the neck and"
      echo "move box decode into the custom bbox parser — the detect head is the usual blocker."
    fi
    echo
    echo "Logs: \`${OUTDIR}/${M}_*.log\`"
    echo
  } >> "$REPORT"
done

echo
echo "==> $REPORT"
[ "$overall_any_pass" = 0 ] && echo "==> No model qualified for DLA. Both stay on GPU; that is a valid outcome, not an error."
exit 0
