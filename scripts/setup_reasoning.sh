#!/usr/bin/env bash
# Runs ON THE JETSON. Stands up the local reasoning endpoint for Phase 2.
#
#   ./scripts/setup_reasoning.sh llamacpp [--serve]     VLM (Cosmos Reason 2) on :8000
#   ./scripts/setup_reasoning.sh llm      [--serve]     agent LLM (Nemotron Nano 9B) on :8001
#   ./scripts/setup_reasoning.sh vllm     [--serve]     try the JP6-targeted vLLM container
#   ./scripts/setup_reasoning.sh stop                   stop whatever is serving
#
# Both backends expose the SAME OpenAI-compatible surface on :8000/v1, which is the whole point —
# scripts/bench_reasoning.py and services/reasoning_service.py never learn which one won.
#
# WHY TWO BACKENDS, and why llamacpp is the default:
#
# Cosmos-Reason2-2B is `Qwen3VLForConditionalGeneration` (model_type qwen3_vl, transformers
# 4.57.0.dev0) — a very new architecture. The documented serving path,
# ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin, is built for JetPack 6 / L4T r36, and this device
# is JetPack 7.2.1 / L4T R39.2 / CUDA 13.2. So the container carries TWO independent risks: the
# JetPack gap, and whether its vLLM is new enough to know qwen3_vl. Try it, but do not sink time
# into it — llama.cpp builds from source against the CUDA that is actually installed here and is
# far less coupled to JetPack internals.
#
# Vision through llama.cpp needs the multimodal projector (`mmproj-*.gguf`) alongside the weights.
# Without --mmproj the server loads happily and answers as a TEXT-ONLY model — it will confidently
# discuss a warehouse it cannot see. That failure is silent and is exactly the kind that survives
# to a demo, so the mmproj file is treated as required, not optional.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

PORT=8000
# The agent LLM runs as a SECOND server on its own port, not as a replacement. The two models do
# different jobs at very different duty cycles: the VLM is called per incident (continuously, so
# it must be cheap), the LLM only per human question (rarely, so it can be bigger). Keeping them
# separate also means restarting one never disturbs the other.
LLM_PORT=8001
LLM_MODEL="models/nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q5_K_M.gguf"
LLM_ALIAS="Nemotron-Nano-9B-v2"
# Quantised GGUF of the agent LLM. Q5_K_M is the size/quality knee for this model on a
# 64 GB Orin: it answers as well as Q8 on the planner's constrained JSON while leaving the
# 20-stream pipeline the memory it needs.
export LLM_REPO_GGUF="${LLM_REPO_GGUF:-bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF}"
LLAMA_DIR="${HOME}/llama.cpp"
MODEL_DIR="models/cosmos"
HF_REPO_GGUF="robertzty/Cosmos-Reason2-2B-GGUF"
# Which model to serve: 2b (default) or 8b. Both are Cosmos Reason 2; the 8B is quantised Q8_0
# for the language tower but keeps its vision projector at BF16, because vision fidelity is
# exactly what the size comparison is testing (see bench/reasoning.md §5b).
MODEL_SIZE="${MODEL_SIZE:-2b}"
# Minimum image tokens per image. llama-server warns at load time:
#
#   "Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding
#    tasks / if you encounter problems with accuracy, try adding --image-min-tokens 1024"
#
# Cosmos Reason 2 IS a Qwen3-VL model, and our PPE crops are small and tall (448px wide), so
# they land far below that. Phase 2.4 saw exactly the symptoms this warning predicts: confident
# but wrong attribute calls, and a traffic cone described as a worker. Default it ON; set
# IMAGE_MIN_TOKENS=0 to reproduce the old behaviour.
IMAGE_MIN_TOKENS="${IMAGE_MIN_TOKENS:-1024}"
HF_REPO_HF="nvidia/Cosmos-Reason2-2B"
LOGDIR="logs"; mkdir -p "$LOGDIR"

# Building for the one arch that exists here rather than a fat binary cuts the compile roughly in
# half, and there is no other GPU to serve. sm_87 is Orin's Ampere compute capability; the value
# is detected rather than assumed so this still builds correctly on a different Jetson.
[ -f build/hardware.env ] && { set -a; . build/hardware.env; set +a; }
CUDA_ARCH="${CUDA_ARCH:-${DETECTED_CUDA_ARCH:-87}}"

# The token lives in a file, not in ~/.bashrc: Ubuntu's default .bashrc returns early for
# non-interactive shells, so an `export` there is invisible to exactly the ssh commands that
# need it (verified — HF_TOKEN read as UNSET over ssh until it was moved here).
load_env

usage() { sed -n '2,12p' "$0"; exit 1; }
ACTION="${1:-}"; shift || true
SERVE=0
prev=""
for a in "$@"; do
  [ "$a" = "--serve" ] && SERVE=1
  [ "$prev" = "--model" ] && MODEL_SIZE="$a"
  prev="$a"
done
case "$MODEL_SIZE" in 2b|8b) ;; *) echo "ERROR: --model must be 2b or 8b"; exit 1 ;; esac


fetch_llm_gguf() {
  [ -f "$LLM_MODEL" ] && { echo "==> ${LLM_MODEL##*/} already present, skipping"; return 0; }
  [ -x build/venv-hf/bin/python3 ] || { echo "ERROR: build/venv-hf missing — run scripts/setup.sh"; exit 1; }
  echo "==> fetching ${LLM_REPO_GGUF} (${LLM_MODEL##*/})"
  mkdir -p "$(dirname "$LLM_MODEL")"
  build/venv-hf/bin/python3 - "$(dirname "$LLM_MODEL")" "$(basename "$LLM_MODEL")" <<'HFPY' || exit 1
import os, sys
from huggingface_hub import hf_hub_download
path = hf_hub_download(os.environ["LLM_REPO_GGUF"], sys.argv[2],
                       local_dir=sys.argv[1], token=os.environ.get("HF_TOKEN"))
print("   ->", path)
HFPY
}


serve_llm() {
  fetch_llm_gguf
  [ -f "$LLM_MODEL" ] || { echo "ERROR: $LLM_MODEL missing"; exit 1; }
  pkill -f "[l]lama-server.*${LLM_PORT}" 2>/dev/null; sleep 1
  echo "==> serving ${LLM_ALIAS} on :${LLM_PORT}"
  # No --mmproj: this is a text-only reasoning model, not a VLM.
  # --ctx-size 8192 holds the tool schema, the live enum vocabulary, a few turns of conversation
  # and a page of retrieved incidents.
  nohup setsid "${LLAMA_DIR}/build/bin/llama-server" \
      --model "$LLM_MODEL" \
      --host 0.0.0.0 --port "$LLM_PORT" \
      -ngl 99 --ctx-size 8192 --parallel 1 --no-warmup \
      --alias "$LLM_ALIAS" \
      > "${LOGDIR}/llm_server.log" 2>&1 < /dev/null &
  local n
  echo -n "==> waiting for /v1/models "
  for n in $(seq 1 180); do
    if curl -sf --max-time 3 "http://127.0.0.1:${LLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo; echo "==> UP: http://127.0.0.1:${LLM_PORT}/v1"; return 0
    fi
    echo -n "."; sleep 2
  done
  echo; echo "!! LLM never came up. Last log lines:"; tail -25 "${LOGDIR}/llm_server.log"
  return 1
}


stop_all() {
  # `pkill -x` matches the PROCESS NAME exactly; `pkill -f` matches the whole command line and is
  # the wrong tool here. Bracket-matching (`[l]lama-server`) is NOT sufficient either: it only
  # stops pkill from matching ITSELF, while any ancestor ssh command line that happens to mention
  # llama-server still matches and the session dies with exit 255. That happened — an ssh command
  # that called this function and then restarted the server killed itself mid-measurement.
  pkill -x llama-server 2>/dev/null && echo "stopped llama-server(s)" || true
  docker rm -f cosmos-vllm 2>/dev/null >/dev/null && echo "stopped cosmos-vllm" || true
}


build_llamacpp() {
  if [ ! -d "$LLAMA_DIR" ]; then
    echo "==> cloning llama.cpp into ${LLAMA_DIR}"
    # Built from source against the CUDA that is actually installed here. The documented vLLM
    # container path is built for an older JetPack and carries two independent risks: the JetPack
    # gap, and whether its vLLM is new enough to know this model's architecture at all.
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" || {
      echo "ERROR: clone failed — check network access"; exit 1; }
  fi
  if [ -x "${LLAMA_DIR}/build/bin/llama-server" ]; then
    echo "==> llama-server already built, skipping"
    return
  fi
  echo "==> building llama.cpp with CUDA (arch ${CUDA_ARCH}, $(nproc) jobs)"
  cmake -S "$LLAMA_DIR" -B "${LLAMA_DIR}/build" \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc || exit 1
  cmake --build "${LLAMA_DIR}/build" --config Release -j "$(nproc)" --target llama-server || exit 1
  echo "==> built ${LLAMA_DIR}/build/bin/llama-server"
}


fetch_gguf() {
  local out="${MODEL_DIR}/gguf"
  mkdir -p "$out"
  # The BF16 weights ship as a 2-part split; llama.cpp follows the split automatically when given
  # part 1, so only the first shard is named downstream.
  if [ -f "${out}/mmproj-Cosmos-Reason2-2B-BF16.gguf" ] \
     && [ -f "${out}/Cosmos-Reason2-2B-BF16-split-00001-of-00002.gguf" ]; then
    echo "==> GGUF already present, skipping"
    return
  fi
  echo "==> fetching ${HF_REPO_GGUF} (weights + mmproj)"
  build/venv-hf/bin/python3 - "$out" <<PY || exit 1
import os, sys
from huggingface_hub import snapshot_download
snapshot_download("${HF_REPO_GGUF}", local_dir=sys.argv[1],
                  token=os.environ.get("HF_TOKEN"),
                  allow_patterns=["*.gguf"])
print("ok")
PY
}


serve_llamacpp() {
  local out weights mmproj alias ctx
  if [ "$MODEL_SIZE" = 8b ]; then
    out="${MODEL_DIR}/gguf8b"
    weights="${out}/Cosmos-Reason2-8B.Q8_0.gguf"
    mmproj="${out}/Cosmos-Reason2-8B.mmproj-bf16.gguf"
    alias="Cosmos-Reason2-8B"
  else
    out="${MODEL_DIR}/gguf"
    weights="${out}/Cosmos-Reason2-2B-BF16-split-00001-of-00002.gguf"
    mmproj="${out}/mmproj-Cosmos-Reason2-2B-BF16.gguf"
    alias="Cosmos-Reason2-2B"
  fi
  [ -f "$weights" ] || { echo "ERROR: $weights missing"; exit 1; }
  [ -f "$mmproj" ]  || { echo "ERROR: $mmproj missing — without it the server is TEXT-ONLY and will answer about images it never saw"; exit 1; }

  stop_all; sleep 1
  echo "==> serving ${alias} on :${PORT} (image-min-tokens=${IMAGE_MIN_TOKENS:-off})"
  # -ngl 99  : every layer on GPU. Orin's memory is unified, so there is no host<->device copy to
  #            trade off — leaving layers on CPU is pure loss here.
  # --ctx-size: 8192 fits 6-8 frames of vision tokens plus the prompt and a short answer. Larger
  #            costs KV cache memory that the 20-stream pipeline needs more than the VLM does.
  # --parallel 1: admission control starts here. Phase 2.4 allows ONE reasoning request in flight
  #            by design; letting the server fan out would hand the GPU to the cold path exactly
  #            when a burst of violations makes the hot path matter most.
  nohup setsid "${LLAMA_DIR}/build/bin/llama-server" \
      --model "$weights" --mmproj "$mmproj" \
      --host 0.0.0.0 --port "$PORT" \
      -ngl 99 --ctx-size 8192 --parallel 1 --no-warmup \
      ${IMAGE_MIN_TOKENS:+--image-min-tokens $IMAGE_MIN_TOKENS} \
      --alias "$alias" \
      > "${LOGDIR}/llama_server.log" 2>&1 < /dev/null &
  wait_ready
}


serve_vllm() {
  local img="ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin"
  docker image inspect "$img" >/dev/null 2>&1 || { echo "ERROR: $img not pulled"; exit 1; }
  stop_all; sleep 1
  echo "==> trying vLLM container (JP6-targeted; failure here is an EXPECTED outcome, not a bug)"
  docker run -d --name cosmos-vllm --runtime nvidia --network host --ipc host \
    -v "$(pwd)/${MODEL_DIR}/${HF_REPO_HF##*/}:/model:ro" \
    "$img" \
    vllm serve /model --served-model-name Cosmos-Reason2-2B \
      --port "$PORT" --max-model-len 8192 --gpu-memory-utilization 0.45 \
      --limit-mm-per-prompt '{"image":8}' \
    > /dev/null || exit 1
  wait_ready "docker logs cosmos-vllm"
}


wait_ready() {
  local logcmd="${1:-tail -40 ${LOGDIR}/llama_server.log}"
  echo -n "==> waiting for /v1/models "
  for i in $(seq 1 150); do
    if curl -sf --max-time 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo
      echo "==> UP: http://127.0.0.1:${PORT}/v1"
      curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 400; echo
      return 0
    fi
    echo -n "."; sleep 2
  done
  echo
  echo "!! endpoint never came up. Last log lines:"
  eval "$logcmd" 2>&1 | tail -30
  return 1
}


case "$ACTION" in
  llamacpp) build_llamacpp; fetch_gguf; [ $SERVE = 1 ] && serve_llamacpp ;;
  llm)      build_llamacpp; fetch_llm_gguf; [ $SERVE = 1 ] && serve_llm ;;
  vllm)     [ $SERVE = 1 ] && serve_vllm || echo "nothing to do without --serve" ;;
  stop)     stop_all ;;
  *)        usage ;;
esac
