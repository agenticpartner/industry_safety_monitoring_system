#!/usr/bin/env bash
# llama.cpp server entrypoint. Fetches the weights on first boot, then serves.
#
#   ROLE=vlm   Cosmos Reason 2 (vision) on :8000  — adjudicates incidents, called continuously
#   ROLE=llm   Nemotron Nano 9B  (text)   on :8001 — answers human questions, called rarely
#
# Both expose the same OpenAI-compatible surface, which is the whole point: neither
# services/reasoning_service.py nor services/agent.py ever learns what is behind it.
#
#   HF_TOKEN         optional; only needed if a repo is gated
#   WEIGHTS_DIR      where the GGUFs live (a volume, so a 12 GB pull happens once)
#   LLAMA_CTX_SIZE   context window, default 8192
#   LLAMA_NGL        layers on GPU, default 99 (all)
set -uo pipefail

ROLE="${ROLE:-vlm}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/weights}"
CTX="${LLAMA_CTX_SIZE:-8192}"
NGL="${LLAMA_NGL:-99}"

die() { printf "\033[31mERROR: %s\033[0m\n" "$1" >&2; exit 1; }
step() { printf "\n\033[1m==> %s\033[0m\n" "$1"; }

# hf_hub_download, not the CLI. The command was `huggingface-cli` and is now `hf`, and a version
# that has made the switch answers the old name with its help text and a non-zero exit — which
# reads as "the repo is wrong" rather than "the command was renamed". The Python API has been
# stable across all of that, resumes a partial download, and is what scripts/setup_reasoning.sh
# uses on the Jetson, so both platforms fetch weights the same way.
fetch() {
  local repo="$1" file="$2" dest="$3"
  if [ -f "${dest}/${file}" ]; then
    echo "    ${file} present ($(du -h "${dest}/${file}" | cut -f1))"
    return 0
  fi
  echo "    fetching ${file} from ${repo}"
  mkdir -p "$dest"
  REPO="$repo" FILE="$file" DEST="$dest" python3 - <<'PYHF' || die "could not fetch ${file} from ${repo}"
import os, sys
from huggingface_hub import hf_hub_download
path = hf_hub_download(os.environ["REPO"], os.environ["FILE"],
                       local_dir=os.environ["DEST"],
                       token=os.environ.get("HF_TOKEN") or None)
print("    ->", path, file=sys.stderr)
PYHF
}

case "$ROLE" in

  vlm)
    PORT="${PORT:-8000}"
    REPO="${VLM_REPO:-robertzty/Cosmos-Reason2-2B-GGUF}"
    DIR="${WEIGHTS_DIR}/cosmos"
    WEIGHTS="${DIR}/Cosmos-Reason2-2B-BF16-split-00001-of-00002.gguf"
    MMPROJ="${DIR}/mmproj-Cosmos-Reason2-2B-BF16.gguf"
    ALIAS="${VLM_ALIAS:-Cosmos-Reason2-2B}"

    step "weights"
    # The BF16 weights ship as a 2-part split. llama.cpp follows the split itself when handed
    # part 1, but BOTH files have to be on disk — given only part 1 it fails at load with a
    # missing-tensor error that does not mention the missing shard.
    fetch "$REPO" "Cosmos-Reason2-2B-BF16-split-00001-of-00002.gguf" "$DIR"
    fetch "$REPO" "Cosmos-Reason2-2B-BF16-split-00002-of-00002.gguf" "$DIR"
    fetch "$REPO" "mmproj-Cosmos-Reason2-2B-BF16.gguf" "$DIR"

    # Without --mmproj the server loads happily and answers as a TEXT-ONLY model — it will
    # confidently describe a warehouse it never saw. Silent, and exactly the kind of failure that
    # survives to a demo, so it is required rather than optional.
    [ -f "$MMPROJ" ] || die "no multimodal projector at ${MMPROJ}"

    # --image-min-tokens 1024: llama-server warns that Qwen-VL models need at least 1024 image
    # tokens for grounding, Cosmos Reason 2 IS a Qwen3-VL, and the PPE crops are small and tall
    # (448px wide) so they land far below that. Phase 2.4 saw exactly the predicted symptoms —
    # confident but wrong attribute calls, a traffic cone read as a worker.
    # --parallel 1: admission control. One reasoning request in flight by design; letting the
    # server fan out hands the GPU to the cold path exactly when a burst makes the hot path
    # matter most.
    step "serving ${ALIAS} on :${PORT}"
    exec llama-server \
        --model "$WEIGHTS" --mmproj "$MMPROJ" \
        --host 0.0.0.0 --port "$PORT" \
        -ngl "$NGL" --ctx-size "$CTX" --parallel 1 --no-warmup \
        --image-min-tokens "${IMAGE_MIN_TOKENS:-1024}" \
        --alias "$ALIAS"
    ;;

  llm)
    PORT="${PORT:-8001}"
    REPO="${LLM_REPO:-bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF}"
    DIR="${WEIGHTS_DIR}/nemotron"
    FILE="${LLM_FILE:-nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q5_K_M.gguf}"
    ALIAS="${LLM_ALIAS:-Nemotron-Nano-9B-v2}"

    step "weights"
    # Q5_K_M is the size/quality knee for this model: it answers as well as Q8 on the planner's
    # constrained JSON at ~2 GB less resident.
    fetch "$REPO" "$FILE" "$DIR"

    # No --mmproj: this is a text-only reasoning model, not a VLM.
    # --ctx-size 8192 holds the tool schema, the live enum vocabulary, a few turns of
    # conversation and a page of retrieved incidents.
    step "serving ${ALIAS} on :${PORT}"
    exec llama-server \
        --model "${DIR}/${FILE}" \
        --host 0.0.0.0 --port "$PORT" \
        -ngl "$NGL" --ctx-size "$CTX" --parallel 1 --no-warmup \
        --alias "$ALIAS"
    ;;

  *)
    die "ROLE must be vlm or llm, got '${ROLE}'"
    ;;
esac
