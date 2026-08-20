# Phase 2.0 gate — can a VLM run on THIS device alongside 20 streams?

**Verdict: PASS.** Cosmos Reason 2 2B runs locally on the AGX Orin under JetPack 7.2, answers
correctly about real warehouse frames, and costs the 20-stream pipeline **3.3%** throughput at the
Phase 2 design point. Both candidate serving stacks work. **Recommendation: llama.cpp.**

Measured 2026-08-16 on the reference AGX Orin 64GB — L4T R39.2.1 / JetPack 7.2.1-b49 / CUDA 13.2,
nvpmodel MAXN. Raw data in `bench/reasoning_*/`, latency JSON in `bench/vlm_latency*.json`.

---

## 1. What was actually at risk, and what happened

The plan named one top risk: the documented serving container targets **JetPack 6 / L4T r36**
and this device is **JetPack 7.2 / L4T R39.2**. Inspecting the model raised a second, independent
risk that the plan did not anticipate:

> `nvidia/Cosmos-Reason2-2B` is **`Qwen3VLForConditionalGeneration`** (`model_type: qwen3_vl`,
> `transformers_version: 4.57.0.dev0`), with DeepStack visual indexes and interleaved M-RoPE.

So the container had to clear two bars, not one. **I expected it to fail. It did not.** The
container ships `torch 2.10.0`, `vllm 0.19.0` and `transformers 4.57.3` — new enough for
`qwen3_vl` — and CUDA is available inside it on R39. Recording this plainly because the plan
budgeted real time for that failure and it did not happen.

llama.cpp also builds cleanly from source against the installed CUDA 13.2 for `sm_87`, and
detects `CUDA0: Orin (62877 MiB)`.

**Both paths work. The choice is now about cost, not viability.**

---

## 2. The decisive measurement: does reasoning break the hot path?

Method (`scripts/bench_reasoning.sh`): one continuously running pipeline, three phases —
**A** baseline with the VLM loaded but idle, **B** 30 reasoning requests back to back (the worst
realistic burst), **C** recovery. fps attributed by log-line offset. Verdict keyed on **phase B**,
never on an average, because an average over A+B+C hides exactly the dip being looked for.

`drop-frame-interval` (`dfi`) decouples analytics rate from ingest rate. The realtime target
scales with it: at `dfi=2` the decoder emits 15 fps/stream, so 300 fps aggregate **is** realtime.

### llama.cpp backend, 20 streams

| dfi | analytics | A baseline | B under reasoning | C recovery | target | margin at B |
|---|---|---|---|---|---|---|
| 0 | 30 fps | 941.1 | **521.8 (−44.6%)** | 941.9 | 600 | **0.87× ✗** |
| 2 | 15 fps | 523.5 | **506.4 (−3.3%)** | 522.6 | 300 | **1.69× ✓** |
| 3 | 10 fps | 340.1 | **327.2 (−3.8%)** | 337.0 | 200 | 1.64× ✓ |

**`dfi=0` fails.** At full 30 fps analytics the GPU is already saturated, so the VLM competes
directly and throughput falls 44.6% — below realtime. This is the number that would have sunk the
plan had the analytics rate not been decoupled.

**`dfi=2` and `dfi=3` both pass, and the margin is effectively identical (1.69× vs 1.64×).**
Since throughput gives no reason to prefer 10 fps, **`dfi=2` (15 fps) is recommended**: same
headroom, but objects move half as far between frames, which is easier on NvSORT's association.
The plan proposed `dfi=3` with `dfi=2` as a fallback; the measurement inverts that preference.

Recovery returns to baseline within 0.2% in every case — no leak, no thermal decay.

### vLLM backend, 20 streams, dfi=2

| | A | B | C | margin |
|---|---|---|---|---|
| vLLM | 525.7 | **458.0 (−12.9%)** | 505.6 (−3.8%) | 1.53× ✓ |

Passes, but costs **4× more hot-path throughput** than llama.cpp and does not fully recover.

---

## 3. Reasoning latency

| | TTFT | total | tok/s |
|---|---|---|---|
| llama.cpp, idle | 0.12 s | 0.83 s | 33.9 |
| llama.cpp, under 20-stream load (dfi=2) | 0.15 s | 1.75 s | 16.2 |
| vLLM, idle | 0.05 s | 0.72 s | 40.1 |
| vLLM, under 20-stream load (dfi=2) | 0.11 s | 1.55 s | 19.0 |

Request shape: 6 frames at 640px + prompt (~447 KB), 192 max tokens — the realistic
event-triggered shape from Phase 2.4, not a single frame and not a whole video.

Both roughly halve in decode rate under pipeline load, but **TTFT stays ~0.15 s**. For a cold
path that streams an explanation onto an already-alerted incident, sub-2-second total is
comfortably inside budget. Nothing an operator waits on.

### Memory

| | model weights | peak system RAM, 20 streams + reasoning |
|---|---|---|
| llama.cpp (BF16 GGUF + mmproj) | ~3.7–4.3 GB | **14.8 GB** |
| vLLM (BF16 safetensors) | 4.24 GiB (its own report) | **40.3 GB** |

Two independent measurements agree on ~4.2 GB of weights, which is a useful cross-check.
vLLM's extra 25 GB is KV-cache reservation from `--gpu-memory-utilization 0.45` (it allocated a
220,672-token cache — 27× more concurrency than a single-request cold path can use). That is
tunable and the gap would narrow; it was left at the documented default here so the comparison
reflects an untuned deployment of each.

**Jetson memory is unified.** `nvidia-smi` reports `[N/A]` on Tegra, so all memory figures come
from `/proc/meminfo`, and model footprint is measured as RAM-with-server-loaded minus
RAM-with-server-stopped — not from a delta taken while the model was already resident.

---

## 4. Does it actually SEE the frames?

The most dangerous failure here is silent. llama.cpp loads happily **without** the `mmproj`
vision projector and then answers as a text-only model — confidently describing a warehouse it
never saw. A leading prompt ("a detector flagged a missing vest — confirm?") makes agreement look
like observation. Latency numbers cannot detect this, so the endpoint was controlled:

| control | input | answer |
|---|---|---|
| neutral question, real frame | warehouse frame | *"a worker in a yellow hard hat and blue overalls… carrying a box… boxes on shelves and a forklift"* |
| literal detail, real frame | warehouse frame | *"The floor is dark blue, and there are two large orange and yellow objects… likely forklifts"* |
| **same leading prompt, blank grey image** | flat grey | **"VERDICT: rejected — There are no visible people in the provided frames."** |

The third row is the one that matters: given every cue to say "confirmed", the model refused
because the pixels did not support it. vLLM passes the same control ("the image is entirely blank,
so no person or vest can be detected"). **Both backends are genuinely grounded in the image.**

`scripts/setup_reasoning.sh` therefore treats `mmproj` as **required** and hard-fails without it,
rather than starting a server that would look healthy and be blind.

---

## 5. Recommendation

**Serve with llama.cpp** (`scripts/setup_reasoning.sh llamacpp --serve`), at `dfi=2`.

- The hot path is the product. 3.3% vs 12.9% impact decides this on its own.
- 14.8 GB vs 40.3 GB peak leaves room for the Phase 2.1–2.7 services on the same box.
- It recovers fully (−0.2% vs −3.8%).
- It builds from source against the installed CUDA, so it is not coupled to a container built
  for a different JetPack.

The cost is ~0.2 s of cold-path latency and ~3 tok/s — irrelevant for a background verifier that
an operator never waits on.

**vLLM remains a live option** and is the better choice if reasoning ever moves to the hot path,
becomes multi-tenant, or migrates to Thor where the real VSS blueprint runs. Its `0.05 s` idle
TTFT is genuinely better. Retest it with `--gpu-memory-utilization` tuned down before dismissing
the impact gap as inherent.

---

## 6. Carried forward

- **`dfi` corrects a premise in the plan.** `drop-frame-interval` drops frames *after* decode, so
  it buys **GPU** headroom, not **decoder** headroom — NVDEC still does full 30 fps work per
  stream. At `dfi=3` the pipeline runs at 99% NVDEC and 3% GPU. The lever works, but for a
  different reason than "everything downstream of the decoder slows down": what it frees is
  precisely the resource reasoning needs.
- **Track-ID stability at 15 fps: VERIFIED, `dfi=2` is safe.** Measured with
  `--track-stats` (8 streams, 130 s each). What matters is not the raw id count but the fraction
  of tracks that live long enough for `rules.py` to reach a verdict at all — a track shorter than
  the 15-frame debounce window never produces one and is pure churn.

  | dfi | unique ids | lifetime p50 | mean | **adjudicable (≥15 frames)** | ephemeral (≤2) |
  |---|---|---|---|---|---|
  | 0 | 1111 | 0.90 s | 1.50 s | **74.4%** | 8.2% |
  | 2 | 1124 | 0.77 s | 1.44 s | **73.6%** | 7.0% |
  | 3 | 1003 | 0.77 s | 1.61 s | 72.1% | 8.2% |

  `dfi=2` costs **0.8 percentage points** of adjudicable tracks against `dfi=0`. NvSORT's Kalman
  motion model absorbs the doubled inter-frame motion; the plan's concern does not materialise at
  15 fps. Lifetimes are reported in **seconds** deliberately — frames/id halves at `dfi=2` purely
  because there are half as many frames, which would make the slower rate look worse for free.

  **But `window_frames: 15` now spans 1.0 s of wall time instead of 0.5 s**, so compliance
  verdicts take twice as long to settle. That is a real behavioural change, not a bug: it is more
  temporal evidence per verdict, arriving later. Phase 2.1 should decide deliberately whether the
  debounce is specified in frames (current) or in seconds — the event model's "exactly one event
  per violation" guarantee depends on which.
- **The verdict format needs work.** One vLLM answer read *"VERDICT: rejected / REASON: …no
  hi-vis vest is visible"* — internally contradictory. Phase 2.4 must constrain the output
  (structured decoding or a validating parser) rather than trusting free text.
- **This is synthetic footage.** Reasoning quality on rendered warehouse video does not predict
  real-plant behaviour, and the `no-vest` precision claim in Phase 2.4's verification step must be
  measured on real imagery before it is quoted to anyone.

## 7. Reproducing

```bash
# on the Jetson
./scripts/setup_reasoning.sh llamacpp --serve          # build + fetch + serve on :8000/v1
python3 scripts/bench_reasoning.py --model Cosmos-Reason2-2B --frames 6 --repeat 5
./scripts/bench_reasoning.sh --model Cosmos-Reason2-2B --streams 20 --repeat 30 --dfi 2
```

`bench_reasoning.sh` refuses to start if another pipeline is already running, and reports
`NO-DETECTIONS` rather than a pass if the pipeline detected nothing — see the note below.

### A measurement failure worth not repeating

The first set of these numbers was wrong. The harness launched the pipeline with `nohup setsid`,
which moves it into a new session, so `$!` captured the transient parent and the cleanup `kill`
hit nothing. **Four 20-stream pipelines ended up running concurrently**, and each successive run
measured the contention: apparent baseline throughput fell 941 → 254 fps across runs, and one run
showed the pipeline getting *25% faster* while the VLM ran — which was an orphaned pipeline from
the previous run hitting its own `timeout` and dying mid-measurement. That very nearly went into
this document as a DVFS finding about the memory controller.

Fixed by launching in-session so `$!` is real, sweeping by name on exit, and **refusing to
benchmark at all if a pipeline is already running**. The same discipline as Phase 1's
zero-detection guard: a benchmark that cannot prove it had the machine to itself is not a
measurement.
