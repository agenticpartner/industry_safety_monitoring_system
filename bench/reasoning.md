# Phase 2.4 — VLM verification: does it actually improve precision?

**Yes — but only after two corrections, and the headline number was nearly reported wrong.**

Cosmos Reason 2 rejects ~a third of the incidents the CV pipeline reports, which is the right
order of magnitude for the "absence of evidence" false positives Phase 1 named as the weakest
part of the system. The hot path held realtime throughout.

**The important caveat, stated up front:** the first version of this layer produced verdicts that
were *fluent, schema-valid, internally consistent, and roughly half wrong*. Every automated check
passed. It was only caught by rendering the images the model was shown and looking at them (§3).
Read §3 before quoting any precision number from this document.

Measured 2026-08-17 on the reference AGX Orin 64GB. Service: `services/reasoning_service.py`.

---

## 1. Results

| | |
|---|---|
| Incidents verified | **40** (20 cameras, one pass) |
| Confirmed | 27 |
| **Rejected** | **13 (33%)** |
| Uncertain | 0 |
| **Verdicts adjudicated by eye** | **10/12 correct** with the corrected image policy (§3b) — was ~50% before |
| Median latency | 2.1 s idle, **3.7 s with a 20-stream pipeline live** |
| Pipeline fps with reasoning running | **459.3** vs 516 baseline (−11%), **1.53× over the 300 fps target** |

PPE violations alone: **21 incidents, 4–6 rejected depending on image set** — all of the form
"the person IS wearing a high-visibility vest". That is exactly the failure mode the inferred
`no-vest` rule was known to produce, and it is the number that justifies this layer.

Sample verdicts, all schema-constrained and traceable to an observation:

```
cam11 ppe_violation high CrossAisle -> REJECTED  person IS wearing a high-visibility vest.
                                                 A worker in a yellow safety vest and hard hat…
cam05 overcrowding  high AisleLeftOC-> CONFIRMED counted 4 people against a limit of 2.
cam09 overcrowding  high CrossAisleOC->REJECTED  counted 3 people, within the limit of 3.
cam07 ppe_violation high None       -> CONFIRMED person is not wearing a high-visibility vest.
```

---

## 2. The design change that made it work

The first implementation asked the VLM for the verdict directly. It returned **100% rejections**,
including an overcrowding incident where it reported 4 people against a limit of 2 — and its
reasons were internally contradictory: *"the area appears clear of people, with only three
individuals visible"*.

Two causes, both in the prompt, not the model:

1. **It primed rejection.** The prompt helpfully explained that the detector miscounts cones as
   people. Phase 2.0 had already established this model follows its prompt closely.
2. **It was asked for a policy judgement without the inputs.** "Does this area look crowded" is
   subjective — the occupancy limit was never in the prompt.

**Fix: the VLM answers perception questions; the verdict is computed in code.**

| | |
|---|---|
| VLM decides | `subject_is_person`, `wearing_high_vis_vest`, `wearing_hard_hat`, `people_visible`, `description` |
| Code decides | verdict, from those answers + the rule that fired + the zone's configured threshold |

A JSON schema constrains *shape* but not *coherence*. Taking the judgement back into code is what
made it reliable — and every verdict is now explainable from its inputs, and unit-testable without
a GPU (`tests/test_reasoning.py`, 25 cases).

---

## 3. Does the model answer about the CROP — and is it RIGHT?

Two different questions. The first was measured and answered yes; the second was **not measured
at first, and the answer turned out to be no**.

### 3a. Agreement between image sets (the wrong metric)

Same 21 PPE incidents, three image sets:

| Image set | Agrees with `both` | confirmed / rejected |
|---|---|---|
| `both` (4 context + 1 crop) | — | 17 / 4 |
| `crop` only | 19/21 (90%) | 15 / 6 |
| `context` only | 16/21 (76%) | 12 / 9 |

The natural reading — "the crop drives the verdict, so the design is fine" — **was wrong**.
Agreement between two configurations cannot detect that BOTH are inaccurate. Nothing here
establishes correctness.

### 3b. Correctness, by rendering the crops and adjudicating them (`verify_verdicts.py`)

The model was shown the exact crop and its verdict rendered next to it, then judged by eye. There
is no ground truth in this dataset, so a human looking at the image IS the ground truth.

**With `both` (context + crop) — roughly half the rejections were WRONG:**

| Camera | What is actually in the crop | Model said | |
|---|---|---|---|
| cam02 | a traffic cone and a box | "worker in yellow hard hat and hi-vis vest" | fabricated |
| cam09 | a fire extinguisher on a rack post | "worker in a light blue hi-vis vest" | fabricated |
| cam05 | worker in a teal shirt, **no vest** | "yellow safety vest" | **false negative** |
| cam08 | black top, grey trousers, **no vest** | "bright yellow hi-vis vest" | **false negative** |
| cam11 | pink overalls, **no hi-vis** | "yellow safety vest" | **false negative** |
| cam12 | teal shirt, **no vest** | "yellow safety vest" | **false negative** |

The pattern is unmistakable: it described a generic warehouse worker in a hi-vis vest almost
regardless of the crop — including twice when there was no person in the crop at all. **The
context frames were bleeding into the answer.**

**With `crop` only — 10 of 12 clearly correct**, and the descriptions become specific and
checkable rather than generic: "teal shirt, beige pants", "white coveralls walking away from the
camera", "bright orange hard hat and a dark shirt", "yellow hi-vis vest with reflective stripes".
cam09's fire extinguisher now returns `subject_is_person=no` → *"the tracked object is not a
person"*.

**So the default is per event type** (`--images auto`):

| Event type | Images | Why |
|---|---|---|
| `ppe_violation` | **crop only** | the question is about one subject; context frames cause confabulation |
| `overcrowding` | context frames | the question is "how many people are in this scene" — there is no subject, and the scene IS the evidence |
| `fire_alert` | context frames | same |

### 3c. The real root cause: `--image-min-tokens`

Everything in §3b was chasing a symptom. llama-server prints this **at load time**, and it was
scrolled past:

```
W load_hparams: Qwen-VL models require at minimum 1024 image tokens to function correctly
                on grounding tasks
W load_hparams: if you encounter problems with accuracy, try adding --image-min-tokens 1024
```

Cosmos Reason 2 **is** a Qwen3-VL model (`model_type: qwen3_vl`, established back in Phase 2.0),
and our PPE crops are small and tall — 448px wide — so they landed far below that floor. The
warning predicted exactly the symptoms observed: confident but wrong attribute calls, and objects
misidentified as people.

Adding `--image-min-tokens 1024` to the **2B**, changing nothing else:

| Camera | Before | After |
|---|---|---|
| cam02 | "a worker in a hi-vis vest" | **"not a person. A traffic cone with orange and white reflective stripes on a yellow base"** |
| cam09 | "worker in a light blue hi-vis vest" | **"not a person. A red fire extinguisher with a white label"** |
| cam07 | "a red high-visibility vest" (rejected) / "an apron" (confirmed) — flipped between runs | **"a pink apron"** — consistently not a vest |
| cam04, cam08 | "yellow hard hat" | **"orange hard hat"** — correct colour |

**Adjudicated by eye: 12/12 correct.** Every incident, including both non-persons and the
borderline apron that previously flipped verdicts between identical runs.

The cone case is the one that motivated this entire phase, and it had failed under every previous
configuration. It was never a model-capacity problem — it was a serving flag, and the serving
stack said so on startup.

**Lesson:** read the model server's load-time warnings. This cost several rounds of prompt
engineering, an image-set ablation, and an 8B download, all chasing a symptom of a documented
misconfiguration.

### 3d. What is still wrong

- **Non-persons are caught inconsistently.** cam09's fire extinguisher was correctly rejected as
  not-a-person; cam02's traffic cone was still described as a worker in a hi-vis vest. Right
  outcome, fabricated reason — the incident is rejected either way, but the stated reason is
  false, and an operator reading it would be misled.
Both issues listed here before — borderline garments flipping between runs, and hard-hat colours
reported wrongly — were **fixed by `--image-min-tokens 1024`** (§3c). They were symptoms of
under-tokenised images, not of model size.

### 3e. The methodological lesson

Every automated signal was green while half the verdicts were wrong: the request succeeded, the
JSON schema validated, the enum was respected, the reasons were fluent and internally consistent,
and the two configurations agreed 90% of the time. **None of that is correctness.** The bug was
only visible by rendering the image the model was shown and looking at it.

`tools/verify_verdicts.py` exists so that check is repeatable instead of a one-off.

## 4. A bug found by looking, not by testing

The crop was originally cut from the **clip** at a fixed `pre_roll` offset. That is wrong twice:

1. When an incident opens near the start of the source, the clip is clamped to `start=0`, so the
   incident sits at its raw offset, not at `pre_roll`.
2. `-ss` before `-i` with `-c copy` snaps the clip to the preceding keyframe, so the clip's true
   start drifts from the requested one by up to a GOP.

The result was a crop of empty floor while the model described a worker in detail. Every
automated check passed: the request succeeded, the schema validated, the verdict was well-formed.
**Only rendering the crop and looking at it revealed it.**

Fixed by cutting the crop from the **source** at the incident's own PTS, which removes both
errors. (And `-accurate_seek` is an INPUT option — placing it after `-i` makes ffmpeg reject the
whole command, which is how the fix failed the first time.)

---

## 5. Cold-path guarantees

- **One request in flight.** llama-server runs `--parallel 1`; the service is single-threaded and
  never queues more.
- **Circuit breaker** — 5 consecutive failures open it for 60 s. While open the service sleeps
  rather than hammering a sick endpoint, and incidents simply stay `unverified`.
- **Crash-safe claiming** — an incident is marked `reasoning` before the slow call, and
  `reclaim_stuck()` returns anything abandoned mid-flight on startup. The service is designed to
  be killable, so that is a normal path.
- **Transient failures return the incident to `new`** and are retried; only real outcomes are
  terminal.
- **Severity is never overwritten.** `severity` is what the CV rules and zones concluded;
  `vlm_verdict` is additive. Overwriting would destroy the ability to ask "how often does the VLM
  disagree with a high-severity detection?" — and VSS models it the same way, as its own filter
  on `video_analytics__get_incidents`.

---

## 5b. 2B vs 8B — and why 2B was chosen

Cosmos Reason 2 **8B** (Q8_0 weights + BF16 vision projector) was downloaded and benchmarked
because verdict instability looked like a model-capacity problem. It was not — it was
`--image-min-tokens` (§3c). With that fixed, the 2B is accurate and stable, and the 8B's cost is
no longer worth paying:

| | **2B (chosen)** | 8B |
|---|---|---|
| Memory (llama-server RSS) | **6.1 GB** | 11.2 GB |
| Median latency, idle | **3.0 s** | 7.3 s |
| Verdict stability, 3 identical runs | **39/40 (97%)** | not completed — decision made first |
| Accuracy adjudicated by eye | **12/12** | — |

The 8B run was stopped once the decision was clear rather than held for completeness; the two
numbers above are enough to justify the choice, and they are the real cost of the alternative.

**The 8B weights are still on the device** at `models/cosmos/gguf8b/` (9.2 GB). Reclaimable with
`rm -rf models/cosmos/gguf8b` if disk is ever needed — kept for now so the comparison can be
finished without a re-download.

**Standing configuration:** 2B, `--image-min-tokens 1024`, crop-only for PPE
(`./scripts/setup_reasoning.sh llamacpp --serve` — both are the defaults).

### Full-stack cost at 20 streams

With **everything** live — zones, events, clips and reasoning:

| | |
|---|---|
| Pipeline | **420.1 fps** = **1.40× over the 300 fps target** |
| Events | 4938 published, **0 dropped**, 0 probe errors |
| Incidents | 30, all with clips, 24 confirmed / 6 rejected |
| Integrity | all checks pass |

## 6. Carried forward

- **−11% hot-path cost, not the −3.3% Phase 2.0 predicted.** The gap is the service's own ffmpeg
  frame extraction (CPU) on top of the VLM's GPU work, running back-to-back over a 40-incident
  backlog. Still 1.53× over target. If it needs reducing: extract frames once at clip time rather
  than at reasoning time.
- **The cone false positives were not the ones rejected.** The demo's headline case — the PPE
  model calling traffic cones and wet-floor signs `human` on cam02 — did not surface, because the
  crop is of the track that OPENED the incident, and incidents merge many tracks (one had 130).
  A cone tracked later in the same incident is never examined. Per-track verification would catch
  it, at the cost of one VLM call per track instead of per incident.
- **`uncertain` never fired** across 40 incidents. Either the model is well-calibrated here or it
  is reluctant to abstain; on this synthetic footage it is not possible to tell which, and real
  plant imagery would be the honest test.
