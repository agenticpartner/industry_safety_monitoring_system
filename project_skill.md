# Project skill — Industry Safety Operations Monitoring

**Read this before changing anything.** It is the compiled state of the system: what it does, how
it is put together, what has been measured, and the behaviours that were verified the hard way on
real hardware. Several entries contradict the SDK documentation. Where they disagree, trust this
file — every claim here was observed on a running device.

Companion material: `README.md` is the human setup guide. `.claude/skills/` holds 28 NVIDIA agent
skills (DeepStream, Jetson, VSS) with exact API and property references — read the relevant
skill's reference docs before generating pipeline code.

---

## 1. What this is

A multi-stream video analytics system for industrial safety monitoring. It watches camera feeds
for PPE compliance (helmet and high-visibility vest) and fire/smoke, turns detections into
incidents, cuts an evidence clip for each one, has a vision-language model adjudicate it, and
surfaces everything on an operator dashboard with a natural-language agent and Telegram alerts.

It runs entirely on one edge device. No cloud inference, no external API calls, no data leaving
the box — the vision model and the language model both run locally.

**Measured target: 20 concurrent 1080p30 H.265 streams in realtime**, tiled to an attached display
and restreamed over RTSP/WebRTC/HLS.

### Reference platform

Every number in this file was measured on this configuration:

| | |
|---|---|
| Board | Jetson AGX Orin Developer Kit, 64 GB, aarch64 |
| L4T / JetPack | R39.2.1 / nvidia-jetpack 7.2.1-b49 |
| DeepStream | 9.1.0 |
| TensorRT | 10.16.2.10 (CUDA 13.2) |
| GStreamer | 1.24.2 |
| Accelerators | GPU + 2× DLA + 2× PVA |
| Power mode | MAXN, clocks locked (`jetson_clocks`) |
| Media | NVIDIA MV3DT synthetic warehouse dataset, transcoded to 1080p30 H.265 |

`scripts/check_hardware.sh` probes all of this and writes `build/hardware.env`, so nothing
downstream needs to hardcode a board, a CUDA arch or a display number.

---

## 2. Architecture

### Video path (one process, system Python)

```
nvurisrcbin ×N ─→ nvstreammux ─→ nvinfer(ppe) ─→ nvinfer(fire) ─→ nvtracker
                                                                      │
                        ┌─────────────────────────────────────────────┘
                        ↓
                  nvdsanalytics ─→ nvmultistreamtiler ─→ nvdsosd ─┬─→ nv3dsink   (display)
                        │                                          └─→ nvv4l2h265enc
                        │                                                 → rtspclientsink
                        └── metadata probe → rules.py → events.py → Redis
```

The probe attaches to the **tracker**, the last per-stream element. This is not a style choice —
see §6, trap 1.

A second topology exists (`pipeline.topology: parallel`) that tees into both detectors and
recombines with `nvdsmetamux`. It only pays off when one model is DLA-resident; with both on GPU
it adds a synchronisation point for no gain. Serial is the default.

### Service path (separate processes, `build/venv-services`)

```
Redis stream ─→ event_service ─→ store.py (SQLite, WAL)
                                     │
                    ┌────────────────┼────────────────┬──────────────┐
                    ↓                ↓                ↓              ↓
             clip_service     reasoning_service   notify_service   api.py :8080
             (ffmpeg -c copy) (VLM :8000)        (Telegram)       ├─ dashboard
                                                                   ├─ agent (LLM :8001)
                                                                   └─ WebSocket feed
```

Every service is a separate process. Killing one must never touch another — verified by stopping
Redis under a running pipeline (§3).

### Why the pipeline and the services are split across two Python environments

The pipeline runs on **system Python**, because that is where `pyservicemaker` lives, and nothing
may be installed there. So `app/events.py` is deliberately dependency-free: it speaks RESP over a
plain socket rather than importing a Redis client. The services run in `build/venv-services` and
use the real client. Do not "simplify" this by adding a dependency to `app/`.

---

## 3. What works, with the number behind it

Every figure below has a script that produced it and a file in `bench/` that records it.

### Throughput

| Streams | fps | Target | Headroom | GPU |
|--:|--:|--:|--:|--:|
| 1 | 254.2 | 30 | 8.47× | 74% |
| 8 | 621.3 | 240 | 2.58× | 99% |
| 16 | 625.9 | 480 | 1.30× | 99% |
| **20** | **633.7** | **600** | **1.05×** | 99% |

Phase 1 configuration (`drop_frame_interval=0`, full 30 fps analytics). Throughput plateaus at
~630 fps, so the practical ceiling is ~21 cameras. **Every row is detection-verified** — a
zero-detection run is recorded as `NO-DETECTIONS`, never as a pass.

At the Phase 2 design point (`drop_frame_interval=2`, 15 fps analytics, zones + events + clips +
reasoning all live) the pipeline holds **509–519 fps against a 300 fps target — 1.53–1.73×**.
Dropping frames after decode buys GPU headroom rather than decoder headroom, which is what makes
the local reasoning layer affordable.

Sizing guidance: 20 streams at 1.05× is the *demonstrated maximum*, not the design point. A busier
shift, a hotter cabinet or a third model breaks it. **Size production at 16 cameras per Orin
(1.30×).**

### Component ceilings

- **Decode**: flat at ~1100 fps from N=4 — about 36 concurrent 1080p30 H.265 streams, above the
  22 in the AGX Orin datasheet. Never the constraint.
- **Inference** (`trtexec`, batch 20): PPE YOLO11n FP16 **929 inf/s**, fire YOLO26s FP16
  **508 inf/s**. Also not the constraint.
- **The tracker is the binding constraint**, and its cost scales with *people, not pixels*. The
  same pipeline reached 723.8 fps on traffic footage vs 633.7 on warehouse footage.

### Latency

| Stage | p50 | p90 |
|---|--:|--:|
| detection → visible on the dashboard | **0.27 s** | 0.56 s |
| detection → evidence clip ready | **2.11 s** | 3.36 s |
| detection → VLM verdict | 12.49 s | 17.47 s |

The VLM number is a **queue, not a latency**. Each verdict costs 5–10 s and they run serially by
design; a cold start produces ~31 incidents in 40 s, so the last one waits minutes. In steady
state incidents arrive as transitions and the adjudicator keeps up easily. Narrate this honestly:
**the alert is instant, the adjudication is not.**

### Correctness properties that are verified

- **2225 detection transitions from 6 cameras in 120 s became 6 incidents.** An event is a state
  *transition*, not a frame observation.
- **Incidents are reference-counted.** One incident folds many tracks and closes only when the
  last clears. Closing on the first clear produced 531 unmatched closes and 310 rows where there
  should have been 6.
- **Event publishing costs the hot path −0.9%** (514.2 → 509.6 fps at 20 streams), 0 drops.
  Stopping Redis mid-run leaves fps flat (515.6 / 517.1 / 508.2 across up/down/restored) with zero
  probe errors, and publishing resumes by itself.
- **Zone analytics cost nothing measurable** — 510 / 514 / 519 fps for base / zones / zones+events,
  all inside run-to-run noise.
- **Clip capture costs nothing measurable** (516.2 fps against a 510–519 baseline), and clips are
  30.0 fps while analytics ran at 15.
- **VLM verdicts: 12/12 correct by eye**, including two non-persons, after `--image-min-tokens
  1024` was set. 13 of 40 incidents rejected on a full run; hot path held 459.3 fps.
- **Two resident model servers cost nothing when idle** — 517.8 fps at 20 streams with both
  loaded, identical to baseline. Only inference competes.
- **Notification policy: 30 incidents → 4 notifications** (9 below the severity floor, 1
  VLM-rejected, 16 still awaiting a verdict). A channel that buzzes constantly gets muted, at
  which point it protects nobody.
- **Agent latency: 11.2 s idle, 23.8 s at 20 streams** (down from 105 s / 218 s before the schema
  fix in §6).

---

## 4. Hard constraints — settled, do not re-litigate

- **Use `pyservicemaker`, not `pyds`.** They are different bindings from the same SDK and only
  `pyservicemaker` is present on a default DeepStream 9.x install.
- **Use `nvinfer`, not `nvinferserver`.** No Triton backend is set up.
- **H.265 only for demo media.** 20 streams is 91% of the NVDEC 1080p30 H.265 ceiling. H.264 has
  a materially lower ceiling and will not hold 20.
- **Python venvs live in `build/`.** Never install into system Python — see §2.
- **The X display is auto-detected**, never assumed. On a Jetson running a desktop session the
  server is commonly on `:1`, and `nv3dsink` against the wrong number fails with "no display
  found". `scripts/env.sh` probes `/tmp/.X11-unix`; `DISPLAY_NUM` in `.env` pins it.
- **Credentials live in `.env`, never in `configs/`.** Those files are committed, and a token in
  one is in the history forever.

---

## 5. Settings that matter, with the cost of getting them wrong

| Setting | Wrong value | Right value | Cost of the wrong one |
|---|---|---|---|
| `nvmultistreamtiler compute-hw` | default (= VIC on Jetson) | `1` (GPU) | −33% throughput |
| `nvvideoconvert` → RGBA before `nvdsosd` | present | **omit** | a full 1080p convert per batch |
| tracker config | stock NvSORT | `configs/tracker_nvsort_tuned.yml` | **zero detections** |
| `pgie_fire.yml interval` | 5 or 11 | `2` | **fire is never detected at all** |
| `demo.yml rules.fire.min_confidence` | 0.40 | `0.35` | fire detected but **no alert raised** |
| `drop_frame_interval` (with reasoning) | 0 | `2` | pipeline drops **below realtime** under a VLM burst |
| `--image-min-tokens` (llama-server) | unset | `1024` | confidently wrong VLM attribute calls |
| `/no_think` (Nemotron system prompt) | absent | first token | empty planner output, 33.7 s → 7.2 s |

Rendering (tiler + OSD) costs ~3% on the GPU path. It is **not** worth removing.

### Two settings worth understanding rather than copying

**The fire `interval` is a method lesson.** It was tuned 5 → 11 for +20% throughput, measured on
warehouse footage that contains no fire. With nothing to detect, skipping frames looks free — you
cannot observe the recall cost of a setting when the class never occurs. Re-measured at 20 streams
with real fire spliced in (`scripts/splice_fire.sh`):

```
interval  fire detections  fps
0         570              312.0   (only 1.04× over target — too tight)
2          28              509.3   <- chosen
5           0              500.1
11          0              510.1   (previously shipped)
```

At 5 and 11 a ten-second fire passes through **completely undetected** while the benchmark reads
510 fps and looks healthy.

**Two thresholds, one decision.** `rules.fire.min_confidence` must never be stricter than the
detector's own `pre-cluster-threshold` in `configs/pgie_fire.yml` (0.35). At 0.40 the pipeline
printed `fire=23` in `--stats` and raised **zero** fire alerts: every detection landed in the
0.35–0.40 band, counted by the stats probe and dropped by the rule. It looked like a working
detector with a broken event path, and neither number alone showed the gap.

---

## 6. Field notes — verified on this build

These were each hit and diagnosed on a running device. Several contradict the SDK docs.

### DeepStream metadata

1. **`object_items` yields transient proxies.** `list(frame_meta.object_items)` and then reading
   attributes **segfaults** (verified by bisection). All metadata access must happen *inline*, in
   a single pass. The docs' "convert to list first if you need multiple iterations" advice is
   wrong here. Never `len()` them either.
2. **An object's `text_params` is not a standalone `osd.Text`.** `osd.TextParams` exposes
   `font_params`, `set_bg_clr`, `text_bg_clr`. `osd.Text` (for `display_meta`) exposes `font`,
   `set_bg_color`, `bg_color`. Mixing them raises `AttributeError`.
3. **A Python exception inside a probe aborts the process** — `terminate called`, SIGABRT. It does
   not propagate as a traceback. Always wrap `handle_metadata` in try/except.
4. **Attach probes to the last per-stream element**, i.e. the tracker. `nvmultistreamtiler`
   composites the batch into ONE frame, so a probe after it sees a single `frame_meta` with
   `source_id` always 0 — all 20 cameras collapse into stream 0 and per-stream state is silently
   wrong. Measured: probe on tracker = 20 frames/batch and 20 source_ids; probe on OSD = 1 and 1.
   It also makes the frame counter count *batches*, under-reporting fps by exactly N.
5. **Events carry a ONE-BASED `camera_id`** to match the OSD ("CAM 01") and `cam01.mp4`, while
   DeepStream's `source_id` is 0-based. Convert at the probe boundary.

### DeepStream elements

6. This install has **`nvdsosd`, not `nvosdbin`**.
7. The tracker's `libnvds_nvmultiobjecttracker.so` needs **`libmosquitto1`** installed, or it fails
   to `dlopen` with a misleading "Failed to initilaize low level lib".
8. **Every sink needs `async=0`** when a `tee` split or dynamic sources are in play, or the
   pipeline deadlocks in PAUSED with no video and no error.
9. Request pads use the template name: `("", "sink_%u")`, never `"sink_0"`.
10. Dynamic-axis ONNX **requires** `infer-dims=3;640;640` in the nvinfer config, or TensorRT fails
    with `setDimensions: Error Code 3`.
11. Custom parser structs must be zero-initialised — `NvDsInferObjectDetectionInfo obj{};` — or
    `rotation_angle` is garbage and boxes render tilted.
12. Parser and `cluster-mode` must match the model's actual output shape, confirmed at runtime by
    the parser's own log line:
    - ppe `{8, 8400}` → pre-NMS (v11) → `NvDsInferParseYoloPreNMS`, `cluster-mode=2`
    - fire `{300, 6}` → post-NMS (v26) → `NvDsInferParseYoloPostNMS`, `cluster-mode=4`
13. **`pipeline.wait()` cannot be interrupted.** It blocks in C++, so SIGINT/SIGTERM are ignored
    and even a timer-thread `pipeline.stop()` does not unblock it. Use `--duration` to bound a run
    and `timeout --signal=KILL` as the exit path.

### The tracker trap

14. **The stock `config_tracker_NvSORT.yml` emits ZERO objects.** It sets
    `minTrackerConfidence: 0.8216`; NvSORT has no visual tracker, so its per-target confidence
    comes from Kalman + IoU association alone and never reaches that. Every target stays in shadow
    mode and nothing is output — while the pipeline looks healthy and runs *faster* than a working
    one, because it is doing no work. This briefly made a broken configuration the best-performing
    entry in the benchmark.

    Use `configs/tracker_nvsort_tuned.yml` (`minTrackerConfidence: 0.40`, `probationAge: 2`,
    `minIouDiff4NewTarget: 0.45`). It matches NvDCF's detection counts to within 0.2% at ~1.3× the
    speed.

    **Rule: never accept a throughput number without proof the pipeline detected something.**

### Zone analytics

15. **`pyservicemaker` reads nvdsanalytics metadata fine** — no `pyds` gap here.
    `obj.nvdsanalytics_obj_items` → `as_nvdsanalytics_obj()` → `roi_status`, `lc_status`,
    `oc_status`; `frame_meta.nvdsanalytics_frame_items` → `as_nvdsanalytics_frame()`. Read them
    inline like everything else.
16. **Object `oc_status` is a LIST, frame `oc_status` is a DICT `{zone: bool}`.** Iterating the
    dict yields keys, so treating it like the list marks every configured zone as overcrowded.
17. **nvdsanalytics takes ONE config with per-stream sections** (`[roi-filtering-stream-0]`), not
    one file per camera. The section suffix is the 0-based source index.
18. **Only one `[overcrowding-stream-N]` section per stream** — INI keys must be unique, so
    several zones share one section and therefore one `object-threshold`. The generator warns
    rather than silently dropping.

### The incident store

19. **`merge_window_s` is a linger period, not a length cap.** It absorbs track-id churn (~26% of
    tracks are too short to adjudicate), so it merges into open incidents *always* and reopens
    ones closed within the window. Bounding by incident start time instead let a 37 s incident
    spawn a second open incident alongside itself.
20. **A merged incident goes silent, so long-open ones are RE-RAISED** (`store.raise_stale`,
    `realert_after_s: 480`). Continuous violation merges rather than reopening — correct, one
    person violating for an hour is one incident — but on a 20-camera run 19 PPE incidents sat
    open for 82 minutes having absorbed ~53,000 detections with nothing raised after the first
    minute. Every fresh alert was fire, because fire is the only thing that starts and stops.
21. **The sweep must run BEFORE the empty-read `continue`.** In steady state almost nothing
    transitions, so `xreadgroup` returns empty most of the time — placed after that `continue`,
    the sweep ran only while events were arriving, which is exactly when it is not needed.
22. **Commit after every DML, even when it matched no rows.** Python's `sqlite3` opens a
    transaction on any DML and holds it until commit; `if rowcount: db.commit()` left an empty
    write transaction open for the life of the process, holding a RESERVED lock. Every other
    writer then failed with "database is locked".
23. **A transient store error is not a malformed entry.** The event service acks what it cannot
    process, so classifying `OperationalError` as "bad" ACKED AND DISCARDED live events during the
    lock window. Transient errors now retry without acking.
24. **`store.connect()` sets `row_factory` itself.** A caller that forgets turns `dict(row)` into
    "dictionary update sequence element #0 has length 32", an error a long way from its cause.
25. **sqlite3 connections are thread-bound.** Opening one on the event loop and passing it into
    `asyncio.to_thread` raises "SQLite objects created in a thread can only be used in that same
    thread". Open inside the worker.
26. **Don't write from a read path.** `sync()` rebuilds the FTS index and WAL allows one writer —
    calling it on a plain listing while the event service is writing gives "database is locked".

### Evidence clips

27. **`nvurisrcbin` smart-record does NOT work on `file://` sources — only RTSP.** The failure is
    **silent**: `start_recording()` returns a session id, raises nothing, and no file is written.
    Verified working on RTSP. File mode therefore cuts the clip from the source with
    `ffmpeg -c copy` — stream copy, no re-encode, no GPU.
28. **Use `buffer_pts`, not `frame_number` or wall-clock, to locate the moment.** `buffer_pts` is
    SOURCE time and survives `drop-frame-interval` (measured: 5.2 s wall = 10.3 s source at
    dfi=2). `frame_number` counts DELIVERED frames, so it runs at the analytics rate. Wall-clock
    is meaningless in file mode because the pipeline is not paced to realtime.
29. **Retention deletes the FILE and keeps the ROW** (`clip_state='expired'`) — the incident
    happened even if the evidence aged out.

### The vision model

30. **The VLM answers PERCEPTION questions; the verdict is computed in code.** Asking the model
    for the verdict directly produced 100% rejections and self-contradictory reasons ("the area
    appears clear of people, with only three individuals visible"). A JSON schema constrains
    shape, not coherence.
31. **Never tell the prompt that the detector is unreliable.** That primes rejection. Neutral
    prompts that only ask what is visible.
32. **Never ask for a policy judgement without its inputs.** "Does this look crowded" is
    subjective; the occupancy threshold comes from `zones.yml` and the comparison happens in code.
33. **The subject crop drives the PPE verdict** — crop-only agrees with the default 19/21,
    context-only only 16/21, and context-only rejects twice as often because the model finds
    *somebody* in a vest and answers about the wrong person.
34. **Cut the crop from the SOURCE at the incident PTS, not from the clip.** The clip's start is
    clamped when an incident opens early AND snaps to a keyframe, so a fixed `pre_roll` seek lands
    in the wrong place. This produced a crop of empty floor that the model still described in
    confident detail — every automated check passed; only looking at the image caught it.
35. **`-accurate_seek` is an INPUT option.** After `-i`, ffmpeg rejects the whole command.
36. **`--image-min-tokens 1024` is REQUIRED.** llama-server warns at load time that Qwen-VL models
    need ≥1024 image tokens for grounding, and Cosmos Reason 2 is a Qwen3-VL model; 448px crops
    fall far below it. Without the flag the model called a traffic cone "a worker in a hi-vis
    vest", reported orange hard hats as yellow, and flipped the same garment between "apron" and
    "vest" on identical input. With it: 12/12 verdicts correct by eye. **Read the model server's
    load-time warnings** — this cost several rounds of prompt engineering and an 8B download, all
    chasing a symptom of a documented misconfiguration.
37. **A VLM without its `mmproj` projector loads happily and is blind** — it answers from the
    prompt and will describe a warehouse it never saw. `setup_reasoning.sh` hard-fails without it.
    Control for this by sending a **blank image with a leading prompt**: a grounded model says "no
    people visible", a blind one agrees with you.
38. **The VLM escalates hazards it was not asked about**, and prose is not a signal. "A large fire
    is engulfing a cardboard box on the floor" appeared inside a *PPE* incident's `description`,
    where nothing could act on it. Every schema now asks `fire_or_smoke_visible` and
    `hazard_visible` + `hazard_description` as their own fields, and `escalation_for()` decides in
    code. **Never grep the description for "fire"** — that is a hidden list of anticipated hazards,
    and it matches "no fire visible" too.
39. **An escalated alert must never escalate.** The first run produced `hazard_alert from
    fire_alert` (a fire IS a hazard) then `fire_alert from hazard_alert` (that hazard IS a fire) —
    one new incident per cycle, each with its own clip and VLM call, forever. Escalation is
    allowed only from detector-originated types.
40. **State the FINDING first in `vlm_reason`, never the bare description.** A clip carries
    pre-roll before the event, so the model's one sentence anchors on the calm opening frames:
    confirmed fires read as "a worker in a yellow hard hat walks through an aisle" on every
    human-readable surface while `fire_or_smoke_visible` was correctly `yes`. The verdict was
    right and everything a person could see said the opposite.

### The agent

41. **`/no_think` at the start of the system prompt is REQUIRED for Nemotron Nano v2.** It is a
    reasoning model and its thinking tokens count against `max_tokens`. The OpenAI-style
    `chat_template_kwargs.thinking` and `reasoning_effort` switches are **silently ignored** by
    this build. `/no_think` cuts completion tokens 340–564 → 91–94 and latency 33.7 s → 7.2 s.
    Without it the planner returns `finish_reason: length` with EMPTY content.
42. **An unbounded string in a JSON schema never terminates under grammar-constrained decoding.**
    `text` was `{"type": "string"}`; the planner wrote a correct plan then padded that one field to
    `max_tokens` on EVERY question — 99 of the agent's 105 s. Same prompt without the schema: 1.5 s
    and 29 tokens. Give every free-text field a `maxLength`, set `additionalProperties: false`,
    and keep `max_tokens` tight: **unused budget is free only while generation terminates.**
    `tests/test_agent.py` asserts this statically because it is invisible at runtime.
43. **A soft fallback hides the failure it protects against.** `ask()` falls back to full-text
    search when planning throws, so the agent always answered and the only symptom was latency —
    for a whole phase the plan `{'tool': 'search_incidents', 'text': <the whole question>}` was
    read as a weak planner when it is the fallback's literal shape. Always surface `plan_error`.
44. **Vocabulary for the planner is built from the DATABASE at runtime**, never hardcoded. Keyword
    regexes (`helmet` → `ppe_violation`) are a hidden list of anticipated questions.
45. **A record's `type` is a fact; its `reason` is a description of a few sampled frames.** They
    disagree routinely and it is not a contradiction. Asked to show the fire clip, the agent read
    a description that never mentioned fire and answered "the clip of the fire incident is not
    available" — while quoting that exact clip. The type wins.
46. **The store counts incidents, not people.** There is no headcount denominator, so "what
    percentage of people were not wearing a helmet" has no answer; `by_label` gives the breakdown
    that does exist. Saying which is which beats inventing a rate.
47. **The model narrates, the database counts.** Aggregate answers carry a per-camera × verdict
    table built from SQL, because the LLM reliably narrates three cameras out of nine.

### Dashboard and API

48. **`/pipeline/status` reports what is RUNNING, not what `demo.yml` says.** It parses the live
    process's argv. Reporting the config value made the header read "1 streams" while 12 ran —
    found only by screenshotting the page.
49. **Browsers cannot play RTSP.** mediamtx republishes as WebRTC (`:8889`, sub-second) and HLS
    (`:8888`, the iOS-Safari fallback). `webrtcAdditionalHosts: [LANIP]` is required or a remote
    viewer negotiates against loopback and never gets a frame.
50. **A black video box has three causes with three different fixes** — mediamtx down, mediamtx up
    with nobody publishing, or the viewer's own WebRTC. `GET /live/status` separates them. The
    middle one is the trap: `POST /rtsp/on` starts the *server*, while the *publisher* is
    `--rtsp-out`, a pipeline **start-time** flag. An encoder cannot be attached to a running
    DeepStream graph, so enabling live view genuinely restarts the pipeline.
51. **mediamtx's control API (`:9997`) is localhost-only by default.** Curling it from a laptop
    returns `{"status":"error","error":"authentication error"}`, which looks like a broken config
    and is not.
52. **Browsers cannot play H.265, and evidence clips inherit it.** Chrome needs a platform hardware
    decoder and Firefox refuses outright, so every clip sat at `readyState 0` showing a black
    rectangle — indistinguishable from a missing clip. `/clips/{id}` transcodes to H.264 lazily and
    caches beside the original: the capture path stays free, and only clips somebody opens are
    paid for. ~1.9 s for a 12 s clip.
53. **The alarm fires before the clip exists.** Alert visible at 0.27 s, clip cut at 2–3 s, so the
    event object captured by `raiseAlarm()` never has a `clip_url`. `openClip()` re-resolves the
    incident and waits for a `pending` clip in the dialog.
54. **Clips honour HTTP Range** (`206` + `Content-Range`, `416` past EOF). Without it a browser
    `<video>` scrub bar is dead and every seek re-downloads the file.
55. **`/analytics/calendar` reports incidents, not people.** A "how many people had PPE violations"
    figure was tried as `SUM(hits)` and came out at **23,122 for 20 incidents**, because `hits`
    counts every re-observation of the same situation.
56. **Never verify live media with a virtual time budget.** It fast-forwards the page's timers, so
    the page screenshots before a single WebRTC frame has arrived — a perfectly working stream
    photographs as a black box. Drive a real browser on a real clock and probe
    `videoWidth`/`readyState` instead of trusting pixels.
57. **Calendar days are LOCAL dates computed in SQL** (`date(ts,'unixepoch','localtime')`) — a
    shift starting at 08:00 must not straddle two cells because the server thinks in UTC.
58. **The donut caps at five zones + Other.** Past that the validated hues stop being separable for
    colour-blind readers. A single zone at 100% is drawn as a `<circle>`, because a full-circle arc
    degenerates to a zero-length path and the slice vanishes.

### Notifications

59. **At-most-once.** `notify_state` is claimed *before* the network call, and a row left in
    `sending` by a crash resolves to `failed`, never retried: a duplicate alert is worse than a
    missing one, because the channel is only trustworthy if a message in it means something new.
60. **Don't select one pending row at a time.** The first version did, and an incident awaiting a
    VLM verdict was re-selected forever while every other pending incident starved behind it —
    `--once` never terminated. It works a batch and defers waiting rows in memory.
61. **Telegram will not preview HEVC** — it arrives as an unplayable attachment. It reuses the
    API's `browser_playable()` cache so a clip watched in the dashboard and one pushed to Telegram
    are the same converted file.

### Operating the device

62. **Never `pkill -f <pattern>` when the pattern also matches your own command line** — over SSH
    it kills the session itself (exit 255). Bracket-matching (`[l]lama-server`) is **not**
    sufficient: it only stops pkill matching *itself*, while any ancestor command line that
    mentions the pattern still matches. Use `pkill -x <name>` to match the process name exactly,
    or `ps -eo args | grep "[s]weep.sh"` and kill by PID.
63. **Never launch a process you intend to kill with `nohup setsid ... &`.** `setsid` moves it to
    a new session, so `$!` captures the transient parent and your cleanup `kill` hits nothing.
    This left **four 20-stream pipelines running concurrently** during a benchmark; each
    successive run measured the contention and apparent throughput fell 941 → 254 fps. Detach the
    *outermost* script, then launch children with a plain `&`. Any benchmark that can be run twice
    must also **refuse to start if a previous instance is still alive.**
64. **Long jobs must be detached**, not held open over SSH: `nohup setsid bash scripts/... > log
    2>&1 < /dev/null &`. A benchmark held open by an SSH session dies when the connection blips.
65. **`~/.bashrc` exports are invisible to `ssh host 'cmd'`.** Ubuntu's default `.bashrc` returns
    early for non-interactive shells. Secrets belong in `.env`, sourced explicitly via
    `load_env()` in `scripts/env.sh`.
66. **uvicorn does not hot-reload without `--reload`.** Restart the services after editing.

---

## 7. DLA policy

**All-or-nothing.** A model goes on DLA only if it compiles with `--useDLACore=N` and
`--reportCapabilityDLA` and **without** `--allowGPUFallback`. A graph split between DLA and GPU
ping-pongs tensors at every subgraph boundary and is typically slower than pure GPU while
appearing to "use DLA". A build that only succeeds with `--allowGPUFallback` is a **failed**
qualification.

DLA requires static batch shapes, so DLA engines are per-batch-size; GPU engines are dynamic.

Both models currently run on GPU, which met the target on its own. `scripts/qualify_dla.sh`
implements the gate and is ready to run.

---

## 8. Where the streams come from

Camera sources are served **from a separate machine** (`scripts/serve_rtsp_sources.sh`), never from
the Jetson. Production cameras are separate network devices; generating 20 streams on the box that
is also running inference burns CPU alongside it and misrepresents the load. The Jetson's own
mediamtx (`scripts/serve_rtsp.sh`) publishes only the demo's tiled **output** — one NVENC session
on dedicated encoder silicon, which does not compete with NVDEC or the GPU.

**Benchmark in `file` mode, demo in `rtsp` mode.** Network jitter and buffering add variance that
is not the Jetson's compute ceiling, so capacity claims come from the file sweep.

---

## 9. File map

### `app/` — the pipeline (system Python)

| File | What it does |
|---|---|
| `safety_pipeline.py` | Builds and runs the DeepStream graph: sources → mux → PPE → fire → tracker → zones → tiler → OSD → sinks. Owns the metadata probe. |
| `rules.py` | Turns raw detections into compliance state. Debounces per-track verdicts, latches fire alerts, counts violations per camera. No I/O — pure logic, unit-tested. |
| `events.py` | The event vocabulary and publisher. Detects state **transitions**, publishes to Redis fire-and-forget. Deliberately dependency-free. |

### `services/` — everything after the pipeline (`build/venv-services`)

| File | What it does |
|---|---|
| `store.py` | The SQLite incident store and its state machine. Reference counting, merging, re-raising. Schema and migrations live here. |
| `event_service.py` | Consumes the Redis stream, drives `store.py`, runs the stale-incident sweep. |
| `clip_service.py` | Cuts an evidence clip per incident from the source at the incident PTS. Stream-copy. Enforces the disk budget. |
| `reasoning_service.py` | The VLM layer. Asks perception questions, computes the verdict in code, escalates fire and hazards. |
| `agent.py` | Question answering: plans a retrieval tool, runs it, writes a grounded answer. Tool names mirror the NVIDIA VSS MCP server. |
| `search_service.py` | Full-text and structured search, plus the aggregate counts the agent and dashboard quote. |
| `api.py` | REST + WebSocket control plane on :8080. Serves the dashboard, incidents, clips (Range + H.264 transcode), analytics, agent, pipeline control. |
| `notify_service.py` | Telegram alerts with the clip attached, under a policy for what is worth interrupting someone for. |
| `media.py` | H.265 → H.264 on first request, cached. |
| `metrics.py` | Parses `tegrastats` into a ring buffer for the dashboard's system chart. |

### `tools/`

| File | What it does |
|---|---|
| `inspect_db.py` | Checks the store's invariants, exits non-zero on violation. Row counts alone cannot tell you the state machine is sound. |
| `verify_verdicts.py` | Renders what the VLM actually saw per incident, so verdicts can be checked by eye. |

### `scripts/`

**Setup** — `check_hardware.sh` (probe), `setup.sh` (install everything), `env.sh` (shared
settings, source don't execute), `setup_reasoning.sh` (build llama.cpp, fetch and serve both
models), `export_models.py` (weights → ONNX), `build_engines.sh` (ONNX → TensorRT + throughput),
`make_streams.sh` (build the demo media set), `make_zones.py` (zone geometry → analytics config,
`--preview` renders polygons on a real frame), `splice_fire.sh` (splice real fire into a camera).

**Running** — `demo_up.sh` (**the one you want**: whole demo up clean, or `--down`),
`run_services.sh` (start/stop/status/reset), `run_demo.sh` (local display instead of dashboard),
`serve_rtsp.sh` (mediamtx for the output), `serve_rtsp_sources.sh` (runs elsewhere, serves camera
sources), `telegram_setup.sh` (find the chat id, send a probe).

**Measurement** — `sweep.sh` (1→20 throughput; refuses zero-detection rows), `decode_sweep.sh`
(NVDEC ceiling), `bench_reasoning.sh`/`.py` (what reasoning costs), `bench_vlm_compare.sh` (2B vs
8B), `measure_alert_latency.py` (end to end), `diagnose_bottleneck.sh` (ablation),
`qualify_dla.sh` (DLA gate), `compare_verdicts.py` (agreement between VLM configs).

### `configs/`

| File | Controls |
|---|---|
| `demo.yml` | Stream count, source mode, frame intervals, rule thresholds, rendering, sinks. **The main dial.** |
| `services.yml` | Redis, the store, clip retention, the reasoning endpoint, the notification policy. |
| `pgie_ppe.yml`, `pgie_fire.yml` | The two detectors: engine, precision, batch, dims, parser, clustering. |
| `tracker_nvsort_tuned.yml` | The tracker. The stock config outputs nothing — see §6 trap 14. |
| `analytics/zones.yml` | Zone geometry per camera, normalised 0–1. |
| `analytics/analytics.txt` | Generated from the above — do not edit by hand. |
| `metamux.txt` | Metadata muxer config for the parallel-inference topology. |

### `dashboard/`

`index.html` is the operator UI: KPIs, live WebRTC wall, system chart, incident calendar, zone
breakdown, per-camera counts, incident feed, fire alarm, agent. One file, no build step, no
external requests. `system.html` is the developer reference behind the brain button — hardware,
models, engine throughput, capacity sweep, latencies, service map, pipeline diagram. Deliberately
**static**: every figure is a measurement with a script behind it, and mixing live values in would
leave a reader unsure which numbers are history and which are now. Update it when a measurement
changes.

Design constraints worth preserving: navy + yellow with **yellow as an accent only** (it is the
highest-contrast element at 8.6:1 on the card navy, so it marks what matters and never fills an
area); the five chart series are validated for lightness band, chroma floor, CVD separation and
contrast — re-run the validator after any theme change, because that is exactly when a palette
silently stops being legible; `prefers-reduced-motion` disables every animation, since a
full-screen red flash is precisely the effect that can harm.

---

## 10. Reading order, if you are new to it

1. This file — the constraints and the field notes.
2. `app/safety_pipeline.py` — how the video path is assembled.
3. `app/rules.py` and `app/events.py` — how pixels become incidents.
4. `services/store.py` — what an incident actually is.
5. `services/api.py` — everything the UI can ask for.

---

## 11. Future improvements

Directions the system is built to grow in. Each is a place where the groundwork already exists.

**Detection quality.** The PPE model has no `no-vest` class, so a vest violation is inferred —
"a person with no overlapping vest box". The rendering already distinguishes evidence strength
(`NO HELMET` upper-case and definite, `no vest?` lower-case and hedged) so an operator can tell
them apart. Fine-tuning on a dataset that labels `no-vest` directly would turn the inferred signal
into a trained one. Helmet detection is already direct and trustworthy.

**Fire recall.** `interval=2` detects fire with margin to improve — 28 detections on a spliced
ten-second event. Lowering the interval, or adding fire-specific footage to the detector's
training mix, raises recall further. `scripts/splice_fire.sh` makes this measurable.

**Real-world footage.** Capacity and accuracy are measured on the MV3DT synthetic warehouse
dataset. Running the same sweeps against real plant footage — different lighting, motion blur,
occlusion — would calibrate what the numbers mean in a specific site.

**DLA offload.** Both models run on GPU, which met the target on its own. `scripts/qualify_dla.sh`
implements the all-or-nothing admission gate and is ready to run; a model that passes frees GPU
budget for a third detector or a higher stream count. The `parallel` topology in `demo.yml`
exists for exactly this case.

**Dynamic stream management.** The stream count is set when the pipeline starts, and
`POST /streams` persists a new value and says a restart is required — which beats a 501 or a
silent no-op. `nvmultiurisrcbin` is the documented path to adding and removing cameras on a live
graph.

**Service supervision.** `scripts/run_services.sh` gives full start/stop/status/reset control and
is the right shape for development. systemd units with restart-on-failure are the natural step for
unattended deployment, and the services are already separate processes with clean lifecycles,
which is the hard part.

**Access control.** The API and dashboard are designed for a trusted LAN and carry no
authentication. Adding auth and TLS at the API boundary extends the same system to wider network
exposure. Worth doing before exposing :8080 beyond a controlled network.

**Reasoning throughput.** VLM adjudication is serial by design — `--parallel 1` is what keeps the
hot-path cost at 3.3%. vLLM is faster per request (TTFT 0.05 s vs llama.cpp's 0.12 s idle) and
becomes the better choice if reasoning ever moves onto the hot path or onto newer silicon, at the
cost of 40.3 GB peak RAM against 14.8 GB. The endpoint is OpenAI-compatible, so
`services/reasoning_service.py` never learns which backend is behind it.

**Cross-camera tracking.** Following a person between cameras needs re-ID embeddings, i.e. NvDCF
or NvDeepSORT — precisely the element that cost the most in the capacity sweep, so the budget
needs re-measuring alongside it. The MV3DT dataset ships `camInfo/*.yml` calibration and a BEV
map, and `.claude/skills/deepstream-run-mv3dt/` covers the multi-view tracking pipeline.

**Track stability at 15 fps.** `drop_frame_interval=2` was chosen partly because objects move half
as far between frames as at 10 fps, which is easier on NvSORT's IoU association and therefore on
the track-ID stability that compliance debouncing depends on. Measuring that directly would
confirm the reasoning.
