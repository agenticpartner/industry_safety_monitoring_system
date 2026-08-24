# Industry Safety Operations Monitoring System

Real-time PPE compliance and fire/smoke detection across **20 concurrent 1080p30 camera streams**
on a single NVIDIA Jetson AGX Orin. Detection, tracking, zone analytics, evidence clips,
vision-language verification, a natural-language agent and phone alerts — all running on the edge
device, with nothing leaving the box.

![Operator dashboard — 20 cameras live, incident feed, KPIs](docs/images/dashboard.png)

The operator dashboard: KPI tiles, the live 20-camera WebRTC wall with detections and zone
overlays, the incident feed, and system utilisation — NVDEC at 99%, GPU at 92%.

---

## What it does

A camera sees a worker without a helmet. Within a second that violation is an incident on the
operator dashboard. Two seconds later an evidence clip is cut from the source video. A local
vision-language model then looks at the clip and confirms or rejects the finding — and if it spots
a fire nobody asked it about, it raises that as its own alert. If the incident clears the
notification policy, it arrives on a phone with the video attached.

| Layer | What happens |
|---|---|
| **Detect** | Two YOLO detectors per frame: PPE (helmet / vest / person) and fire/smoke |
| **Track** | NvSORT assigns stable per-person IDs so a single bad frame cannot flip a verdict |
| **Reason about zones** | Per-camera polygons — a missing vest is `medium` in a walkway and `high` in a forklift route |
| **Turn into incidents** | An event is a state *transition*, not a frame observation. 2225 observations → 6 incidents |
| **Capture evidence** | An H.265 clip cut from the source at the incident timestamp, no re-encode |
| **Verify** | A local VLM answers perception questions; the verdict is computed in code |
| **Alert** | Dashboard toast and alarm, plus Telegram with the clip — under a policy for what is worth interrupting someone for |
| **Ask** | A local LLM answers questions over the incident database, grounded in SQL |

### An alert, end to end

<table>
<tr>
<td width="50%"><img src="docs/images/fire-alarm.png" alt="Full-viewport fire alarm"></td>
<td width="50%"><img src="docs/images/evidence-clip.png" alt="Evidence clip with the VLM finding"></td>
</tr>
<tr>
<td><b>Fire raises a full-viewport alarm</b> with a Web Audio siren — no asset to fetch. Audio
needs a user gesture, hence the explicit sound toggle, and <code>prefers-reduced-motion</code>
disables every animation, because a full-screen red flash is exactly the effect that can harm.</td>
<td><b>The evidence clip, with the model's finding first.</b> Cut from the source at the incident
timestamp and transcoded to H.264 on demand, because browsers cannot decode H.265. The operator
can confirm the violation or mark it a false positive.</td>
</tr>
</table>

### Measured, not estimated

| | |
|---|---|
| Throughput at 20 streams | **633.7 fps** against a 600 fps target (1.05×) |
| With the full Phase 2 stack live | 509–519 fps against a 300 fps target (1.53–1.73×) |
| Detection → visible on the dashboard | **0.27 s** median |
| Detection → evidence clip ready | **2.11 s** median |
| VLM verdict accuracy | 12/12 correct by eye, including two non-persons |
| Cost of the event layer on the hot path | −0.9% |
| Cost of zone analytics | none measurable |
| Notification policy in practice | 30 incidents → 4 phone notifications |

Every figure has a script behind it and a file in [`bench/`](bench/) that records it. The
`/system` page in the dashboard presents the same numbers with the architecture.

**Sizing note:** 20 streams at 1.05× is the demonstrated maximum, not the design point. For
production, size at **16 cameras per Orin** (1.30× headroom).

### On live RTSP cameras

The figures above are file mode, run flat out — a ceiling. Live cameras pace the pipeline, so the
question is whether it keeps up. Measured on 20 cameras over the LAN:

| | |
|---|---|
| Detection → visible on the dashboard | **0.12 s** median (0.19 s p90) |
| Detection → evidence clip ready | 11.1 s median |
| VLM verdict | 22.9 s median, of which 6.7 s is inference |
| Capacity, same configuration | **484.6 fps** against 300 needed — **1.6×** |
| GPU, steady state | ~30% |

**Realtime, with 1.6× headroom.** The analytics rate on this rig is ~190 fps against the ~290 the
cameras send, and the gap is ingest rather than compute — 70% of the GPU is idle. Twenty TCP
streams arrive from one host over one NIC and `nvstreammux` drops whatever misses its 40 ms batch
window. Real cameras are separate devices, so that ceiling is the simulator's, not the Jetson's.

---

## Architecture

![DeepStream pipeline graph](docs/images/pipeline.png)

One DeepStream graph built with `pyservicemaker`. Detection runs batched across all cameras;
everything after the tracker is per-frame metadata work. The probe attaches to the **tracker**,
the last per-stream element — after the tiler, all cameras have been composited into a single
frame and per-stream state is silently wrong.

From there the metadata leaves the pipeline over Redis, and the services that consume it —
incident store, clip capture, VLM adjudication, notifications, API — run as separate processes
that cannot stall the hot path. Verified: stopping Redis under a running pipeline leaves fps flat
and publishing resumes by itself.

The same diagram, with hardware detail, per-model throughput and the full service map, is on the
dashboard's `/system` page.

---

## Hardware requirements

Two platforms. **Jetson** is the reference target and runs on the host. **x86_64 with a discrete
NVIDIA GPU** runs in Docker — see [Deploying on x86](#deploying-on-x86).

The platform-specific parts are detected, not assumed: power modes and locked clocks are Tegra
tools and are skipped elsewhere, DLA exists only on Jetson, and the dashboard's system chart reads
`tegrastats` on Jetson and `nvidia-smi` on a discrete card. `./scripts/check_hardware.sh` reports
which platform it found.

| | Minimum | Reference platform |
|---|---|---|
| Board | Jetson Orin (any) | **AGX Orin 64 GB** |
| JetPack / L4T | JetPack 6+ | JetPack 7.2.1 / L4T R39.2.1 |
| DeepStream | 7.0+ | **9.1.0** |
| TensorRT | 8.6+ | 10.16.2.10 |
| RAM | 32 GB | 64 GB |
| Free disk | 25 GB | 40 GB+ |
| Python bindings | `pyservicemaker` | ships with DeepStream 9.x |

On a smaller Orin (NX, Nano) everything runs — just lower `MAX_STREAMS`. The 20-stream figure is
91% of the AGX Orin 64 GB NVDEC ceiling for 1080p30 H.265 and will not transfer.

Check your box before installing anything:

```bash
./scripts/check_hardware.sh
```

It reports PASS / WARN / FAIL per item and exits non-zero if something is genuinely missing. Every
check exists because getting it wrong produces a failure a long way from its cause.

---

## Quick start — Jetson

A host install. For x86 see [Deploying on x86](#deploying-on-x86), which is a container and needs
none of this.

```bash
git clone <your-fork-url> industry_safety_monitoring_system
cd industry_safety_monitoring_system

# 1. check the hardware, then install everything
./scripts/setup.sh

# 2. add your own tokens  (see "Bring your own tokens" below)
cp .env.example .env && chmod 600 .env
$EDITOR .env

# 3. build the demo media set from your own footage
cp /path/to/your/footage/*.mp4 media/src/
./scripts/make_streams.sh 20

# 4. detector weights -> ONNX -> TensorRT engines
./build/venv-export/bin/python3 scripts/export_models.py all
./scripts/build_engines.sh all

# 5. optional: the local reasoning layer (VLM + agent LLM)
./scripts/setup_reasoning.sh llamacpp --serve     # vision model on :8000
./scripts/setup_reasoning.sh llm --serve          # agent LLM on :8001

# 6. bring the whole demo up
./scripts/demo_up.sh 20
```

Then open `http://<your-jetson>:8080/`.

Steps 3 and 4 take a while — engine builds are several minutes each, and the reasoning layer
compiles llama.cpp from source and downloads ~12 GB of weights. Steps 1–4 are the minimum for a
working detection pipeline; step 5 is optional and everything degrades gracefully without it.

### What `setup.sh` installs

| | |
|---|---|
| System packages | ffmpeg, redis-server, **libmosquitto1**, cmake, build-essential |
| `build/venv-export` | CPU torch + ultralytics + onnx — for `.pt` → ONNX only |
| `build/venv-services` | FastAPI, uvicorn, redis, PyYAML — the API and services |
| `build/venv-hf` | huggingface_hub — model downloads |
| `build/bin/mediamtx` | RTSP / WebRTC / HLS republisher |
| `models/parser/*.so` | the custom YOLO output parser that nvinfer loads |
| `.env` | created from `.env.example`, `chmod 600` |
| Performance mode | MAXN + `jetson_clocks` |

`libmosquitto1` looks unrelated and is not optional: the DeepStream tracker `dlopen`s it, and
without it the tracker fails to load with a message that names the tracker, not the library.

Flags: `--check-only`, `--skip-apt`, `--no-perf-mode`, `--yes`. It is idempotent — re-run it
freely.

---

## Deploying on x86

From nothing to a running system on any cloud GPU instance. Every step below was run on an L4.

### 1. The instance

Docker and the NVIDIA Container Toolkit, and a GPU the toolkit can see:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L
```

~60 GB of free disk: the image is ~33 GB, the model weights ~11 GB, and the clip budget 4 GB.

DeepStream 9.1 documents driver 590+ for dGPU. Below that the container supplies its own newer
user-mode driver through CUDA forward compatibility — measured working on 580.126.20 with an L4.

### 2. Clone

```bash
git clone <your-fork-url> industry_safety_monitoring_system
cd industry_safety_monitoring_system
```

### 3. Put the footage in place

`media/` is gitignored, so a clone has none — the demo set is 454 MB and travels separately from
the repository. Copy it to `media/` at the repo root:

```bash
rsync -avz --progress ./media/cam*.mp4 user@instance:industry_safety_monitoring_system/media/
```

The names are the contract. `media/cam01.mp4 … cam20.mp4`, and the number **is** the camera id
that `configs/analytics/zones.yml`, the OSD label, the clip prefixes and the dashboard all key on
— renumbering the files renames every camera. Each clip is 1920x1080, 30 fps, H.265.

To build a set from your own footage instead, drop it in `media/src/` and:

```bash
docker compose -f docker/compose.yml --profile media run --rm media
```

That re-encodes whatever it finds into the 20-camera set, hardware-encoded through `hevc_nvenc`.
With `media/src/` empty it falls back to the H.265 sample DeepStream ships, which contains no
helmets, vests or fire — it satisfies the decode benchmark and nothing else.

### 4. Credentials, if you want them — optional

Nothing here needs a token. The model weights come from public GGUF repositories and download
anonymously; `HF_TOKEN` is only for a gated repository, and Telegram alerts are off by default.

If you do want either, the file has to exist **before** the containers are created:

```bash
cp .env.example .env && chmod 600 .env      # credentials
$EDITOR .env
```

`env_file` is read at container *creation*, so a `.env` written afterwards needs
`up -d --force-recreate app` — `restart` keeps the environment the container was built with.

### 5. Run it — three modes

Pick one. They differ only in where the video comes from; everything downstream — detection,
tracking, zones, incidents, the VLM, the dashboard — is identical.

| | source | evidence clips | use |
|---|---|---|---|
| **A** | files in `media/` | cut from the `.mp4` with `ffmpeg -c copy` | demos, benchmarks, repeatable numbers |
| **B** | those same files, served as RTSP | `nvurisrcbin` smart record | exercising the live path with no cameras |
| **C** | real cameras | `nvurisrcbin` smart record | production |

#### A. Video files — the default

```bash
docker compose -f docker/compose.yml up -d --build
```

20 file sources from `media/`, run flat out. This is the mode to benchmark in: no network, no
external host, repeatable.

#### B. The same files, served as RTSP

Publishes `media/` as 20 RTSP cameras on `:8654`, then points the pipeline at them:

```bash
ISMS_SOURCE=rtsp ISMS_RTSP_BASE=rtsp://127.0.0.1:8654 \
  docker compose -f docker/compose.yml --profile sources up -d
```

This is the only way to exercise the live path — reconnection, TCP transport, smart-record
evidence clips, subject crops — without cameras. The restreamers burn CPU beside the inference
they feed, so throughput measured here describes a box doing both jobs;
`scripts/serve_rtsp_sources.sh` explains why that is not a deployment topology.

Sources are real-time paced, so 20 cameras at `drop_frame_interval: 2` produce 300 fps of
analytics and the pipeline keeping pace at ~300 fps is the expected result, not a ceiling.

#### C. Real cameras

One URL per camera. **Position in the list is the camera id** that zones, the OSD label, the clip
prefixes and the dashboard all key on:

```bash
ISMS_SOURCE=rtsp \
ISMS_RTSP_URLS="rtsp://user:pw@10.0.0.11/Streaming/Channels/101,rtsp://user:pw@10.0.0.12/axis-media/media.amp" \
  docker compose -f docker/compose.yml up -d
```

A fleet on one server with numbered mounts can use `ISMS_RTSP_BASE=rtsp://10.0.0.5:8554` instead,
which fills in `cam01 … camNN` from `sources.rtsp_pattern`.

Camera URLs come from the environment because they carry credentials and `configs/demo.yml` is
committed — the same reason `HF_TOKEN` lives in `.env`.

#### Making a mode stick

The commands above set the mode for one invocation. To make it the default for this deployment,
put the same names in `docker/.env` (see `docker/.env.example`) and plain `docker compose up -d`
uses them:

```ini
ISMS_SOURCE=rtsp
ISMS_RTSP_BASE=rtsp://127.0.0.1:8654
```

Switching modes later means recreating the app container, not restarting it:

```bash
ISMS_SOURCE=file docker compose -f docker/compose.yml up -d --force-recreate app
```

### First boot, and watching it

First boot builds the TensorRT engines against the GPU that is present (~8 minutes) and pulls
~11 GB of model weights. Both land on named volumes, so every later start skips them.

The reasoning containers are slow to arrive and nothing waits for them: the dashboard and the
pipeline come up immediately, and the reasoning layer joins when the VLM answers. Until then
incidents read `unverified`.

```bash
curl -s localhost:8080/health                       # every component, one line
docker compose -f docker/compose.yml logs -f app    # engine build, then the pipeline
docker compose -f docker/compose.yml --profile sources down   # stop; volumes survive
```


### What runs

Four containers on host networking, so `configs/services.yml` keeps addressing `127.0.0.1` and one
set of configs stays correct on both platforms.

| | |
|---|---|
| `redis` | the event bus |
| `app` | pipeline, services, dashboard, mediamtx |
| `vlm` | Cosmos Reason 2 on `:8000`, adjudicates incidents |
| `llm` | Nemotron Nano 9B on `:8001`, answers questions |

Everything except the dashboard (`:8080`) and mediamtx's RTSP and WebRTC listeners (`:8554`,
`:8889`) binds loopback. Those three are what a browser reaches, and there is no authentication in
front of them.

Three things are not in the image. **Engines** are specific to the GPU, driver and TensorRT that
built them, so they are built on first boot and stamped with the compute capability that did it;
moving the volume to another card rebuilds. **Media** is mounted. **Model weights** are pulled to a
volume. ONNX *is* baked, from a separate build stage — it is architecture-independent.

### Configuration

**File mode is the default.** `docker compose up -d` with nothing else set runs 20 file sources
from `media/`; RTSP is opt-in.

Two env files, with different jobs, because Compose treats them differently:

| | |
|---|---|
| `docker/.env` | what the deployment **is** — source mode, stream count, camera addresses. Compose reads it to substitute `${...}` in `compose.yml`, and looks for it beside the compose file. Template: `docker/.env.example` |
| `<repo>/.env` | **credentials**, injected into the containers by `env_file`. The same file `scripts/env.sh` sources on the Jetson. Template: `.env.example` |

An `ISMS_*` value in the repo-root `.env` does not work: Compose interpolates from `docker/`, and
the `environment:` block then overrides whatever `env_file` supplied. It fails silently and the
deployment runs with the default. Either use `docker/.env`, or set it on the command line.

Settings, in `docker/.env` or per-command:

| | |
|---|---|
| `ISMS_STREAMS` | cameras, default 20 |
| `ISMS_SOURCE` | `file` or `rtsp` |
| `ISMS_RTSP_URLS` | comma-separated per-camera URLs; position is the camera id |
| `ISMS_RTSP_BASE` | camera server, overrides `sources.rtsp_base` |
| `ISMS_WIPE` | wipe previous incidents on start, default 1 |
| `ISMS_RGB_CAPTURE` | subject crops; unset follows the source mode — on for `rtsp`, off for `file` |
| `CUDA_ARCHS` | compute capabilities llama.cpp is built for, default `80;86;89;90` |

Subject crops need CUDA torch in the image, ~3.5 GB; `--build-arg WITH_RGB_CAPTURE=0` leaves it
out and the VLM falls back to context frames. Only RTSP needs it — in file mode the crop is cut
from the source `.mp4` at the incident PTS.

### Measured on an L4

24 GB VRAM, 20 streams, `drop_frame_interval: 2`, zones and events on, full reasoning stack up.

| | file sources | 20 RTSP cameras |
|---|---|---|
| Throughput | **616 fps** flat out | **301 fps**, keeping pace with 20 real-time cameras |
| Cost of the reasoning layer | -0.5% | |
| VRAM | | **15631 MiB of 23034** - 7028 LLM, 5620 VLM, 2948 pipeline |
| VLM verdict | | 3.2-6.2 s PPE, 8.0-9.5 s overcrowding |

Engine throughput is 1869 inf/s for PPE and 801 inf/s for fire, against budgets of 300 and 100.

---


## Bring your own tokens

**Nothing in this repository ships with credentials, and no feature falls back to somebody else's
account.** You supply your own. Copy the template and fill it in:

```bash
cp .env.example .env
chmod 600 .env
```

`.env` is gitignored and must stay that way.

The same file, the same place, on both platforms. On Jetson `scripts/env.sh` sources it; on x86
`docker/compose.yml` reads it into the containers that need it. It is optional in both cases —
without it the features that need a token stay off. The image itself ships no `.env`, so nothing
in it can overwrite yours.

On x86 the reasoning layer needs no token at all: it serves quantised GGUF builds
(`robertzty/Cosmos-Reason2-2B-GGUF`, `bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF`), both
public, and `hf_hub_download` is called with no token when the variable is unset. The 11 GB pull
on this deployment ran anonymously. `HF_TOKEN` matters for the Jetson path, which can fetch
`nvidia/Cosmos-Reason2-2B` itself, and that repository is gated.

### Hugging Face token

Needed only for the optional reasoning layer, which downloads the vision and language model
weights. The detector models are public and need no token.

1. Go to <https://huggingface.co/settings/tokens>
2. Create a token with **read** scope — that is enough
3. Put it in `.env` as `HF_TOKEN=hf_...`

### Telegram bot token and chat id

Needed only if you enable phone alerts. They are **off by default** — an alert pushed to someone's
phone cannot be unsent, so turning them on is a deliberate act.

1. Message [@BotFather](https://t.me/BotFather) in Telegram, send `/newbot`, copy the token
2. Add the bot to a group, or just send it a direct message
3. Get the chat id — either let the helper find it:
   ```bash
   ./scripts/telegram_setup.sh
   ```
   or do it by hand:
   ```bash
   curl "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
   ```
   and read `.result[].message.chat.id` out of the response
4. Put both in `.env`:
   ```
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_CHAT_ID=...
   ```
5. Turn the feature on in `configs/services.yml`:
   ```yaml
   notify:
     telegram:
       enabled: true
   ```
6. Restart: `./scripts/demo_up.sh --down && ./scripts/demo_up.sh`

Check it with `curl http://localhost:8080/notify/status`, and preview the policy without sending
anything using `notify_service.py --dry-run`.

Credentials are read from a file rather than your shell profile on purpose: Ubuntu's default
`.bashrc` returns early for non-interactive shells, so an `export` there is invisible to exactly
the systemd, `nohup` and `ssh host 'cmd'` invocations that need it.

---

## Running it

### The one command

```bash
./scripts/demo_up.sh          # wipe previous incidents, 20 streams, live view on
./scripts/demo_up.sh 12       # 12 streams
./scripts/demo_up.sh 20 --keep    # keep existing incidents instead of wiping
./scripts/demo_up.sh --down   # stop everything
```

Wiping is the default because the usual reason to restart is to show the system detecting things
live, and a feed pre-filled with an old run makes new alerts impossible to pick out.

`--down` deliberately **leaves the two model servers running** on :8000 and :8001. They cost
nothing when idle (measured: 517.8 fps at 20 streams with both loaded, identical to baseline) and
take minutes to reload. To free their ~12 GB as well: `pkill -x llama-server`.

### Where to look

| | |
|---|---|
| Dashboard | `http://<jetson>:8080/` |
| System reference | `http://<jetson>:8080/system` |
| API browser | `http://<jetson>:8080/docs` |
| Tiled wall (RTSP) | `rtsp://<jetson>:8554/safety` |
| Live view (WebRTC) | `http://<jetson>:8889/safety` |
| Service logs | `logs/` |

### Timing to expect

An alert appears in **under a second**. Its evidence clip follows in **2–3 s**. The VLM verdict
takes **5–10 s per incident and runs serially**, so a cold start (~30 incidents in the first 40 s)
takes a few minutes to fully adjudicate. That is queueing, not a stall — in steady state incidents
arrive as transitions and the adjudicator keeps up easily.

### Real cameras instead of files

Set `pipeline.source_mode: rtsp` and list the cameras:

```yaml
sources:
  rtsp_urls:
    - rtsp://user:pw@10.0.0.11/Streaming/Channels/101      # Hikvision
    - rtsp://user:pw@10.0.0.12/axis-media/media.amp        # Axis
    - rtsp://10.0.0.13:554/cam/realmonitor?channel=1&subtype=0   # Dahua
```

List position is the camera id — zones, the OSD and clip names all key on it. An empty
`rtsp_urls` falls back to `rtsp_base` + `rtsp_pattern`, the demo's numbered mounts on one server.

`sources.rtsp` carries reconnect, TCP transport and a 200 ms jitterbuffer. Reconnect is the one
that matters: `nvurisrcbin` defaults it to *never*, so a camera that blips goes black until the
pipeline restarts.

The simulator from earlier serves the same purpose here, from another machine:

```bash
./scripts/serve_rtsp_sources.sh start 20   # prints the rtsp_base line to paste in
```

It loops a compressed stream, which restarts the GOP and occasionally makes the fire detector
fire on decode smear — an artifact of the simulator, not the detector. The same footage in file
mode detects nothing.

### Evidence over RTSP

There is no local source file, so both come off the pipeline:

- **Clips** — `nvurisrcbin` smart record, straight from its ring buffer. Full source frame rate,
  no re-encode. Slower to appear than file mode (11 s vs 2 s) because the window has to elapse.
- **Subject crops** — captured in the probe at the moment of detection, behind a valve that
  costs nothing when idle (484.6 → 481.3 fps at 20 streams). Needs torch:

```bash
python3 -m venv --system-site-packages build/venv-pipeline   # sees system pyservicemaker
build/venv-pipeline/bin/pip install torch
build/venv-pipeline/bin/python3 app/safety_pipeline.py --source rtsp --rgb-capture ...
```

Without `--rgb-capture` the VLM falls back to context frames — weaker, but no extra dependency.

---

## Configuration

Two files, split by lifecycle. The pipeline gets restarted constantly during tuning; the services
are meant to stay up.

### `configs/demo.yml` — the pipeline

| Knob | What it does |
|---|---|
| `pipeline.streams` | 1–20. How many cameras. |
| `pipeline.source_mode` | `file` (benchmarking — repeatable) or `rtsp` (demo) |
| `pipeline.drop_frame_interval` | `2` = 15 fps analytics. **This is what makes the local reasoning layer affordable.** |
| `pipeline.topology` | `serial` (default) or `parallel` (only pays off with a DLA-resident model) |
| `rules.*.min_confidence` | Per-rule thresholds. **Never stricter than the detector's own `pre-cluster-threshold`.** |
| `rules.window_frames` / `flip_ratio` | Debouncing — how many frames must agree before a verdict flips |
| `sinks.display` / `sinks.rtsp_out` | Local monitor and/or network output |
| `render.compute_hw` | `gpu`. The Jetson default is VIC, which costs 33% throughput. |

### `configs/services.yml` — everything downstream

Redis, the incident store (including `realert_after_s`, which re-raises incidents left open too
long), clip retention and disk budget, the reasoning endpoint, and the notification policy —
`always_types`, `confirmed_types`, `min_severity`, rate limits.

### The rest

`configs/pgie_ppe.yml` and `pgie_fire.yml` are the detectors. `tracker_nvsort_tuned.yml` is the
tracker — **do not point this at the stock `config_tracker_NvSORT.yml`**, which emits zero objects
while appearing to run faster. `analytics/zones.yml` is the zone geometry;
`scripts/make_zones.py --preview N` renders the polygons onto a real frame so you can check them
by eye.

Full reasoning behind every value is in [`project_skill.md`](project_skill.md).

---

## Tests

```bash
./tests/run_all.sh            # everything that can run on this machine
./tests/run_all.sh --logic    # only the dependency-free tests
```

Nothing in `tests/` talks to a network, a model server, a GPU or a real Redis. `test_rules.py` and
`test_events.py` need no dependencies at all and run on a laptop — that is deliberate, because the
code that decides whether a worker is compliant is the part most likely to be subtly wrong and
should not need a Jetson and a video feed to check. See [`tests/README.md`](tests/README.md).

To validate a live incident database against the store's invariants:

```bash
python3 tools/inspect_db.py --check-only
```

---

## Models

Downloaded at setup time, not committed.

| Purpose | Model | Source | Licence |
|---|---|---|---|
| PPE / helmet / vest | YOLOv11n, 4 classes | `melihuzunoglu/ppe-detection` | AGPL-3.0 |
| Fire / smoke | YOLOv26s | `SalahALHaismawi/yolov26-fire-detection` | MIT |
| Vision verification | Cosmos Reason 2 (2B) | `nvidia/Cosmos-Reason2-2B` | NVIDIA Open Model |
| Agent | Nemotron Nano 9B v2 | NVIDIA | NVIDIA Open Model |

The PPE model has **no `no-vest` class**, so a vest violation is inferred — "a person with no
overlapping vest box". The UI reflects the difference in evidence strength: `NO HELMET` is
upper-case and definite, `no vest?` is lower-case and hedged. Helmet detection is direct and is
the trustworthy signal.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Pipeline runs fast, zero detections | Stock NvSORT config — `minTrackerConfidence` is unreachable without a visual tracker, so every target sits in shadow mode | Use `configs/tracker_nvsort_tuned.yml` |
| `Failed to initilaize low level lib` | The tracker `dlopen`s libmosquitto | `sudo apt install libmosquitto1` |
| `setDimensions: Error Code 3` | Dynamic-axis ONNX without explicit dims | `infer-dims=3;640;640` in the nvinfer config |
| `nv3dsink: no display found` | Wrong X display number | Auto-detected by `scripts/env.sh`; pin with `DISPLAY_NUM` in `.env` |
| `Could not get EGL display connection`, then `nvinfer ... Failed to set buffer pool to active` | `DISPLAY` points at an X server this user cannot reach — often the login-screen greeter. The error names nvinfer but the model is fine | Leave `DISPLAY` **unset** and run headless; `scripts/env.sh` now probes with `xdpyinfo` and only exports a display that answers |
| Pipeline deadlocks in PAUSED, no error | A sink missing `async=0` with a `tee` in the graph | Set `async=0` on every sink |
| Black video box in the dashboard | Three different causes | `curl localhost:8080/live/status` — it names which one and the next action |
| Evidence clip plays as a black rectangle | Browsers cannot decode H.265 | Already handled — `/clips/{id}` transcodes lazily. Check `logs/api.log` |
| Agent answers slowly, always with the same plan | The soft fallback is firing | Check `plan_error` in the response |
| `database is locked` | A write from a read path, or an uncommitted empty transaction | See `project_skill.md` §6, traps 23 and 27 |
| Throughput varies wildly between runs | Clocks not locked, or a previous pipeline still alive | `sudo jetson_clocks`; `ps -eo args \| grep '[s]afety_pipeline'` |

The full trap list — 67 verified behaviours, several of which contradict the SDK docs — is in
[`project_skill.md`](project_skill.md) §6.

---

## Repository layout

```
app/            the DeepStream pipeline and the rules that turn detections into events
services/       everything downstream: store, clips, VLM, agent, API, alerts
tools/          operator utilities — database integrity, verdict inspection
tests/          unit tests, none of which need hardware
scripts/        setup, build, run, and measurement
configs/        pipeline, model, tracker, zone and service configuration
dashboard/      the operator UI and the developer reference page
models/         label files and the custom TensorRT output parser
bench/          measured results — the source for every number quoted anywhere
requirements/   Python dependencies, one file per virtualenv
docker/         the x86 images, compose file and entrypoints
.claude/skills/ 28 NVIDIA agent skills (DeepStream, Jetson, VSS)
```

---

## For AI coding assistants

Two things are here for you:

**[`project_skill.md`](project_skill.md)** — the compiled state of the project. Architecture, every
measured number with the script behind it, the settled constraints, 66 verified field notes, and
where the system is built to grow. Read it before changing anything.

**[`.claude/skills/`](.claude/skills/)** — 28 NVIDIA agent skills covering DeepStream development,
Jetson tuning and the VSS platform, with exact API and property references. Read the relevant
skill's reference docs before generating pipeline code; they contain the precise property names
that guesswork gets wrong.

---

## Licence

AGPL-3.0 — see [LICENSE](LICENSE).

The PPE detector is a YOLOv11 model and the export path uses Ultralytics, both AGPL-3.0, which is
why this project is too. The NVIDIA agent skills under `.claude/skills/` are NVIDIA's own work
under their own terms (Apache-2.0 / CC-BY-4.0 / MIT — each skill's `skill-card.md` says which) and
are **not** covered by the AGPL. Full attribution in [NOTICE](NOTICE).
