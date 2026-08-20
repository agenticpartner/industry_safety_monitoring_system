# Capacity findings — industrial safety demo on Jetson AGX Orin 64GB

**Result: 20 concurrent 1080p30 H.265 streams run in realtime on realistic industrial footage —
but with only 1.05× margin.** Every step from 1 to 20 cameras holds 30 fps/stream through the
full chain: decode, PPE detection, fire/smoke detection, tracking, compliance logic, tiling, OSD.

Device: JetPack 7.2.1 / L4T R39.2.1, DeepStream 9.1, TensorRT 10.16.2, MAXN, clocks locked.
Media: NVIDIA MV3DT synthetic **warehouse** dataset, transcoded to 1080p30 H.265.

---

## 1. End-to-end capacity

`scripts/sweep.sh --label warehouse` — full pipeline, headless, timed wall-clock to EOS.
Every row is detection-verified; a zero-detection run is refused, not recorded as a pass.

| Streams | fps | Target | Holds? | Headroom | GPU | RAM |
|--:|--:|--:|:--:|--:|--:|--:|
| 1  | 254.2 |  30 | YES | 8.47× | 74% |  8.0 GB |
| 2  | 385.2 |  60 | YES | 6.42× | 84% |  8.2 GB |
| 4  | 515.0 | 120 | YES | 4.29× | 94% |  8.4 GB |
| 8  | 621.3 | 240 | YES | 2.58× | 99% |  8.9 GB |
| 12 | 633.2 | 360 | YES | 1.75× | 99% |  9.4 GB |
| 16 | 625.9 | 480 | YES | 1.30× | 99% |  9.8 GB |
| 18 | 626.7 | 540 | YES | 1.16× | 99% | 10.1 GB |
| **20** | **633.7** | **600** | **YES** | **1.05×** | 99% | 10.4 GB |

Throughput plateaus at **~630 fps**, so the practical ceiling is ~21 cameras. On the earlier
traffic footage the same pipeline reached 723.8 fps (1.20×) — warehouse scenes have more people
per frame, and **tracker cost scales with people, not pixels**. The warehouse number is the one
to quote.

**1.05× is thin.** It holds on this footage at this ambient temperature with clocks locked. A
busier shift, a hotter cabinet, or a third model would break it. For production I would size at
**16 cameras per Orin (1.30×)** and treat 20 as the demonstrated maximum, not the design point.

Steady-state throughput measured *inside* the pipeline is higher (~880 fps at N=20) because
wall-clock-to-EOS includes ~10 s of TensorRT engine load. The sweep deliberately reports the
conservative wall-clock figure.

## 2. Component ceilings

**Decode** (NVDEC only): flat at **~1100 fps** from N=4 up — a real hardware ceiling divided
among streams, implying **~36 concurrent 1080p30 H.265 streams**, well above the 22 in the AGX
Orin datasheet. Never a constraint here.

**Inference** (`trtexec`, batch 20, `--noDataTransfers`):

| Model | Arch | Precision | Inferences/s |
|---|---|---|--:|
| ppe  | YOLO11n (2.6M) | FP16 | 929.2 |
| ppe  | YOLO11n | INT8¹ | 1342.1 |
| fire | YOLO26s (9.5M) | FP16 | 507.8 |

¹ No calibration cache — a speed upper bound only; accuracy meaningless.

**Inference is not the bottleneck.** Raising the PPE interval from 1→3 (cutting inference by
two-thirds) moved throughput only 633→676 fps (1.05×→1.12×) while losing 30% of `no-helmet`
detections. Not worth it; interval stays at 1. **The tracker is the binding constraint.**

## 3. What actually cost throughput

### 3a. Jetson defaults the tiler and converter to VIC — pin them to GPU

`compute-hw` defaults to `Default` = **VIC** on Jetson, a fixed-function block far slower than
the GPU for 1080p scale/convert. At N=8: VIC 41.9 fps, **GPU 62.8 fps**.

### 3b. The NV12→RGBA convert before `nvdsosd` was unnecessary

`nvdsosd` GPU mode takes NV12 directly. Removing the convert plus 3a took N=8 from 41.1 → 63.5 fps.

**Rendering now costs ~3%** (N=20: 479.7 fps metadata-only vs 466.3 rendered; the tiler alone
measured 1064 vs 1081 fps in isolation). There is no speed-vs-visualisation trade-off left to
make — keep the tiled display.

Output path cost at N=20, timed to EOS: headless 630.9 · local display 590.7 (−6%) ·
RTSP/NVENC 575.2 (−9%).

### 3c. Tracker choice and fire interval

| Config @ N=20 (traffic) | fps | Detections |
|---|--:|---|
| NvDCF_perf + fire interval 5 | 466.5 | ✓ |
| NvSORT **tuned** + fire interval 11 | **724.2** | ✓ |
| NvSORT **stock** + fire interval 11 | 700.5 | **NONE** |

NvSORT (Kalman + IoU, no visual features) replaced NvDCF_perf: same detection counts to within
0.2%, ~1.3× the speed. Fire interval 5→11 (infer every 12th frame, 400 ms latency) is free —
fire evolves over seconds.

## 4. Bugs found (all fixed)

1. **Stock `config_tracker_NvSORT.yml` emits ZERO objects.** `minTrackerConfidence: 0.8216` is
   unreachable without a visual tracker, so every target sits in shadow mode. The pipeline looks
   healthy and runs *faster* — briefly making a broken config the best benchmark result. Fixed by
   `configs/tracker_nvsort_tuned.yml`; `sweep.sh` now refuses any zero-detection row.
2. **The compliance probe was attached after the tiler.** `nvmultistreamtiler` composites the
   batch into ONE frame, so the probe saw a single `frame_meta` with `source_id` always 0 — all
   20 cameras collapsed into stream 0 and per-stream state was silently wrong. Measured directly:
   probe on tracker = 20 frames/batch, 20 source_ids; probe on OSD = 1 and 1. It also made the
   frame counter count *batches*, under-reporting fps by exactly N. Probes must attach to the
   last per-stream element.
3. **The fps probe folded startup into its average** — `t0` stamped before `pipeline.start()`, so
   ~20 s of engine deserialisation was averaged in. Now windowed; the sweep uses wall-clock.
4. **`--stats` was nested inside the `--fps` branch**, so a stats-only run printed nothing and
   the new zero-detection guard flagged every row. (The guard failed safe, which is correct.)
5. **`ComplianceTracker.violation_count()` scanned every stream's tracks every frame**, and
   `_expire()` swept all state per frame — quadratic across a batch. Now per-stream and throttled.

## 5. Demo quality — the honest weak spot

Visual output verified by rendering frames (`--snapshot`): 20 tiles, per-camera violation
banners, red/green person boxes, terse labels.

**Most violations shown are `no vest?`, not `NO HELMET`.** That is the inferred-rule weakness
flagged at the start, now visible in practice: the PPE model has **no `no-vest` class**, so a
vest violation is "a person with no overlapping vest box" — absence of evidence. In this
warehouse many workers wear dark clothing, so the rule fires almost everywhere. Helmet violations
are direct (`no-helmet` is a trained class) and are the trustworthy signal.

Rendering reflects this: `NO HELMET` is upper-case and definite, `no vest?` is lower-case and
hedged, so an operator can tell the evidence strengths apart. Closing the gap properly requires
fine-tuning on a dataset that labels `no-vest`.

## 6. Caveats

- **Synthetic footage.** The MV3DT warehouse dataset is rendered, not real. Detection rates on
  real plant footage will differ, likely worse (lighting, motion blur, occlusion).
- **File sources, not RTSP.** Capacity is measured in file mode deliberately — network jitter is
  not the Jetson's compute ceiling. RTSP sources are served from a separate machine for realism
  (`scripts/serve_rtsp_sources.sh`), never from the Jetson, which would burn CPU beside inference.
- **GPU sits at 99% from N=8 up.** Throughput headroom exists; compute headroom does not.
- **DLA unused** (`dla0`/`dla1` off in every row). Never needed — GPU met the target. The
  all-or-nothing qualification gate (`scripts/qualify_dla.sh`) is written and ready.
- **Phase 5 will disturb all of this.** Cross-camera tracking needs re-ID embeddings, i.e.
  NvDCF/NvDeepSORT — precisely the element that cost the most here. Budget must be re-measured.
  The MV3DT dataset ships `camInfo/*.yml` calibration and a BEV map, so that groundwork exists.
