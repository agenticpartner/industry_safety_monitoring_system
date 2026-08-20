# Phase 2.3 — evidence clips: what works on this build

**Verdict: clips work, are full 30 fps, and cost the pipeline nothing measurable.** But
`nvurisrcbin` smart-record — the mechanism the Phase 2 plan assumed — **does not work on file
sources**, which is the mode the demo and every benchmark run in. Two backends, chosen by source
type.

Measured 2026-08-17 on the reference AGX Orin 64GB, DS 9.1 / JetPack 7.2.1.

---

## 1. Smart-record is RTSP-only. This is real, not a misconfiguration.

`gst-inspect-1.0 nvurisrcbin` says it three times:

```
smart-record        : ... Sources must be of type source-type-rtsp
smart-rec-container : ... Sources must be of type source-type-rtsp
type                : Set the type of source. Use source-type-rtsp to use smart record features
```

Tested both ways with the same code, changing only the URI:

| Source | `type` | `start_recording()` | File written? |
|---|---|---|---|
| `file:///…/cam01.mp4` | (auto) | returns session id `0`, no error | **no — nothing, ever** |
| `rtsp://127.0.0.1:8554/…` | `2` (rtsp) | returns session id `0` | **yes — 8.71 MB mp4** |

The file-source failure is **silent**: a session id comes back, no exception is raised, and no
file appears. Anything that trusts the return value will report success forever.

On the RTSP path the `sr-done` callback returns a populated record, which is what a production
integration would index against the incident:

```
RecordingInfo(session_id=0, file_name=cam01_-1_00000_20260816-185107_400375.mp4,
              file_directory=/tmp/cliptest, duration=39174, container_type=MP4,
              width=1920, height=1080, contains_video=1, contains_audio=0)
```

### The API is nicer than the docs suggest

The skill reference documents smart recording through a Kafka `smart_recording_action` controller,
i.e. cloud-triggered. For local triggering, `pyservicemaker.Pipeline` exposes it directly and no
signal emission is needed:

```python
session_id = pipeline.start_recording(source_name, start_time, duration, callback)
pipeline.stop_recording(source_name)
pipeline.stop_recording_by_session_id(session_id)
```

### Caveat on the RTSP path — not yet production-tuned

The RTSP clip captured **530 packets over a 39 s span** where the same stream delivered a clean
30 fps to an `ffmpeg` client (436 packets / 14.53 s). Frames are missing in ~0.22 s gaps
throughout. That points at RTP transport loss into `nvurisrcbin` (it defaults to UDP;
`udp-buffer-size` and `select-rtp-protocol` are the knobs), **not** at smart-record itself. It was
not chased because file mode is what the demo runs. **Tune and re-measure before relying on the
RTSP path for real evidence.**

---

## 2. File mode: cut the window out of the source

`services/clip_service.py` cuts the clip with `ffmpeg -c copy` — stream copy, so no re-encode, no
GPU, no quality loss, and the clip is inherently at the source's own frame rate.

### Finding the moment

Incidents carry **`source_pts_ns`** = `frame_meta.buffer_pts`, the frame's position in SOURCE
time. Wall-clock would be wrong: in file mode the pipeline runs faster or slower than realtime
depending on load and `drop-frame-interval`, so elapsed seconds say nothing about where in the
video the incident is. Measured at `dfi=2`: **5.2 s of wall time advanced `buffer_pts` by 10.3 s
of source.**

Sources loop, so the offset is `pts % duration`. That is exact rather than approximate **because
every loop replays identical content** — landing in the wrong loop still lands on the right frame.

Two fields were considered and one rejected:

| Field | Behaviour | Verdict |
|---|---|---|
| `buffer_pts` | tracks source time; survives `drop-frame-interval` | **used** |
| `frame_number` | counts DELIVERED frames, so it runs at the analytics rate, not the source rate | rejected |

`buffer_pts` does show occasional discontinuities across loop boundaries (one run jumped
95.7 s → 125.1 s). It does not matter here for the same reason the modulo is safe: every loop is
the same content.

---

## 3. Results at 20 streams

| | |
|---|---|
| Pipeline fps with clip capture live | **516.2** (baseline range 510–519 → no measurable cost) |
| Events | 5399 published, **0 dropped** |
| Incidents | 30, **30/30 clips `ready`, 0 failed** |
| Clip properties | hevc 1920×1080, **362 packets / 12.067 s = exactly 30.0 fps** |
| Clip size | ~5.5 MB for a 12 s clip |

### The Phase 2 claim, verified

> "Evidence clips stay smooth… recorded clips remain full 30 fps even though analytics runs at 10."

**Confirmed, by a different mechanism than the plan expected.** Analytics ran at 15 fps
(`dfi=2`); clips are 30.0 fps. The plan's reasoning was that smart-record taps the encoded stream
ahead of the decoder — true, but only available on RTSP. In file mode the clip is cut from the
original file, which gives the same guarantee for a different reason: the source is never decoded
at all.

### Retention

Disk budget enforced on every pass, oldest first. Verified on real data: 168 MB → 53 MB against a
60 MB budget, 21 clips deleted.

**Deleting a clip does not delete its incident.** The row stays and moves to
`clip_state='expired'` — the incident still happened, only the evidence aged out. Verified: all 30
rows survived.

---

## 4. Why the clip service polls the store instead of consuming the bus

The bus carries raw per-track transitions; the store folds them into incidents. A clip belongs to
an incident, and only the store knows which transitions merged into which. A bus consumer would
have to duplicate that logic and would still race the writer.

Polling `clip_state='pending'` is restartable, naturally rate-limited, and makes the backlog
inspectable with a `SELECT`. It is also the exact shape Phase 2.4's reasoning service needs, so
that phase inherits a proven pattern rather than inventing one.

---

## 5. Carried forward

- **RTSP smart-record needs transport tuning** before it can be trusted as evidence (§1).
- **Clips attach to a camera-level incident, not a person.** Accepted for this demo. A clip shows
  the situation on that camera at that moment, which is what an operator reviews anyway.
- **`-ss` before `-i` snaps to the nearest keyframe.** The demo media has a 1 s closed GOP
  (`scripts/make_streams.sh`), so the snap is at most 1 s — well inside the 6 s pre-roll. Media
  with longer GOPs would need `-ss` after `-i` (slower, frame-exact) or a bigger pre-roll.
