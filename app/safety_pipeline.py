#!/usr/bin/env python3
"""
Industrial safety monitoring pipeline — DeepStream 9.1 / pyservicemaker on Jetson AGX Orin.

    python3 app/safety_pipeline.py [--streams N] [--topology serial|parallel]
                                   [--source file|rtsp] [--no-display] [--rtsp-out] [--fps]

Everything configurable lives in configs/demo.yml; the flags above are overrides so the
benchmark sweep can drive this without editing files.

Topology (see configs/demo.yml `topology`):

  serial    sources -> mux -> nvinfer(ppe) -> nvinfer(fire) -> tracker -> tiler -> osd -> sinks
            Simple and known-good. Both models run on GPU, one after the other.

  parallel  sources -> mux -> tee =+=> nvinfer(ppe)  =+=> nvdsmetamux -> tracker -> ...
                                 +=> nvinfer(fire) =+
            Only worth using when one model is DLA-resident: branching is what lets GPU and DLA
            work on the same batch concurrently instead of queueing. With both models on GPU it
            adds a synchronisation point for no gain, so serial is the default.

Four DeepStream/pyservicemaker footguns this file is written around, all found the hard way on
this exact build (DS 9.1 / JetPack 7.2). Each one fails loudly but unhelpfully:

  1. `object_items` yields TRANSIENT proxies. Materialising them (`list(frame.object_items)`)
     and reading attributes afterwards SEGFAULTS. Everything must be read inline, in one pass.
     The SDK docs' "convert to list first" advice does not hold here.
  2. An object's `text_params` is an `osd.TextParams`, which is NOT shaped like a standalone
     `osd.Text`: it has `font_params` / `set_bg_clr` / `text_bg_clr`, not `font` / `set_bg_color`
     / `bg_color`. Using the wrong names raises AttributeError -> process abort.
  3. Any uncaught exception inside a probe becomes `terminate called` (SIGABRT) rather than a
     Python traceback, so handle_metadata wraps everything in try/except.
  4. Every sink needs async=0 when a tee is in the graph, or the pipeline deadlocks in PAUSED
     with no video and no error message.

Also: this install ships `nvdsosd` (needs an explicit RGBA nvvideoconvert in front), not the
`nvosdbin` wrapper, and the tracker's low-level lib needs `libmosquitto1` installed.
"""

from __future__ import annotations

import argparse
import glob
import os
import queue
import sys
import threading
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules import (  # noqa: E402
    Box, ComplianceTracker, COLOR_FIRE, PPE_HUMAN, VIOLATION,
)
from events import (  # noqa: E402
    EventEmitter, TransitionDetector, FireTransitionDetector, OvercrowdingDetector,
    SEVERITY_RANK,
)

from pyservicemaker import (  # noqa: E402
    Pipeline, Probe, BatchMetadataOperator, BufferOperator, osd,
)

ROOT = Path(__file__).resolve().parent.parent
UNTRACKED = 0xFFFFFFFFFFFFFFFF

PPE_GIE_ID = 1
FIRE_GIE_ID = 2

# nvinfer batch-size is pinned to the batch the engines were built for, NOT to the stream count.
# The engines carry a dynamic profile (min=1, opt=max=20), so one engine serves every step of the
# 1->20 sweep; keying batch-size to the stream count instead would force a ~7-minute TensorRT
# rebuild at every step. nvinfer processes whatever partial batch nvstreammux hands it.
ENGINE_BATCH = 20


# ---------------------------------------------------------------------------------------------
# OSD / compliance probe
# ---------------------------------------------------------------------------------------------
class SafetyOverlay(BatchMetadataOperator):
    """Applies the compliance verdict to each person box and draws per-tile status."""

    def __init__(self, cfg: dict, fire_labels: list[str], ppe_labels: list[str]):
        super().__init__()
        self.tracker = ComplianceTracker(cfg)
        self.fire_labels = fire_labels
        self.frames = 0
        # t0 is deliberately NOT set here. This object is built before pipeline.start(), and
        # TensorRT engine deserialisation plus tracker init take tens of seconds — folding that
        # into the average makes a healthy pipeline look several times slower than it is. It is
        # stamped on the first buffer instead.
        self.t0 = None
        self.window_frames = 0
        self.window_t0 = None
        self.report_fps = False
        # Verdicts computed from the PREVIOUS frame, keyed (stream_id, track_id). See the
        # one-frame-lag note in _handle_frame.
        self._pending: dict[int, dict[int, dict]] = {}
        self._errors = 0
        self.report_stats = False
        self.count_only = False
        self.class_counts: dict[str, int] = {}
        self.ppe_labels = ppe_labels
        # Track-stability accounting, off by default — it is pure measurement overhead and this
        # runs inside the hot-path probe. {stream: {track_id: [first_t, last_t, frames_seen]}}
        self.track_stats = False
        self._tracks: dict[int, dict[int, list]] = {}
        # Event publishing. `emitter` is None when disabled, and every call site checks — the
        # probe must cost nothing at all when events are off, since that is how the Phase 1
        # benchmark numbers were produced and they have to stay reproducible.
        self.emitter: EventEmitter | None = None
        self._transitions: TransitionDetector | None = None
        self._fire_transitions: FireTransitionDetector | None = None
        self._oc_transitions: OvercrowdingDetector | None = None
        # {camera_id: {zone_name: severity}} from configs/analytics/zones.yml. Empty when zones
        # are off, which makes every zone lookup a cheap dict miss rather than a branch.
        self.zone_severity: dict[int, dict[str, str]] = {}
        # Separate budgets for object and frame records. A single shared counter is starved by
        # objects — there are dozens per frame and only one frame record — so the frame-level
        # metadata looked absent when it was merely last in the queue.
        self.dump_analytics = 0
        self.dump_analytics_frame = 0
        self._zone_hits: dict[str, int] = {}

    def enable_events(self, emitter: EventEmitter) -> None:
        self.emitter = emitter
        self._transitions = TransitionDetector(emitter, VIOLATION)
        self._fire_transitions = FireTransitionDetector(emitter)
        self._oc_transitions = OvercrowdingDetector(emitter)

    def _object_zones(self, obj) -> list[str]:
        """ROI labels this object currently sits inside, from nvdsanalytics object user meta.

        Read INLINE, inside the caller's single pass over `object_items` — the object handle is a
        transient proxy and touching it after its iteration step segfaults on this build.
        """
        if not self.zone_severity:
            return []
        out: list[str] = []
        try:
            for um in obj.nvdsanalytics_obj_items:
                aoi = um.as_nvdsanalytics_obj()
                if aoi is None:
                    continue
                if self.dump_analytics > 0:
                    self.dump_analytics -= 1
                    print(f"[analytics-obj] attrs={[a for a in dir(aoi) if not a.startswith('_')]}"
                          f" roi_status={getattr(aoi, 'roi_status', None)!r}"
                          f" lc_status={getattr(aoi, 'lc_status', None)!r}"
                          f" oc_status={getattr(aoi, 'oc_status', None)!r}", flush=True)
                rs = getattr(aoi, "roi_status", None)
                if rs:
                    out.extend(str(r) for r in rs)
        except Exception:  # noqa: BLE001
            # Analytics metadata is an enrichment, never a requirement. If the shape is not what
            # we expect, events still carry zone=None rather than the pipeline dying.
            return out
        return out

    def _frame_overcrowding(self, frame_meta, stream_id: int) -> list[str]:
        """Zone names currently over their occupancy limit."""
        if not self.zone_severity:
            return []
        out: list[str] = []
        try:
            for um in frame_meta.nvdsanalytics_frame_items:
                afm = um.as_nvdsanalytics_frame()
                if afm is None:
                    continue
                if self.dump_analytics_frame > 0:
                    self.dump_analytics_frame -= 1
                    print(f"[analytics-frame] attrs="
                          f"{[a for a in dir(afm) if not a.startswith('_')]} "
                          f"oc_status={getattr(afm, 'oc_status', None)!r} "
                          f"obj_in_roi_cnt={getattr(afm, 'obj_in_roi_cnt', None)!r}", flush=True)
                oc = getattr(afm, "oc_status", None)
                if not oc:
                    continue
                # FRAME-level oc_status is a dict {zone: bool} — verified at runtime, and it
                # differs from the OBJECT-level oc_status, which is a plain list of the
                # overcrowding zones an object sits in. Iterating the dict directly yields keys,
                # so every configured zone would read as overcrowded whether or not it is.
                if isinstance(oc, dict):
                    out.extend(str(k) for k, v in oc.items() if v)
                else:
                    out.extend(str(z) for z in oc)
        except Exception:  # noqa: BLE001
            return out
        return out

    def zone_of(self, stream_id: int, zones: list[str]) -> tuple[str | None, str | None]:
        """Pick the zone that matters and its severity.

        A person can be inside several overlapping zones (a walkway ROI and an overcrowding ROI
        covering the same floor). The one worth putting on the incident is the most severe, since
        that is what changes the operator's response — reporting "AisleLeft" when the person is
        also standing in the forklift route would understate the finding.
        """
        table = self.zone_severity.get(stream_id + 1) or {}
        best_name, best_sev, best_rank = None, None, -1
        for z in zones:
            sev = table.get(z)
            if sev is None:
                continue
            rank = SEVERITY_RANK.get(sev, 0)
            if rank > best_rank:
                best_name, best_sev, best_rank = z, sev, rank
        return best_name, best_sev

    def handle_metadata(self, batch_meta):
        # An uncaught exception in a probe does not raise into Python — pybind11 turns it into
        # `terminate called` and the whole process aborts with SIGABRT. Swallow and report
        # instead, so one bad frame can never take down a 20-camera demo.
        try:
            for frame_meta in batch_meta.frame_items:
                if not self.count_only:
                    self._handle_frame(batch_meta, frame_meta)
                self.frames += 1
        except Exception as e:  # noqa: BLE001
            self._errors += 1
            if self._errors <= 5:
                import traceback
                print(f"[probe-error] {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()

        now = time.monotonic()
        if self.t0 is None:
            self.t0 = self.window_t0 = now

        # NOTE: stats must NOT be nested under report_fps. They were, which meant a --stats run
        # without --fps printed nothing — and the sweep's zero-detection guard then flagged every
        # row as broken. Reporting is driven by the window, and each flag is independent.
        if ((self.report_fps or self.report_stats or self.track_stats)
                and self.frames - self.window_frames >= 300):
            wdt = now - self.window_t0
            if self.report_fps and wdt > 0:
                # Report the RATE OVER THE LAST WINDOW, not the cumulative average — a cumulative
                # figure keeps dragging startup cost forward and never settles on the true rate.
                inst = (self.frames - self.window_frames) / wdt
                total = self.frames / max(now - self.t0, 1e-9)
                print(f"[fps] {inst:.1f} frames/s aggregate (avg {total:.1f})", flush=True)
            self.window_frames = self.frames
            self.window_t0 = now
            if self.report_stats:
                # Proof the pipeline is doing real work, not just moving buffers: which classes
                # fired and how many people are currently judged non-compliant.
                counts = " ".join(f"{k}={v}" for k, v in sorted(self.class_counts.items()))
                viol = sum(len([1 for v in p.values() if v["state"] == VIOLATION])
                           for p in self._pending.values())
                ev = ""
                if self.emitter is not None:
                    e = self.emitter.stats()
                    # `dropped` is printed even when zero. A silent drop counter is how a queue
                    # that is quietly losing events looks exactly like one that is healthy.
                    ev = (f" | events pub={e['published']} drop={e['dropped']} "
                          f"q={e['queued']} err={e['errors']}")
                zh = ""
                if self._zone_hits:
                    top = sorted(self._zone_hits.items(), key=lambda kv: -kv[1])[:4]
                    zh = " | zones " + " ".join(f"{k}={v}" for k, v in top)
                print(f"[stats] detections: {counts or 'none'} | violations_now={viol} "
                      f"| probe_errors={self._errors}{ev}{zh}", flush=True)
            if self.track_stats:
                self.report_track_stability()

    def _handle_frame(self, batch_meta, frame_meta):
        stream_id = frame_meta.source_id

        # CRITICAL: ObjectMetadata handles yielded by `object_items` are transient proxies —
        # holding one past its iteration step and reading it later segfaults (verified on this
        # build; the SDK docs' "convert to list first" advice does NOT hold here). So everything
        # is done in ONE inline pass: read attributes, paint immediately, and copy the geometry
        # into our own Box dataclass, which is safe to keep.
        #
        # That means the verdict painted on a person is the one computed from the PREVIOUS
        # frame. Verdicts are already debounced over ~15 frames, so one frame of lag (33 ms) is
        # imperceptible — and it buys a single-pass probe with no dangling references.
        pending = self._pending.get(stream_id, {})
        ppe_boxes: list[Box] = []
        fire_boxes: list[Box] = []
        # {track_id: (zone_name, zone_severity)} gathered in the SAME inline pass as everything
        # else, because the object handles cannot be revisited afterwards.
        track_zones: dict[int, tuple[str | None, str | None]] = {}

        for obj in frame_meta.object_items:
            r = obj.rect_params
            is_fire = obj.unique_component_id == FIRE_GIE_ID
            track_id = obj.object_id if obj.object_id != UNTRACKED else -1
            box = Box(
                cls=obj.class_id, conf=obj.confidence,
                left=r.left, top=r.top, width=r.width, height=r.height,
                track_id=track_id,
            )

            if is_fire:
                fire_boxes.append(box)
                if self.report_stats:
                    name = (self.fire_labels[box.cls] if box.cls < len(self.fire_labels)
                            else f"fire{box.cls}")
                    self.class_counts[name] = self.class_counts.get(name, 0) + 1
                r.border_width = 2
                r.border_color = osd.Color(*COLOR_FIRE)
                continue

            ppe_boxes.append(box)
            if self.report_stats:
                name = (self.ppe_labels[box.cls] if box.cls < len(self.ppe_labels)
                        else f"cls{box.cls}")
                self.class_counts[name] = self.class_counts.get(name, 0) + 1
            if box.cls != PPE_HUMAN:
                # PPE items are evidence, not findings — the person box carries the verdict.
                # Their default nvdsosd class labels ("helmet 2245") otherwise pile up on top of
                # the person labels and make a 20-tile view unreadable, so blank them.
                r.border_width = 1
                obj.text_params.display_text = b""
                continue

            if self.zone_severity and track_id != -1:
                zname, zsev = self.zone_of(stream_id, self._object_zones(obj))
                if zname is not None:
                    track_zones[track_id] = (zname, zsev)
                    if self.report_stats:
                        self._zone_hits[zname] = self._zone_hits.get(zname, 0) + 1

            if self.track_stats and track_id != -1:
                # Lifetime is accumulated in SECONDS, not frames. Frames are not comparable across
                # drop-frame-interval settings — at dfi=2 a 30-frame track covers twice the wall
                # time of a 30-frame track at dfi=0, so a frame count would make the slower
                # analytics rate look better for free. Seconds is the honest unit: it answers
                # "how long does one person keep one ID", which is what debouncing depends on.
                st = self._tracks.setdefault(stream_id, {})
                now_t = time.monotonic()
                rec = st.get(track_id)
                if rec is None:
                    st[track_id] = [now_t, now_t, 1]
                else:
                    rec[1] = now_t
                    rec[2] += 1

            verdict = pending.get(track_id)
            if verdict is None:
                # New track: nothing decided yet. Neutral until the debounce window fills.
                # Blank the label too — otherwise nvdsosd falls back to its own class name
                # ("human 2088") and a single un-adjudicated person breaks the visual language
                # of the display, where text means "a verdict was reached".
                r.border_width = 2
                r.border_color = osd.Color(0.6, 0.6, 0.6, 1.0)
                obj.text_params.display_text = b""
                continue
            self._paint(obj, r, verdict, track_id)

        # Fold this frame in and stash the result for the next one.
        verdicts = self.tracker.update(stream_id, ppe_boxes)
        self._pending[stream_id] = {
            v["box"].track_id: v for v in verdicts if v["box"].track_id != -1
        }

        # Publish state CHANGES only. `verdicts` is already debounced by rules.py, so a transition
        # here means the supermajority flipped, not that one frame looked different.
        # Events carry a ONE-BASED camera id, matching what the OSD paints ("CAM 01") and the
        # media filenames (cam01.mp4). DeepStream's source_id is zero-based, and letting that
        # leak into stored incidents meant the dashboard would say cam00 for the tile labelled
        # CAM 01 — a mismatch that only ever surfaces when someone is trying to find the camera.
        cam_id = stream_id + 1
        # Position in the SOURCE video, not wall-clock. The clip service uses this to find the
        # moment; see the note on Event.source_pts_ns.
        pts = getattr(frame_meta, "buffer_pts", None)
        if self._transitions is not None:
            self._transitions.update(cam_id, verdicts, track_zones, pts)
            self._transitions.expire()
        if self._oc_transitions is not None:
            self._oc_transitions.update(
                cam_id, self._frame_overcrowding(frame_meta, stream_id), pts)

        self._draw_status(batch_meta, frame_meta, stream_id, fire_boxes, pts)

    def report_track_stability(self) -> None:
        """Print per-ID lifetime statistics.

        Reported PERIODICALLY, from inside the reporting window, not once at the end. `--duration`
        calls `pipeline.stop()` from a timer thread but `pipeline.wait()` stays blocked in C++, so
        an end-of-run report is simply never reached — and the exit path that does work
        (`timeout --signal=KILL`) leaves no chance to print anything either. A periodic line
        survives both, at the cost of the reader wanting the LAST one.

        The question this answers: can the compliance state machine still debounce? `rules.py`
        needs a person to hold ONE id across its ~15-frame window before it will flip a verdict,
        so what matters is the fraction of tracks that survive long enough to be adjudicated at
        all. Tracks shorter than that never produce a verdict — they are pure churn, and at 20
        cameras they also inflate the violation counter as the same person is re-adjudicated
        under a new id.
        """
        if not self.track_stats:
            return
        lifetimes: list[float] = []
        frames: list[int] = []
        for tracks in self._tracks.values():
            for first_t, last_t, seen in tracks.values():
                lifetimes.append(last_t - first_t)
                frames.append(seen)
        if not lifetimes:
            print("[track] no tracked objects — nothing to report", flush=True)
            return

        lifetimes.sort()
        frames.sort()
        n = len(lifetimes)

        def pct(xs, q):
            return xs[min(len(xs) - 1, int(len(xs) * q))]

        # A track seen fewer times than the debounce window can never reach a verdict.
        window = self.tracker.window_frames
        adjudicable = sum(1 for f in frames if f >= window)
        ephemeral = sum(1 for f in frames if f <= 2)

        print(f"[track] {n} unique ids across {len(self._tracks)} streams", flush=True)
        print(f"[track] lifetime s : p50 {pct(lifetimes, 0.5):.2f}  p90 {pct(lifetimes, 0.9):.2f} "
              f"max {lifetimes[-1]:.2f}  mean {sum(lifetimes) / n:.2f}", flush=True)
        print(f"[track] frames/id  : p50 {pct(frames, 0.5)}  p90 {pct(frames, 0.9)} "
              f"max {frames[-1]}  mean {sum(frames) / n:.1f}", flush=True)
        print(f"[track] adjudicable (>={window} frames): {adjudicable}/{n} "
              f"({100.0 * adjudicable / n:.1f}%)   ephemeral (<=2 frames): {ephemeral}/{n} "
              f"({100.0 * ephemeral / n:.1f}%)", flush=True)

    def _paint(self, obj, rect, verdict: dict, track_id: int) -> None:
        r, g, b, a = verdict["color"]
        rect.border_color = osd.Color(r, g, b, a)
        rect.border_width = 4 if verdict["state"] == VIOLATION else 2

        # NOTE: an object's `text_params` is an osd.TextParams, which is NOT the same shape as a
        # standalone osd.Text — it exposes font_params / set_bg_clr / text_bg_clr, and touching
        # `.font` / `.bg_color` here raises AttributeError, which aborts the process.
        tp = obj.text_params
        tp.display_text = verdict["label"].encode("ascii", "ignore")
        fp = tp.font_params
        fp.name = osd.FontFamily.Serif
        fp.size = 10
        fp.color = osd.Color(1.0, 1.0, 1.0, 1.0)
        tp.set_bg_clr = True
        tp.text_bg_clr = osd.Color(r * 0.5, g * 0.5, b * 0.5, 0.8)

    def _draw_status(self, batch_meta, frame_meta, stream_id: int, fire_boxes: list[Box],
                     pts: int | None = None) -> None:
        """Per-tile banner: camera id, violation count, and a latched fire alert."""
        fire = self.tracker.update_fire(stream_id, fire_boxes, self.fire_labels)
        if self._fire_transitions is not None:
            # `pts` must be threaded through here too. It was passed to the PPE and overcrowding
            # detectors but not this one, so fire alerts — the most serious incident type — were
            # the only ones that never got an evidence clip.
            self._fire_transitions.update(stream_id + 1, fire, pts)   # 1-based, as above
        violations = self.tracker.violation_count(stream_id)

        parts = [f"CAM {stream_id + 1:02d}"]
        if violations:
            parts.append(f"{violations} PPE VIOLATION{'S' if violations > 1 else ''}")
        if fire:
            parts.append(f"** {fire['label']} **")

        if violations or fire:
            colour = COLOR_FIRE if fire else (1.0, 0.15, 0.1, 1.0)
        else:
            colour = (0.7, 0.7, 0.7, 1.0)

        display_meta = batch_meta.acquire_display_meta()
        text = osd.Text()
        text.display_text = "  |  ".join(parts).encode("ascii", "ignore")
        text.x_offset = 12
        text.y_offset = 12
        text.font.name = osd.FontFamily.Serif
        text.font.size = 14
        text.font.color = osd.Color(*colour)
        text.set_bg_color = True
        text.bg_color = osd.Color(0.0, 0.0, 0.0, 0.6)
        display_meta.add_text(text)
        frame_meta.append(display_meta)


# ---------------------------------------------------------------------------------------------
# pipeline construction
# ---------------------------------------------------------------------------------------------
def load_zone_severity(streams: int) -> dict[int, dict[str, str]]:
    """{camera_id (1-based): {zone_name: severity}} from configs/analytics/zones.yml.

    Read directly from the same YAML that generates the nvdsanalytics config, rather than from a
    second derived file — one source of truth means a zone cannot exist geometrically while being
    unknown to the severity logic, which would silently drop its escalation.
    """
    path = ROOT / "configs/analytics/zones.yml"
    if not path.exists():
        return {}
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from make_zones import severity_map  # noqa: E402  (shares the same_as resolution)
        return severity_map(yaml.safe_load(path.read_text()), streams)
    except Exception as e:  # noqa: BLE001
        print(f"[zones] could not load severities: {type(e).__name__}: {e}", flush=True)
        return {}


def resolve_sources(cfg: dict, count: int, mode: str) -> list[str]:
    if mode == "rtsp":
        # The environment wins over the config file, for both forms.
        #
        # A camera address is deployment configuration, not a property of the software: the same
        # image runs against a different fleet at every site, and configs/demo.yml ships a
        # CHANGE-ME placeholder precisely because it cannot know one. More pointedly, real camera
        # URLs carry credentials — `rtsp://user:pw@host/...` — and configs/demo.yml is committed,
        # so a fleet written in there is a password in the repository history forever. That is the
        # same reasoning that keeps HF_TOKEN and the Telegram credentials in .env, which
        # scripts/env.sh already sources into the environment this reads.
        #
        #   ISMS_RTSP_URLS   comma-separated, one per camera; position is the camera id
        #   ISMS_RTSP_BASE   overrides sources.rtsp_base, keeping rtsp_pattern
        env_urls = [u.strip() for u in os.environ.get("ISMS_RTSP_URLS", "").split(",") if u.strip()]

        # Explicit per-camera URLs win when present. This is the production path: real cameras
        # are individual devices with their own hostnames, credentials and vendor-specific paths
        # (`rtsp://user:pw@10.0.0.5/Streaming/Channels/101`), and no printf pattern spans a
        # mixed fleet. `rtsp_base` + `rtsp_pattern` stays for the demo, where twenty streams do
        # come off one server with numbered mounts.
        urls = env_urls or [u for u in (cfg["sources"].get("rtsp_urls") or []) if str(u).strip()]
        if urls:
            if len(urls) < count:
                where = "ISMS_RTSP_URLS" if env_urls else "sources.rtsp_urls"
                raise SystemExit(
                    f"{where} lists {len(urls)} camera(s) but {count} streams were "
                    f"requested — add the missing URLs or lower pipeline.streams")
            # Camera N is urls[N-1]: the position in this list IS the camera id the dashboard,
            # the zone config and the clip prefixes all key on.
            return [str(u).strip() for u in urls[:count]]
        base = (os.environ.get("ISMS_RTSP_BASE") or cfg["sources"]["rtsp_base"]).rstrip("/")
        pattern = cfg["sources"]["rtsp_pattern"]
        return [f"{base}/{pattern % (i + 1)}" for i in range(count)]

    files = sorted(glob.glob(str(ROOT / cfg["sources"]["file_glob"])))
    if not files:
        raise SystemExit(f"no media matching {cfg['sources']['file_glob']} — run scripts/make_streams.sh")
    if len(files) < count:
        raise SystemExit(f"need {count} clips, found {len(files)} — run scripts/make_streams.sh {count}")
    return [f"file://{os.path.abspath(f)}" for f in files[:count]]


def _detect_display() -> str | None:
    """First X display this user can actually TALK to, or None for headless.

    A socket in /tmp/.X11-unix proves an X server exists, not that we may connect to it — on a
    device sitting at the login screen, :0 belongs to the display manager's greeter and is
    root-owned with no xauth cookie for the service user. So each candidate is probed with
    `xdpyinfo` rather than assumed.

    Returning None matters as much as returning a display. A set-but-unusable DISPLAY makes
    nvbufsurftransform fail its EGL init and the whole pipeline refuses to reach PLAYING with
    "Could not get EGL display connection" — reported against nvinfer, which is not the cause.
    Unset, DeepStream takes the headless path and everything works.
    """
    import glob
    import subprocess
    for sock in sorted(glob.glob("/tmp/.X11-unix/X*")):
        disp = ":" + sock.rsplit("/X", 1)[-1]
        try:
            probe = subprocess.run(["xdpyinfo"], env={**os.environ, "DISPLAY": disp},
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue          # no xdpyinfo installed, or it hung — treat as unusable
        if probe.returncode == 0:
            return disp
    return None


# Incident types urgent enough to interrupt a camera's post-recording cooldown. The cooldown is
# only a disk-and-churn guard, and a fire is exactly the thing nobody should later discover had
# no video because the camera was resting. Kept as types rather than a severity test because
# severity is escalated by zone — a routine violation in a forklift aisle is `high`, and that
# should not preempt.
PREEMPT_TYPES = frozenset({"fire_alert", "hazard_alert"})

_RTP_TRANSPORT = {"auto": 7, "udp": 1, "udp-mcast": 2, "tcp": 4, "http": 10, "tls": 20}


def rtsp_props(cfg: dict) -> dict:
    """Reconnection and transport settings for live RTSP sources.

    **`rtsp-reconnect-interval` defaults to 0, which means never reconnect.** A camera that
    stops sending — a network blip, a PoE reset, a switch reboot — is then gone for the life of
    the pipeline: the source posts EOS, nvstreammux retires it, and that tile stays black with
    nothing in the log but a single "Could not read from resource". Observed here on 3 of 20
    streams (cam06, cam10, cam17) during a routine run, on a quiet LAN with all 20 publishers
    still healthy. On real cameras this is not an edge case, it is Tuesday.

    Transport defaults to TCP rather than the element's `udp-udpmcast-tcp`. Twenty 1080p streams
    is enough traffic that UDP datagram loss shows up as exactly the read failures above, and
    RTSP-over-TCP costs a little latency to remove a whole class of them. Set
    `sources.rtsp.transport: auto` to restore the element default.
    """
    r = (cfg.get("sources") or {}).get("rtsp") or {}
    transport = str(r.get("transport", "tcp")).lower()
    if transport not in _RTP_TRANSPORT:
        print(f"[rtsp] unknown transport {transport!r}, falling back to tcp", flush=True)
        transport = "tcp"
    return {
        # Force a reconnect when no data has arrived for this long.
        "rtsp-reconnect-interval": int(r.get("reconnect_interval_s", 10)),
        # And when the FIRST connection errors, so a camera that is slow to boot — or a demo
        # started before its source server — is retried instead of abandoned.
        "init-rtsp-reconnect-interval": int(r.get("init_reconnect_interval_s", 10)),
        # -1 = keep trying. A camera that has been down for an hour should still rejoin.
        "rtsp-reconnect-attempts": int(r.get("reconnect_attempts", -1)),
        "select-rtp-protocol": _RTP_TRANSPORT[transport],
        # Jitterbuffer. 100 ms (the default) is tight for anything crossing a real network.
        "latency": int(r.get("latency_ms", 200)),
        # Be explicit: the property help says smart record needs source-type-rtsp, and relying
        # on URI-scheme auto-detection makes evidence capture depend on a guess.
        "type": 2,
    }


def load_services_cfg() -> dict:
    path = ROOT / "configs/services.yml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def clip_settings(svc: dict) -> tuple[Path, int, int, int]:
    """(clips_dir, pre_roll_s, post_roll_s, cache_s) from services.yml.

    Read from the SAME keys `clip_service.py` uses, so an evidence clip is the same length
    whichever backend produced it and the two paths cannot drift apart.
    """
    clips = (svc.get("clips") or {})
    d = Path(clips.get("dir", "data/clips"))
    if not d.is_absolute():
        d = ROOT / d
    pre = int(clips.get("pre_roll_s", 6))
    return (d, pre, int(clips.get("post_roll_s", 6)),
            int(clips.get("cache_s", pre * 2 + 5)))


def _smart_record_props(cfg: dict, args) -> dict | None:
    """nvurisrcbin smart-record properties, or None when it is switched off."""
    if getattr(args, "no_smart_record", False):
        return None
    svc = load_services_cfg()
    clips_dir, pre_s, post_s, cache_s = clip_settings(svc)
    clips_dir.mkdir(parents=True, exist_ok=True)
    # The ring buffer has to hold the pre-roll PLUS however long the trigger took to reach the
    # recorder — see the lag compensation in SmartRecorder._dispatch, which spends that margin.
    if cache_s <= pre_s:
        print(f"[smart-rec] WARNING clips.cache_s={cache_s}s does not cover pre_roll_s={pre_s}s "
              f"— clips will be missing part of the lead-up", flush=True)
    return {
        # 2 = smart-rec-multi: respond to BOTH cloud messages and local/API events. 1 (cloud) is
        # the common copy-paste value and would ignore Pipeline.start_recording() entirely.
        "smart-record": 2,
        # The cache is the ONLY history that exists over RTSP; a cache shorter than the pre-roll
        # silently truncates the part of the clip an operator actually wants (the lead-up).
        "smart-rec-cache": cache_s,
        "smart-rec-dir-path": str(clips_dir),
        "smart-rec-container": 0,            # MP4
        "smart-rec-mode": 1,                 # video only — these streams carry no audio
        # Backstop if a stop event never arrives, so a stuck recording cannot grow without bound.
        "smart-rec-default-duration": pre_s + post_s,
    }


class SmartRecorder:
    """Evidence clips for RTSP sources, cut from nvurisrcbin's own ring buffer.

    ## Why this exists at all

    In `file` mode `clip_service.py` cuts the window out of `media/camNN.mp4` with `ffmpeg -c
    copy`, locating the moment by `source_pts_ns`. That is impossible over RTSP: the Jetson never
    has the source, only the live stream passing through it. Production cameras are separate
    network devices, so this is the REAL deployment shape, not a corner case — and smart record is
    the mechanism DeepStream provides for it. `nvurisrcbin` continuously caches the incoming
    ENCODED stream (`smart-rec-cache` seconds of it) and can dump a window out of that cache on
    demand: no decode, no re-encode, and the clip comes out at the full SOURCE frame rate rather
    than the `drop-frame-interval` analytics rate. Measured on this build: 1080p30 H.265 out of a
    15 fps analytics pipeline.

    It is RTSP-only, and not by choice — `smart-rec-container` says so in its own property help
    ("Sources must be of type source-type-rtsp") and file sources fail SILENTLY, returning a
    session id and writing nothing.

    ## Off the hot path

    `request()` is called from inside the DeepStream probe, i.e. on the GStreamer streaming
    thread, so it only puts a tuple on a bounded queue. A worker thread does the
    `start_recording()` call, exactly as `EventEmitter` defers its socket writes — the hot path
    must never wait on the clip layer.

    ## One recording per camera at a time, shared by every incident inside it

    The probe emits one transition per TRACK, but a clip belongs to a moment on a CAMERA: five
    people starting a violation together are five events and one piece of video. So a camera
    already recording does not start a second overlapping recording; the later events attach to
    the one in flight and all of them end up pointing at the same file. That mirrors what the
    store does with incidents, and it is why the pending list is a shared mutable list — events
    arriving mid-window land in the list the callback will read.

    **A camera is released by its CALLBACK, never by a timer.** An earlier version expired the
    window after `pre+post` seconds, which raced: the window opened up a fraction before the
    callback arrived, the next event started a fresh recording and replaced the pending list, and
    the in-flight callback then attached its clip to the wrong events while the incident that
    triggered it sat in `clip_state='recording'` forever. Measured on a 20-camera run: 159
    recordings started, 20 files produced, 4 incidents orphaned. Completion is the only honest
    signal that a source is free again; the deadline below is a watchdog, not the normal path.

    **A source must be explicitly stopped before it will record again**, and this is the part
    that is not in any documentation. `smart-rec-default-duration` ends the recording, writes the
    file and fires the callback — but leaves the element's session OPEN. Every later
    `start_recording()` on that source is then refused with `Element srcN is already recording`,
    printed by the element and invisible to the caller, which still returns a session id. Probed
    directly: refused at callback+5s and still refused at callback+50s, i.e. it never clears on
    its own. Without the `stop_recording()` in `_rearm()` below, each camera yields exactly ONE
    evidence clip per pipeline lifetime and every incident after the first silently gets none.

    The callback can also fire TWICE for one recording (same filename, same instant — once for
    the auto-stop and once for the explicit stop), so completions are de-duplicated by filename.

    ## An incident with no clip is a correct outcome

    Smart record can only run one session per camera at a time, so a camera cannot always be
    recording when an incident opens. When it cannot, the incident gets NO clip and says so —
    it is never pointed at a nearby file. A clip covers `[trigger-pre, trigger+post]` and
    nothing else; attaching one that ends before the incident began produces evidence of a
    different moment, which the VLM then adjudicates in good faith and gets wrong. `fire_alert`
    and `hazard_alert` preempt the cooldown so the incidents that most need video are the least
    likely to go without.
    """

    def __init__(self, pipeline: Pipeline, emitter, clips_dir: Path,
                 pre_s: int, post_s: int, streams: int, cooldown_s: float = 30.0,
                 cache_s: int = 17):
        self.pipeline = pipeline
        self.emitter = emitter
        self.clips_dir = clips_dir
        # The binding is start_recording(str, int, int, Callable[[RecordingInfo], None]) -> int:
        # both times are ints and a float raises TypeError.
        self.pre_s = max(1, int(pre_s))
        self.post_s = max(1, int(post_s))
        self.cache_s = max(self.pre_s, int(cache_s))
        self.streams = streams
        self.cooldown_s = float(cooldown_s)
        self._q: queue.Queue = queue.Queue(maxsize=512)
        self._lock = threading.Lock()
        self._active: set[int] = set()               # cameras with a recording in flight
        self._deadline: dict[int, float] = {}        # watchdog: callback overdue past this
        self._cooldown_until: dict[int, float] = {}  # cam_id -> monotonic, set on completion
        self._pending: dict[int, list[str]] = {}     # cam_id -> event_ids sharing that recording
        self._seen_files: set[str] = set()           # completions already handled (dedupe)
        self.started = 0
        self.completed = 0
        self.failed = 0
        self.attached = 0
        self.stalled = 0
        self.skipped = 0     # incidents that got no clip because the camera was cooling down
        self.lagged = 0      # triggers that reached the worker more than 1s late
        threading.Thread(target=self._run, name="smart-recorder", daemon=True).start()

    # -- called from the probe -------------------------------------------------------------
    def request(self, ev) -> None:
        """Ask for a clip covering this moment. Never blocks, never raises."""
        try:
            # The request time is carried, not read at dispatch: the worker can be several
            # seconds behind during a burst, and the pre-roll has to reach back to the moment
            # the incident actually happened rather than to whenever we got round to it.
            self._q.put_nowait(("record", ev.camera_id,
                                (ev.event_id, ev.type, ev.severity, time.monotonic())))
        except queue.Full:
            self.failed += 1

    def stats(self) -> dict[str, int]:
        return {"started": self.started, "done": self.completed,
                "attached": self.attached, "failed": self.failed,
                "stalled": self.stalled, "no_clip": self.skipped,
                "lagged": self.lagged}

    # -- worker ---------------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            try:
                kind, camera_id, payload = self._q.get()
            except Exception:  # noqa: BLE001
                continue
            try:
                if kind == "rearm":
                    self._rearm(camera_id)
                else:
                    self._dispatch(camera_id, *payload)
            except Exception as e:  # noqa: BLE001
                self.failed += 1
                print(f"[smart-rec] {kind} failed cam{camera_id:02d}: "
                      f"{type(e).__name__}: {e}", flush=True)

    def _rearm(self, camera_id: int) -> None:
        """Close the finished session so this source can record again.

        Done on the worker rather than inside the completion callback, which runs on a
        DeepStream thread that should not be held. Only after the stop lands is the camera
        marked free — releasing it earlier would let the next event issue a `start_recording()`
        that the element silently refuses.
        """
        try:
            self.pipeline.stop_recording(f"src{camera_id - 1}")
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            print(f"[smart-rec] cam{camera_id:02d} stop_recording failed: "
                  f"{type(e).__name__}: {e}", flush=True)
        with self._lock:
            self._active.discard(camera_id)
            self._deadline.pop(camera_id, None)
            self._cooldown_until[camera_id] = time.monotonic() + self.cooldown_s

    def _dispatch(self, camera_id: int, event_id: str, etype: str = "",
                  severity: str = "", requested_at: float | None = None) -> None:
        if not (1 <= camera_id <= self.streams):
            return
        now = time.monotonic()
        total = self.pre_s + self.post_s
        with self._lock:
            if camera_id in self._active:
                if now < self._deadline.get(camera_id, 0.0):
                    # A recording is in flight RIGHT NOW, so it genuinely covers this moment —
                    # attaching is truthful. This is the only sharing that is.
                    self._pending.setdefault(camera_id, []).append(event_id)
                    self.attached += 1
                    return
                # Watchdog. The callback is long overdue, so treat the session as lost rather
                # than leaving this camera unable to record for the rest of the run. The
                # incidents waiting on it are released to `clip_service`'s stuck-row sweep,
                # which marks them failed with a reason.
                self._active.discard(camera_id)
                self._pending.pop(camera_id, None)
                self.stalled += 1
                print(f"[smart-rec] cam{camera_id:02d} recording never completed — releasing",
                      flush=True)
            if now < self._cooldown_until.get(camera_id, 0.0) and etype not in PREEMPT_TYPES:
                # Cooling down, and this is not urgent enough to interrupt it. The incident gets
                # NO clip, and that is the correct answer.
                #
                # An earlier version pointed it at the clip recorded just before, to reach 100%
                # coverage. That was wrong in the way that matters: a clip only covers
                # [trigger-pre, trigger+post], and anything arriving during the cooldown is by
                # definition after that window has closed — so the attached video could never
                # contain the incident. It was caught in the field, not by a test: a SMOKE alert
                # on cam14 carried a clip ending 10.6 s before the smoke was detected, the VLM
                # dutifully reported "no flames or smoke visible", and a CRITICAL fire alert was
                # rejected on footage of a different moment. Coverage is not the goal; evidence
                # that shows the incident is.
                self.skipped += 1
                return
        with self._lock:
            self._active.add(camera_id)
            # Generous grace: the file is only finalised after the full window has elapsed.
            self._deadline[camera_id] = now + total + 30.0
            self._pending[camera_id] = [event_id]

        # Reach back past the dispatch lag as well as the pre-roll. `start_time` is measured
        # from NOW, not from when the incident happened, so a trigger that waited behind
        # nineteen other cameras would otherwise produce a clip that begins after the moment it
        # is meant to document. Measured before this compensation: 7 of 29 incidents had clips
        # starting up to 11.9 s AFTER the incident — evidence of the aftermath, not the event.
        #
        # Bounded by the ring buffer, which is all the history that exists; a lag longer than
        # the cache cannot be recovered, and the clip then starts as early as it can.
        lag = max(0.0, now - requested_at) if requested_at else 0.0
        start_back = int(min(self.cache_s, self.pre_s + lag))
        sid = self.pipeline.start_recording(f"src{camera_id - 1}", start_back,
                                            total + start_back - self.pre_s, self._on_done)
        self.started += 1
        if lag > 1.0:
            self.lagged += 1
        print(f"[smart-rec] cam{camera_id:02d} recording session={sid} "
              f"(-{start_back}s +{self.post_s}s, lag {lag:.1f}s) for {event_id[:8]}", flush=True)

    # -- completion callback (called from a DeepStream thread) -----------------------------
    def _on_done(self, info) -> None:
        """Publish the finished clip against every incident that was waiting on it.

        Wrapped whole: this is invoked from C++, so an escaping exception aborts the process
        rather than raising — the same trap the metadata probe is written around.
        """
        cam_id = 0
        try:
            name = getattr(info, "file_name", "") or ""
            directory = getattr(info, "file_directory", "") or str(self.clips_dir)
            # The camera comes from the filename, not the session id. Every source sets its own
            # `smart-rec-file-prefix` (cam07_...), and session ids restart per source — the
            # 20-stream run returned session=0 for all twenty — so the prefix is the only
            # unambiguous way back to the camera.
            #
            # Split on the separator rather than slicing two characters: nvurisrcbin appends
            # `_<sessionid>_<counter>_<timestamp>_<pid>`, so the prefix is everything before the
            # first underscore and a fixed [3:5] slice would silently read cam100 as camera 10.
            head = name.split("_", 1)[0]
            if head[:3] == "cam" and head[3:].isdigit():
                cam_id = int(head[3:])
            with self._lock:
                # One recording can fire this callback twice with the same filename — once for
                # the auto-stop, once for our explicit stop. Claiming the pending list twice
                # would attach the clip and then log the same file as unwanted.
                duplicate = name in self._seen_files
                if not duplicate:
                    self._seen_files.add(name)
                    event_ids = self._pending.pop(cam_id, [])
            if duplicate:
                return
            path = Path(directory) / name
            try:
                uri = str(path.relative_to(ROOT))
            except ValueError:
                # Stored relative to the repo root when possible so the database stays portable
                # between the device and a laptop copy; absolute otherwise.
                uri = str(path)
            self.completed += 1
            for eid in event_ids:
                self.emitter.emit_raw({"kind": "clip_ready", "event_id": eid, "clip_uri": uri})
            if event_ids:
                print(f"[smart-rec] cam{cam_id:02d} -> {name} "
                      f"({path.stat().st_size / 1e6:.1f} MB) for {len(event_ids)} incident(s)",
                      flush=True)
            else:
                print(f"[smart-rec] {name} finished with no incident waiting on it", flush=True)
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            print(f"[smart-rec] callback error: {type(e).__name__}: {e}", flush=True)
        finally:
            # ALWAYS re-arm, on every exit path including the error one. A camera whose session
            # is never stopped is a camera that silently produces no further evidence for the
            # rest of the run, which is the worst possible failure for this feature — it looks
            # exactly like "no incidents happened".
            if cam_id:
                try:
                    self._q.put_nowait(("rearm", cam_id, None))
                except queue.Full:
                    self.failed += 1


class FrameCapture(BufferOperator):
    """Crops the violating subject out of the LIVE frame, at the instant it was flagged.

    ## Why this and not a crop from the evidence clip

    Both other routes are eliminated by measurement, not preference. Cropping the source at the
    incident PTS is meaningless over RTSP — `source_pts_ns` is running time on a live stream and
    indexes no file. Cropping the smart-record clip fails at EVERY offset: an offset sweep found
    the subject in 1 of 4 cases at best and 0 of 4 at most offsets, because a stored bbox and a
    clip describe different moments by construction (the bbox is from the transition that opened
    the incident; the clip is recorded later, and incidents merge many tracks over time).

    Here the frame and the box come from the same instant, so that mismatch cannot arise.

    ## The valve is the whole reason this is affordable

    Extraction needs an RGB NVMM surface, which needs an `nvvideoconvert` the main path cannot
    supply (`nvdsosd` takes NV12 or RGBA only). So the branch is a tee behind a valve that DROPS
    by default: with the gate shut no buffer reaches the convert and `handle_buffer` is never
    called at all — measured at 20 streams as 481.3 vs 484.6 fps, i.e. nothing. The gate opens
    only while a crop is owed, which makes the captured frame about one frame later than the
    detection (~66 ms at 15 fps analytics) rather than exact. That is a different order of error
    from the clip attempt this replaces.

    Cost when it does run: 8.58 ms p50 end to end. The crop is taken on the GPU and only the crop
    is copied to host — moving the whole 1080p frame would be ~6 MB a time.

    ## One crop per incident, not per transition

    The probe emits an opening event per TRACK, and on this footage that is a near-continuous
    stream: 2360 crops in three minutes across 20 cameras. Each request opens the gate, so at
    that rate the gate is NEVER shut and the RGB convert runs on every frame of every stream —
    precisely the cost the valve exists to avoid. Measured that way: 174-235 fps against a 300
    fps realtime target, i.e. the pipeline stops keeping up. The gating is only as good as the
    trigger rate behind it.

    So requests are rate-limited per (camera, type) on the store's own `merge_window_s`. Not an
    arbitrary throttle: inside that window the store MERGES transitions into a single incident,
    so the second and later crops describe an incident that already has one. The first request
    after a quiet period is the one that opens the incident, and that is the crop worth taking.
    """

    def __init__(self, pipeline: Pipeline, emitter, crops_dir: Path, streams: int,
                 cooldown_s: float = 30.0, pad: float = 0.35, width: int = 448,
                 valve: str = "cap_valve"):
        super().__init__()
        # Imported here, not at module scope: the pipeline must still run on plain system python
        # when capture is off, and torch only exists in the --system-site-packages venv.
        import torch  # noqa: PLC0415
        from torch.utils.dlpack import from_dlpack  # noqa: PLC0415
        import cv2  # noqa: PLC0415
        self._from_dlpack = from_dlpack
        self._cv2 = cv2
        self._torch = torch
        # Whether reading the extracted surface needs an explicit device sync — see the note in
        # handle_buffer. Only on a discrete GPU, so it is decided once here rather than per frame.
        self._needs_sync = bool(torch.cuda.is_available()) and not Path("/etc/nv_tegra_release").exists()
        self.pipeline = pipeline
        self.emitter = emitter
        self.crops_dir = crops_dir
        self.streams = streams
        self.pad = pad
        self.width = width
        self.valve = valve
        self.cooldown_s = float(cooldown_s)
        self._cool: dict[tuple[int, str], float] = {}
        crops_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pending: dict[int, list[tuple[str, tuple]]] = {}   # 0-based stream -> [(id, bbox)]
        self._gate_open = False
        self.captured = 0
        self.failed = 0
        self.skipped = 0
        self.merged = 0

    def stats(self) -> dict[str, int]:
        return {"captured": self.captured, "failed": self.failed, "no_bbox": self.skipped,
                "merged": self.merged}

    # -- called from the probe thread -------------------------------------------------------
    def request(self, ev) -> None:
        """Ask for a crop of this incident's subject. Never blocks, never raises."""
        bbox = getattr(ev, "bbox", None)
        if not bbox or len(bbox) != 4:
            # Overcrowding and fire have no subject box — the scene is the evidence, and
            # context frames already serve them.
            self.skipped += 1
            return
        sid = ev.camera_id - 1
        if not (0 <= sid < self.streams):
            return
        now = time.monotonic()
        key = (ev.camera_id, ev.type)
        with self._lock:
            if now < self._cool.get(key, 0.0):
                # Same situation as one already captured; the store will merge this transition
                # into the incident that crop belongs to.
                self.merged += 1
                return
            self._cool[key] = now + self.cooldown_s
            self._pending.setdefault(sid, []).append(
                (ev.event_id, tuple(bbox), ev.type, ev.track_id))
            need_open = not self._gate_open
            self._gate_open = True
        if need_open:
            self._gate(False)          # valve `drop` False == gate OPEN

    def _gate(self, drop: bool) -> None:
        try:
            self.pipeline[self.valve].set({"drop": drop})
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            print(f"[capture] valve {'close' if drop else 'open'} failed: "
                  f"{type(e).__name__}: {e}", flush=True)

    # -- streaming thread, only while the gate is open ---------------------------------------
    def handle_buffer(self, buffer) -> None:
        # Wrapped whole: an exception escaping a probe is SIGABRT, not a traceback.
        try:
            with self._lock:
                if not self._pending:
                    return
            for frame in buffer.batch_meta.frame_items:
                sid = frame.source_id
                with self._lock:
                    reqs = self._pending.pop(sid, None)
                if not reqs:
                    continue
                # batch_id, NOT source_id: extract() indexes the position in this batch, and a
                # partial batch makes the two diverge.
                gpu = self._from_dlpack(buffer.extract(frame.batch_id))
                # `extract()` hands back the surface itself, not a copy, and `from_dlpack` wraps it
                # without copying either — so the device-to-host copy in _write races the
                # `nvvideoconvert` that produces it. They are on different CUDA streams and the raw
                # capsule carries no stream for DLPack's exchange protocol to synchronise on.
                #
                # On Jetson the memory is unified and this almost never surfaces. On a discrete GPU
                # it does: measured here, 12 of 81 crops came back as flat green — R and B at zero,
                # G saturated — which is a partially written surface, not a colour-format error.
                # Intermittent and silent, and the VLM answers about them in good faith.
                #
                # A full device sync is the blunt instrument, and it is the right one: this runs
                # only while the valve is open, which is once per incident rather than per frame.
                if self._needs_sync:
                    self._torch.cuda.synchronize()
                for event_id, bbox, etype, track_id in reqs:
                    self._write(gpu, bbox, event_id, sid + 1, etype, track_id)
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            if self.failed <= 3:
                print(f"[capture] {type(e).__name__}: {e}", flush=True)
        finally:
            with self._lock:
                done = not self._pending
                if done and self._gate_open:
                    self._gate_open = False
                else:
                    done = False
            if done:
                self._gate(True)

    def _write(self, gpu, bbox, event_id: str, camera_id: int, etype: str,
               track_id: int) -> None:
        h, w = int(gpu.shape[0]), int(gpu.shape[1])
        left, top, bw, bh = (float(v) for v in bbox)
        px, py = bw * self.pad, bh * self.pad
        x0 = max(0, min(w - 2, int(left - px)))
        y0 = max(0, min(h - 2, int(top - py)))
        x1 = max(x0 + 2, min(w, int(left + bw + px)))
        y1 = max(y0 + 2, min(h, int(top + bh + py)))
        # Crop on the GPU, then copy only the crop across.
        host = gpu[y0:y1, x0:x1].contiguous().cpu().numpy()
        if self.width and host.shape[1] > self.width:
            scale = self.width / host.shape[1]
            host = self._cv2.resize(
                host, (self.width, max(1, int(host.shape[0] * scale))),
                interpolation=self._cv2.INTER_AREA)
        ok, enc = self._cv2.imencode(".jpg", host[:, :, ::-1],   # RGB surface -> BGR for cv2
                                     [self._cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            self.failed += 1
            return
        # Named for the incident, which is the whole contract with the reasoning service: it
        # looks for this path and needs no schema change or extra message to find it.
        path = self.crops_dir / f"{event_id}.jpg"
        path.write_bytes(enc.tobytes())
        self.captured += 1
        # Report it the way clips are reported. The store resolves this to an incident through
        # `incident_tracks` on (camera, type, track) rather than by event_id: the crop is named
        # for a TRANSITION, and measured on a live run only 1 of 138 transitions is itself an
        # incident row — every other crop would be discarded on an event_id match.
        try:
            uri = str(path.relative_to(ROOT))
        except ValueError:
            uri = str(path)
        self.emitter.emit_raw({"kind": "crop_ready", "event_id": event_id,
                               "camera_id": camera_id, "type": etype, "track_id": track_id,
                               "crop_uri": uri})


def tile_layout(count: int, cfg: dict) -> tuple[int, int]:
    rows, cols = cfg["pipeline"]["tiler"].get("rows"), cfg["pipeline"]["tiler"].get("cols")
    if rows and cols:
        return int(rows), int(cols)
    cols = int(count ** 0.5 + 0.999)
    rows = (count + cols - 1) // cols
    return rows, cols


def _display_q(p: Pipeline, name: str):
    """A one-deep leaky queue in front of a display or encode sink.

    Without this the sink back-pressures the entire pipeline: rendering 20 tiles on a GPU that is
    already at 99% dropped end-to-end throughput from 634 fps to 39 fps, i.e. the video wall was
    throttling the analytics. `leaky=2` discards the OLDEST buffer, so the sink always shows the
    most recent frame and simply skips whatever it could not keep up with, while decode, inference
    and tracking continue at full rate.

    This is the point the display must never be able to slow down the thing it is displaying.
    """
    p.add("queue", name, {"leaky": 2, "max-size-buffers": 1,
                          "max-size-bytes": 0, "max-size-time": 0})
    return name


def _q(p: Pipeline, name: str, size: int = 4):
    """A thread boundary.

    GStreamer runs a chain of elements in a single streaming thread unless a queue splits it.
    With no queues, decode, inference, tracking and compositing serialise even though they use
    different hardware units — measured as a ~7x throughput loss on this pipeline. `leaky=2`
    (downstream) keeps a slow stage from back-pressuring the decoders in a live deployment.
    """
    p.add("queue", name, {"leaky": 2, "max-size-buffers": size,
                          "max-size-bytes": 0, "max-size-time": 0})
    return name


def build(cfg: dict, args) -> tuple[Pipeline, SafetyOverlay]:
    count = args.streams
    mode = args.source
    live = mode == "rtsp"
    uris = resolve_sources(cfg, count, mode)
    rows, cols = tile_layout(count, cfg)

    mux = cfg["pipeline"]["muxer"]
    tiler = cfg["pipeline"]["tiler"]

    # Labels are checked into the repo (they are the class-index contract the parser and nvinfer
    # configs depend on), but don't hard-fail if one is missing — rules.py has a sane fallback.
    labels_path = ROOT / "models/fire/config/labels.txt"
    fire_labels = labels_path.read_text().split() if labels_path.exists() else []
    if not fire_labels:
        print(f"[warn] {labels_path} missing — falling back to built-in fire class names",
              flush=True)
    ppe_labels_path = ROOT / "models/ppe/config/labels.txt"
    ppe_labels = ppe_labels_path.read_text().split() if ppe_labels_path.exists() else []

    p = Pipeline("safety")

    # ---- sources ----
    # drop-frame-interval decouples the ANALYTICS rate from the INGEST rate. `=3` means the
    # decoder emits every 3rd frame, i.e. 10 fps out of a 30 fps stream.
    #
    # This is the headroom lever for Phase 2, and it is a different lever from nvinfer's
    # `interval`: nvinfer's slows inference ONLY, leaving the tracker, tiler and OSD running at
    # full rate — and Phase 1 measured the TRACKER, not inference, as the bottleneck, which is why
    # that lever barely moved (1.05x -> 1.12x) while costing 30% of no-helmet detections.
    # Dropping at the decoder slows EVERYTHING downstream, tracker included.
    #
    # Two consequences worth knowing rather than discovering: the live tiled view becomes 10 fps
    # (fine for a monitoring wall), and tracking association gets harder because objects move ~3x
    # further between frames. NvSORT's Kalman motion model should cope, but track-ID stability is
    # to be re-verified, not assumed — 15 fps (=2) is the fallback if it degrades.
    #
    # Evidence clips are NOT affected: smart-record captures the incoming ENCODED stream ahead of
    # the decoder, so recordings stay a full 30 fps.
    dfi = int(args.drop_frame_interval if args.drop_frame_interval is not None
              else cfg["pipeline"].get("drop_frame_interval", 0) or 0)
    # Smart record, RTSP only. `nvurisrcbin` keeps `smart-rec-cache` seconds of the incoming
    # ENCODED stream and can dump a window out of it on demand, which is the only way to get an
    # evidence clip when the source is a camera on the network rather than a file on disk.
    # File sources fail SILENTLY here (a session id is returned and nothing is written), so this
    # is gated on the mode rather than left on and hoped for.
    sr = _smart_record_props(cfg, args) if mode == "rtsp" else None
    # Reconnection and transport. Live sources only: these properties are meaningless for
    # file:// and setting `type=rtsp` on one would be a lie the element acts on.
    rtsp = rtsp_props(cfg) if mode == "rtsp" else None
    if rtsp is not None:
        print(f"[rtsp] transport={[k for k, v in _RTP_TRANSPORT.items() if v == rtsp['select-rtp-protocol']][0]}"
              f" reconnect={rtsp['rtsp-reconnect-interval']}s"
              f" latency={rtsp['latency']}ms", flush=True)

    for i, uri in enumerate(uris):
        props = {"uri": uri}
        if rtsp is not None:
            props.update(rtsp)
        if mode == "file" and cfg["sources"].get("loop", True):
            props["file-loop"] = 1
        if dfi > 0:
            props["drop-frame-interval"] = dfi
        if sr is not None:
            props.update(sr)
            # Unique per source: the completion callback reports a filename but not which camera
            # produced it, so the prefix is what carries that back. The property help says the
            # same thing ("for unique file names every source must be provided with a unique
            # prefix"), and sharing one prefix would also let two cameras overwrite each other.
            props["smart-rec-file-prefix"] = f"cam{i + 1:02d}"
        p.add("nvurisrcbin", f"src{i}", props)

    p.add("nvstreammux", "mux", {
        "batch-size": count,
        "width": mux["width"],
        "height": mux["height"],
        "batched-push-timeout": mux["batched_push_timeout_us"],
        "live-source": 1 if live else 0,
    })
    for i in range(count):
        p.link((f"src{i}", "mux"), ("", "sink_%u"))

    ppe_cfg = str(ROOT / cfg["inference"]["ppe"]["config"])
    fire_cfg = str(ROOT / cfg["inference"]["fire"]["config"])
    fire_on = cfg["inference"]["fire"].get("enabled", True)

    # ---- inference ----
    if args.topology == "parallel" and fire_on:
        p.add("tee", "infer_tee")
        p.add("queue", "q_ppe", {"leaky": 2, "max-size-buffers": 4})
        p.add("queue", "q_fire", {"leaky": 2, "max-size-buffers": 4})
        p.add("nvinfer", "pgie_ppe", {"config-file-path": ppe_cfg, "batch-size": ENGINE_BATCH})
        p.add("nvinfer", "pgie_fire", {"config-file-path": fire_cfg, "batch-size": ENGINE_BATCH})
        p.add("nvdsmetamux", "metamux", {"config-file": str(ROOT / "configs/metamux.txt")})

        p.link("mux", "infer_tee")
        p.link(("infer_tee", "q_ppe"), ("src_%u", ""))
        p.link(("infer_tee", "q_fire"), ("src_%u", ""))
        p.link("q_ppe", "pgie_ppe")
        p.link("q_fire", "pgie_fire")
        # sink_0 must be the PPE branch — metamux.txt sets it as the active pad.
        p.link(("pgie_ppe", "metamux"), ("", "sink_%u"))
        p.link(("pgie_fire", "metamux"), ("", "sink_%u"))
        head = "metamux"
    else:
        p.add("nvinfer", "pgie_ppe", {"config-file-path": ppe_cfg, "batch-size": ENGINE_BATCH})
        p.link("mux", _q(p, "q_mux"), "pgie_ppe")
        head = "pgie_ppe"
        if fire_on:
            p.add("nvinfer", "pgie_fire", {"config-file-path": fire_cfg, "batch-size": ENGINE_BATCH})
            p.link("pgie_ppe", _q(p, "q_ppe_out"), "pgie_fire")
            head = "pgie_fire"

    # ---- tracker ----
    # Not decoration: the compliance state machine needs stable IDs to debounce verdicts.
    if cfg["tracker"].get("enabled", True):
        tcfg = cfg["tracker"]["config"]
        tcfg = tcfg if os.path.isabs(tcfg) else str(ROOT / tcfg)
        p.add("nvtracker", "tracker", {
            "ll-config-file": tcfg,
            "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
            "tracker-width": 960, "tracker-height": 544,
        })
        p.link(head, _q(p, "q_pre_track"), "tracker")
        head = "tracker"

    # ---- zone analytics ----
    # MUST come after the tracker: nvdsanalytics keys line-crossing and direction on object ids,
    # so without stable ids it can do ROI presence but nothing temporal.
    #
    # The config is GENERATED from configs/analytics/zones.yml by scripts/make_zones.py — its
    # section suffixes are source indices, so it is regenerated whenever the stream count or the
    # zone geometry changes.
    an_cfg = ROOT / "configs/analytics/analytics.txt"
    if args.zones and an_cfg.exists():
        p.add("nvdsanalytics", "analytics", {"config-file": str(an_cfg)})
        p.link(head, _q(p, "q_pre_analytics"), "analytics")
        head = "analytics"
        print(f"[zones] nvdsanalytics enabled ({an_cfg.name})", flush=True)
    elif args.zones:
        print(f"[zones] {an_cfg} missing — run scripts/make_zones.py --generate", flush=True)

    probe_point = head  # last element that still carries one frame_meta PER STREAM

    # ---- frame-capture branch (optional) -----------------------------------------------------
    # To crop the violating subject out of the live frame we need the PIXELS, and
    # `Buffer.extract()` will only hand back a tensor for an RGB surface in NVMM. The main path
    # cannot simply be converted: `nvdsosd` accepts NV12 or RGBA and nothing else, so an RGB main
    # path breaks rendering outright.
    #
    # Hence a tee. The capture branch sits behind a `valve` that DROPS by default, so when no
    # incident is pending the branch costs a tee and a closed gate — the convert never runs. The
    # probe opens it for a moment when it needs a frame, which makes the captured frame about one
    # frame later than the detection rather than exact. At 15 fps analytics that is ~66 ms, which
    # is a different order of error entirely from the clip-crop attempt this replaces (that one
    # could not find the subject at ANY offset, because a stored bbox and a smart-record clip
    # describe different moments by construction).
    if args.rgb_capture:
        p.add("tee", "cap_tee")
        p.link(head, "cap_tee")
        p.add("queue", "q_cap", {"leaky": 2, "max-size-buffers": 1,
                                 "max-size-bytes": 0, "max-size-time": 0})
        # drop-mode=1 (forward-sticky-events) so caps and segment still reach the convert while
        # the gate is shut; with the default drop-all the branch has no caps to renegotiate from
        # when it opens.
        p.add("valve", "cap_valve", {"drop": True, "drop-mode": 1})
        # compute-hw=1 (GPU) is REQUIRED, not a tuning preference. Jetson's nvvideoconvert
        # defaults to VIC, which cannot produce RGB from NV12: the branch builds and links fine
        # and then every buffer dies with "buffer transform failed" the moment the valve opens.
        # Cost nothing to diagnose only because the valve made it look like a valve problem.
        p.add("nvvideoconvert", "cap_conv", {"compute-hw": 1})
        p.add("capsfilter", "cap_caps",
              {"caps": "video/x-raw(memory:NVMM), format=RGB"})
        # async=0: there is a tee in the graph, and a sink left on async=1 parks the whole
        # pipeline in PAUSED with no video and no error (trap 8).
        p.add("fakesink", "cap_sink", {"sync": 0, "qos": 0, "async": 0})
        p.link(("cap_tee", "q_cap"), ("src_%u", ""))
        p.link("q_cap", "cap_valve", "cap_conv", "cap_caps", "cap_sink")
        head = "cap_tee"
        print("[capture] RGB frame-capture branch added (valve closed until a crop is needed)",
              flush=True)

    def _link_downstream(first_q: str, *rest: str) -> None:
        """Link the rendering chain, taking a request pad when the capture tee is in the way."""
        if args.rgb_capture:
            p.link(("cap_tee", first_q), ("src_%u", ""))
            p.link(first_q, *rest)
        else:
            p.link(head, first_q, *rest)

    # ---- tiler + osd ----
    if args.no_osd:
        # Diagnostic: straight from inference to a fakesink. Everything between here and the
        # sink (tile compositing, NV12->RGBA conversion, OSD draw) is skipped, so the resulting
        # fps is the decode+inference ceiling of the assembled pipeline.
        p.add("fakesink", "sink", {"sync": 0, "qos": 0, "async": 0})
        p.link(head, "sink")
        overlay = SafetyOverlay(cfg, fire_labels, ppe_labels)
        overlay.report_fps = args.fps
        overlay.report_stats = args.stats
        overlay.count_only = args.no_probe
        overlay.track_stats = args.track_stats
        overlay.streams = count
        p.attach(probe_point, Probe("safety-overlay", overlay))
        return p, overlay

    # compute-hw: 0=Default (VIC on Jetson), 1=GPU, 2=VIC.
    #
    # The default matters enormously here. VIC is a fixed-function block that is far slower than
    # the GPU for 1080p scaling and colour conversion, and it serialises — measured as the single
    # largest cost in the whole pipeline (see bench/findings.md). Both the tiler and the OSD
    # colour convert are pinned to GPU unless demo.yml says otherwise.
    render = cfg.get("render", {}) or {}
    compute_hw = {"gpu": 1, "vic": 2, "default": 0}[str(render.get("compute_hw", "gpu")).lower()]

    p.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": cols,
        "width": tiler["width"], "height": tiler["height"],
        "compute-hw": compute_hw,
    })

    # This install ships `nvdsosd`, not the `nvosdbin` wrapper. nvdsosd's GPU mode accepts NV12
    # as well as RGBA, so the RGBA conversion is optional — and skipping it removes a full
    # 1920x1080 colour convert per batch.
    if render.get("osd_rgba", False):
        p.add("nvvideoconvert", "osd_conv", {"compute-hw": compute_hw})
        p.add("capsfilter", "osd_caps", {"caps": "video/x-raw(memory:NVMM), format=RGBA"})
        p.add("nvdsosd", "osd", {"process-mode": 1})
        _link_downstream(_q(p, "q_pre_tile"), "tiler", _q(p, "q_pre_conv"),
                         "osd_conv", "osd_caps", "osd")
    elif render.get("osd", True):
        p.add("nvdsosd", "osd", {"process-mode": 1})
        _link_downstream(_q(p, "q_pre_tile"), "tiler", _q(p, "q_pre_osd"), "osd")
    else:
        # Diagnostic only: tiler with no OSD, to separate compositing cost from draw cost.
        p.add("identity", "osd", {"silent": 1})
        _link_downstream(_q(p, "q_pre_tile"), "tiler", _q(p, "q_pre_osd"), "osd")

    # ---- sinks ----
    want_display = cfg["sinks"]["display"].get("enabled", True) and not args.no_display
    # Configured-on is not the same as available. Dropping the sink costs a local preview;
    # keeping it against an unreachable X server costs the entire pipeline (see _detect_display).
    if want_display and not (os.environ.get("DISPLAY") or _detect_display()):
        print("!! display sink enabled but no reachable X server — running headless.\n"
              "!!   The dashboard and RTSP output are unaffected. Set DISPLAY_NUM in .env "
              "to force one.", flush=True)
        want_display = False
    want_rtsp = cfg["sinks"]["rtsp_out"].get("enabled", False) or args.rtsp_out

    # nv3dsink is Tegra. DGX Spark is aarch64 but SBSA/dGPU-class — nveglglessink, same as x86.
    # Do not key this on architecture: `platform.processor()` is often empty on Linux aarch64.
    sink_type = "nv3dsink" if os.path.exists("/etc/nv_tegra_release") else "nveglglessink"
    # async=0 everywhere: with a tee in the graph, a sink left on async=1 parks the pipeline in
    # PAUSED forever with no error. Set unconditionally so toggling sinks can't reintroduce it.
    common = {"sync": 0, "qos": 0, "async": 0}

    if want_display and want_rtsp:
        p.add("tee", "sink_tee")
        _display_q(p, "q_disp")
        p.add("queue", "q_enc", {"leaky": 2, "max-size-buffers": 2})
        p.link("osd", "sink_tee")
        p.link(("sink_tee", "q_disp"), ("src_%u", ""))
        p.link(("sink_tee", "q_enc"), ("src_%u", ""))
        p.add(sink_type, "sink", common)
        p.link("q_disp", "sink")
        _add_rtsp_branch(p, cfg, "q_enc", common)
    elif want_rtsp:
        p.add("queue", "q_enc", {"leaky": 2, "max-size-buffers": 4})
        p.link("osd", "q_enc")
        _add_rtsp_branch(p, cfg, "q_enc", common)
    elif want_display:
        p.add(sink_type, "sink", common)
        p.link("osd", _display_q(p, "q_disp"), "sink")
    elif args.snapshot:
        # Visual verification path: render the OSD output to JPEGs so the boxes, colours and
        # labels can actually be looked at, rather than inferred from counters.
        p.add("nvvideoconvert", "snap_conv")
        p.add("jpegenc", "snap_enc", {"quality": 90})
        p.add("multifilesink", "sink", {"location": args.snapshot, "sync": 0, "async": 0})
        p.link("osd", "snap_conv", "snap_enc", "sink")
    else:
        p.add("fakesink", "sink", common)
        p.link("osd", "sink")

    overlay = SafetyOverlay(cfg, fire_labels, ppe_labels)
    overlay.report_fps = args.fps
    overlay.report_stats = args.stats
    # Diagnostic: same probe, same fps line format, but no metadata touched — isolates the cost
    # of Python metadata handling from the cost of the GStreamer pipeline itself.
    overlay.count_only = args.no_probe
    overlay.track_stats = args.track_stats
    overlay.streams = count
    if args.zones:
        overlay.zone_severity = load_zone_severity(count)
        overlay.dump_analytics = args.dump_analytics
        overlay.dump_analytics_frame = args.dump_analytics

    # CRITICAL: attach to `probe_point` (pre-tiler), NOT to the OSD.
    #
    # nvmultistreamtiler composites the batch into ONE frame, so a probe downstream of it sees a
    # single frame_meta per batch with source_id always 0 — every camera collapses into one and
    # per-stream compliance state is silently wrong. Measured directly: probe on the tracker sees
    # 20 frames/batch and 20 distinct source_ids; probe on the OSD sees 1 and 1. It also makes a
    # frame counter there count BATCHES, under-reporting fps by exactly N.
    #
    # Object metadata painted here still reaches the OSD — the tiler remaps object coordinates
    # into tile space but preserves the metadata, so colours and labels render correctly.
    # (Probes must go on processing elements, never sinks — those raise "Probe failure".)
    if not args.no_attach:
        print(f"[probe] attached to '{probe_point}' (pre-tiler, per-stream metadata)", flush=True)
        p.attach(probe_point, Probe("safety-overlay", overlay))
    return p, overlay


def _add_rtsp_branch(p: Pipeline, cfg: dict, src: str, common: dict) -> None:
    """Encode the tiled output and push it to the local mediamtx instance.

    `realtime` paces the sink to the media clock (sync=1). It matters: with file sources and
    sync=0 the pipeline runs flat out — measured ~1.7x realtime at 20 streams — and the published
    stream plays back visibly sped up. Pacing also makes the run behave like production, where
    cameras deliver at 30 fps and the spare capacity simply sits idle.
    Turn it off only to benchmark the encode path.
    """
    rtsp = cfg["sinks"]["rtsp_out"]
    url = f"rtsp://127.0.0.1:{rtsp['port']}{rtsp['mount']}"
    realtime = bool(rtsp.get("realtime", True))
    p.add("nvvideoconvert", "enc_conv")
    p.add("nvv4l2h264enc", "enc", {
        "bitrate": int(rtsp["bitrate_kbps"]) * 1000,
        "iframeinterval": 30,
        "insert-sps-pps": 1,
    })
    p.add("h264parse", "enc_parse")
    p.add("rtspclientsink", "rtsp_sink",
          {"location": url, "sync": 1 if realtime else 0, "async": 0})
    p.link(src, "enc_conv", "enc", "enc_parse", "rtsp_sink")
    print(f"[rtsp] publishing to {url} "
          f"({'realtime-paced' if realtime else 'flat out'})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/demo.yml"))
    ap.add_argument("--streams", type=int)
    ap.add_argument("--topology", choices=["serial", "parallel"])
    ap.add_argument("--source", choices=["file", "rtsp"])
    ap.add_argument("--no-display", action="store_true", help="headless (benchmarking)")
    ap.add_argument("--rtsp-out", action="store_true", help="also publish RTSP")
    ap.add_argument("--fps", action="store_true", help="print aggregate fps periodically")
    ap.add_argument("--snapshot", metavar="PATH",
                    help="write JPEG frames of the rendered output (visual verification)")
    ap.add_argument("--no-attach", action="store_true",
                    help="attach no probe at all (isolates probe-attachment overhead)")
    ap.add_argument("--no-osd", action="store_true",
                    help="bypass tiler + RGBA convert + nvdsosd (isolates rendering cost)")
    ap.add_argument("--no-tracker", action="store_true",
                    help="drop nvtracker (isolates NvDCF cost)")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the Python compliance probe entirely (isolates its cost)")
    ap.add_argument("--duration", type=float, default=0,
                    help="stop cleanly after N seconds (benchmarking). 0 = run until EOS/Ctrl-C")
    ap.add_argument("--stats", action="store_true",
                    help="print per-class detection counts and live violation count")
    ap.add_argument("--no-fire", action="store_true",
                    help="run the PPE detector only (useful before the fire engine exists)")
    ap.add_argument("--zones", action="store_true", default=None,
                    help="enable nvdsanalytics zone analytics (ROI presence, overcrowding). "
                         "Zone geometry: configs/analytics/zones.yml")
    ap.add_argument("--no-zones", action="store_true",
                    help="force zones off (benchmark parity with Phase 1)")
    ap.add_argument("--dump-analytics", type=int, default=0, metavar="N",
                    help="print the first N raw nvdsanalytics metadata records and their "
                         "attributes, then stop printing (API shape diagnostic)")
    ap.add_argument("--events", action="store_true",
                    help="publish compliance state transitions to Redis (see events.redis in "
                         "configs/services.yml). Off by default so benchmark runs stay "
                         "comparable to Phase 1")
    ap.add_argument("--no-events", action="store_true",
                    help="force events off even if services.yml enables them")
    ap.add_argument("--rgb-capture", action="store_true",
                    help="add the valve-gated RGB tee branch used to crop the violating subject "
                         "out of the live frame. Off by default until its cost at full stream "
                         "count is measured")
    ap.add_argument("--no-smart-record", action="store_true",
                    help="disable nvurisrcbin smart-record evidence clips in rtsp mode. Clips "
                         "are the only way to get evidence video off a live camera, so this is "
                         "for isolating its cost, not for normal running")
    ap.add_argument("--track-stats", action="store_true",
                    help="report per-track id lifetimes when the run ends. Use to verify that "
                         "tracking still holds ids long enough to debounce after changing "
                         "--drop-frame-interval")
    ap.add_argument("--drop-frame-interval", type=int, default=None,
                    help="decoder emits every Nth frame (3 = 10fps analytics from a 30fps "
                         "stream). Decouples analytics rate from ingest rate; overrides "
                         "pipeline.drop_frame_interval in demo.yml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    args.streams = args.streams or int(cfg["pipeline"]["streams"])
    args.topology = args.topology or cfg["pipeline"].get("topology", "serial")
    args.source = args.source or cfg["pipeline"]["source_mode"]
    # Zones default OFF so a benchmark run stays comparable with the Phase 1 and 2.0 numbers;
    # the demo turns them on explicitly (scripts/run_demo.sh), same convention as --events.
    args.zones = bool(args.zones) and not args.no_zones
    if args.no_fire:
        cfg["inference"]["fire"]["enabled"] = False
    if args.no_tracker:
        cfg["tracker"]["enabled"] = False

    if args.source == "file" and not args.no_display:
        # `display: null` in demo.yml means "use whatever DISPLAY the environment already has"
        # (scripts/env.sh probes /tmp/.X11-unix for a live X socket). Pin it in the config only
        # when auto-detection picks the wrong server — nv3dsink fails with an unhelpful
        # "no display found" against the wrong number.
        _disp = cfg["sinks"]["display"].get("display") or _detect_display()
        if _disp:
            os.environ.setdefault("DISPLAY", str(_disp))
        else:
            # Explicitly clear it: an inherited DISPLAY pointing at a server we cannot reach is
            # what breaks EGL, so headless must mean genuinely unset.
            os.environ.pop("DISPLAY", None)

    rows, cols = tile_layout(args.streams, cfg)
    _dfi = (args.drop_frame_interval if args.drop_frame_interval is not None
            else cfg["pipeline"].get("drop_frame_interval", 0) or 0)
    # The analytics rate is printed, not just the drop interval, because "drop-frame-interval=3"
    # is easy to misread as "drop 3 frames". It means the opposite: keep every 3rd.
    _rate = f"{30 // _dfi}fps analytics" if _dfi > 0 else "30fps analytics"
    print(f"==> {args.streams} streams | {args.source} | topology={args.topology} | "
          f"tiles {rows}x{cols} | dfi={_dfi} ({_rate}) | "
          f"DISPLAY={os.environ.get('DISPLAY', '-')}", flush=True)

    pipeline, overlay = build(cfg, args)

    # Events are wired AFTER build() and before start(): the emitter opens a socket and starts a
    # thread, and neither belongs in graph construction.
    svc_path = ROOT / "configs/services.yml"
    svc = yaml.safe_load(svc_path.read_text()) if svc_path.exists() else {}
    ev_cfg = (svc.get("events") or {})
    want_events = (args.events or ev_cfg.get("enabled", False)) and not args.no_events
    emitter = None
    if want_events:
        r = ev_cfg.get("redis") or {}
        emitter = EventEmitter(
            host=r.get("host", "127.0.0.1"), port=int(r.get("port", 6379)),
            stream=r.get("stream", "safety:events"), maxlen=int(r.get("maxlen", 100000)),
            queue_size=int(ev_cfg.get("queue_size", 2048)),
        )
        overlay.enable_events(emitter)
        print(f"[events] publishing transitions to redis://{r.get('host', '127.0.0.1')}:"
              f"{r.get('port', 6379)}/{r.get('stream', 'safety:events')}", flush=True)

    # Smart record rides on the emitter: it needs somewhere to report the finished filename, and
    # the event stream is already the channel every other incident fact travels on. No emitter
    # means no way to tell anyone a clip exists, so there is nothing to enable.
    recorder = None
    if emitter is not None and args.source == "rtsp" and not args.no_smart_record:
        clips_dir, pre_s, post_s, cache_s = clip_settings(svc)
        # Cooldown is purely a disk-and-churn guard now, NOT a correctness mechanism — nothing
        # is attached during it, so a longer one only means more incidents with no video. Kept
        # short and tied to the clip's own tail; `clips.cooldown_s` in services.yml overrides.
        cooldown = float((svc.get("clips") or {}).get("cooldown_s", post_s))
        recorder = SmartRecorder(pipeline, emitter, clips_dir, pre_s, post_s, args.streams,
                                 cooldown_s=cooldown, cache_s=cache_s)
        # Both are set here rather than inside SmartRecorder so the emitter has exactly one
        # owner deciding its policy, and so a run with recording disabled leaves the event
        # payloads byte-identical to the file-mode ones.
        emitter.clip_mode = "smart_record"
        print(f"[smart-rec] enabled: {clips_dir} (-{pre_s}s +{post_s}s per incident)", flush=True)

    capture = None
    if emitter is not None and args.rgb_capture:
        capture = FrameCapture(pipeline, emitter, ROOT / "data/crops", args.streams,
                               cooldown_s=float((svc.get("store") or {}).get("merge_window_s", 30)))
        # Attached to the convert, i.e. downstream of the valve — so while the gate is shut this
        # operator is not called at all, rather than called and returning early.
        pipeline.attach("cap_conv", Probe("frame-capture", capture))
        print(f"[capture] writing subject crops to {ROOT / 'data/crops'}", flush=True)

    # One hook, several consumers. Fanned out here rather than chaining assignments so a new
    # consumer cannot silently replace an existing one — and so a raising hook cannot stop the
    # others: this runs on the streaming thread, where an escaping exception is SIGABRT.
    hooks = [h for h in (recorder.request if recorder else None,
                         capture.request if capture else None) if h]
    if hooks and emitter is not None:
        def _on_open(ev, _hooks=tuple(hooks)):
            for h in _hooks:
                try:
                    h(ev)
                except Exception:  # noqa: BLE001
                    pass
        emitter.on_open = _on_open

    pipeline.start()

    # pipeline.wait() blocks inside C++, so Python signal handlers never run — SIGINT and
    # SIGTERM are both ignored until the pipeline ends on its own. A timer thread calling
    # pipeline.stop() is the only reliable way to bound a run, which is what the sweep needs.
    if args.duration > 0:
        def _stop():
            print(f"[duration] {args.duration}s elapsed, stopping", flush=True)
            pipeline.stop()
        timer = threading.Timer(args.duration, _stop)
        timer.daemon = True
        timer.start()

    pipeline.wait()
    # Only reachable on a clean stop (EOS). `timeout --signal=KILL` bypasses this, which is why
    # both of these also report periodically from inside the probe.
    overlay.report_track_stability()
    if recorder is not None:
        print(f"[smart-rec] final: {recorder.stats()}", flush=True)
    if capture is not None:
        print(f"[capture] final: {capture.stats()}", flush=True)
    if emitter is not None:
        emitter.close()
        print(f"[events] final: {emitter.stats()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
