"""
Event schema and hot-path publisher for the safety pipeline.

An *event* is a **state transition**, not a frame observation. The compliance tracker already
decides, per (camera, track), whether someone is compliant; this module turns the moments when
that verdict CHANGES into durable records. A person standing without a helmet for two minutes is
one event with a duration, not 3600 of them.

## Why this file has no third-party imports

It is imported by `app/safety_pipeline.py`, which must run on the **system** Python 3.12 because
that is where `pyservicemaker` lives — and project policy is that nothing gets installed into
system Python (venvs live in `build/`). So the Redis publisher here speaks RESP over a plain
socket in ~30 lines rather than depending on `redis-py`. The cold-path services under `services/`
run in `build/venv-services` and use the real client; they have no such constraint.

That split is not a workaround, it is the right shape: the hot path should carry as little
third-party code as possible.

## Why publishing is fire-and-forget

`EventEmitter.emit()` is called from inside a DeepStream probe, i.e. on the GStreamer streaming
thread. Anything that blocks there — a TCP connect, a slow broker, a DNS lookup — directly costs
pipeline throughput, and the whole Phase 2 architecture rests on the hot path never being blocked
by anything downstream of it. So `emit()` only puts a dict on a bounded in-memory queue and
returns; a daemon thread does the socket work.

When the queue is full it **drops the oldest** and counts the drop. Dropping events is a bad
outcome; stalling twenty camera streams is a worse one, and the drop is visible via `stats()`
rather than silent. If Redis is down the emitter keeps failing and retrying in the background and
the pipeline never notices — the plan's stated failure mode ("events stay unverified; nothing
lost" from the pipeline's point of view).
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field, asdict
from typing import Any

# ---- event types -----------------------------------------------------------------------------
PPE_VIOLATION = "ppe_violation"
FIRE_ALERT = "fire_alert"
OVERCROWDING = "overcrowding"
# Raised by the reasoning service, never by the detectors: something the VLM saw in an incident's
# frames that no trained class covers — a spill, a fall, a blocked exit, a toppled load. The
# detectors can only find what they were trained on; the VLM is the only component in the system
# that can report a hazard nobody anticipated. Its job here is to get an operator LOOKING at the
# clip, not to adjudicate — which is why it lands as an alert with evidence attached.
HAZARD_ALERT = "hazard_alert"

# ---- lifecycle states (the plan's new -> alerted -> reasoning -> verified|unverified) ----------
STATE_NEW = "new"

# ---- severity --------------------------------------------------------------------------------
# Helmet is DIRECT evidence (the model has an explicit `no-helmet` class trained on the negative
# case); vest is INFERRED from the absence of an overlapping vest box and false-positives on
# occlusion and distance. Encoding that difference as severity is what lets the dashboard and the
# reasoning service treat them differently instead of pretending they are equally trustworthy.
SEV_CRITICAL = "critical"   # fire / smoke
SEV_HIGH = "high"           # no helmet — direct detection; or ANY violation in a restricted zone
SEV_MEDIUM = "medium"       # no vest — inferred from absence

# Used to combine two independent severity opinions — what the rule says, and what the ZONE says.
# "no vest" is medium in a corridor and high in a forklift aisle; the incident takes the worse of
# the two, because that is what changes the operator's response.
SEVERITY_RANK = {SEV_MEDIUM: 1, SEV_HIGH: 2, SEV_CRITICAL: 3}


def worst(*severities: str | None) -> str:
    """Highest-ranked severity among the arguments; ignores None."""
    best, rank = SEV_MEDIUM, 0
    for s in severities:
        r = SEVERITY_RANK.get(s or "", 0)
        if r > rank:
            best, rank = s, r          # type: ignore[assignment]
    return best


@dataclass
class Event:
    """One safety incident. Field names mirror VSS incident records so this ports to Thor.

    `zone`, `clip_uri`, `vlm_verdict` and `vlm_reason` are declared now and populated by later
    phases (2.2 zones, 2.3 clips, 2.4 reasoning). They are part of the schema from the start so
    the SQLite table and the API shape do not have to change under a running system.
    """
    camera_id: int
    type: str
    severity: str
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Wall-clock, NOT time.monotonic(). rules.py uses monotonic internally because it only ever
    # compares durations, but an event record has to mean something to a human hours later.
    ts: float = field(default_factory=time.time)
    track_id: int = -1
    label: str = ""
    bbox: tuple[float, float, float, float] | None = None   # left, top, width, height
    confidence: float = 0.0
    zone: str | None = None                 # Phase 2.2
    clip_uri: str | None = None             # Phase 2.3
    vlm_verdict: str | None = None          # Phase 2.4
    vlm_reason: str | None = None           # Phase 2.4
    state: str = STATE_NEW
    # Set on the closing event of a violation so the record carries how long it lasted.
    duration_s: float | None = None
    ended: bool = False
    # SOURCE timestamp of the frame the incident opened on, in nanoseconds — `frame_meta.buffer_pts`.
    #
    # This is what lets the clip service find the moment in the source. It is deliberately NOT
    # wall-clock `ts`: with file sources the pipeline can run faster or slower than realtime
    # (flat-out in benchmarks, paced in the demo), so wall-clock says nothing about WHERE in the
    # video the incident is. `buffer_pts` tracks source time regardless of how fast the pipeline
    # is consuming it — verified at drop-frame-interval=2, where 5.2s of wall time advanced
    # buffer_pts by 10.3s of source.
    source_pts_ns: int | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


# ---------------------------------------------------------------------------------------------
# minimal RESP client
# ---------------------------------------------------------------------------------------------
def _resp(*args: str) -> bytes:
    """Encode a Redis command as a RESP array. Enough for XADD; not a general client."""
    out = [f"*{len(args)}\r\n".encode()]
    for a in args:
        b = a.encode("utf-8", "replace")
        out.append(b"$%d\r\n%s\r\n" % (len(b), b))
    return b"".join(out)


class EventEmitter:
    """Non-blocking publisher of Events to a Redis Stream.

    Usage from the probe is deliberately trivial:  emitter.emit(event)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 6379,
                 stream: str = "safety:events", maxlen: int = 100_000,
                 queue_size: int = 2048, enabled: bool = True):
        self.host, self.port, self.stream = host, port, stream
        self.maxlen = maxlen
        self.enabled = enabled
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._published = 0
        self._dropped = 0
        self._errors = 0
        self._last_connect_attempt = 0.0
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(target=self._run, name="event-emitter", daemon=True)
            self._thread.start()

    # -- called from the hot path ---------------------------------------------------------------
    def emit(self, ev: Event) -> None:
        """Queue an event. Never blocks, never raises."""
        if not self.enabled:
            return
        try:
            self._q.put_nowait(ev)
        except queue.Full:
            # Drop the OLDEST, keep the newest: during a burst the most recent safety state is
            # the one an operator needs. Best-effort — another thread may have drained it.
            try:
                self._q.get_nowait()
                self._dropped += 1
                self._q.put_nowait(ev)
            except (queue.Empty, queue.Full):
                self._dropped += 1

    def stats(self) -> dict[str, int]:
        return {"published": self._published, "dropped": self._dropped,
                "errors": self._errors, "queued": self._q.qsize()}

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._close_sock()

    # -- background thread ----------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ev = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            self._publish(ev)
        # Best-effort flush of whatever is left, so a clean shutdown does not lose the tail.
        while True:
            try:
                self._publish(self._q.get_nowait())
            except queue.Empty:
                break

    def _connect(self) -> bool:
        # Rate-limit reconnects. Without this, a down broker turns into a tight connect() loop
        # that burns a core the pipeline needs.
        now = time.monotonic()
        if now - self._last_connect_attempt < 2.0:
            return False
        self._last_connect_attempt = now
        try:
            s = socket.create_connection((self.host, self.port), timeout=1.0)
            s.settimeout(1.0)
            # TCP_NODELAY: these are tiny, latency-sensitive writes; Nagle would batch them into
            # 40ms clumps for no benefit.
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = s
            return True
        except OSError:
            self._errors += 1
            self._sock = None
            return False

    def _close_sock(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _publish(self, ev: Event) -> None:
        if self._sock is None and not self._connect():
            self._dropped += 1
            return
        # MAXLEN ~ bounds the stream so a long-running demo cannot fill the disk. `~` is the
        # approximate form, which lets Redis trim on whole nodes and is markedly cheaper.
        cmd = _resp("XADD", self.stream, "MAXLEN", "~", str(self.maxlen), "*",
                    "data", ev.to_json())
        try:
            self._sock.sendall(cmd)
            # The reply MUST be drained. Never reading it leaves replies accumulating in the
            # kernel receive buffer until the window closes and sendall() starts blocking — which
            # would push broker backpressure onto the pipeline, the exact thing this design is
            # supposed to prevent.
            self._sock.recv(4096)
            self._published += 1
        except OSError:
            self._errors += 1
            self._close_sock()
            self._dropped += 1


# ---------------------------------------------------------------------------------------------
# transition detection
# ---------------------------------------------------------------------------------------------
class TransitionDetector:
    """Turns per-frame compliance verdicts into start/end events.

    Holds only the previous verdict per (stream, track), so it is O(tracks) per frame and adds no
    scanning to the probe.

    A track that simply disappears (person leaves frame, or the tracker loses the id) is closed
    by `expire()` rather than left open forever. That matters more than it looks: ~26% of tracks
    at 15fps analytics live too briefly to be adjudicated at all, so open-ended violations would
    otherwise accumulate indefinitely.
    """

    def __init__(self, emitter: EventEmitter, violation_state: str):
        self.emitter = emitter
        self.violation_state = violation_state
        # {(stream, track): (state, started_ts, label, severity)}
        self._open: dict[tuple[int, int], tuple[str, float, str, str]] = {}
        self._seen: dict[tuple[int, int], float] = {}
        self._last_expire = 0.0

    @staticmethod
    def severity_for(label: str, zone_severity: str | None = None) -> str:
        # "NO HELMET" is the direct signal and outranks the inferred vest rule when both fire.
        # The ZONE can raise it further: the same missing vest is a different finding in a
        # forklift route than in a corridor. This is the whole point of Phase 2.2.
        rule = SEV_HIGH if "HELMET" in label.upper() else SEV_MEDIUM
        return worst(rule, zone_severity)

    def update(self, stream_id: int, verdicts: list[dict],
               zones: dict[int, tuple[str | None, str | None]] | None = None,
               source_pts_ns: int | None = None) -> None:
        """`zones` maps track_id -> (zone_name, zone_severity); absent when analytics is off.

        `source_pts_ns` is the frame's position in the SOURCE video, carried onto opening events
        so the clip service can find the moment later.
        """
        now = time.monotonic()
        wall = time.time()
        zones = zones or {}
        for v in verdicts:
            box = v["box"]
            if box.track_id == -1:
                continue                      # untracked: cannot be debounced, cannot be an event
            key = (stream_id, box.track_id)
            self._seen[key] = now
            is_viol = v["state"] == self.violation_state
            prev = self._open.get(key)
            zone_name, zone_sev = zones.get(box.track_id, (None, None))

            if is_viol and prev is None:
                sev = self.severity_for(v["label"], zone_sev)
                self._open[key] = (v["state"], wall, v["label"], sev)
                self.emitter.emit(Event(
                    camera_id=stream_id, type=PPE_VIOLATION, severity=sev,
                    track_id=box.track_id, label=v["label"], zone=zone_name,
                    bbox=(box.left, box.top, box.width, box.height),
                    confidence=box.conf, ts=wall, source_pts_ns=source_pts_ns,
                ))
            elif is_viol and prev is not None:
                # Re-emit only when the SEVERITY changes — either the violation grew ("no vest?"
                # became "NO HELMET + no vest?") or the person walked into a restricted zone.
                # Both matter; a label change that leaves severity alone does not.
                sev = self.severity_for(v["label"], zone_sev)
                if sev != prev[3]:
                    self._open[key] = (v["state"], prev[1], v["label"], sev)
                    self.emitter.emit(Event(
                        camera_id=stream_id, type=PPE_VIOLATION, severity=sev,
                        track_id=box.track_id, label=v["label"], zone=zone_name,
                        bbox=(box.left, box.top, box.width, box.height),
                        confidence=box.conf, ts=wall, source_pts_ns=source_pts_ns,
                    ))
            elif not is_viol and prev is not None:
                self._close(key, stream_id, box.track_id, prev, wall)

    def _close(self, key, stream_id: int, track_id: int, prev, wall: float) -> None:
        _, started, label, sev = prev
        del self._open[key]
        self.emitter.emit(Event(
            camera_id=stream_id, type=PPE_VIOLATION, severity=sev,
            track_id=track_id, label=label, ts=wall,
            duration_s=max(0.0, wall - started), ended=True,
        ))

    def expire(self, ttl: float = 5.0, every: float = 1.0) -> None:
        """Close violations whose track has gone away. Throttled, like rules.py's own expiry."""
        now = time.monotonic()
        if now - self._last_expire < every:
            return
        self._last_expire = now
        wall = time.time()
        stale = [k for k, t in self._seen.items() if now - t > ttl]
        for key in stale:
            del self._seen[key]
            prev = self._open.get(key)
            if prev is not None:
                self._close(key, key[0], key[1], prev, wall)


class OvercrowdingDetector:
    """Zone occupancy limits, from nvdsanalytics frame-level `oc_status`.

    Keyed on (camera, zone) rather than on tracks — overcrowding is a property of a place, not of
    a person, and the whole point is that no individual is at fault.

    nvdsanalytics reports `oc_status` per frame, so it flickers as people cross the threshold
    boundary. `min_frames` requires the condition to hold for several consecutive analytics frames
    before an event fires, which is the same hysteresis idea `rules.py` applies to compliance —
    without it a zone sitting exactly on its limit emits a stream of start/end pairs.
    """

    def __init__(self, emitter: EventEmitter, min_frames: int = 5, clear_frames: int = 15):
        self.emitter = emitter
        self.min_frames = min_frames
        self.clear_frames = clear_frames
        self._streak: dict[tuple[int, str], int] = {}
        self._clear: dict[tuple[int, str], int] = {}
        self._open: dict[tuple[int, str], float] = {}

    @staticmethod
    def zone_track_id(zone: str) -> int:
        """A stable synthetic track id derived from the zone name.

        The store identifies a contributing track by (camera, type, track_id). Overcrowding has no
        track — it is a property of a place — so every zone on a camera would otherwise share
        track_id=-1 and collapse into one incident. A zone-derived id keeps them distinct.

        `crc32`, not `hash()`: Python randomises string hashing per process, so a restart would
        produce different ids and orphan every open incident.
        """
        return -(zlib.crc32(zone.encode()) % 1_000_000) - 2   # < -1, never a real track id

    def update(self, stream_id: int, crowded_zones: list[str],
               source_pts_ns: int | None = None) -> None:
        wall = time.time()
        crowded = set(crowded_zones)

        for zone in crowded:
            key = (stream_id, zone)
            self._clear[key] = 0
            self._streak[key] = self._streak.get(key, 0) + 1
            if self._streak[key] >= self.min_frames and key not in self._open:
                self._open[key] = wall
                self.emitter.emit(Event(camera_id=stream_id, type=OVERCROWDING,
                                        severity=SEV_HIGH, zone=zone, ts=wall,
                                        track_id=self.zone_track_id(zone),
                                        source_pts_ns=source_pts_ns,
                                        label=f"OVERCROWDED {zone}"))

        for key in list(self._open):
            if key[0] != stream_id or key[1] in crowded:
                continue
            # Clearing needs a LONGER streak than opening: an occupancy that dips for one frame as
            # someone is briefly missed by the detector has not actually resolved.
            self._clear[key] = self._clear.get(key, 0) + 1
            if self._clear[key] >= self.clear_frames:
                started = self._open.pop(key)
                self._streak[key] = 0
                self.emitter.emit(Event(camera_id=stream_id, type=OVERCROWDING,
                                        severity=SEV_HIGH, zone=key[1], ts=wall,
                                        track_id=self.zone_track_id(key[1]),
                                        label=f"OVERCROWDED {key[1]}",
                                        duration_s=max(0.0, wall - started), ended=True))

        for key in list(self._streak):
            if key[0] == stream_id and key[1] not in crowded and key not in self._open:
                self._streak[key] = 0


class FireTransitionDetector:
    """Fire alerts latch for `latch_seconds`, so the latch itself is the event boundary."""

    def __init__(self, emitter: EventEmitter):
        self.emitter = emitter
        self._active: dict[int, tuple[float, str]] = {}

    def update(self, stream_id: int, fire: dict | None,
               source_pts_ns: int | None = None) -> None:
        """`source_pts_ns` is what makes a fire alert produce an evidence clip.

        It was missing here while the PPE and overcrowding detectors both carried it, so every
        real fire alert landed with `source_pts_ns = None`, the store marked it `clip_state =
        'skipped'` ("cannot be located in the video"), and the most serious incident type in the
        system was the only one with no video attached. Nothing failed — the clip was correctly
        skipped for an event that genuinely could not be placed in the source.
        """
        wall = time.time()
        prev = self._active.get(stream_id)
        if fire and prev is None:
            self._active[stream_id] = (wall, fire["label"])
            self.emitter.emit(Event(camera_id=stream_id, type=FIRE_ALERT,
                                    severity=SEV_CRITICAL, label=fire["label"], ts=wall,
                                    source_pts_ns=source_pts_ns))
        elif not fire and prev is not None:
            started, label = prev
            del self._active[stream_id]
            self.emitter.emit(Event(camera_id=stream_id, type=FIRE_ALERT,
                                    severity=SEV_CRITICAL, label=label, ts=wall,
                                    duration_s=max(0.0, wall - started), ended=True))
