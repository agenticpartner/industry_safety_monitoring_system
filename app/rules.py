"""
Compliance state machine for the industrial safety demo.

Turns per-frame detections into a stable per-person verdict. The hard part is not the geometry,
it's the stability: raw per-frame association flickers constantly (a helmet box drops for two
frames and a compliant worker flashes red), which reads as a broken system. So every verdict
goes through a per-track rolling window and only flips on a supermajority.

Two rules, and they are NOT equally trustworthy:

  helmet — DIRECT. The model has an explicit `no-helmet` class, trained on the negative case.
           A `no-helmet` box on a person is positive evidence of a violation.

  vest   — INFERRED. There is no `no-vest` class. We report a violation when a `human` has no
           overlapping `vest` box, which is absence-of-evidence, not evidence-of-absence. It
           false-positives on occlusion, side-on poses, and distance. It is held to a higher
           confidence bar, skipped entirely for people too small to judge, and rendered with
           distinct wording so an operator can tell the two rule strengths apart.

This module is pure logic over plain tuples — no DeepStream imports — so it can be unit-tested
on a laptop without a Jetson.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

# --- class ids, from models/*/config/labels.txt (see export_summary.json) -------------------
PPE_HELMET = 0
PPE_HUMAN = 1
PPE_NO_HELMET = 2
PPE_VEST = 3

FIRE_FIRE = 0
FIRE_OTHER = 1
FIRE_SMOKE = 2

# --- verdicts -------------------------------------------------------------------------------
COMPLIANT = "compliant"
VIOLATION = "violation"
UNKNOWN = "unknown"

# BGR-ish RGBA tuples in the 0..1 range nvdsosd expects.
COLOR_COMPLIANT = (0.0, 0.85, 0.2, 1.0)
COLOR_VIOLATION = (1.0, 0.15, 0.1, 1.0)
COLOR_UNKNOWN = (0.6, 0.6, 0.6, 1.0)
COLOR_FIRE = (1.0, 0.35, 0.0, 1.0)


@dataclass(frozen=True)
class Box:
    """Axis-aligned detection in frame coordinates."""
    cls: int
    conf: float
    left: float
    top: float
    width: float
    height: float
    track_id: int = -1

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


def containment(inner: Box, outer: Box) -> float:
    """Fraction of `inner` that falls inside `outer`.

    Deliberately not IoU: a helmet is tiny next to a person, so their IoU is near zero even when
    the helmet is unambiguously on that person's head. What matters is how much of the small box
    sits inside the big one.
    """
    if inner.area <= 0:
        return 0.0
    ix = max(0.0, min(inner.right, outer.right) - max(inner.left, outer.left))
    iy = max(0.0, min(inner.bottom, outer.bottom) - max(inner.top, outer.top))
    return (ix * iy) / inner.area


@dataclass
class PersonState:
    """Rolling compliance history for one tracked person on one stream."""
    helmet_votes: deque = field(default_factory=deque)
    vest_votes: deque = field(default_factory=deque)
    helmet_verdict: str = UNKNOWN
    vest_verdict: str = UNKNOWN
    last_seen: float = field(default_factory=time.monotonic)

    def vote(self, window: deque, value: str, current: str,
             window_frames: int, flip_ratio: float) -> str:
        """Add a vote and return the (possibly unchanged) verdict.

        A verdict only flips when the window is full enough to be meaningful AND a supermajority
        agrees. Anything less and we keep the previous verdict — hysteresis is what stops the
        boxes strobing between red and green.
        """
        window.append(value)
        while len(window) > window_frames:
            window.popleft()

        # Need at least half a window before committing to anything.
        if len(window) < max(3, window_frames // 2):
            return current if current != UNKNOWN else value

        for candidate in (VIOLATION, COMPLIANT):
            if window.count(candidate) / len(window) >= flip_ratio:
                return candidate
        return current


class ComplianceTracker:
    """Per-stream, per-track compliance state with debouncing and fire latching."""

    def __init__(self, cfg: dict):
        rules = cfg.get("rules", {})
        self.window_frames = int(rules.get("window_frames", 15))
        self.flip_ratio = float(rules.get("flip_ratio", 0.6))

        helmet = rules.get("helmet", {}) or {}
        vest = rules.get("vest", {}) or {}
        fire = rules.get("fire", {}) or {}

        self.helmet_enabled = bool(helmet.get("enabled", True))
        self.helmet_min_conf = float(helmet.get("min_confidence", 0.35))

        self.vest_enabled = bool(vest.get("enabled", True))
        self.vest_min_conf = float(vest.get("min_confidence", 0.45))
        self.vest_containment = float(vest.get("containment", 0.5))
        self.vest_min_person_h = float(vest.get("min_person_height_px", 80))

        self.fire_enabled = bool(fire.get("enabled", True))
        self.fire_min_conf = float(fire.get("min_confidence", 0.40))
        self.fire_latch = float(fire.get("latch_seconds", 2.0))
        self.fire_ignored = {c.lower() for c in (fire.get("ignore_classes") or [])}

        # State is bucketed PER STREAM. A single flat dict keyed (stream, track) forces every
        # per-frame query to scan all streams' tracks, which is O(streams x tracks) per frame and
        # therefore quadratic across a batch — that alone collapses throughput as cameras scale.
        self._states: dict[int, dict[int, PersonState]] = {}
        self._violations: dict[int, int] = {}
        self._fire_until: dict[int, float] = {}
        self._fire_label: dict[int, str] = {}
        self._last_expire = 0.0

    # -- helmet: direct evidence ------------------------------------------------------------
    def _helmet_vote(self, person: Box, helmets: list[Box], bare: list[Box]) -> str:
        best_bare = max((containment(b, person) for b in bare), default=0.0)
        best_helm = max((containment(h, person) for h in helmets), default=0.0)

        # A bare head inside the person box is positive evidence; prefer whichever signal is
        # more strongly contained so a helmet on a neighbour doesn't clear this person.
        if best_bare >= self.vest_containment and best_bare >= best_helm:
            return VIOLATION
        if best_helm >= self.vest_containment:
            return COMPLIANT
        return UNKNOWN

    # -- vest: inferred from absence ---------------------------------------------------------
    def _vest_vote(self, person: Box, vests: list[Box]) -> str:
        # Too small to judge: abstain rather than manufacture a violation.
        if person.height < self.vest_min_person_h:
            return UNKNOWN
        best = max((containment(v, person) for v in vests), default=0.0)
        if best >= self.vest_containment:
            return COMPLIANT
        return VIOLATION

    def update(self, stream_id: int, ppe_boxes: Iterable[Box]) -> list[dict]:
        """Fold one frame of PPE detections into state; return per-person render instructions."""
        boxes = list(ppe_boxes)
        persons = [b for b in boxes if b.cls == PPE_HUMAN]
        helmets = [b for b in boxes if b.cls == PPE_HELMET and b.conf >= self.helmet_min_conf]
        bare = [b for b in boxes if b.cls == PPE_NO_HELMET and b.conf >= self.helmet_min_conf]
        vests = [b for b in boxes if b.cls == PPE_VEST and b.conf >= self.vest_min_conf]

        now = time.monotonic()
        out: list[dict] = []
        violations = 0

        stream_states = self._states.setdefault(stream_id, {})
        for person in persons:
            st = stream_states.get(person.track_id)
            if st is None:
                st = stream_states[person.track_id] = PersonState()
            st.last_seen = now

            if self.helmet_enabled:
                st.helmet_verdict = st.vote(
                    st.helmet_votes, self._helmet_vote(person, helmets, bare),
                    st.helmet_verdict, self.window_frames, self.flip_ratio)
            if self.vest_enabled:
                st.vest_verdict = st.vote(
                    st.vest_votes, self._vest_vote(person, vests),
                    st.vest_verdict, self.window_frames, self.flip_ratio)

            # Labels are terse because each tile is only ~480x270 at 20 cameras. Helmet is a
            # DIRECT detection so it shouts; vest is INFERRED from absence, so it is lower-case
            # and hedged — an operator should be able to tell the two evidence strengths apart
            # at a glance.
            reasons = []
            if st.helmet_verdict == VIOLATION:
                reasons.append("NO HELMET")
            if st.vest_verdict == VIOLATION:
                reasons.append("no vest?")

            if reasons:
                state, color = VIOLATION, COLOR_VIOLATION
                violations += 1
            elif COMPLIANT in (st.helmet_verdict, st.vest_verdict):
                state, color = COMPLIANT, COLOR_COMPLIANT
            else:
                state, color = UNKNOWN, COLOR_UNKNOWN

            out.append({
                "box": person,
                "state": state,
                "color": color,
                "label": " + ".join(reasons) if reasons else
                         ("OK" if state == COMPLIANT else "?"),
            })

        # Counted while we were already walking this frame's people, so the per-tile banner is
        # a dict lookup rather than a scan.
        self._violations[stream_id] = violations
        self._expire(now)
        return out

    def update_fire(self, stream_id: int, fire_boxes: Iterable[Box],
                    labels: list[str] | None = None) -> dict | None:
        """Latch a fire/smoke alert so a flickering flame stays visible on screen."""
        if not self.fire_enabled:
            return None
        now = time.monotonic()

        for b in fire_boxes:
            name = self._fire_class_name(b.cls, labels)
            if name in self.fire_ignored or b.cls == FIRE_OTHER:
                continue
            if b.conf < self.fire_min_conf:
                continue
            self._fire_until[stream_id] = now + self.fire_latch
            self._fire_label[stream_id] = name.upper()

        if self._fire_until.get(stream_id, 0.0) > now:
            return {"label": self._fire_label.get(stream_id, "FIRE"), "color": COLOR_FIRE}
        return None

    @staticmethod
    def _fire_class_name(cls: int, labels: list[str] | None) -> str:
        """Resolve a fire-model class id to a name.

        Falls back to the model's known class order rather than the raw index, so a caller that
        forgets to pass labels gets "fire" on screen instead of "0".
        """
        if labels and cls < len(labels):
            return labels[cls].lower()
        default = {FIRE_FIRE: "fire", FIRE_OTHER: "other", FIRE_SMOKE: "smoke"}
        return default.get(cls, f"class{cls}")

    def violation_count(self, stream_id: int) -> int:
        return self._violations.get(stream_id, 0)

    def _expire(self, now: float, ttl: float = 5.0, every: float = 1.0) -> None:
        """Drop tracks we haven't seen recently so state doesn't grow without bound.

        Throttled to once a second. Sweeping every frame means every camera pays a full scan of
        every other camera's tracks on every frame — pure overhead, since a track that went
        stale 20 ms ago is no more expired than one that went stale now.
        """
        if now - self._last_expire < every:
            return
        self._last_expire = now
        for sid, tracks in self._states.items():
            stale = [t for t, st in tracks.items() if now - st.last_seen > ttl]
            for t in stale:
                del tracks[t]
