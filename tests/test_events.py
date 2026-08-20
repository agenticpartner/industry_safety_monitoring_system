"""
Unit tests for event transition detection.

The property under test is the one the whole event model rests on: **a violation produces exactly
one event, no matter how many frames it spans.** Everything else in Phase 2 — alerting, clip
capture, reasoning, the incident feed — assumes that and gets flooded if it is wrong.

Runs on a laptop: no DeepStream, no Redis, no Jetson.
"""

from __future__ import annotations

import queue
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from events import (  # noqa: E402
    Event, EventEmitter, TransitionDetector, FireTransitionDetector, OvercrowdingDetector,
    PPE_VIOLATION, FIRE_ALERT, OVERCROWDING, SEV_HIGH, SEV_MEDIUM, SEV_CRITICAL,
)
from rules import Box, COMPLIANT, VIOLATION, UNKNOWN  # noqa: E402


class FakeEmitter:
    """Captures events instead of publishing. Same surface as EventEmitter.emit."""

    def __init__(self):
        self.events: list[Event] = []

    def emit(self, ev: Event) -> None:
        self.events.append(ev)

    def of_type(self, t: str) -> list[Event]:
        return [e for e in self.events if e.type == t]


def verdict(track_id: int, state: str, label: str = "NO HELMET") -> dict:
    return {
        "box": Box(cls=1, conf=0.9, left=10, top=10, width=50, height=120, track_id=track_id),
        "state": state,
        "color": (1.0, 0.0, 0.0, 1.0),
        "label": label,
    }


class TestTransitions(unittest.TestCase):
    def setUp(self):
        self.em = FakeEmitter()
        self.td = TransitionDetector(self.em, VIOLATION)

    def test_compliant_produces_no_events(self):
        for _ in range(50):
            self.td.update(0, [verdict(1, COMPLIANT, "OK")])
        self.assertEqual(self.em.events, [])

    def test_unknown_produces_no_events(self):
        for _ in range(50):
            self.td.update(0, [verdict(1, UNKNOWN, "?")])
        self.assertEqual(self.em.events, [])

    def test_sustained_violation_is_exactly_one_event(self):
        """The core property. 300 frames of the same violation is ONE incident."""
        for _ in range(300):
            self.td.update(0, [verdict(1, VIOLATION)])
        opens = [e for e in self.em.of_type(PPE_VIOLATION) if not e.ended]
        self.assertEqual(len(opens), 1)
        self.assertEqual(opens[0].track_id, 1)
        self.assertEqual(opens[0].severity, SEV_HIGH)

    def test_violation_end_emits_closing_event(self):
        for _ in range(10):
            self.td.update(0, [verdict(1, VIOLATION)])
        for _ in range(10):
            self.td.update(0, [verdict(1, COMPLIANT, "OK")])
        evs = self.em.of_type(PPE_VIOLATION)
        self.assertEqual(len(evs), 2)
        self.assertFalse(evs[0].ended)
        self.assertTrue(evs[1].ended)
        self.assertIsNotNone(evs[1].duration_s)

    def test_flapping_produces_one_pair_per_episode(self):
        """Two separate episodes are two incidents — not merged, not one each frame."""
        for _ in range(5):
            self.td.update(0, [verdict(1, VIOLATION)])
        for _ in range(5):
            self.td.update(0, [verdict(1, COMPLIANT, "OK")])
        for _ in range(5):
            self.td.update(0, [verdict(1, VIOLATION)])
        evs = self.em.of_type(PPE_VIOLATION)
        self.assertEqual([e.ended for e in evs], [False, True, False])

    def test_untracked_objects_never_emit(self):
        """track_id -1 means the tracker never locked on: it cannot be debounced or followed."""
        for _ in range(50):
            self.td.update(0, [verdict(-1, VIOLATION)])
        self.assertEqual(self.em.events, [])

    def test_streams_are_independent(self):
        self.td.update(0, [verdict(1, VIOLATION)])
        self.td.update(1, [verdict(1, VIOLATION)])
        opens = [e for e in self.em.of_type(PPE_VIOLATION) if not e.ended]
        self.assertEqual(sorted(e.camera_id for e in opens), [0, 1])

    def test_same_track_id_on_different_streams_is_two_incidents(self):
        """Track ids are only unique per stream; keying on id alone would collapse cameras."""
        for _ in range(20):
            self.td.update(0, [verdict(7, VIOLATION)])
            self.td.update(3, [verdict(7, VIOLATION)])
        opens = [e for e in self.em.of_type(PPE_VIOLATION) if not e.ended]
        self.assertEqual(len(opens), 2)

    def test_severity_escalates_when_helmet_joins_vest(self):
        for _ in range(3):
            self.td.update(0, [verdict(1, VIOLATION, "no vest?")])
        for _ in range(3):
            self.td.update(0, [verdict(1, VIOLATION, "NO HELMET + no vest?")])
        evs = [e for e in self.em.of_type(PPE_VIOLATION) if not e.ended]
        self.assertEqual([e.severity for e in evs], [SEV_MEDIUM, SEV_HIGH])

    def test_severity_does_not_re_emit_when_unchanged(self):
        """A label change that does not change severity is not worth an event."""
        for _ in range(3):
            self.td.update(0, [verdict(1, VIOLATION, "NO HELMET")])
        for _ in range(3):
            self.td.update(0, [verdict(1, VIOLATION, "NO HELMET + no vest?")])
        opens = [e for e in self.em.of_type(PPE_VIOLATION) if not e.ended]
        self.assertEqual(len(opens), 1)

    def test_vanished_track_is_closed_by_expire(self):
        """A person who walks out of frame must not leave an incident open forever."""
        self.td.update(0, [verdict(1, VIOLATION)])
        # Force the track stale without sleeping.
        self.td._seen[(0, 1)] = time.monotonic() - 99
        self.td._last_expire = 0.0
        self.td.expire(ttl=5.0, every=0.0)
        evs = self.em.of_type(PPE_VIOLATION)
        self.assertEqual(len(evs), 2)
        self.assertTrue(evs[1].ended)

    def test_expire_is_idempotent(self):
        self.td.update(0, [verdict(1, VIOLATION)])
        self.td._seen[(0, 1)] = time.monotonic() - 99
        for _ in range(3):
            self.td._last_expire = 0.0
            self.td.expire(ttl=5.0, every=0.0)
        self.assertEqual(len(self.em.of_type(PPE_VIOLATION)), 2)


class TestZoneSeverity(unittest.TestCase):
    """Phase 2.2: the same violation is a different finding depending on where it happens."""

    def setUp(self):
        self.em = FakeEmitter()
        self.td = TransitionDetector(self.em, VIOLATION)

    def test_vest_violation_in_restricted_zone_is_escalated(self):
        self.td.update(0, [verdict(1, VIOLATION, "no vest?")],
                       {1: ("ForkliftAisle", SEV_HIGH)})
        ev = self.em.of_type(PPE_VIOLATION)[0]
        self.assertEqual(ev.severity, SEV_HIGH)
        self.assertEqual(ev.zone, "ForkliftAisle")

    def test_vest_violation_in_ordinary_zone_stays_medium(self):
        self.td.update(0, [verdict(1, VIOLATION, "no vest?")], {1: ("AisleLeft", SEV_MEDIUM)})
        ev = self.em.of_type(PPE_VIOLATION)[0]
        self.assertEqual(ev.severity, SEV_MEDIUM)
        self.assertEqual(ev.zone, "AisleLeft")

    def test_zone_never_downgrades_a_helmet_violation(self):
        """A missing helmet is high on its own; a benign zone must not soften it."""
        self.td.update(0, [verdict(1, VIOLATION, "NO HELMET")], {1: ("AisleLeft", SEV_MEDIUM)})
        self.assertEqual(self.em.of_type(PPE_VIOLATION)[0].severity, SEV_HIGH)

    def test_walking_into_a_restricted_zone_re_emits(self):
        """Severity rising mid-incident is worth an update; the operator's response changes."""
        self.td.update(0, [verdict(1, VIOLATION, "no vest?")], {1: ("AisleLeft", SEV_MEDIUM)})
        self.td.update(0, [verdict(1, VIOLATION, "no vest?")], {1: ("ForkliftAisle", SEV_HIGH)})
        sevs = [e.severity for e in self.em.of_type(PPE_VIOLATION) if not e.ended]
        self.assertEqual(sevs, [SEV_MEDIUM, SEV_HIGH])

    def test_staying_in_the_same_zone_does_not_re_emit(self):
        for _ in range(50):
            self.td.update(0, [verdict(1, VIOLATION, "no vest?")], {1: ("ForkliftAisle", SEV_HIGH)})
        self.assertEqual(len([e for e in self.em.of_type(PPE_VIOLATION) if not e.ended]), 1)

    def test_no_zone_information_still_emits(self):
        """Zones are an enrichment — with analytics off, events must still flow."""
        self.td.update(0, [verdict(1, VIOLATION, "no vest?")])
        ev = self.em.of_type(PPE_VIOLATION)[0]
        self.assertIsNone(ev.zone)
        self.assertEqual(ev.severity, SEV_MEDIUM)


class TestOvercrowding(unittest.TestCase):
    def setUp(self):
        self.em = FakeEmitter()
        self.oc = OvercrowdingDetector(self.em, min_frames=3, clear_frames=5)

    def test_requires_a_streak_before_firing(self):
        """One frame over the limit is noise, not an incident."""
        self.oc.update(0, ["ZoneA"])
        self.oc.update(0, ["ZoneA"])
        self.assertEqual(self.em.events, [])
        self.oc.update(0, ["ZoneA"])
        self.assertEqual(len(self.em.of_type(OVERCROWDING)), 1)

    def test_sustained_overcrowding_is_one_event(self):
        for _ in range(100):
            self.oc.update(0, ["ZoneA"])
        self.assertEqual(len([e for e in self.em.of_type(OVERCROWDING) if not e.ended]), 1)

    def test_brief_dip_does_not_close_the_incident(self):
        for _ in range(5):
            self.oc.update(0, ["ZoneA"])
        self.oc.update(0, [])          # one frame under the limit
        for _ in range(5):
            self.oc.update(0, ["ZoneA"])
        self.assertEqual(len([e for e in self.em.of_type(OVERCROWDING) if e.ended]), 0)

    def test_sustained_clear_closes_with_duration(self):
        for _ in range(5):
            self.oc.update(0, ["ZoneA"])
        for _ in range(6):
            self.oc.update(0, [])
        closed = [e for e in self.em.of_type(OVERCROWDING) if e.ended]
        self.assertEqual(len(closed), 1)
        self.assertIsNotNone(closed[0].duration_s)

    def test_zones_are_independent(self):
        for _ in range(5):
            self.oc.update(0, ["ZoneA", "ZoneB"])
        opens = [e for e in self.em.of_type(OVERCROWDING) if not e.ended]
        self.assertEqual(sorted(e.zone for e in opens), ["ZoneA", "ZoneB"])

    def test_zone_track_ids_are_distinct_and_stable(self):
        """The store keys contributing tracks by id; zones must not collide on one camera."""
        a = OvercrowdingDetector.zone_track_id("ZoneA")
        b = OvercrowdingDetector.zone_track_id("ZoneB")
        self.assertNotEqual(a, b)
        self.assertEqual(a, OvercrowdingDetector.zone_track_id("ZoneA"))
        # Must never look like a real track id (-1 means "untracked", >=0 is a real one).
        self.assertLess(a, -1)
        self.assertLess(b, -1)


class TestFireTransitions(unittest.TestCase):
    def setUp(self):
        self.em = FakeEmitter()
        self.fd = FireTransitionDetector(self.em)

    def test_latched_fire_is_one_event(self):
        for _ in range(100):
            self.fd.update(0, {"label": "FIRE", "color": (1, 0, 0, 1)})
        opens = [e for e in self.em.of_type(FIRE_ALERT) if not e.ended]
        self.assertEqual(len(opens), 1)
        self.assertEqual(opens[0].severity, SEV_CRITICAL)

    def test_fire_clears_and_closes(self):
        self.fd.update(0, {"label": "SMOKE", "color": (1, 0, 0, 1)})
        self.fd.update(0, None)
        evs = self.em.of_type(FIRE_ALERT)
        self.assertEqual([e.ended for e in evs], [False, True])
        self.assertEqual(evs[1].label, "SMOKE")

    def test_no_fire_no_events(self):
        for _ in range(20):
            self.fd.update(0, None)
        self.assertEqual(self.em.events, [])


class TestEmitterBackpressure(unittest.TestCase):
    """The hot path must never block, even with no broker listening."""

    def test_emit_never_blocks_when_queue_is_full(self):
        em = EventEmitter(enabled=False)          # no thread, so the queue cannot drain
        em.enabled = True
        em._q = queue.Queue(maxsize=4)
        start = time.monotonic()
        for _ in range(500):
            em.emit(Event(camera_id=0, type=PPE_VIOLATION, severity=SEV_HIGH))
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 1.0, "emit() blocked — this would stall the pipeline")
        self.assertGreater(em.stats()["dropped"], 0, "overflow must be counted, not silent")
        self.assertLessEqual(em._q.qsize(), 4)

    def test_disabled_emitter_is_inert(self):
        em = EventEmitter(enabled=False)
        for _ in range(100):
            em.emit(Event(camera_id=0, type=PPE_VIOLATION, severity=SEV_HIGH))
        self.assertEqual(em.stats()["published"], 0)
        self.assertEqual(em._q.qsize(), 0)

    def test_unreachable_broker_does_not_raise(self):
        """Redis being down is a normal operating condition, not an error the pipeline sees."""
        em = EventEmitter(host="127.0.0.1", port=1, queue_size=8)
        for _ in range(20):
            em.emit(Event(camera_id=0, type=PPE_VIOLATION, severity=SEV_HIGH))
        time.sleep(0.5)
        em.close()
        self.assertEqual(em.stats()["published"], 0)


class TestEventSchema(unittest.TestCase):
    def test_event_ids_are_unique(self):
        ids = {Event(camera_id=0, type=PPE_VIOLATION, severity=SEV_HIGH).event_id
               for _ in range(1000)}
        self.assertEqual(len(ids), 1000)

    def test_json_round_trip_preserves_fields(self):
        import json
        ev = Event(camera_id=3, type=PPE_VIOLATION, severity=SEV_HIGH, track_id=42,
                   label="NO HELMET", bbox=(1.0, 2.0, 3.0, 4.0), confidence=0.87)
        d = json.loads(ev.to_json())
        self.assertEqual(d["camera_id"], 3)
        self.assertEqual(d["track_id"], 42)
        self.assertEqual(d["bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(d["state"], "new")
        # Fields later phases fill must exist in the schema from the start, so the SQLite table
        # and API shape never have to change under a running system.
        for k in ("zone", "clip_uri", "vlm_verdict", "vlm_reason"):
            self.assertIn(k, d)

    def test_timestamp_is_wall_clock_not_monotonic(self):
        """An incident record has to mean something to a human hours later."""
        ev = Event(camera_id=0, type=PPE_VIOLATION, severity=SEV_HIGH)
        self.assertAlmostEqual(ev.ts, time.time(), delta=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
