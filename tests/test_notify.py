#!/usr/bin/env python3
"""Tests for the notification policy.

    python3 tests/test_notify.py

The policy is the whole product here. Everything else is transport, but WHICH incidents reach
somebody's phone decides whether the channel is useful or muted — and a muted channel protects
nobody. So the decisions are pure functions over an incident row, and they are tested.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))

from notify_service import (  # noqa: E402
    caption_for, clip_ready_or_gave_up, esc, should_notify,
)

CFG = {
    "always_types": ["fire_alert", "hazard_alert"],
    "confirmed_types": ["ppe_violation", "overcrowding"],
    "min_severity": "high",
    "clip_wait_s": 20,
}


def row(**kw):
    base = {"event_id": "a" * 32, "ts": time.time(), "camera_id": 7, "type": "ppe_violation",
            "severity": "high", "zone": None, "label": None, "vlm_verdict": None,
            "vlm_reason": None, "clip_state": "ready", "clip_uri": "data/clips/x.mp4"}
    base.update(kw)
    return base


class TestPolicy(unittest.TestCase):
    def test_fire_goes_immediately_without_waiting_for_a_verdict(self):
        send, why = should_notify(row(type="fire_alert", severity="critical"), CFG, time.time())
        self.assertTrue(send, "waiting for adjudication on a fire is indefensible")
        self.assertIn("always", why)

    def test_vlm_raised_hazard_goes_immediately(self):
        send, _ = should_notify(row(type="hazard_alert", severity="high"), CFG, time.time())
        self.assertTrue(send)

    def test_unadjudicated_ppe_waits_rather_than_being_dropped(self):
        send, why = should_notify(row(type="ppe_violation", vlm_verdict=None), CFG, time.time())
        self.assertFalse(send)
        self.assertEqual(why, "waiting for VLM verdict",
                         "must be distinguishable from a decision, so the caller leaves it pending")

    def test_confirmed_ppe_is_sent(self):
        send, why = should_notify(row(vlm_verdict="confirmed"), CFG, time.time())
        self.assertTrue(send)
        self.assertIn("confirmed", why)

    def test_rejected_ppe_never_reaches_a_phone(self):
        # The headline false positive on this footage is a traffic cone tracked as a worker.
        send, why = should_notify(row(vlm_verdict="rejected"), CFG, time.time())
        self.assertFalse(send)
        self.assertIn("rejected", why)

    def test_severity_floor_applies(self):
        send, why = should_notify(row(severity="medium", vlm_verdict="confirmed"),
                                  CFG, time.time())
        self.assertFalse(send)
        self.assertIn("below", why)

    def test_severity_floor_does_not_silence_fire(self):
        # Fire is critical by construction; this guards against a config that would mute it.
        send, _ = should_notify(row(type="fire_alert", severity="critical"), CFG, time.time())
        self.assertTrue(send)

    def test_unknown_type_is_not_sent(self):
        send, why = should_notify(row(type="something_new"), CFG, time.time())
        self.assertFalse(send)
        self.assertIn("not in notify policy", why)


class TestClipWait(unittest.TestCase):
    def test_ready_clip_proceeds_with_a_path(self):
        ok, path = clip_ready_or_gave_up(row(clip_state="ready"), CFG, time.time())
        self.assertTrue(ok)
        self.assertIsNotNone(path)

    def test_pending_clip_waits(self):
        ok, path = clip_ready_or_gave_up(
            row(clip_state="pending", ts=time.time()), CFG, time.time())
        self.assertFalse(ok, "the clip is the point; give it a moment")
        self.assertIsNone(path)

    def test_pending_clip_gives_up_eventually(self):
        # A fire alert must not be delayed indefinitely by a clip that will never cut.
        ok, path = clip_ready_or_gave_up(
            row(clip_state="pending", ts=time.time() - 999), CFG, time.time())
        self.assertTrue(ok)
        self.assertIsNone(path, "sends text-only rather than nothing")

    def test_skipped_clip_does_not_wait(self):
        ok, path = clip_ready_or_gave_up(row(clip_state="skipped", clip_uri=None),
                                         CFG, time.time())
        self.assertTrue(ok)
        self.assertIsNone(path)


class TestCaption(unittest.TestCase):
    def test_names_camera_and_zone(self):
        c = caption_for(row(camera_id=3, zone="SpillZone", label="NO HELMET",
                            severity="high", vlm_verdict="confirmed"))
        self.assertIn("cam03", c)
        self.assertIn("SpillZone", c)
        self.assertIn("NO HELMET", c)

    def test_escapes_html_so_one_angle_bracket_cannot_kill_the_message(self):
        c = caption_for(row(vlm_reason="a <person> & a cone"))
        self.assertIn("&lt;person&gt;", c)
        self.assertNotIn("<person>", c)

    def test_esc_handles_ampersand_first(self):
        # &lt; must not become &amp;lt;
        self.assertEqual(esc("<a & b>"), "&lt;a &amp; b&gt;")


if __name__ == "__main__":
    unittest.main(verbosity=2)
