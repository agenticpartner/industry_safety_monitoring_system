"""
Unit tests for the compliance state machine.

rules.py has no DeepStream imports precisely so this can run on a laptop: the logic that decides
whether a worker is compliant is the part most likely to be subtly wrong, and it should not need
a Jetson and a video feed to check.

    python3 -m pytest tests/test_rules.py -q      (or: python3 tests/test_rules.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from rules import (  # noqa: E402
    Box, ComplianceTracker, COMPLIANT, UNKNOWN, VIOLATION,
    PPE_HELMET, PPE_HUMAN, PPE_NO_HELMET, PPE_VEST, containment,
)

CFG = {
    "rules": {
        "window_frames": 15,
        "flip_ratio": 0.6,
        "helmet": {"enabled": True, "min_confidence": 0.35},
        "vest": {"enabled": True, "min_confidence": 0.45,
                 "containment": 0.5, "min_person_height_px": 80},
        "fire": {"enabled": True, "min_confidence": 0.40,
                 "latch_seconds": 2.0, "ignore_classes": ["other"]},
    }
}

def person(track=1, left=100, top=100, w=60, h=200):
    return Box(PPE_HUMAN, 0.9, left, top, w, h, track)

def helmet(left=115, top=105, w=30, h=25):
    return Box(PPE_HELMET, 0.8, left, top, w, h)

def bare_head(left=115, top=105, w=30, h=25):
    return Box(PPE_NO_HELMET, 0.8, left, top, w, h)

def vest(left=110, top=160, w=40, h=70):
    return Box(PPE_VEST, 0.8, left, top, w, h)


def settle(tracker, boxes, frames=20, stream=0):
    """Feed the same frame repeatedly so verdicts pass the debounce window."""
    out = []
    for _ in range(frames):
        out = tracker.update(stream, boxes)
    return out


# --- geometry --------------------------------------------------------------------------------

def test_containment_is_not_iou():
    """A helmet is tiny next to a person, so IoU would be ~0 even when fully worn."""
    p, h = person(), helmet()
    assert containment(h, p) == 1.0        # helmet entirely inside the person box
    assert containment(p, h) < 0.1         # and the reverse is near zero — hence not IoU

def test_containment_partial():
    p = person(left=100, top=100, w=100, h=100)     # 100..200 x 100..200
    half_out = Box(PPE_VEST, 0.9, 150, 150, 100, 100)  # only a quarter overlaps
    assert abs(containment(half_out, p) - 0.25) < 1e-6


# --- helmet: direct signal -------------------------------------------------------------------

def test_helmet_worn_is_compliant():
    t = ComplianceTracker(CFG)
    out = settle(t, [person(), helmet(), vest()])
    assert out[0]["state"] == COMPLIANT

def test_bare_head_is_violation():
    t = ComplianceTracker(CFG)
    out = settle(t, [person(), bare_head(), vest()])
    assert out[0]["state"] == VIOLATION
    assert "NO HELMET" in out[0]["label"]

def test_low_confidence_helmet_is_ignored():
    """Below min_confidence the helmet shouldn't count as evidence either way."""
    t = ComplianceTracker(CFG)
    faint = Box(PPE_HELMET, 0.10, 115, 105, 30, 25)
    out = settle(t, [person(), faint, vest()])
    # No helmet evidence + vest present -> not a helmet violation.
    assert "NO HELMET" not in out[0]["label"]


# --- vest: inferred from absence -------------------------------------------------------------

def test_missing_vest_is_inferred_violation():
    t = ComplianceTracker(CFG)
    out = settle(t, [person(), helmet()])
    assert out[0]["state"] == VIOLATION
    assert "vest" in out[0]["label"].lower()

def test_small_person_abstains_on_vest():
    """Too far away to judge: abstain rather than manufacture a violation."""
    t = ComplianceTracker(CFG)
    tiny = person(h=40)  # below min_person_height_px
    out = settle(t, [tiny, helmet(left=105, top=102, w=15, h=12)])
    assert "vest" not in out[0]["label"].lower()

def test_vest_on_neighbour_does_not_clear_person():
    t = ComplianceTracker(CFG)
    far_vest = Box(PPE_VEST, 0.9, 900, 160, 40, 70)   # nowhere near our person
    out = settle(t, [person(), helmet(), far_vest])
    assert "vest" in out[0]["label"].lower()


# --- debouncing: the thing that makes the demo not look broken -------------------------------

def test_single_bad_frame_does_not_flip_verdict():
    """One dropped helmet detection must not strobe a compliant worker to red."""
    t = ComplianceTracker(CFG)
    good = [person(), helmet(), vest()]
    settle(t, good, frames=20)
    out = t.update(0, [person(), vest()])          # helmet momentarily missing
    assert out[0]["state"] == COMPLIANT

def test_sustained_change_does_flip_verdict():
    """A genuine change must still get through — hysteresis, not deafness."""
    t = ComplianceTracker(CFG)
    settle(t, [person(), helmet(), vest()], frames=20)
    out = settle(t, [person(), bare_head(), vest()], frames=20)
    assert out[0]["state"] == VIOLATION

def test_tracks_are_independent_per_stream():
    """Same track id on two cameras must not share state."""
    t = ComplianceTracker(CFG)
    settle(t, [person(track=7), bare_head(), vest()], frames=20, stream=0)
    out = settle(t, [person(track=7), helmet(), vest()], frames=20, stream=1)
    assert out[0]["state"] == COMPLIANT
    assert t.violation_count(0) == 1
    assert t.violation_count(1) == 0


# --- fire latching ---------------------------------------------------------------------------

def test_fire_latches_after_detection_stops():
    t = ComplianceTracker(CFG)
    fire = Box(0, 0.9, 10, 10, 50, 50)              # class 0 = fire
    assert t.update_fire(0, [fire])["label"] == "FIRE"
    assert t.update_fire(0, []) is not None          # still latched

def test_other_class_is_not_an_alert():
    """`other` is a training-set catch-all, not a safety signal."""
    t = ComplianceTracker(CFG)
    other = Box(1, 0.99, 10, 10, 50, 50)             # class 1 = other
    assert t.update_fire(0, [other], ["fire", "other", "smoke"]) is None

def test_low_confidence_fire_is_ignored():
    t = ComplianceTracker(CFG)
    faint = Box(0, 0.10, 10, 10, 50, 50)
    assert t.update_fire(0, [faint]) is None


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}  {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}  {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
