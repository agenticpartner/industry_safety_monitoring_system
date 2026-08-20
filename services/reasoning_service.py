#!/usr/bin/env python3
"""
Reasoning service — a VLM re-checks incidents the CV pipeline already reported.

    python3 services/reasoning_service.py [--once] [--limit N] [--dry-run]

**CV proposes, VLM disposes.** The detector is fast and literal; it flags a person with no
overlapping vest box. It also, on this footage, flags traffic cones and wet-floor signs as people.
A VLM looking at the actual pixels can say "that is a traffic cone, not a worker" — and that
rejection is the number that justifies this whole layer.

## This is a COLD path and must stay one

It reads incidents that are **already stored and already alerted on** and enriches them
afterwards. Nothing waits for it:

* Separate process, `nice`-able, killable at any moment.
* Polls SQLite exactly like `clip_service.py` — restartable, inspectable with a SELECT, and
  naturally rate-limited. That pattern was proven in 2.3, so it is reused rather than reinvented.
* **One request in flight.** Not a thread pool: the VLM shares the iGPU with 20 camera streams,
  and Phase 2.0 measured that a sustained burst costs the pipeline 3.3% at `dfi=2` — which holds
  precisely because llama-server runs `--parallel 1` and we never queue more.
* **Circuit breaker.** Consecutive failures open it; while open the service sleeps instead of
  hammering a sick endpoint. Incidents stay `unverified` and the product keeps working, less
  richly — the plan's stated degradation mode.

## The verdict is schema-constrained, not parsed out of prose

Phase 2.0 caught the model emitting `VERDICT: rejected` next to "no hi-vis vest is visible" —
internally contradictory, and impossible to act on. llama.cpp supports OpenAI `response_format:
json_schema` with enum enforcement (verified on this build), so the verdict cannot be a sentence
and cannot be a value outside the enum.

## Severity is NOT overwritten

`severity` records what the CV rules and zones concluded. `vlm_verdict` is a separate, additive
field. Overwriting severity with the VLM's opinion would destroy the ability to ask the question
that matters — "how often does the VLM disagree with a high-severity detection?" — and it is also
how VSS models it: `vlm_verdict` is its own filter on `video_analytics__get_incidents`, not a
severity override.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
# `app/` holds events.py, which this module needs for the escalation event types. Declared at
# module level rather than inside main() so importing the module — as the tests do — is enough
# to use it; a path set up only on the service's own startup path is not a real dependency
# declaration, it is a coincidence that happens to hold at runtime.
sys.path.insert(0, str(ROOT / "app"))
from store import connect  # noqa: E402

# Verdict vocabulary. `confirmed` / `rejected` match the VSS
# `video_analytics__get_incidents(vlm_verdict=...)` filter exactly, so an agent or dashboard
# written against this keeps working against a real VSS backend. VSS's third value, `unverified`,
# is our NULL (never processed); `uncertain` is the distinct case of "the VLM looked and could not
# tell", which is worth separating from "nobody looked yet".
CONFIRMED, REJECTED, UNCERTAIN = "confirmed", "rejected", "uncertain"

# Incident lifecycle: new -> reasoning -> verified | unverified
ST_NEW, ST_REASONING, ST_VERIFIED, ST_UNVERIFIED = "new", "reasoning", "verified", "unverified"

# ---------------------------------------------------------------------------------------------
# Ask the VLM what it SEES. Decide the verdict here.
# ---------------------------------------------------------------------------------------------
# The first version of this file asked the model for the verdict directly and got 100% rejections,
# including an overcrowding incident where it reported 4 people against a limit of 2. The reasons
# it gave were internally contradictory — "the area appears clear of people, with only three
# individuals visible". Two causes, both mine:
#
#   1. The prompt primed rejection by explaining that the detector makes mistakes. Phase 2.0
#      already established this model follows its prompt closely.
#   2. It was asked to make a POLICY judgement it did not have the inputs for. "Does this look
#      crowded" is subjective; the occupancy limit was never in the prompt.
#
# So the split is now: the VLM answers **perception** questions (is that a person, is there a
# vest, how many people), and the verdict is **computed** from those answers plus the rule that
# fired. A schema can constrain shape but not coherence — taking the judgement back into code is
# what actually makes it reliable, and it makes every verdict explainable from its inputs.
CANNOT_TELL = "cannot_tell"
YES_NO = {"type": "string", "enum": ["yes", "no", CANNOT_TELL]}

# Deliberately free text rather than an enum of hazard types. An enum is a list of hazards
# somebody thought of in advance, and the entire value of asking a VLM is that it can report the
# one nobody anticipated — a spill, a fall, a blocked fire exit, a toppled pallet load. The
# detectors already cover the anticipated classes; this field exists for everything else.
#
# `maxLength` is mandatory, not cosmetic: an unbounded string under grammar-constrained decoding
# never terminates and will run to max_tokens on every call. That bug cost the agent 99 of its
# 105 seconds — see bench/agent.md §10.
HAZARD_TEXT = {"type": "string", "maxLength": 90}
# The prompts ask for "one sentence"; the schema has to enforce it, because a prompt is a request
# and a grammar is a rule. These were unbounded — the same defect as the agent's, found by the
# schema scan added to the tests rather than by noticing the latency.
DESC_TEXT = {"type": "string", "maxLength": 300}

SCHEMAS = {
    "ppe_violation": {
        "name": "ppe_observation",
        "schema": {
            "type": "object",
            "properties": {
                # The dominant false positive on this footage is not a PPE misjudgement — it is
                # the detector calling a traffic cone or a wet-floor sign a person. Asking this
                # first, as its own boolean, turns that into something measurable.
                "subject_is_person": YES_NO,
                "wearing_high_vis_vest": YES_NO,
                "wearing_hard_hat": YES_NO,
                "people_visible": {"type": "integer"},
                # Asked on EVERY incident type, not just fire_alert. The VLM was already seeing
                # fire while adjudicating PPE incidents and mentioning it in `description` ("a
                # large fire is engulfing a cardboard box on the floor") — where nothing could
                # act on it, because prose is not a signal. Asking it as a boolean turns an
                # incidental observation into something the code can escalate on.
                "fire_or_smoke_visible": YES_NO,
                "hazard_visible": YES_NO,
                "hazard_description": HAZARD_TEXT,
                "description": DESC_TEXT,
            },
            "required": ["subject_is_person", "wearing_high_vis_vest", "wearing_hard_hat",
                         "people_visible", "fire_or_smoke_visible", "hazard_visible",
                         "description"],
        },
    },
    "overcrowding": {
        "name": "occupancy_observation",
        "schema": {
            "type": "object",
            "properties": {
                "people_visible": {"type": "integer"},
                "fire_or_smoke_visible": YES_NO,
                "hazard_visible": YES_NO,
                "hazard_description": HAZARD_TEXT,   # see the note in ppe_observation
                "description": DESC_TEXT,
            },
            "required": ["people_visible", "fire_or_smoke_visible", "hazard_visible",
                         "description"],
        },
    },
    "fire_alert": {
        "name": "fire_observation",
        "schema": {
            "type": "object",
            "properties": {
                "fire_or_smoke_visible": YES_NO,
                "hazard_visible": YES_NO,
                "hazard_description": HAZARD_TEXT,
                "people_visible": {"type": "integer"},
                "description": DESC_TEXT,
            },
            "required": ["fire_or_smoke_visible", "people_visible", "description"],
        },
    },
}

# Neutral prompts. They describe the scene and ask what is visible; they do NOT mention what the
# detector concluded or that it is unreliable, because saying either biases the answer.
PROMPTS = {
    "ppe_violation": (
        "These are frames from a fixed overhead camera in a warehouse (camera {camera_id}"
        "{zone_txt}). The final image is a close crop of one object the system is tracking.\n\n"
        "Answer only from what is visible:\n"
        "- subject_is_person: is the object in the close crop a human being? Traffic cones, "
        "wet-floor signs, boxes, pallets and machinery are NOT people.\n"
        "- wearing_high_vis_vest: is that subject wearing a high-visibility safety vest "
        "(bright yellow/orange, usually with reflective stripes)?\n"
        "- wearing_hard_hat: is that subject wearing a hard hat / safety helmet?\n"
        "- people_visible: how many people are visible across all the frames.\n"
        "- fire_or_smoke_visible: are there actual flames or smoke anywhere in these frames?\n"
        "- hazard_visible: apart from that, is anything in these frames an immediate "
        "safety hazard or emergency — a spill, a fall, someone on the floor, a blocked "
        "exit, a collapsed or toppling load, a collision? Answer 'no' for an ordinary "
        "working scene.\n"
        "- hazard_description: if yes, name it in a few words. Empty if no.\n"
        "- description: one sentence describing the subject. If flames or smoke are "
        "visible in any frame, say so.\n"
        "Answer '{cannot_tell}' for anything the images do not let you determine."
    ),
    "overcrowding": (
        "These are frames from a fixed overhead camera in a warehouse (camera {camera_id}"
        "{zone_txt}).\n\n"
        "Answer only from what is visible:\n"
        "- people_visible: the number of human beings you can count. Traffic cones, wet-floor "
        "signs, boxes, pallets and machinery are NOT people. Give your best single number.\n"
        "- fire_or_smoke_visible: are there actual flames or smoke anywhere in these frames?\n"
        "- hazard_visible: apart from that, is anything in these frames an immediate "
        "safety hazard or emergency — a spill, a fall, someone on the floor, a blocked "
        "exit, a collapsed or toppling load, a collision? Answer 'no' for an ordinary "
        "working scene.\n"
        "- hazard_description: if yes, name it in a few words. Empty if no.\n"
        "- description: one sentence describing the scene. If flames, smoke or a hazard "
        "are visible in ANY of the frames, describe those rather than the calm part of "
        "the scene."
    ),
    "fire_alert": (
        "These are frames from a fixed overhead camera in a warehouse (camera {camera_id}"
        "{zone_txt}).\n\n"
        "Answer only from what is visible:\n"
        "- fire_or_smoke_visible: can you see actual flames or smoke?\n"
        "- hazard_visible: apart from that, is anything in these frames an immediate "
        "safety hazard or emergency — a spill, a fall, someone on the floor, a blocked "
        "exit, a collapsed or toppling load, a collision? Answer 'no' for an ordinary "
        "working scene.\n"
        "- hazard_description: if yes, name it in a few words. Empty if no.\n"
        "- people_visible: how many people are visible.\n"
        "- description: one sentence describing the scene. If flames, smoke or a hazard "
        "are visible in ANY of the frames, describe those rather than the calm part of "
        "the scene."
    ),
}


def escalation_for(etype: str, obs: dict) -> tuple[str, str, str, str] | None:
    """Should this observation raise a NEW alert of its own? (type, severity, label, why)

    The VLM is the only component that can report a hazard nobody trained a class for. It was
    already seeing them — "a large fire is engulfing a cardboard box on the floor" turned up
    inside a *PPE* incident's description, where nothing could act on it because prose is not a
    signal. Both fire and the general hazard are now asked as their own fields, and the decision
    to raise an alert is made HERE, in code, from those answers. Never by grepping the
    description for words like "fire": that is a hidden list of anticipated hazards, and it would
    also fire on "no fire visible".

    Escalations are ALERTS, not verdicts. Their job is to get an operator looking at the clip.
    """
    from events import FIRE_ALERT, HAZARD_ALERT, SEV_CRITICAL, SEV_HIGH  # noqa: PLC0415

    # Escalate ONLY from detector-originated incidents. An escalated alert must never escalate
    # again, or the two rules feed each other forever — observed on the first run:
    #
    #   ESCALATED hazard_alert from fire_alert   (a fire is, correctly, also a hazard)
    #   ESCALATED fire_alert   from hazard_alert (that hazard is, correctly, a fire)
    #   ... and round again, one new incident per cycle, each with its own clip and VLM call.
    #
    # Excluding the alert types is enough because they are the only types this function creates,
    # so the cycle has no other entry point.
    if etype in (FIRE_ALERT, HAZARD_ALERT):
        return None

    if obs.get("fire_or_smoke_visible") == "yes" and etype != FIRE_ALERT:
        return FIRE_ALERT, SEV_CRITICAL, "FIRE (seen by VLM)", "Fire or smoke"

    if obs.get("hazard_visible") == "yes":
        what = (obs.get("hazard_description") or "").strip()
        # A hazard the model cannot name is not actionable; treating a bare "yes" as an alert
        # would flood the operator with unreviewable notifications.
        if not what:
            return None
        return HAZARD_ALERT, SEV_HIGH, f"HAZARD: {what[:60]}", what
    return None


def escalate(emitter, alert_type: str, severity: str, label: str, why: str,
             camera_id: int, zone: str | None, pts_ns: int | None,
             obs: dict, from_event_id: str) -> None:
    """Publish an escalated alert on the normal event path.

    Emitted as an open/close PAIR with a unique track id, exactly like `POST /alerts/test`, and
    for the same reason: a single opening event with the default `track_id = -1` collides with
    every other synthetic emission on that camera and gets silently absorbed as a duplicate,
    while leaving an incident open forever that swallows all later ones.

    `source_pts_ns` is carried over from the originating incident, so the escalated alert gets
    its own evidence clip cut from the same moment in the source — which is the whole point:
    the operator is being asked to LOOK.
    """
    from events import Event  # noqa: PLC0415 — dependency-free module, imported lazily

    desc = (obs.get("description") or "").strip()
    track = -(abs(hash(from_event_id)) % 2_000_000_000) - 2
    common = dict(camera_id=camera_id, type=alert_type, severity=severity,
                  track_id=track, label=label, zone=zone, source_pts_ns=pts_ns,
                  vlm_verdict=CONFIRMED,
                  vlm_reason=f"{why} seen by the VLM while reviewing incident "
                             f"{from_event_id[:8]}. {desc}"[:600])
    emitter.emit(Event(**common))
    emitter.emit(Event(**common, ended=True, duration_s=1.0))


def decide(etype: str, label: str, obs: dict, threshold: int | None) -> tuple[str, str]:
    """Turn observations into (verdict, reason). All policy lives here, none in the model."""
    desc = (obs.get("description") or "").strip()
    people = obs.get("people_visible")

    if etype == "overcrowding":
        if threshold is None:
            return UNCERTAIN, f"no occupancy limit configured for this zone; saw {people} people"
        if not isinstance(people, int):
            return UNCERTAIN, desc
        if people > threshold:
            return CONFIRMED, f"counted {people} people against a limit of {threshold}. {desc}"
        return REJECTED, f"counted {people} people, within the limit of {threshold}. {desc}"

    if etype == "fire_alert":
        v = obs.get("fire_or_smoke_visible")
        if v == "yes":
            # State the FINDING first, exactly as the PPE and overcrowding branches do. Returning
            # the bare description made confirmed fires read as "a worker in a yellow hard hat
            # walks through an aisle" — because a clip carries pre-roll before the flames and the
            # model's one sentence anchors on the opening frames. The verdict was right and every
            # human-readable surface (feed, Telegram caption, agent answer) said the opposite;
            # the agent even refused to show "the fire clip" while quoting that very clip.
            return CONFIRMED, f"flames or smoke visible. {desc}"
        if v == "no":
            return REJECTED, f"no flames or smoke visible. {desc}"
        return UNCERTAIN, desc

    # ---- ppe_violation ----
    if obs.get("subject_is_person") == "no":
        # The headline case: the detector tracked a cone or a sign as a worker.
        return REJECTED, f"the tracked object is not a person. {desc}"
    if obs.get("subject_is_person") == CANNOT_TELL:
        return UNCERTAIN, f"cannot tell whether the tracked object is a person. {desc}"

    up = (label or "").upper()
    # A violation label can name both ("NO HELMET + no vest?"). Confirming either is enough for
    # the incident to stand, so the checks are ordered by evidence strength: helmet is a DIRECT
    # detection, vest is inferred from absence.
    checks = []
    if "HELMET" in up:
        checks.append(("wearing_hard_hat", "hard hat"))
    if "VEST" in up:
        checks.append(("wearing_high_vis_vest", "high-visibility vest"))
    if not checks:
        return UNCERTAIN, f"unrecognised violation label {label!r}. {desc}"

    saw_missing, saw_present, unsure = [], [], []
    for field, name in checks:
        v = obs.get(field)
        (saw_missing if v == "no" else saw_present if v == "yes" else unsure).append(name)

    if saw_missing:
        return CONFIRMED, f"person is not wearing a {' or '.join(saw_missing)}. {desc}"
    if unsure:
        return UNCERTAIN, f"cannot determine whether the {' or '.join(unsure)} is present. {desc}"
    return REJECTED, f"person IS wearing a {' and '.join(saw_present)}. {desc}"


def zone_thresholds() -> dict[tuple[int, str], int]:
    """{(camera_id, zone_name): object-threshold} from configs/analytics/zones.yml.

    Read from the same file that generates the nvdsanalytics config, so the limit the VLM is
    judged against is by construction the limit that fired the incident.
    """
    try:
        import yaml
        sys.path.insert(0, str(ROOT / "scripts"))
        from make_zones import resolve_camera
        cfg = yaml.safe_load((ROOT / "configs/analytics/zones.yml").read_text())
        out: dict[tuple[int, str], int] = {}
        for cam in range(1, 33):
            entry = resolve_camera(cfg, cam)
            if not entry:
                continue
            for z in entry.get("zones") or []:
                if z.get("kind") == "overcrowding":
                    out[(cam, z["name"])] = int(z.get("threshold", 4))
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[reasoning] could not load zone thresholds: {type(e).__name__}: {e}", flush=True)
        return {}


class CircuitBreaker:
    """Stop hammering an endpoint that is failing.

    Without this, a VLM that is down turns the service into a tight retry loop that burns a core
    the pipeline needs — the cold path degrading the hot path, which is the one thing the
    architecture must not allow.
    """

    def __init__(self, threshold: int = 5, cooldown_s: float = 60.0):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.opened_at = 0.0

    @property
    def is_open(self) -> bool:
        if self.failures < self.threshold:
            return False
        if time.monotonic() - self.opened_at >= self.cooldown_s:
            # Half-open: let one request through to test the water.
            self.failures = self.threshold - 1
            return False
        return True

    def record(self, ok: bool) -> None:
        if ok:
            self.failures = 0
        else:
            self.failures += 1
            if self.failures == self.threshold:
                self.opened_at = time.monotonic()


def b64_jpeg(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


class FrameSet:
    """Pulls the images a verdict is made from, out of the already-captured clip."""

    def __init__(self, work_dir: Path, n_frames: int, pre_roll_s: float, crop_pad: float = 0.35):
        self.work_dir = work_dir
        self.n_frames = n_frames
        self.pre_roll_s = pre_roll_s
        self.crop_pad = crop_pad
        work_dir.mkdir(parents=True, exist_ok=True)

    def _run(self, cmd: list[str]) -> bool:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def build(self, clip: Path, bbox: list | None,
              source: Path | None = None, source_offset_s: float | None = None,
              mode: str = "both") -> list[Path]:
        """`mode`: both | context | crop — which images the verdict is allowed to see.

        Exists so the question "does the model actually answer about the CROP, or is it just
        describing the scene?" can be measured instead of assumed. It is a real risk: when the
        crop was accidentally empty floor, the model still confidently described a worker in a
        hi-vis vest, which it could only have taken from the context frames.
        """
        for f in self.work_dir.glob("*.jpg"):
            f.unlink()
        out: list[Path] = []

        # Context frames spread across the clip. Downscaled to 640 wide: the VLM does not need
        # 1080p to tell a person from a traffic cone, and request size drives latency.
        if mode in ("both", "context") and self._run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip),
                 "-vf", f"fps={self.n_frames}/12,scale=640:-2",
                 "-frames:v", str(self.n_frames),
                 str(self.work_dir / "ctx_%02d.jpg")]):
            out.extend(sorted(self.work_dir.glob("ctx_*.jpg")))

        # A crop of exactly what the detector flagged, at exactly the frame it flagged it on.
        #
        # Cut from the SOURCE, not from the clip. Two errors made the clip unusable for this, and
        # both were caught by rendering a crop and looking at it — it was empty floor while the
        # model confidently described a worker in a hi-vis vest, having answered from the context
        # frames instead:
        #
        #   1. The incident is NOT always `pre_roll` seconds into the clip. When it opens near the
        #      start of the file the clip start is clamped to 0, so the incident sits at its raw
        #      offset instead.
        #   2. `-ss` before `-i` with `-c copy` snaps the clip to the preceding keyframe, so the
        #      clip's true start drifts from the requested one by up to a GOP (1s here).
        #
        # Seeking the source at the incident's own PTS removes both. This frame is re-encoded
        # anyway, so an accurate seek costs nothing worth counting.
        if (mode in ("both", "crop") and bbox and len(bbox) == 4
                and source and source.exists() and source_offset_s is not None):
            left, top, w, h = bbox
            pad_x, pad_y = w * self.crop_pad, h * self.crop_pad
            x = max(0, int(left - pad_x))
            y = max(0, int(top - pad_y))
            cw = max(32, int(w + 2 * pad_x))
            ch = max(32, int(h + 2 * pad_y))
            crop = self.work_dir / "crop.jpg"
            # `-ss` goes BEFORE `-i` (fast input seek). No `-accurate_seek` flag: it is an INPUT
            # option, so placing it after `-i` makes ffmpeg reject the whole command — and it is
            # on by default for decoded output anyway, which this is, since the crop is
            # re-encoded rather than stream-copied.
            if self._run(["ffmpeg", "-y", "-loglevel", "error",
                          "-ss", f"{max(0.0, source_offset_s):.3f}", "-i", str(source),
                          "-frames:v", "1",
                          # exact=0 clamps an out-of-bounds crop instead of failing outright.
                          "-vf", f"crop={cw}:{ch}:{x}:{y}:exact=0,scale=448:-2",
                          str(crop)]) and crop.exists() and crop.stat().st_size > 0:
                out.append(crop)
        return out


class VLMClient:
    def __init__(self, endpoint: str, model: str, timeout: float, max_tokens: int = 200):
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def observe(self, prompt: str, frames: list[Path], schema: dict) -> dict:
        content: list[dict] = [{"type": "image_url", "image_url": {"url": b64_jpeg(f)}}
                               for f in frames]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            # Near-greedy: a safety verdict must be reproducible. If the same frames can yield
            # 'confirmed' and 'rejected' on consecutive calls, the layer launders randomness as
            # judgement.
            "temperature": 0.1,
            "response_format": {"type": "json_schema", "json_schema": schema},
        }
        req = urllib.request.Request(
            self.endpoint, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.load(resp)
        text = body["choices"][0]["message"]["content"]
        out = json.loads(text)
        # The schema guarantees shape, not sanity. A negative person count means the decode went
        # wrong, and storing it would quietly corrupt the precision numbers this layer exists to
        # produce.
        if not isinstance(out.get("people_visible"), int) or out["people_visible"] < 0:
            raise ValueError(f"bad people_visible: {out.get('people_visible')!r}")
        return out


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def load_cfg() -> dict:
    try:
        import yaml
        return yaml.safe_load((ROOT / "configs/services.yml").read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}


def reclaim_stuck(db: sqlite3.Connection, stale_s: float = 300.0) -> int:
    """Return incidents abandoned mid-flight to the queue.

    A crash or kill between "mark reasoning" and "write verdict" would otherwise strand a record
    in `reasoning` forever. The service is explicitly designed to be killable, so this is a normal
    path, not an exceptional one.
    """
    cur = db.execute(
        "UPDATE events SET state = ? WHERE state = ? AND ts < ?",
        (ST_NEW, ST_REASONING, time.time() - stale_s))
    db.commit()
    return cur.rowcount


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--frames", type=int, default=4, help="context frames from the clip")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0, help="stop after N incidents (0 = unlimited)")
    ap.add_argument("--once", action="store_true", help="drain what is ready, then exit")
    ap.add_argument("--dry-run", action="store_true", help="do everything except write verdicts")
    ap.add_argument("--images", choices=["auto", "both", "context", "crop"], default="auto",
                    help="which images the VLM sees. 'auto' (default) picks per event type: "
                         "crop-only for PPE, context for overcrowding/fire. Override to "
                         "reproduce the accuracy comparison in bench/reasoning.md")
    ap.add_argument("--redo", action="store_true",
                    help="re-verify incidents that already have a verdict")
    args = ap.parse_args()

    cfg = load_cfg()
    r_cfg = cfg.get("reasoning") or {}
    store_cfg = cfg.get("store") or {}
    clip_cfg = cfg.get("clips") or {}

    db_path = Path(args.db or store_cfg.get("path", "data/events.db"))
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    endpoint = args.endpoint or r_cfg.get("endpoint", "http://127.0.0.1:8000/v1")
    model = args.model or r_cfg.get("model", "Cosmos-Reason2-2B")
    pre_roll = float(clip_cfg.get("pre_roll_s", 6))

    if not shutil.which("ffmpeg"):
        print("[reasoning] ERROR: ffmpeg not on PATH", flush=True)
        return 2

    db = connect(db_path)
    if args.redo:
        db.execute("UPDATE events SET state = ?, vlm_verdict = NULL, vlm_reason = NULL",
                   (ST_NEW,))
        db.commit()
    n = reclaim_stuck(db)
    if n:
        print(f"[reasoning] reclaimed {n} incident(s) stuck in '{ST_REASONING}'", flush=True)

    # Publishing escalations goes back through Redis on the same path the pipeline uses, so an
    # escalated alert is indistinguishable downstream from a detected one — same store, same
    # clip service, same dashboard. `app/events.py` is dependency-free by design (it has to run
    # on system python inside the pipeline), so importing it here costs nothing.
    emitter = None
    try:
        sys.path.insert(0, str(ROOT / "app"))
        from events import EventEmitter  # noqa: PLC0415
        redis_cfg = (cfg.get("events") or {}).get("redis") or {}
        emitter = EventEmitter(host=redis_cfg.get("host", "127.0.0.1"),
                               port=int(redis_cfg.get("port", 6379)),
                               stream=redis_cfg.get("stream", "safety:events"))
    except Exception as e:  # noqa: BLE001
        # Escalation is additive. Losing it must not stop incidents being adjudicated, which is
        # this service's actual job.
        print(f"[reasoning] escalation disabled ({type(e).__name__}: {str(e)[:80]})", flush=True)

    frames = FrameSet(ROOT / "build/reasoning_frames", args.frames, pre_roll)
    vlm = VLMClient(endpoint, model, args.timeout)
    breaker = CircuitBreaker(threshold=int(r_cfg.get("breaker_threshold", 5)),
                             cooldown_s=float(r_cfg.get("breaker_cooldown_s", 60)))
    thresholds = zone_thresholds()
    source_durations: dict[Path, float | None] = {}

    print(f"[reasoning] db={db_path} endpoint={endpoint} model={model} "
          f"frames={args.frames}{' DRY-RUN' if args.dry_run else ''}", flush=True)

    running = True

    def _stop(_s, _f):
        nonlocal running
        running = False
        print("[reasoning] stopping", flush=True)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    done = 0
    counts: dict[str, int] = {}
    latencies: list[float] = []

    while running:
        if breaker.is_open:
            print(f"[reasoning] circuit OPEN ({breaker.failures} consecutive failures) — "
                  f"sleeping; incidents stay unverified", flush=True)
            time.sleep(min(10.0, breaker.cooldown_s))
            continue

        # Only incidents with a clip are reasonable-about. Highest severity first, then oldest:
        # if the backlog never drains, the operator should at least have verdicts on the ones
        # that matter.
        row = db.execute(
            "SELECT event_id, camera_id, type, severity, label, zone, bbox, clip_uri, "
            "       source_pts_ns "
            "  FROM events "
            " WHERE state = ? AND clip_state = 'ready' AND clip_uri IS NOT NULL "
            " ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 ELSE 2 END, ts "
            " LIMIT 1", (ST_NEW,)).fetchone()

        if row is None:
            if args.once:
                break
            time.sleep(args.interval)
            continue

        event_id, camera_id, etype, severity, label, zone, bbox_json, clip_uri, pts_ns = row
        clip = ROOT / clip_uri
        if not clip.exists():
            db.execute("UPDATE events SET state = ?, vlm_reason = ? WHERE event_id = ?",
                       (ST_UNVERIFIED, "clip file missing", event_id))
            db.commit()
            counts["clip-missing"] = counts.get("clip-missing", 0) + 1
            continue

        # Claim it before the slow part, so a second instance cannot pick up the same incident.
        db.execute("UPDATE events SET state = ? WHERE event_id = ?", (ST_REASONING, event_id))
        db.commit()

        bbox = json.loads(bbox_json) if bbox_json else None
        # Where in the SOURCE the incident happened — the same modulo the clip service used, so
        # the crop lands on the exact frame the detector fired on.
        source = ROOT / "media" / f"cam{camera_id:02d}.mp4"
        src_dur = source_durations.get(source)
        if src_dur is None and source.exists():
            src_dur = probe_duration(source)
            source_durations[source] = src_dur
        offset = ((pts_ns / 1e9) % src_dur) if (pts_ns and src_dur) else None
        # Per-type image policy. This is NOT a tuning preference — it was measured by rendering
        # the crops and adjudicating them by eye (bench/reasoning.md §3):
        #
        #   context+crop  ~50% of PPE rejections WRONG. The model described a generic worker in a
        #                 hi-vis vest almost regardless of the crop — on two incidents the crop
        #                 held a traffic cone and a fire extinguisher and it still described a
        #                 worker. The context frames were bleeding into the answer.
        #   crop-only     7/8 correct, with specifically accurate descriptions ("teal shirt and
        #                 beige pants", "red apron", "reflective stripes").
        #
        # Overcrowding keeps the context frames because its question is "how many people are in
        # this scene" — there is no subject to crop, and the scene IS the evidence.
        mode = args.images
        if mode == "auto":
            mode = "crop" if etype == "ppe_violation" else "context"
        imgs = frames.build(clip, bbox, source, offset, mode=mode)
        if not imgs:
            db.execute("UPDATE events SET state = ?, vlm_reason = ? WHERE event_id = ?",
                       (ST_UNVERIFIED, "could not extract frames from clip", event_id))
            db.commit()
            counts["no-frames"] = counts.get("no-frames", 0) + 1
            continue

        zone_txt = f", zone '{zone}'" if zone else ""
        prompt = PROMPTS.get(etype, PROMPTS["ppe_violation"]).format(
            camera_id=camera_id, zone_txt=zone_txt, cannot_tell=CANNOT_TELL)
        if mode == "crop" and etype == "ppe_violation":
            # Only one image is sent, so "the final image is a close crop" is misleading.
            prompt = prompt.replace(
                " The final image is a close crop of one object the system is tracking.",
                " This image is a close crop of the single object the system is tracking.")
        if mode == "context":
            # Do not tell the model there is a crop when none was sent — a prompt describing an
            # image that is not there is its own source of confabulation.
            prompt = prompt.replace(
                " The final image is a close crop of one object the system is tracking.", "")
            prompt = prompt.replace("the object in the close crop", "the person being tracked")
        schema = SCHEMAS.get(etype, SCHEMAS["ppe_violation"])

        t0 = time.monotonic()
        try:
            obs = vlm.observe(prompt, imgs, schema)
            dt = time.monotonic() - t0
            breaker.record(True)
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError,
                json.JSONDecodeError, OSError) as e:
            dt = time.monotonic() - t0
            breaker.record(False)
            msg = f"{type(e).__name__}: {str(e)[:120]}"
            # Back to `new`, not a terminal state: a transient endpoint failure should be retried,
            # and the breaker is what stops that becoming a hot loop.
            db.execute("UPDATE events SET state = ? WHERE event_id = ?", (ST_NEW, event_id))
            db.commit()
            counts["error"] = counts.get("error", 0) + 1
            print(f"[reasoning] ERROR {event_id[:8]} after {dt:.1f}s: {msg}", flush=True)
            time.sleep(1.0)
            continue

        latencies.append(dt)
        people = obs.get("people_visible")
        verdict, reason = decide(etype, label, obs, thresholds.get((camera_id, zone or "")))
        state = ST_VERIFIED if verdict in (CONFIRMED, REJECTED) else ST_UNVERIFIED

        if not args.dry_run:
            db.execute(
                "UPDATE events SET vlm_verdict = ?, vlm_reason = ?, state = ? "
                " WHERE event_id = ?",
                (verdict, reason[:600], state, event_id))
            db.commit()

        # ---- hazard escalation ------------------------------------------------------------
        # The VLM looked at this incident's frames and saw something dangerous that the incident
        # was not about. That is a real observation from a real model on real frames, and it
        # publishes through the normal Redis path so it becomes a first-class incident with its
        # own clip — and, for fire, trips the dashboard alarm like any other fire.
        esc = escalation_for(etype, obs) if not args.dry_run and emitter is not None else None
        if esc is not None:
            alert_type, esc_sev, esc_label, why = esc
            escalate(emitter, alert_type, esc_sev, esc_label, why,
                     camera_id, zone, pts_ns, obs, event_id)
            counts[f"escalated-{alert_type}"] = counts.get(f"escalated-{alert_type}", 0) + 1
            print(f"[reasoning] ESCALATED {alert_type} from {etype} {event_id[:8]} "
                  f"cam{camera_id:02d} | {esc_label}", flush=True)

        counts[verdict] = counts.get(verdict, 0) + 1
        done += 1
        print(f"[reasoning] {event_id[:8]} cam{camera_id:02d} {etype:14s} {severity:8s} "
              f"{str(zone):14s} -> {verdict.upper():9s} people={people} {dt:5.1f}s | "
              f"{reason[:90]}", flush=True)

        if args.limit and done >= args.limit:
            break

    if latencies:
        latencies.sort()
        med = latencies[len(latencies) // 2]
        print(f"\n[reasoning] {dict(sorted(counts.items()))} | median {med:.1f}s "
              f"over {len(latencies)} calls", flush=True)
    else:
        print(f"\n[reasoning] {dict(sorted(counts.items()))}", flush=True)

    pending = db.execute(
        "SELECT COUNT(*) FROM events WHERE state = ? AND clip_state = 'ready'",
        (ST_NEW,)).fetchone()[0]
    print(f"[reasoning] backlog: {pending} incident(s) awaiting reasoning", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
