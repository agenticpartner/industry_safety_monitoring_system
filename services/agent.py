#!/usr/bin/env python3
"""
Q&A agent over incident records — LLM-planned retrieval, grounded answers, clip citations.

    python3 services/agent.py "which cameras had confirmed violations in a vehicle zone?"
    python3 services/agent.py --chat                      # conversational, follow-ups
    python3 services/agent.py --tool video_analytics__get_incidents --args '{"max_count":5}'
    python3 services/agent.py --list-tools

## Tool names match VSS on purpose

`video_analytics__get_incidents`, `video_analytics__get_incident`,
`video_analytics__get_sensor_ids` are the exact names and parameters the VSS blueprint's MCP
server exposes on port 9901, including `vlm_verdict` ∈ `confirmed | rejected | unverified`
(`unverified` = nobody has looked yet = our `vlm_verdict IS NULL`). A dashboard or agent written
against these keeps working against a real VSS backend on Thor — "replace a service, not rewrite
the system". Tools without a VSS counterpart are namespaced separately so the line stays visible.

## No hardcoded questions, and no hardcoded vocabulary

An earlier version matched keywords with regexes (`helmet|hard ?hat` → `ppe_violation`). That is
a hidden list of anticipated questions: it answers what was foreseen and silently mis-files
everything else. It is gone.

Instead the LLM plans the retrieval, and the **enums it must choose from are built from the
database at runtime** — the zones, cameras, event types and severities that actually exist in the
data right now. Add a camera or rename a zone and the agent follows, with no code change. Rules
remain, but only as guardrails that cannot invent anything:

* the plan is schema-constrained to those live enums, so an unknown zone cannot be produced;
* arguments are validated and coerced before any SQL runs;
* every citation is checked against the rows actually retrieved, and invented ids are dropped;
* if planning fails entirely, retrieval degrades to full-text search rather than to an error.

An answer can be clumsily worded. It cannot cite an incident that does not exist, filter on a zone
that was never configured, or invent a camera.

## Why a bigger model here than for verification

The reasoning service runs per incident and continuously — its model has to be cheap. The agent
runs **per human question**, which is a very low duty cycle. That difference is what makes
Nemotron Nano 9B affordable for planning and synthesis while Cosmos Reason 2 2B stays the right
choice for per-incident vision.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_service import search, summarise, sync  # noqa: E402
from store import connect  # noqa: E402


# ---------------------------------------------------------------------------------------------
# Live vocabulary — the enums the planner may choose from, read from the data
# ---------------------------------------------------------------------------------------------
def vocabulary(db: sqlite3.Connection) -> dict:
    """What actually exists in the store right now.

    Read from the database rather than declared in code so the agent tracks the deployment: a new
    camera, a renamed zone or a new event type is immediately selectable without an edit here.
    """
    col = lambda sql: [r[0] for r in db.execute(sql) if r[0] is not None]  # noqa: E731
    return {
        "sensors": [f"cam{c:02d}" for c in col(
            "SELECT DISTINCT camera_id FROM events ORDER BY camera_id")],
        "event_types": col("SELECT DISTINCT type FROM events ORDER BY type"),
        "severities": col("SELECT DISTINCT severity FROM events ORDER BY severity"),
        "zones": col("SELECT DISTINCT zone FROM events WHERE zone IS NOT NULL ORDER BY zone"),
        "verdicts": ["confirmed", "rejected", "unverified", "any"],
    }


# ---------------------------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------------------------
def _row_public(r: dict, includes: list[str] | None = None) -> dict:
    includes = includes or []
    out = {
        "id": r["event_id"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["ts"])),
        "sensor": f"cam{r['camera_id']:02d}",
        "type": r["type"],
        "severity": r["severity"],
        "zone": r.get("zone"),
        "description": r.get("label"),
        "vlm_verdict": r.get("vlm_verdict") or "unverified",
        "state": r.get("state"),
        "has_clip": bool(r.get("clip_uri")),
    }
    if r.get("duration_s") is not None:
        out["duration_s"] = round(r["duration_s"], 1)
    if "info" in includes:
        out["vlm_reason"] = r.get("vlm_reason")
        out["hits"] = r.get("hits")
        out["clip_uri"] = r.get("clip_uri")
    if "objectIds" in includes:
        out["track_id"] = r.get("track_id")
    return out


def _cam(source) -> int | None:
    if source in (None, "", "any"):
        return None
    m = re.search(r"(\d+)", str(source))
    return int(m.group(1)) if m else None


def _iso(v):
    if v in (None, "", "any"):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(str(v), fmt))
        except ValueError:
            continue
    return None


def video_analytics__get_incidents(db, source=None, source_type="sensor", start_time=None,
                                   end_time=None, max_count=10, includes=None,
                                   vlm_verdict=None, **_) -> dict:
    """VSS-compatible incident listing. `source` is a sensor id like 'cam02'."""
    rows = search(db, camera_id=_cam(source), vlm_verdict=vlm_verdict,
                  since_ts=_iso(start_time), until_ts=_iso(end_time), limit=int(max_count))
    return {"incidents": [_row_public(r, includes) for r in rows], "count": len(rows)}


def video_analytics__get_incident(db, id=None, includes=None, **_) -> dict:
    """VSS-compatible single-incident fetch."""
    if not id:
        return {"error": "id is required"}
    r = db.execute("SELECT * FROM events WHERE event_id = ? OR event_id LIKE ?",
                   (id, f"{id}%")).fetchone()
    if not r:
        return {"error": f"no incident {id}"}
    return {"incident": _row_public(dict(r), includes or ["info", "objectIds"])}


def video_analytics__get_sensor_ids(db, **_) -> dict:
    """VSS-compatible sensor listing, with how much each has to say."""
    rows = db.execute("SELECT camera_id, COUNT(*) n, MAX(ts) last FROM events "
                      "GROUP BY camera_id ORDER BY camera_id").fetchall()
    return {"sensors": [
        {"id": f"cam{r['camera_id']:02d}", "incidents": r["n"],
         "last_incident": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["last"]))}
        for r in rows]}


def search_incidents(db, text=None, sensor=None, event_type=None, severity=None, zone=None,
                     vlm_verdict=None, hours=None, open_only=False, max_count=10, **_) -> dict:
    """Ours: structured filters + full-text search over incident records."""
    # Only rebuild the FTS index when text search will actually use it. `sync()` WRITES, and the
    # event service holds the single WAL writer slot while incidents are landing — an unnecessary
    # write here turns a read query into "database is locked".
    if text:
        sync(db)
    since = (time.time() - float(hours) * 3600) if hours else None
    rows = search(db, text=text, camera_id=_cam(sensor), event_type=event_type,
                  severity=severity, zone=zone, vlm_verdict=vlm_verdict,
                  since_ts=since, open_only=bool(open_only), limit=int(max_count))
    # Retrieval returns a ranked PAGE, so counting the rows answers "how many did you show me",
    # not "how many are there". Rather than warn the model about that and hope, the exact counts
    # for the same filter set are attached to the result: the numbers it needs are already in the
    # payload, computed in SQL, so it never has to do arithmetic over rows. `filters` travels with
    # them because a total without its predicate is ambiguous — a bare "23" was read as
    # "23 confirmed" when it meant "23 in total".
    agg = summarise(db, camera_id=_cam(sensor), event_type=event_type, severity=severity,
                    zone=zone, vlm_verdict=vlm_verdict, since_ts=since,
                    open_only=bool(open_only))
    return {"incidents": [_row_public(r, ["info"]) for r in rows],
            "count": len(rows), "total_matching": agg["incidents"],
            "truncated": agg["incidents"] > len(rows),
            "counts": agg}


def get_clips(db, ids=None, text=None, event_type=None, zone=None, vlm_verdict=None,
              sensor=None, hours=None, max_count=5, **_) -> dict:
    """Ours: evidence clips for the incidents that best match, ranked by confidence."""
    if ids:
        rows = []
        for i in (ids if isinstance(ids, list) else [ids]):
            r = db.execute("SELECT * FROM events WHERE event_id = ? OR event_id LIKE ?",
                           (i, f"{i}%")).fetchone()
            if r:
                rows.append(dict(r))
    else:
        if text:
            sync(db)
        rows = search(db, text=text, camera_id=_cam(sensor), event_type=event_type, zone=zone,
                      vlm_verdict=vlm_verdict,
                      since_ts=(time.time() - float(hours) * 3600) if hours else None,
                      limit=int(max_count) * 3)
    scored = [r for r in (dict(x) for x in rows) if r.get("clip_uri")]
    for r in scored:
        r["_confidence"] = confidence(r)
    scored.sort(key=lambda r: -r["_confidence"])
    out = []
    for r in scored[:int(max_count)]:
        p = ROOT / r["clip_uri"]
        out.append({"id": r["event_id"][:8], "sensor": f"cam{r['camera_id']:02d}",
                    "type": r["type"], "zone": r.get("zone"),
                    "vlm_verdict": r.get("vlm_verdict") or "unverified",
                    "confidence": round(r["_confidence"], 2),
                    "clip": r["clip_uri"], "exists": p.exists(),
                    "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None,
                    "reason": r.get("vlm_reason")})
    return {"clips": out, "count": len(out)}


def get_summary(db, hours=None, vlm_verdict=None, event_type=None, zone=None, **_) -> dict:
    """Ours: aggregate analytics — counts by type, severity, verdict and camera.

    This is the tool for any "how many" / "which cameras" question. It counts over the whole
    table rather than over a retrieved page, so its numbers are the real ones.
    """
    return summarise(db, float(hours) if hours else None,
                     vlm_verdict=vlm_verdict, event_type=event_type, zone=zone)


TOOLS = {
    "video_analytics__get_incidents": video_analytics__get_incidents,
    "video_analytics__get_incident": video_analytics__get_incident,
    "video_analytics__get_sensor_ids": video_analytics__get_sensor_ids,
    "search_incidents": search_incidents,
    "get_clips": get_clips,
    "get_summary": get_summary,
}


def confidence(r: dict) -> float:
    """How much an incident deserves an operator's attention, 0..1.

    Not a probability — a ranking signal, combining evidence strength with how much the CV layer
    saw. It is computed here rather than asked of the LLM for the same reason verdicts are
    (Phase 2.4): a model asked to score confidence produces a fluent number with nothing behind
    it, while this one is reproducible and explainable from the record.
    """
    verdict = {"confirmed": 1.0, "unverified": 0.55, "rejected": 0.1}.get(
        r.get("vlm_verdict") or "unverified", 0.5)
    sev = {"critical": 1.0, "high": 0.8, "medium": 0.5}.get(r.get("severity"), 0.5)
    # More contributing tracks = more independent looks at the same situation. Saturates fast:
    # the difference between 1 and 10 sightings matters, between 100 and 200 does not.
    hits = min(1.0, (r.get("hits") or 1) / 20.0)
    return round(0.5 * verdict + 0.35 * sev + 0.15 * hits, 3)


# ---------------------------------------------------------------------------------------------
# LLM planning + synthesis
# ---------------------------------------------------------------------------------------------
def plan_schema(vocab: dict) -> dict:
    """Tool-call schema whose enums come from the live data, not from a literal list here.

    ## `maxLength` on the free-text fields is load-bearing, not decoration

    Every other property here is an enum or a number, so the grammar forces termination. `text`
    was an unbounded string, and under grammar-constrained decoding an unbounded string has no
    reason to stop: the planner emitted a correct plan and then kept writing the `text` value
    until it hit `max_tokens`, EVERY TIME. The call then raised "completion truncated", `ask()`
    swallowed it, and the agent silently fell back to a raw full-text search.

    That single missing constraint cost **99 of the agent's 105 seconds** and disabled planning
    entirely — the "planner keeps choosing search_incidents" behaviour recorded in bench/agent.md
    was this bug, not a weak model. Without the schema the same prompt answers in 1.5 s and 29
    tokens, which is what proved the grammar was the cause rather than the reasoning tokens.

    `additionalProperties: false` closes the other escape route: with it absent the grammar lets
    the model invent further keys once the required ones are written.
    """
    return {
        "name": "tool_call",
        "schema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": list(TOOLS)},
                "sensor": {"type": "string", "enum": vocab["sensors"] + ["any"]},
                "event_type": {"type": "string", "enum": vocab["event_types"] + ["any"]},
                "severity": {"type": "string", "enum": vocab["severities"] + ["any"]},
                "zone": {"type": "string", "enum": vocab["zones"] + ["any"]},
                "vlm_verdict": {"type": "string", "enum": vocab["verdicts"]},
                "hours": {"type": "number"},
                # A search phrase, not an essay, and NOT required: bounded but mandatory, the
                # model padded it to the bound with repetition ("violation, confirmed, ppe,
                # violation, confirmed, ppe, ...") because it had to write something. That is
                # harmless for an aggregate but poisons FTS relevance on a search. Optional lets
                # it be omitted when the question has no free-text part.
                "text": {"type": "string", "maxLength": 80},
                "max_count": {"type": "integer"},
            },
            "required": ["tool", "sensor", "event_type", "severity", "zone", "vlm_verdict",
                         "hours", "max_count"],
            "additionalProperties": False,
        },
    }


ANSWER_SCHEMA = {
    "name": "grounded_answer",
    "schema": {
        "type": "object",
        "properties": {
            # Bounded for the same reason as `text` above. The answer prompt asks for two to four
            # sentences; 900 characters is well clear of that and still terminates the grammar.
            "answer": {"type": "string", "maxLength": 900},
            "cited_ids": {"type": "array", "items": {"type": "string", "maxLength": 40},
                          "maxItems": 12},
        },
        "required": ["answer", "cited_ids"],
        "additionalProperties": False,
    },
}


class LLM:
    def __init__(self, endpoint: str, model: str, timeout: float = 180.0):
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout

    def json_call(self, messages: list[dict], schema: dict, max_tokens: int = 800) -> dict:
        payload = {
            "model": self.model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.2,
            "response_format": {"type": "json_schema", "json_schema": schema},
            # Nemotron Nano v2 is a REASONING model: it emits thinking tokens before the answer.
            # Both switches below are sent, but MEASURED ON THIS BUILD NEITHER IS HONOURED — the
            # reasoning is still generated and still consumes the token budget. They are kept
            # because a later llama.cpp will respect them, and they cost nothing.
            #
            # The consequence is the real lesson: at max_tokens=300 the planner returned
            # `finish_reason: length` with EMPTY content, which looks exactly like llama.cpp bug
            # #20268 ("stops in Reasoning and returns no answer") but is simply the budget being
            # eaten before the JSON starts. The budget must cover reasoning + the answer.
            "chat_template_kwargs": {"thinking": False},
            "reasoning_effort": "none",
        }
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            choice = json.load(resp)["choices"][0]
        content = choice["message"]["content"]
        if choice.get("finish_reason") == "length":
            # Truncated mid-JSON. Try to salvage before giving up: a schema-constrained object
            # writes its properties in order, so a truncated one usually contains every field
            # that matters and is missing only a closing brace. Throwing it away meant the
            # caller fell back to a far worse plan, having already paid the full generation cost.
            salvaged = _salvage_json(content)
            if salvaged is not None:
                return salvaged
            # Distinct from a refusal and from a model failure: the budget was too small, and
            # saying so points at the fix instead of at the model.
            raise ValueError(
                f"completion truncated at max_tokens={max_tokens} and could not be salvaged — "
                f"raise it, or bound the free-text fields in the schema with maxLength "
                f"(an unbounded string under a grammar never terminates)")
        if not content.strip():
            # Distinguish "ran out of budget mid-reasoning" from "the model refused". The first
            # is a configuration error and must say so, not be reported as a model failure.
            raise ValueError(
                f"empty completion (finish_reason={choice.get('finish_reason')}); "
                f"raise max_tokens above {max_tokens} — reasoning tokens consume it first")
        return json.loads(_strip_reasoning(content))


def _salvage_json(text: str) -> dict | None:
    """Recover an object from a truncated JSON completion, or None if nothing usable is there.

    Walks back from the end, closing the structure at each plausible cut point. A grammar-
    constrained object emits its properties in schema order, so the fields that were already
    written are valid — only the tail is missing. Recovering them beats discarding a completion
    that has already been paid for.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    for cut in range(len(body), 0, -1):
        frag = body[:cut].rstrip().rstrip(",")
        if not frag:
            break
        for suffix in ("}", '"}', '"]}',  "]}"):
            try:
                out = json.loads(frag + suffix)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(out, dict) and out:
                return out
    return None


def _strip_reasoning(text: str) -> str:
    """Pull the JSON object out of a response that may be wrapped in prose or think tags.

    A reasoning model can prepend `<think>…</think>` or a sentence even under a JSON schema,
    depending on build. Rather than trusting the whole string to parse, take the outermost
    braces — which is exact for a schema-constrained object and salvages the rest.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("{"):
        return text
    i, j = text.find("{"), text.rfind("}")
    return text[i:j + 1] if i >= 0 and j > i else text


# `/no_think` is Nemotron Nano v2's OWN control for suppressing reasoning, and it is the switch
# that actually works. Measured on this build: the OpenAI-style `chat_template_kwargs.thinking`
# and `reasoning_effort` fields are silently ignored, while `/no_think` in the system prompt cuts
# completion tokens from 340-564 down to 91-94 — a 4-6x reduction — and removes the truncation
# that was corrupting the JSON. Put it FIRST in the system message.
NO_THINK = "/no_think\n"

PLAN_SYSTEM = (
    NO_THINK
    + "You route warehouse-safety questions to one retrieval tool. Choose the tool and its "
    "filters. Use 'any' for a filter the question does not constrain — do not guess. Put words "
    "that are not filters (objects, clothing, activities) into `text` for full-text search; "
    "OMIT `text` entirely when the question has no such words. Never pad or repeat it. "
    "`hours` is how far back to look; use 0 for no time limit.\n\n"
    "TOOLS\n"
    "  search_incidents               general search over incidents (default choice)\n"
    "  get_clips                      when the question asks for video, clips or footage\n"
    "  get_summary                    counts and statistics; how many, totals, breakdowns,\n"
    "                                 and per-camera counts. Its filters work like the search\n"
    "                                 filters, so use it for any counting question about a\n"
    "                                 subset too.\n"
    "  video_analytics__get_sensor_ids  which cameras exist and how active they are\n"
    "  video_analytics__get_incidents   recent incidents, optionally for one sensor\n"
    "\n"
    # A rule about the SHAPE of the answer, not about any particular question: retrieval returns a
    # ranked page, so it can never support a total. Stated as a property of the tools so it holds
    # for questions nobody anticipated.
    "CHOOSING\n"
    "  search_incidents and get_clips return a limited page of examples — they cannot tell you\n"
    "  how many exist in total. If answering needs a count, a total, or a per-camera or per-type\n"
    "  breakdown, choose get_summary.\n"
)


def plan(llm: LLM, question: str, vocab: dict, history: list[dict]) -> dict:
    """Ask the LLM which tool to call and with what arguments. Never trusted blindly."""
    ctx = ""
    if history:
        # Follow-ups ("what about camera 5?", "show me that clip") only make sense with the
        # previous turns in view.
        prior = "\n".join(f"Q: {h['q']}\nA: {h['a'][:200]}" for h in history[-3:])
        ctx = f"\nEarlier in this conversation:\n{prior}\n"
    msgs = [
        {"role": "system", "content": PLAN_SYSTEM
         + f"\nVALID VALUES\n  sensors: {', '.join(vocab['sensors']) or '(none)'}"
           f"\n  event_types: {', '.join(vocab['event_types']) or '(none)'}"
           f"\n  severities: {', '.join(vocab['severities']) or '(none)'}"
           f"\n  zones: {', '.join(vocab['zones']) or '(none)'}"},
        {"role": "user", "content": f"{ctx}\nQuestion: {question}"},
    ]
    # A plan is ~30-90 completion tokens with `/no_think`, so 300 is already generous. It used to
    # be 2000 on the reasoning "unused budget is free" argument — which is true only while
    # generation terminates. It did not: an unbounded `text` string in the schema ran to the
    # ceiling every single time, and the ceiling therefore set the latency. 2000 tokens at ~28
    # tok/s is 70+ seconds of pure waste per question.
    #
    # The real fix is the `maxLength` in plan_schema; this cap is the belt to its braces. If some
    # future prompt does run away again, it costs ~10 s rather than ~100 s, and `plan_error`
    # surfaces it instead of it hiding behind a fallback.
    return llm.json_call(msgs, plan_schema(vocab), max_tokens=300)


def execute(db, p: dict) -> tuple[str, dict]:
    """Run the planned tool. Validation happens here, not in the model."""
    tool = p.get("tool") if p.get("tool") in TOOLS else "search_incidents"
    clean = lambda k: None if p.get(k) in (None, "", "any") else p[k]  # noqa: E731
    args = {
        "sensor": clean("sensor"),
        "event_type": clean("event_type"),
        "severity": clean("severity"),
        "zone": clean("zone"),
        "vlm_verdict": clean("vlm_verdict"),
        "text": clean("text"),
        "hours": p.get("hours") or None,
        "max_count": max(1, min(int(p.get("max_count") or 10), 50)),
    }
    if tool == "get_summary":
        # The aggregate honours the same filters as retrieval, so "how many confirmed per camera"
        # is one counted query rather than a counted page of search results.
        return tool, TOOLS[tool](db, hours=args["hours"], vlm_verdict=args["vlm_verdict"],
                                 event_type=args["event_type"], zone=args["zone"])
    if tool == "video_analytics__get_sensor_ids":
        return tool, TOOLS[tool](db)
    if tool == "video_analytics__get_incidents":
        return tool, TOOLS[tool](db, source=args["sensor"], vlm_verdict=args["vlm_verdict"],
                                 max_count=args["max_count"], includes=["info"])
    return tool, TOOLS[tool](db, **args)


ANSWER_SYSTEM = (
    NO_THINK
    + "You are a warehouse safety assistant. Answer using ONLY the records provided. Never invent "
    "incidents, cameras, zones or times. If the records do not answer the question, say so "
    "plainly. Be concrete: name cameras and zones. Two to four sentences.\n"
    # A record's `type` is what the detector and the rules concluded; `reason` is one description
    # of a few frames sampled from the clip. They routinely disagree in a way that looks like a
    # contradiction but is not: a fire clip starts BEFORE the flames (pre-roll), so its
    # description can be "a worker walks through an aisle" while the record is a fire_alert.
    # Asked to show the fire clip, the model read the description and answered "the clip of the
    # fire incident is not available" while quoting that very clip.
    "A record's type and label are facts established by the system. `reason` is only a "
    "description of some frames from the clip and may not mention the event that triggered the "
    "incident — never conclude from `reason` that a record is not what its type says it is.\n"
    # Asked "what percentage of people were not wearing a helmet or vest", the honest answer is
    # that there is no denominator: the store records incidents, never a headcount of everyone
    # who walked through. Stating that and giving the counts beats inventing a rate.
    "These records count INCIDENTS, not people, and there is no record of how many people were "
    "observed in total. A percentage or rate of people therefore cannot be computed — say so, "
    "and give the incident counts instead. The `label` says which equipment was missing.\n"
    "cited_ids must be the id values of the records you actually used."
)


def synthesise(llm: LLM, question: str, tool: str, result: dict,
               history: list[dict], context: list[dict] | None = None) -> dict:
    # `get_summary` also has an "incidents" key, but it holds a COUNT, not a list. Take the
    # value only when it is actually a list of rows — otherwise an aggregate answer crashes on
    # `int + list` while trying to build the citation set.
    rows = next((v for v in (result.get("incidents"), result.get("clips"))
                 if isinstance(v, list)), [])
    context = context or []
    if not rows and not context and tool not in ("get_summary",
                                                 "video_analytics__get_sensor_ids"):
        return {"answer": "No incidents match that question.", "cited_ids": []}

    if tool in ("get_summary", "video_analytics__get_sensor_ids"):
        body = json.dumps(result, indent=1)[:2000]
    else:
        body = "\n".join(
            f"- id={r['id'][:8]} {r.get('sensor')} {r.get('type')} "
            f"severity={r.get('severity')} zone={r.get('zone')} "
            f"verdict={r.get('vlm_verdict')} at {r.get('timestamp', '')} "
            f"{'clip=yes' if r.get('has_clip') or r.get('clip') else 'clip=no'} "
            f"\"{r.get('description') or ''}\""
            + (f" reason=\"{r.get('reason') or r.get('vlm_reason')}\""
               if r.get("reason") or r.get("vlm_reason") else "")
            for r in rows[:14])

    msgs = [{"role": "system", "content": ANSWER_SYSTEM}]
    for h in history[-3:]:
        msgs.append({"role": "user", "content": h["q"]})
        msgs.append({"role": "assistant", "content": h["a"][:400]})
    user = f"QUESTION: {question}\n\nRECORDS (from {tool}):\n{body or '(none matched)'}"
    if context:
        ctx = "\n".join(
            f"- id={r['id'][:8]} {r.get('sensor')} {r.get('type')} zone={r.get('zone')} "
            f"verdict={r.get('vlm_verdict')} \"{r.get('description') or ''}\""
            + (f" reason=\"{r.get('vlm_reason')}\"" if r.get("vlm_reason") else "")
            for r in context[:8])
        user += ("\n\nNOTHING matched the question's filters. These are the nearby records that "
                 "DO exist — explain the absence using them; do not present them as matches:\n"
                 + ctx)
    # The backstop for when planning routes a counting question to retrieval. The listed rows are
    # a ranked page, so counting them yields a confident, wrong total. The exact counts for the
    # same filters are supplied instead — a rule about the shape of the data, which therefore
    # holds for questions nobody anticipated rather than for a list of expected ones.
    if result.get("counts"):
        c = result["counts"]
        user += ("\n\nEXACT COUNTS for these filters "
                 f"({json.dumps(c['filters']) if c['filters'] else 'no filters — ALL incidents'})"
                 f":\n  total={c['incidents']}  by_verdict={json.dumps(c['by_verdict'])}"
                 f"\n  by_camera={json.dumps(c['by_camera'])}"
                 f"\n  by_camera_verdict={json.dumps(c['by_camera_verdict'])}"
                 f"\n  by_type={json.dumps(c['by_type'])}"
                 f"\n  by_zone={json.dumps(c['by_zone'])}"
                 # The label is the only field that says WHICH equipment was missing.
                 f"\n  by_label={json.dumps(c.get('by_label') or {})}")
        if result.get("truncated"):
            user += (f"\nThe {result['count']} records listed above are only the top matches, "
                     f"not the whole set.")
        user += ("\nUse these counts for any number you state. Do NOT count the listed records. "
                 "If the question asks about a subset (a verdict, type or zone) that the filters "
                 "above did not restrict, take it from the matching breakdown rather than from "
                 "the total.")
    msgs.append({"role": "user", "content": user})

    try:
        # Bounded by ANSWER_SCHEMA (900 chars of answer + at most 12 ids) at roughly 360 tokens
        # worst case, so 600 is headroom rather than a limit. Kept low for the same reason as the
        # planner's: a ceiling only costs nothing while generation terminates, and the whole
        # 99-second planning bug was generation that did not.
        out = llm.json_call(msgs, ANSWER_SCHEMA, max_tokens=600)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError,
            json.JSONDecodeError) as e:
        # Retrieval was deterministic and already succeeded — a dead or confused LLM must not
        # throw the rows away.
        return {"answer": f"(LLM unavailable: {type(e).__name__}) "
                          f"{len(rows)} matching record(s) retrieved.",
                "cited_ids": [r["id"][:8] for r in rows[:5]], "llm_error": str(e)[:150]}

    allrows = rows + context
    valid = {r["id"][:8] for r in allrows} | {r["id"] for r in allrows}
    cited = [c for c in out.get("cited_ids", []) if c[:8] in {v[:8] for v in valid}]
    dropped = [c for c in out.get("cited_ids", []) if c[:8] not in {v[:8] for v in valid}]
    res = {"answer": out.get("answer", ""), "cited_ids": cited}
    if dropped:
        res["dropped_citations"] = dropped
    return res


def ask(db, question: str, llm: LLM, history: list[dict] | None = None) -> dict:
    history = history or []
    vocab = vocabulary(db)
    try:
        p = plan(llm, question, vocab, history)
        plan_error = None
    except Exception as e:  # noqa: BLE001
        # Planning is the only step allowed to fail soft: full-text search over the raw question
        # still returns something useful, which is better than an error page.
        p, plan_error = {"tool": "search_incidents", "text": question, "max_count": 10}, \
            f"{type(e).__name__}: {str(e)[:100]}"
    tool, result = execute(db, p)

    # A plan that matches nothing is usually OVER-constrained, not evidence of absence: the model
    # guessed a filter the question never asked for. Rather than reporting "no incidents" — which
    # is a confident wrong answer — drop the least-committed filters in order and retry.
    # Ordered by how likely each is to have been invented: severity and sensor are guessed far
    # more often than the zone or verdict the question actually named.
    relaxed = []
    for drop in ("severity", "sensor", "event_type", "hours"):
        if (result.get("count") or 0) > 0 or tool in ("get_summary",
                                                      "video_analytics__get_sensor_ids"):
            break
        if p.get(drop) in (None, "", "any", 0):
            continue
        p = dict(p, **{drop: "any"})
        relaxed.append(drop)
        tool, result = execute(db, p)

    # A genuinely empty result is still worth explaining. "No incidents match" is true but
    # unhelpful; "there are no CONFIRMED ones — the only incident in that zone was rejected as a
    # traffic cone" is the same fact made useful. So when nothing matched, fetch the neighbouring
    # rows under the question's strongest filter alone and let the answer explain the absence
    # against them. They are passed as CONTEXT, and the citation check still applies, so this
    # cannot manufacture a match that does not exist.
    context = []
    if not (result.get("count") or 0) and tool not in ("get_summary",
                                                       "video_analytics__get_sensor_ids"):
        for keep in ("zone", "sensor", "event_type"):
            if p.get(keep) in (None, "", "any"):
                continue
            _, near = execute(db, {"tool": "search_incidents", keep: p[keep], "max_count": 8})
            context = near.get("incidents", [])
            if context:
                break

    out = synthesise(llm, question, tool, result, history, context=context)
    if relaxed:
        out["relaxed_filters"] = relaxed
    if context:
        out["context_rows"] = len(context)
    out.update({"tool": tool,
                "plan": {k: v for k, v in p.items() if v not in (None, "", "any", 0)},
                "result": result})
    if plan_error:
        out["plan_error"] = plan_error
    return out


def render(res: dict) -> None:
    print(f"   tool: {res['tool']}   plan: "
          f"{ {k: v for k, v in res['plan'].items() if k != 'tool'} }")
    if res.get("plan_error"):
        print(f"   !! planning fell back to full-text: {res['plan_error']}")
    print(f"\n{res['answer']}\n")
    # Same trap as in synthesise(): get_summary's "incidents" is a count, not a list.
    rows = next((v for v in (res["result"].get("incidents"), res["result"].get("clips"))
                 if isinstance(v, list)), [])
    by = {r["id"][:8]: r for r in rows}
    for c in res.get("cited_ids", []):
        r = by.get(c[:8])
        if r:
            extra = (f" conf={r['confidence']}" if "confidence" in r else "")
            clip = r.get("clip") or ("clip" if r.get("has_clip") else "")
            print(f"  [{c[:8]}] {r.get('sensor')} {r.get('type')} {r.get('severity')} "
                  f"zone={r.get('zone')} {r.get('vlm_verdict')}{extra} {clip}")
    if res.get("dropped_citations"):
        print(f"  !! dropped invented citation(s): {res['dropped_citations']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*")
    ap.add_argument("--db", default=None)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8001/v1",
                    help="agent LLM (Nemotron). The VLM on :8000 is a different model.")
    ap.add_argument("--model", default="Nemotron-Nano-9B-v2")
    ap.add_argument("--chat", action="store_true", help="conversational mode with follow-ups")
    ap.add_argument("--tool")
    ap.add_argument("--args", default="{}")
    ap.add_argument("--list-tools", action="store_true")
    ap.add_argument("--vocab", action="store_true", help="print the live enum vocabulary")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.list_tools:
        for name, fn in TOOLS.items():
            print(f"{name}\n    {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0

    db = connect(Path(a.db) if a.db else ROOT / "data/events.db")
    db.row_factory = sqlite3.Row

    if a.vocab:
        print(json.dumps(vocabulary(db), indent=2))
        return 0
    if a.tool:
        if a.tool not in TOOLS:
            print(f"unknown tool {a.tool}; --list-tools to see them")
            return 1
        print(json.dumps(TOOLS[a.tool](db, **json.loads(a.args)), indent=2))
        return 0

    llm = LLM(a.endpoint, a.model)

    if a.chat:
        history: list[dict] = []
        print("Ask about incidents. Follow-ups keep context. Ctrl-D to exit.\n")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not q:
                continue
            res = ask(db, q, llm, history)
            render(res)
            history.append({"q": q, "a": res["answer"]})
            print()

    if not a.question:
        ap.print_help()
        return 1
    res = ask(db, " ".join(a.question), llm)
    if a.json:
        print(json.dumps({k: v for k, v in res.items() if k != "result"}, indent=2))
    else:
        print(f"Q: {' '.join(a.question)}")
        render(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
