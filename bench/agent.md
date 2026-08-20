# Phase 2.5 — search and the Q&A agent

**Two models, two jobs.** Cosmos Reason 2 2B on `:8000` verifies incidents (per incident,
continuously, must be cheap). Nemotron Nano 9B v2 on `:8001` plans retrieval and writes answers
(per human question, rarely, can be bigger). The duty-cycle difference is what makes the second
model affordable at all.

Measured 2026-08-17 on the reference AGX Orin 64GB.

---

## 1. The 2B cannot do this job — measured, not assumed

Cosmos Reason 2 2B was tried first, since it was already serving. Asked to plan
*"show me clips of confirmed violations in the spill zone"* it produced valid JSON that was
semantically wrong:

```
sensor=cam20  event_type=overcrowding  severity=high  zone=SpillZone  verdict=confirmed
```

Three of five filters invented — the question named no sensor, no event type and no severity. It
also returned invalid JSON when asked to write the answer. A schema constrains shape, never
judgement; this is the same finding as the verdicts in 2.4, in a different place.

Nemotron Nano 9B v2 on the same question:

```
tool=get_clips  zone=SpillZone  verdict=confirmed  text="violation spill zone confirmed clips"
```

Correct tool, correct filters, nothing invented.

---

## 2. `/no_think` — the single most important setting

Nemotron Nano v2 is a **reasoning** model: it emits thinking tokens before its answer, and those
tokens are counted against `max_tokens`. At `max_tokens=300` the planner returned
`finish_reason: length` with **empty content**, which looks exactly like llama.cpp bug #20268
("stops in Reasoning and returns no answer") but is nothing of the kind.

The OpenAI-style switches are **silently ignored** by this build:

| Approach | Completion tokens | Result |
|---|---|---|
| `chat_template_kwargs: {thinking: false}` | 340–564 | ignored |
| `reasoning_effort: "none"` | 340–564 | ignored |
| **`/no_think` at the start of the system prompt** | **91–94** | **works** |

A **4–6× token reduction**, and end-to-end query latency fell from **33.7 s to 7.2 s**.

`/no_think` is Nemotron's own control token, not an API parameter. Both API switches are still
sent — they cost nothing and a later llama.cpp may honour them — but `/no_think` is what does the
work.

The related lesson: `max_tokens` is a **ceiling covering reasoning plus answer**, so it is set
generously (2000) rather than tuned. Actual usage stays near 100. A truncated plan silently
degrades retrieval, so unused budget is much cheaper than too little — and both `finish_reason:
length` and empty content now raise an error naming the fix instead of failing soft.

---

## 3. No hardcoded questions; vocabulary comes from the data

An earlier draft matched keywords with regexes (`helmet|hard ?hat` → `ppe_violation`). That is a
hidden list of anticipated questions: it answers what was foreseen and silently mis-files
everything else. It is gone.

The LLM plans retrieval, and **the enums it may choose from are built from the database at
runtime** — the sensors, zones, event types and severities that exist right now. Add a camera or
rename a zone and the agent follows with no code change.

Rules remain, but only as guardrails that cannot invent anything:

| Guardrail | What it prevents |
|---|---|
| plan schema uses live enums | a zone or camera that does not exist is *unexpressible* |
| arguments validated and clamped (`max_count` 1–50) | a model asking for 10 000 rows |
| citations checked against retrieved rows | invented provenance — observed and dropped (`by_type`) |
| empty result ⇒ relax guessed filters, retry | a wrong guess reported as "nothing happened" |
| empty result ⇒ fetch neighbouring rows as CONTEXT | "no matches" where "no *confirmed* ones, but here is the rejected one" is the useful answer |
| planning failure ⇒ full-text search | an error page where a decent answer was available |

**An answer can be clumsily worded. It cannot cite an incident that does not exist.**

---

## 4. What it does

### Grounded answers, including about absence

> **Q:** show me clips of confirmed violations in the spill zone
> **A:** There are no clips of confirmed violations in the SpillZone. The only nearby record
> (id=d6527cbb) shows a rejected verdict for a PPE violation in the SpillZone, as the tracked
> object was a traffic cone, not a person.

Correctly answers a negative question *and* explains why, using the context rows.

### Analytics

> **Q:** how many incidents are there and what types?
> **A:** There are 30 incidents in total. The types are overcrowding (10 incidents) and
> ppe_violation (20 incidents).

Routed to `get_summary`; the guardrail dropped its attempt to cite `by_type` as an incident id.

### Follow-ups that carry context

| Turn | Plan | |
|---|---|---|
| which cameras had confirmed PPE violations? | `event_type=ppe_violation verdict=confirmed` | lists cameras |
| **what about camera 5?** | `sensor=cam05` **+ carried `event_type=ppe_violation`** | context retained |
| **show me the clip for that one** | **`get_clips`**, carried `sensor=cam05` **and the incident id** | describes the clip's actual contents |

### Confidence-ranked clip retrieval

Clips are returned ranked by a confidence score computed **in code** — `0.5·verdict +
0.35·severity + 0.15·corroborating tracks`, with the track count saturating so sighting volume
cannot dominate. It is a ranking signal, not a probability, and it is computed rather than asked
of the model for the same reason verdicts are: a model asked to score confidence produces a
fluent number with nothing behind it.

---

## 5. FTS5 rather than embeddings

The plan called for text embeddings (`sqlite-vec` or FAISS). SQLite **FTS5** is used instead:

* an incident is mostly **structured** — camera, zone, type, severity, verdict, time. "When did
  someone last enter zone 3 without a helmet" is a WHERE clause, answered exactly rather than
  approximately;
* its free text is one sentence of concrete nouns ("traffic cone", "hi-vis vest"), which lexical
  matching handles well;
* an embedding model is **another always-on GPU tenant**, competing with 20 camera streams.

The honest limit: FTS5 has no synonymy — "near a vehicle" will not match "forklift". If that
matters, add `sqlite-vec` over `vlm_reason` *alongside* this, not instead of it. VSS uses Cosmos
Embed1 for genuine **video** search, which is a much larger capability than what this replaces.

---

## 6. VSS compatibility

`video_analytics__get_incidents`, `video_analytics__get_incident` and
`video_analytics__get_sensor_ids` match the VSS MCP server's names and parameters exactly,
including `vlm_verdict ∈ {confirmed, rejected, unverified}` where `unverified` is our
`vlm_verdict IS NULL`. A dashboard or agent written against these keeps working against a real
VSS backend on Thor. Tools with no VSS counterpart (`search_incidents`, `get_clips`,
`get_summary`) are namespaced separately so the boundary stays visible.

---

## 7. Carried forward

- **Planning quality is not yet measured across a question set.** Every example here was checked
  by hand. A fixed set of questions with expected filters would turn "it looked right" into a
  number, and that has not been done.
- **Two models are resident** (~28 GB free of 61 GB with both loaded). If memory gets tight, the
  agent LLM is the one to unload — it is only needed while someone is asking questions, whereas
  the VLM is needed continuously.
- **`/no_think` is load-bearing.** Without it, planning silently truncates. It is in
  `PLAN_SYSTEM` and `ANSWER_SYSTEM`; if either is edited, keep it first.

---

## 8. Counting questions: the LLM must never do the arithmetic (Phase 2.7 follow-up)

Found by asking the dashboard's agent panel a question no test covered:
**"Which cameras had confirmed violations, and how many each?"**

Ground truth is 13 confirmed across 9 cameras
(cam01·1 cam03·1 cam04·1 cam05·2 cam08·1 cam09·2 cam10·1 cam11·3 cam12·1).

The answer was:

> "Cameras cam09 and cam12 had confirmed overcrowding violations (2 each), and cam10 had a
> confirmed PPE violation. Cameras cam03 and cam11 also had confirmed violations (1 each for
> cam03 and 2 for cam11)."

Five cameras named out of nine, two of those five counts wrong. Fluent, specific, and wrong —
the failure mode that automated checks never catch, because every field is well-formed.

### Why

1. The planner chose `search_incidents`, which returns a **ranked page** (`max_count=10`).
2. The model counted the rows it could see. Ten rows, drawn from 23, ordered by relevance —
   so the "counts" were counts of a sample, and the sample included `rejected` rows.
3. `get_summary` — the aggregate tool that should have been chosen — **had no per-camera
   breakdown at all**, only `cameras_affected`. Even the right tool could not have answered it.

The generalisation: **a retrieval tool can never support a total.** This is a property of the
tool's shape, not of any particular question, so the fix belongs in the plumbing rather than in a
list of question phrasings to detect. (Keyword-matching "how many" would be exactly the hidden
list of anticipated questions that §3 removed.)

### Fixes

- `summarise()` gained `by_camera`, `by_camera_type`, `by_camera_verdict`, `by_zone`, and the
  full filter set (`camera_id`, `severity`, `open_only` on top of type/zone/verdict/hours). It
  shares its WHERE clause with `search()` via `_filters()`, so a summary and a retrieval asked
  the same question describe the same set.
- `summarise()` returns the `filters` it applied. A bare total is ambiguous: **23** (all
  incidents) was read as "23 confirmed" in one intermediate version of this fix — a right number
  attached to the wrong predicate.
- **Every retrieval now carries the exact aggregate for its own filters** (`counts`, plus
  `total_matching` / `truncated`). The model no longer has to count anything; the numbers are in
  the payload, computed in SQL.
- `/agent/chat` returns `counts`, and the dashboard renders it as a per-camera × verdict table
  under the prose.

### Result

Numbers are now exact. The prose still under-enumerates — the model narrates three or four
cameras out of nine even with all nine in front of it — so **the table, not the sentence, is the
answer** to an aggregate question. That is the same split as Phase 2.4, where the VLM answers
perception questions and the verdict is computed in code: the model writes the narrative, the
database produces the numbers.

~~Still open: the planner picks `search_incidents` for these questions even with an explicit
"choose get_summary for counts" rule in `PLAN_SYSTEM`.~~ **Wrong diagnosis — see §10.** The
planner was not choosing anything. Every plan call was failing and `ask()` was silently falling
back to a full-text search on the raw question. With the real bug fixed, the planner picks
`get_summary` for exactly these questions.

## 9. Agent latency, measured

Same question, same 23-incident database, Nemotron Nano 9B Q5_K_M on `:8001`:

| Pipeline state | End-to-end |
|---|---|
| stopped | **104-109 s** |
| 12 streams, GPU ~95%, NVDEC 99% | **218 s** |

Two LLM round trips (plan + synthesis). Contention with the hot path **doubles** it, which is
consistent with §5's "two resident models cost nothing when idle — only inference competes".

Even uncontended, ~105 s is too slow to feel interactive in a demo. Not addressed here; the
options are a smaller planner model, collapsing plan+synthesis into one call, or streaming the
answer so the wait is visible rather than blank. Recorded so the number is known rather than
discovered in front of an audience.

---

## 10. The 105-second agent was one missing `maxLength`

`bench/alert_latency.md` measures the alert path in milliseconds; the agent answered in 105 s
idle and 218 s under load. The assumption in §9 was that a 9B model on Orin is simply slow. It is
not.

### Measuring instead of assuming

Raw model speed, pipeline stopped:

```
prompt_tok  prefill_ms  prefill_tok/s  gen_tok  gen_tok/s
        29         297           97.5        3       27.5
       323         738          438.0        3       28.2
      1223        2109          579.9        3       28.1
      2423        2821          858.9        3       28.2
```

28 tok/s generation, 580-860 tok/s prefill. That predicts a **~12 s** agent call, not 105 s. So
the model was not the problem, and the next step was to time the phases rather than theorise:

```
vocabulary()         0.00s
plan() FAILED       99.33s  ValueError: completion truncated at max_tokens=2000
sync() FTS           0.00s
execute()            0.00s
```

**99 of the 105 seconds were a single wasted generation**, and planning was failing on every
question. Timing the LLM calls confirmed it: total 107.6 s, of which the LLM accounted for 8.7 s
across *one* call — the plan call never appeared, because it threw.

### The cause

`plan_schema`'s `text` property was `{"type": "string"}` — unbounded. Under grammar-constrained
decoding an unbounded string has no reason to terminate. The proof is one A/B:

| | latency | finish_reason | tokens |
|---|---|---|---|
| with the JSON schema | 22.4 s | `length` | 400 (the cap) |
| same prompt, no schema | **1.5 s** | `stop` | 29 |

The model emitted a *correct* plan (`get_summary`, `vlm_verdict: confirmed`) and then kept
writing the `text` value until it hit the ceiling. `max_tokens=2000` — set generously on the
"unused budget is free" reasoning in §1 — meant the ceiling was 2000 tokens at 28 tok/s.

That argument was right about reasoning tokens and wrong here: **unused budget is free only while
generation terminates.** When it does not, `max_tokens` stops being a ceiling and becomes the
latency.

### Why it hid for a whole phase

`ask()` catches planning failures and falls back to a full-text search on the raw question,
because a degraded answer beats an error page. That is good behaviour and it is exactly what
concealed this: the agent always answered, the answers were plausible, and the only symptom was
latency. §8's "the planner picks search_incidents for counting questions" was this bug — the plan
in every logged result (`{'tool': 'search_incidents', 'text': <the whole question>}`) is the
fallback's literal shape, which should have been the tell.

### Fixes

- `maxLength` on every free-text field in both schemas, and `additionalProperties: false` to
  close the other runaway route (inventing new keys after the required ones).
- `text` made **optional**. Bounded but mandatory, the model padded it to the bound with
  repetition — harmless for an aggregate, but it poisons FTS relevance on a search.
- Plan `max_tokens` 2000 → 300, answer 2000 → 600. Not the fix; the blast radius if it recurs.
- Truncated completions are now **salvaged** (`_salvage_json`) rather than discarded. A
  grammar-constrained object writes properties in order, so a truncated one usually has every
  field that matters and is missing only a closing brace.
- `tests/test_agent.py` asserts every non-enum string in both schemas is bounded, both are
  closed, and `text` is optional. The bug was invisible at runtime, so the guard is static.

### Result

| Condition | Before | After | |
|---|---|---|---|
| pipeline stopped | 105 s | **11.2 s** mean (9.5-13.7 over 4 question shapes) | 9.4x |
| 20 streams running | 218 s | **23.8 s** mean (19.3-32.2) | 9.2x |

Planning now succeeds on every question tried, and picks the right tool: `get_summary` for
counting questions, `get_clips` for clip requests, `search_incidents` for a single-camera lookup.

Contention still costs ~2x, consistent with §5. Remaining options if it needs to be faster:
collapse plan+synthesis into one call, or stream the answer so the wait is visible rather than
blank. Neither is needed at 24 s.
