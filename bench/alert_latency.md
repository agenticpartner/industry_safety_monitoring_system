# Alert latency at 20 streams

Measured with `scripts/measure_alert_latency.py`, run **on the Jetson** (comparing a Jetson-clock
`event.ts` against a Mac wall clock would fold in unbounded silent skew).

Conditions: 20 streams, file sources, `--zones --events`, RTSP output on, reasoning and clip
services running, database wiped immediately before the run. 31 incidents.

| Stage | n | min | p50 | p90 | max |
|---|---|---|---|---|---|
| emit → visible (queryable via API) | 31 | 0.03 s | **0.27 s** | 0.56 s | 0.59 s |
| emit → clip ready | 30 | 0.52 s | **2.11 s** | 3.36 s | 3.61 s |
| emit → VLM verdict | 3 | 7.26 s | 12.49 s | 17.47 s | 17.47 s |

The dashboard adds its own delivery on top: a WebSocket push (immediate) with an 8 s feed poll as
the backstop. So an operator sees an alert **well inside a second**.

## The VLM verdict number is a queue, not a latency

Only 3 of 31 incidents were adjudicated inside the 5.5 minute window; 28 were still unverified at
the end. That is not a stall — `reasoning_service.log` shows steady progress at **5-10 s per
incident, serially**. The arrival pattern is what makes it look bad: starting a 20-stream pipeline
against a clean database produces ~31 incidents in the first ~40 s, and a serial adjudicator
working at ~6.5 s each needs ~3.5 minutes to drain that burst. The last incident in the queue
waits minutes even though each one costs seconds.

This matters for how the demo is narrated: **the alert is instant, the adjudication is not.** An
incident appears on the dashboard in under a second and is marked `unverified`; its verdict lands
later. Claiming a sub-second verified alert would be false.

Worth noting the burst is an artifact of starting cold — in steady state incidents arrive as
transitions (2225 observations became 6 incidents in the Phase 2.1 measurement), so the
adjudicator keeps up easily. The queue only forms at startup or during an unusual surge.

## Method note: prime before measuring

The first version of this script reported a suspiciously uniform 43.4 s for eighteen incidents at
once. That was not latency — it was each incident's AGE at the script's first poll, because the
first poll sees the whole existing table and treats every row as newly arrived. The script now
records everything present at startup as backlog and measures only what appears while it watches.

Same trap as the dashboard's `primed` flag, which exists to stop history from toasting on first
load. Any "time to first observation" measurement over a pre-existing table has it.
