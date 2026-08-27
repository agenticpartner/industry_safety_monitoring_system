# Tests

Unit tests only. Nothing here talks to a network, a model server, a GPU or a real Redis — a test
that needed any of those would be a benchmark, and those live in `scripts/` and write their
results to `bench/`.

```bash
./tests/run_all.sh       # 175 tests across 7 files
./tests/run_all.sh -v    # verbose, per-test names
```

Each file also runs standalone:

```bash
python3 tests/test_rules.py
./build/venv-services/bin/python3 tests/test_store.py
```

The runner prefers `build/venv-services` when it exists, because that is the interpreter the
services actually run under, and falls back to system `python3` otherwise — saying which it used.

## What each file covers

| Test | Covers |
|---|---|
| `test_rules.py` | the compliance state machine — debouncing, vest containment, fire latching |
| `test_events.py` | the event vocabulary and transition detection |
| `test_store.py` | the incident state machine: reference counting, merging, re-raising |
| `test_agent.py` | the agent's deterministic half — vocabulary, tools, schema bounds, confidence |
| `test_reasoning.py` | verdict computation and hazard escalation, with the VLM stubbed |
| `test_clip_service.py` | clip windowing, PTS arithmetic, retention |
| `test_notify.py` | the notification policy: what is worth interrupting someone for |

The suite needs no third-party packages: the services import PyYAML lazily, inside the functions
that read a config file, and no test reaches those paths. `app/rules.py` has no DeepStream imports
at all, deliberately — the code that decides whether a worker is compliant is the part most likely
to be subtly wrong, and checking it should not require a GPU and a video feed.

## Two tests worth knowing about

**`test_store.py`** exists because of a real failure: merging N tracks into one incident and then
closing that incident on the *first* track to clear produced 531 unmatched closes against 306 real
ones, turning a handful of ongoing situations into 310 short-lived rows. The invariants it asserts
are the ones a `SELECT COUNT(*)` cannot check.

**`test_agent.py`** asserts statically that every free-text field in both JSON schemas has a
`maxLength`. An unbounded string under grammar-constrained decoding never terminates: the planner
wrote a correct plan and then padded one field to `max_tokens` on every question. That is
invisible at runtime — it looks like a slow model — so it is checked in a test instead.

## Integrity check

`tools/inspect_db.py --check-only` validates a live incident database against the same invariants
and exits non-zero on violation, so it can gate a run the way these tests gate a change.
