# Benchmarks

Measured results. Every performance number quoted in the dashboard, the `/system` page,
`README.md` and `project_skill.md` comes from a file in here, and every file in here was produced
by a script in `scripts/`.

## Reports

| File | What it measures | Script |
|---|---|---|
| `findings.md` | End-to-end capacity 1→20 streams, component ceilings, what actually cost throughput | `sweep.sh`, `decode_sweep.sh`, `diagnose_bottleneck.sh` |
| `decode_ceiling.md` | The NVDEC decode ceiling in isolation | `decode_sweep.sh` |
| `vlm_feasibility.md` | Whether local reasoning fits at all — llama.cpp vs vLLM, and what each costs the hot path | `bench_reasoning.sh` |
| `reasoning.md` | VLM verdict quality and the prompt design that produced it | `bench_vlm_compare.sh`, `compare_verdicts.py` |
| `agent.md` | Agent latency, planning failures and retrieval grounding | — |
| `clip_capture.md` | Evidence clip timing, PTS handling and cost | — |
| `alert_latency.md` | Detection → alert → clip → verdict, end to end | `measure_alert_latency.py` |

## Raw data

`*.csv` are the sweep outputs the reports are built from. `demo_20cam.jpg` is the 20-camera tiled
output.

Per-run detritus — `sweep_*/`, `engines/`, `dla/` and `*.log` — is gitignored. It is regenerated
by re-running the script and is large.

## Two rules these benchmarks follow

**A throughput number is never accepted without proof the pipeline detected something.**
`sweep.sh` records a zero-detection run as `NO-DETECTIONS`, never as a pass. This is not
paranoia: a misconfigured tracker once produced the *best* result in the whole sweep by emitting
no objects at all and therefore doing no work.

**A setting is never tuned against footage that lacks the thing it detects.** The fire inference
interval was tuned to 11 on warehouse footage containing no fire, where skipping frames looks
free. Re-measured with real fire spliced in (`scripts/splice_fire.sh`), a ten-second fire passed
through completely undetected at that setting while the benchmark read 510 fps and looked healthy.

Both stories are written up in `project_skill.md` §5 and §6.
