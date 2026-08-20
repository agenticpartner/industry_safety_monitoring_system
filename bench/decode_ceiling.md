# Phase 1 — NVDEC decode ceiling (measured)

**Verdict: decode is not the bottleneck. 20 streams passes with 1.83× headroom.**

Device: Jetson AGX Orin 64GB, JetPack 7.2.1 / L4T R39.2.1, MAXN, clocks locked.
Media: 1920×1080, 30 fps, H.265 (HEVC), 1443 frames/clip.
Method: `scripts/decode_sweep.sh` — N × `filesrc ! qtdemux ! h265parse ! nvv4l2decoder ! fakesink`
with `sync=false`, i.e. flat out. No inference, no tracker, no OSD, no display.

| Streams | Elapsed | Aggregate fps | Per-stream fps | Realtime target | Holds? | Headroom | Peak RAM |
|--------:|--------:|--------------:|---------------:|----------------:|:------:|---------:|---------:|
| 1  |  1.4 s | 1054.4 | 1054.4 |  30 | YES | 35.14× | 2456 MB |
| 2  |  2.7 s | 1077.1 |  538.5 |  60 | YES | 17.95× | 2500 MB |
| 4  |  5.3 s | 1092.8 |  273.2 | 120 | YES |  9.10× | 2601 MB |
| 8  | 10.5 s | 1099.1 |  137.3 | 240 | YES |  4.57× | 2805 MB |
| 12 | 15.7 s | 1101.6 |   91.8 | 360 | YES |  3.06× | 3053 MB |
| 16 | 21.0 s | 1100.7 |   68.7 | 480 | YES |  2.29× | 3287 MB |
| 18 | 23.6 s | 1101.7 |   61.2 | 540 | YES |  2.04× | 3435 MB |
| 20 | 26.2 s | 1102.0 |   55.1 | 600 | YES |  1.83× | 3582 MB |

## Reading of the numbers

**Aggregate throughput is flat at ~1100 fps from N=4 upward.** That flatness is the signature of a
genuine hardware ceiling: NVDEC saturates and then divides its fixed capacity among however many
streams are asked of it. It is not a per-stream limit, and it is not CPU contention (GR3D stayed at
3%, CPU was never the constraint).

**Implied ceiling: ~1102 / 30 ≈ 36 concurrent 1080p30 H.265 streams.**

This is materially better than the 22× 1080p30 H.265 figure in the AGX Orin datasheet. The
datasheet number is a conservative product-spec figure; measured NVDEC throughput on this JetPack
7.2 / L4T R39.2 build is higher. **The plan's "20 streams is 91% of the ceiling, essentially no
margin" risk is therefore resolved — the real margin is ~1.8×.**

Caveats on that claim, stated plainly:
- Measured with `sync=false` on file sources. Live RTSP adds jitter buffering and network
  handling, which costs CPU rather than NVDEC, but has not been measured yet.
- All 20 clips were the same asset (`--fast` mode). NVDEC load depends on codec, resolution and
  frame rate rather than content, so this does not bias the ceiling — but bitrate does vary with
  content, and a high-motion 8 Mbps stream is more work than this 4 Mbps sample.
- `NVDEC0` in tegrastats reports a clock (99 MHz), not a utilisation percentage, so it confirms the
  engine was active but is not itself the evidence of saturation. The flat aggregate curve is.

## Consequence for the build

Decode is comfortably solved, so **the 20-stream target now rests entirely on inference
throughput** (600 frames/s arriving at the muxer). The accelerator work — DLA qualification and the
parallel-branch topology — is where the remaining risk lives, and Phase 2 goes there next.

Peak RAM at 20 streams was 3.6 GB of 62.9 GB. Memory is a non-issue at this scale.

Raw data: `bench/decode_sweep.csv`, `bench/tegrastats_decode.log`.
