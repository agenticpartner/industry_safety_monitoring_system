#!/usr/bin/env python3
"""
Measure a local VLM/LLM endpoint the way the Phase 2 cold path will actually use it.

    python3 scripts/bench_reasoning.py --endpoint http://127.0.0.1:8000/v1 \
            --model nvidia/Cosmos-Reason2-2B --frames 6 --repeat 5

Runtime-agnostic on purpose: vLLM and llama.cpp's server both speak OpenAI
`/v1/chat/completions`, so this harness does not care which one survives the Phase 2.0 gate.
That is the point — the gate is allowed to change the serving stack without invalidating the
measurement.

What it reports and why each number is the one that matters:

  ttft      time to first token. The dashboard streams the VLM's explanation, so this is what an
            operator perceives as "the reasoning appeared", not the total.
  total     wall time for the whole response. This is what bounds the reasoning service's
            per-request timeout and therefore its queue depth.
  tok/s     decode rate. Only useful for predicting `total` at a different answer length.
  mem       system RAM delta across the run. Jetson memory is UNIFIED — there is no separate GPU
            pool to query, and `nvidia-smi` reports [N/A] on Tegra. `free` is the honest proxy for
            "how much of the 61 GB the model is holding", and it is the number that decides
            whether a VLM and a 20-stream pipeline can coexist.

Deliberately NOT measured here: the throughput hit on a running pipeline. That needs the pipeline
running concurrently and is orchestrated by scripts/bench_reasoning.sh, which drives this script.

Frames are sent as base64 data URLs in the OpenAI `image_url` content form, which is what both
servers accept. Reasoning in Phase 2.4 is event-triggered over a handful of frames pulled from an
already-captured clip, so `--frames 4..8` is the realistic request shape — not one frame, and not
a whole video.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def mem_used_mb() -> int:
    """System memory in use, MB.

    Tegra has no discrete VRAM: `nvidia-smi --query-gpu=memory.used` returns [N/A] on this board
    (verified). Model weights land in the same pool as everything else, so this is the only
    meaningful "how much did the model cost" figure available.
    """
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            avail_kb = int(line.split()[1])
        elif line.startswith("MemTotal:"):
            total_kb = int(line.split()[1])
    return (total_kb - avail_kb) // 1024


def extract_frames(clip: Path, count: int, out_dir: Path, stride: float = 1.0) -> list[Path]:
    """Pull `count` JPEGs out of a clip, `stride` seconds apart.

    Sampled a second apart rather than consecutively because that is what the reasoning service
    will do: consecutive frames at 30 fps are near-identical and waste context on a model that
    only gets a handful of images.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if len(existing) >= count:
        return existing[:count]

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip),
         "-vf", f"fps=1/{stride},scale=640:-2", "-frames:v", str(count),
         str(out_dir / "frame_%02d.jpg")],
        check=True,
    )
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if len(frames) < count:
        raise SystemExit(f"ffmpeg produced {len(frames)} frames, wanted {count}")
    return frames[:count]


def data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_request(model: str, frames: list[Path], prompt: str, max_tokens: int) -> dict:
    content: list[dict] = [{"type": "image_url", "image_url": {"url": data_url(f)}}
                           for f in frames]
    content.append({"type": "text", "text": prompt})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        # Near-greedy. A safety verdict must be reproducible run to run — if the same frames can
        # yield "violation" and "not a violation" on consecutive calls, the verification layer is
        # worse than useless because it launders randomness as a judgement.
        "temperature": 0.1,
        "stream": True,
    }


def one_request(endpoint: str, payload: dict, timeout: float) -> tuple[float, float, str, int]:
    """Fire one streaming request. Returns (ttft, total, text, chunks)."""
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    t0 = time.monotonic()
    ttft = None
    pieces: list[str] = []
    chunks = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                delta = json.loads(body)["choices"][0].get("delta", {})
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            piece = delta.get("content") or ""
            if not piece:
                continue
            # First token that carries actual text — role-only preamble chunks do not count as
            # "the answer started", and counting them flatters ttft by a chunk's worth of latency.
            if ttft is None:
                ttft = time.monotonic() - t0
            pieces.append(piece)
            chunks += 1

    total = time.monotonic() - t0
    return (ttft if ttft is not None else total), total, "".join(pieces), chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--clip", default=str(ROOT / "media/cam01.mp4"))
    ap.add_argument("--frame-dir", default=str(ROOT / "build/bench_frames"))
    ap.add_argument("--frames", type=int, default=6,
                    help="images per request (4-8 is the realistic event-triggered shape)")
    ap.add_argument("--repeat", type=int, default=5,
                    help="requests to time; the median is reported, not the mean")
    ap.add_argument("--warmup", type=int, default=1,
                    help="untimed requests first — the first call pays graph capture and cache "
                         "allocation and is not representative of steady state")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--prompt", default=(
        "You are reviewing frames from a fixed industrial-safety camera in a warehouse. "
        "A computer-vision detector flagged a person as not wearing a high-visibility vest. "
        "Look at the frames and answer in this exact form:\n"
        "VERDICT: confirmed | rejected | uncertain\n"
        "REASON: one sentence citing what you actually see."))
    ap.add_argument("--json", metavar="PATH", help="also write raw results as JSON")
    args = ap.parse_args()

    frames = extract_frames(Path(args.clip), args.frames, Path(args.frame_dir))
    payload = build_request(args.model, frames, args.prompt, args.max_tokens)
    kb = len(json.dumps(payload)) // 1024
    print(f"==> {args.model} @ {args.endpoint} | {len(frames)} frames | request {kb} KB",
          flush=True)

    mem_before = mem_used_mb()

    for i in range(args.warmup):
        try:
            _, total, text, _ = one_request(args.endpoint, payload, args.timeout)
        except urllib.error.URLError as e:
            print(f"[FAIL] endpoint unreachable or refused: {e}", flush=True)
            return 2
        print(f"[warmup {i + 1}] {total:.2f}s", flush=True)

    mem_loaded = mem_used_mb()
    rows = []
    for i in range(args.repeat):
        ttft, total, text, chunks = one_request(args.endpoint, payload, args.timeout)
        rows.append({"ttft_s": ttft, "total_s": total, "chunks": chunks, "text": text})
        print(f"[{i + 1}/{args.repeat}] ttft {ttft:.2f}s  total {total:.2f}s  "
              f"{chunks} tok  {chunks / max(total, 1e-9):.1f} tok/s", flush=True)

    mem_after = mem_used_mb()
    ttfts = sorted(r["ttft_s"] for r in rows)
    totals = sorted(r["total_s"] for r in rows)

    print("\n--- median over %d requests ---" % args.repeat, flush=True)
    print(f"ttft         {statistics.median(ttfts):.2f}s   (min {ttfts[0]:.2f} max {ttfts[-1]:.2f})",
          flush=True)
    print(f"total        {statistics.median(totals):.2f}s   (min {totals[0]:.2f} max {totals[-1]:.2f})",
          flush=True)
    print(f"tok/s        {statistics.median([r['chunks'] / r['total_s'] for r in rows]):.1f}",
          flush=True)
    # NOTE these are deltas ACROSS THE RUN, not the model's footprint. If the server was already
    # serving when this started — the normal case, and what bench_reasoning.sh does — the weights
    # are already resident at the first sample and the delta is just KV cache and activations.
    # It can even go negative as caches are reclaimed. To size the model itself, compare RAM with
    # the server stopped against RAM with it loaded (measured separately: ~3.7-4.3 GB for
    # Cosmos-Reason2-2B BF16 + mmproj).
    print(f"mem  start {mem_before} MB -> after warmup {mem_loaded} MB -> end {mem_after} MB "
          f"(run delta {mem_after - mem_before:+d} MB; NOT the model footprint)", flush=True)

    # Proof it answered, not just that it responded fast. A server that returns empty strings in
    # 40 ms would otherwise look like the best result in the table — the same trap as the
    # zero-detection tracker in Phase 1.
    print("\n--- last answer ---\n" + (rows[-1]["text"].strip() or "<EMPTY — NOT A PASS>"),
          flush=True)
    if not rows[-1]["text"].strip():
        print("\n[FAIL] endpoint streamed no text. Fast and useless is not a pass.", flush=True)
        return 3

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"model": args.model, "endpoint": args.endpoint, "frames": len(frames),
             "mem_idle_mb": mem_before, "mem_loaded_mb": mem_loaded, "results": rows},
            indent=2))
        print(f"\nwrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
