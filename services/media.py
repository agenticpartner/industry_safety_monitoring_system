#!/usr/bin/env python3
"""Make an evidence clip playable by things that are not this device.

The demo media is H.265, because 20 concurrent 1080p30 streams do not fit the NVDEC budget in
H.264 on an AGX Orin. Clips are cut from it with `ffmpeg -c copy` — free, lossless, and it
inherits the codec.

**Almost nothing outside the device will play H.265.** Chrome needs a platform hardware decoder
and refuses without one, Firefox has no HEVC support at all, and Telegram will not preview it —
it arrives as a file attachment nobody can open inline, which defeats the point of attaching it.
So every consumer needs the same H.264 proxy, built once and shared.

This lives in its own module because both `api.py` (serving `/clips/{id}`) and
`notify_service.py` (attaching to a Telegram message) need it, and the notifier importing the
whole FastAPI application — which constructs the app, an LLM client and a metrics sampler — to
get at one function is the kind of coupling that eventually starts a second `tegrastats` or
holds a database handle nobody expected.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def browser_playable(path: Path) -> Path:
    """Return a path any client can decode, transcoding H.265 to H.264 on first request.

    Transcoding lazily rather than at capture time keeps the capture path free (its whole design
    point) and pays only for clips somebody actually opens. The H.265 original stays as the
    archival evidence; this is a viewing proxy, cached beside it under `h264/`.
    """
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=20)
        codec = (probe.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return path
    if codec not in ("hevc", "h265"):
        return path

    proxy = path.parent / "h264" / path.name
    # Rebuild if the source is newer, so a re-cut clip is not served from a stale proxy.
    if proxy.exists() and proxy.stat().st_mtime >= path.stat().st_mtime and proxy.stat().st_size:
        return proxy
    proxy.parent.mkdir(parents=True, exist_ok=True)

    # NVENC is not reachable from ffmpeg on this build (see scripts/splice_fire.sh), so this is
    # libx264 on the CPU. `ultrafast` + `faststart` on a ~12 s clip costs a couple of seconds,
    # once, on a cold path. Written to a temp name and renamed so a concurrent reader — the API
    # and the notifier genuinely do race for the same clip — can never observe a partial file.
    tmp = proxy.with_suffix(".part.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path),
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
             "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(tmp)],
            check=True, capture_output=True, timeout=180)
        tmp.replace(proxy)
    except (OSError, subprocess.SubprocessError):
        tmp.unlink(missing_ok=True)
        # Serving the H.265 beats serving nothing: Safari plays it, and the alternative is a 500.
        return path
    return proxy
