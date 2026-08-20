#!/usr/bin/env python3
"""
Render a labelled contact sheet of incident crops + verdicts, so a human can check whether the
VLM was actually RIGHT.

    python3 tools/verify_verdicts.py [--verdict rejected] [--limit 12] [--out build/sheet.jpg]

Everything else in this project measures whether the reasoning layer *runs*: latency, throughput
cost, schema validity, internal coherence. None of that establishes whether a verdict is
**correct**. "13 of 40 rejected" is only good news if those 13 people really are wearing vests —
and the one time a crop was inspected by eye it turned out to be empty floor while the model
described a worker in confident detail.

So this renders the exact image the model was shown, with the verdict and the model's own words
underneath, and lets a person adjudicate. It is deliberately manual: there is no ground truth in
this dataset, so a human looking at the crop IS the ground truth.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(ROOT / "data/events.db"))
    ap.add_argument("--verdict", default=None, help="filter: confirmed | rejected | uncertain")
    ap.add_argument("--type", default="ppe_violation")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "build/verdict_sheet.jpg"))
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    q = ("SELECT event_id, camera_id, type, severity, zone, label, bbox, source_pts_ns, "
         "       vlm_verdict, vlm_reason FROM events "
         " WHERE bbox IS NOT NULL AND source_pts_ns IS NOT NULL AND vlm_verdict IS NOT NULL")
    params: list = []
    if args.type:
        q += " AND type = ?"
        params.append(args.type)
    if args.verdict:
        q += " AND vlm_verdict = ?"
        params.append(args.verdict)
    rows = db.execute(q + " ORDER BY camera_id LIMIT ?", (*params, args.limit)).fetchall()
    if not rows:
        print("no matching incidents")
        return 1

    work = ROOT / "build/verify_crops"
    work.mkdir(parents=True, exist_ok=True)
    for f in work.glob("*.jpg"):
        f.unlink()

    CW, CH, PAD, TXT = 300, 380, 10, 74
    tiles: list[tuple[Image.Image, sqlite3.Row]] = []
    durations: dict[Path, float | None] = {}

    for r in rows:
        src = ROOT / "media" / f"cam{r['camera_id']:02d}.mp4"
        if src not in durations:
            durations[src] = probe_duration(src)
        dur = durations[src]
        if not dur:
            continue
        offset = (r["source_pts_ns"] / 1e9) % dur
        left, top, w, h = json.loads(r["bbox"])
        px, py = w * 0.35, h * 0.35
        x, y = max(0, int(left - px)), max(0, int(top - py))
        cw, ch = max(32, int(w + 2 * px)), max(32, int(h + 2 * py))
        out = work / f"{r['event_id'][:8]}.jpg"
        # Identical extraction to reasoning_service.FrameSet — this must show the model's input,
        # not a prettier approximation of it.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{offset:.3f}",
                        "-i", str(src), "-frames:v", "1",
                        "-vf", f"crop={cw}:{ch}:{x}:{y}:exact=0,scale=448:-2", str(out)],
                       capture_output=True, timeout=60)
        if out.exists() and out.stat().st_size:
            tiles.append((Image.open(out).convert("RGB"), r))

    if not tiles:
        print("no crops could be rendered")
        return 1

    cols = min(args.cols, len(tiles))
    rows_n = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (CW + PAD) + PAD, rows_n * (CH + TXT + PAD) + PAD),
                      (24, 26, 30))
    d = ImageDraw.Draw(sheet)

    for i, (img, r) in enumerate(tiles):
        cx = PAD + (i % cols) * (CW + PAD)
        cy = PAD + (i // cols) * (CH + TXT + PAD)
        img.thumbnail((CW, CH))
        sheet.paste(img, (cx + (CW - img.width) // 2, cy + (CH - img.height) // 2))
        colour = {"confirmed": (255, 90, 80), "rejected": (90, 210, 120)}.get(
            r["vlm_verdict"], (220, 200, 90))
        d.rectangle([cx - 2, cy - 2, cx + CW + 2, cy + CH + 2], outline=colour, width=3)
        head = f"cam{r['camera_id']:02d} {r['vlm_verdict'].upper()}  [{r['label'][:16]}]"
        d.text((cx, cy + CH + 6), head, fill=colour)
        reason = (r["vlm_reason"] or "")[:150]
        for j in range(0, len(reason), 44):
            d.text((cx, cy + CH + 22 + (j // 44) * 12), reason[j:j + 44], fill=(200, 200, 205))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=88)
    print(f"wrote {args.out}  ({len(tiles)} incidents)")
    for _, r in tiles:
        print(f"  cam{r['camera_id']:02d} {r['vlm_verdict']:9s} {r['label'][:18]:18s} "
              f"{(r['vlm_reason'] or '')[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
