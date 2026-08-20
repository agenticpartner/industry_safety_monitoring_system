#!/usr/bin/env python3
"""
Score verdict STABILITY across repeated runs, and agreement between models.

    python3 scripts/compare_verdicts.py --dir bench/vlm_compare --runs 3 --models "2b 8b"

Stability is the metric that can be computed without labels: run the same incidents N times and
see whether the answers hold. It is also the defect that motivated the comparison — Phase 2.4
found the same red tabard called "an apron" (confirmed) and then "a hi-vis vest" (rejected),
opposite verdicts on identical input.

**This deliberately does NOT score correctness.** A model can be perfectly stable and perfectly
wrong; the two failure modes are independent, and conflating them is how "the pipeline runs" gets
mistaken for "the answers are right". Correctness is adjudicated by eye from the contact sheets.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

LINE = re.compile(
    r"\[reasoning\] (\w{8}) cam(\d+) (\S+)\s+\S+\s+\S+\s+-> (\w+)\s+people=(\S+)")


def parse(path: Path) -> dict[str, tuple[str, str, str]]:
    """{event_id: (type, verdict, people)}"""
    out: dict[str, tuple[str, str, str]] = {}
    if not path.exists():
        return out
    for ln in path.read_text(errors="replace").splitlines():
        m = LINE.match(ln)
        if m:
            out[m.group(1)] = (m.group(3), m.group(4).lower(), m.group(5))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="bench/vlm_compare")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--models", default="2b 8b")
    args = ap.parse_args()

    d = Path(args.dir)
    models = args.models.split()
    per_model: dict[str, list[dict]] = {
        m: [parse(d / f"{m}_run{i}.log") for i in range(1, args.runs + 1)] for m in models
    }

    print("=== verdict stability (same incidents, same model, %d runs) ===" % args.runs)
    stable_sets: dict[str, dict[str, str]] = {}
    for m in models:
        runs = [r for r in per_model[m] if r]
        if not runs:
            print(f"  {m}: no runs found")
            continue
        ids = set(runs[0])
        for r in runs[1:]:
            ids &= set(r)
        if not ids:
            print(f"  {m}: no incidents common to all runs")
            continue

        flips: list[tuple[str, list[str]]] = []
        consensus: dict[str, str] = {}
        for eid in sorted(ids):
            verdicts = [r[eid][1] for r in runs]
            consensus[eid] = Counter(verdicts).most_common(1)[0][0]
            if len(set(verdicts)) > 1:
                flips.append((eid, verdicts))
        stable_sets[m] = consensus
        n = len(ids)
        print(f"  {m}: {n - len(flips)}/{n} incidents gave the SAME verdict every run "
              f"({100.0 * (n - len(flips)) / n:.0f}% stable)")
        for eid, vs in flips[:8]:
            # People counts drifting is expected; a VERDICT flipping is the problem.
            print(f"      FLIP {eid}: {' -> '.join(vs)}")

    if len(models) >= 2 and all(m in stable_sets for m in models[:2]):
        a, b = models[0], models[1]
        common = set(stable_sets[a]) & set(stable_sets[b])
        if common:
            agree = sum(stable_sets[a][e] == stable_sets[b][e] for e in common)
            print(f"\n=== cross-model agreement (consensus verdicts) ===")
            print(f"  {a} vs {b}: {agree}/{len(common)} agree "
                  f"({100.0 * agree / len(common):.0f}%)")
            for e in sorted(common):
                if stable_sets[a][e] != stable_sets[b][e]:
                    print(f"      DIFFER {e}: {a}={stable_sets[a][e]}  {b}={stable_sets[b][e]}")

    print("\n=== verdict mix (consensus) ===")
    for m in models:
        if m not in stable_sets:
            continue
        c = Counter(stable_sets[m].values())
        print(f"  {m}: " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))

    print("\n=== resources ===")
    for m in models:
        f = d / f"{m}_memory_mb.txt"
        mem = f.read_text().strip() if f.exists() else "?"
        lat = []
        for i in range(1, args.runs + 1):
            p = d / f"{m}_run{i}.log"
            if p.exists():
                lat += [float(x) for x in
                        re.findall(r"(\d+\.\d)s \|", p.read_text(errors="replace"))]
        lat.sort()
        med = f"{lat[len(lat) // 2]:.1f}s" if lat else "?"
        print(f"  {m}: model memory {mem} MB | median latency {med} over {len(lat)} calls")

    print("\nNOTE: stability is not accuracy. A model can be perfectly stable and perfectly")
    print("      wrong. Adjudicate the contact sheets by eye before choosing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
