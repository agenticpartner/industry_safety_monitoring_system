#!/usr/bin/env python3
"""
System metrics sampler — parses `tegrastats` into a ring buffer for the dashboard.

    python3 services/metrics.py --seconds 5      # print samples, for checking the parser

Jetson has no `nvidia-smi` worth using (it reports `[N/A]` for GPU memory on Tegra), so
`tegrastats` is the source of truth for GPU, NVDEC, EMC, thermals and power.

## One sampler, not one per request

`tegrastats` is a long-running process that prints a line per interval. Spawning it per HTTP
request would cost a process launch and a full interval of latency for every poll, and several
concurrent dashboards would spawn several copies. Instead one background thread owns a single
`tegrastats` and appends to a bounded deque; every reader gets the same buffer for free.

The buffer is bounded on purpose: a dashboard left open for a week must not grow the API's memory.
At the default 2s interval, 1800 samples is an hour of history in a few hundred KB.

## Why these fields

`GR3D_FREQ` (GPU), `NVDEC` and `EMC_FREQ` are the three that actually explain this system's
behaviour, and each has already been decisive during Phase 2:

* **GPU** — at `dfi=2` the pipeline sits near 3% GPU, which is what makes local reasoning
  affordable at all.
* **NVDEC** — at `dfi=3` the pipeline is decoder-bound at 99% NVDEC while the GPU idles, which is
  the finding that corrected the plan's headroom assumption.
* **EMC** — memory-controller frequency once looked like it explained a throughput anomaly (it
  did not — the real cause was contention), and it is still the first thing to check when
  throughput moves without GPU load moving.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import threading
import time
from collections import deque

# tegrastats emits one long line; each field is pulled out independently so a firmware change
# that adds or removes a block cannot break the rest of the parse.
RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
RE_SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
RE_CPU = re.compile(r"CPU \[([^\]]+)\]")
RE_EMC = re.compile(r"EMC_FREQ (\d+)%")
RE_GPU = re.compile(r"GR3D_FREQ (\d+)%")
RE_NVDEC = re.compile(r"NVDEC\d* (\d+)%")
RE_NVENC = re.compile(r"NVENC\d* (\d+)%")
RE_VIC = re.compile(r"VIC (\d+)%")
RE_TEMP = re.compile(r"(\w+)@([\d.]+)C")
RE_POWER = re.compile(r"(VDD_\w+|VIN_\w+) (\d+)mW")


def parse(line: str) -> dict | None:
    """One tegrastats line -> a flat sample. Returns None for lines that are not samples."""
    m_ram = RE_RAM.search(line)
    if not m_ram:
        return None
    used, total = int(m_ram.group(1)), int(m_ram.group(2))

    cpus: list[int] = []
    m_cpu = RE_CPU.search(line)
    if m_cpu:
        for part in m_cpu.group(1).split(","):
            p = part.split("%")[0]
            if p.replace(".", "").isdigit():
                cpus.append(int(float(p)))

    def pct(rx) -> int | None:
        m = rx.search(line)
        return int(m.group(1)) if m else None

    temps = {k: float(v) for k, v in RE_TEMP.findall(line)}
    power = {k: int(v) for k, v in RE_POWER.findall(line)}

    return {
        "t": time.time(),
        # Percentages, all 0-100, so they share one axis on the chart. Mixing a percentage with a
        # megabyte count on one plot would need two scales, which is the chart mistake to avoid —
        # RAM is therefore carried as a percentage AND as raw MB for the tooltip.
        "gpu": pct(RE_GPU) or 0,
        "cpu": round(sum(cpus) / len(cpus)) if cpus else 0,
        "cpu_max": max(cpus) if cpus else 0,
        "nvdec": pct(RE_NVDEC) or 0,
        "nvenc": pct(RE_NVENC) or 0,
        "emc": pct(RE_EMC) or 0,
        "vic": pct(RE_VIC) or 0,
        "ram": round(100 * used / total) if total else 0,
        "ram_used_mb": used,
        "ram_total_mb": total,
        "temp_gpu": temps.get("gpu"),
        "temp_cpu": temps.get("cpu"),
        "temp_tj": temps.get("tj"),
        "power_mw": sum(power.values()) or None,
        "power_gpu_soc_mw": power.get("VDD_GPU_SOC"),
    }


class Sampler:
    """Owns a single `tegrastats` process and a bounded history."""

    def __init__(self, interval_ms: int = 2000, keep: int = 1800):
        self.interval_ms = interval_ms
        self.buf: deque[dict] = deque(maxlen=keep)
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tegrastats", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # `tegrastats --stop` first: only one instance may run, and a leftover from a previous
        # process (a killed benchmark, say) makes this one exit immediately with no error.
        subprocess.run(["sudo", "-n", "tegrastats", "--stop"],
                       capture_output=True, timeout=10)
        try:
            self._proc = subprocess.Popen(
                ["sudo", "-n", "tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except (OSError, subprocess.SubprocessError) as e:
            self.error = f"could not start tegrastats: {e}"
            return
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            s = parse(line)
            if s:
                self.buf.append(s)
                self.error = None
            elif "sudo" in line.lower() or "password" in line.lower():
                # Passwordless sudo is required; say so rather than silently producing no data.
                self.error = line.strip()[:160]

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        subprocess.run(["sudo", "-n", "tegrastats", "--stop"], capture_output=True, timeout=10)

    def current(self) -> dict | None:
        return self.buf[-1] if self.buf else None

    def history(self, minutes: float = 5.0, max_points: int = 300) -> list[dict]:
        """Recent samples, decimated to at most `max_points`.

        Decimation happens server-side: sending 1800 points to redraw a 600px-wide chart wastes
        bandwidth and paints multiple samples onto the same pixel column.
        """
        cutoff = time.time() - minutes * 60
        rows = [s for s in self.buf if s["t"] >= cutoff]
        if len(rows) <= max_points:
            return rows
        step = len(rows) / max_points
        return [rows[int(i * step)] for i in range(max_points)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--interval", type=int, default=1000)
    a = ap.parse_args()
    s = Sampler(interval_ms=a.interval)
    s.start()
    time.sleep(a.seconds)
    s.stop()
    if s.error:
        print(f"ERROR: {s.error}")
    for row in s.buf:
        print(f"gpu={row['gpu']:3d}% cpu={row['cpu']:3d}% nvdec={row['nvdec']:3d}% "
              f"emc={row['emc']:3d}% ram={row['ram']:3d}% ({row['ram_used_mb']}MB) "
              f"tj={row['temp_tj']}C pwr={row['power_mw']}mW")
    print(f"\n{len(s.buf)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
