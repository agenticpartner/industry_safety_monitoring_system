#!/usr/bin/env python3
"""
System metrics sampler for the dashboard, over whichever telemetry the platform has.

    python3 services/metrics.py --seconds 5      # print samples, for checking the parser

Two backends behind one `Sampler()`, chosen by what is actually present:

* **Jetson — `tegrastats`.** Tegra has no `nvidia-smi` worth using (it reports `[N/A]` for GPU
  memory), so tegrastats is the source of truth for GPU, NVDEC, EMC, thermals and power.
* **Discrete GPU — `nvidia-smi`.** No tegrastats exists. `nvidia-smi` supplies the same fields
  under different names, plus CPU read from `/proc/stat`, which tegrastats folds in for free.

Both emit the SAME sample dict, so `dashboard/system.html` and `/system` never learn which one
ran. Where a field has no counterpart it is `None` rather than `0`: an absent VIC is not an idle
VIC, and a chart that draws a flat zero for a block this machine does not have is lying.

## One sampler, not one per request

Each backend is a long-running process that prints a line per interval. Spawning one per HTTP
request would cost a process launch and a full interval of latency for every poll, and several
concurrent dashboards would spawn several copies. Instead one background thread owns a single
process and appends to a bounded deque; every reader gets the same buffer for free.

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

## The one field that means something different on each platform

`ram` is **the memory the GPU draws from**, which is why it maps to different things: on Jetson
that is the unified 64 GB the pipeline and both model servers share, and on a discrete card it is
VRAM. Reporting host RAM on a dGPU would be the wrong number — 128 GB of system memory the GPU
cannot touch says nothing about whether the next engine will allocate. Host RAM is carried
alongside as `sys_ram_*` for anyone who wants it.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
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


def is_jetson() -> bool:
    """Tegra is identified by the device tree, never by the architecture alone.

    `platform.machine() == "aarch64"` is also every ARM server and every Apple VM, and one of
    those with a discrete card must take the nvidia-smi path.
    """
    return os.path.exists("/etc/nv_tegra_release") or os.path.exists("/proc/device-tree/model")


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
        "sys_ram": None,
        "sys_ram_used_mb": None,
        "sys_ram_total_mb": None,
        "temp_gpu": temps.get("gpu"),
        "temp_cpu": temps.get("cpu"),
        "temp_tj": temps.get("tj"),
        "power_mw": sum(power.values()) or None,
        "power_gpu_soc_mw": power.get("VDD_GPU_SOC"),
    }


class _CpuTimes:
    """Per-core utilisation from /proc/stat, as deltas between consecutive reads.

    tegrastats reports this already; nvidia-smi does not report host CPU at all, so on a discrete
    GPU it is read here on the cadence nvidia-smi sets. Deltas, not the raw counters: /proc/stat
    is monotonic since boot, so the instantaneous figure is the difference between two samples and
    the FIRST sample can only ever be `None`.
    """

    def __init__(self) -> None:
        self._prev: dict[str, tuple[int, int]] = {}

    def read(self) -> tuple[int | None, int | None]:
        """(mean, max) percent busy across cores, or (None, None) on the first call."""
        try:
            with open("/proc/stat") as fh:
                lines = [ln for ln in fh if ln.startswith("cpu") and not ln.startswith("cpu ")]
        except OSError:
            return None, None

        pcts: list[float] = []
        for ln in lines:
            parts = ln.split()
            name, vals = parts[0], [int(v) for v in parts[1:]]
            if len(vals) < 4:
                continue
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)   # idle + iowait
            total = sum(vals)
            prev = self._prev.get(name)
            self._prev[name] = (total, idle)
            if prev is None:
                continue
            d_total, d_idle = total - prev[0], idle - prev[1]
            if d_total > 0:
                pcts.append(100.0 * (d_total - d_idle) / d_total)

        if not pcts:
            return None, None
        return round(sum(pcts) / len(pcts)), round(max(pcts))


def _host_ram_mb() -> tuple[int | None, int | None]:
    try:
        with open("/proc/meminfo") as fh:
            info = {}
            for ln in fh:
                k, _, v = ln.partition(":")
                info[k] = int(v.split()[0])
        total = info["MemTotal"] // 1024
        avail = info.get("MemAvailable", info.get("MemFree", 0)) // 1024
        return total - avail, total
    except (OSError, KeyError, ValueError, IndexError):
        return None, None


class _BaseSampler:
    """Bounded history plus the reader API. Subclasses supply `_run`."""

    def __init__(self, interval_ms: int = 2000, keep: int = 1800):
        self.interval_ms = interval_ms
        self.buf: deque[dict] = deque(maxlen=keep)
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.error: str | None = None
        self.backend = "none"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self.backend, daemon=True)
        self._thread.start()

    def _run(self) -> None:            # pragma: no cover - overridden
        raise NotImplementedError

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()

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


class TegrastatsSampler(_BaseSampler):
    """Owns a single `tegrastats` process and a bounded history."""

    def __init__(self, interval_ms: int = 2000, keep: int = 1800):
        super().__init__(interval_ms, keep)
        self.backend = "tegrastats"

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
        super().stop()
        subprocess.run(["sudo", "-n", "tegrastats", "--stop"], capture_output=True, timeout=10)


class NvSmiSampler(_BaseSampler):
    """Owns a single looping `nvidia-smi` and a bounded history.

    GPU 0 only, matching `gpu-id: 0` in both nvinfer configs. A second card would need pipeline
    changes before it would need telemetry.

    `--loop` takes whole seconds and floors at 1, so a sub-second interval_ms cannot be honoured;
    the requested value is rounded up rather than silently ignored.
    """

    # Order matters: it is the order of the CSV columns parsed below.
    QUERY = ("utilization.gpu,utilization.memory,utilization.decoder,utilization.encoder,"
             "memory.used,memory.total,temperature.gpu,power.draw")

    def __init__(self, interval_ms: int = 2000, keep: int = 1800):
        super().__init__(interval_ms, keep)
        self.backend = "nvidia-smi"
        self._cpu = _CpuTimes()

    def _run(self) -> None:
        if not shutil.which("nvidia-smi"):
            self.error = "nvidia-smi not found"
            return
        secs = max(1, round(self.interval_ms / 1000))
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self.QUERY}",
                 "--format=csv,noheader,nounits", "--id=0", "--loop", str(secs)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except (OSError, subprocess.SubprocessError) as e:
            self.error = f"could not start nvidia-smi: {e}"
            return

        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            s = self._parse(line)
            if s:
                self.buf.append(s)
                self.error = None
            elif line.strip():
                self.error = line.strip()[:160]

    def _parse(self, line: str) -> dict | None:
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 8:
            return None

        def num(v: str, cast=int):
            # Fields a card does not implement come back as "[N/A]" or "[Not Supported]". Those
            # are None, not zero — see the note at the top about drawing a flat zero for silicon
            # this machine does not have.
            try:
                return cast(v)
            except (TypeError, ValueError):
                return None

        gpu, mem_util, dec, enc, used, total, temp, power = (
            num(cols[0]), num(cols[1]), num(cols[2]), num(cols[3]),
            num(cols[4]), num(cols[5]), num(cols[6], float), num(cols[7], float))

        cpu_mean, cpu_max = self._cpu.read()
        sys_used, sys_total = _host_ram_mb()

        return {
            "t": time.time(),
            "gpu": gpu or 0,
            "cpu": cpu_mean or 0,
            "cpu_max": cpu_max or 0,
            "nvdec": dec or 0,
            "nvenc": enc or 0,
            # `utilization.memory` is the fraction of time the memory bus was being read or
            # written — the same question EMC_FREQ answers on Tegra, which is why it lands in the
            # same field rather than in one the dashboard would have to learn about.
            "emc": mem_util or 0,
            # No VIC on a discrete card. None, so the chart omits the series instead of drawing
            # an idle block that does not exist.
            "vic": None,
            # VRAM: the memory the GPU draws from. See the module docstring.
            "ram": round(100 * used / total) if used is not None and total else 0,
            "ram_used_mb": used,
            "ram_total_mb": total,
            "sys_ram": round(100 * sys_used / sys_total) if sys_used is not None and sys_total else None,
            "sys_ram_used_mb": sys_used,
            "sys_ram_total_mb": sys_total,
            "temp_gpu": temp,
            # No separate CPU package sensor over nvidia-smi, and no Tegra junction sensor. The
            # dashboard reads temp_tj, so the GPU temperature answers it — on a discrete card it
            # is the only die there is.
            "temp_cpu": None,
            "temp_tj": temp,
            "power_mw": round(power * 1000) if power is not None else None,
            "power_gpu_soc_mw": round(power * 1000) if power is not None else None,
        }


def Sampler(interval_ms: int = 2000, keep: int = 1800) -> _BaseSampler:  # noqa: N802
    """The right sampler for this machine.

    Deliberately a function rather than a class picked at import: the callers only ever construct
    one and read from it, and a factory keeps `from metrics import Sampler` working unchanged on
    both platforms.
    """
    return (TegrastatsSampler if is_jetson() else NvSmiSampler)(interval_ms, keep)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--interval", type=int, default=1000)
    a = ap.parse_args()
    s = Sampler(interval_ms=a.interval)
    print(f"backend: {s.backend}")
    s.start()
    time.sleep(a.seconds)
    s.stop()
    if s.error:
        print(f"ERROR: {s.error}")

    def f(v, unit=""):
        return "  n/a" if v is None else f"{v:3d}{unit}"

    for row in s.buf:
        print(f"gpu={f(row['gpu'])}% cpu={f(row['cpu'])}% nvdec={f(row['nvdec'])}% "
              f"emc={f(row['emc'])}% ram={f(row['ram'])}% ({row['ram_used_mb']}MB) "
              f"tj={row['temp_tj']}C pwr={row['power_mw']}mW")
    print(f"\n{len(s.buf)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
