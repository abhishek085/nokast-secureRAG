"""
Hardware + throughput telemetry for the DGX Spark (GB10 Grace-Blackwell).

The GB10 uses unified memory, so `nvidia-smi` reports "Not Supported" for the
device memory fields. This monitor therefore stitches together:
  - GPU util / temperature / power / SM+mem clocks   <- nvidia-smi --query-gpu (works)
  - unified memory used                              <- /proc/meminfo (MemTotal-MemAvailable)
  - per-process GPU memory                           <- nvidia-smi --query-compute-apps
sampled on a background thread, plus integrates power over time to report energy (Wh).

Usage:
    from src.telemetry import HardwareMonitor, Throughput
    with HardwareMonitor("gen-smoke") as mon:
        ... do work ...
        mon.mark(samples=10, tokens=3400)      # optional throughput counters
    summary = mon.summary                       # dict, also written to results/telemetry/
"""
import os
import json
import time
import threading
import subprocess
from typing import Dict, List, Optional, Any

NVSMI = "nvidia-smi"
_GPU_FIELDS = "utilization.gpu,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory"


def _run(cmd: List[str], timeout: float = 4.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def _meminfo_used_gb() -> Optional[float]:
    try:
        total = avail = None
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1])
                if total is not None and avail is not None:
                    break
        if total is None or avail is None:
            return None
        return (total - avail) / 1024 / 1024  # kB -> GiB
    except Exception:
        return None


def sample_once() -> Dict[str, Any]:
    """One telemetry snapshot. Missing fields come back as None, never raise."""
    snap: Dict[str, Any] = {"t": time.time()}
    out = _run([NVSMI, f"--query-gpu={_GPU_FIELDS}", "--format=csv,noheader,nounits"])
    if out.strip():
        parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]

        def num(i):
            try:
                return float(parts[i])
            except (ValueError, IndexError):
                return None
        snap.update(
            gpu_util=num(0), temp_c=num(1), power_w=num(2),
            sm_clock_mhz=num(3), mem_clock_mhz=num(4),
        )
    snap["mem_used_gb"] = _meminfo_used_gb()

    # per-process GPU memory (sum of compute apps), in MiB
    apps = _run([NVSMI, "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"])
    gpu_mib = 0.0
    for line in apps.strip().splitlines():
        try:
            gpu_mib += float(line.strip())
        except ValueError:
            pass
    snap["gpu_proc_mem_gb"] = gpu_mib / 1024 if gpu_mib else None
    return snap


def _stats(vals: List[Optional[float]]) -> Dict[str, Optional[float]]:
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs:
        return {"mean": None, "max": None, "min": None}
    return {"mean": round(sum(xs) / len(xs), 2), "max": round(max(xs), 2), "min": round(min(xs), 2)}


class HardwareMonitor:
    def __init__(self, tag: str, interval: float = 1.0, out_dir: str = "results/telemetry"):
        self.tag = tag
        self.interval = interval
        self.out_dir = out_dir
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._samples_done = 0
        self._tokens_done = 0
        self.summary: Dict[str, Any] = {}

    # ---- throughput counters (call as work completes) ----
    def mark(self, samples: int = 0, tokens: int = 0):
        self._samples_done += samples
        self._tokens_done += tokens

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append(sample_once())
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._t0 = time.time()
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 2 + 2)
        elapsed = max(time.time() - self._t0, 1e-9)

        # energy via trapezoidal integration of power over sample timestamps
        energy_wh = None
        pw = [(s["t"], s.get("power_w")) for s in self.samples if s.get("power_w") is not None]
        if len(pw) >= 2:
            joules = 0.0
            for (t0, p0), (t1, p1) in zip(pw, pw[1:]):
                joules += (p0 + p1) / 2 * (t1 - t0)
            energy_wh = round(joules / 3600, 4)

        self.summary = {
            "tag": self.tag,
            "elapsed_s": round(elapsed, 2),
            "n_telemetry_samples": len(self.samples),
            "gpu_util_pct": _stats([s.get("gpu_util") for s in self.samples]),
            "temp_c": _stats([s.get("temp_c") for s in self.samples]),
            "power_w": _stats([s.get("power_w") for s in self.samples]),
            "sm_clock_mhz": _stats([s.get("sm_clock_mhz") for s in self.samples]),
            "unified_mem_used_gb": _stats([s.get("mem_used_gb") for s in self.samples]),
            "gpu_proc_mem_gb": _stats([s.get("gpu_proc_mem_gb") for s in self.samples]),
            "energy_wh": energy_wh,
            "throughput": {
                "samples": self._samples_done,
                "tokens": self._tokens_done,
                "samples_per_s": round(self._samples_done / elapsed, 3) if self._samples_done else None,
                "tokens_per_s": round(self._tokens_done / elapsed, 1) if self._tokens_done else None,
            },
        }

        os.makedirs(self.out_dir, exist_ok=True)
        base = os.path.join(self.out_dir, self.tag)
        with open(base + ".summary.json", "w") as f:
            json.dump(self.summary, f, indent=2)
        with open(base + ".series.jsonl", "w") as f:
            for s in self.samples:
                f.write(json.dumps(s) + "\n")
        return self.summary

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        self.print_summary()
        return False

    def print_summary(self):
        s = self.summary
        if not s:
            return
        tp = s["throughput"]
        print("\n" + "=" * 60)
        print(f"HARDWARE / THROUGHPUT  [{s['tag']}]")
        print("=" * 60)
        print(f"  elapsed:        {s['elapsed_s']} s   ({s['n_telemetry_samples']} samples)")
        print(f"  GPU util:       mean {s['gpu_util_pct']['mean']}%   max {s['gpu_util_pct']['max']}%")
        print(f"  temperature:    mean {s['temp_c']['mean']}C   max {s['temp_c']['max']}C")
        print(f"  power:          mean {s['power_w']['mean']}W   max {s['power_w']['max']}W")
        print(f"  SM clock:       mean {s['sm_clock_mhz']['mean']} MHz")
        print(f"  unified mem:    peak {s['unified_mem_used_gb']['max']} GiB used")
        print(f"  GPU proc mem:   peak {s['gpu_proc_mem_gb']['max']} GiB")
        print(f"  energy:         {s['energy_wh']} Wh")
        if tp["samples_per_s"]:
            print(f"  throughput:     {tp['samples_per_s']} samples/s   {tp['tokens_per_s']} tok/s")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # quick self-test: monitor 5s of idle/used GPU
    with HardwareMonitor("selftest", interval=1.0) as mon:
        time.sleep(5)
        mon.mark(samples=0, tokens=0)
