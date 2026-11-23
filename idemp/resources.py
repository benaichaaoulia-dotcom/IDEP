"""resource controls for the IDEP runner.

GPU script contract: explicit device selection with
fail-fast (NO silent CPU fallback — `allow_cpu` must be set explicitly for
CPU-only preflight checks), placement assertions, a first-step watchdog, and
a GPU-free unit-testable guard. nvidia-smi is queried at launch so the chosen
GPU is the idle/least-loaded one with enough free VRAM (ties broken by lowest
memory used); the query output is recorded for the experiment artifact.

Also: 80% per-process CPU cap (thread pools) and a GpuMonitor daemon thread
that samples utilization/memory to <log_dir>/gpu_monitor.jsonl.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import torch

try:
    from loguru import logger as _loguru
except ImportError:  # pragma: no cover
    import logging
    _loguru = logging.getLogger("idemp.resources")


class DeviceGuardError(RuntimeError):
    """Raised when the requested/available device cannot serve the run."""


GPU_REQUIRED_MSG = ("GPU REQUIRED BUT UNAVAILABLE: CUDA is not available and "
                    "--allow-cpu was not set (no silent CPU fallback)")


def query_gpus() -> list:
    """nvidia-smi snapshot: index, memory used/total (MiB), utilization (%)."""
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({"index": int(parts[0]), "mem_used_mib": int(parts[1]),
                         "mem_total_mib": int(parts[2]),
                         "util_pct": int(parts[3])})
    return gpus


def choose_gpu(gpus: list, min_free_mib: int = 23000) -> int:
    """Pick the idle/least-loaded GPU with enough free VRAM.

    Drop GPUs without `min_free_mib` free; of the rest pick the lowest
    utilization, breaking ties by lowest memory used. Raises DeviceGuardError
    when no GPU qualifies (never a silent CPU fallback).
    """
    eligible = [g for g in gpus
                if g["mem_total_mib"] - g["mem_used_mib"] >= min_free_mib]
    if not eligible:
        raise DeviceGuardError(
            f"no GPU with >= {min_free_mib} MiB free: "
            f"{[(g['index'], g['mem_used_mib'], g['mem_total_mib']) for g in gpus]}")
    best = sorted(eligible, key=lambda g: (g["util_pct"], g["mem_used_mib"]))[0]
    return best["index"]


def resolve_device(gpu: int = 0, allow_cpu: bool = False, min_free_mib: int = 23000,
                  record_path: str | None = None) -> torch.device:
    """Explicit device resolution with fail-fast.

    `gpu=-1` -> auto-select the least-loaded eligible GPU (query recorded).
    `allow_cpu=True` -> CPU device (explicit preflight path only).
    Raises DeviceGuardError when CUDA is unavailable without allow_cpu, or no
    GPU qualifies — the caller must NOT fall back to CPU silently.
    """
    if not torch.cuda.is_available():
        if allow_cpu:
            dev = torch.device("cpu")
            _loguru.info("device resolved: cpu (EXPLICIT --allow-cpu preflight path)")
            return dev
        raise DeviceGuardError(GPU_REQUIRED_MSG)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    gpus = query_gpus()
    chosen = choose_gpu(gpus, min_free_mib) if gpu == -1 else int(gpu)
    if chosen not in {g["index"] for g in gpus}:
        raise DeviceGuardError(f"gpu index {chosen} not present in nvidia-smi")
    record = {"gpus": gpus, "chosen": chosen, "min_free_mib": min_free_mib,
              "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if record_path:
        os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)
    torch.cuda.set_device(chosen)
    dev = torch.device(f"cuda:{chosen}")
    _loguru.info("device resolved: {} ({}) — nvidia-smi query recorded",
                 dev, torch.cuda.get_device_name(dev))
    return dev


def assert_params_on_device(model, dev: torch.device) -> None:
    """Placement assertion: every parameter on `dev`."""
    n_on = sum(1 for p in model.parameters() if p.device == dev)
    n_all = sum(1 for _ in model.parameters())
    if n_on != n_all:
        raise DeviceGuardError(f"only {n_on}/{n_all} params on {dev}")


def apply_cpu_cap(fraction: float = 0.8) -> int:
    """Per-process CPU cap (thread pools); returns threads set."""
    n = max(1, int((os.cpu_count() or 1) * fraction))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    torch.set_num_threads(n)
    _loguru.info("cpu cap: {} threads ({}% of {})", n, int(fraction * 100),
                 os.cpu_count())
    return n


class GpuMonitor(threading.Thread):
    """Daemon thread sampling GPU utilization/memory to a JSONL file.

    GPU-free safe: on a CPU device it records {"device": "cpu"} lines so the
    artifact always has the monitor stream. Start after device resolution;
    stop() is idempotent and joins with a short timeout.
    """

    def __init__(self, log_dir: str, run_name: str, interval_s: float = 30.0,
                 dev: torch.device | None = None):
        super().__init__(daemon=True, name="idemp-gpu-monitor")
        self.path = os.path.join(log_dir, f"{run_name}_gpu_monitor.jsonl")
        self.interval_s = interval_s
        self.dev = dev
        self._stop_evt = threading.Event()

    def _sample(self) -> dict:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if self.dev is not None and self.dev.type == "cuda" and torch.cuda.is_available():
            try:
                i = self.dev.index or 0
                props = torch.cuda.get_device_properties(i)
                alloc = torch.cuda.memory_allocated(i) / 1024 ** 3
                res = torch.cuda.memory_reserved(i) / 1024 ** 3
                util = -1
                try:
                    gpus = query_gpus()
                    for g in gpus:
                        if g["index"] == i:
                            util = g["util_pct"]
                except Exception:  # nvidia-smi hiccup — keep the torch stats
                    pass
                rec.update({"device": str(self.dev), "util_pct": util,
                            "mem_alloc_gib": round(alloc, 3),
                            "mem_reserved_gib": round(res, 3),
                            "mem_total_gib": round(props.total_memory / 1024 ** 3, 3)})
            except Exception as e:  # pragma: no cover
                rec.update({"device": str(self.dev), "error": str(e)})
        else:
            rec.update({"device": "cpu"})
        return rec

    def run(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            while not self._stop_evt.is_set():
                f.write(json.dumps(self._sample()) + "\n")
                f.flush()
                self._stop_evt.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=5.0)


class FirstStepWatchdog:
    """First-step watchdog: the first training step must
    complete within `max_seconds`; an unsupported-argument error, a silent CPU
    fallback, or implausible throughput fails fast BEFORE the long run."""

    def __init__(self, max_seconds: float):
        self.max_seconds = float(max_seconds)
        self.t0 = None
        self.checked = False

    def start(self) -> None:
        self.t0 = time.time()

    def check(self) -> None:
        if self.t0 is None:
            return
        dt = time.time() - self.t0
        self.checked = True
        _loguru.info("first step completed in {:.2f}s (budget {:.2f}s)",
                     dt, self.max_seconds)
        if dt > self.max_seconds:
            raise DeviceGuardError(
                f"FIRST-STEP WATCHDOG: first step took {dt:.2f}s > "
                f"{self.max_seconds:.2f}s budget — stop and repair the script "
                f"before the long run (device/throughput check)")
