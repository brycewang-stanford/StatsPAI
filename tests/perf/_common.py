"""Shared helpers for the StatsPAI Track C performance benchmark.

Each benchmark runs an estimator at a series of sample sizes, records
wall-clock time and peak memory across `n_reps` repetitions, and
writes a JSON record under results/. The companion R scripts run the
canonical R reference at the same sample sizes; compare_perf.py then
emits a Markdown rollup and a log-log scaling figure.

Hardware reported in results/_hardware.json so cross-machine
reproducibility is auditable.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _git_commit_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _jax_record() -> dict[str, Any]:
    rec: dict[str, Any] = {}
    try:
        import jax  # type: ignore
        import jaxlib  # type: ignore

        rec["jax"] = jax.__version__
        rec["jaxlib"] = jaxlib.__version__
        try:
            devs = jax.devices()
            rec["jax_default_platform"] = devs[0].platform
            rec["jax_devices"] = [str(d) for d in devs]
        except Exception:
            pass
        rec["jax_platforms_env"] = os.environ.get("JAX_PLATFORMS")
        rec["jax_platform_name_env"] = os.environ.get("JAX_PLATFORM_NAME")
    except ImportError:
        pass
    return rec


def _gpu_record() -> dict[str, Any]:
    """Best-effort NVIDIA GPU snapshot via nvidia-smi. None on non-CUDA hosts."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            lines = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append(
                        {
                            "name": parts[0],
                            "driver": parts[1],
                            "memory_total_mb": float(parts[2]),
                        }
                    )
            return {"gpus": gpus} if gpus else {}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {}


def hardware_record() -> dict[str, Any]:
    rec: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if _HAS_PSUTIL:
        rec["cpu_logical"] = psutil.cpu_count(logical=True)
        rec["cpu_physical"] = psutil.cpu_count(logical=False)
        rec["mem_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    sha = _git_commit_sha()
    if sha:
        rec["git_commit"] = sha
    jax_rec = _jax_record()
    if jax_rec:
        rec["jax"] = jax_rec
    gpu_rec = _gpu_record()
    if gpu_rec:
        rec["gpu"] = gpu_rec
    return rec


def gpu_hbm_peak_mb() -> float | None:
    """Return current GPU HBM used (MiB) on device 0, or None if unavailable.

    Caller is expected to record before/after around a workload and take the
    max; this helper is intentionally a single snapshot so callers can control
    sampling cadence.
    """
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            return float(info.used) / (1024**2)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except (ImportError, Exception):
        return None


@dataclass
class TimingResult:
    estimator: str
    side: str  # "py" or "R"
    n: int
    n_reps: int
    median_time_s: float
    iqr_time_s: float
    min_time_s: float
    max_time_s: float
    peak_mem_mb: float | None = None
    peak_gpu_hbm_mb: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def time_repeat(
    fn: Callable[[], Any], n_reps: int = 5, warmup: int = 1
) -> tuple[float, float, float, float, float | None]:
    """Returns (median, iqr, min, max, peak_mem_mb) over n_reps."""
    for _ in range(warmup):
        fn()
        gc.collect()
    times: list[float] = []
    peak_mem = None
    for _ in range(n_reps):
        gc.collect()
        if _HAS_PSUTIL:
            proc = psutil.Process()
            mem_before = proc.memory_info().rss / (1024**2)
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if _HAS_PSUTIL:
            mem_after = proc.memory_info().rss / (1024**2)
            peak = max(mem_before, mem_after)
            peak_mem = peak if peak_mem is None else max(peak_mem, peak)
    times.sort()
    median = statistics.median(times)
    if len(times) >= 4:
        q1 = statistics.median(times[: len(times) // 2])
        q3 = statistics.median(times[(len(times) + 1) // 2 :])
        iqr = q3 - q1
    else:
        iqr = max(times) - min(times)
    return median, iqr, min(times), max(times), peak_mem


def write_results(
    estimator: str,
    side: str,
    rows: list[TimingResult],
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    out = RESULTS_DIR / f"{estimator}_{side}.json"
    payload = {
        "estimator": estimator,
        "side": side,
        "rows": [r.to_dict() for r in rows],
        "hardware": hardware_record(),
        "extra": dict(extra or {}),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
