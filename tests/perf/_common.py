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
import platform
import statistics
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
    return rec


@dataclass
class TimingResult:
    estimator: str
    side: str           # "py" or "R"
    n: int
    n_reps: int
    median_time_s: float
    iqr_time_s: float
    min_time_s: float
    max_time_s: float
    peak_mem_mb: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def time_repeat(fn: Callable[[], Any], n_reps: int = 5,
                warmup: int = 1) -> tuple[float, float, float, float, float | None]:
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
        q3 = statistics.median(times[(len(times) + 1) // 2:])
        iqr = q3 - q1
    else:
        iqr = max(times) - min(times)
    return median, iqr, min(times), max(times), peak_mem


def write_results(estimator: str, side: str, rows: list[TimingResult],
                  *, extra: Mapping[str, Any] | None = None) -> Path:
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
