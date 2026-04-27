"""Run a single Python HDFE Poisson backend on one dataset and emit JSON.

Usage::

    python3 run_python.py --dataset small --backend pyfixest [--repeats 3]
    python3 run_python.py --dataset medium --backend statspai

Designed to be invoked as a **subprocess** by ``run_baseline.py`` so that
peak RSS, GC pressure, and JIT state are isolated per backend per
dataset. Output (stdout) is a single JSON object; warnings and
diagnostics go to stderr.

Two backends are supported:

* ``pyfixest`` — calls ``pyfixest.fepois`` directly (the upstream Python
  implementation that ``sp.fepois`` currently wraps).
* ``statspai`` — calls ``sp.fepois``. Today this is a thin wrapper over
  pyfixest (so numbers should match), but we measure it independently
  to track future divergence as we move to a Rust backend.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent


def _read_dataset(name: str) -> pd.DataFrame:
    csv = HERE / "data" / f"{name}.csv.gz"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found — run datasets.py first to materialise the data."
        )
    dtypes = {"y": np.int64, "x1": np.float64, "x2": np.float64, "fe1": np.int32, "fe2": np.int32}
    return pd.read_csv(csv, dtype=dtypes)


class PeakRSSSampler:
    """Background thread that polls process RSS every ``interval`` seconds.

    macOS ``getrusage`` reports cumulative max-RSS that does not reset
    across calls within a process; sampling RSS from psutil gives a
    per-call peak when the process is short-lived (this script).
    """

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.proc = psutil.Process()
        self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss
            except psutil.Error:
                break
            if rss > self.peak:
                self.peak = rss
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakRSSSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def _extract_coefs(fit: Any) -> Dict[str, float]:
    """Pull (x1, x2) coefficients out of whichever object the backend returns."""
    # pyfixest >=0.18: .coef() returns a pandas Series.
    if hasattr(fit, "coef") and callable(fit.coef):
        try:
            ser = fit.coef()
            if isinstance(ser, pd.Series):
                return {k: float(ser[k]) for k in ("x1", "x2") if k in ser.index}
        except Exception:  # pragma: no cover  - defensive
            pass
    # statspai EconometricResults: .params is a dict-like or Series.
    if hasattr(fit, "params"):
        params = fit.params
        if isinstance(params, pd.Series):
            return {k: float(params[k]) for k in ("x1", "x2") if k in params.index}
        if isinstance(params, dict):
            return {k: float(params[k]) for k in ("x1", "x2") if k in params}
    raise RuntimeError(f"could not extract coefficients from {type(fit).__name__}")


def _extract_se(fit: Any) -> Dict[str, float]:
    # pyfixest path: fit.se() -> Series
    if hasattr(fit, "se") and callable(fit.se):
        try:
            ser = fit.se()
            if isinstance(ser, pd.Series):
                return {k: float(ser[k]) for k in ("x1", "x2") if k in ser.index}
        except Exception:
            pass
    # statspai EconometricResults: .std_errors is a Series
    for attr in ("std_errors", "bse"):
        val = getattr(fit, attr, None)
        if isinstance(val, pd.Series):
            return {k: float(val[k]) for k in ("x1", "x2") if k in val.index}
    return {}


def _run_pyfixest(df: pd.DataFrame) -> Any:
    import pyfixest as pf
    return pf.fepois(
        fml="y ~ x1 + x2 | fe1 + fe2",
        data=df,
        vcov="iid",
        fixef_rm="singleton",
        iwls_tol=1e-8,
        iwls_maxiter=25,
    )


def _run_statspai(df: pd.DataFrame) -> Any:
    import statspai as sp
    return sp.fepois(
        fml="y ~ x1 + x2 | fe1 + fe2",
        data=df,
        vcov="iid",
        fixef_rm="singleton",
        iwls_tol=1e-8,
        iwls_maxiter=25,
    )


BACKENDS: Dict[str, Callable[[pd.DataFrame], Any]] = {
    "pyfixest": _run_pyfixest,
    "statspai": _run_statspai,
}


def _bench(fn: Callable[[], Any], n_warmup: int, n_runs: int) -> Dict[str, Any]:
    """One warmup, then ``n_runs`` timed calls; return wall-time stats and last fit."""
    last_fit: Any = None
    for _ in range(n_warmup):
        last_fit = fn()
    runs = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        last_fit = fn()
        runs.append(time.perf_counter() - t0)
    return {
        "wall_runs": runs,
        "wall_min": min(runs),
        "wall_mean": sum(runs) / len(runs),
        "wall_max": max(runs),
        "fit": last_fit,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one Python HDFE Poisson backend.")
    ap.add_argument("--dataset", required=True, choices=("small", "medium", "large"))
    ap.add_argument("--backend", required=True, choices=tuple(BACKENDS))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    print(f"[run_python] loading {args.dataset}…", file=sys.stderr, flush=True)
    df = _read_dataset(args.dataset)
    fn = lambda: BACKENDS[args.backend](df)
    rss_before = psutil.Process().memory_info().rss

    print(
        f"[run_python] backend={args.backend}  warmup={args.warmup}  repeats={args.repeats}…",
        file=sys.stderr,
        flush=True,
    )
    err: Dict[str, Any] | None = None
    timing: Dict[str, Any] | None = None
    peak = rss_before
    try:
        with PeakRSSSampler() as sampler:
            timing = _bench(fn, n_warmup=args.warmup, n_runs=args.repeats)
        peak = sampler.peak
    except Exception as exc:  # pragma: no cover  - reported in JSON
        err = {"type": type(exc).__name__, "message": str(exc)}

    out: Dict[str, Any] = {
        "dataset": args.dataset,
        "backend": args.backend,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "n_rows": int(len(df)),
        "rss_before_bytes": int(rss_before),
        "peak_rss_bytes": int(peak),
        "peak_rss_delta_bytes": int(peak - rss_before),
        "wall": None,
        "coefs": None,
        "se": None,
        "error": err,
    }
    if timing is not None and err is None:
        out["wall"] = {k: timing[k] for k in ("wall_runs", "wall_min", "wall_mean", "wall_max")}
        try:
            out["coefs"] = _extract_coefs(timing["fit"])
            out["se"] = _extract_se(timing["fit"])
        except Exception as exc:
            out["error"] = {"type": type(exc).__name__, "message": str(exc)}

    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0 if err is None else 1


if __name__ == "__main__":
    sys.exit(main())
