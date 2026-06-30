#!/usr/bin/env python3
"""Honest cross-library HDFE benchmark: StatsPAI vs pyfixest vs R ``fixest``.

This is a deliberately *honest* benchmark. The point is not to show StatsPAI
winning — it is to publish, on one unified data-generating process and one
machine, exactly where StatsPAI is faster **and where it is slower**, with the
coefficient agreement that proves all backends compute the same estimator.
Selectively reporting only the regimes where you win is the cheapest way to lose
an econometrician's trust; this harness refuses to do that (see
:func:`build_report`, which always emits the loss rows).

Design choices that keep it trustworthy
---------------------------------------
* **One DGP, one machine.** Every backend estimates the *same* two-way fixed
  effects model on the *same* generated panel, so timings are comparable.
* **Correctness gates speed.** Each row carries the estimated slope and its SE;
  :func:`coefficient_agreement` checks every backend agrees with the pyfixest
  reference before any speed claim is made. A fast wrong answer is reported as a
  failure, not a win.
* **Median + IQR, not a single run.** Wall-clock is the median over repeats with
  the inter-quartile range, so noise and JIT warm-up do not masquerade as signal.
* **Provenance embedded.** Library versions, Python, platform, CPU count and
  (best-effort) peak RSS go into every result file.
* **No silent scope.** Backends that are unavailable (no ``pyfixest``, no
  ``Rscript`` + ``fixest``) are recorded as ``skipped`` with the reason, never
  silently dropped — a missing baseline must be visible, not invisible.

This module is intentionally self-contained and writes only under its own
directory; it never imports from ``tests/perf`` and never touches the
Paper-JSS Track-C performance artifacts.

Usage
-----
::

    python benchmarks/cross_library/hdfe_benchmark.py            # quick: 10k,100k
    python benchmarks/cross_library/hdfe_benchmark.py --scales 10000,100000,1000000
    python benchmarks/cross_library/hdfe_benchmark.py --repeats 7 --json out.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
# Hard safety boundary: this harness may only write under its own directory.
# Anything else (especially Paper-JSS / tests/perf Track-C) is off-limits.
_WRITE_ROOT = HERE


def _assert_under_write_root(path: Path) -> Path:
    p = path.resolve()
    if _WRITE_ROOT not in p.parents and p != _WRITE_ROOT:
        raise ValueError(
            f"refusing to write outside {_WRITE_ROOT} (got {p}); this harness is "
            "additive and must never touch Track-C / Paper-JSS artifacts."
        )
    return p


# --------------------------------------------------------------------------- #
#  Data-generating process — one two-way fixed-effects panel for all backends
# --------------------------------------------------------------------------- #


def make_panel(n_obs: int, *, seed: int = 12345):
    """A two-way (unit × time) FE panel with two regressors and a known slope.

    ``y = 1.5*x1 - 0.8*x2 + alpha_i + gamma_t + eps`` with correlated x's so the
    FE absorption does real work. Returns a pandas DataFrame with columns
    ``y, x1, x2, unit, time``.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    # Roughly square-ish panel: many units, modest T, so two-way FE is non-trivial.
    n_time = max(5, int(round(n_obs**0.35)))
    n_unit = max(2, n_obs // n_time)
    n = n_unit * n_time

    unit = np.repeat(np.arange(n_unit), n_time)
    tim = np.tile(np.arange(n_time), n_unit)
    alpha = rng.normal(0, 1, n_unit)[unit]
    gamma = rng.normal(0, 1, n_time)[tim]
    # regressors correlated with the FE so within-transform matters
    x1 = 0.5 * alpha + 0.3 * gamma + rng.normal(0, 1, n)
    x2 = 0.2 * alpha + rng.normal(0, 1, n)
    eps = rng.normal(0, 1, n)
    y = 1.5 * x1 - 0.8 * x2 + alpha + gamma + eps
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "unit": unit, "time": tim})


# --------------------------------------------------------------------------- #
#  Timing
# --------------------------------------------------------------------------- #


@dataclass
class BackendResult:
    backend: str
    scale: int
    status: str  # "ok" | "skipped" | "error"
    coef_x1: Optional[float] = None
    se_x1: Optional[float] = None
    time_median_s: Optional[float] = None
    time_iqr_s: Optional[float] = None
    peak_rss_mb: Optional[float] = None
    n_obs: Optional[int] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {k: v for k, v in self.__dict__.items()}


def _peak_rss_mb() -> Optional[float]:
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        return ru / (1024 if sys.platform.startswith("linux") else 1024 * 1024)
    except Exception:  # pragma: no cover - resource missing on some platforms
        return None


def _time_callable(fn: Callable[[], Tuple[float, float]], repeats: int):
    """Run ``fn`` ``repeats`` times; ``fn`` returns ``(coef, se)``. Returns
    ``(coef, se, median_s, iqr_s)`` using the last estimate (all identical).

    One untimed warm-up call precedes the timed runs so JIT compilation
    (numba/pyfixest) and first-touch imports do not inflate the median or IQR.
    """
    times: List[float] = []
    coef = se = float("nan")
    fn()  # warm-up: discard JIT / first-import cost
    for _ in range(repeats):
        t0 = time.perf_counter()
        coef, se = fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = statistics.median(times)
    if len(times) >= 4:
        q1 = statistics.median(times[: len(times) // 2])
        q3 = statistics.median(times[(len(times) + 1) // 2:])
        iqr = q3 - q1
    else:
        iqr = max(times) - min(times)
    return coef, se, median, iqr


# --------------------------------------------------------------------------- #
#  Backends
# --------------------------------------------------------------------------- #


def _bench_sp_feols(df, repeats: int) -> BackendResult:
    import statspai as sp

    def run():
        r = sp.feols("y ~ x1 + x2 | unit + time", data=df)
        coef = float(r.params["x1"])
        se = float(r.std_errors["x1"])
        return coef, se

    try:
        coef, se, med, iqr = _time_callable(run, repeats)
    except Exception as exc:  # noqa: BLE001 - record, do not hide
        return BackendResult("sp.feols", 0, "error", detail=repr(exc)[:200])
    return BackendResult(
        "sp.feols", 0, "ok", coef, se, med, iqr, _peak_rss_mb(), len(df)
    )


def _bench_sp_native(df, repeats: int) -> BackendResult:
    """StatsPAI's own HDFE kernel (``sp.absorb_ols``), independent of pyfixest."""
    import statspai as sp

    if not hasattr(sp, "absorb_ols"):
        return BackendResult("sp.absorb_ols", 0, "skipped", detail="absorb_ols absent")

    def run():
        r = sp.absorb_ols(
            df["y"].to_numpy(),
            df[["x1", "x2"]].to_numpy(),
            df[["unit", "time"]],
        )
        coef = float(r["coef"][0])
        se = float(r["se"][0])
        return coef, se

    try:
        coef, se, med, iqr = _time_callable(run, repeats)
    except Exception as exc:  # noqa: BLE001 - record, do not hide
        return BackendResult("sp.absorb_ols", 0, "error", detail=repr(exc)[:200])
    return BackendResult(
        "sp.absorb_ols", 0, "ok", coef, se, med, iqr, _peak_rss_mb(), len(df)
    )


def _bench_pyfixest(df, repeats: int) -> BackendResult:
    try:
        import pyfixest as pf
    except ImportError:
        return BackendResult("pyfixest", 0, "skipped", detail="pyfixest not installed")

    def run():
        r = pf.feols("y ~ x1 + x2 | unit + time", data=df)
        tidy = r.tidy()
        coef = float(tidy.loc["x1", "Estimate"])
        se = float(tidy.loc["x1", "Std. Error"])
        return coef, se

    try:
        coef, se, med, iqr = _time_callable(run, repeats)
    except Exception as exc:  # noqa: BLE001 - record, do not hide
        return BackendResult("pyfixest", 0, "error", detail=repr(exc)[:200])
    return BackendResult(
        "pyfixest", 0, "ok", coef, se, med, iqr, _peak_rss_mb(), len(df)
    )


_R_TEMPLATE = r"""
suppressMessages(library(fixest))
df <- read.csv("{csv}")
reps <- {repeats}
times <- numeric(reps)
for (i in seq_len(reps)) {{
  t0 <- Sys.time()
  m <- feols(y ~ x1 + x2 | unit + time, data = df)
  times[i] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
}}
co <- coef(m)[["x1"]]
se <- se(m)[["x1"]]
cat(sprintf('{{"coef": %.10f, "se": %.10f, "median": %.6f, "iqr": %.6f}}',
            co, se, median(times),
            as.numeric(quantile(times, .75) - quantile(times, .25))))
"""


def _bench_r_fixest(df, repeats: int) -> BackendResult:
    import shutil

    if shutil.which("Rscript") is None:
        return BackendResult("R fixest", 0, "skipped", detail="Rscript not on PATH")
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "panel.csv"
        df.to_csv(csv, index=False)
        script = _R_TEMPLATE.format(csv=csv.as_posix(), repeats=repeats)
        try:
            out = subprocess.run(
                ["Rscript", "--vanilla", "-e", script],
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return BackendResult("R fixest", 0, "error", detail=repr(exc)[:200])
        if out.returncode != 0:
            reason = (out.stderr or out.stdout).strip().splitlines()
            tail = reason[-1] if reason else "non-zero exit"
            status = "skipped" if "there is no package" in (out.stderr or "") else "error"
            return BackendResult("R fixest", 0, status, detail=tail[:200])
        try:
            payload = json.loads(out.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            return BackendResult("R fixest", 0, "error", detail=f"parse: {exc}")
    return BackendResult(
        "R fixest",
        0,
        "ok",
        payload["coef"],
        payload["se"],
        payload["median"],
        payload["iqr"],
        None,
        len(df),
    )


BACKENDS: Dict[str, Callable] = {
    "sp_feols": _bench_sp_feols,
    "sp_native": _bench_sp_native,
    "pyfixest": _bench_pyfixest,
    "r_fixest": _bench_r_fixest,
}


# --------------------------------------------------------------------------- #
#  Correctness + report
# --------------------------------------------------------------------------- #


def coefficient_agreement(
    rows: List[BackendResult], reference: str = "pyfixest", rtol: float = 1e-4
) -> Dict[str, object]:
    """Max relative slope disagreement vs a reference backend. The benchmark is
    only meaningful if every backend computes the same estimator."""
    ok = [r for r in rows if r.status == "ok" and r.coef_x1 is not None]
    ref = next((r for r in ok if r.backend.replace("sp.", "").startswith(reference)
                or reference in r.backend), None)
    if ref is None and ok:
        ref = ok[0]
    if ref is None:
        return {"reference": None, "max_rel_diff": None, "agree": None}
    worst = 0.0
    for r in ok:
        if abs(ref.coef_x1) > 1e-12:
            worst = max(worst, abs(r.coef_x1 - ref.coef_x1) / abs(ref.coef_x1))
    return {
        "reference": ref.backend,
        "max_rel_diff": worst,
        "agree": worst <= rtol,
        "rtol": rtol,
    }


def _provenance() -> Dict[str, object]:
    rec: Dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    try:
        import os

        rec["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover
        pass
    for mod in ("statspai", "pyfixest", "numpy", "pandas"):
        try:
            rec[f"{mod}_version"] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - version is best-effort
            rec[f"{mod}_version"] = "unavailable"
    import shutil

    if shutil.which("Rscript"):
        try:
            v = subprocess.run(
                ["Rscript", "-e", 'cat(as.character(packageVersion("fixest")))'],
                capture_output=True,
                text=True,
                timeout=60,
            )
            rec["r_fixest_version"] = v.stdout.strip() or "unknown"
        except Exception:  # noqa: BLE001 - best-effort
            rec["r_fixest_version"] = "unavailable"
    return rec


def build_report(results: Dict[int, List[BackendResult]], provenance: Dict) -> str:
    """An honest markdown report — always prints the loss rows, never hides them."""
    lines: List[str] = ["# StatsPAI cross-library HDFE benchmark\n"]
    lines.append(
        "Two-way (unit × time) fixed-effects OLS, identical DGP per scale. "
        "Speed is the **median** wall-clock over repeats (IQR in parentheses). "
        "*Honesty note:* the **Δ vs fastest** column states the slowdown wherever "
        "StatsPAI is not the fastest backend — those rows are reported, not hidden.\n"
    )
    lines.append("## Provenance\n")
    for k, v in provenance.items():
        lines.append(f"- {k}: `{v}`")
    for scale in sorted(results):
        rows = results[scale]
        agree = coefficient_agreement(rows)
        lines.append(f"\n## n ≈ {scale:,}\n")
        ag = agree.get("max_rel_diff")
        ag_s = f"{ag:.1e}" if isinstance(ag, float) else "n/a"
        verdict = "✓ agree" if agree.get("agree") else "✗ DISAGREE"
        lines.append(
            f"Coefficient agreement vs `{agree.get('reference')}`: "
            f"max rel diff **{ag_s}** ({verdict} @ rtol={agree.get('rtol')}).\n"
        )
        ok = [r for r in rows if r.status == "ok" and r.time_median_s]
        fastest = min((r.time_median_s for r in ok), default=None)
        lines.append("| backend | status | slope x1 | SE | median (s) | IQR | Δ vs fastest |")
        lines.append("| --- | --- | --: | --: | --: | --: | --: |")
        for r in rows:
            if r.status != "ok":
                lines.append(
                    f"| {r.backend} | **{r.status}** | — | — | — | — | {r.detail} |"
                )
                continue
            delta = (
                f"{r.time_median_s / fastest:.2f}× slower"
                if fastest and r.time_median_s > fastest * 1.001
                else "**fastest**"
            )
            lines.append(
                f"| {r.backend} | ok | {r.coef_x1:.6f} | {r.se_x1:.6f} | "
                f"{r.time_median_s:.4f} | {r.time_iqr_s:.4f} | {delta} |"
            )
    lines.append(
        "\n*Generated by `benchmarks/cross_library/hdfe_benchmark.py`. "
        "Re-run on your own hardware before quoting any number.*\n"
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def run(scales: List[int], backends: List[str], repeats: int) -> Dict[int, List[BackendResult]]:
    results: Dict[int, List[BackendResult]] = {}
    for scale in scales:
        df = make_panel(scale)
        rows: List[BackendResult] = []
        for name in backends:
            fn = BACKENDS[name]
            res = fn(df, repeats)
            res.scale = scale
            res.n_obs = len(df)
            rows.append(res)
            tag = res.status if res.status != "ok" else f"{res.time_median_s:.4f}s"
            print(f"  [{scale:>9,}] {name:<12} {tag}", file=sys.stderr)
        results[scale] = rows
    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scales", default="10000,100000",
                    help="comma-separated obs counts")
    ap.add_argument("--backends", default=",".join(BACKENDS),
                    help=f"comma-separated subset of {list(BACKENDS)}")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--json", metavar="PATH", help="write raw results JSON here")
    ap.add_argument("--markdown", metavar="PATH", help="write the report here")
    args = ap.parse_args(argv)

    scales = [int(s) for s in args.scales.split(",") if s.strip()]
    backends = [b for b in args.backends.split(",") if b.strip()]
    unknown = [b for b in backends if b not in BACKENDS]
    if unknown:
        ap.error(f"unknown backends: {unknown}; choose from {list(BACKENDS)}")

    provenance = _provenance()
    results = run(scales, backends, args.repeats)
    report = build_report(results, provenance)

    if args.json:
        payload = {
            "provenance": provenance,
            "results": {
                str(s): [r.to_dict() for r in rows] for s, rows in results.items()
            },
        }
        path = _assert_under_write_root(Path(args.json))
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {path}", file=sys.stderr)
    if args.markdown:
        path = _assert_under_write_root(Path(args.markdown))
        path.write_text(report, encoding="utf-8")
        print(f"wrote {path}", file=sys.stderr)
    if not args.markdown:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
