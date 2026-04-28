"""End-to-end OLS HDFE benchmark — ``sp.fast.feols`` vs pyfixest vs R fixest.

Mirror of ``run_baseline.py`` (Phase 0 fepois bench) for the linear path.
Usage:

    python3 benchmarks/hdfe/run_feols_bench.py [--small] [--medium]

Reports wall-clock + coef diff vs R ``fixest::feols`` (the reference);
falls back to pyfixest as the cross-check when R is unavailable. Output
is JSON to stdout and a copy to ``benchmarks/hdfe/feols_bench.json``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

def _make_panel(n: int, fe1_card: int, fe2_card: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fe1 = rng.integers(0, fe1_card, size=n)
    fe2 = rng.integers(0, fe2_card, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a1 = rng.normal(0, 0.5, size=fe1_card)[fe1]
    a2 = rng.normal(0, 0.3, size=fe2_card)[fe2]
    cl = rng.normal(0, 0.4, size=fe1_card)[fe1]
    eps = cl + rng.normal(size=n)
    y = 0.30 * x1 - 0.20 * x2 + a1 + a2 + eps
    return pd.DataFrame({
        "y": y, "x1": x1, "x2": x2,
        "fe1": fe1.astype(np.int32), "fe2": fe2.astype(np.int32),
    })


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _time(fn, n_warm: int = 1, n_runs: int = 3) -> Dict[str, Any]:
    """Run ``fn()`` ``n_warm`` warmups + ``n_runs`` timed runs; return summary."""
    for _ in range(n_warm):
        fn()
    times: List[float] = []
    last_result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        last_result = fn()
        times.append(time.perf_counter() - t0)
    return {
        "wall_min": min(times),
        "wall_mean": float(np.mean(times)),
        "n_runs": n_runs,
        "result": last_result,
    }


def _bench_statspai(df: pd.DataFrame) -> Dict[str, Any]:
    import statspai as sp
    def _run():
        return sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df, vcov="iid")
    summary = _time(_run)
    fit = summary.pop("result")
    summary["coef"] = {k: float(fit.coef()[k]) for k in ("x1", "x2")}
    summary["se"] = {k: float(fit.se()[k]) for k in ("x1", "x2")}
    return summary


def _bench_pyfixest(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        import pyfixest as pf
    except ImportError:
        return {"available": False, "reason": "pyfixest not installed"}
    def _run():
        return pf.feols(
            fml="y ~ x1 + x2 | fe1 + fe2", data=df,
            vcov="iid", fixef_rm="singleton",
        )
    summary = _time(_run)
    fit = summary.pop("result")
    summary["available"] = True
    summary["coef"] = {k: float(fit.coef()[k]) for k in ("x1", "x2")}
    summary["se"] = {k: float(fit.se()[k]) for k in ("x1", "x2")}
    return summary


def _bench_r_fixest(df: pd.DataFrame, tmp_csv: Path) -> Dict[str, Any]:
    if shutil.which("Rscript") is None:
        return {"available": False, "reason": "Rscript not on PATH"}
    df.to_csv(tmp_csv, index=False)
    r_script = (
        "suppressMessages({library(data.table); library(fixest); library(jsonlite)})\n"
        f"d <- fread('{tmp_csv}')\n"
        "# Warm + 3 timed runs to mirror the Python harness\n"
        "for (i in 1:1) { invisible(feols(y ~ x1 + x2 | fe1 + fe2, data=d)) }\n"
        "ts <- numeric(3)\n"
        "for (i in 1:3) {\n"
        "  t0 <- proc.time()['elapsed']\n"
        "  f <- feols(y ~ x1 + x2 | fe1 + fe2, data=d)\n"
        "  ts[i] <- as.numeric(proc.time()['elapsed'] - t0)\n"
        "}\n"
        "out <- list(\n"
        "  wall_min = min(ts), wall_mean = mean(ts), n_runs = 3,\n"
        "  coef = as.list(coef(f)), se = as.list(se(f))\n"
        ")\n"
        "cat(toJSON(out, auto_unbox=TRUE, digits=14))\n"
    )
    proc = subprocess.run(
        ["Rscript", "-e", r_script], capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        return {"available": False, "reason": f"Rscript failed: {proc.stderr[:200]}"}
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    out["available"] = True
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

DATASETS = {
    "small":  dict(n=100_000,   fe1_card=1_000,   fe2_card=50,    seed=42),
    "medium": dict(n=1_000_000, fe1_card=100_000, fe2_card=1_000, seed=43),
}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true",
                        help="Run only the 100k-row dataset")
    parser.add_argument("--medium", action="store_true",
                        help="Run only the 1M-row dataset")
    args = parser.parse_args(argv)

    sizes = []
    if args.small:
        sizes = ["small"]
    elif args.medium:
        sizes = ["medium"]
    else:
        sizes = ["small", "medium"]

    out_dir = Path(__file__).resolve().parent
    tmp_csv = out_dir / "_bench_panel.csv"

    report: Dict[str, Any] = {"datasets": {}}
    for size in sizes:
        cfg = DATASETS[size]
        df = _make_panel(**cfg)
        ds_report: Dict[str, Any] = {"config": cfg, "n_rows": len(df)}
        print(f"=== {size} ({len(df):,} rows, fe1={cfg['fe1_card']:,}, "
              f"fe2={cfg['fe2_card']:,}) ===")
        ds_report["statspai"] = _bench_statspai(df)
        print(f"  statspai     : {ds_report['statspai']['wall_min']*1000:.0f} ms")
        ds_report["pyfixest"] = _bench_pyfixest(df)
        if ds_report["pyfixest"].get("available"):
            print(f"  pyfixest     : {ds_report['pyfixest']['wall_min']*1000:.0f} ms")
        ds_report["fixest"] = _bench_r_fixest(df, tmp_csv)
        if ds_report["fixest"].get("available"):
            print(f"  R fixest     : {ds_report['fixest']['wall_min']*1000:.0f} ms")

        # Cross-backend coef diff: reference is fixest if available, else pyfixest
        if ds_report["fixest"].get("available"):
            ref = ds_report["fixest"]
            ref_name = "fixest"
        elif ds_report["pyfixest"].get("available"):
            ref = ds_report["pyfixest"]
            ref_name = "pyfixest"
        else:
            ref = None
            ref_name = "none"
        if ref is not None:
            diffs = {
                k: abs(ds_report["statspai"]["coef"][k] - float(ref["coef"][k]))
                for k in ("x1", "x2")
            }
            ds_report["coef_max_abs_diff_vs_" + ref_name] = max(diffs.values())
            print(f"  coef diff vs {ref_name}: max {max(diffs.values()):.3e}")

        report["datasets"][size] = ds_report

    out_json = out_dir / "feols_bench.json"
    with out_json.open("w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nWrote {out_json}")
    if tmp_csv.exists():
        tmp_csv.unlink()
    return report


if __name__ == "__main__":
    main()
