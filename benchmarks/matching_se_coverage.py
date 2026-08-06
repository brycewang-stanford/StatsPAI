#!/usr/bin/env python3
"""Monte-Carlo coverage of the matching standard errors.

Parity fixtures prove StatsPAI reproduces Stata.  They say nothing about
whether *either* package's standard error covers the truth.  This study
measures that directly: simulate from a design whose ATT is known, fit
``sp.match`` under every ``se_method`` x ``replace`` x ``n_matches``
combination, and count how often the nominal 95% interval contains it.

The quantities reported per cell:

``coverage``      share of replications whose 95% CI contains the true ATT
                  (nominal 0.95; below ~0.93 at 1000 reps is a real gap)
``mean_se``       average reported standard error
``sd_estimate``   standard deviation of the point estimates across
                  replications -- the *actual* sampling variability
``se_ratio``      ``mean_se / sd_estimate``; 1.0 means the reported SE is
                  correctly sized, < 1 means it is anti-conservative
``bias``          mean(estimate) - true ATT

``se_ratio`` is the diagnostic that matters: coverage conflates SE sizing
with bias, whereas ``se_ratio`` isolates whether the variance formula is
right.

Run::

    PYTHONPATH=src python benchmarks/matching_se_coverage.py
    PYTHONPATH=src python benchmarks/matching_se_coverage.py --quick

Writes ``benchmarks/results/matching_se_coverage.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import statspai as sp

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / "results"

#: Constant treatment effect, so the ATT equals TRUE_ATT by construction.
TRUE_ATT = 2.0
N_OBS = 400

ANALYTIC_SE = ["ai", "psmatch2", "abadie_imbens"]
K_GRID = [1, 2, 4]
REPLACE_GRID = [True, False]

#: Treatment-assignment intercepts giving three control-pool regimes.
#:
#: Matching *without replacement* needs ``k * N_treated <= N_control`` before
#: it can even form the requested matches; as that constraint binds, the good
#: partners get taken first and the estimator degrades toward the raw
#: difference in means.  That bias has nothing to do with any standard-error
#: formula, so the pools are separated to keep the two effects apart:
#:
#:   "rich"     ~10% treated (~9 controls each) -- k=4 without replacement fits
#:   "moderate" ~27% treated (~3 controls each) -- k=1 fits, k=4 does not
#:   "thin"     ~50% treated (equal arms)       -- even k=1 exhausts the pool
POOLS = {"rich": -2.2, "moderate": -1.2, "thin": 0.0}


def simulate(seed: int, pool: str = "rich", n: int = N_OBS) -> pd.DataFrame:
    """Selection on observables with a homogeneous treatment effect."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    index = POOLS[pool] + 0.9 * x1 - 0.5 * x2
    d = rng.binomial(1, 1 / (1 + np.exp(-index)))
    y = (
        1.0
        + TRUE_ATT * d
        + 0.7 * x1
        - 0.3 * x2
        + rng.normal(scale=0.5, size=n)
    )
    return pd.DataFrame({"x1": x1, "x2": x2, "d": d, "y": y})


def run_cell(
    se_method: str,
    replace: bool,
    k: int,
    reps: int,
    boot_reps: int,
    seed0: int,
    pool: str = "rich",
) -> Dict[str, Any]:
    estimates: List[float] = []
    ses: List[float] = []
    covered: List[bool] = []
    n_failed = 0

    for r in range(reps):
        df = simulate(seed0 + r, pool=pool)
        kwargs: Dict[str, Any] = dict(
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            n_matches=k,
            replace=replace,
            se_method=se_method,
        )
        if se_method == "bootstrap":
            kwargs["bootstrap_reps"] = boot_reps
            kwargs["bootstrap_seed"] = seed0 + r
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sp.match(df, **kwargs)
        except Exception:
            n_failed += 1
            continue
        if not (np.isfinite(res.estimate) and np.isfinite(res.se)):
            n_failed += 1
            continue
        estimates.append(float(res.estimate))
        ses.append(float(res.se))
        lo, hi = res.ci
        covered.append(bool(lo <= TRUE_ATT <= hi))

    if len(estimates) < 2:
        return {
            "pool": pool,
            "se_method": se_method,
            "replace": replace,
            "k": k,
            "n_usable": len(estimates),
            "n_failed": n_failed,
        }

    est = np.asarray(estimates)
    se = np.asarray(ses)
    sd_est = float(np.std(est, ddof=1))
    return {
        "pool": pool,
        "se_method": se_method,
        "replace": replace,
        "k": k,
        "n_usable": int(len(est)),
        "n_failed": int(n_failed),
        "coverage": float(np.mean(covered)),
        "mean_se": float(np.mean(se)),
        "sd_estimate": sd_est,
        "se_ratio": float(np.mean(se) / sd_est) if sd_est > 0 else float("nan"),
        "bias": float(np.mean(est) - TRUE_ATT),
    }


def to_markdown(rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    lines = [
        "# Matching standard errors: Monte-Carlo coverage",
        "",
        f"Generated by `benchmarks/matching_se_coverage.py`. "
        f"n = {meta['n_obs']}, true ATT = {meta['true_att']}, "
        f"{meta['reps']} replications "
        f"({meta['boot_sim_reps']} for bootstrap cells, "
        f"{meta['boot_reps']} bootstrap draws each).",
        "",
        "`se_ratio` = mean reported SE / actual sampling SD. **1.00 is",
        "correct; below 1 means the interval is too narrow.**",
        "",
        "| pool | se_method | replace | k | coverage | se_ratio | mean_se "
        "| sd_est | bias | n |",
        "| :-: | --- | :-: | :-: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if "coverage" not in r:
            lines.append(
                f"| {r['pool']} | `{r['se_method']}` | {r['replace']} | "
                f"{r['k']} | — | — | — | — | — | {r['n_usable']} |"
            )
            continue
        flag = ""
        if r["coverage"] < 0.90:
            flag = " ⚠️"
        elif r["coverage"] < 0.93:
            flag = " ⚡"
        lines.append(
            f"| {r['pool']} | `{r['se_method']}` | {r['replace']} | "
            f"{r['k']} | {r['coverage']:.3f}{flag} | {r['se_ratio']:.3f} | "
            f"{r['mean_se']:.4f} | {r['sd_estimate']:.4f} | "
            f"{r['bias']:+.4f} | {r['n_usable']} |"
        )
    lines += [
        "",
        "⚠️ coverage < 0.90 &nbsp;&nbsp; ⚡ coverage < 0.93 "
        "(nominal 0.95)",
        "",
        "**Read the `bias` column before the coverage column.** In the "
        "`thin` pool the arms are the same size, so matching *without "
        "replacement* exhausts the control pool and the estimator degrades "
        "toward the raw difference in means. The resulting coverage "
        "collapse is a property of that design, not of any standard error. "
        "The `rich` pool (~3 controls per treated) is the setting "
        "propensity-score matching is meant for, and is where the SE "
        "diagnostics are interpretable.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot-sim-reps", type=int, default=200)
    ap.add_argument("--boot-reps", type=int, default=99)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = ap.parse_args()

    reps, boot_sim_reps, boot_reps = args.reps, args.boot_sim_reps, args.boot_reps
    if args.quick:
        reps, boot_sim_reps, boot_reps = 40, 15, 25

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for pool in POOLS:
        print(f"--- control pool: {pool} ---", flush=True)
        for se_method in ANALYTIC_SE + ["bootstrap"]:
            cell_reps = boot_sim_reps if se_method == "bootstrap" else reps
            for replace in REPLACE_GRID:
                for k in K_GRID:
                    t = time.time()
                    row = run_cell(
                        se_method,
                        replace,
                        k,
                        cell_reps,
                        boot_reps,
                        seed0=10_000,
                        pool=pool,
                    )
                    rows.append(row)
                    cov = row.get("coverage")
                    print(
                        f"  {pool:5s} {se_method:15s} replace={str(replace):5s} "
                        f"k={k}  coverage="
                        f"{cov if cov is None else round(cov, 3)}  "
                        f"[{time.time() - t:.0f}s]",
                        flush=True,
                    )

    shares = {
        pool: float(
            np.mean([simulate(10_000 + i, pool=pool)["d"].mean() for i in range(20)])
        )
        for pool in POOLS
    }
    meta = {
        "n_obs": N_OBS,
        "treated_share_by_pool": {k: round(v, 3) for k, v in shares.items()},
        "true_att": TRUE_ATT,
        "reps": reps,
        "boot_sim_reps": boot_sim_reps,
        "boot_reps": boot_reps,
        "elapsed_s": round(time.time() - t0, 1),
        "statspai_version": getattr(sp, "__version__", "unknown"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "matching_se_coverage.json").write_text(
        json.dumps({"meta": meta, "cells": rows}, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "matching_se_coverage.md").write_text(
        to_markdown(rows, meta), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'matching_se_coverage.md'} "
          f"({meta['elapsed_s']}s)")


if __name__ == "__main__":
    main()
