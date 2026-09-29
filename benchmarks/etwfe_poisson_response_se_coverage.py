#!/usr/bin/env python3
"""Monte-Carlo coverage of the Poisson ETWFE response-scale standard errors.

``sp.etwfe(family='poisson', fe='unit')`` reports the ATT in counts (the
average of ``mu_1 - mu_0`` over treated observations) with a delta-method
SE.  ``response_se='profile'`` (default) differentiates through the
profiled unit effect; ``response_se='margins'`` reproduces Stata
``jwdid ..., method(ppmlhdfe)`` + ``estat`` (``margins`` with the absorbed
effects held fixed, differentiated through ``ppmlhdfe``'s ``_cons``).
The parity tests show ``'margins'`` equals Stata; this study asks which
convention covers the truth.

DGP: ``N`` units x 8 periods, cohorts 4 / 6 / never (30 / 30 / 40%), unit
effects ``c_i ~ N(0.5, c_sd^2)``, a linear period trend, a constant
log-point effect 0.3, Poisson counts (optionally gamma-Poisson with
dispersion ``od``).  Two estimands:

``conditional``  mean over the realised treated rows of the true
                 ``mu_1 - mu_0`` -- the ATT of the sample's units
``population``   the same cells with ``exp(c_i)`` replaced by its mean
                 over the unit-effect distribution

Per cell: ``sd_cond`` = sd(estimate - conditional ATT), ``sd_est`` =
sd(estimate), the mean SE of each convention and the share of 95% CIs
covering each estimand.  Seeds are spaced ``seed0 + 100000 * r``.

Run::

    PYTHONPATH=src python benchmarks/etwfe_poisson_response_se_coverage.py
    PYTHONPATH=src python benchmarks/etwfe_poisson_response_se_coverage.py --quick

Writes ``benchmarks/results/etwfe_poisson_response_se_coverage.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import statspai as sp

TAU = 0.3
T = 8
CELLS = [
    # (N, overdispersion, c_sd)
    (100, 0.0, 0.8),
    (400, 0.0, 0.8),
    (100, 0.5, 0.8),
    (400, 0.5, 0.8),
    (400, 0.0, 0.3),
    (400, 0.0, 1.5),
]


def _one_cell(args) -> Dict[str, Any]:
    N, od, c_sd, reps, seed0 = args
    gam = np.linspace(0.0, 0.4, T)
    Ec = np.exp(0.5 + c_sd**2 / 2)
    rows: List[Dict[str, float]] = []
    for r in range(reps):
        rng = np.random.default_rng(seed0 + 100000 * r)
        g = rng.choice([4, 6, 0], size=N, p=[0.3, 0.3, 0.4])
        c = rng.normal(0.5, c_sd, size=N)
        ii = np.repeat(np.arange(N), T)
        tt = np.tile(np.arange(1, T + 1), N)
        gi = g[ii]
        D = (gi > 0) & (tt >= gi)
        base = np.exp(c[ii] + gam[tt - 1])
        lam = base * np.exp(TAU * D)
        if od > 0:
            lam = lam * rng.gamma(1 / od, od, size=lam.size)
        df = pd.DataFrame({"id": ii, "t": tt, "g": gi, "y": rng.poisson(lam)})
        row = {
            "cond": float((base[D] * (np.exp(TAU) - 1)).mean()),
            "pop": float((Ec * np.exp(gam[tt[D] - 1]) * (np.exp(TAU) - 1)).mean()),
        }
        for m in ("profile", "margins"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = sp.etwfe(
                    df,
                    y="y",
                    group="id",
                    time="t",
                    first_treat="g",
                    family="poisson",
                    fe="unit",
                    response_se=m,
                )
            row["est"] = f.estimate
            row["se_" + m] = f.se
        rows.append(row)
    d = pd.DataFrame(rows)
    out: Dict[str, Any] = {
        "N": N,
        "od": od,
        "c_sd": c_sd,
        "reps": reps,
        "sd_cond": float((d["est"] - d["cond"]).std()),
        "sd_est": float(d["est"].std()),
    }
    for m in ("profile", "margins"):
        se = d["se_" + m]
        out["se_" + m] = float(se.mean())
        out["cov_cond_" + m] = float((abs(d["est"] - d["cond"]) <= 1.96 * se).mean())
        out["cov_pop_" + m] = float((abs(d["est"] - d["pop"]) <= 1.96 * se).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="100 reps per cell")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    reps = 100 if args.quick else args.reps
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_one_cell, [(*c, reps, args.seed) for c in CELLS]))
    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stem = out_dir / "etwfe_poisson_response_se_coverage"
    meta = {"statspai": sp.__version__, "seconds": round(time.time() - t0, 1)}
    stem.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "cells": res}, indent=2), encoding="utf-8"
    )
    lines = [
        "# Poisson ETWFE response-scale SE coverage",
        "",
        f"StatsPAI {meta['statspai']}, {reps} replications per cell, nominal 95%.",
        "",
        "| N | od | c_sd | sd(est - cond) | se profile | se margins | sd(est) "
        "| cov cond profile | cov cond margins | cov pop profile | cov pop margins |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res:
        lines.append(
            f"| {r['N']} | {r['od']} | {r['c_sd']} | {r['sd_cond']:.4f} "
            f"| {r['se_profile']:.4f} | {r['se_margins']:.4f} | {r['sd_est']:.4f} "
            f"| {r['cov_cond_profile']:.3f} | {r['cov_cond_margins']:.3f} "
            f"| {r['cov_pop_profile']:.3f} | {r['cov_pop_margins']:.3f} |"
        )
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
