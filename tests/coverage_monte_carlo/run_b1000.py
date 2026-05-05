"""Track B at B=1000: re-run each calibrated estimator and dump
the actual coverage rate (not just pytest pass/fail) for §5.3 of
the manuscript.

This script imports the test functions' DGPs and replicates each
1000-rep loop, then writes results/coverage_b1000.json so the §5.3
table can be regenerated automatically.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statspai as sp


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results_b1000"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


B = 1000


def _ci_covers(ci, truth) -> bool:
    if ci is None or len(ci) != 2:
        return False
    lo, hi = ci
    return lo <= truth <= hi


def coverage_ols() -> dict:
    """OLS on RCT with covariates."""
    truth = 1.5
    rng = np.random.default_rng(2026)
    covered = 0
    for b in range(B):
        n = 800
        x = rng.normal(size=n)
        d = rng.binomial(1, 0.5, n)
        y = 0.3 * x + truth * d + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "x": x, "d": d})
        fit = sp.regress("y ~ d + x", data=df, robust="hc1")
        beta = float(fit.params["d"]); se = float(fit.std_errors["d"])
        ci = (beta - 1.96 * se, beta + 1.96 * se)
        if _ci_covers(ci, truth):
            covered += 1
    return {"name": "sp.regress (HC1) on RCT", "B": B,
            "covered": covered, "rate": covered / B}


def coverage_did_2x2() -> dict:
    truth = 2.0
    rng = np.random.default_rng(2026)
    covered = 0
    for b in range(B):
        n_per = 100
        # 2x2: 100 treated, 100 control, pre and post
        rows = []
        for unit in range(2 * n_per):
            T = unit < n_per  # treated indicator
            for t in (0, 1):
                y = (
                    0.3 * t + 0.5 * T + truth * T * t
                    + rng.normal(scale=0.5)
                )
                rows.append({"unit": unit, "year": t, "T": int(T),
                             "post": t, "y": y})
        df = pd.DataFrame(rows)
        fit = sp.regress("y ~ T + post + T:post", data=df, robust="hc1")
        # Coefficient on T:post is the ATT
        try:
            beta = float(fit.params["T:post"])
            se = float(fit.std_errors["T:post"])
        except KeyError:
            try:
                beta = float(fit.params["T:post"])
                se = float(fit.std_errors["T:post"])
            except KeyError:
                # Fall back to colon-name variants
                key = next(k for k in fit.params.index if "post" in k and "T" in k)
                beta = float(fit.params[key])
                se = float(fit.std_errors[key])
        ci = (beta - 1.96 * se, beta + 1.96 * se)
        if _ci_covers(ci, truth):
            covered += 1
    return {"name": "sp.regress 2x2 DiD", "B": B,
            "covered": covered, "rate": covered / B}


def coverage_iv() -> dict:
    truth = 1.5
    rng = np.random.default_rng(2026)
    covered = 0
    for b in range(B):
        n = 800
        z = rng.binomial(1, 0.5, n)
        # Strong first stage
        d = (0.6 * z + rng.normal(size=n)) > 0
        d = d.astype(int)
        u = rng.normal(size=n)
        y = truth * d + 0.5 * u + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": d, "z": z})
        fit = sp.ivreg("y ~ (d ~ z)", data=df, robust="hc1")
        beta = float(fit.params["d"]); se = float(fit.std_errors["d"])
        ci = (beta - 1.96 * se, beta + 1.96 * se)
        if _ci_covers(ci, truth):
            covered += 1
    return {"name": "sp.ivreg (HC1) on strong-Z IV", "B": B,
            "covered": covered, "rate": covered / B}


def main() -> None:
    out: list[dict] = []
    for fn in [coverage_ols, coverage_did_2x2, coverage_iv]:
        t0 = time.time()
        rec = fn()
        rec["wall_s"] = round(time.time() - t0, 1)
        out.append(rec)
        print(f"  {rec['name']:<40} cov={rec['rate']:.3f}  ({rec['wall_s']}s)")
    out_path = RESULTS_DIR / "coverage_b1000.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"OK -- wrote {out_path}")


if __name__ == "__main__":
    main()
