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
        beta = float(fit.params["d"])
        se = float(fit.std_errors["d"])
        ci = (beta - 1.96 * se, beta + 1.96 * se)
        if _ci_covers(ci, truth):
            covered += 1
    return {
        "name": "sp.regress (HC1) on RCT",
        "B": B,
        "covered": covered,
        "rate": covered / B,
    }


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
                y = 0.3 * t + 0.5 * T + truth * T * t + rng.normal(scale=0.5)
                rows.append({"unit": unit, "year": t, "T": int(T), "post": t, "y": y})
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
    return {
        "name": "sp.regress 2x2 DiD",
        "B": B,
        "covered": covered,
        "rate": covered / B,
    }


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
        beta = float(fit.params["d"])
        se = float(fit.std_errors["d"])
        ci = (beta - 1.96 * se, beta + 1.96 * se)
        if _ci_covers(ci, truth):
            covered += 1
    return {
        "name": "sp.ivreg (HC1) on strong-Z IV",
        "B": B,
        "covered": covered,
        "rate": covered / B,
    }


def coverage_cs() -> dict:
    """Callaway--Sant'Anna simple ATT on a homogeneous staggered DGP.

    Same DGP as ``test_cs_staggered_ci_coverage`` (n_units=200, 8 periods,
    cohorts {3,5,7,never}); promoted from the B=200 pytest cap to a full
    B=1000 materialised audit.
    """
    truth = 1.5
    covered = 0
    cohorts = [3, 5, 7, 0]
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n_units = 200
        rows = []
        for i in range(n_units):
            g = cohorts[i % 4]
            ui = rng.normal(scale=0.5)
            for t in range(1, 9):
                post = 1 if (g > 0 and t >= g) else 0
                y = 0.2 * t + truth * post + ui + rng.normal(scale=0.8)
                rows.append({"i": i, "t": t, "g": g, "y": y})
        df = pd.DataFrame(rows)
        r = sp.callaway_santanna(df, y="y", g="g", t="t", i="i", estimator="reg")
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    return {
        "name": "sp.callaway_santanna simple ATT (staggered)",
        "B": B,
        "covered": covered,
        "rate": covered / B,
    }


def coverage_ebalance() -> dict:
    """Entropy balancing on a CIA DGP (same DGP as the pytest row)."""
    truth = 2.0
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n = 500
        X1 = rng.normal(size=n)
        X2 = rng.normal(size=n)
        p = 1 / (1 + np.exp(-(-0.3 + 0.5 * X1 - 0.3 * X2)))
        d = (rng.uniform(0, 1, n) < p).astype(int)
        y = 1.0 + 1.5 * X1 - 0.8 * X2 + truth * d + rng.normal(scale=0.8, size=n)
        df = pd.DataFrame({"y": y, "d": d, "X1": X1, "X2": X2})
        r = sp.ebalance(df, y="y", treat="d", covariates=["X1", "X2"])
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    return {
        "name": "sp.ebalance (CIA, ATT)",
        "B": B,
        "covered": covered,
        "rate": covered / B,
    }


def coverage_dml() -> dict:
    """DML IRM ATE via ``sp.causal_question(design='dml')`` (binary D)."""
    truth = 1.0
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n = 500
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        p = 1 / (1 + np.exp(-(0.4 * x1 - 0.2 * x2)))
        d = rng.binomial(1, p)
        y = 0.5 + truth * d + 0.6 * x1 + 0.3 * x2 + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
        q = sp.causal_question(
            treatment="d", outcome="y", design="dml", covariates=["x1", "x2"], data=df
        )
        r = q.estimate()
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    return {
        "name": "sp.causal_question(design='dml') IRM ATE",
        "B": B,
        "covered": covered,
        "rate": covered / B,
    }


def coverage_causal_forest() -> dict:
    """Causal-forest population ATE via cross-fit AIPW-IF (binary D).

    Same DGP as ``test_causal_forest_aipw_ci_coverage``; the ATE summary
    is the doubly-robust AIPW influence-function mean -- the same
    estimator grf::average_treatment_effect reports.
    """
    truth = 1.0
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n = 500
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        p = 1 / (1 + np.exp(-(0.5 * x1)))
        d = rng.binomial(1, p)
        y = 0.5 + truth * d + 0.7 * x1 + 0.3 * x2 + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
        q = sp.causal_question(
            treatment="d",
            outcome="y",
            design="causal_forest",
            covariates=["x1", "x2"],
            data=df,
        )
        r = q.estimate(n_estimators=30, random_state=seed)
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    return {
        "name": "sp.causal_question(design='causal_forest') AIPW ATE",
        "B": B,
        "covered": covered,
        "rate": covered / B,
    }


def coverage_panel_fe() -> dict:
    """Two-way FE panel on a known-coefficient DGP (same as the pytest row)."""
    truth = 1.5
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n_units, n_time = 50, 6
        rows = []
        for i in range(n_units):
            ai = rng.normal()
            for t in range(n_time):
                d = rng.binomial(1, 0.5)
                y = ai + 0.3 * t + truth * d + rng.normal(scale=0.8)
                rows.append({"i": i, "t": t, "d": d, "y": y})
        df = pd.DataFrame(rows)
        r = sp.panel(df, formula="y ~ d", entity="i", time="t", method="fe")
        lo, hi = r.conf_int().loc["d"].values
        if lo <= truth <= hi:
            covered += 1
    return {
        "name": "sp.panel two-way FE",
        "B": B,
        "covered": covered,
        "rate": covered / B,
    }


def coverage_sdid() -> dict:
    """SDID placebo-SE 95% CI on a one-treated-unit factor-model DGP.

    Same DGP as ``test_synth_sdid_ci_coverage``. Classic Abadie SCM has no
    analytic CI; SDID (Arkhangelsky et al. 2021) does, and placebo is the
    recommended variance estimator for a single treated unit.
    """
    truth = 3.0
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n_ctrl, t0, t1 = 20, 12, 6
        n_t = t0 + t1
        f = rng.normal(size=n_t)
        units, times, ys = [], [], []
        for u in range(n_ctrl + 1):
            loading = rng.uniform(0.5, 1.5)
            ai = rng.normal()
            for t in range(n_t):
                base = ai + loading * f[t] + rng.normal(scale=0.5)
                eff = truth if (u == 0 and t >= t0) else 0.0
                units.append(u)
                times.append(t)
                ys.append(base + eff)
        df = pd.DataFrame({"u": units, "t": times, "y": ys})
        r = sp.sdid(
            df,
            outcome="y",
            unit="u",
            time="t",
            treated_unit=0,
            treatment_time=t0,
            se_method="placebo",
            n_reps=100,
            seed=seed,
        )
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    return {
        "name": "sp.sdid placebo (1 treated)",
        "B": B,
        "covered": covered,
        "rate": covered / B,
    }


def main() -> None:
    out: list[dict] = []
    # The canonical Track-B artifact is frozen at the published seven DGPs
    # while JOSS #10604 is in review (the manuscript narrative pins "seven").
    # ``coverage_panel_fe`` and ``coverage_sdid`` are kept available for a
    # post-review expansion but are intentionally NOT in the canonical list;
    # panel/sdid coverage is still exercised by the pytest rows in
    # ``test_coverage.py``.
    fns = [
        coverage_ols,
        coverage_did_2x2,
        coverage_iv,
        coverage_cs,
        coverage_ebalance,
        coverage_dml,
        coverage_causal_forest,
    ]
    for fn in fns:
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
