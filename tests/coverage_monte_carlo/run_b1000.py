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
from scipy.stats import t as t_dist

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


class _Draws:
    """Per-draw estimate / SE / interval, summarised beyond a coverage rate.

    A coverage rate alone cannot say *why* an interval over- or
    under-covers. Recording every draw lets the table report the bias, the
    Monte Carlo SD of the estimator, the mean reported SE, their ratio (the
    SE calibration), and the mean interval length next to the rate.
    """

    def __init__(self, truth: float) -> None:
        self.truth = truth
        self.est: list[float] = []
        self.se: list[float] = []
        self.lo: list[float] = []
        self.hi: list[float] = []

    def add(self, est, se, ci) -> bool:
        self.est.append(float(est))
        self.se.append(float(se) if se is not None else float("nan"))
        self.lo.append(float(ci[0]))
        self.hi.append(float(ci[1]))
        return _ci_covers(ci, self.truth)

    def summary(self, name: str) -> dict:
        est = np.asarray(self.est)
        se = np.asarray(self.se)
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        covered = int(np.sum((lo <= self.truth) & (self.truth <= hi)))
        mc_sd = float(np.std(est, ddof=1))
        mean_se = float(np.nanmean(se))
        return {
            "name": name,
            "B": int(est.size),
            "covered": covered,
            "rate": covered / est.size,
            "truth": self.truth,
            "bias": float(np.mean(est) - self.truth),
            "mc_sd": mc_sd,
            "mean_se": mean_se,
            "median_se": float(np.nanmedian(se)),
            "se_sd_ratio": mean_se / mc_sd if mc_sd > 0 else float("nan"),
            "rmse": float(np.sqrt(np.mean((est - self.truth) ** 2))),
            "mean_ci_length": float(np.mean(hi - lo)),
            "failures": int(np.sum(~np.isfinite(est))),
        }


def coverage_ols() -> dict:
    """OLS on RCT with covariates."""
    truth = 1.5
    rng = np.random.default_rng(2026)
    acc = _Draws(truth)
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
        acc.add(beta, se, ci)
    return acc.summary("sp.regress (HC1) on RCT")


def coverage_did_2x2() -> dict:
    truth = 2.0
    rng = np.random.default_rng(2026)
    acc = _Draws(truth)
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
        acc.add(beta, se, ci)
    return acc.summary("sp.regress 2x2 DiD")


def coverage_iv() -> dict:
    truth = 1.5
    rng = np.random.default_rng(2026)
    acc = _Draws(truth)
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
        acc.add(beta, se, ci)
    return acc.summary("sp.ivreg (HC1) on strong-Z IV")


def coverage_cs() -> dict:
    """Callaway--Sant'Anna simple ATT on a homogeneous staggered DGP.

    Same DGP as ``test_cs_staggered_ci_coverage`` (n_units=200, 8 periods,
    cohorts {3,5,7,never}); promoted from the B=200 pytest cap to a full
    B=1000 materialised audit.
    """
    truth = 1.5
    acc = _Draws(truth)
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
        acc.add(r.estimate, r.se, r.ci)
    return acc.summary("sp.callaway_santanna simple ATT (staggered)")


def coverage_ebalance() -> dict:
    """Entropy balancing on a CIA DGP (same DGP as the pytest row)."""
    truth = 2.0
    acc = _Draws(truth)
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
        acc.add(r.estimate, r.se, r.ci)
    return acc.summary("sp.ebalance (CIA, ATT)")


def coverage_dml() -> dict:
    """DML IRM ATE via ``sp.causal_question(design='dml')`` (binary D)."""
    truth = 1.0
    acc = _Draws(truth)
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
        acc.add(r.estimate, r.se, r.ci)
    return acc.summary("sp.causal_question(design='dml') IRM ATE")


def coverage_dml_plr() -> dict:
    """DML partially linear model (the suite's DML member) on a PLR DGP.

    The IRM row above answers a different question (binary treatment,
    interactive model). The twelve-estimator suite lists ``sp.dml(model=
    "plr")``, so it gets its own row: continuous treatment, nonlinear
    nuisances, ``theta = 1``, five-fold cross-fitting with the default
    learners.
    """
    truth = 1.0
    acc = _Draws(truth)
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n = 500
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        d = 0.5 * x1 + 0.3 * np.sin(2 * x2) + rng.normal(size=n)
        y = truth * d + x1 + 0.5 * x2**2 + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
        r = sp.dml(
            df, y="y", d="d", X=["x1", "x2"], model="plr", n_folds=5, random_state=seed
        )
        acc.add(r.estimate, r.se, r.ci)
    return acc.summary("sp.dml(model='plr') theta")


def coverage_causal_forest() -> dict:
    """Causal-forest population ATE via cross-fit AIPW-IF (binary D).

    Same DGP as ``test_causal_forest_aipw_ci_coverage``; the ATE summary
    is the doubly-robust AIPW influence-function mean -- the same
    estimator grf::average_treatment_effect reports.
    """
    truth = 1.0
    acc = _Draws(truth)
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
        # grf-default 2,000 trees: since 1.31 the ATE is the forest's own AIPW,
        # and a 30-tree forest leaves rows without out-of-bag predictions.
        r = q.estimate(random_state=seed)
        acc.add(r.estimate, r.se, r.ci)
    return acc.summary("sp.causal_question(design='causal_forest') AIPW ATE")


def coverage_panel_fe() -> dict:
    """Two-way FE panel on a known-coefficient DGP (same as the pytest row)."""
    truth = 1.5
    acc = _Draws(truth)
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
        acc.add(r.params["d"], r.std_errors["d"], (lo, hi))
    return acc.summary("sp.panel two-way FE")


def coverage_fast_feols() -> dict:
    """``sp.fast.feols`` on the same two-way FE DGP, unit-clustered CR1.

    The suite's HDFE member is ``sp.fast.feols``; the row above runs
    ``sp.panel``, a different entry point of the same absorption family.
    This row measures the member itself on identical draws (same seeds),
    with unit and period effects absorbed and CR1 errors by unit.
    """
    truth = 1.5
    acc = _Draws(truth)
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
        r = sp.fast.feols("y ~ d | i + t", data=df, vcov="cr1", cluster="i")
        est, se = float(r.coef()["d"]), float(r.se()["d"])
        # fixest's clustered inference: t with G - 1 degrees of freedom.
        crit = float(t_dist.ppf(0.975, n_units - 1))
        acc.add(est, se, (est - crit * se, est + crit * se))
    return acc.summary("sp.fast.feols two-way FE (CR1 by unit)")


def coverage_sdid() -> dict:
    """SDID placebo-SE 95% CI on a one-treated-unit factor-model DGP.

    Same DGP as ``test_synth_sdid_ci_coverage``. Classic Abadie SCM has no
    analytic CI; SDID (Arkhangelsky et al. 2021) does, and placebo is the
    recommended variance estimator for a single treated unit.
    """
    truth = 3.0
    acc = _Draws(truth)
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
        acc.add(r.estimate, r.se, r.ci)
    return acc.summary("sp.sdid placebo (1 treated)")


def coverage_rd() -> dict:
    """Sharp RD robust bias-corrected CI on the known-jump DGP.

    Same DGP as ``test_rd_sharp_ci_coverage``; materialised at B=1000 so
    the twelve-estimator suite's RD member carries a direct coverage row
    rather than only the slow-suite pytest check.
    """
    truth = 1.0
    acc = _Draws(truth)
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n = 1000
        x = rng.uniform(-1, 1, n)
        y = (
            2
            + 3 * x
            + x**2
            + truth * (x >= 0).astype(int)
            + rng.normal(scale=0.4, size=n)
        )
        df = pd.DataFrame({"y": y, "x": x})
        r = sp.rdrobust(df, y="y", x="x", c=0.0)
        acc.add(r.estimate, r.se, r.ci)
    return acc.summary("sp.rdrobust sharp (robust CI)")


def coverage_sun_abraham() -> dict:
    """Sun--Abraham overall ATT on the homogeneous staggered DGP.

    Same DGP as ``coverage_cs`` (cohorts {3,5,7,never}, homogeneous
    truth), so the interaction-weighted aggregation is checked directly
    rather than only through cross-estimator agreement.
    """
    truth = 1.5
    acc = _Draws(truth)
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
        # The committed B=1,000 row ran the pre-1.41 defaults (equal-weight
        # event-time headline, normal intervals); pinned so it reproduces.
        r = sp.sun_abraham(
            df,
            y="y",
            g="g",
            t="t",
            i="i",
            aggregation="event_time",
            crit_dist="normal",
        )
        acc.add(r.estimate, r.se, r.ci)
    return acc.summary("sp.sun_abraham overall ATT (staggered)")


def main() -> None:
    out: list[dict] = []
    # JOSS #10604 is published, so the artifact is no longer frozen at the
    # seven-DGP set: the HDFE/panel-FE, SDID, sharp-RD, and Sun--Abraham
    # members of the twelve-estimator suite now carry materialised
    # B=1,000 rows too, answering the "seven of twelve" coverage gap.
    fns = [
        coverage_ols,
        coverage_did_2x2,
        coverage_iv,
        coverage_cs,
        coverage_sun_abraham,
        coverage_panel_fe,
        coverage_fast_feols,
        coverage_rd,
        coverage_sdid,
        coverage_ebalance,
        coverage_dml,
        coverage_dml_plr,
        coverage_causal_forest,
    ]
    only = set(os.environ.get("STATSPAI_B1000_ONLY", "").split(",")) - {""}
    if only:
        fns = [f for f in fns if f.__name__ in only]
    for fn in fns:
        t0 = time.time()
        rec = fn()
        rec["wall_s"] = round(time.time() - t0, 1)
        out.append(rec)
        print(f"  {rec['name']:<40} cov={rec['rate']:.3f}  ({rec['wall_s']}s)")
    out_path = RESULTS_DIR / os.environ.get("STATSPAI_B1000_OUT", "coverage_b1000.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"OK -- wrote {out_path}")


if __name__ == "__main__":
    main()
