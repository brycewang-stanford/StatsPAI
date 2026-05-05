"""Monte Carlo 95% CI coverage validation for every major estimator.

For each estimator, we generate B independent draws of a deterministic
DGP with known population parameter, build the 95% CI on each draw,
and check that the CI covers the truth at least 92% of the time
(Wilson 95% lower band for B=300 with nominal 0.95).

Failures indicate SE miscalibration — a class of bug that recovery
tests cannot detect.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# Default draws per test.  Override via env STATSPAI_MC_DRAWS=N.
# CI default: 300 (about 45-60s per test).  Set higher for deep audits.
B_DEFAULT = int(os.environ.get('STATSPAI_MC_DRAWS', 300))

# Nominal coverage
NOMINAL_COVERAGE = 0.95


def _coverage_rate(truths_in_ci: int, B: int) -> float:
    return truths_in_ci / B


def _wilson_bounds(B: int, conf: float = 0.99) -> tuple:
    """Return (lo, hi) Wilson-score bounds around nominal 0.95 for B draws.

    Uses ``conf`` confidence for the test-of-coverage itself (default
    99% — want to be permissive so only gross SE bugs fail).  For B=30
    this gives roughly [0.83, 0.99]; for B=300, [0.92, 0.98].
    """
    from scipy.stats import norm
    p = NOMINAL_COVERAGE
    z = norm.ppf((1 + conf) / 2)
    denom = 1 + z**2 / B
    centre = (p + z**2 / (2 * B)) / denom
    half = z * (p * (1 - p) / B + z**2 / (4 * B**2))**0.5 / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _assert_calibrated(covered: int, B: int, label: str,
                       upper_slack: float = 0.02) -> None:
    """Assert empirical coverage is within the Wilson band of nominal 0.95."""
    rate = _coverage_rate(covered, B)
    lo, hi = _wilson_bounds(B)
    # Most bugs manifest as under-coverage; upper bound is looser.
    assert lo <= rate <= hi + upper_slack, (
        f"{label} 95% CI coverage = {rate:.3f} outside Wilson band "
        f"[{lo:.3f}, {hi+upper_slack:.3f}] (B={B})"
    )


# ---------------------------------------------------------------------------
# OLS on RCT — baseline: should be essentially exactly calibrated
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ols_rct_ci_coverage():
    """OLS on RCT data with robust SE: 95% CI must cover truth in [0.92, 0.98]."""
    B = B_DEFAULT
    truth = 1.5
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n = 500
        d = rng.binomial(1, 0.5, n)
        x = rng.normal(size=n)
        y = 1.0 + 0.5 * x + truth * d + rng.normal(size=n)
        df = pd.DataFrame({'y': y, 'd': d, 'x': x})
        r = sp.regress('y ~ d + x', data=df, robust='hc1')
        ci = r.conf_int()
        # conf_int returns a DataFrame with 2 columns; row 'd' gives CI
        lo, hi = ci.loc['d'].values
        if lo <= truth <= hi:
            covered += 1
    _assert_calibrated(covered, B, 'OLS RCT')


# ---------------------------------------------------------------------------
# Classic 2x2 DID
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_did_2x2_ci_coverage():
    """Classic 2x2 DID on a homogeneous DGP: coverage must be calibrated."""
    B = B_DEFAULT
    truth = 2.0
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n = 300
        rows = []
        for i in range(n):
            treat = 1 if i < n // 2 else 0
            ui = rng.normal(scale=0.5)
            for t in [0, 1]:
                y = 1.0 + 0.3 * t + 0.5 * treat + truth * treat * t + ui + \
                    rng.normal(scale=0.7)
                rows.append({'i': i, 't': t, 'treated': treat,
                             'post': t, 'y': y})
        df = pd.DataFrame(rows)
        r = sp.did(df, y='y', treat='treated', time='t', post='post')
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    _assert_calibrated(covered, B, 'DID 2x2')


# ---------------------------------------------------------------------------
# CS staggered DID — smaller B since CS is slower
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cs_staggered_ci_coverage():
    """CS2021 on homogeneous staggered DGP: coverage must be calibrated.

    This guards the simple-ATT aggregation against influence-function
    scaling bugs: the group-time IFs are estimated on treated/control
    subsets but aggregated over the full unit universe.
    """
    B = min(B_DEFAULT, 200)   # CS is slow; cap at 200
    truth = 1.5
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n_units = 200
        cohorts = [3, 5, 7, 0]
        rows = []
        for i in range(n_units):
            g = cohorts[i % 4]
            ui = rng.normal(scale=0.5)
            for t in range(1, 9):
                post = 1 if (g > 0 and t >= g) else 0
                y = 0.2 * t + truth * post + ui + rng.normal(scale=0.8)
                rows.append({'i': i, 't': t, 'g': g, 'y': y})
        df = pd.DataFrame(rows)
        r = sp.callaway_santanna(df, y='y', g='g', t='t', i='i',
                                 estimator='reg')
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    _assert_calibrated(covered, B, 'CS2021 staggered')


# ---------------------------------------------------------------------------
# Sharp RD
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_rd_sharp_ci_coverage():
    """Sharp RD (rdrobust) on known-jump DGP: coverage must be calibrated."""
    B = B_DEFAULT
    truth = 1.0
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n = 1000
        x = rng.uniform(-1, 1, n)
        y = 2 + 3*x + x**2 + truth * (x >= 0).astype(int) + \
            rng.normal(scale=0.4, size=n)
        df = pd.DataFrame({'y': y, 'x': x})
        r = sp.rdrobust(df, y='y', x='x', c=0.0)
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    _assert_calibrated(covered, B, 'RD sharp')


# ---------------------------------------------------------------------------
# IV — strong instrument
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_iv_strong_ci_coverage():
    """2SLS with strong instrument: coverage must be calibrated."""
    B = B_DEFAULT
    truth = 1.5
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n = 600
        z = rng.binomial(1, 0.5, n)
        u = rng.normal(size=n)
        d = (0.2 + 0.6 * z + 0.3 * u +
             rng.normal(scale=0.3, size=n) > 0.5).astype(int)
        y = 1.0 + truth * d + 0.5 * u + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({'y': y, 'd': d, 'z': z})
        r = sp.ivreg('y ~ (d ~ z)', data=df, robust='hc1')
        ci = r.conf_int()
        lo, hi = ci.loc['d'].values
        if lo <= truth <= hi:
            covered += 1
    _assert_calibrated(covered, B, 'IV 2SLS')


# ---------------------------------------------------------------------------
# Matching — entropy balancing
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ebalance_ci_coverage():
    """Entropy balancing on CIA DGP: coverage must be calibrated."""
    B = min(B_DEFAULT, 200)
    truth = 2.0
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n = 500
        X1 = rng.normal(size=n)
        X2 = rng.normal(size=n)
        lin = -0.3 + 0.5 * X1 - 0.3 * X2
        p = 1 / (1 + np.exp(-lin))
        d = (rng.uniform(0, 1, n) < p).astype(int)
        y = 1.0 + 1.5 * X1 - 0.8 * X2 + truth * d + \
            rng.normal(scale=0.8, size=n)
        df = pd.DataFrame({'y': y, 'd': d, 'X1': X1, 'X2': X2})
        r = sp.ebalance(df, y='y', treat='d', covariates=['X1', 'X2'])
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    # Matching SEs are often conservative (wider CIs -> higher coverage);
    # allow 4% upper slack instead of default 2%.
    _assert_calibrated(covered, B, 'Ebalance CIA', upper_slack=0.04)


# ---------------------------------------------------------------------------
# DML — interactive regression (binary treatment, ATE) via causal_question
#
# Tests that the Neyman-orthogonal moment + cross-fit produce a
# well-calibrated 95% CI on a binary-D conditional-ignorability DGP.
# This is the canonical non-IV DML branch picked by the dispatcher.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dml_irm_ci_coverage():
    """sp.causal_question(design='dml') on binary D, no instruments:
    Neyman-orthogonal SE must be calibrated to ~95%."""
    B = min(B_DEFAULT, 200)   # cross-fit is slower; cap at 200
    truth = 1.0
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n = 500
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        # Conditional ignorability — propensity depends on (x1,x2).
        p = 1 / (1 + np.exp(-(0.4 * x1 - 0.2 * x2)))
        d = rng.binomial(1, p)
        y = 0.5 + truth * d + 0.6 * x1 + 0.3 * x2 + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
        q = sp.causal_question(
            treatment="d", outcome="y", design="dml",
            covariates=["x1", "x2"], data=df,
        )
        r = q.estimate()    # auto-picks IRM
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    # DML IRM SEs are sometimes slightly conservative on small n with
    # GBM nuisance; allow standard upper slack.
    _assert_calibrated(covered, B, "DML IRM (causal_question)")


# ---------------------------------------------------------------------------
# Causal Forest — AIPW influence function for population ATE
#
# The forest provides heterogeneous tau(x); ATE inference comes from
# cross-fit AIPW (van der Laan & Robins 2003; Chernozhukov et al. 2018),
# matching what grf::average_treatment_effect does in R. This is the
# inference path that replaced the previous NaN-SE fallback.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_causal_forest_aipw_ci_coverage():
    """sp.causal_question(design='causal_forest'): the ATE 95% CI from
    the cross-fit AIPW-IF must cover truth at ~95%."""
    # AIPW-IF + 30-tree forest is comparatively cheap; cap at 200 to
    # keep total wall-clock under ~3 minutes.
    B = min(B_DEFAULT, 200)
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
            treatment="d", outcome="y", design="causal_forest",
            covariates=["x1", "x2"], data=df,
        )
        r = q.estimate(n_estimators=30, random_state=seed)
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    # AIPW-IF with GBM nuisance can over-cover slightly when the
    # propensity model under-fits — allow 5% upper slack rather than
    # 2% so a 96-99% empirical coverage doesn't fail on small B.
    _assert_calibrated(covered, B, "Causal Forest AIPW-IF",
                       upper_slack=0.05)


# ---------------------------------------------------------------------------
# Fast-mode coverage: one cheap smoke test that ALWAYS runs (not slow)
# ---------------------------------------------------------------------------

def test_fast_ols_coverage_smoke():
    """B=50 smoke test — runs in normal CI to catch catastrophic SE bugs."""
    B = 50
    truth = 1.0
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        n = 300
        x = rng.normal(size=n)
        y = truth * x + rng.normal(size=n)
        df = pd.DataFrame({'y': y, 'x': x})
        r = sp.regress('y ~ x', data=df, robust='hc1')
        ci = r.conf_int()
        lo, hi = ci.loc['x'].values
        if lo <= truth <= hi:
            covered += 1
    rate = covered / B
    # With B=50, wider band needed (Wilson [0.84, 1.00])
    assert 0.84 <= rate <= 1.0, (
        f"OLS smoke coverage = {rate:.3f} outside [0.84, 1.0] (B={B})"
    )
