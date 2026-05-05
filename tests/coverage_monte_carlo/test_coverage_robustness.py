r"""Monte Carlo CI coverage under *robustness* DGPs.

The canonical Track B suite (``test_coverage.py``) checks that 95% CIs
hit nominal coverage on **well-specified** DGPs.  This sibling suite
checks the same estimators on DGPs that violate or stress the
identification assumptions (weak instruments, heterogeneous DiD timing,
severe propensity-score overlap loss).

The contract is intentionally different from the canonical suite.  These
tests do **not** assert that empirical coverage equals the nominal 0.95.
They assert that empirical coverage falls within a *documented band*
that we have measured at pilot time.  The band serves two purposes:

1. ``regression``: if the band is violated (coverage drops below the
   floor, or jumps above the ceiling), some estimator-internal change
   has shifted the calibration of the inference procedure on this DGP.
   Either the estimator improved (e.g. a fix lifts coverage) or it
   regressed (e.g. a refactor broke an SE term).  Either way, a
   maintainer should investigate.

2. ``honest documentation``: weak instruments are *expected* to under-
   cover with HC1 SEs.  AIPW with severe overlap loss is expected to
   over-cover (because the IF blows up).  Pretending these failures
   don't exist would be marketing, not science.  The bands let us
   record the magnitude of the textbook failure mode while still using
   the band as a regression guard against silent drift.

Bands were measured by direct simulation at pilot time and are
reproducible from a fixed seed schedule.  See ``run_robustness_b1000.py``
for the explicit-rate harness used in §5.3 of the JSS draft.

References
----------
- Stock, J. H. and Yogo, M. (2005). Testing for weak instruments in
  linear IV regression.  In *Identification and Inference for
  Econometric Models*, ed. D. W. K. Andrews and J. H. Stock,
  pp. 80-108.
- Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences
  with multiple time periods. JoE 225(2), 200-230.
- Crump, R. K., Hotz, V. J., Imbens, G. W., and Mitnik, O. A. (2009).
  Dealing with limited overlap in estimation of average treatment
  effects. Biometrika 96(1), 187-199.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# Default draws per test.  Override via env STATSPAI_MC_DRAWS=N.
B_DEFAULT = int(os.environ.get('STATSPAI_MC_DRAWS', 300))


def _coverage_rate(covered: int, B: int) -> float:
    return covered / B


def _assert_documented_band(covered: int, B: int, label: str,
                            lo: float, hi: float) -> None:
    """Assert empirical coverage lies in the documented [lo, hi] band.

    For robustness DGPs this band reflects the *known textbook
    failure-mode envelope* (under-cover for weak IV, over-cover for
    AIPW under poor overlap, etc.) rather than the Wilson band around
    nominal 0.95.
    """
    rate = _coverage_rate(covered, B)
    assert lo <= rate <= hi, (
        f"{label} coverage = {rate:.3f} outside documented "
        f"[{lo:.3f}, {hi:.3f}] band (B={B}). If this is a true "
        f"improvement, widen the band; if it is a regression, "
        f"investigate the SE / IF computation."
    )


# ---------------------------------------------------------------------------
# Robustness 1: Weak instrument
#
# DGP: linear y = truth*d + u; first stage d = pi*z + eu*u + noise with
#      pi = 0.10 (deliberately weak), endogeneity loading eu = 1.5.
# Pilot at B=300 yields F_med ~ 3 (Stock-Yogo 10% size critical value
# for one instrument is 16.4) and HC1-SE coverage ~ 0.91.  This is the
# canonical weak-IV under-coverage that motivated LIML / Anderson-Rubin
# inference.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_iv_weak_instrument_undercoverage():
    """Weak first stage (F_med ~ 3): HC1 2SLS CI under-covers truth.

    Documented band: 0.85 <= rate <= 0.95.  A regression below 0.85
    indicates additional SE breakage; above 0.95 means the under-
    coverage has been (somehow) corrected and the band can tighten.
    """
    B = B_DEFAULT
    truth = 1.0
    pi = 0.10            # weak first-stage coefficient
    eu = 1.5             # endogeneity loading
    n = 600
    covered = 0
    with warnings.catch_warnings():
        # Suppress the (correct, intentional) weak-instrument warnings
        # from sp.ivreg; we are testing precisely that regime.
        warnings.simplefilter("ignore")
        for seed in range(B):
            rng = np.random.default_rng(seed)
            z = rng.normal(size=n)
            u = rng.normal(size=n)
            d = pi * z + eu * u + rng.normal(scale=0.4, size=n)
            y = truth * d + u + rng.normal(scale=0.5, size=n)
            df = pd.DataFrame({"y": y, "d": d, "z": z})
            r = sp.ivreg("y ~ (d ~ z)", data=df, robust="hc1")
            ci = r.conf_int()
            lo, hi = ci.loc["d"].values
            if lo <= truth <= hi:
                covered += 1
    _assert_documented_band(covered, B, "Weak-IV (pi=0.10)",
                            lo=0.85, hi=0.95)


# ---------------------------------------------------------------------------
# Robustness 2: Heterogeneous-timing CS-DiD
#
# DGP: 4 cohorts {3, 5, 7, 0=never-treated}, n_units=200, T=8.
#      Cohort-specific ATT magnitudes tau_g in {1.0, 2.0, 3.0} plus
#      linear-in-time-since-treatment dynamics (slope 0.5).
# Population simple ATT computed analytically as the equal-weighted
# mean of ATT(g, t) over treated (g, t>=g) cells = 2.583.
# Pilot at B=200 yields cov ~ 0.94 — Callaway-Sant'Anna 2021 IS designed
# for this, so this row is a "robustness pass" rather than documented
# under-coverage.
# ---------------------------------------------------------------------------


# Population simple ATT under the heterogeneous DGP defined below.
# Computed analytically as the equal-weighted mean of ATT(g, t) over
# treated cells.  Hard-coded so the test fails if the DGP drifts.
_CS_HET_TRUTH = 2.583333333333333

# Cohort-specific level effects
_CS_HET_TAU_G = {3: 1.0, 5: 2.0, 7: 3.0}


def _cs_het_population_truth() -> float:
    """Recompute the population simple ATT for the heterogeneous DGP."""
    cells = []
    for g in (3, 5, 7):
        for t in range(g, 9):  # T = 8
            tst = t - g
            cells.append(_CS_HET_TAU_G[g] + 0.5 * tst)
    return float(np.mean(cells))


@pytest.mark.slow
def test_cs_heterogeneous_timing_coverage():
    """CS-DiD on cohort-and-dynamic heterogeneity DGP: must remain
    calibrated.  Documented band: 0.90 <= rate <= 0.97 (Wilson band
    at B=200; tightens to [0.92, 0.96] at B=1000).
    """
    # Sanity-check: hard-coded truth matches the analytic recomputation.
    assert abs(_cs_het_population_truth() - _CS_HET_TRUTH) < 1e-9
    B = min(B_DEFAULT, 200)   # CS is slow
    truth = _CS_HET_TRUTH
    n_units = 200
    T = 8
    cohorts = [3, 5, 7, 0]
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_units):
            g = cohorts[i % len(cohorts)]
            ui = rng.normal(scale=0.5)
            for t in range(1, T + 1):
                if g > 0 and t >= g:
                    att_eff = _CS_HET_TAU_G[g] + 0.5 * (t - g)
                else:
                    att_eff = 0.0
                y = 0.2 * t + att_eff + ui + rng.normal(scale=0.8)
                rows.append({'i': i, 't': t, 'g': g, 'y': y})
        df = pd.DataFrame(rows)
        r = sp.callaway_santanna(df, y='y', g='g', t='t', i='i',
                                 estimator='reg')
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    _assert_documented_band(covered, B, "CS-DiD heterogeneous timing",
                            lo=0.90, hi=0.97)


# ---------------------------------------------------------------------------
# Robustness 3: AIPW under severe propensity-overlap loss
#
# DGP: propensity model p(x) = sigmoid(-1.5 + 2.0*x1) → typical
#      propensity-score range covers [0.04, 0.96], so the AIPW IF gets
#      large 1/p and 1/(1-p) terms in the tails.  This is the
#      Crump-Hotz-Imbens-Mitnik (2009) limited-overlap regime.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_causal_forest_overlap_loss():
    """sp.causal_question(design='causal_forest') under severe overlap
    loss: AIPW-IF coverage is documented at pilot time.

    Documented band: 0.85 <= rate <= 0.99.  AIPW under poor overlap
    can either under-cover (IF tail variance under-estimated) or over-
    cover (CIs blow up); the wide band reflects both possibilities.
    """
    B = min(B_DEFAULT, 200)
    truth = 1.0
    n = 500
    covered = 0
    for seed in range(B):
        rng = np.random.default_rng(seed)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        # Strong selection: propensity ranges across roughly [0.04, 0.96]
        lin = -1.5 + 2.0 * x1
        p = 1.0 / (1.0 + np.exp(-lin))
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
    _assert_documented_band(covered, B, "Causal Forest AIPW (overlap loss)",
                            lo=0.85, hi=0.99)
