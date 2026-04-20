"""
Regression tests for the post-review fixes.

Each test pins a behaviour that a specific fix changed, so future edits
that regress one of them fail here with a clear attribution.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.dml import DoubleMLPLR
from statspai.principal_strat import principal_strat


# ---------------------------------------------------------------------
# Fix #1 — MSM IRLS eta/eta_clip alignment (binomial path stability)
# ---------------------------------------------------------------------

def test_msm_binomial_does_not_diverge_on_extreme_eta():
    """
    Extreme confounder range exercises the IRLS linear-predictor clipping
    path. The previous bug (z = unclipped_eta + residual from clipped p)
    would produce NaN/Inf or very large coefficients for any row with
    |η| > 30. After the fix, IRLS converges to a finite estimate.
    """
    rng = np.random.default_rng(0)
    n_units = 150
    T = 4
    rows = []
    for i in range(n_units):
        V = rng.normal()
        L_t = rng.normal()
        A_prev = 0.0
        for t in range(T):
            # Very strong confounder signal for A → extreme linear
            # predictor in the outcome model after stacking.
            L_t = L_t + 0.5 * A_prev + rng.normal(0, 0.3)
            A = float(rng.binomial(1, 1 / (1 + np.exp(-2.5 * L_t))))
            A_prev = A
            rows.append({'id': i, 'time': t, 'A': A, 'L_lag': L_t, 'V': V})
    panel = pd.DataFrame(rows)
    prob = 1 / (1 + np.exp(-(0.5 * panel.groupby('id')['A'].cumsum() + 5 * panel['V'])))
    panel['Y'] = rng.binomial(1, prob)

    r = sp.msm(
        panel, y='Y', treat='A',
        id='id', time='time',
        time_varying=['L_lag'], baseline=['V'],
        family='binomial', exposure='cumulative',
    )
    # Finite estimate and finite, strictly positive SE
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.se) and r.se > 0
    # Reasonable magnitude — no coefficient-blowup
    assert abs(r.estimate) < 10


# ---------------------------------------------------------------------
# Fix #2 — DML n_rep>1 SE aggregation (Chernozhukov et al. 2018 eq. 3.7)
# ---------------------------------------------------------------------

def test_dml_n_rep_se_reflects_between_rep_variance():
    """
    For n_rep > 1, the aggregated SE must include between-rep dispersion:
        σ̂² = median_r ( se_r² + (θ̂_r − θ̂_med)² )

    We verify this by setting up a DGP where splits-randomness gives
    non-trivial between-rep variance, then comparing the aggregated SE
    to median(se_r). Aggregated SE must be ≥ median(se_r).
    """
    rng = np.random.default_rng(2026)
    n = 600
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = np.cos(X1) + X2 + rng.normal(0, 0.5, n)
    Y = 2.0 * D + np.sin(X1) + X2**2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})

    # Fit with n_rep=5 and grab the per-rep theta/se that the aggregator
    # stored under model_info when n_rep > 1.
    r = sp.dml(df, y='y', treat='d', covariates=['x1', 'x2'],
               n_rep=5)
    thetas = np.asarray(r.model_info['theta_all_reps'])
    ses = np.asarray(r.model_info['se_all_reps'])
    theta_med = float(np.median(thetas))
    within_only = float(np.median(ses))
    full = float(np.sqrt(np.median(ses**2 + (thetas - theta_med) ** 2)))

    # Reported SE should match the "full" formula to high precision
    assert abs(r.se - full) < 1e-10
    # And should be at least as large as within_only (median(se_r))
    assert r.se >= within_only - 1e-10


# ---------------------------------------------------------------------
# Fix #3 — Zhang-Rubin k-floor degeneracy (pi_always tiny → NaN bounds)
# ---------------------------------------------------------------------

def test_zhang_rubin_bounds_nan_when_no_always_takers():
    """
    A DGP with ONLY compliers and never-takers (no always-takers) should
    produce NaN SACE bounds rather than a spurious non-zero interval
    built from a forced k=1 slice.
    """
    rng = np.random.default_rng(42)
    n = 1500
    D = rng.binomial(1, 0.5, n).astype(float)
    # 80% complier (S=D), 20% never (S=0) — ZERO always-takers
    u = rng.uniform(0, 1, n)
    types = np.where(u < 0.8, 'C', 'N')
    S = np.where(types == 'N', 0.0, D).astype(float)
    Y = np.where(types == 'C', 2.0 * D + rng.normal(0, 0.3, n),
                 rng.normal(0, 0.3, n))
    df = pd.DataFrame({'y': Y, 'd': D, 's': S})

    res = principal_strat(df, y='y', treat='d', strata='s',
                          method='monotonicity', n_boot=40, seed=0)
    sace_lo = res.bounds.loc[0, 'estimate']
    sace_hi = res.bounds.loc[1, 'estimate']
    # With pi_always ≈ 0, SACE bounds are undefined: NaN on the point
    # estimate (bootstrap NaN propagation is acceptable).
    assert np.isnan(sace_lo)
    assert np.isnan(sace_hi)


# ---------------------------------------------------------------------
# Fix #4 — IIVM subgroup-fit minimum size (no GBM on ≤2 rows)
# ---------------------------------------------------------------------

def test_iivm_constant_fits_rather_than_overfitting_tiny_subgroup():
    """
    Smoke test: very small sample exercising a tiny Z-arm subgroup.
    Previously the subgroup fit ran on 2 rows and produced wild
    predictions. After the fix, subgroups below MIN_SUBGROUP_FIT=10
    fall back to the constant subgroup mean — results are stable (SE
    stays finite, estimate doesn't explode).
    """
    rng = np.random.default_rng(99)
    n = 200
    X = rng.normal(0, 1, n)
    Z = (rng.uniform(0, 1, n) < 0.1).astype(float)  # very few Z=1
    u = rng.uniform(0, 1, n)
    D = np.where(u < 0.7, Z, 1.0).astype(float)
    Y = 1.5 * D + 0.5 * X + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'x': X})

    r = sp.dml(df, y='y', treat='d', covariates=['x'],
               model='iivm', instrument='z', n_folds=2)
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.se) and r.se > 0
    # Not checking recovery precision — small n + rare Z=1 is meant
    # to stress the fallback, not give a good estimate. We just
    # verify that the estimator didn't blow up.


# ---------------------------------------------------------------------
# Fix #5 — IIVM no-variation early error
# ---------------------------------------------------------------------

def test_iivm_raises_when_treatment_is_constant():
    """No treatment variation ⇒ LATE not identified ⇒ early ValueError."""
    rng = np.random.default_rng(0)
    n = 500
    Z = rng.binomial(1, 0.5, n).astype(float)
    D = np.ones(n, dtype=float)  # every unit treated
    Y = 2.0 * D + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'x': rng.normal(0, 1, n)})
    with pytest.raises(ValueError, match='variation in D'):
        sp.dml(df, y='y', treat='d', covariates=['x'],
               model='iivm', instrument='z')


def test_iivm_raises_when_instrument_is_constant():
    rng = np.random.default_rng(0)
    n = 500
    Z = np.ones(n, dtype=float)  # no instrument variation
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 2.0 * D + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'x': rng.normal(0, 1, n)})
    with pytest.raises(ValueError, match='variation in Z'):
        sp.dml(df, y='y', treat='d', covariates=['x'],
               model='iivm', instrument='z')


# ---------------------------------------------------------------------
# Fix #6 — Proximal first-stage F only for k_w == 1
# ---------------------------------------------------------------------

def test_proximal_first_stage_F_is_none_for_multi_W():
    """
    With multiple endogenous W, the summed-RSS 'F' had no null
    distribution. The fix returns None and emits a RuntimeWarning
    rather than a misleading numeric F.
    """
    rng = np.random.default_rng(7)
    n = 2000
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    # Two Z proxies and two W proxies
    Z1 = 0.9 * U + rng.normal(0, 0.3, n)
    Z2 = 0.9 * U + rng.normal(0, 0.3, n)
    W1 = 0.9 * U + rng.normal(0, 0.3, n)
    W2 = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z1': Z1, 'z2': Z2, 'w1': W1, 'w2': W2})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = sp.proximal(df, y='y', treat='d',
                        proxy_z=['z1', 'z2'], proxy_w=['w1', 'w2'])
    assert r.model_info['first_stage_F'] is None
    # Expect a warning about the unimplemented multi-W F
    assert any('Cragg-Donald' in str(w.message)
               or 'k_w' in str(w.message) for w in caught)


def test_proximal_first_stage_F_reported_for_single_W():
    """Regression guard for k_w == 1 still receiving a numeric F."""
    rng = np.random.default_rng(7)
    n = 1500
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.9 * U + rng.normal(0, 0.3, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})
    r = sp.proximal(df, y='y', treat='d', proxy_z=['z'], proxy_w=['w'])
    assert r.model_info['first_stage_F'] is not None
    assert r.model_info['first_stage_F'] > 10  # strong proxies
