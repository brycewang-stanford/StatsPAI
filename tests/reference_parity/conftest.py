"""Shared deterministic DGPs for reference parity tests.

All DGPs here are fully deterministic given a seed and return
dataframes with ``attrs['true_effect']`` recording the known truth
so recovery tests can verify bias is within Monte-Carlo tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# DID DGPs
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def did_2x2_data():
    """Classic 2-group, 2-period DID with true ATT = 2.0.

    N=2000 units (1000 treated, 1000 control), 2 periods.
    Treatment effect is homogeneous, so CS/SA/Wooldridge must
    all recover the same ATT.  Large N keeps us far from the
    3-SE boundary so recovery tests don't flake on sampling draws.
    """
    rng = np.random.default_rng(2024)
    n_units = 2000
    rows = []
    for i in range(n_units):
        treated = 1 if i < n_units // 2 else 0
        ui = rng.normal(scale=0.4)
        for t in [0, 1]:
            y = 1.0 + 0.3 * t + 0.5 * treated + 2.0 * treated * t + ui + rng.normal(scale=0.5)
            rows.append({'i': i, 't': t, 'treated': treated,
                         'post': t, 'd': treated * t,
                         'g': 1 if treated else 0, 'y': y})
    df = pd.DataFrame(rows)
    df.attrs['true_effect'] = 2.0
    return df


@pytest.fixture(scope='session')
def did_staggered_homogeneous():
    """Staggered DID, homogeneous effect = 1.5 (cohort-invariant, time-invariant).

    600 units, 8 periods, 3 treatment cohorts + never-treated.
    Because effect is homogeneous, TWFE, CS, SA, Wooldridge must agree
    (no staggering bias).
    """
    rng = np.random.default_rng(314)
    n_units = 600
    cohorts = [3, 5, 7, 0]
    rows = []
    for i in range(n_units):
        g = cohorts[i % 4]
        ui = rng.normal(scale=0.5)
        for t in range(1, 9):
            post = 1 if (g > 0 and t >= g) else 0
            y = 0.2 * t + 1.5 * post + ui + rng.normal(scale=0.8)
            rows.append({'i': i, 't': t, 'g': g, 'post': post, 'y': y})
    df = pd.DataFrame(rows)
    df.attrs['true_effect'] = 1.5
    return df


@pytest.fixture(scope='session')
def did_staggered_heterogeneous():
    """Staggered DID, heterogeneous effect (later cohorts = larger effect).

    TWFE will be biased (contamination from already-treated controls);
    CS, SA, Wooldridge should be unbiased.

    Population simple ATT (unit-uniform weights) = 1.6
    (cohort-3 effect=1.0, cohort-5 effect=1.5, cohort-7 effect=2.0,
     same number of units per cohort -> simple average at horizon h=0
     is (1+1.5+2)/3 = 1.5; weighted by post-periods per cohort).
    """
    rng = np.random.default_rng(42)
    n_units = 600
    cohorts = [3, 5, 7, 0]
    cohort_effects = {3: 1.0, 5: 1.5, 7: 2.0, 0: 0.0}
    rows = []
    for i in range(n_units):
        g = cohorts[i % 4]
        te = cohort_effects[g]
        ui = rng.normal(scale=0.5)
        for t in range(1, 9):
            post = 1 if (g > 0 and t >= g) else 0
            y = 0.2 * t + te * post + ui + rng.normal(scale=0.8)
            rows.append({'i': i, 't': t, 'g': g, 'post': post, 'y': y,
                         'cohort_effect': te})
    df = pd.DataFrame(rows)
    df.attrs['cohort_effects'] = cohort_effects
    return df


# ---------------------------------------------------------------------------
# RD DGPs
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def rd_sharp_data():
    """Sharp RD with known ATE at cutoff = 1.0.

    Running variable x ~ Uniform(-1, 1), cutoff = 0.
    m_0(x) = 2 + 3x + x^2 (left)
    m_1(x) = 3 + 3x + x^2 (right) -> jump = 1.0
    N = 2000.
    """
    rng = np.random.default_rng(2025)
    n = 2000
    x = rng.uniform(-1, 1, n)
    d = (x >= 0).astype(int)
    y = 2 + 3*x + x**2 + 1.0 * d + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({'y': y, 'x': x, 'd': d})
    df.attrs['true_effect'] = 1.0
    df.attrs['cutoff'] = 0.0
    return df


@pytest.fixture(scope='session')
def rd_fuzzy_data():
    """Fuzzy RD with known LATE at cutoff = 0.8.

    Takeup probability: 0.1 below cutoff, 0.8 above.
    Compliers have effect = 0.8.
    """
    rng = np.random.default_rng(99)
    n = 3000
    x = rng.uniform(-1, 1, n)
    # Probability of treatment jumps from 0.1 to 0.8 at cutoff
    p = np.where(x >= 0, 0.8, 0.1)
    d = (rng.uniform(0, 1, n) < p).astype(int)
    y = 1.0 + 2*x + 0.8 * d + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame({'y': y, 'x': x, 'd': d})
    df.attrs['true_effect'] = 0.8
    df.attrs['cutoff'] = 0.0
    return df


# ---------------------------------------------------------------------------
# IV DGPs
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def iv_strong_data():
    """Strong-instrument IV with known LATE = 1.5.

    N = 2000.
    Z ~ Bernoulli(0.5), compliance: P(D=1|Z=1) - P(D=1|Z=0) = 0.6.
    Effect on compliers = 1.5.
    """
    rng = np.random.default_rng(77)
    n = 2000
    z = rng.binomial(1, 0.5, n)
    u = rng.normal(size=n)
    # First stage: d = 0.2 + 0.6 z + eps
    d = (0.2 + 0.6 * z + 0.3 * u + rng.normal(scale=0.3, size=n) > 0.5).astype(int)
    # Outcome: y = 1 + 1.5 * d + 0.5 * u + noise (u is confounder)
    y = 1.0 + 1.5 * d + 0.5 * u + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({'y': y, 'd': d, 'z': z})
    df.attrs['true_effect'] = 1.5
    return df


# ---------------------------------------------------------------------------
# Synth DGPs
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def synth_factor_model_data():
    """Factor-model DGP where SCM is exactly unbiased.

    10 control units, 1 treated, 30 periods (20 pre, 10 post).
    True post-treatment effect = -5.0.
    Factor loadings make one convex combination of controls
    exactly match treated unit in expectation.
    """
    rng = np.random.default_rng(1999)
    T, T0 = 30, 20
    n_controls = 10
    # 2 latent factors
    f1 = np.cumsum(rng.normal(scale=0.3, size=T)) + 10
    f2 = np.cumsum(rng.normal(scale=0.3, size=T)) + 5

    # Treated unit: loadings (0.6, 0.4) — matchable by convex combo of controls
    # if there exists weights w such that sum w_j * lambda_j ≈ (0.6, 0.4)
    treated_loadings = np.array([0.6, 0.4])
    # Give control units loadings spanning a simplex around treated
    control_loadings = rng.dirichlet([1, 1], size=n_controls)
    # Ensure at least one combination can match
    control_loadings[0] = np.array([0.8, 0.2])
    control_loadings[1] = np.array([0.4, 0.6])

    rows = []
    for unit in range(n_controls + 1):
        is_treated = (unit == n_controls)
        lam = treated_loadings if is_treated else control_loadings[unit]
        y_base = lam[0] * f1 + lam[1] * f2
        noise = rng.normal(scale=0.2, size=T)
        y = y_base + noise
        if is_treated:
            # Post-period effect = -5
            y[T0:] += -5.0
        for t in range(T):
            rows.append({
                'unit': unit,
                'year': 2000 + t,
                'y': y[t],
                'treat': 1 if (is_treated and t >= T0) else 0,
                'is_treated_unit': int(is_treated),
            })
    df = pd.DataFrame(rows)
    df.attrs['true_effect'] = -5.0
    df.attrs['treatment_year'] = 2020
    df.attrs['treated_unit'] = n_controls
    return df


# ---------------------------------------------------------------------------
# Matching DGPs
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def matching_cia_data():
    """Selection-on-observables DGP with known ATT = 2.0.

    N = 3000 (1500 treated, 1500 control).
    Propensity score depends on 3 covariates (X1, X2, X3).
    Treatment effect is homogeneous = 2.0.
    Under CIA, matching / IPW / DR should all recover 2.0.
    """
    rng = np.random.default_rng(55)
    n = 3000
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    X3 = rng.binomial(1, 0.4, size=n)
    # Propensity score
    lin = -0.3 + 0.5 * X1 - 0.3 * X2 + 0.4 * X3
    p = 1 / (1 + np.exp(-lin))
    d = (rng.uniform(0, 1, n) < p).astype(int)
    # Outcome
    y0 = 1.0 + 1.5 * X1 - 0.8 * X2 + 0.6 * X3 + rng.normal(scale=0.8, size=n)
    y = y0 + 2.0 * d
    df = pd.DataFrame({'y': y, 'd': d, 'X1': X1, 'X2': X2, 'X3': X3})
    df.attrs['true_effect'] = 2.0
    return df
