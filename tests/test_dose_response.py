"""Tests for the continuous-treatment dose-response family (``sp.dose_response``,
``sp.vcnet``).

These estimators target the causal dose-response curve E[Y(t)] for a
continuous exposure under unconfoundedness — a common public-health setting
(dose, BMI, pollutant concentration). The module previously had no dedicated
test file; these checks pin its statistical behaviour on simple DGPs with a
known sign and shape.
"""
import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _linear_dgp(n=1000, slope=1.5, seed=0):
    """Y depends linearly on a continuous treatment t, confounded by x.

    t = 0.7*x + noise; y = slope*t + 0.5*x + noise. The GPS adjustment in
    ``sp.dose_response`` should remove the x-confounding and recover a
    positive dose-response.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    t = 0.7 * x + rng.normal(0, 1, n)
    y = slope * t + 0.5 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"t": t, "x": x, "y": y})


def _null_dgp(n=1000, seed=0):
    """Treatment t is independent of the outcome y (no dose-response)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    t = rng.normal(0, 1, n)
    y = 0.5 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"t": t, "x": x, "y": y})


def test_dose_response_detects_positive_effect():
    df = _linear_dgp(slope=1.5, seed=0)
    res = sp.dose_response(
        df, y="y", treat="t", covariates=["x"],
        n_bootstrap=100, n_dose_points=15, random_state=0,
    )
    # E[Y(t75)] - E[Y(t25)] must be positive and clearly bounded away from 0.
    assert res.estimate > 0
    assert res.ci[0] < res.estimate < res.ci[1]
    assert res.ci[0] > 0          # CI excludes the null for a strong effect
    assert res.se > 0
    assert res.n_obs == len(df)
    assert "Dose-Response" in res.summary()


def test_dose_response_null_effect_covers_zero():
    df = _null_dgp(seed=1)
    res = sp.dose_response(
        df, y="y", treat="t", covariates=["x"],
        n_bootstrap=100, n_dose_points=12, random_state=0,
    )
    # No true dose-response: the IQR contrast should be near zero and its
    # confidence interval should cover zero.
    assert abs(res.estimate) < 0.5
    assert res.ci[0] < 0 < res.ci[1]


def test_dose_response_is_reproducible_with_random_state():
    df = _linear_dgp(seed=2)
    a = sp.dose_response(df, y="y", treat="t", covariates=["x"],
                         n_bootstrap=50, random_state=7)
    b = sp.dose_response(df, y="y", treat="t", covariates=["x"],
                         n_bootstrap=50, random_state=7)
    assert a.estimate == pytest.approx(b.estimate, rel=1e-9, abs=1e-9)


def test_vcnet_curve_is_increasing_for_positive_slope():
    rng = np.random.default_rng(3)
    n = 1000
    x = rng.normal(0, 1, n)
    t = rng.normal(0, 1, n)
    y = 1.5 * t + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"t": t, "x": x, "y": y})

    res = sp.vcnet(df, y="y", treatment="t", covariates=["x"],
                   n_bootstrap=20, random_state=0)
    t_grid = np.asarray(res.t_grid, dtype=float)
    mu_hat = np.asarray(res.mu_hat, dtype=float)

    assert t_grid.shape == mu_hat.shape
    assert len(t_grid) > 1
    # Monotone-increasing dose-response: fitted curve tracks the grid.
    assert np.corrcoef(t_grid, mu_hat)[0, 1] > 0.8
    # Pointwise CI brackets the point estimate.
    ci_lo = np.asarray(res.ci_lo, dtype=float)
    ci_hi = np.asarray(res.ci_hi, dtype=float)
    assert np.all(ci_lo <= mu_hat + 1e-8)
    assert np.all(mu_hat <= ci_hi + 1e-8)
