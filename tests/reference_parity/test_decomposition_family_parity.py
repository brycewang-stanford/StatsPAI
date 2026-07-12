"""Analytical parity: distributional / gap-decomposition estimators.

* ``sp.rifreg`` at the ``mean`` statistic reduces to OLS bit-for-bit (the
  recentered influence function of the mean is the outcome itself).
* ``sp.shapley_inequality`` factor contributions sum to the total index
  (Shapley additivity), and a factor with no explanatory role gets ~0.
* ``sp.fairlie`` (nonlinear Blinder-Oaxaca) satisfies ``gap = explained +
  unexplained`` and attributes a pure composition gap to the explained part.
* ``sp.ffl_decompose`` (Firpo-Fortin-Lemieux) satisfies the full aggregate
  identity ``gap = composition + structure + reweight_error + spec_error`` and
  ``gap = stat_a - stat_b``.

Analytical evidence tier (exact identities and known-truth recovery on
deterministic DGPs).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


# --------------------------------------------------------------------------
# RIF regression at the mean == OLS
# --------------------------------------------------------------------------
def test_rifreg_mean_reduces_to_ols():
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x": x})
    res = sp.rifreg("y ~ x", df, statistic="mean")
    design = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    assert float(res.params["Intercept"]) == pytest.approx(beta[0], abs=1e-8)
    assert float(res.params["x"]) == pytest.approx(beta[1], abs=1e-8)


# --------------------------------------------------------------------------
# Shapley inequality decomposition
# --------------------------------------------------------------------------
def test_shapley_inequality_shares_sum_to_total():
    # Shapley shares reconstruct the inequality of the *predicted* outcome
    # I(y_hat), which equals the total index I(y) up to the regression residual.
    # A near-perfect fit (tiny noise) drives that residual to zero, exposing the
    # exact additivity of the Shapley values.
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    latent = 2.0 * x1 + 0.0 * x2 + rng.normal(0, 0.01, n)  # x2 irrelevant, R^2~1
    y = latent - latent.min() + 1.0
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    res = sp.shapley_inequality(df, y="y", x=["x1", "x2"], index="theil_t")
    contrib = res.shapley["contribution"].to_numpy()
    assert contrib.sum() == pytest.approx(float(res.total), rel=1e-4)


def test_shapley_inequality_irrelevant_factor_near_zero():
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    latent = 2.0 * x1 + 0.0 * x2 + rng.normal(0, 0.01, n)
    y = latent - latent.min() + 1.0
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    res = sp.shapley_inequality(df, y="y", x=["x1", "x2"], index="theil_t")
    share = dict(zip(res.shapley["variable"], res.shapley["pct_of_total"]))
    assert share["x2"] < 1.0  # < 1% of the total
    assert share["x1"] > 95.0


# --------------------------------------------------------------------------
# Fairlie nonlinear decomposition
# --------------------------------------------------------------------------
def test_fairlie_identity_and_pure_composition_gap():
    rng = np.random.default_rng(0)
    n = 3000
    g = rng.integers(0, 2, n)
    x = rng.normal(0.5 * g, 1.0, n)  # groups differ only in X location
    p = 1.0 / (1.0 + np.exp(-(-0.5 + 1.2 * x)))
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({"y": y, "g": g, "x": x})
    res = sp.fairlie(df, y="y", group="g", x=["x"], seed=0, n_sim=200)
    # Aggregate identity.
    assert float(res.gap) == pytest.approx(
        float(res.explained) + float(res.unexplained), abs=1e-8
    )
    # Same coefficients across groups -> the gap is essentially all "explained".
    assert abs(float(res.unexplained)) < 0.2 * abs(float(res.gap))


# --------------------------------------------------------------------------
# FFL (Firpo-Fortin-Lemieux) decomposition
# --------------------------------------------------------------------------
def test_ffl_full_aggregate_identity():
    rng = np.random.default_rng(0)
    n = 3000
    g = rng.integers(0, 2, n)
    x = rng.normal(0.5 * g, 1.0, n)
    y = 1.0 + 2.0 * x + 0.5 * g + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "g": g, "x": x})
    res = sp.ffl_decompose(df, y="y", group="g", x=["x"], stat="mean")
    assert float(res.gap) == pytest.approx(
        float(res.stat_a) - float(res.stat_b), abs=1e-8
    )
    total = (
        float(res.composition)
        + float(res.structure)
        + float(res.reweight_error)
        + float(res.spec_error)
    )
    assert total == pytest.approx(float(res.gap), abs=1e-8)
