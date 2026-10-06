"""Tests for ``sp.bayes_mte`` — Bayesian Marginal Treatment Effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.bayes import BayesianMTEResult, bayes_mte

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------


def _mte_dgp(n, mte_fn, seed):
    """Latent-index selection + linear outcome where MTE is a function
    of the latent index."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    U = rng.normal(size=n)
    linpred = 0.8 * Z + U
    D = (linpred > 0).astype(float)
    # Bring linpred into propensity-ish scale [0, 1] for the MTE call
    u_obs = 1 / (1 + np.exp(-linpred))
    tau = np.array([mte_fn(u) for u in u_obs])
    Y = 1.0 + tau * D + 0.3 * rng.normal(size=n)
    return pd.DataFrame({"y": Y, "d": D, "z": Z})


@pytest.fixture
def flat_mte_data():
    return _mte_dgp(600, mte_fn=lambda u: 1.5, seed=601)


def _index_model_dgp(n, mte_coefs, seed, strength=1.5):
    """Heckman-Vytlacil index model: ``D = 1{p(Z) > U_D}`` with
    ``U_D ~ U(0, 1)`` and ``MTE(u) = sum_k c_k u^k``."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    U = rng.uniform(size=n)
    D = (1.0 / (1.0 + np.exp(-strength * Z)) > U).astype(float)
    tau = sum(c * U**k for k, c in enumerate(mte_coefs))
    Y = 1.0 + tau * D + 0.3 * rng.normal(size=n)
    return pd.DataFrame({"y": Y, "d": D, "z": Z})


@pytest.fixture
def monotone_mte_data():
    # MTE(u) = 0.5 + 2 u: units with more resistance to treatment gain more.
    # Local IV identifies the slope from the variation in the propensity
    # only, so it needs a few thousand observations and a strong instrument.
    return _index_model_dgp(4000, (0.5, 2.0), seed=602)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_bayes_mte_returns_expected_result(flat_mte_data):
    r = bayes_mte(
        flat_mte_data,
        y="y",
        treat="d",
        instrument="z",
        poly_u=2,
        draws=300,
        tune=300,
        chains=2,
        progressbar=False,
    )
    assert isinstance(r, BayesianMTEResult)
    assert r.estimand.startswith("ATE")
    assert isinstance(r.mte_curve, pd.DataFrame)
    assert len(r.mte_curve) == 19  # default grid
    assert {"u", "posterior_mean", "hdi_low", "hdi_high"}.issubset(r.mte_curve.columns)
    assert np.isfinite(r.ate)
    assert np.isfinite(r.att)
    assert np.isfinite(r.atu)


def test_bayes_mte_top_level_export():
    assert sp.bayes_mte is bayes_mte
    assert sp.BayesianMTEResult is BayesianMTEResult


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_bayes_mte_flat_mte_curve_is_approximately_flat(flat_mte_data):
    """On a constant-effect DGP the MTE curve's SD across u should
    be much smaller than its mean."""
    r = bayes_mte(
        flat_mte_data,
        y="y",
        treat="d",
        instrument="z",
        poly_u=2,
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_state=31,
    )
    means = r.mte_curve["posterior_mean"].values
    # Constant DGP => across u, MTE should be roughly constant
    assert np.std(means) < 0.5 * np.abs(np.mean(means)), (
        f"Flat DGP: MTE std {np.std(means):.3f} vs mean "
        f"{np.mean(means):.3f} should indicate near-constant curve"
    )


def test_bayes_mte_monotone_mte_recovers_slope(monotone_mte_data):
    """An increasing MTE is recovered: intercept 0.5, slope 2."""
    r = bayes_mte(
        monotone_mte_data,
        y="y",
        treat="d",
        instrument="z",
        poly_u=1,
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_state=32,
    )
    means = r.mte_curve["posterior_mean"].values
    assert means[-1] > means[0]
    b = r.trace.posterior["b_mte"].values.reshape(-1, 2)
    assert (b[:, 1] > 0).mean() > 0.97
    assert abs(b[:, 1].mean() - 2.0) < 4 * b[:, 1].std()
    assert abs(b[:, 0].mean() - 0.5) < 4 * b[:, 0].std()
    # ATE = 0.5 + 2 / 2 = 1.5
    assert r.hdi_lower < 1.5 < r.hdi_upper


# ---------------------------------------------------------------------------
# Configurability
# ---------------------------------------------------------------------------


def test_bayes_mte_custom_u_grid(flat_mte_data):
    grid = np.array([0.1, 0.5, 0.9])
    r = bayes_mte(
        flat_mte_data,
        y="y",
        treat="d",
        instrument="z",
        u_grid=grid,
        poly_u=1,
        draws=300,
        tune=300,
        chains=2,
        progressbar=False,
    )
    assert len(r.mte_curve) == 3
    np.testing.assert_allclose(r.mte_curve["u"].values, grid)


def test_bayes_mte_poly_u_one(flat_mte_data):
    r = bayes_mte(
        flat_mte_data,
        y="y",
        treat="d",
        instrument="z",
        poly_u=1,
        draws=300,
        tune=300,
        chains=2,
        progressbar=False,
    )
    assert r.model_info["poly_u"] == 1


def test_bayes_mte_with_covariates(flat_mte_data):
    df = flat_mte_data.copy()
    rng = np.random.default_rng(33)
    df["age"] = rng.normal(size=len(df))
    r = bayes_mte(
        df,
        y="y",
        treat="d",
        instrument="z",
        covariates=["age"],
        poly_u=2,
        draws=300,
        tune=300,
        chains=2,
        progressbar=False,
    )
    assert r.model_info["covariates"] == ["age"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_bayes_mte_missing_column_raises(flat_mte_data):
    with pytest.raises(ValueError, match="not found"):
        bayes_mte(
            flat_mte_data,
            y="nope",
            treat="d",
            instrument="z",
            draws=50,
            tune=50,
            chains=1,
            progressbar=False,
        )


def test_bayes_mte_non_binary_treat_raises(flat_mte_data):
    df = flat_mte_data.copy()
    df["d"] = df["d"] * 2.0
    with pytest.raises(ValueError, match="binary"):
        bayes_mte(
            df,
            y="y",
            treat="d",
            instrument="z",
            draws=50,
            tune=50,
            chains=1,
            progressbar=False,
        )
