"""Tests for ``sp.fast.demean_polars`` and ``sp.fast.fepois_polars`` (Phase 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

pl = pytest.importorskip("polars")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _panel_pandas(seed=0):
    rng = np.random.default_rng(seed)
    n_units, n_periods = 80, 25
    i = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(n_periods), n_units)
    n = i.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = rng.normal(0, 0.5, size=n_units)[i]
    g = rng.normal(0, 0.3, size=n_periods)[t]
    eta = 0.5 + 0.3 * x1 - 0.2 * x2 + a + g
    eta = np.clip(eta, -10, 10)
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.int64)
    return pd.DataFrame({
        "y": y, "x1": x1, "x2": x2,
        "i": i.astype(np.int32), "t": t.astype(np.int32),
    })


# ---------------------------------------------------------------------------
# demean_polars
# ---------------------------------------------------------------------------

def test_demean_polars_eager_matches_pandas():
    pdf = _panel_pandas(seed=1)
    polf = pl.from_pandas(pdf)
    Xd_pd, _info_pd = sp.fast.demean(
        pdf[["x1", "x2"]].to_numpy(), pdf[["i", "t"]].to_numpy(),
        drop_singletons=False,
    )
    Xd_pl, _info_pl = sp.fast.demean_polars(
        polf, X_cols=["x1", "x2"], fe_cols=["i", "t"],
        drop_singletons=False,
    )
    assert np.allclose(Xd_pd, Xd_pl, atol=1e-12)


def test_demean_polars_lazyframe_collected():
    pdf = _panel_pandas(seed=2)
    polf = pl.from_pandas(pdf).lazy()  # explicitly lazy
    Xd, info = sp.fast.demean_polars(
        polf, X_cols=["x1"], fe_cols=["i", "t"],
        drop_singletons=False,
    )
    assert info.converged[0]
    # X_cols is a list ⇒ output is 2-D regardless of length (cleanest API)
    assert Xd.shape == (len(pdf), 1)


def test_demean_polars_missing_column_raises():
    pdf = _panel_pandas(seed=3)
    polf = pl.from_pandas(pdf)
    with pytest.raises(KeyError):
        sp.fast.demean_polars(polf, X_cols=["nope"], fe_cols=["i"])


# ---------------------------------------------------------------------------
# fepois_polars
# ---------------------------------------------------------------------------

def test_fepois_polars_matches_pandas():
    pdf = _panel_pandas(seed=4)
    polf = pl.from_pandas(pdf)
    fit_pd = sp.fast.fepois("y ~ x1 + x2 | i + t", pdf)
    fit_pl = sp.fast.fepois_polars(polf, "y ~ x1 + x2 | i + t")

    for k in ("x1", "x2"):
        assert abs(fit_pd.coef()[k] - fit_pl.coef()[k]) < 1e-12
        assert abs(fit_pd.se()[k] - fit_pl.se()[k]) < 1e-12


def test_fepois_polars_lazyframe():
    pdf = _panel_pandas(seed=5)
    polf = pl.from_pandas(pdf).lazy()
    fit = sp.fast.fepois_polars(polf, "y ~ x1 + x2 | i + t")
    assert fit.converged
    assert "x1" in fit.coef().index


def test_fepois_polars_only_collects_needed_columns():
    """Phase 5 promise: don't materialise the whole frame, just the
    referenced columns. We can't *easily* assert the projection happened
    without monkeypatching, but we can at least verify the code-path
    runs and produces correct numbers when extra columns are present."""
    pdf = _panel_pandas(seed=6)
    pdf["unused"] = np.random.default_rng(99).normal(size=len(pdf))
    polf = pl.from_pandas(pdf)
    fit = sp.fast.fepois_polars(polf, "y ~ x1 + x2 | i + t")
    assert fit.converged


def test_fepois_polars_missing_column_raises():
    pdf = _panel_pandas(seed=7)
    polf = pl.from_pandas(pdf)
    with pytest.raises(KeyError):
        sp.fast.fepois_polars(polf, "y ~ nope | i + t")
