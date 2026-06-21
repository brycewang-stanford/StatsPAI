"""Tests for the PyArrow table path behind ``sp.fast.*_polars`` adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

pa = pytest.importorskip("pyarrow")


def _panel_pandas(seed=0):
    rng = np.random.default_rng(seed)
    n_units, n_periods = 50, 16
    i = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(n_periods), n_units)
    n = i.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = rng.normal(0, 0.4, size=n_units)[i]
    g = rng.normal(0, 0.25, size=n_periods)[t]
    eta = np.clip(0.35 + 0.25 * x1 - 0.15 * x2 + a + g, -8, 8)
    y = rng.poisson(np.exp(eta)).astype(np.int64)
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "i": i.astype(np.int32),
            "t": t.astype(np.int32),
        }
    )


def _arrow_table(pdf: pd.DataFrame):
    return pa.Table.from_pandas(pdf, preserve_index=False)


def test_demean_polars_arrow_table_matches_pandas():
    pdf = _panel_pandas(seed=1)
    arrow = _arrow_table(pdf)

    expected, _ = sp.fast.demean(
        pdf[["x1", "x2"]].to_numpy(),
        pdf[["i", "t"]].to_numpy(),
        drop_singletons=False,
    )
    observed, info = sp.fast.demean_polars(
        arrow,
        X_cols=["x1", "x2"],
        fe_cols=["i", "t"],
        drop_singletons=False,
    )

    assert info.converged == [True, True]
    assert np.allclose(observed, expected, atol=1e-12)


def test_demean_polars_arrow_accepts_scalar_column_names():
    pdf = _panel_pandas(seed=2)
    arrow = _arrow_table(pdf)

    observed, _ = sp.fast.demean_polars(
        arrow,
        X_cols="x1",
        fe_cols="i",
        drop_singletons=False,
    )
    expected, _ = sp.fast.demean(
        pdf[["x1"]].to_numpy(),
        pdf[["i"]].to_numpy(),
        drop_singletons=False,
    )

    assert observed.shape == (len(pdf), 1)
    assert np.allclose(observed, expected, atol=1e-12)


def test_demean_polars_arrow_validates_columns():
    pdf = _panel_pandas(seed=3)
    arrow = _arrow_table(pdf)

    with pytest.raises(MethodIncompatibility, match="X_cols"):
        sp.fast.demean_polars(arrow, X_cols=[], fe_cols=["i"])
    with pytest.raises(MethodIncompatibility, match="missing columns"):
        sp.fast.demean_polars(arrow, X_cols=["missing"], fe_cols=["i"])
    with pytest.raises(MethodIncompatibility, match="fe_cols"):
        sp.fast.demean_polars(arrow, X_cols=["x1"], fe_cols=[])


def test_fepois_polars_arrow_matches_pandas():
    pdf = _panel_pandas(seed=4)
    arrow = _arrow_table(pdf)

    fit_pd = sp.fast.fepois("y ~ x1 + x2 | i + t", pdf)
    fit_arrow = sp.fast.fepois_polars(arrow, "y ~ x1 + x2 | i + t")

    for name in ("x1", "x2"):
        assert fit_arrow.coef()[name] == pytest.approx(fit_pd.coef()[name], abs=1e-12)
        assert fit_arrow.se()[name] == pytest.approx(fit_pd.se()[name], abs=1e-12)


def test_fepois_polars_arrow_missing_column_raises_taxonomy_error():
    pdf = _panel_pandas(seed=5)
    arrow = _arrow_table(pdf)

    with pytest.raises(MethodIncompatibility, match="missing columns"):
        sp.fast.fepois_polars(arrow, "y ~ nope | i + t")
