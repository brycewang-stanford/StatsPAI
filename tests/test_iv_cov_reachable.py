"""Coverage campaign — remaining *reachable* IV input-coercion and validation
branches (1-D arrays, Series/DataFrame name extraction, alternative propensity
models, and cov-type / dimensionality validation errors).

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). These are the input-handling and
error-raising lines reachable by feeding 1-D vectors / pandas objects and by
mis-specifying options — distinct from the genuinely-defensive
``except LinAlgError`` pinv fallbacks (those are ``# pragma: no cover`` in src).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(71)
    n = 500
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    d = 0.8 * z1 + 0.6 * z2 + 0.3 * x + 0.6 * u + 0.5 * rng.standard_normal(n)
    y = 1.0 + 2.0 * d + 0.4 * x + u + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2, "x": x})


@pytest.fixture(scope="module")
def binary_df():
    rng = np.random.default_rng(72)
    n = 800
    z = rng.uniform(-2, 2, n)
    x = rng.standard_normal(n)
    v = rng.standard_normal(n)
    d = ((0.9 * z + 0.3 * x - 0.4 * v) > 0).astype(float)
    y = 1.0 + 1.3 * d + 0.4 * x + 0.5 * v + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "d": d, "z": z, "x": x})


# ─── Kleibergen–Paap validation + Series/array name handling ─────────────


def test_kp_cluster_without_cluster_raises(df):
    with pytest.raises(ValueError, match="cluster"):
        sp.iv.kleibergen_paap_rk(
            endog=df[["d"]], instruments=df[["z1", "z2"]], cov_type="cluster"
        )


def test_kp_unknown_cov_type_raises(df):
    with pytest.raises(ValueError, match="[Uu]nknown cov_type"):
        sp.iv.kleibergen_paap_rk(
            endog=df[["d"]], instruments=df[["z1", "z2"]], cov_type="bogus"
        )


def test_kp_1d_series_inputs(df):
    # 1-D Series endog + 1-D instrument array exercise the reshape/name branches
    res = sp.iv.kleibergen_paap_rk(endog=df["d"], instruments=df["z1"].to_numpy())
    assert np.isfinite(res.rk_f)


# ─── MTE: linear propensity model + 1-D array inputs ─────────────────────


def test_mte_linear_propensity_1d_arrays(binary_df):
    y = binary_df["y"].to_numpy()
    d = binary_df["d"].to_numpy()
    z = binary_df["z"].to_numpy()  # 1-D instrument → reshape branch
    res = sp.iv.mte(y=y, treatment=d, instruments=z, propensity_model="linear")
    assert res is not None


# ─── post_lasso: array inputs, no exog ───────────────────────────────────


def test_post_lasso_array_no_exog(df):
    res = sp.iv.bch_post_lasso_iv(
        y=df["y"].to_numpy(),
        endog=df["d"].to_numpy(),
        instruments=df[["z1", "z2"]],  # DataFrame → column-name branch
    )
    assert res is not None


# ─── JIVE: DataFrame / Series name extraction ────────────────────────────


def test_jive_dataframe_series_names(df):
    res = sp.iv.ujive(
        y=df["y"], endog=df[["d"]], instruments=df[["z1", "z2"]], exog=df[["x"]]
    )
    # endog name should come from the DataFrame column, not a positional default
    assert "d" in [str(i) for i in res.params.index]


# ─── IVMTE: 1-D array inputs ─────────────────────────────────────────────


def test_ivmte_1d_array_inputs(binary_df):
    res = sp.iv.ivmte_bounds(
        y=binary_df["y"].to_numpy(),
        treatment=binary_df["d"].to_numpy(),
        instruments=binary_df["z"].to_numpy(),  # 1-D → reshape branch
        bounds_outcome=(-5.0, 8.0),
    )
    assert res is not None
