"""Coverage campaign — StatsPAI's native HDFE OLS (``panel/feols.py``).

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). ``sp.hdfe_ols`` is StatsPAI's own
high-dimensional-fixed-effects OLS (the ``panel/feols.py`` ``feols`` /
``FEOLSResult``; distinct from the pyfixest-backed ``sp.feols``). This exercises
its calling surface: two-way HDFE, cluster-robust and heteroskedastic SEs,
analytic weights, the no-fixed-effect OLS fallback, the wild-cluster-bootstrap
path, result accessors, and the formula-parse error — plus the ``sp.demean`` /
``sp.absorb_ols`` HDFE primitives.

Assertions are real: the HDFE slope recovers the true effect (=2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n_e, n_t = 60, 8
    N = n_e * n_t
    ent = np.repeat(np.arange(n_e), n_t)
    tm = np.tile(np.arange(n_t), n_e)
    x1 = rng.standard_normal(N)
    x2 = rng.standard_normal(N)
    fe_e = np.repeat(rng.standard_normal(n_e), n_t)
    fe_t = np.tile(rng.standard_normal(n_t), n_e)
    y = 2.0 * x1 - 1.0 * x2 + fe_e + fe_t + rng.standard_normal(N)
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "id": ent,
            "time": tm,
            "w": rng.uniform(0.5, 1.5, N),
        }
    )


def _b(res, name="x1"):
    return float(res.params[name])


def test_hdfe_twoway(df):
    res = sp.hdfe_ols("y ~ x1 + x2 | id + time", df)
    assert abs(_b(res) - 2.0) < 0.5
    assert isinstance(res.summary(), str)
    assert res.std_errors["x1"] > 0
    assert isinstance(repr(res), str)


def test_hdfe_cluster(df):
    res = sp.hdfe_ols("y ~ x1 + x2 | id", df, cluster="id")
    assert abs(_b(res) - 2.0) < 0.5


def test_hdfe_cluster_omits_fe_nested_in_cluster_dof():
    df = sp.dgp_panel(n_units=200, n_periods=10, seed=0)

    res = sp.hdfe_ols("y ~ x | unit + time", df, cluster="unit")

    assert res.dof_fe == 209
    assert res.cluster_info["dof_fe_cluster"] == 10
    assert res.cluster_info["nested_fe"] == ["unit"]
    assert res.std_errors["x"] == pytest.approx(0.02517069037042562, rel=1e-12)


def test_hdfe_cluster_all_nested_fe_keeps_intercept_dof():
    df = sp.dgp_panel(n_units=200, n_periods=10, seed=0)

    res = sp.hdfe_ols("y ~ x | unit", df, cluster="unit")

    assert res.cluster_info["dof_fe_cluster"] == 1
    assert res.cluster_info["nested_fe"] == ["unit"]
    assert res.std_errors["x"] == pytest.approx(0.02680753371253867, rel=1e-12)


def test_hdfe_cluster_keeps_full_dof_when_no_fe_nested():
    df = sp.dgp_panel(n_units=200, n_periods=10, seed=0)
    rng = np.random.default_rng(123)
    df["cl_alt"] = rng.integers(0, 50, len(df))

    res = sp.hdfe_ols("y ~ x | unit + time", df, cluster="cl_alt")

    assert res.cluster_info["dof_fe_cluster"] == res.dof_fe == 209
    assert res.cluster_info["nested_fe"] == []
    assert res.std_errors["x"] == pytest.approx(0.01963832219003795, rel=1e-12)


def test_hdfe_hetero(df):
    res = sp.hdfe_ols("y ~ x1 + x2 | id", df, se_type="hetero")
    assert np.isfinite(_b(res))


def test_hdfe_weights(df):
    res = sp.hdfe_ols("y ~ x1 + x2 | id", df, weights="w")
    assert np.isfinite(_b(res))


def test_hdfe_no_fixed_effects(df):
    # no '|' → the plain-OLS (_ols_no_fe) fallback
    res = sp.hdfe_ols("y ~ x1 + x2", df)
    assert np.isfinite(_b(res))


def test_hdfe_no_fixed_effects_weights_match_manual_wls():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 1.5, 2.6, 1.2, 4.0, 1.7, 6.6],
            "d": [0, 0, 0, 0, 1, 1, 1, 1],
            "post": [0, 1, 0, 1, 0, 1, 0, 1],
            "w": [1.0, 1.0, 8.0, 8.0, 1.0, 1.0, 8.0, 8.0],
        }
    )
    df["_did"] = df["d"] * df["post"]

    res = sp.hdfe_ols("y ~ d + post + _did", df, weights="w")

    X = np.column_stack(
        [
            np.ones(len(df)),
            df[["d", "post", "_did"]].to_numpy(dtype=float),
        ]
    )
    y = df["y"].to_numpy(dtype=float)
    w = df["w"].to_numpy(dtype=float)
    beta = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y * w))

    assert res.params["_did"] == pytest.approx(beta[3], abs=1e-12)
    unweighted = sp.hdfe_ols("y ~ d + post + _did", df)
    assert res.params["_did"] != pytest.approx(unweighted.params["_did"])


def test_hdfe_wild_cluster_bootstrap(df):
    res = sp.hdfe_ols("y ~ x1 | id", df, cluster="id", wild=True, wild_n_boot=199)
    assert res is not None


def test_hdfe_bad_formula_raises(df):
    with pytest.raises((ValueError, KeyError)):
        sp.hdfe_ols("this is not a formula", df)


# ─── HDFE primitives ─────────────────────────────────────────────────────


def test_demean_and_absorb(df):
    x = df[["y", "x1"]].to_numpy(dtype=float)
    fe = df[["id"]]
    out = sp.demean(x, fe=fe)
    arr = out[0] if isinstance(out, tuple) else out
    assert np.asarray(arr).shape[0] == len(df)
    res = sp.absorb_ols(
        df["y"].to_numpy(dtype=float), df[["x1", "x2"]].to_numpy(dtype=float), fe=fe
    )
    assert res is not None
