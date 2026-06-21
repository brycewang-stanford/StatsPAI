"""Coverage tests for assorted statspai.rd modules.

Covers hte (rdhte / rdbwhte / rdhte_lincom), honest_ci (rd_honest),
bias_aware (rd_bias_aware_fuzzy), dashboard (rd_dashboard / rd_compare /
rd_robustness_table), rd_flex, rd_discrete, rdmulti (rdmc / rdms), rdml
(rd_forest / rd_boost / rd_lasso / rd_cate_summary), the small frontier
estimators, and the unified RD dispatcher (sp.rd.fit). Real synthetic data.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statspai as sp
from statspai.core.results import CausalResult

_HAS_SK = importlib.util.find_spec("sklearn") is not None


def _make_sharp(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    Z = rng.normal(0, 1, n)
    Y = 0.5 * X + 3.0 * (X >= 0) + 0.3 * Z + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": Y, "x": X, "z": Z, "z2": rng.normal(0, 1, n)})


def _make_hte(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    Z = rng.normal(0, 1, n)
    tau = 2.0 + 1.5 * Z
    Y = 0.5 * X + tau * (X >= 0) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": Y, "x": X, "z": Z})


def _make_fuzzy(n=2500, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    prob = 0.15 + 0.7 * (X >= 0)
    D = (rng.uniform(0, 1, n) < prob).astype(float)
    Y = 0.5 * X + 2.0 * D + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": Y, "x": X, "d": D})


# ---------------------------------------------------------------- hte


def test_rdhte_and_lincom_and_bwhte():
    df = _make_hte()
    res = sp.rdhte(df, y="y", x="x", z="z", c=0, n_eval=8)
    assert isinstance(res, CausalResult)
    assert abs(res.estimate - 2.0) < 1.0
    h = sp.rdbwhte(df, y="y", x="x", z="z", c=0)
    assert h > 0
    n = len(res.detail)
    lc = sp.rdhte_lincom(res, weights=np.ones(n) / n)
    assert "estimate" in lc


def test_rdhte_multiple_covariates():
    df = _make_hte()
    df["z2"] = np.random.default_rng(0).normal(0, 1, len(df))
    res = sp.rdhte(df, y="y", x="x", z=["z", "z2"], c=0, n_eval=5)
    assert res.se > 0


# ---------------------------------------------------------------- honest_ci


def test_rd_honest_mse_and_flci_and_manual_M():
    df = _make_sharp()
    r_mse = sp.rd_honest(df, y="y", x="x", c=0, opt_criterion="mse")
    assert r_mse.model_info["honest_ci"][0] < r_mse.model_info["honest_ci"][1]
    r_flci = sp.rd_honest(df, y="y", x="x", c=0, opt_criterion="flci")
    assert r_flci.se >= 0
    r_M = sp.rd_honest(df, y="y", x="x", c=0, M=5.0)
    assert np.isfinite(r_M.estimate)
    np.testing.assert_allclose(
        [
            r_mse.estimate,
            r_mse.se,
            r_mse.ci[0],
            r_mse.ci[1],
            r_mse.model_info["bias_bound"],
        ],
        [
            3.075221537406432,
            0.06848787439334914,
            2.667657858640401,
            3.4827852161724624,
            0.1366649557886817,
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [
            r_flci.estimate,
            r_flci.se,
            r_flci.ci[0],
            r_flci.ci[1],
            r_flci.model_info["bias_bound"],
        ],
        [
            3.09668701925409,
            0.09801637617962299,
            2.8453998509496996,
            3.3479741875584805,
            0.02958930054859988,
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [r_M.estimate, r_M.se, r_M.ci[0], r_M.ci[1], r_M.model_info["bias_bound"]],
        [
            3.075221537406432,
            0.06848787439334914,
            2.8305431453530514,
            3.319899929459812,
            0.05522231243235646,
        ],
        atol=1e-12,
    )


@pytest.mark.parametrize("kernel", ["triangular", "epanechnikov", "uniform"])
def test_rd_honest_kernels(kernel):
    df = _make_sharp()
    res = sp.rd_honest(df, y="y", x="x", c=0, kernel=kernel, h=0.4)
    assert np.isfinite(res.estimate)


# ---------------------------------------------------------------- bias_aware


def test_rd_bias_aware_fuzzy():
    df = _make_fuzzy()
    res = sp.rd_bias_aware_fuzzy(df, y="y", x="x", fuzzy="d", c=0, n_grid=81)
    assert isinstance(res, CausalResult)
    assert res.ci[0] < res.ci[1]


def test_rd_bias_aware_fuzzy_manual_M_and_cluster():
    df = _make_fuzzy()
    df["g"] = (np.arange(len(df)) // 25).astype(int)
    res = sp.rd_bias_aware_fuzzy(
        df, y="y", x="x", fuzzy="d", c=0, M_y=2.0, M_d=2.0, cluster="g", n_grid=81
    )
    assert np.isfinite(res.estimate)


# ---------------------------------------------------------------- dashboard


def test_rd_dashboard():
    df = _make_sharp()
    out = sp.rd_dashboard(df, y="y", x="x", c=0, covs=["z"])
    assert out is not None
    plt.close("all")


def test_rd_compare_default_and_custom():
    df = _make_sharp()
    out = sp.rd_compare(df, y="y", x="x", c=0)
    assert isinstance(out, pd.DataFrame)
    assert len(out) >= 1
    plt.close("all")


def test_rd_robustness_table():
    df = _make_sharp()
    out = sp.rd_robustness_table(
        df,
        y="y",
        x="x",
        c=0,
        kernels=("triangular",),
        bwselects=("mserd",),
        polynomials=(1,),
    )
    assert isinstance(out, pd.DataFrame)
    assert len(out) >= 1


# ---------------------------------------------------------------- rd_flex / discrete


@pytest.mark.parametrize("learner", ["boost", "forest", "ridge", "lasso"])
def test_rd_flex_learners(learner):
    if not _HAS_SK:
        pytest.skip("sklearn not installed")
    df = _make_sharp()
    res = sp.rd_flex(
        df, y="y", x="x", c=0, W=["z", "z2"], learner=learner, n_folds=3, random_state=0
    )
    assert isinstance(res, CausalResult)
    assert res.se > 0


def test_rd_discrete_bsd_and_bm():
    rng = np.random.default_rng(1)
    n = 1500
    X = rng.integers(-8, 9, n).astype(float)
    Y = 0.3 * X + 3.0 * (X >= 0) + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": Y, "x": X})
    r_bsd = sp.rd_discrete(df, y="y", x="x", c=0, method="bsd")
    assert r_bsd.ci[0] < r_bsd.ci[1]
    r_bm = sp.rd_discrete(df, y="y", x="x", c=0, method="bm")
    assert np.isfinite(r_bm.estimate)


def test_rd_discrete_errors():
    df = pd.DataFrame({"y": [1.0, 2, 3], "x": [-1.0, 0, 1]})
    with pytest.raises(ValueError, match="not found"):
        sp.rd_discrete(df, y="nope", x="x", c=0)
    with pytest.raises(ValueError, match="method"):
        sp.rd_discrete(df, y="y", x="x", c=0, method="bad")


# ---------------------------------------------------------------- rdmulti


def test_rdmc_multi_cutoff():
    rng = np.random.default_rng(2)
    n = 3000
    X = rng.uniform(-2, 3, n)
    Y = 0.5 * X + 2.0 * (X >= 0) + 1.0 * (X >= 1) + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": Y, "x": X})
    res = sp.rdmc(df, y="y", x="x", cutoffs=[0.0, 1.0])
    assert np.isfinite(res.pooled_estimate)
    assert res.pooled_se > 0


def test_rdms_multi_score():
    rng = np.random.default_rng(3)
    n = 2000
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    treat = ((x1 >= 0) & (x2 >= 0)).astype(int)
    Y = 0.3 * x1 + 0.3 * x2 + 2.0 * treat + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"y": Y, "x1": x1, "x2": x2})
    res = sp.rdms(df, y="y", x1="x1", x2="x2", cutoff1=0, cutoff2=0, bandwidth=0.6)
    assert res is not None


# ---------------------------------------------------------------- rdml


def test_rd_lasso():
    df = _make_sharp()
    res = sp.rd_lasso(df, y="y", x="x", c=0, covs=["z", "z2"])
    assert abs(res.estimate - 3.0) < 1.0


def test_rd_forest_and_boost():
    if not _HAS_SK:
        pytest.skip("sklearn not installed")
    df = _make_hte()
    rf = sp.rd_forest(df, y="y", x="x", c=0, covs=["z"], n_trees=40, seed=0)
    assert rf.model_info is not None
    rb = sp.rd_boost(df, y="y", x="x", c=0, covs=["z"], n_estimators=40, seed=0)
    assert np.isfinite(rb.estimate)


def test_rd_cate_summary():
    if not _HAS_SK:
        pytest.skip("sklearn not installed")
    df = _make_hte()
    out = sp.rd_cate_summary(
        df, y="y", x="x", c=0, covs=["z"], methods=["lasso"], seed=0
    )
    assert isinstance(out, dict)


# ---------------------------------------------------------------- dispatcher


def test_rd_fit_dispatcher_passthrough():
    df = _make_sharp()
    for method in ["rdrobust", "honest", "rkd"]:
        res = sp.rd.fit(df, y="y", x="x", c=0, method=method)
        assert hasattr(res, "estimate")
    # lasso requires candidate covariates
    res = sp.rd.fit(df, y="y", x="x", c=0, method="lasso", covs=["z", "z2"])
    assert hasattr(res, "estimate")


def test_rd_fit_dispatcher_aliases_and_errors():
    df = _make_sharp()
    with pytest.raises(ValueError, match="Unknown method"):
        sp.rd.fit(df, y="y", x="x", c=0, method="not_a_method")
    with pytest.raises(TypeError):
        sp.rd.fit(df, y="y", x="x", c=0, method=123)


def test_rd_fit_bias_aware_requires_fuzzy():
    df = _make_fuzzy()
    with pytest.raises(ValueError, match="requires fuzzy"):
        sp.rd.fit(df, y="y", x="x", c=0, method="bias_aware_fuzzy")
    res = sp.rd.fit(df, y="y", x="x", c=0, method="bias_aware_fuzzy", fuzzy="d")
    assert np.isfinite(res.estimate)


def test_rd_fit_rdit_via_dispatcher():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-01", periods=400, freq="D")
    xd = (dates - dates[200]).days.values.astype(float)
    Y = 0.005 * xd + 1.5 * (xd >= 0) + rng.normal(0, 0.4, 400)
    df = pd.DataFrame({"y": Y, "date": dates})
    res = sp.rd.fit(df, y="y", x="date", c="2010-07-20", method="rdit", h=120.0)
    assert np.isfinite(res.estimate)
