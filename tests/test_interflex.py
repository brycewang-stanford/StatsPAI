"""sp.interflex: interaction effects with diagnostics (Hainmueller, Mummolo and Xu 2019).

Numerical parity with R interflex is pinned by Track A module 87; these
tests cover the analytic identities and the API contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.regression.interflex import l_kurtosis, r_density


def _sample(seed: int = 0, n: int = 300, nonlinear: bool = True):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    d = rng.binomial(1, 0.5, n).astype(float)
    me = 1.0 + 0.5 * x - (0.8 * x**2 if nonlinear else 0.0)
    y = 0.5 + 0.3 * x + d * me + rng.normal(0.0, 0.5, n)
    return pd.DataFrame({"y": y, "d": d, "x": x})


def test_linear_matches_ols_interaction_and_delta_method():
    df = _sample(nonlinear=False)
    res = sp.interflex(df, y="y", d="d", x="x", estimator="linear", x_eval=[0.0, 1.0])
    reg = sp.regress("y ~ x + d + d:x", data=df, vce="hc1")
    b = reg.params
    assert res.detail["me"].iloc[0] == pytest.approx(b["d"], rel=1e-10)
    assert res.detail["me"].iloc[1] == pytest.approx(b["d"] + b["d:x"], rel=1e-10)
    assert res.detail["se"].iloc[0] == pytest.approx(reg.std_errors["d"], rel=1e-8)
    assert res.estimand == "ATE"


def test_binning_recovers_bin_specific_effects():
    df = _sample(nonlinear=True)
    res = sp.interflex(df, y="y", d="d", x="x", estimator="binning", nbins=3)
    d = res.detail
    assert list(d["bin"]) == [1, 2, 3]
    assert d["n"].sum() == len(df)
    # Concave profile: middle bin has the largest effect.
    assert d["me"].iloc[1] > d["me"].iloc[0] and d["me"].iloc[1] > d["me"].iloc[2]
    tests = res.model_info["tests"]
    assert tests["p_wald"] < 0.01 and tests["p_lr"] < 0.01
    assert 0.0 < tests["x_lkurtosis"] < 0.3


def test_linear_restriction_not_rejected_when_true():
    df = _sample(nonlinear=False, seed=4)
    res = sp.interflex(df, y="y", d="d", x="x", estimator="binning", nbins=3)
    assert res.model_info["tests"]["p_wald"] > 0.05


def test_kernel_tracks_the_true_nonlinear_profile():
    df = _sample(nonlinear=True, n=800, seed=2)
    grid = [-1.0, 0.0, 1.0]
    res = sp.interflex(df, y="y", d="d", x="x", estimator="kernel", bw=0.5, x_eval=grid)
    truth = 1.0 + 0.5 * np.array(grid) - 0.8 * np.array(grid) ** 2
    assert np.abs(res.detail["me"].to_numpy() - truth).max() < 0.35
    assert res.se is None


def test_kernel_bootstrap_is_seeded():
    df = _sample(n=150)
    a = sp.interflex(
        df,
        y="y",
        d="d",
        x="x",
        estimator="kernel",
        bw=0.8,
        neval=4,
        vce="bootstrap",
        n_boot=20,
        seed=3,
    )
    b = sp.interflex(
        df,
        y="y",
        d="d",
        x="x",
        estimator="kernel",
        bw=0.8,
        neval=4,
        vce="bootstrap",
        n_boot=20,
        seed=3,
    )
    assert a.se == b.se and a.se > 0
    assert "ci_lower" in a.detail.columns


def test_r_density_integrates_to_one_and_is_symmetric_for_symmetric_data():
    x = np.concatenate([np.linspace(-2, 2, 101)])
    xg, yg, bw = r_density(x)
    assert bw > 0
    assert np.trapz(yg, xg) == pytest.approx(1.0, abs=2e-3)
    assert np.allclose(yg, yg[::-1], atol=1e-10)


def test_l_kurtosis_reference_values():
    # Uniform: tau4 = 0; exact for large n up to sampling error.
    x = np.linspace(0, 1, 2001)
    assert abs(l_kurtosis(x)) < 1e-3
    # Normal: tau4 ≈ 0.1226.
    rng = np.random.default_rng(0)
    assert abs(l_kurtosis(rng.normal(size=200000)) - 0.1226) < 0.01


def test_input_validation():
    df = _sample()
    with pytest.raises(ValueError):
        sp.interflex(df, y="y", d="d", x="x", estimator="kernel")
    with pytest.raises(ValueError):
        sp.interflex(df, y="y", d="d", x="x", estimator="loess")
    df2 = df.copy()
    df2["d"] = df2["d"] + 1  # coded 1/2
    with pytest.raises(ValueError):
        sp.interflex(df2, y="y", d="d", x="x")


def test_registry_and_citation():
    df = _sample()
    res = sp.interflex(df, y="y", d="d", x="x")
    assert "hainmueller2019much" in res.cite()
    assert sp.describe_function("interflex")["name"] == "interflex"


def test_fixed_kernel_and_stata_wald_conventions():
    df = _sample(nonlinear=True, seed=5)
    df["z"] = np.random.default_rng(1).normal(size=len(df))
    adaptive = sp.interflex(
        df, y="y", d="d", x="x", estimator="kernel", bw=0.7, neval=4
    )
    fixed = sp.interflex(
        df, y="y", d="d", x="x", estimator="kernel", bw=0.7, neval=4, adaptive=False
    )
    assert not np.allclose(adaptive.detail["me"], fixed.detail["me"])
    assert fixed.model_info["adaptive"] is False
    r_conv = sp.interflex(df, y="y", d="d", x="x", z=["z"], estimator="binning")
    st_conv = sp.interflex(
        df,
        y="y",
        d="d",
        x="x",
        z=["z"],
        estimator="binning",
        wald_full_moderate=False,
        wald_test="F",
    )
    assert r_conv.model_info["tests"]["df"] == st_conv.model_info["tests"]["df"] + 4
    assert st_conv.model_info["tests"]["wald_test"] == "F"
    with pytest.raises(ValueError):
        sp.interflex(df, y="y", d="d", x="x", estimator="binning", wald_test="t")


def test_interflex_plot_returns_a_figure():
    import matplotlib

    matplotlib.use("Agg")
    df = _sample()
    res = sp.interflex(df, y="y", d="d", x="x", estimator="binning")
    fig = sp.interflex_plot(res)
    assert fig is not None and len(fig.axes) >= 1
    kern = sp.interflex(df, y="y", d="d", x="x", estimator="kernel", bw=0.8, neval=6)
    assert sp.interflex_plot(kern, show_hist=False) is not None
