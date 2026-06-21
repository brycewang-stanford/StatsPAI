"""Coverage tests for statspai.rd.locrand (local randomization inference).

Exercises rdrandinf across test statistics (diffmeans / ksmirnov / ranksum /
all), polynomial adjustment, covariate partialling, fuzzy Wald, plus
rdwinselect, rdsensitivity, and rdrbounds (Rosenbaum bounds). Real synthetic
RD data; permutation p-values in [0,1] and structural properties asserted.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statspai as sp
from statspai.core.results import CausalResult


def _make_sharp(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    Z = rng.normal(0, 1, n)
    Y = 0.5 * X + 3.0 * (X >= 0) + 0.3 * Z + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": Y, "x": X, "z": Z, "z2": rng.normal(0, 1, n)})


def _make_fuzzy(n=2500, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    prob = 0.15 + 0.7 * (X >= 0)
    D = (rng.uniform(0, 1, n) < prob).astype(float)
    Y = 0.5 * X + 2.0 * D + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": Y, "x": X, "d": D})


@pytest.mark.parametrize("stat", ["diffmeans", "ksmirnov", "ranksum", "all"])
def test_rdrandinf_statistics(stat):
    df = _make_sharp()
    res = sp.rdrandinf(
        df, y="y", x="x", c=0, wl=-0.3, wr=0.3, statistic=stat, n_perms=150, seed=1
    )
    assert isinstance(res, CausalResult)
    assert 0.0 <= res.pvalue <= 1.0


def test_rdrandinf_polynomial_adjustment():
    df = _make_sharp()
    res = sp.rdrandinf(df, y="y", x="x", c=0, wl=-0.4, wr=0.4, p=1, n_perms=150, seed=2)
    assert 0.0 <= res.pvalue <= 1.0


def test_rdrandinf_with_covariates():
    df = _make_sharp()
    res = sp.rdrandinf(
        df, y="y", x="x", c=0, wl=-0.3, wr=0.3, covs=["z", "z2"], n_perms=150, seed=3
    )
    assert 0.0 <= res.pvalue <= 1.0


def test_rdrandinf_fuzzy():
    df = _make_fuzzy()
    res = sp.rdrandinf(
        df, y="y", x="x", c=0, wl=-0.3, wr=0.3, fuzzy="d", n_perms=150, seed=4
    )
    assert 0.0 <= res.pvalue <= 1.0
    assert np.isfinite(res.estimate)


def test_rdrandinf_window_required():
    df = _make_sharp()
    # wl/wr are mandatory: omitting them raises, pointing to rdwinselect.
    with pytest.raises(ValueError, match="Window bounds"):
        sp.rdrandinf(df, y="y", x="x", c=0, covs=["z"], n_perms=120, seed=5)


def test_rdwinselect():
    df = _make_sharp()
    out = sp.rdwinselect(df, x="x", c=0, covs=["z", "z2"], nwindows=6, seed=7)
    assert isinstance(out, pd.DataFrame)
    assert len(out) >= 3
    np.testing.assert_allclose(
        out[["window_left", "window_right", "n_left", "n_right", "p_value"]]
        .head(3)
        .to_numpy(),
        np.array(
            [
                [-0.083237, 0.083237, 98.0, 77.0, 0.394],
                [-0.266358, 0.266358, 285.0, 250.0, 0.384],
                [-0.449480, 0.449480, 449.0, 410.0, 0.680],
            ]
        ),
        atol=5e-7,
    )


def test_rdsensitivity():
    df = _make_sharp()
    out = sp.rdsensitivity(df, y="y", x="x", c=0, nwindows=5, n_perms=100, seed=8)
    assert isinstance(out, pd.DataFrame)
    assert "estimate" in out.columns
    np.testing.assert_allclose(
        out[["window", "estimate", "se", "pvalue"]].head(3).to_numpy(),
        np.array(
            [
                [0.099884, 3.013447, 0.066839, 0.0],
                [0.312139, 3.117404, 0.034816, 0.0],
                [0.524393, 3.250576, 0.027918, 0.0],
            ]
        ),
        atol=5e-7,
    )
    plt.close("all")


def test_rdsensitivity_explicit_wlist():
    df = _make_sharp()
    out = sp.rdsensitivity(
        df, y="y", x="x", c=0, wlist=[0.1, 0.2, 0.3], n_perms=80, seed=9
    )
    assert len(out) >= 1
    plt.close("all")


def test_rdrbounds():
    df = _make_sharp()
    out = sp.rdrbounds(
        df,
        y="y",
        x="x",
        c=0,
        wl=-0.3,
        wr=0.3,
        gamma_list=[1.0, 1.5, 2.0],
        n_perms=100,
        seed=10,
    )
    assert isinstance(out, pd.DataFrame)
    assert "gamma" in out.columns
    # p-value bounds should be valid probabilities
    pcols = [c for c in out.columns if "pval" in c.lower() or "p_" in c.lower()]
    for col in pcols:
        vals = out[col].dropna()
        assert ((vals >= -1e-9) & (vals <= 1 + 1e-9)).all()
    np.testing.assert_allclose(
        out[["gamma", "pvalue_upper", "pvalue_lower"]].to_numpy(),
        np.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        atol=1e-12,
    )
