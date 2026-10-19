"""``sp.cuped``: identities of the CUPED estimator and known-truth Monte Carlo.

There is no canonical reference implementation of CUPED; the estimator is
defined by a formula (Deng et al. 2013), so the tests check that formula
exactly and the two properties that justify it on a known DGP: unbiasedness
and a variance reduction of rho^2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _data(n=1000, rho=0.8, effect=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = rng.permutation(np.repeat([0, 1], n // 2))
    y = effect * d + rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x": x, "x2": rng.normal(size=n)})


def test_estimate_is_the_adjusted_difference_in_means():
    df = _data()
    r = sp.cuped(df, "y", "d", ["x"])
    theta = np.cov(df.x, df.y)[0, 1] / np.var(df.x, ddof=1)
    assert r.model_info["theta"]["x"] == pytest.approx(theta, rel=1e-12)
    m = df.groupby("d").mean()
    expected = (m.y[1] - m.y[0]) - theta * (m.x[1] - m.x[0])
    assert r.estimate == pytest.approx(expected, rel=1e-12)
    adj = df.y - theta * (df.x - df.x.mean())
    dim = sp.difference_in_means(df.assign(a=adj), "a", "d")
    assert r.se == pytest.approx(dim.se, rel=1e-12)
    assert r.ci == pytest.approx(dim.ci, rel=1e-12)


def test_zero_theta_is_the_difference_in_means():
    df = _data()
    r = sp.cuped(df, "y", "d", ["x"], theta=[0.0])
    dim = sp.difference_in_means(df, "y", "d")
    assert r.estimate == pytest.approx(dim.estimate, rel=1e-12)
    assert r.se == pytest.approx(dim.se, rel=1e-12)
    assert r.model_info["variance_reduction"] == pytest.approx(0.0, abs=1e-12)


def test_multiple_covariates_use_the_pooled_projection():
    df = _data()
    r = sp.cuped(df, "y", "d", ["x", "x2"])
    X = df[["x", "x2"]].to_numpy()
    Xc = X - X.mean(0)
    theta = np.linalg.lstsq(Xc, df.y - df.y.mean(), rcond=None)[0]
    np.testing.assert_allclose(list(r.model_info["theta"].values()), theta, rtol=1e-12)


def test_lin_method_is_lm_lin():
    df = _data()
    a = sp.cuped(df, "y", "d", ["x"], method="lin")
    b = sp.lm_lin(df, "y", "d", ["x"])
    assert a.estimate == b.estimate and a.se == b.se


def test_unbiased_with_variance_reduction_rho_squared():
    # Known truth: effect 0.3, corr(x, y | d) = 0.8, so the variance of the
    # estimate falls by rho^2 = 0.64. 400 replications: the MC SE of the mean
    # estimate is about 0.0024 and of the variance ratio about 0.02.
    est, ratio = [], []
    for s in range(400):
        r = sp.cuped(_data(n=600, seed=100 + s), "y", "d", ["x"])
        est.append(r.estimate)
        ratio.append(1 - r.model_info["variance_reduction"])
    est = np.array(est)
    assert abs(est.mean() - 0.3) < 4 * est.std(ddof=1) / np.sqrt(len(est))
    assert np.mean(ratio) == pytest.approx(1 - 0.64, abs=0.02)
    # The reported SE matches the sampling SD of the estimate.
    assert est.std(ddof=1) == pytest.approx(np.sqrt(0.36 * 4 / 600), rel=0.1)


def test_cupac_captures_nonlinear_signal():
    # y depends on x^2 only; with x symmetric, Cov(x, y) = 0, so linear CUPED
    # removes nothing while an out-of-fold nonlinear prediction removes most
    # of the variance (Var(x^2) = 2 against noise variance 0.25).
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.normal(size=n)
    d = rng.permutation(np.repeat([0, 1], n // 2))
    df = pd.DataFrame({"x": x, "d": d, "y": 0.2 * d + x**2 + 0.5 * rng.normal(size=n)})
    lin = sp.cuped(df, "y", "d", ["x"])
    cup = sp.cuped(df, "y", "d", ["x"], method="cupac", seed=0)
    assert lin.model_info["variance_reduction"] < 0.02
    assert cup.model_info["variance_reduction"] > 0.6
    assert abs(cup.estimate - 0.2) < 4 * cup.se


def test_invalid_inputs():
    df = _data()
    with pytest.raises(MethodIncompatibility):
        sp.cuped(df, "y", "d", ["x"], method="nope")
    with pytest.raises(MethodIncompatibility):
        sp.cuped(df, "y", "d", [])
    with pytest.raises(MethodIncompatibility):
        sp.cuped(df, "y", "d", ["x"], theta=[1.0, 2.0])
    with pytest.raises(MethodIncompatibility):
        sp.cuped(df.assign(g=["a"] * len(df)), "y", "d", ["g"])
