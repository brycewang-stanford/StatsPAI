"""``sp.aipw(propensity=)``: the treatment mechanism of a randomised trial."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility


def _trial(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    stratum = (x > 0).astype(int)
    p = np.where(stratum == 1, 0.7, 0.4)
    a = rng.binomial(1, p)
    y = 1.5 * a + x + 0.5 * a * x + rng.normal(size=n)
    return pd.DataFrame({"x": x, "a": a, "y": y, "p": p})


def test_matches_the_formula_with_the_design_probability() -> None:
    df = _trial()
    res = sp.aipw(
        df, y="y", treat="a", covariates=["x"], propensity="p", cross_fit=False
    )
    X = np.column_stack([np.ones(len(df)), df["x"]])
    y, a, p = (df[c].to_numpy(float) for c in ("y", "a", "p"))
    mu = {}
    for arm in (0, 1):
        beta = np.linalg.lstsq(X[a == arm], y[a == arm], rcond=None)[0]
        mu[arm] = X @ beta
    psi = mu[1] - mu[0] + a * (y - mu[1]) / p - (1 - a) * (y - mu[0]) / (1 - p)
    np.testing.assert_allclose(res.estimate, psi.mean(), rtol=1e-12)
    np.testing.assert_allclose(res.se, psi.std(ddof=1) / np.sqrt(len(df)), rtol=1e-12)
    assert res.model_info["propensity"] == "known"
    assert res.model_info["n_propensity_clipped"] == 0


def test_scalar_probability_equals_a_constant_column() -> None:
    df = _trial().assign(half=0.5)
    kw = dict(y="y", treat="a", covariates=["x"], seed=3)
    one = sp.aipw(df, propensity=0.5, **kw)
    col = sp.aipw(df, propensity="half", **kw)
    assert one.estimate == col.estimate and one.se == col.se


def test_unbiased_whatever_the_outcome_regression() -> None:
    # With the true propensity the estimator is consistent even though a
    # linear outcome regression is wrong here; the mean over replications
    # sits on the truth.
    truth = 1.5 + 0.5 * 0.0 + 0.8  # E[1.5 + 0.5 x + 0.8 x^2] with x ~ N(0, 1)
    est = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        n = 400
        x = rng.normal(size=n)
        p = 1 / (1 + np.exp(-x))
        a = rng.binomial(1, p)
        y = a * (1.5 + 0.5 * x + 0.8 * x**2) + np.sin(2 * x) + rng.normal(size=n)
        df = pd.DataFrame({"x": x, "a": a, "y": y, "p": p})
        est.append(
            sp.aipw(
                df, y="y", treat="a", covariates=["x"], propensity="p", cross_fit=False
            ).estimate
        )
    est = np.asarray(est)
    # Four Monte Carlo standard errors.
    assert abs(est.mean() - truth) < 4 * est.std(ddof=1) / np.sqrt(len(est))


def test_invalid_use_is_refused() -> None:
    df = _trial()
    kw = dict(y="y", treat="a", covariates=["x"])
    with pytest.raises(MethodIncompatibility, match="strictly between"):
        sp.aipw(df, propensity=1.0, **kw)
    with pytest.raises(MethodIncompatibility, match="sandwich"):
        sp.aipw(df, propensity=0.5, cross_fit=False, se_method="sandwich", **kw)
