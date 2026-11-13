"""The standard error of ``sp.sbw``.

Balancing weights make the weighted control mean insensitive to the part
of the outcome that is linear in the balanced covariates, so the sampling
error of the effect comes from what is left: the residuals of the outcome
on those covariates. Through 1.43.1 ``sp.sbw`` measured each arm's spread
around its weighted mean instead, which also counts the variation that
balancing removed. In the SE calibration screen
(``tests/reliability/se_screen/causal/weighting.py``) that standard error
was about 1.5 times the sampling standard deviation of the ATT. The
default is now the linearized (calibration) variance; ``vce='naive'``
keeps the old one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

X = ["x1", "x2"]


def _data(seed: int = 0, n: int = 600, noise: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x1 + 0.5 * x2))).astype(int)
    tau = 1 + 0.5 * x1 + rng.normal(scale=0.7, size=n)
    y = 2 + 3 * x1 - x2 + tau * d + noise * rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "tau": tau})


def test_exactly_linear_control_outcome_leaves_only_the_treated_spread():
    """Controls on a plane: their residuals vanish and the ATT's variance
    is var(Y_treated - plane) / n_treated, here the variance of the
    individual effects over the number of treated."""
    df = _data(noise=0.0)
    r = sp.sbw(df, "d", X, y="y")
    tau_treated = df.loc[df["d"] == 1, "tau"].to_numpy()
    expected = np.std(tau_treated, ddof=1) / np.sqrt(len(tau_treated))
    assert r.se == pytest.approx(expected, rel=1e-8)
    # and the point estimate is the treated units' own average effect, up
    # to the balance tolerance times the slopes (0.02 sd per covariate)
    assert abs(r.estimate - tau_treated.mean()) < 0.02 * (3 + 1) * 1.1


@pytest.mark.parametrize("estimand", ["att", "atc", "ate"])
def test_linearized_variance_is_the_stated_formula(estimand):
    df = _data(seed=3)
    r = sp.sbw(df, "d", X, y="y", estimand=estimand)
    w = np.asarray(r.weights, dtype=float)
    Y, T = df["y"].to_numpy(), df["d"].to_numpy() == 1
    D = np.column_stack([np.ones(len(df)), df[X].to_numpy()])

    def fit(mask):
        ww = w[mask] / w[mask].sum()
        A = D[mask].T @ (ww[:, None] * D[mask])
        beta = np.linalg.solve(A, D[mask].T @ (ww * Y[mask]))
        return beta, ww, Y[mask] - D[mask] @ beta

    if estimand == "att":
        b0, w0, e0 = fit(~T)
        var = np.var(Y[T] - D[T] @ b0, ddof=1) / T.sum() + np.sum(w0**2 * e0**2)
    elif estimand == "atc":
        b1, w1, e1 = fit(T)
        var = np.var(Y[~T] - D[~T] @ b1, ddof=1) / (~T).sum() + np.sum(w1**2 * e1**2)
    else:
        b1, w1, e1 = fit(T)
        b0, w0, e0 = fit(~T)
        var = (
            np.sum(w1**2 * e1**2)
            + np.sum(w0**2 * e0**2)
            + np.var(D @ (b1 - b0), ddof=1) / len(df)
        )
    assert r.se == pytest.approx(np.sqrt(var), rel=1e-10)
    assert r.model_info["vce"] == "linearized"
    lo, hi = r.ci
    assert (hi - lo) / 2 == pytest.approx(1.959963984540054 * r.se, rel=1e-10)


def test_naive_variance_is_the_pre_change_number():
    """``vce='naive'`` is the fixed-weights variance around the arm means."""
    df = _data(seed=5)
    r = sp.sbw(df, "d", X, y="y", vce="naive")
    w = np.asarray(r.weights, dtype=float)
    Y, T = df["y"].to_numpy(), df["d"].to_numpy() == 1
    w_c = w[~T]
    mu_c = np.sum(w_c * Y[~T]) / w_c.sum()
    var = np.var(Y[T], ddof=1) / T.sum() + np.sum(w_c**2 * (Y[~T] - mu_c) ** 2)
    assert r.se == pytest.approx(np.sqrt(var), rel=1e-12)
    # same weights, same estimate under either variance
    lin = sp.sbw(df, "d", X, y="y")
    assert lin.estimate == r.estimate
    np.testing.assert_array_equal(lin.weights, r.weights)
    # the covariates explain most of the outcome here, so the naive
    # standard error is far above the linearized one
    assert r.se > 1.5 * lin.se


def test_squares_of_a_binary_covariate_do_not_break_the_regression():
    """``include_squares`` on a 0/1 column balances the same column twice."""
    df = _data(seed=7)
    df["b"] = (df["x2"] > 0).astype(float)
    r = sp.sbw(df, "d", ["x1", "b"], y="y", include_squares=True, delta=0.05)
    plain = sp.sbw(df, "d", ["x1", "b"], y="y", delta=0.05)
    assert np.isfinite(r.se) and r.se > 0
    # adding x1^2 (and the redundant b^2) changes the fit only a little
    assert r.se == pytest.approx(plain.se, rel=0.15)


def test_unknown_vce_is_refused():
    with pytest.raises(ValueError, match="vce"):
        sp.sbw(_data(), "d", X, y="y", vce="robust")
