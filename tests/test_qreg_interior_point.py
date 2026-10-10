"""The interior-point quantile-regression solver and its vertex finish.

``_qreg_frisch_newton`` may only return the exact minimiser of the check
loss. The tests state that directly (interpolation of ``k`` rows, the
subgradient condition, no better point nearby), compare with the HiGHS
solution the function used to return, and check that designs without a
unique minimiser are handed back to HiGHS.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.regression import quantile as Q


def _design(n, k, seed, kind="normal"):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    err = rng.standard_t(2, n) if kind == "t2" else rng.normal(size=n)
    if kind == "hetero":
        err = err * (1 + 0.5 * np.abs(X[:, -1]))
    return X @ rng.normal(size=k) + err, X


def _check_loss(Y, X, beta, tau):
    r = Y - X @ beta
    return float(np.sum(r * (tau - (r < 0))))


def _highs(Y, X, tau, monkeypatch):
    monkeypatch.setattr(Q, "_FN_MIN_N", 10**12)
    try:
        return Q._qreg_fit(Y, X, tau)
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize("kind", ["normal", "hetero", "t2"])
@pytest.mark.parametrize("tau", [0.05, 0.5, 0.9])
def test_solution_is_the_unique_minimiser(kind, tau, monkeypatch):
    Y, X = _design(6000, 4, 3, kind)
    n, k = X.shape
    beta = Q._qreg_frisch_newton(Y, X, tau)
    assert beta is not None

    # a vertex: exactly k observations are interpolated
    r = Y - X @ beta
    h = np.argsort(np.abs(r))[:k]
    np.testing.assert_allclose(r[h], 0.0, atol=1e-9 * np.abs(Y).max())
    assert np.abs(r).min() < np.sort(np.abs(r))[k] * 1e-6

    # the subgradient condition of Koenker and Bassett, strictly
    rest = np.setdiff1d(np.arange(n), h)
    g = np.linalg.solve(X[h].T, X[rest].T @ (tau - (r[rest] < 0)))
    assert np.all(g > -tau) and np.all(g < 1 - tau)

    # no nearby point does better
    rng = np.random.default_rng(0)
    base = _check_loss(Y, X, beta, tau)
    for _ in range(50):
        assert _check_loss(Y, X, beta + 1e-4 * rng.normal(size=k), tau) > base

    # and it is the vertex HiGHS finds; 1e-9 is the conditioning of the
    # k x k basis solve, both answers solve the same system
    np.testing.assert_allclose(beta, _highs(Y, X, tau, monkeypatch), rtol=1e-9)


def test_design_without_a_unique_minimiser_goes_to_highs(monkeypatch):
    # The median of y on a dummy alone: with an even count in a cell any
    # value between the two middle observations minimises the loss.
    rng = np.random.default_rng(1)
    n = 6000
    d = (np.arange(n) % 2).astype(float)
    X = np.column_stack([np.ones(n), d])
    Y = 1 + d + rng.normal(size=n)
    assert Q._qreg_frisch_newton(Y, X, 0.5) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = Q._qreg_fit(Y, X, 0.5)
    np.testing.assert_array_equal(got, _highs(Y, X, 0.5, monkeypatch))


def test_qreg_reports_the_same_numbers_through_either_solver(monkeypatch):
    Y, X = _design(8000, 3, 5, "hetero")
    df = pd.DataFrame({"y": Y, "a": X[:, 1], "b": X[:, 2]})
    fast = sp.qreg(df, "y ~ a + b", quantile=0.25)
    monkeypatch.setattr(Q, "_FN_MIN_N", 10**12)
    slow = sp.qreg(df, "y ~ a + b", quantile=0.25)
    np.testing.assert_allclose(fast.params.values, slow.params.values, rtol=1e-9)
    np.testing.assert_allclose(
        fast.std_errors.values, slow.std_errors.values, rtol=1e-7
    )


def test_cost_is_linear_in_n():
    # 200,000 rows: about 0.2 s per fit, against 10 s for the LP solver.
    Y, X = _design(200_000, 4, 2)
    start = time.perf_counter()
    beta = Q._qreg_frisch_newton(Y, X, 0.5)
    assert beta is not None
    assert time.perf_counter() - start < 15.0
