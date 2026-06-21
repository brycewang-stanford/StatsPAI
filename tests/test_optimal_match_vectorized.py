"""Equivalence + correctness guard for the vectorised Mahalanobis distance
matrix in ``optimal_match``.

The per-treated-unit Python loop was replaced with ``scipy.cdist`` (~3-5x
faster, lower memory). This test pins the new path to the exact reference
formula ``sqrt((x-y)' VI (x-y))`` to machine precision so the speed-up cannot
silently change which controls get matched.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.matching.optimal import _distance_matrix
import statspai as sp


def _reference_mahalanobis(X_treat, X_ctrl):
    X_all = np.vstack([X_treat, X_ctrl])
    cov = np.cov(X_all, rowvar=False) + 1e-8 * np.eye(X_all.shape[1])
    cov_inv = np.linalg.inv(cov)
    D = np.empty((X_treat.shape[0], X_ctrl.shape[0]))
    for i in range(X_treat.shape[0]):
        diff = X_ctrl - X_treat[i]
        D[i] = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))
    return D


@pytest.mark.parametrize("k", [2, 5, 8])
def test_vectorized_mahalanobis_matches_reference(k):
    rng = np.random.RandomState(k)
    X_treat = rng.randn(120, k)
    X_ctrl = rng.randn(300, k)
    fast = _distance_matrix(X_treat, X_ctrl, "mahalanobis")
    ref = _reference_mahalanobis(X_treat, X_ctrl)
    assert fast.shape == (120, 300)
    np.testing.assert_allclose(fast, ref, rtol=0, atol=1e-9)


def test_euclidean_branch_unchanged():
    rng = np.random.RandomState(0)
    X_treat = rng.randn(50, 3)
    X_ctrl = rng.randn(80, 3)
    D = _distance_matrix(X_treat, X_ctrl, "euclidean")
    # Spot-check against the direct norm definition.
    expected = np.linalg.norm(X_treat[0] - X_ctrl[5])
    assert D[0, 5] == pytest.approx(expected)


def test_optimal_match_recovers_att():
    rng = np.random.RandomState(3)
    n = 2000
    x1, x2, x3 = rng.randn(n), rng.randn(n), rng.randn(n)
    ps = 1.0 / (1.0 + np.exp(-(0.7 * x1 + 0.4 * x2 - 1.6)))
    t = (rng.uniform(size=n) < ps).astype(int)
    y = 2.0 * t + 1.2 * x1 + x2 + 0.5 * x3 + rng.randn(n)
    df = pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2, "x3": x3})
    r = sp.optimal_match(
        df,
        treatment="t",
        outcome="y",
        covariates=["x1", "x2", "x3"],
        metric="mahalanobis",
    )
    assert r.ate == pytest.approx(2.0, abs=0.4)
    assert r.n_matched == int(t.sum())
