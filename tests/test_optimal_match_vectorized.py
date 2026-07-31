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

import statspai as sp
from statspai.matching.optimal import _distance_matrix


def _reference_mahalanobis(X_treat, X_ctrl):
    """Loop reference for the cdist path.

    Uses the pooled *within-group* covariance (Rubin 1980), which is the
    metric ``optimal_match`` builds. Before v1.21 both this helper and the
    implementation used the total sample covariance; that was a different
    metric, inflated along the direction the group means differ in.
    """
    n1, n0 = len(X_treat), len(X_ctrl)
    s1 = np.atleast_2d(np.cov(X_treat, rowvar=False))
    s0 = np.atleast_2d(np.cov(X_ctrl, rowvar=False))
    cov = ((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2)
    cov = cov + 1e-8 * np.eye(cov.shape[0])
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
