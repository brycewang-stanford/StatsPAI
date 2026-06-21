"""Validate the canonical core/_vcov sandwich wrappers against statsmodels.

This pins the NEW primitive (currently unwired) so that the future
one-estimator-at-a-time migration off the ~18 hand-rolled sandwich copies has
a trusted reference to migrate onto.
"""

import numpy as np
import pytest

from statspai.core._vcov import (
    cluster_robust_vcov,
    hc_vcov,
    sandwich_vcov,
    cluster_correction_factor,
)


@pytest.fixture
def ols_data():
    rng = np.random.default_rng(0)
    n = 400
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta = np.array([1.0, 2.0, -0.5])
    y = X @ beta + rng.normal(size=n) * (1.0 + 0.5 * np.abs(X[:, 1]))  # heterosk.
    clusters = rng.integers(0, 25, size=n)
    bhat = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ bhat
    return X, y, resid, clusters


def test_hc1_matches_statsmodels(ols_data):
    sm = pytest.importorskip("statsmodels.api")
    X, y, resid, _ = ols_data
    res = sm.OLS(y, X).fit()
    expected = res.cov_HC1
    got = hc_vcov(X, resid, hc_type="hc1")
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_hc0_matches_statsmodels(ols_data):
    sm = pytest.importorskip("statsmodels.api")
    X, y, resid, _ = ols_data
    res = sm.OLS(y, X).fit()
    np.testing.assert_allclose(
        hc_vcov(X, resid, hc_type="hc0"), res.cov_HC0, rtol=1e-10, atol=1e-12
    )


def test_cluster_stata_matches_statsmodels(ols_data):
    sm = pytest.importorskip("statsmodels.api")
    X, y, resid, clusters = ols_data
    res = sm.OLS(y, X).fit()
    rob = res.get_robustcov_results(
        cov_type="cluster", groups=clusters, use_correction=True
    )
    expected = rob.cov_params()
    got = cluster_robust_vcov(X, resid, clusters, correction="stata")
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-10)


def test_correction_factor_relationships():
    # cgm = none * G/(G-1);  stata = cgm * (N-1)/(N-K)
    G, N, K = 25, 400, 3
    assert cluster_correction_factor(G, N, K, "none") == 1.0
    assert cluster_correction_factor(G, N, K, "cgm") == pytest.approx(G / (G - 1))
    assert cluster_correction_factor(G, N, K, "stata") == pytest.approx(
        (G / (G - 1)) * ((N - 1) / (N - K))
    )
    assert cluster_correction_factor(G, N, K, "stacked") == pytest.approx(
        (G / (G - 1)) * (N / (N - K))
    )


def test_dof_adjust_override(ols_data):
    X, y, resid, clusters = ols_data
    base = cluster_robust_vcov(X, resid, clusters, correction="none")
    scaled = cluster_robust_vcov(X, resid, clusters, dof_adjust=2.0)
    np.testing.assert_allclose(scaled, 2.0 * base, rtol=1e-12)


def test_single_cluster_does_not_explode(ols_data):
    X, y, resid, _ = ols_data
    one = np.zeros(len(resid), dtype=int)
    V = cluster_robust_vcov(X, resid, one, correction="stata")
    assert np.all(np.isfinite(V))


def test_cluster_rejects_missing_labels(ols_data):
    X, y, resid, clusters = ols_data
    labels = clusters.astype(float)
    labels[0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        cluster_robust_vcov(X, resid, labels, correction="stata")

    labels_obj = np.array([f"g{x}" for x in clusters], dtype=object)
    labels_obj[0] = None
    with pytest.raises(ValueError, match="missing values"):
        cluster_robust_vcov(X, resid, labels_obj, correction="stata")


def test_unknown_correction_raises():
    with pytest.raises(ValueError, match="Unknown cluster correction"):
        cluster_correction_factor(10, 100, 3, "bogus")


# --- generic sandwich_vcov (covers MLE bread + precomputed scores) ----------


def test_sandwich_vcov_cluster_equals_cluster_robust_vcov(ols_data):
    X, y, resid, clusters = ols_data
    XtX_inv = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]
    got = sandwich_vcov(XtX_inv, scores, clusters=clusters, correction="liang_zeger")
    expected = cluster_robust_vcov(X, resid, clusters, correction="liang_zeger")
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_sandwich_vcov_hc_equals_hc0(ols_data):
    X, y, resid, _ = ols_data
    XtX_inv = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]
    got = sandwich_vcov(XtX_inv, scores, correction="none")
    expected = XtX_inv @ (scores.T @ scores) @ XtX_inv
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_sandwich_vcov_hc1(ols_data):
    X, y, resid, _ = ols_data
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]
    got = sandwich_vcov(XtX_inv, scores, correction="hc1")
    expected = (n / (n - k)) * XtX_inv @ (scores.T @ scores) @ XtX_inv
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_sandwich_vcov_cr1_matches_cr1_formula(ols_data):
    X, y, resid, clusters = ols_data
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]
    G = int(np.unique(clusters).shape[0])
    meat = np.zeros((k, k))
    for g in np.unique(clusters):
        s = scores[clusters == g].sum(axis=0)
        meat += np.outer(s, s)
    scale = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    expected = scale * XtX_inv @ meat @ XtX_inv
    got = sandwich_vcov(XtX_inv, scores, clusters=clusters, correction="cr1")
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-13)


def test_sandwich_vcov_mle_bread(ols_data):
    # Generic bread (not (X'X)^{-1}) — e.g. inv(Hessian) in an MLE model.
    X, y, resid, _ = ols_data
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X + 0.3 * np.eye(k))  # arbitrary SPD bread
    scores = X * resid[:, None]
    got = sandwich_vcov(bread, scores, correction="none")
    expected = bread @ (scores.T @ scores) @ bread
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_sandwich_vcov_hc_rejects_cluster_correction(ols_data):
    X, y, resid, _ = ols_data
    XtX_inv = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]
    with pytest.raises(ValueError, match="Without clusters"):
        sandwich_vcov(XtX_inv, scores, correction="stata")
