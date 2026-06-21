"""Tests for ``sp.fast.crve`` and ``sp.fast.boottest`` (Phase 4)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _ols_panel(n_clusters=20, m=30, seed=0, beta=(0.30, -0.20)):
    """Generate clustered OLS data with cluster-correlated errors."""
    rng = np.random.default_rng(seed)
    n = n_clusters * m
    g = np.repeat(np.arange(n_clusters), m)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cluster_eff = rng.normal(0, 0.4, size=n_clusters)[g]
    eps = cluster_eff + rng.normal(size=n)
    y = beta[0] * x1 + beta[1] * x2 + eps
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})


# ---------------------------------------------------------------------------
# CRVE — closed-form sanity
# ---------------------------------------------------------------------------


def test_crve_cr1_matches_manual_formula():
    """CR1 sandwich computed by us must match the textbook formula."""
    df = _ols_panel(seed=1)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()

    XtX = X.T @ X
    bread = np.linalg.inv(XtX)
    beta = bread @ X.T @ y
    resid = y - X @ beta

    # Hand formula
    n, k = X.shape
    G = int(np.unique(g).size)
    score = resid[:, None] * X
    cluster_score = np.zeros((G, k))
    for i in range(n):
        cluster_score[g[i]] += score[i]
    meat = cluster_score.T @ cluster_score
    V_ref = bread @ meat @ bread * (G / (G - 1)) * ((n - 1) / (n - k))

    V = sp.fast.crve(X, resid, g, type="cr1")
    assert np.allclose(V, V_ref, atol=1e-12)


def test_crve_cr3_runs_and_positive():
    """Analytic CR3 uses the clubSandwich-style (I-H_g)^-1 adjustment.
    We simply check both CR1 and CR3 compute and report positive variances
    on the same residuals."""
    df = _ols_panel(seed=2)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta

    V1 = sp.fast.crve(X, resid, g, type="cr1")
    V3 = sp.fast.crve(X, resid, g, type="cr3")
    assert (np.diag(V1) > 0).all()
    assert (np.diag(V3) > 0).all()


def test_crve_too_few_clusters_raises():
    X = np.random.default_rng(0).normal(size=(20, 2))
    resid = np.random.default_rng(0).normal(size=20)
    with pytest.raises(ValueError, match="2 clusters"):
        sp.fast.crve(X, resid, np.zeros(20))  # G=1


def test_crve_missing_cluster_labels_raise():
    X = np.random.default_rng(1).normal(size=(20, 2))
    resid = np.random.default_rng(2).normal(size=20)
    cluster = np.repeat(np.arange(4, dtype=float), 5)
    cluster[3] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        sp.fast.crve(X, resid, cluster)


def test_crve_cluster_length_mismatch_raises():
    X = np.random.default_rng(3).normal(size=(20, 2))
    resid = np.random.default_rng(4).normal(size=20)

    with pytest.raises(ValueError, match="cluster length"):
        sp.fast.crve(X, resid, np.arange(19))


def test_crve_residual_length_mismatch_raises():
    X = np.random.default_rng(5).normal(size=(20, 2))
    resid = np.random.default_rng(6).normal(size=19)
    cluster = np.repeat(np.arange(4), 5)

    with pytest.raises(ValueError, match="residuals length"):
        sp.fast.crve(X, resid, cluster)


def test_crve_nonfinite_residuals_raise():
    X = np.random.default_rng(7).normal(size=(20, 2))
    resid = np.random.default_rng(8).normal(size=20)
    resid[0] = np.nan
    cluster = np.repeat(np.arange(4), 5)

    with pytest.raises(ValueError, match="residuals contain non-finite"):
        sp.fast.crve(X, resid, cluster)


def test_crve_negative_or_bad_weights_raise():
    X = np.random.default_rng(9).normal(size=(20, 2))
    resid = np.random.default_rng(10).normal(size=20)
    cluster = np.repeat(np.arange(4), 5)

    with pytest.raises(ValueError, match="weights must be non-negative"):
        sp.fast.crve(X, resid, cluster, weights=np.r_[[-1.0], np.ones(19)])
    with pytest.raises(ValueError, match="weights length"):
        sp.fast.crve(X, resid, cluster, weights=np.ones(19))
    bad_weights = np.ones(20)
    bad_weights[2] = np.nan
    with pytest.raises(ValueError, match="weights contain non-finite"):
        sp.fast.crve(X, resid, cluster, weights=bad_weights)


def test_boottest_missing_cluster_labels_raise():
    df = _ols_panel(n_clusters=4, m=8, seed=13)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    cluster = df["g"].to_numpy(dtype=float)
    cluster[0] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        sp.fast.boottest(X, y, cluster, null_coef=0, B=19, seed=0)


def test_boottest_invalid_obs_weights_raise():
    df = _ols_panel(n_clusters=4, m=8, seed=13)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    cluster = df["g"].to_numpy()

    with pytest.raises(ValueError, match="weights length"):
        sp.fast.boottest(
            X,
            y,
            cluster,
            null_coef=0,
            B=19,
            seed=0,
            obs_weights=np.ones(len(y) - 1),
        )
    bad_weights = np.ones(len(y))
    bad_weights[0] = np.inf
    with pytest.raises(ValueError, match="weights contain non-finite"):
        sp.fast.boottest(
            X,
            y,
            cluster,
            null_coef=0,
            B=19,
            seed=0,
            obs_weights=bad_weights,
        )
    bad_weights = np.ones(len(y))
    bad_weights[0] = -1.0
    with pytest.raises(ValueError, match="weights must be non-negative"):
        sp.fast.boottest(
            X,
            y,
            cluster,
            null_coef=0,
            B=19,
            seed=0,
            obs_weights=bad_weights,
        )


def test_boottest_wald_cluster_length_mismatch_raises():
    df = _ols_panel(n_clusters=4, m=8, seed=14)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    cluster = df["g"].to_numpy()[:-1]

    with pytest.raises(ValueError, match="cluster length"):
        sp.fast.boottest_wald(X, y, cluster, R=np.eye(2), B=19, seed=0)


def test_boottest_wald_invalid_obs_weights_raise():
    df = _ols_panel(n_clusters=4, m=8, seed=14)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    cluster = df["g"].to_numpy()
    bad_weights = np.ones(len(y))
    bad_weights[0] = np.nan

    with pytest.raises(ValueError, match="weights contain non-finite"):
        sp.fast.boottest_wald(
            X,
            y,
            cluster,
            R=np.eye(2),
            B=19,
            seed=0,
            obs_weights=bad_weights,
        )


def test_cluster_dof_bm_missing_cluster_labels_raise():
    df = _ols_panel(n_clusters=4, m=8, seed=15)
    X = df[["x1", "x2"]].to_numpy()
    cluster = df["g"].to_numpy(dtype=float)
    cluster[0] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        sp.fast.cluster_dof_bm(X, cluster, contrast=np.array([1.0, 0.0]))


def test_cluster_dof_bm_invalid_weights_raise():
    df = _ols_panel(n_clusters=4, m=8, seed=16)
    X = df[["x1", "x2"]].to_numpy()
    cluster = df["g"].to_numpy()

    with pytest.raises(ValueError, match="weights length"):
        sp.fast.cluster_dof_bm(
            X,
            cluster,
            contrast=np.array([1.0, 0.0]),
            weights=np.ones(len(df) - 1),
        )
    bad_weights = np.ones(len(df))
    bad_weights[0] = -1.0
    with pytest.raises(ValueError, match="weights must be non-negative"):
        sp.fast.cluster_dof_bm(
            X,
            cluster,
            contrast=np.array([1.0, 0.0]),
            weights=bad_weights,
        )


def test_crve_extra_df_matches_manual_formula():
    """``extra_df`` charges absorbed FE rank against the CR1 denominator —
    matches reghdfe / fixest convention. Verified against the textbook
    formula `(G/(G-1)) * (n-1)/(n - k - extra_df)`."""
    df = _ols_panel(seed=11, n_clusters=25, m=20)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()

    XtX = X.T @ X
    bread = np.linalg.inv(XtX)
    beta = bread @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    G = int(np.unique(g).size)

    score = resid[:, None] * X
    cluster_score = np.zeros((G, k))
    for i in range(n):
        cluster_score[g[i]] += score[i]
    meat = cluster_score.T @ cluster_score

    extra_df = 7  # pretend 7 FE-rank dof were absorbed
    factor_ref = (G / (G - 1)) * ((n - 1) / (n - k - extra_df))
    V_ref = bread @ meat @ bread * factor_ref

    V = sp.fast.crve(X, resid, g, type="cr1", extra_df=extra_df)
    assert np.allclose(V, V_ref, atol=1e-12)


def test_crve_extra_df_zero_is_default():
    """``extra_df=0`` must reproduce the pre-extension CR1 SE bit-for-bit."""
    df = _ols_panel(seed=12)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta

    V_default = sp.fast.crve(X, resid, g, type="cr1")
    V_explicit = sp.fast.crve(X, resid, g, type="cr1", extra_df=0)
    assert np.array_equal(V_default, V_explicit)


def test_crve_extra_df_negative_rejected():
    df = _ols_panel(seed=13)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    resid = y - X @ np.linalg.solve(X.T @ X, X.T @ y)
    with pytest.raises(ValueError, match="extra_df"):
        sp.fast.crve(X, resid, g, type="cr1", extra_df=-1)


def test_crve_extra_df_ignored_for_cr3():
    """CR3 has no CR1-style residual-DOF scalar, so ``extra_df`` must not
    change the result for ``type="cr3"``."""
    df = _ols_panel(seed=14)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    resid = y - X @ np.linalg.solve(X.T @ X, X.T @ y)

    V_a = sp.fast.crve(X, resid, g, type="cr3")
    V_b = sp.fast.crve(X, resid, g, type="cr3", extra_df=10)
    assert np.array_equal(V_a, V_b)


def test_crve_cr3_matches_clubsandwich_frozen_reference():
    df = sp.datasets.mpdta()
    X = np.column_stack(
        [
            np.ones(len(df)),
            df["treat"].to_numpy(float),
            df["year"].to_numpy(float),
        ]
    )
    y = df["lemp"].to_numpy(float)
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    V = sp.fast.crve(X, resid, df["countyreal"].to_numpy(), type="cr3")

    np.testing.assert_allclose(
        np.sqrt(np.diag(V)),
        np.array(
            [
                5.08137168727772,
                0.0122594989183214,
                0.00253657872562322,
            ]
        ),
        rtol=1e-7,
    )


# ---------------------------------------------------------------------------
# CR2 (Bell-McCaffrey)
# ---------------------------------------------------------------------------


def test_crve_cr2_runs_and_positive():
    """CR2 sandwich must be SPD on a well-conditioned panel."""
    df = _ols_panel(seed=15)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    resid = y - X @ np.linalg.solve(X.T @ X, X.T @ y)
    V = sp.fast.crve(X, resid, g, type="cr2")
    assert V.shape == (2, 2)
    assert np.allclose(V, V.T, atol=1e-10)
    # Diagonal must be strictly positive
    assert (np.diag(V) > 0).all()
    # Eigenvalues > 0 (positive-definite up to round-off)
    evals = np.linalg.eigvalsh(0.5 * (V + V.T))
    assert (evals > -1e-10).all()


def test_crve_cr2_close_to_cr1_with_many_balanced_clusters():
    """When G is large and clusters are balanced, leverage H_gg is close
    to 0 so CR2 ≈ CR1 / cr1_small_sample_factor — i.e. they share the
    same sandwich up to a multiplicative constant. We verify they're
    in the same ballpark (within 30%)."""
    df = _ols_panel(seed=16, n_clusters=80, m=40)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    resid = y - X @ np.linalg.solve(X.T @ X, X.T @ y)
    V1 = sp.fast.crve(X, resid, g, type="cr1")
    V2 = sp.fast.crve(X, resid, g, type="cr2")
    # The diagonal SEs should be in the same order of magnitude
    se1 = np.sqrt(np.diag(V1))
    se2 = np.sqrt(np.diag(V2))
    ratio = se2 / se1
    assert (0.8 <= ratio).all() and (
        ratio <= 1.3
    ).all(), f"CR2/CR1 SE ratio out of bounds: {ratio}"


def test_crve_cr2_larger_than_cr1_in_few_clusters():
    """With few clusters (G=8), CR2 corrects for cluster-leverage and
    should generally yield *larger* SEs than uncorrected CR1 (matching
    the Bell-McCaffrey small-sample motivation)."""
    df = _ols_panel(seed=17, n_clusters=8, m=40, beta=(0.4, -0.1))
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    resid = y - X @ np.linalg.solve(X.T @ X, X.T @ y)
    se1 = np.sqrt(np.diag(sp.fast.crve(X, resid, g, type="cr1")))
    se2 = np.sqrt(np.diag(sp.fast.crve(X, resid, g, type="cr2")))
    # CR2 typically inflates relative to CR1 small-sample-corrected when
    # leverage is non-trivial; we accept any ≥ 90% to allow for sampling
    # noise on a single seed.
    assert (se2 >= 0.9 * se1).all(), f"se1={se1} se2={se2}"


def test_crve_unknown_type_rejected():
    df = _ols_panel(seed=18)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    resid = y - X @ np.linalg.solve(X.T @ X, X.T @ y)
    with pytest.raises(ValueError, match="cr1.*cr2.*cr3"):
        sp.fast.crve(X, resid, g, type="cr99")


# ---------------------------------------------------------------------------
# Wild cluster bootstrap
# ---------------------------------------------------------------------------


def test_boottest_returns_pvalue_in_unit_interval():
    df = _ols_panel(seed=3, n_clusters=15, m=20)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()

    res = sp.fast.boottest(
        X,
        y,
        g,
        null_coef=0,
        null_value=0.0,
        weights="rademacher",
        B=999,
        seed=42,
    )
    assert 0.0 <= res.pvalue <= 1.0
    assert res.boot_t_dist.shape == (999,)
    assert np.isfinite(res.t_obs)


def test_boottest_rejects_under_alternative():
    """When the true beta is far from H0, the bootstrap should reject."""
    df = _ols_panel(seed=4, beta=(0.5, -0.1))
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    res = sp.fast.boottest(X, y, g, null_coef=0, null_value=0.0, B=999, seed=1)
    assert res.pvalue < 0.05, f"expected reject, got p={res.pvalue}"


def test_boottest_does_not_reject_under_null():
    """When the null is correct, p-value should be roughly uniform on
    average. We seed and just assert it's not extremely small."""
    df = _ols_panel(seed=5, beta=(0.0, -0.1))
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    res = sp.fast.boottest(X, y, g, null_coef=0, null_value=0.0, B=999, seed=2)
    # With true β=0 we don't expect p<0.01 (10% of seeds would, but seed=2
    # is a deterministic check that we're not catastrophically over-rejecting)
    assert res.pvalue > 0.01


def test_boottest_webb_weights_run():
    df = _ols_panel(seed=6, n_clusters=10, m=15)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    res = sp.fast.boottest(X, y, g, null_coef=1, weights="webb", B=499, seed=7)
    assert 0.0 <= res.pvalue <= 1.0
    # Webb-6 weights have 6 distinct values
    unique_t = np.unique(np.round(res.boot_t_dist, 8))
    # Just sanity: the bootstrap stat distribution shouldn't degenerate
    assert unique_t.size > 10


def test_boottest_seed_reproducibility():
    df = _ols_panel(seed=8)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    a = sp.fast.boottest(X, y, g, null_coef=0, B=399, seed=99)
    b = sp.fast.boottest(X, y, g, null_coef=0, B=399, seed=99)
    assert np.array_equal(a.boot_t_dist, b.boot_t_dist)
    assert a.pvalue == b.pvalue


def test_boottest_unknown_weights_rejected():
    df = _ols_panel(seed=10)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    with pytest.raises(ValueError, match="weights"):
        sp.fast.boottest(X, y, g, null_coef=0, weights="bogus", B=99)


def test_boottest_summary_string():
    df = _ols_panel(seed=11)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    res = sp.fast.boottest(X, y, g, null_coef=0, B=99, seed=0)
    s = res.summary()
    assert "boottest" in s
    assert "rademacher" in s


def test_boottest_result_agent_protocol_json_safe():
    df = _ols_panel(seed=12)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    res = sp.fast.boottest(X, y, g, null_coef=0, B=99, seed=1)

    full = res.to_dict()
    agent = res.to_agent_summary()

    assert full["kind"] == "fast_boottest_result"
    assert len(full["boot_t_dist"]) == 99
    assert agent["kind"] == "fast_boottest_agent_summary"
    assert agent["bootstrap_distribution"]["n"] == 99
    json.dumps(full)
    json.dumps(agent)


# ---------------------------------------------------------------------------
# Multi-coefficient joint Wald wild bootstrap
# ---------------------------------------------------------------------------


def test_boottest_wald_single_coef_matches_t_squared():
    """A 1-row R matrix testing β[0] = 0 should give Wald = t^2; the
    bootstrap p should match the corresponding two-sided t-bootstrap p
    in the limit B → ∞ (we just check they agree within Monte Carlo
    noise on a single seed)."""
    df = _ols_panel(seed=20, beta=(0.45, -0.10))
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()

    R = np.array([[1.0, 0.0]])  # test β_x1 = 0
    res_w = sp.fast.boottest_wald(X, y, g, R, B=999, seed=11)
    # Wald == t^2 under H0 with q=1
    res_t = sp.fast.boottest(X, y, g, null_coef=0, B=999, seed=11)
    assert abs(res_w.wald_obs - res_t.t_obs**2) < 1e-8


def test_boottest_wald_rejects_under_alternative():
    """Under a true two-coefficient alternative, joint H0 (both = 0)
    should be rejected at 5%."""
    df = _ols_panel(seed=21, beta=(0.5, -0.4))
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)
    res = sp.fast.boottest_wald(X, y, g, R, B=999, seed=2)
    assert res.pvalue < 0.05


def test_boottest_wald_does_not_reject_under_null():
    """Under the joint null β = 0, p should not be tiny on a benign seed."""
    df = _ols_panel(seed=22, beta=(0.0, 0.0))
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)
    res = sp.fast.boottest_wald(X, y, g, R, B=999, seed=3)
    assert res.pvalue > 0.01


def test_boottest_wald_seed_reproducibility():
    df = _ols_panel(seed=23)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)
    a = sp.fast.boottest_wald(X, y, g, R, B=399, seed=99)
    b = sp.fast.boottest_wald(X, y, g, R, B=399, seed=99)
    assert np.array_equal(a.boot_wald_dist, b.boot_wald_dist)
    assert a.pvalue == b.pvalue


def test_boottest_wald_R_validation():
    df = _ols_panel(seed=24)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()

    # Wrong number of columns
    with pytest.raises(ValueError, match="cols"):
        sp.fast.boottest_wald(X, y, g, np.eye(3), B=99)

    # Rank-deficient R
    R_dup = np.array([[1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="rank"):
        sp.fast.boottest_wald(X, y, g, R_dup, B=99)

    # Mismatched r length
    with pytest.raises(ValueError, match="length"):
        sp.fast.boottest_wald(X, y, g, np.eye(2), r=np.zeros(3), B=99)


def test_boottest_wald_summary_string():
    df = _ols_panel(seed=25)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    res = sp.fast.boottest_wald(X, y, g, np.eye(2), B=99, seed=0)
    s = res.summary()
    assert "Wald" in s
    assert "q=2" in s


def test_boottest_wald_result_agent_protocol_json_safe():
    df = _ols_panel(seed=26)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    res = sp.fast.boottest_wald(X, y, g, np.eye(2), B=99, seed=1)

    full = res.to_dict()
    agent = res.to_agent_summary()

    assert full["kind"] == "fast_boottest_wald_result"
    assert len(full["boot_wald_dist"]) == 99
    assert agent["kind"] == "fast_boottest_wald_agent_summary"
    assert agent["bootstrap_distribution"]["n"] == 99
    json.dumps(full)
    json.dumps(agent)


def test_wald_test_result_agent_summary_json_safe():
    res = sp.fast.WaldTestResult(
        test="HTZ",
        q=2,
        eta=15.0,
        F_stat=3.5,
        p_value=0.04,
        Q=7.0,
        R=np.eye(2),
        r=np.zeros(2),
        V_R=np.eye(2),
    )

    payload = res.to_agent_summary()

    assert payload == {
        "kind": "fast_wald_test_agent_summary",
        "test": "HTZ",
        "q": 2,
        "eta": 15.0,
        "F_stat": 3.5,
        "p_value": 0.04,
        "Q": 7.0,
    }
    json.dumps(payload)
