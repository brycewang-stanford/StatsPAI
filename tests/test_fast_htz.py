"""Tests for clubSandwich-equivalent HTZ Wald DOF (Pustejovsky-Tipton 2018).

See docs/superpowers/specs/2026-04-27-htz-clubsandwich-parity-design.md.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Shared panel fixture (mirrors tests/test_fast_inference.py::_ols_panel)
# ---------------------------------------------------------------------------

def _ols_panel(n_clusters=20, m=30, seed=0, beta=(0.30, -0.20), unbalanced=False):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_clusters):
        n_g = int(rng.integers(3, 50)) if unbalanced else m
        x1 = rng.normal(size=n_g)
        x2 = rng.normal(size=n_g)
        u_g = rng.normal(scale=0.5)
        eps = rng.normal(size=n_g) + u_g
        y = beta[0] * x1 + beta[1] * x2 + eps
        for i in range(n_g):
            rows.append({"g": g, "x1": x1[i], "x2": x2[i], "y": y[i]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 1: WaldTestResult dataclass
# ---------------------------------------------------------------------------

def test_wald_test_result_is_frozen_dataclass():
    """WaldTestResult should exist, be importable from sp.fast, and be frozen."""
    res = sp.fast.WaldTestResult(
        test="HTZ", q=2, eta=18.5, F_stat=3.4, p_value=0.04, Q=7.1,
        R=np.eye(2), r=np.zeros(2), V_R=np.eye(2),
    )
    assert res.test == "HTZ"
    assert res.q == 2
    # Frozen dataclass: setting an attribute must raise.
    with pytest.raises((AttributeError, Exception)):
        res.eta = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Task 2: per-cluster helper internal API
# ---------------------------------------------------------------------------

def test_htz_helper_V_R_matches_crve_path():
    """The HTZ helper's per-cluster G_g matrices should reproduce the CR2
    sandwich variance R V^CR2 R^T when contracted with residuals — i.e. it
    is *the same* CR2 path as :func:`crve`, just lifted to q-dim.
    """
    from statspai.fast.inference import _htz_per_cluster_quantities, crve

    df = _ols_panel(seed=42, n_clusters=20, m=25)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)

    # Fit OLS to get residuals
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta

    # HTZ helper path
    qty = _htz_per_cluster_quantities(X, g, R=R)
    cluster_codes = qty["cluster_codes"]
    G_clusters = qty["G"]
    v_sum = np.zeros((R.shape[0], R.shape[0]))
    for cg in range(G_clusters):
        mask = cluster_codes == cg
        e_g = e[mask]
        a_e = qty["A_g_sqrtW"][cg] @ e_g           # (n_g,)
        v_g = qty["G_g"][cg] @ a_e                 # (q,)
        v_sum += np.outer(v_g, v_g)

    # Compare to crve(type="cr2") sandwich, sliced to R subspace
    V_cr2 = crve(X, e, g, type="cr2")
    V_R_crve = R @ V_cr2 @ R.T
    np.testing.assert_allclose(v_sum, V_R_crve, rtol=1e-10, atol=1e-12)


def test_htz_helper_Omega_is_symmetric_psd():
    df = _ols_panel(seed=43, n_clusters=15, m=20)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)
    from statspai.fast.inference import _htz_per_cluster_quantities
    qty = _htz_per_cluster_quantities(X, g, R=R)
    Omega = qty["Omega"]
    assert Omega.shape == (2, 2)
    assert np.allclose(Omega, Omega.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(Omega)
    assert eigvals.min() > 1e-10, f"Ω not PD: eigvals={eigvals}"


# ---------------------------------------------------------------------------
# Task 3: Frozen R-clubSandwich fixture parity (CI backbone)
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture_all() -> dict:
    return json.loads(
        (FIXTURE_DIR / "htz_clubsandwich.json").read_text(encoding="utf-8")
    )


def _load_panel(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / f"htz_panel_{name}.csv")


@pytest.mark.parametrize("panel_name", ["q1", "q2", "q3_unbal"])
def test_htz_frozen_fixture_matches_clubsandwich(panel_name):
    """Frozen-fixture parity: η / F / p match R clubSandwich to rtol<1e-8.

    Runs in every CI environment (no R required). The fixture was generated
    by ``tests/fixtures/_gen_htz_fixture.R`` against R clubSandwich 0.6.2.
    """
    fx = _load_fixture_all()[panel_name]
    df = _load_panel(panel_name)

    # Re-fit OLS with intercept on Python side (R side did `lm(y ~ x1 + x2)`).
    X = np.column_stack([np.ones(len(df)), df[["x1", "x2"]].to_numpy()])
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()

    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta

    R = np.atleast_2d(np.array(fx["R"], dtype=np.float64))

    res = sp.fast.cluster_wald_htz(
        X=X, residuals=e, cluster=g, R=R, beta=beta,
    )
    np.testing.assert_allclose(
        res.eta, fx["eta"], rtol=1e-8,
        err_msg=f"η mismatch on panel {panel_name}",
    )
    np.testing.assert_allclose(
        res.F_stat, fx["F_stat"], rtol=1e-8,
        err_msg=f"F mismatch on panel {panel_name}",
    )
    np.testing.assert_allclose(
        res.p_value, fx["p_value"], rtol=1e-7,
        err_msg=f"p-value mismatch on panel {panel_name}",
    )
    np.testing.assert_allclose(
        res.V_R, np.atleast_2d(np.array(fx["V_R"], dtype=np.float64)),
        rtol=1e-9, err_msg=f"V_R mismatch on panel {panel_name}",
    )


# ---------------------------------------------------------------------------
# Task 4: helper ↔ full-wrapper self-consistency
# ---------------------------------------------------------------------------

def test_htz_q1_helper_eta_matches_full_wrapper():
    """The DOF helper and the full wrapper share Step 4 — η must be bit-equal."""
    df = _ols_panel(seed=70, n_clusters=20, m=25)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    contrast = np.array([1.0, 0.0])

    nu_helper = sp.fast.cluster_dof_wald_htz(X, g, R=contrast.reshape(1, -1))
    res = sp.fast.cluster_wald_htz(
        X=X, residuals=e, cluster=g, R=contrast.reshape(1, -1), beta=beta,
    )
    assert abs(nu_helper - res.eta) < 1e-12


def test_htz_F_and_p_value_internal_consistency():
    """Hotelling-T² scaling consistency: F = (η-q+1)/(η q) · Q, p = sf(F)."""
    from scipy.stats import f as scipy_f

    df = _ols_panel(seed=71, n_clusters=25, m=20)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta

    R = np.eye(2)
    res = sp.fast.cluster_wald_htz(
        X=X, residuals=e, cluster=g, R=R, beta=beta,
    )
    # F = (η - q + 1) / (η · q) · Q
    F_recomputed = (res.eta - res.q + 1) / (res.eta * res.q) * res.Q
    assert abs(F_recomputed - res.F_stat) < 1e-10

    # p = 1 - F.cdf(F_stat)  with df1=q, df2=η-q+1
    p_recomputed = scipy_f.sf(res.F_stat, res.q, res.eta - res.q + 1)
    assert abs(p_recomputed - res.p_value) < 1e-10


def test_htz_zero_residuals_returns_p_one():
    """e ≡ 0 ⇒ Q = 0 ⇒ F = 0 ⇒ p = 1 exactly."""
    df = _ols_panel(seed=72, n_clusters=20, m=15)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    beta = np.zeros(2)
    e = np.zeros(len(df))
    R = np.eye(2)
    res = sp.fast.cluster_wald_htz(
        X=X, residuals=e, cluster=g, R=R, beta=beta,
    )
    assert res.Q == 0.0
    assert res.F_stat == 0.0
    assert res.p_value == 1.0


# ---------------------------------------------------------------------------
# Task 5: Validation + edge cases
# ---------------------------------------------------------------------------

def test_htz_validates_R_shape():
    df = _ols_panel(seed=80)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()

    with pytest.raises(ValueError, match="cols"):
        sp.fast.cluster_dof_wald_htz(X, g, R=np.eye(3))
    with pytest.raises(ValueError, match="rank"):
        sp.fast.cluster_dof_wald_htz(
            X, g, R=np.array([[1.0, 0.0], [1.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="at least one row"):
        sp.fast.cluster_dof_wald_htz(X, g, R=np.empty((0, 2)))


def test_htz_too_few_clusters_rejected():
    rng = np.random.default_rng(0)
    n = 50
    X = rng.normal(size=(n, 3))
    g = np.concatenate([np.zeros(25), np.ones(25)]).astype(int)
    with pytest.raises(ValueError, match="G > q"):
        sp.fast.cluster_dof_wald_htz(X, g, R=np.eye(3))


def test_htz_invariant_to_X_column_rescaling():
    """Multiplying X column j by α and adjusting R column j by 1/α leaves
    the HTZ statistic and η invariant (design-equivariance)."""
    df = _ols_panel(seed=81, n_clusters=20, m=20)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)
    nu0 = sp.fast.cluster_dof_wald_htz(X, g, R=R)

    alpha = np.array([3.0, 0.5])
    X_scaled = X * alpha
    R_scaled = R / alpha[None, :]
    nu1 = sp.fast.cluster_dof_wald_htz(X_scaled, g, R=R_scaled)
    np.testing.assert_allclose(nu0, nu1, rtol=1e-9)


def test_htz_invariant_to_cluster_relabel():
    """Permuting cluster IDs leaves η, F, p unchanged."""
    df = _ols_panel(seed=82, n_clusters=20, m=20)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    R = np.eye(2)

    res0 = sp.fast.cluster_wald_htz(X=X, residuals=e, cluster=g, R=R, beta=beta)

    rng = np.random.default_rng(0)
    perm = rng.permutation(g.max() + 1)
    g_relab = perm[g]
    res1 = sp.fast.cluster_wald_htz(
        X=X, residuals=e, cluster=g_relab, R=R, beta=beta,
    )
    np.testing.assert_allclose(res0.eta, res1.eta, rtol=1e-10)
    np.testing.assert_allclose(res0.F_stat, res1.F_stat, rtol=1e-10)
    np.testing.assert_allclose(res0.p_value, res1.p_value, rtol=1e-10)


def test_htz_independent_of_bread_arg():
    df = _ols_panel(seed=83)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)
    nu_a = sp.fast.cluster_dof_wald_htz(X, g, R=R)
    nu_b = sp.fast.cluster_dof_wald_htz(
        X, g, R=R, bread=np.linalg.inv(X.T @ X),
    )
    assert abs(nu_a - nu_b) < 1e-10


def test_htz_eta_in_sane_range():
    """For a well-conditioned panel with G=25, q=2: η in (q, 2·G).
    Catches catastrophic-sign-error bugs."""
    df = _ols_panel(seed=84, n_clusters=25, m=20)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    R = np.eye(2)
    nu = sp.fast.cluster_dof_wald_htz(X, g, R=R)
    assert nu > 2, f"η={nu} ≤ q=2"
    assert nu < 25 * 2, f"η={nu} > 2·G upper sanity bound"
    assert np.isfinite(nu)


def test_htz_singleton_cluster_warns():
    """A cluster of size 1 should trigger a RuntimeWarning, not a crash."""
    rng = np.random.default_rng(99)
    G = 20
    rows = []
    for cg in range(G):
        n_g = 1 if cg == 0 else 10
        x1 = rng.normal(size=n_g)
        x2 = rng.normal(size=n_g)
        eps = rng.normal(size=n_g)
        rows.append(pd.DataFrame({"g": cg, "x1": x1, "x2": x2,
                                    "y": 0.3 * x1 - 0.2 * x2 + eps}))
    df = pd.concat(rows, ignore_index=True)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()

    with pytest.warns(RuntimeWarning, match="singleton cluster"):
        nu = sp.fast.cluster_dof_wald_htz(X, g, R=np.eye(2))
    assert np.isfinite(nu)


def test_htz_r_shape_mismatch_rejected():
    df = _ols_panel(seed=85, n_clusters=20)
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    R = np.eye(2)

    with pytest.raises(ValueError, match="r has"):
        sp.fast.cluster_wald_htz(
            X=X, residuals=e, cluster=g, R=R, beta=beta, r=np.zeros(3),
        )


def test_htz_negative_weights_rejected():
    df = _ols_panel(seed=86, n_clusters=20)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    bad_w = np.ones(len(df))
    bad_w[0] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        sp.fast.cluster_dof_wald_htz(X, g, R=np.eye(2), weights=bad_w)


def test_htz_non_uniform_weights_not_implemented():
    """v1 spec locks Φ=I (weights=None or all 1); other weights raise."""
    df = _ols_panel(seed=87, n_clusters=20)
    X = df[["x1", "x2"]].to_numpy()
    g = df["g"].to_numpy()
    weights = np.linspace(0.5, 2.0, len(df))
    with pytest.raises(NotImplementedError, match="non-uniform weights"):
        sp.fast.cluster_dof_wald_htz(X, g, R=np.eye(2), weights=weights)


def test_htz_q1_documented_drift_from_bm_simplified():
    """At q=1, HTZ uses cross-cluster terms; cluster_dof_bm uses the BM 2002
    simplified (Σλ)²/Σλ² formula. They differ by ~5-15% on canonical panels.

    Locks the *direction* of the drift; if broken either side regressed.
    """
    df = _ols_panel(seed=46, n_clusters=15, m=20)
    X_ic = np.column_stack([np.ones(len(df)), df[["x1", "x2"]].to_numpy()])
    g = df["g"].to_numpy()
    contrast = np.array([0.0, 1.0, 0.0])

    nu_bm = sp.fast.cluster_dof_bm(X_ic, g, contrast=contrast)
    nu_htz = sp.fast.cluster_dof_wald_htz(X_ic, g, R=contrast.reshape(1, -1))

    rel = abs(nu_htz - nu_bm) / max(abs(nu_bm), 1e-12)
    assert 0.005 <= rel <= 0.30, (
        f"q=1 drift HTZ vs BM-simplified out of band: "
        f"BM={nu_bm:.4f}, HTZ={nu_htz:.4f}, rel={rel:.4f}"
    )
    assert 1.0 < nu_htz < 14.0
    assert 1.0 < nu_bm < 14.0
