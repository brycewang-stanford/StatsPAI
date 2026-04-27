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
