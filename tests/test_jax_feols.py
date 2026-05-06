"""Parity tests for ``sp.fast.feols_jax`` vs ``sp.fast.feols``.

Skips automatically when JAX is not installed (``pip install jax jaxlib``
to activate). The tests run on the JAX CPU path; correctness on GPU is
inferred from JAX's device-routing semantics — same JIT-compiled
function, only the device changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")  # whole module skips if jax missing

from statspai.fast import feols, feols_jax, jax_device_info


# Float64 + same RNG seed → bit-comparable parity to within QR vs
# normal-equation rounding. Empirically this lands within ~1e-10 on
# CPU JAX even with non-deterministic XLA scheduling.
_ATOL = 1e-9


def _make_panel(n: int = 1_000, n_firm: int = 50, n_year: int = 10,
                seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firm, size=n)
    year = rng.integers(0, n_year, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    fe_firm = rng.normal(size=n_firm)[firm]
    fe_year = rng.normal(size=n_year)[year]
    y = 1.5 * x1 - 0.4 * x2 + fe_firm + fe_year + 0.5 * rng.normal(size=n)
    return pd.DataFrame({
        "y": y, "x1": x1, "x2": x2,
        "firm": firm, "year": year,
        "w": rng.uniform(0.5, 1.5, size=n),
        "cluster": firm,
    })


# ---------------------------------------------------------------------------
# Coefficient + SE parity
# ---------------------------------------------------------------------------

def test_iid_no_fe_matches_numpy_feols():
    df = _make_panel(seed=1)
    fit_np = feols("y ~ x1 + x2", df, vcov="iid")
    fit_jx = feols_jax("y ~ x1 + x2", df, vcov="iid")

    np.testing.assert_allclose(fit_jx.coef_vec, fit_np.coef_vec, atol=_ATOL)
    np.testing.assert_allclose(fit_jx.vcov_matrix, fit_np.vcov_matrix, atol=_ATOL)
    assert fit_jx.df_resid == fit_np.df_resid
    assert fit_jx.backend == "statspai-jax"


def test_iid_with_fe_matches_numpy_feols():
    df = _make_panel(seed=2)
    fit_np = feols("y ~ x1 + x2 | firm + year", df, vcov="iid")
    fit_jx = feols_jax("y ~ x1 + x2 | firm + year", df, vcov="iid")

    np.testing.assert_allclose(fit_jx.coef_vec, fit_np.coef_vec, atol=_ATOL)
    np.testing.assert_allclose(fit_jx.vcov_matrix, fit_np.vcov_matrix, atol=_ATOL)
    assert fit_jx.fe_cardinality == fit_np.fe_cardinality
    assert fit_jx.df_resid == fit_np.df_resid


def test_hc1_with_fe_matches_numpy_feols():
    df = _make_panel(seed=3)
    fit_np = feols("y ~ x1 + x2 | firm + year", df, vcov="hc1")
    fit_jx = feols_jax("y ~ x1 + x2 | firm + year", df, vcov="hc1")

    np.testing.assert_allclose(fit_jx.coef_vec, fit_np.coef_vec, atol=_ATOL)
    np.testing.assert_allclose(fit_jx.vcov_matrix, fit_np.vcov_matrix, atol=_ATOL)


def test_cr1_with_fe_matches_numpy_feols():
    """``cr1`` is delegated to ``crve`` — must round-trip identically."""
    df = _make_panel(seed=4)
    fit_np = feols(
        "y ~ x1 + x2 | firm + year", df,
        vcov="cr1", cluster="cluster",
    )
    fit_jx = feols_jax(
        "y ~ x1 + x2 | firm + year", df,
        vcov="cr1", cluster="cluster",
    )
    np.testing.assert_allclose(fit_jx.coef_vec, fit_np.coef_vec, atol=_ATOL)
    np.testing.assert_allclose(fit_jx.vcov_matrix, fit_np.vcov_matrix, atol=_ATOL)


def test_weighted_iid_matches_numpy_feols():
    df = _make_panel(seed=5)
    fit_np = feols("y ~ x1 + x2 | firm", df, vcov="iid", weights="w")
    fit_jx = feols_jax("y ~ x1 + x2 | firm", df, vcov="iid", weights="w")

    np.testing.assert_allclose(fit_jx.coef_vec, fit_np.coef_vec, atol=_ATOL)
    np.testing.assert_allclose(fit_jx.vcov_matrix, fit_np.vcov_matrix, atol=_ATOL)


def test_within_r2_matches_numpy_feols():
    df = _make_panel(seed=6)
    fit_np = feols("y ~ x1 + x2 | firm + year", df, vcov="iid")
    fit_jx = feols_jax("y ~ x1 + x2 | firm + year", df, vcov="iid")
    assert abs(fit_jx.r_squared_within - fit_np.r_squared_within) < _ATOL


# ---------------------------------------------------------------------------
# Validation + error paths
# ---------------------------------------------------------------------------

def test_invalid_vcov_raises():
    df = _make_panel()
    with pytest.raises(ValueError, match="vcov="):
        feols_jax("y ~ x1", df, vcov="hc3")


def test_cr1_without_cluster_raises():
    df = _make_panel()
    with pytest.raises(ValueError, match="cr1"):
        feols_jax("y ~ x1", df, vcov="cr1")


def test_cluster_without_cr1_raises():
    df = _make_panel()
    with pytest.raises(ValueError, match="cluster="):
        feols_jax("y ~ x1", df, vcov="iid", cluster="cluster")


def test_invalid_dtype_raises():
    df = _make_panel()
    with pytest.raises(ValueError, match="dtype="):
        feols_jax("y ~ x1", df, dtype="bfloat16")


def test_float32_runs_with_relaxed_tol():
    """float32 mode trades precision for GPU speed; verify it still
    produces sane numbers (~3 sig figs)."""
    df = _make_panel(seed=7)
    fit_np = feols("y ~ x1 + x2 | firm", df, vcov="iid")
    fit_jx = feols_jax(
        "y ~ x1 + x2 | firm", df, vcov="iid", dtype="float32",
    )
    # Coefficients should match to ~3-4 decimals; this is the explicit
    # precision/speed tradeoff documented in the docstring.
    np.testing.assert_allclose(fit_jx.coef_vec, fit_np.coef_vec, atol=1e-3)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def test_jax_device_info_when_jax_present():
    info = jax_device_info()
    assert "jax" in info
    assert "not installed" not in info
