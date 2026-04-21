"""
Tests for LPCMCI + DYNOTEARS (time-series causal discovery).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_svar_data(T: int = 300, seed: int = 31) -> pd.DataFrame:
    """SVAR with known structure:
        X_t = 0.5 X_{t-1} + eps_x
        Y_t = 0.4 X_{t-1} + 0.3 Y_{t-1} + eps_y
        Z_t = 0.6 Y_{t-1} + eps_z
    """
    rng = np.random.default_rng(seed)
    X = np.zeros(T)
    Y = np.zeros(T)
    Z = np.zeros(T)
    for t in range(1, T):
        X[t] = 0.5 * X[t - 1] + rng.normal(0, 0.5)
        Y[t] = 0.4 * X[t - 1] + 0.3 * Y[t - 1] + rng.normal(0, 0.5)
        Z[t] = 0.6 * Y[t - 1] + rng.normal(0, 0.5)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


# -------------------------------------------------------------------------
# lpcmci
# -------------------------------------------------------------------------


def test_lpcmci_recovers_directed_chain():
    df = _make_svar_data(T=400, seed=31)
    res = sp.lpcmci(df, variables=["X", "Y", "Z"], tau_max=2, alpha=0.05)
    assert res.variables == ["X", "Y", "Z"]
    # There must be a directed lag-1 edge from X to Y
    assert res.edge_types[1, 0, 1] == "-->", (
        "Expected X --> Y at lag 1, got "
        f"{res.edge_types[1, 0, 1]!r}"
    )
    # Directed lag-1 Y -> Z
    assert res.edge_types[1, 1, 2] == "-->", res.edge_types[1, 1, 2]
    # Summary + frame work
    assert "LPCMCI" in res.summary()
    frame = res.to_frame()
    assert set(frame.columns) == {"lag", "from", "to", "type", "p_value"}
    assert (frame["type"] == "-->").sum() >= 2


def test_lpcmci_needs_numeric_variables():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="at least 2 variables"):
        sp.lpcmci(df, tau_max=1)


def test_lpcmci_short_timeseries_errors():
    df = pd.DataFrame({
        "a": np.random.randn(4), "b": np.random.randn(4),
    })
    with pytest.raises(ValueError, match="too short"):
        sp.lpcmci(df, tau_max=3)


# -------------------------------------------------------------------------
# dynotears
# -------------------------------------------------------------------------


def test_dynotears_recovers_svar_coefficients():
    df = _make_svar_data(T=500, seed=37)
    res = sp.dynotears(
        df, variables=["X", "Y", "Z"], lag=1,
        lambda_w=0.01, lambda_a=0.01, threshold=0.1,
    )
    assert res.variables == ["X", "Y", "Z"]
    # Contemporaneous should be near-zero (DGP has none).
    assert (np.abs(res.W) > 0.2).sum() == 0
    # Lagged: X->Y ~ 0.4, Y->Z ~ 0.6
    A1 = res.A[0]
    # A[i, j] = coefficient of X_{t-1}^i on X_t^j
    assert abs(A1[0, 1] - 0.4) < 0.2, f"X->Y at lag 1: {A1[0, 1]:.3f}"
    assert abs(A1[1, 2] - 0.6) < 0.2, f"Y->Z at lag 1: {A1[1, 2]:.3f}"
    # Self-lags: X->X ~ 0.5, Y->Y ~ 0.3
    assert abs(A1[0, 0] - 0.5) < 0.25, f"X->X at lag 1: {A1[0, 0]:.3f}"


def test_dynotears_contemporaneous_acyclicity():
    df = _make_svar_data(T=200, seed=41)
    res = sp.dynotears(df, lag=1, lambda_w=0.02, lambda_a=0.02, threshold=0.05)
    # h(W) should be close to 0 (acyclic).
    from scipy import linalg as _la
    h = np.trace(_la.expm(res.W * res.W)) - res.W.shape[0]
    assert h < 1e-4, h
    # Diagonal of W must be exactly 0.
    np.testing.assert_array_equal(np.diag(res.W), np.zeros(3))


def test_dynotears_lag_zero():
    """lag=0 should still run and return an empty A tensor."""
    df = _make_svar_data(T=200, seed=43)
    res = sp.dynotears(df, lag=0, lambda_w=0.02, threshold=0.05)
    assert res.A.shape == (0, 3, 3)
    assert res.lag == 0


def test_dynotears_summary_frame():
    df = _make_svar_data(T=300, seed=47)
    res = sp.dynotears(df, lag=1, lambda_w=0.02, lambda_a=0.02, threshold=0.1)
    s = res.summary()
    assert "DYNOTEARS" in s
    f = res.to_frame()
    assert set(f.columns) == {"lag", "from", "to", "coef"}


def test_timeseries_cd_in_registry():
    fns = set(sp.list_functions())
    assert "lpcmci" in fns
    assert "dynotears" in fns
