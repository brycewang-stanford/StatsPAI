"""Tests for shift_share_political (Park & Xu, arXiv:2603.00135, 2026)."""

import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp


def _political_panel(units=40, T=6, K=5, seed=0):
    rng = np.random.default_rng(seed)
    # Unit shares on a simplex over K exposure categories
    shares_arr = rng.dirichlet(np.ones(K), size=units)
    shares = pd.DataFrame(
        shares_arr,
        columns=[f"s{k}" for k in range(K)],
        index=range(units),
    )
    # National shocks per category
    shocks = pd.Series(
        rng.normal(size=K),
        index=[f"s{k}" for k in range(K)],
    )
    # Build a long panel — y depends linearly on shares @ shocks (the
    # shift-share exposure), plus unit FEs + noise.
    rows = []
    ssiv = shares_arr @ shocks.to_numpy()
    tau_true = 1.5
    for u in range(units):
        u_fe = rng.normal(0, 0.2)
        for t in range(T):
            d_u = ssiv[u] + rng.normal(0, 0.3)
            y_u = u_fe + tau_true * d_u + rng.normal(0, 0.3)
            rows.append({"unit": u, "time": t, "y": y_u, "d": d_u})
    return pd.DataFrame(rows), shares, shocks, tau_true


def test_shift_share_political_returns_point_estimate():
    df, shares, shocks, _ = _political_panel(seed=0)
    r = sp.shift_share_political(
        df, unit="unit", time="time", outcome="y", endog="d",
        shares=shares, shocks=shocks, leave_one_out=False,
    )
    assert r is not None
    est = getattr(r, "estimate", None) or getattr(r, "coefficient", None)
    assert est is not None
    assert np.isfinite(est)


def test_shift_share_political_registered():
    assert "shift_share_political" in sp.list_functions()


def test_shift_share_political_exposes_rotemberg_diagnostics():
    """Park-Xu recommend Rotemberg top-K as a default diagnostic."""
    df, shares, shocks, _ = _political_panel(seed=1)
    r = sp.shift_share_political(
        df, unit="unit", time="time", outcome="y", endog="d",
        shares=shares, shocks=shocks, leave_one_out=False,
    )
    attrs = [a for a in dir(r) if not a.startswith("_")]
    has_rot = any("rotemberg" in a.lower() or "top" in a.lower() or
                   "weights" in a.lower() for a in attrs)
    assert has_rot, f"result missing Rotemberg-style diagnostic; attrs: {attrs[:8]}"


def test_shift_share_political_rejects_mismatched_shocks():
    df, shares, shocks, _ = _political_panel(seed=2)
    # Shocks indexed by wrong category names
    bad_shocks = pd.Series(
        np.random.randn(5),
        index=[f"different{k}" for k in range(5)],
    )
    with pytest.raises((KeyError, ValueError)):
        sp.shift_share_political(
            df, unit="unit", time="time", outcome="y", endog="d",
            shares=shares, shocks=bad_shocks, leave_one_out=False,
        )
