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


# ---------------------------------------------------------------------------
# model_info / FE metadata pickup (panel variant)
# ---------------------------------------------------------------------------

def _panel_long_df(units=20, T=4, K=3, seed=0):
    """Build a long-format panel + per-period shares & shocks for the
    panel variant, which requires shocks-by-time (not a flat Series).
    """
    rng = np.random.default_rng(seed)
    shares = pd.DataFrame(
        rng.dirichlet(np.ones(K), size=units),
        columns=[f"s{k}" for k in range(K)],
        index=range(units),
    )
    shocks = pd.DataFrame(
        rng.normal(size=(T, K)),
        index=range(T),
        columns=[f"s{k}" for k in range(K)],
    )
    rows = []
    for u in range(units):
        u_fe = rng.normal(0, 0.2)
        for t in range(T):
            b = float((shares.loc[u] * shocks.loc[t]).sum())
            d = b + rng.normal(0, 0.1)
            y = u_fe + 0.5 * d + rng.normal(0, 0.1)
            rows.append({"u": u, "t": t, "y": y, "d": d})
    return pd.DataFrame(rows), shares, shocks


@pytest.mark.parametrize("fe_mode,expected", [
    ("two-way", "u+t"),
    ("unit", "u"),
    ("time", "t"),
    ("none", ""),
])
def test_panel_writes_fe_to_model_info(fe_mode, expected):
    """Bartik panel result exposes FE variable names via the canonical
    ``model_info['fixed_effects']`` key (pyfixest-compatible ``"unit+time"``
    convention) — not just the mode string in ``diagnostics['fe']``.
    """
    df, shares, shocks = _panel_long_df(seed=0)
    r = sp.shift_share_political_panel(
        df, unit="u", time="t", outcome="y", endog="d",
        shares=shares, shocks=shocks, fe=fe_mode,
    )
    # Canonical key for the output layer
    assert r.model_info["fixed_effects"] == expected
    # Backwards-compat: the mode string is still in diagnostics
    assert r.diagnostics["fe"] == fe_mode


def test_panel_fe_picked_up_by_diagnostic_extractor():
    """End-to-end: ``extract_fe_cluster_indicators`` now turns Bartik panel
    FE metadata into AER-style per-variable rows (``"U FE"``, ``"T FE"``).
    """
    from statspai.output._diagnostics import extract_fe_cluster_indicators
    df, shares, shocks = _panel_long_df(seed=1)
    r = sp.shift_share_political_panel(
        df, unit="u", time="t", outcome="y", endog="d",
        shares=shares, shocks=shocks, fe="two-way",
    )
    rows = extract_fe_cluster_indicators([r])
    assert rows["U FE"] == ["Yes"]
    assert rows["T FE"] == ["Yes"]


def test_panel_fe_none_drops_fe_rows_entirely():
    df, shares, shocks = _panel_long_df(seed=2)
    r = sp.shift_share_political_panel(
        df, unit="u", time="t", outcome="y", endog="d",
        shares=shares, shocks=shocks, fe="none",
    )
    from statspai.output._diagnostics import extract_fe_cluster_indicators
    rows = extract_fe_cluster_indicators([r])
    assert all(not k.endswith(" FE") for k in rows.keys())
    assert "Fixed Effects" not in rows
