"""Tests for shift_share_political (Park & Xu, arXiv:2603.00135, 2026)."""

import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp
from statspai.exceptions import (
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)


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


def _manual_2sls_slope(y, d, z):
    y_arr = np.asarray(y, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    stage1 = np.column_stack([np.ones_like(z_arr), z_arr])
    first_stage, *_ = np.linalg.lstsq(stage1, d_arr, rcond=None)
    d_hat = stage1 @ first_stage
    stage2 = np.column_stack([np.ones_like(d_hat), d_hat])
    second_stage, *_ = np.linalg.lstsq(stage2, y_arr, rcond=None)
    return float(second_stage[1])


def test_shift_share_political_returns_point_estimate():
    df, shares, shocks, _ = _political_panel(seed=0)
    r = sp.shift_share_political(
        df,
        unit="unit",
        time="time",
        outcome="y",
        endog="d",
        shares=shares,
        shocks=shocks,
        leave_one_out=False,
    )
    panel = df.sort_values(["unit", "time"])
    first = panel.groupby("unit").first()
    last = panel.groupby("unit").last()
    dy = last["y"] - first["y"]
    dx = last["d"] - first["d"]
    z = shares.loc[dy.index].to_numpy(float) @ shocks.to_numpy(float)
    np.testing.assert_allclose(r.estimate, _manual_2sls_slope(dy, dx, z))
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
        df,
        unit="unit",
        time="time",
        outcome="y",
        endog="d",
        shares=shares,
        shocks=shocks,
        leave_one_out=False,
    )
    attrs = [a for a in dir(r) if not a.startswith("_")]
    has_rot = any(
        "rotemberg" in a.lower() or "top" in a.lower() or "weights" in a.lower()
        for a in attrs
    )
    assert has_rot, f"result missing Rotemberg-style diagnostic; attrs: {attrs[:8]}"


def test_shift_share_political_rejects_mismatched_shocks():
    df, shares, shocks, _ = _political_panel(seed=2)
    # Shocks indexed by wrong category names
    bad_shocks = pd.Series(
        np.random.randn(5),
        index=[f"different{k}" for k in range(5)],
    )
    with pytest.raises(MethodIncompatibility, match="no overlap"):
        sp.shift_share_political(
            df,
            unit="unit",
            time="time",
            outcome="y",
            endog="d",
            shares=shares,
            shocks=bad_shocks,
            leave_one_out=False,
        )


def test_shift_share_political_accepts_scalar_covariate():
    df, shares, shocks, _ = _political_panel(seed=3)
    df["pre_cov"] = df.groupby("unit")["y"].transform("first")
    r = sp.shift_share_political(
        df,
        unit="unit",
        time="time",
        outcome="y",
        endog="d",
        shares=shares,
        shocks=shocks,
        covariates="pre_cov",
        leave_one_out=False,
    )
    assert list(r.share_balance["covariate"]) == ["pre_cov"]


def test_shift_share_political_rejects_contract_errors():
    df, shares, shocks, _ = _political_panel(seed=4)
    with pytest.raises(MethodIncompatibility, match="shares"):
        sp.shift_share_political(
            df,
            unit="unit",
            time="time",
            outcome="y",
            endog="d",
            shares=shares.to_numpy(),
            shocks=shocks,
            leave_one_out=False,
        )
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.shift_share_political(
            df,
            unit="unit",
            time="time",
            outcome="y",
            endog="d",
            shares=shares,
            shocks=shocks,
            alpha=1.0,
            leave_one_out=False,
        )


def test_shift_share_political_rejects_nonfinite_shares():
    df, shares, shocks, _ = _political_panel(seed=5)
    bad = shares.copy()
    bad.iloc[0, 0] = np.inf
    with pytest.raises(NumericalInstability, match="non-finite"):
        sp.shift_share_political(
            df,
            unit="unit",
            time="time",
            outcome="y",
            endog="d",
            shares=bad,
            shocks=shocks,
            leave_one_out=False,
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


def _manual_panel_shift_share_slope(df, shares, shocks, *, fe):
    work = df.sort_values(["u", "t"]).reset_index(drop=True).copy()
    work["z"] = [
        float((shares.loc[row.u] * shocks.loc[row.t]).sum())
        for row in work.itertuples(index=False)
    ]
    cols = ["y", "d", "z"]
    if fe in ("two-way", "unit"):
        work[cols] = work[cols].sub(work.groupby("u")[cols].transform("mean"))
    if fe in ("two-way", "time"):
        work[cols] = work[cols].sub(work.groupby("t")[cols].transform("mean"))
    return _manual_2sls_slope(work["y"], work["d"], work["z"])


@pytest.mark.parametrize(
    "fe_mode,expected",
    [
        ("two-way", "u+t"),
        ("unit", "u"),
        ("time", "t"),
        ("none", ""),
    ],
)
def test_panel_writes_fe_to_model_info(fe_mode, expected):
    """Bartik panel result exposes FE variable names via the canonical
    ``model_info['fixed_effects']`` key (pyfixest-compatible ``"unit+time"``
    convention) — not just the mode string in ``diagnostics['fe']``.
    """
    df, shares, shocks = _panel_long_df(seed=0)
    r = sp.shift_share_political_panel(
        df,
        unit="u",
        time="t",
        outcome="y",
        endog="d",
        shares=shares,
        shocks=shocks,
        fe=fe_mode,
    )
    np.testing.assert_allclose(
        r.estimate,
        _manual_panel_shift_share_slope(df, shares, shocks, fe=fe_mode),
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
        df,
        unit="u",
        time="t",
        outcome="y",
        endog="d",
        shares=shares,
        shocks=shocks,
        fe="two-way",
    )
    rows = extract_fe_cluster_indicators([r])
    assert rows["U FE"] == ["Yes"]
    assert rows["T FE"] == ["Yes"]


def test_panel_fe_none_drops_fe_rows_entirely():
    df, shares, shocks = _panel_long_df(seed=2)
    r = sp.shift_share_political_panel(
        df,
        unit="u",
        time="t",
        outcome="y",
        endog="d",
        shares=shares,
        shocks=shocks,
        fe="none",
    )
    from statspai.output._diagnostics import extract_fe_cluster_indicators

    rows = extract_fe_cluster_indicators([r])
    assert all(not k.endswith(" FE") for k in rows.keys())
    assert "Fixed Effects" not in rows


def test_panel_rejects_bad_fe_cluster_and_missing_shock_row():
    df, shares, shocks = _panel_long_df(seed=3)
    with pytest.raises(MethodIncompatibility, match="fe must"):
        sp.shift_share_political_panel(
            df,
            unit="u",
            time="t",
            outcome="y",
            endog="d",
            shares=shares,
            shocks=shocks,
            fe="bad",
        )
    with pytest.raises(MethodIncompatibility, match="cluster must"):
        sp.shift_share_political_panel(
            df,
            unit="u",
            time="t",
            outcome="y",
            endog="d",
            shares=shares,
            shocks=shocks,
            cluster="bad",
        )
    with pytest.raises(MethodIncompatibility, match="shocks row missing"):
        sp.shift_share_political_panel(
            df,
            unit="u",
            time="t",
            outcome="y",
            endog="d",
            shares=shares,
            shocks=shocks.drop(index=0),
        )


def test_panel_rejects_uncovered_units_with_clear_error():
    df, shares, shocks = _panel_long_df(seed=4)
    with pytest.raises(DataInsufficient, match="missing Bartik IV"):
        sp.shift_share_political_panel(
            df,
            unit="u",
            time="t",
            outcome="y",
            endog="d",
            shares=shares.iloc[:-1],
            shocks=shocks,
        )
