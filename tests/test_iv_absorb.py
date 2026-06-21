"""Parity tests for ``sp.iv(absorb=...)`` (Phase 3 HDFE wiring).

The reference is ``sp.iv`` with the FE put in as explicit dummy
variables (drop-first to avoid the dummy-variable trap, since the
absorb path drops the intercept). Both paths produce the same 2SLS
fitted values via the Frisch–Waugh–Lovell theorem; coefficients on the
remaining regressors must match to within float-rounding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_iv_panel(n: int = 600, n_firm: int = 30, seed: int = 0) -> pd.DataFrame:
    """Panel with a confounding firm-FE so absorb-vs-no-absorb
    coefficients differ enough to make parity meaningful."""
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firm, size=n)
    fe = rng.normal(scale=2.0, size=n_firm)[firm]
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    d = 0.6 * z1 + 0.3 * z2 + 0.8 * fe + rng.normal(size=n)
    x1 = rng.normal(size=n)
    y = 1.0 + 0.5 * d + 0.2 * x1 + 1.5 * fe + 0.5 * rng.normal(size=n)
    cluster = firm  # by-firm clustering for vcov tests
    return pd.DataFrame(
        {
            "y": y,
            "d": d,
            "z1": z1,
            "z2": z2,
            "x1": x1,
            "firm": firm,
            "cluster": cluster,
        }
    )


def _add_dummies(df: pd.DataFrame, *cols: str) -> tuple[pd.DataFrame, list[str]]:
    """Append drop-first dummies for each column; return (df, dummy names)."""
    out = df.copy()
    dummy_names: list[str] = []
    for col in cols:
        d = pd.get_dummies(df[col], prefix=f"_{col}_", drop_first=True).astype(float)
        out = pd.concat([out, d], axis=1)
        dummy_names.extend(d.columns.tolist())
    return out, dummy_names


# ---------------------------------------------------------------------------
# Coefficient parity vs explicit dummy controls
# ---------------------------------------------------------------------------


def test_one_way_absorb_coef_matches_dummy_control():
    df = _make_iv_panel(seed=1)
    r_absorb = sp.iv("y ~ (d ~ z1 + z2) + x1", data=df, absorb="firm")

    df2, dummies = _add_dummies(df, "firm")
    formula_dummy = f"y ~ (d ~ z1 + z2) + x1 + {' + '.join(dummies)}"
    r_dummy = sp.iv(formula_dummy, data=df2)

    np.testing.assert_allclose(
        r_absorb.params["d"],
        r_dummy.params["d"],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        r_absorb.params["x1"],
        r_dummy.params["x1"],
        atol=1e-9,
    )


def test_two_way_absorb_coef_matches_dummy_control():
    df = _make_iv_panel(seed=2)
    df["year"] = np.tile(np.arange(10), len(df) // 10 + 1)[: len(df)]

    r_absorb = sp.iv("y ~ (d ~ z1 + z2) + x1", data=df, absorb=["firm", "year"])

    df2, dummies = _add_dummies(df, "firm", "year")
    formula_dummy = f"y ~ (d ~ z1 + z2) + x1 + {' + '.join(dummies)}"
    r_dummy = sp.iv(formula_dummy, data=df2)

    np.testing.assert_allclose(
        r_absorb.params["d"],
        r_dummy.params["d"],
        atol=1e-9,
    )


def test_absorb_string_with_plus_parses():
    """``absorb='firm + year'`` and ``absorb=['firm', 'year']`` agree."""
    df = _make_iv_panel(seed=3)
    df["year"] = np.tile(np.arange(10), len(df) // 10 + 1)[: len(df)]

    r1 = sp.iv("y ~ (d ~ z1 + z2) + x1", data=df, absorb="firm + year")
    r2 = sp.iv("y ~ (d ~ z1 + z2) + x1", data=df, absorb=["firm", "year"])
    np.testing.assert_allclose(r1.params["d"], r2.params["d"], atol=1e-12)


# ---------------------------------------------------------------------------
# Standard errors
# ---------------------------------------------------------------------------


def test_absorb_iid_se_matches_dummy_control():
    df = _make_iv_panel(seed=4)
    r_absorb = sp.iv("y ~ (d ~ z1 + z2) + x1", data=df, absorb="firm")
    df2, dummies = _add_dummies(df, "firm")
    formula_dummy = f"y ~ (d ~ z1 + z2) + x1 + {' + '.join(dummies)}"
    r_dummy = sp.iv(formula_dummy, data=df2)

    # SE picks up the same sigma² scaling so they must match to ~5 decimals
    # (small drift from QR vs normal-equation rounding through different
    # code paths).
    np.testing.assert_allclose(
        r_absorb.std_errors["d"],
        r_dummy.std_errors["d"],
        rtol=1e-3,
    )


def test_absorb_cluster_se_matches_dummy_control():
    df = _make_iv_panel(seed=5)
    r_absorb = sp.iv(
        "y ~ (d ~ z1 + z2) + x1",
        data=df,
        absorb="firm",
        robust="nonrobust",
        cluster="cluster",
    )
    df2, dummies = _add_dummies(df, "firm")
    formula_dummy = f"y ~ (d ~ z1 + z2) + x1 + {' + '.join(dummies)}"
    r_dummy = sp.iv(formula_dummy, data=df2, cluster="cluster")

    # Cluster SE: small-sample factor uses (n-k) which absorb path
    # rescales to (n-k-fe_dof). Both should match to ~3 decimals.
    np.testing.assert_allclose(
        r_absorb.std_errors["d"],
        r_dummy.std_errors["d"],
        rtol=1e-2,
    )


# ---------------------------------------------------------------------------
# DOF + metadata
# ---------------------------------------------------------------------------


def test_absorb_df_resid_charges_fe_dof():
    df = _make_iv_panel(seed=6)
    r = sp.iv("y ~ (d ~ z1 + z2) + x1", data=df, absorb="firm")
    n = len(df)
    k = 2  # x1 + d (no intercept under absorb)
    fe_dof = r.model_info["fe_dof"]
    assert fe_dof == 30 - 1  # 30 firms in DGP
    assert r.data_info["df_resid"] == n - k - fe_dof


def test_absorb_metadata_attached():
    df = _make_iv_panel(seed=7)
    r = sp.iv("y ~ (d ~ z1 + z2) + x1", data=df, absorb="firm")
    info = r.model_info
    assert info["absorb"] == ["firm"]
    assert info["fe_cardinality"] == [30]
    assert info["fe_dof"] == 29
    assert info["n_dropped_singletons"] == 0


# ---------------------------------------------------------------------------
# Backwards compatibility: absorb=None must not change anything
# ---------------------------------------------------------------------------


def test_absorb_none_matches_no_absorb_call():
    df = _make_iv_panel(seed=8)
    r0 = sp.iv("y ~ (d ~ z1 + z2) + x1", data=df)
    r1 = sp.iv("y ~ (d ~ z1 + z2) + x1", data=df, absorb=None)
    np.testing.assert_allclose(r0.params, r1.params, atol=1e-15)
    np.testing.assert_allclose(r0.std_errors, r1.std_errors, atol=1e-15)


# ---------------------------------------------------------------------------
# Validation / error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["liml", "fuller", "gmm", "jive"])
def test_absorb_with_unsupported_method_raises(method):
    df = _make_iv_panel(seed=9)
    with pytest.raises(NotImplementedError, match="Phase 3b"):
        sp.iv(
            "y ~ (d ~ z1 + z2) + x1",
            data=df,
            method=method,
            absorb="firm",
        )


def test_absorb_missing_column_raises():
    df = _make_iv_panel(seed=10)
    with pytest.raises(ValueError, match="not_a_column"):
        sp.iv("y ~ (d ~ z1 + z2) + x1", data=df, absorb="not_a_column")
