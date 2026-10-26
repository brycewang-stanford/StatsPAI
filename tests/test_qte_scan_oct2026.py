"""Option scan (2026-10-27): quantile / distributional treatment effects.

Each test states the correct behaviour; every one failed before the fix
landed with it (the defect is named in the test's comment or docstring).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _qr_frame(n: int = 500, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    y = 1.0 + (1.0 + 0.5 * d) * rng.standard_t(4, size=n) + 0.8 * d + 0.6 * x
    return pd.DataFrame({"y": y, "d": d, "x": x})


# ----------------------------------------------------------------------
# 4. conditional_qr is the quantile-regression coefficient (= sp.qreg = rq)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("tau", [0.25, 0.5, 0.9])
def test_conditional_qr_equals_qreg_coefficient(tau):
    df = _qr_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = sp.qte(
            df,
            y="y",
            treatment="d",
            controls=["x"],
            method="conditional_qr",
            quantiles=[tau],
            n_boot=2,
        )
        ref = sp.qreg(df, "y ~ d + x", quantile=tau)
    assert got.effects[0] == pytest.approx(float(ref.params["d"]), rel=1e-8, abs=1e-10)


# ----------------------------------------------------------------------
# 10. distributional_te: missing values and treatment coding
# ----------------------------------------------------------------------
def _dte_frame(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 2, n)
    return pd.DataFrame({"y": 1.0 + 1.5 * d + rng.normal(size=n), "d": d})


def test_distributional_te_drops_or_rejects_missing_outcome():
    df = _dte_frame()
    holed = df.copy()
    holed.loc[7, "y"] = np.nan
    kw = dict(y="y", treatment="d", method="ipw", quantiles=[0.5], n_boot=20, seed=0)
    try:
        got = sp.distributional_te(holed, **kw)
    except ValueError:
        return
    ref = sp.distributional_te(holed.dropna().reset_index(drop=True), **kw)
    assert np.isfinite(got.qte_effects).all()
    np.testing.assert_allclose(got.qte_effects, ref.qte_effects, rtol=0, atol=1e-12)


def test_distributional_te_rejects_non_binary_treatment():
    df = _dte_frame()
    df["d"] = df["d"] + 1
    with pytest.raises(ValueError):
        sp.distributional_te(
            df, y="y", treatment="d", method="ipw", quantiles=[0.5], n_boot=5, seed=0
        )


# ----------------------------------------------------------------------
# 11. panel_qtet: duplicated (unit, time) rows
# ----------------------------------------------------------------------
def test_panel_qtet_rejects_duplicated_unit_time_rows():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        a = rng.normal()
        treated = int(i >= 30)
        for per in (0, 1, 2):
            rows.append(
                (i, per, a + rng.normal() + 2.0 * treated * (per == 2), treated)
            )
    df = pd.DataFrame(rows, columns=["id", "per", "y", "d"])
    # unit 3 appears twice in every period: arrays stay the same length, so
    # nothing crashes and the unit is silently double-counted
    dup = pd.concat([df, df[df.id == 3]], ignore_index=True)
    with pytest.raises(ValueError):
        sp.panel_qtet(
            dup,
            y="y",
            treat="d",
            unit="id",
            time="per",
            t=2,
            tmin1=1,
            tmin2=0,
            se="none",
        )


# ----------------------------------------------------------------------
# 13. qte: treatment truncation
# ----------------------------------------------------------------------
def test_qte_rejects_fractional_treatment():
    df = _dte_frame()
    df["d"] = df["d"].astype(float)
    df.loc[0, "d"] = 0.6
    with pytest.raises(ValueError):
        sp.qte(df, y="y", treatment="d", quantiles=[0.5])
