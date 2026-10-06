"""``tsfilter`` against Stata 18 ``tsfilter`` and statsmodels.

All filters are deterministic linear maps of the series, so cycles and
trends are compared digit for digit on the committed synthetic file
``_fixtures/tsfilter.csv`` (``_generate_tsfilter_data.py``; reference
``tsfilter_Stata.csv`` from ``_generate_tsfilter_Stata.do``, Stata 18 MP).

Tolerance ``EXACT`` = 1e-9, relative to the largest absolute value of the
reference column: each side solves a band system (hp, bw) or sums up to
160 products (cf); observed errors are below 1e-12. statsmodels'
``hpfilter`` uses a sparse LU and agrees with both Stata and us to 1e-10
only, so its assert is at 1e-8.

Stata's ``stationary`` variants of ``cf`` and its ``cf, smaorder()`` are
not implemented and not compared.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.timeseries.tsfilter import tsfilter

FIX = Path(__file__).parent / "_fixtures"
EXACT = 1e-9


def rel(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # the filter must be undefined exactly where the reference is
    assert np.array_equal(np.isnan(a), np.isnan(b))
    ok = ~np.isnan(b)
    return float(np.max(np.abs(a - b)[ok]) / np.max(np.abs(b[ok])))


@pytest.fixture(scope="module")
def y() -> pd.Series:
    return pd.read_csv(FIX / "tsfilter.csv")["y"]


@pytest.fixture(scope="module")
def stata() -> pd.DataFrame:
    return pd.read_csv(FIX / "tsfilter_Stata.csv")


CASES = {
    "hp1600": {"method": "hp", "smooth": 1600},
    "hp100": {"method": "hp", "smooth": "annual"},  # smooth(6.25)
    "bk": {"method": "bk", "low": 6, "high": 32, "K": 12},
    "bk_st": {"method": "bk", "stationary": True},
    "bk_2_8_3": {"method": "bk", "low": 2, "high": 8, "K": 3},
    "cf": {"method": "cf", "drift": False},  # Stata's default
    "cf_drift": {"method": "cf", "drift": True},
    "bw": {"method": "bw", "high": 32, "order": 2},
    "bw_o4": {"method": "bw", "high": 12, "order": 4},
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_stata_tsfilter(y: pd.Series, stata: pd.DataFrame, case: str) -> None:
    res = tsfilter(y, **CASES[case])
    assert rel(res.cycle, stata[f"c_{case}"]) < EXACT
    assert rel(res.trend, stata[f"t_{case}"]) < EXACT


def test_statsmodels_hp(y: pd.Series) -> None:
    from statsmodels.tsa.filters.hp_filter import hpfilter

    cycle, trend = hpfilter(y, 1600)
    res = tsfilter(y, method="hp")
    assert rel(res.cycle, cycle) < 1e-8
    assert rel(res.trend, trend) < 1e-8


def test_statsmodels_bk(y: pd.Series) -> None:
    from statsmodels.tsa.filters.bk_filter import bkfilter

    ref = bkfilter(y, 6, 32, 12).to_numpy()
    res = tsfilter(y, method="bk")
    assert rel(res.cycle.to_numpy()[12:-12], ref) < EXACT


@pytest.mark.parametrize("drift", [True, False])
def test_statsmodels_cf(y: pd.Series, drift: bool) -> None:
    from statsmodels.tsa.filters.cf_filter import cffilter

    cycle, trend = cffilter(y, 6, 32, drift=drift)
    res = tsfilter(y, method="cf", drift=drift)
    assert rel(res.cycle, cycle) < EXACT
    if drift:
        # statsmodels' trend is the series less the drift line less the
        # cycle; ours, like Stata's, is the series less the cycle
        n = len(y)
        line = np.arange(n) * (y.iloc[-1] - y.iloc[0]) / (n - 1)
        assert rel(res.trend - line, trend) < EXACT
    else:
        assert rel(res.trend, trend) < EXACT


def test_hamilton_is_the_ols_residual(y: pd.Series) -> None:
    import statsmodels.api as sm

    v = y.to_numpy()
    n, h, p = v.size, 8, 4
    start = h + p - 1
    X = sm.add_constant(
        np.column_stack([v[start - h - j : n - h - j] for j in range(p)])
    )
    fit = sm.OLS(v[start:], X).fit()
    res = tsfilter(y, method="hamilton")
    assert int(res.cycle.isna().sum()) == start
    # four lags of a series near 1000 are close to collinear: the two
    # least-squares codes agree to 1e-10 in the residual, not to 1e-13
    assert np.max(np.abs(res.cycle.to_numpy()[start:] - fit.resid)) < 1e-8
    assert np.max(np.abs(res.params["coef"] / fit.params - 1)) < 1e-8
