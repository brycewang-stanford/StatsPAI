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

``cf`` with ``stationary=True`` is the one place where Stata is not the
reference: Stata 18 does not compute what its manual documents ([TS]
tsfilter cf, Methods and formulas: "all weights are set to the ideal
filter weight"). ``test_stata_cf_stationary_is_not_its_documented_formula``
rebuilds Stata's numbers from the rule it actually applies and shows the
size of the gap; ``sp.tsfilter`` follows the manual and is tested against
a direct evaluation of that formula in ``tests/test_tsfilter.py``.

The one-sided Hodrick-Prescott filter has no Stata counterpart; it is
compared with statsmodels' two-sided ``hpfilter`` on expanding windows.
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
    # fixed-length symmetric filter, Stata's smaorder()
    "cf_sma": {"method": "cf", "sma_order": 12, "drift": False},
    "cf_sma_dr": {"method": "cf", "sma_order": 12, "drift": True},
    "cf_sma5": {"method": "cf", "low": 2, "high": 8, "sma_order": 5, "drift": False},
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


def _ideal(low: float, high: float, count: int) -> np.ndarray:
    a, b = 2 * np.pi / high, 2 * np.pi / low
    j = np.arange(1, count + 1)
    return np.concatenate(
        ([(b - a) / np.pi], (np.sin(j * b) - np.sin(j * a)) / (np.pi * j))
    )


@pytest.mark.parametrize("drift", [False, True])
def test_stata_cf_stationary_is_not_its_documented_formula(
    y: pd.Series, stata: pd.DataFrame, drift: bool
) -> None:
    """What Stata 18 computes for ``tsfilter cf, stationary``.

    Manual: every weight is the ideal one. Stata: at the interior dates
    the first and the last observation get the ideal weight of the next
    smaller lag (``b_{T-t-1}`` on ``y_T`` where the manual has
    ``b_{T-t}``, ``b_{t-2}`` on ``y_1`` for ``b_{t-1}``), and the two end
    dates keep the sum-to-zero weights of the random-walk filter.
    """
    v = y.to_numpy()
    n = v.size
    if drift:
        v = v - np.arange(n) * (v[-1] - v[0]) / (n - 1)
    b = _ideal(6, 32, n - 1)
    idx = np.arange(n)
    doc = b[np.abs(idx[:, None] - idx[None, :])]
    theirs = doc.copy()
    for t in range(1, n - 1):
        theirs[t, n - 1] = b[n - 2 - t]
        theirs[t, 0] = b[t - 1]
    end = -(0.5 * b[0] + b[1 : n - 1].sum())
    theirs[0] = b[idx]
    theirs[0, 0], theirs[0, n - 1] = 0.5 * b[0], end
    theirs[n - 1] = b[n - 1 - idx]
    theirs[n - 1, n - 1], theirs[n - 1, 0] = 0.5 * b[0], end
    col = stata["c_cf_st_dr" if drift else "c_cf_st"]
    assert rel(theirs @ v, col) < EXACT  # Stata rebuilt exactly
    ours = tsfilter(y, method="cf", stationary=True, drift=drift).cycle
    np.testing.assert_allclose(ours, doc @ v, rtol=0, atol=1e-9)
    # the gap is not rounding: tens of units on a series near 1000 with a
    # cycle of standard deviation about 1.5
    gap = np.abs(ours.to_numpy() - col.to_numpy())
    assert gap[1:-1].max() > 10.0
    # at the end dates Stata returns its random-walk values unchanged
    rw = stata["c_cf_drift" if drift else "c_cf"]
    assert col.iloc[0] == rw.iloc[0] and col.iloc[-1] == rw.iloc[-1]


@pytest.mark.parametrize("drift", [False, True])
def test_stata_cf_stationary_smaorder_is_not_its_documented_formula(
    y: pd.Series, stata: pd.DataFrame, drift: bool
) -> None:
    """Manual: with ``smaorder(q) stationary`` the outermost weight is the
    ideal ``b_q``. Stata 18 uses ``b_{q-1}`` there."""
    v = y.to_numpy()
    n, q = v.size, 12
    if drift:
        v = v - np.arange(n) * (v[-1] - v[0]) / (n - 1)
    b = _ideal(6, 32, q)
    col = stata["c_cf_st_sma_dr" if drift else "c_cf_st_sma"]
    theirs = b.copy()
    theirs[q] = b[q - 1]
    ref = np.full(n, np.nan)
    ref[q : n - q] = np.convolve(v, np.concatenate((theirs[:0:-1], theirs)), "valid")
    assert rel(ref, col) < EXACT  # Stata rebuilt exactly
    res = tsfilter(y, method="cf", sma_order=q, stationary=True, drift=drift)
    np.testing.assert_allclose(res.params["weights"], b, rtol=0, atol=1e-15)
    doc = np.full(n, np.nan)
    doc[q : n - q] = np.convolve(v, np.concatenate((b[:0:-1], b)), "valid")
    np.testing.assert_allclose(res.cycle, doc, rtol=0, atol=1e-9)
    # the two differ by (b_{q-1} - b_q) (y_{t-q} + y_{t+q})
    gap = (res.cycle - col).to_numpy()[q : n - q]
    expected = (b[q] - b[q - 1]) * (v[: n - 2 * q] + v[2 * q :])
    np.testing.assert_allclose(gap, expected, rtol=0, atol=1e-9)
    assert np.abs(gap).min() > 1.0


def test_one_sided_hp_against_statsmodels_expanding(y: pd.Series) -> None:
    from statsmodels.tsa.filters.hp_filter import hpfilter

    res = tsfilter(y, method="hp", one_sided=True)
    v = y.to_numpy()
    ref = np.array([hpfilter(v[: t + 1], 1600)[1][-1] for t in range(2, v.size)])
    # statsmodels' sparse LU carries about 1e-10 (see the two-sided test)
    assert rel(res.trend.to_numpy()[2:], ref) < 1e-8
