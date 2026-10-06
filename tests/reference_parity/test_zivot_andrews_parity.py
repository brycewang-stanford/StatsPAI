"""``sp.zivot_andrews`` against ``urca::ur.za`` 1.3-4.

Reference: ``_fixtures/zivot_andrews_R.json`` from
``_fixtures/_generate_zivot_andrews_R.R`` on the simulated
``_fixtures/zivot_andrews.csv``: the minimum t statistic, the break point,
the whole path of statistics, and the regression at the chosen break, for
three break models, two lag orders and two series. OLS on both sides: 1e-8.
``trim=0`` is urca's search over every date.

Second reference: the Stata command ``zandrews`` 1.0.5 (Baum, SSC) under
Stata 18, ``_fixtures/zivot_andrews_Stata.csv`` and ``..._Stata_path.csv``
from ``_generate_zivot_andrews_Stata.do``: 48 cases over two series, three
break models, fixed lags and its AIC / BIC / t-test choice, three trims.
The chosen lag, the break observation and the list of candidate dates are
compared exactly; the statistic at every candidate date to 1e-8 (OLS on
both sides, observed 5e-13).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.timeseries.zivot_andrews import zivot_andrews

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "zivot_andrews_R.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(FIX / "zivot_andrews.csv")


@pytest.mark.parametrize(
    "case", REF["cases"], ids=lambda c: f"{c['series']}-{c['model']}-{c['lag']}"
)
def test_matches_urca(case):
    res = zivot_andrews(
        DATA, case["series"], model=case["model"], lags=case["lag"], trim=0
    )
    assert res.statistic == pytest.approx(case["stat"], rel=1e-8)
    assert res.break_index == case["bpoint"]
    assert list(res.critical_values.values()) == case["cval"]
    # urca reports the path from the first candidate on; ours is indexed
    # by the number of observations before the break
    theirs = np.array(case["tstats"], dtype=float)
    ours = res.path.to_numpy()[: theirs.size]
    both = np.isfinite(theirs) & np.isfinite(ours)
    assert both.sum() > 150
    np.testing.assert_allclose(ours[both], theirs[both], rtol=1e-7)
    # regression at the break: urca orders (const, y.l1, trend, lags, du, dt)
    np.testing.assert_allclose(res.coefficients["coef"], case["coef"], rtol=1e-7)
    np.testing.assert_allclose(res.coefficients["se"], case["se"], rtol=1e-7)


STATA = pd.read_csv(FIX / "zivot_andrews_Stata.csv")
STATA_PATH = pd.read_csv(FIX / "zivot_andrews_Stata_path.csv")
_METHOD = {"AIC": "aic", "BIC": "bic", "TTest": "ttest"}


@pytest.mark.parametrize(
    "row",
    [r for r in STATA.itertuples()],
    ids=lambda r: f"{r.series}-{r.model}-{r.lagmethod}{r.maxlags}-{r.trim}",
)
def test_matches_stata_zandrews(row):
    kw = {"model": row.model, "trim": row.trim, "trim_rule": "zandrews"}
    if row.lagmethod == "input":
        kw["lags"] = int(row.maxlags)
    else:
        # zandrews ignores maxlags() here and searches up to int(T^0.25),
        # which is the default of lag_rule='zandrews'
        kw.update(lags=_METHOD[row.lagmethod], lag_rule="zandrews", lag_alpha=row.level)
    res = zivot_andrews(DATA, row.series, **kw)
    assert res.lags == row.bestlag
    assert res.break_index + 1 == row.tminobs  # first observation after
    assert res.n_obs == row.nobs
    assert res.statistic == pytest.approx(row.tmin, rel=1e-8)
    theirs = STATA_PATH[STATA_PATH["id"] == row.id]
    # same candidate dates, same statistic at each of them
    assert np.array_equal(res.path.index.to_numpy() + 1, theirs["obs"].to_numpy())
    np.testing.assert_allclose(res.path.to_numpy(), theirs["t"].to_numpy(), rtol=1e-8)


def test_stata_fixture_exercises_the_lag_choice():
    chosen = STATA[STATA["lagmethod"] != "input"].groupby("lagmethod")["bestlag"]
    assert all(v.nunique() >= 2 for _, v in chosen)
