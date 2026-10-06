"""``sp.zivot_andrews`` against ``urca::ur.za`` 1.3-4.

Reference: ``_fixtures/zivot_andrews_R.json`` from
``_fixtures/_generate_zivot_andrews_R.R`` on the simulated
``_fixtures/zivot_andrews.csv``: the minimum t statistic, the break point,
the whole path of statistics, and the regression at the chosen break, for
three break models, two lag orders and two series. OLS on both sides: 1e-8.
``trim=0`` is urca's search over every date.
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
