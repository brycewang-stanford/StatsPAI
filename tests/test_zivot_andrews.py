"""``sp.zivot_andrews``: behaviour and argument checks."""

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.timeseries.zivot_andrews import zivot_andrews


def _broken_trend(seed=0, T=200, at=120):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    return 0.05 * t + 3.0 * (t >= at) + rng.normal(scale=0.5, size=T)


def test_level_shift_is_found_and_the_unit_root_rejected():
    res = zivot_andrews(_broken_trend(), model="intercept")
    assert res.reject["1%"]
    assert abs(res.break_index - 120) <= 3
    assert "rejected at the 5% level" in res.summary()


def test_random_walk_rejects_at_about_the_nominal_rate():
    # 5% test on 300 random walks with drift, T = 150: the rejection rate
    # stays below 10% (binomial s.e. at 0.05 is 0.013; the asymptotic
    # values are somewhat liberal at this length).
    rejections = 0
    for s in range(300):
        rng = np.random.default_rng(500 + s)
        y = np.cumsum(0.1 + rng.normal(size=150))
        rejections += zivot_andrews(y, model="both").reject["5%"]
    assert rejections / 300 < 0.10


def test_trimming_restricts_the_candidates():
    y = _broken_trend(1)
    full = zivot_andrews(y, trim=0)
    cut = zivot_andrews(y, trim=0.2)
    assert full.path.index.min() == 1 and full.path.index.max() == 199
    assert cut.path.index.min() == 40 and cut.path.index.max() == 160
    assert cut.statistic >= full.statistic - 1e-12


def test_dataframe_time_label_and_lag_selection():
    y = _broken_trend(2)
    df = pd.DataFrame({"year": np.arange(1900, 2100), "gdp": y})
    res = zivot_andrews(df, "gdp", time="year", lags="bic", max_lags=4)
    assert res.break_label == 1900 + res.break_index - 1
    assert 0 <= res.lags <= 4
    assert list(res.coefficients.index[:3]) == ["_cons", "L.y", "trend"]


def test_leading_missing_values_are_trimmed_and_gaps_refused():
    y = _broken_trend(3)
    padded = np.concatenate([[np.nan, np.nan], y])
    assert zivot_andrews(padded).statistic == pytest.approx(
        zivot_andrews(y).statistic, rel=1e-12
    )
    gap = y.copy()
    gap[50] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing"):
        zivot_andrews(gap)


def test_argument_errors():
    y = _broken_trend(4)
    with pytest.raises(MethodIncompatibility, match="model"):
        zivot_andrews(y, model="level")
    with pytest.raises(MethodIncompatibility, match="trim"):
        zivot_andrews(y, trim=0.6)
    with pytest.raises(MethodIncompatibility, match="lags"):
        zivot_andrews(y, lags=-1)
    with pytest.raises(MethodIncompatibility, match="lags"):
        zivot_andrews(y, lags="hq")
    with pytest.raises(DataInsufficient):
        zivot_andrews(y[:8], lags=3)
