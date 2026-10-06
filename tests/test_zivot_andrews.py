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


# --- lag order chosen by the data ----------------------------------------


def _ar_break(seed=3, T=160):
    """Trend-stationary AR(3) errors around a level shift at 90."""
    rng = np.random.default_rng(seed)
    e = np.zeros(T)
    for i in range(3, T):
        e[i] = 0.3 * e[i - 1] + 0.3 * e[i - 3] + rng.normal(scale=0.6)
    t = np.arange(T)
    return 0.08 * t + 2.5 * (t >= 90) + e


def _fit(y, k, drop, tb, model):
    """Level regression at one break date with statsmodels: t ratio of
    rho - 1, t ratio of the last lag, residual sum of squares, N, K."""
    import statsmodels.api as sm

    n = y.size
    dy = np.diff(y)
    t = np.arange(drop + 1, n + 1, dtype=float)
    cols = [np.ones(n - drop), y[drop - 1 : n - 1], t]
    cols += [dy[drop - 1 - j : n - 1 - j] for j in range(1, k + 1)]
    if tb is not None:
        if model in ("intercept", "both"):
            cols.append((t > tb).astype(float))
        if model in ("trend", "both"):
            cols.append(np.where(t > tb, t - tb, 0.0))
    X = np.column_stack(cols)
    f = sm.OLS(y[drop:], X).fit()
    last = abs(f.tvalues[2 + k]) if k else np.nan
    return (f.params[1] - 1) / f.bse[1], last, f.ssr, X.shape[0], X.shape[1]


def _choose(y, top, tb, model, how):
    fits = {k: _fit(y, k, top + 1, tb, model) for k in range(top + 1)}
    if how == "ttest":
        for k in range(top, 0, -1):
            if fits[k][1] >= 1.6448536269514722:
                return k
        return 0
    crit = {}
    for k, (_, _, ssr, m, K) in fits.items():
        pen = 2.0 if how == "aic" else np.log(m)
        crit[k] = m * np.log(ssr / m) + pen * K
    return min(crit, key=crit.get)


@pytest.mark.parametrize("how", ["aic", "bic", "ttest"])
@pytest.mark.parametrize("model", ["intercept", "both"])
def test_lag_order_at_every_break_matches_a_direct_evaluation(how, model):
    y = _ar_break()
    top = 5
    res = zivot_andrews(y, model=model, lags=how, max_lags=top, lag_selection="break")
    lo, hi = 24, 160 - 24  # floor(0.15 * 160) at each end
    assert res.path.index.min() == lo and res.path.index.max() == hi
    ks, ts = [], []
    for tb in range(lo, hi + 1):
        k = _choose(y, top, tb, model, how)
        ks.append(k)
        ts.append(_fit(y, k, k + 1, tb, model)[0])
    assert res.lags_path.tolist() == ks
    assert len(set(ks)) > 1  # the order does move with the break date
    # two OLS codes on the same regressions
    np.testing.assert_allclose(res.path.to_numpy(), ts, rtol=1e-8)
    best = int(np.argmin(ts))
    assert res.break_index == lo + best
    assert res.lags == ks[best]
    assert res.n_obs == 160 - ks[best] - 1
    assert res.coefficients.shape[0] == 3 + ks[best] + (2 if model == "both" else 1)
    assert res.lag_method == f"{how}, break"
    assert f"({how}, break)" in res.summary()


@pytest.mark.parametrize("how", ["aic", "bic", "ttest"])
def test_lag_order_chosen_once_is_the_no_break_choice(how):
    y = _ar_break(11)
    res = zivot_andrews(y, lags=how, max_lags=6)  # lag_selection='once'
    k = _choose(y, 6, None, "intercept", how)
    assert res.lags == k and res.lags_path is None
    fixed = zivot_andrews(y, lags=k)
    assert res.statistic == fixed.statistic
    assert res.break_index == fixed.break_index
    assert res.lag_method == f"{how}, once"
    assert fixed.lag_method == "fixed"


def test_ttest_level_moves_the_order():
    y = _ar_break(11)
    strict = zivot_andrews(y, lags="ttest", max_lags=8, lag_alpha=1e-12)
    loose = zivot_andrews(y, lags="ttest", max_lags=8, lag_alpha=0.999)
    assert strict.lags == 0 and loose.lags == 8


def test_zandrews_trim_rule_candidates():
    y = _ar_break()
    res = zivot_andrews(y, lags=2, trim=0.1, trim_rule="zandrews")
    # m = floor(0.1 * 160 + 0.49) = 16: from m + lags to T - m - 1
    assert res.path.index.min() == 18 and res.path.index.max() == 143
    plain = zivot_andrews(y, lags=2, trim=0.1)
    assert plain.path.index.min() == 16 and plain.path.index.max() == 144
    # same regressions where both search
    common = res.path.index.intersection(plain.path.index)
    np.testing.assert_array_equal(res.path[common], plain.path[common])


def test_new_option_errors():
    y = _ar_break()
    with pytest.raises(MethodIncompatibility, match="lag_selection"):
        zivot_andrews(y, lags="aic", lag_selection="each")
    with pytest.raises(MethodIncompatibility, match="lag_rule"):
        zivot_andrews(y, lags="aic", lag_rule="stata")
    with pytest.raises(MethodIncompatibility, match="trim_rule"):
        zivot_andrews(y, trim_rule="round")
    with pytest.raises(MethodIncompatibility, match="lag_alpha"):
        zivot_andrews(y, lags="ttest", lag_alpha=0.0)
    with pytest.raises(MethodIncompatibility, match="once"):
        zivot_andrews(y, lags="bic", lag_rule="zandrews", lag_selection="break")
    with pytest.raises(MethodIncompatibility, match="max_lags"):
        zivot_andrews(y, lags="bic", max_lags=-2)
    with pytest.raises(DataInsufficient):
        zivot_andrews(y[:12], lags="aic", max_lags=7, lag_selection="break")
