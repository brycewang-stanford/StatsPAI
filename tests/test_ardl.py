"""sp.ardl (AR / ADL forecasting regressions) and the partial, robust sup-F
of sp.structural_break.

References: ``statsmodels.tsa.ardl.ARDL`` (an independent implementation of
the same regression), closed forms computed here, and known
data-generating processes. The published-number check, against the RATS
output of Stock & Watson's chapter 15, is the opt-in
``tests/external_parity/test_stock_watson_4e_ch15.py``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.timeseries.structural_break import hansen_supf_pvalue


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(20261003)
    n = 260
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.normal()
        y[t] = 0.5 + 0.4 * y[t - 1] - 0.2 * y[t - 2] + 0.7 * x[t - 1] + rng.normal()
    idx = pd.period_range("1955Q1", periods=n, freq="Q")
    return pd.DataFrame({"y": y, "x": x}, index=idx)


# ------------------------------------------------------------- estimation
@pytest.mark.parametrize("p, q", [(1, 1), (2, 1), (2, 3), (4, 2)])
def test_coefficients_match_statsmodels_ardl(df, p, q):
    ARDL = pytest.importorskip("statsmodels.tsa.ardl").ARDL
    plain = df.reset_index(drop=True)
    ref = ARDL(plain.y, p, plain[["x"]], q, causal=True, trend="c").fit()
    got = sp.ardl(plain, "y", "x", lags=p, x_lags=q, vce="nonrobust")
    want = {"Intercept": ref.params["const"]}
    want.update({f"y_L{k}": ref.params[f"y.L{k}"] for k in range(1, p + 1)})
    want.update({f"x_L{k}": ref.params[f"x.L{k}"] for k in range(1, q + 1)})
    assert list(got.params.index) == list(want)
    # same OLS problem on the same rows
    np.testing.assert_allclose(got.params.to_numpy(), list(want.values()), rtol=1e-9)
    # (statsmodels' .nobs counts from the AR order alone; the rows used are
    # the ones the deepest lag leaves, on both sides)
    assert got.nobs == len(plain) - max(p, q)
    assert got.method == f"ADL({p}, {q})"


def test_ar_is_ols_on_lagged_columns(df):
    got = sp.ardl(df, "y", lags=2, vce="hc0")
    lagged = df.assign(y_L1=df.y.shift(1), y_L2=df.y.shift(2)).dropna()
    want = sp.regress("y ~ y_L1 + y_L2", data=lagged, robust="hc0")
    np.testing.assert_array_equal(got.params.to_numpy(), want.params.to_numpy())
    np.testing.assert_array_equal(got.std_errors.to_numpy(), want.std_errors.to_numpy())
    assert got.method == "AR(2)" and got.nobs == len(df) - 2
    # information criteria as defined: ln(SSR/T) + K * penalty / T
    n, k = got.nobs, 3
    np.testing.assert_allclose(got.bic, np.log(got.rss / n) + k * np.log(n) / n)
    np.testing.assert_allclose(got.aic, np.log(got.rss / n) + k * 2 / n)
    np.testing.assert_allclose(got.ser, np.sqrt(got.rss / (n - k)))


def test_sample_fixes_the_rows_while_lags_come_from_before(df):
    res = sp.ardl(df, "y", "x", lags=2, x_lags=2, sample=("1962Q1", "2015Q4"))
    assert res.nobs == len(pd.period_range("1962Q1", "2015Q4", freq="Q"))
    # the forecast is for the period after the sample, from data in hand
    b = res.params
    hand = (
        b["Intercept"]
        + b["y_L1"] * df.y["2015Q4"]
        + b["y_L2"] * df.y["2015Q3"]
        + b["x_L1"] * df.x["2015Q4"]
        + b["x_L2"] * df.x["2015Q3"]
    )
    fc = res.forecast()
    np.testing.assert_allclose(fc["forecast"].item(), hand, rtol=1e-12)
    np.testing.assert_allclose(fc["rmsfe"].item(), res.ser)
    np.testing.assert_allclose(
        fc["upper"].item() - fc["lower"].item(), 2 * 1.959963984540054 * res.ser
    )


def test_forecast_matches_statsmodels(df):
    ARDL = pytest.importorskip("statsmodels.tsa.ardl").ARDL
    plain = df.reset_index(drop=True)
    ref = ARDL(plain.y, 2, plain[["x"]], 1, causal=True, trend="c").fit()
    got = sp.ardl(plain, "y", "x", lags=2, x_lags=1)
    b = ref.params
    hand = (
        b["const"]
        + b["y.L1"] * plain.y.iloc[-1]
        + b["y.L2"] * plain.y.iloc[-2]
        + b["x.L1"] * plain.x.iloc[-1]
    )
    np.testing.assert_allclose(got.forecast()["forecast"].item(), hand, rtol=1e-9)


# ---------------------------------------------------------- lag selection
@pytest.mark.parametrize("criterion", ["bic", "aic"])
def test_lag_selection_uses_one_sample_and_picks_the_minimum(df, criterion):
    res = sp.ardl(df, "y", lags=criterion, max_lags=6)
    table = res.ic_table
    assert (table["nobs"] == len(df) - 6).all()
    assert res.lags == int(table[criterion].idxmin())
    # each row is the criterion of that order on the common rows
    rows = (df.index[6], df.index[-1])
    for p in (0, 2, 5):
        one = sp.ardl(df, "y", lags=p, sample=rows)
        np.testing.assert_allclose(table.loc[p, criterion], getattr(one, criterion))


def test_bic_finds_the_true_order_and_aic_never_picks_fewer(df):
    bic = sp.ardl(df, "y", "x", lags="bic", x_lags=1, max_lags=6)
    aic = sp.ardl(df, "y", "x", lags="aic", x_lags=1, max_lags=6)
    assert bic.lags == 2  # the DGP is AR(2) in y given x
    assert aic.lags >= bic.lags


def test_x_lags_same_ties_the_two_orders(df):
    res = sp.ardl(df, "y", "x", lags="bic", x_lags="same", max_lags=4)
    assert res.x_lags == {"x": res.lags}
    assert res.method == f"ADL({res.lags}, {res.lags})"


# ---------------------------------------------------------------- granger
def test_granger_is_the_joint_test_on_the_lags_of_x(df):
    res = sp.ardl(df, "y", "x", lags=2, x_lags=2, vce="hc1")
    got = res.granger()
    want = sp.test(res.model, "x_L1 x_L2")
    assert got == want
    assert got["pvalue"] < 1e-6  # x[t-1] has a coefficient of 0.7


def test_granger_does_not_reject_for_an_unrelated_series():
    rng = np.random.default_rng(8)
    reject = 0
    for _ in range(300):
        e = rng.normal(size=160)
        y = np.zeros(160)
        for t in range(1, 160):
            y[t] = 0.5 * y[t - 1] + e[t]
        data = pd.DataFrame({"y": y, "noise": rng.normal(size=160)})
        res = sp.ardl(data, "y", "noise", lags=1, x_lags=2)
        reject += res.granger()["pvalue"] < 0.05
    # nominal 5%; binomial SE with 300 draws is 0.013
    assert 0.02 <= reject / 300 <= 0.09


# ------------------------------------------------------------------- poos
def test_poos_is_recursive_re_estimation(df):
    res = sp.ardl(df, "y", "x", lags=1, x_lags=1)
    out = res.poos(start="2010Q1")
    assert out.index[0] == pd.Period("2010Q1") and out.index[-1] == df.index[-1]
    # recompute three of the forecasts with data before the date only
    for date in (out.index[0], out.index[7], out.index[-1]):
        pos = df.index.get_loc(date)
        past = df.iloc[:pos]
        lagged = past.assign(y_L1=past.y.shift(1), x_L1=past.x.shift(1)).dropna()
        fit = sp.regress("y ~ y_L1 + x_L1", data=lagged)
        hand = (
            fit.params["Intercept"]
            + fit.params["y_L1"] * df.y.iloc[pos - 1]
            + fit.params["x_L1"] * df.x.iloc[pos - 1]
        )
        np.testing.assert_allclose(out.loc[date, "forecast"], hand, rtol=1e-10)
    err = out["error"].to_numpy()
    np.testing.assert_allclose(out.attrs["rmsfe"], np.sqrt(np.mean(err**2)))
    np.testing.assert_allclose(out.attrs["bias"], err.mean())
    assert out.attrs["n_forecasts"] == len(out)


def test_poos_rmsfe_is_near_the_innovation_sd_when_the_model_is_stable(df):
    out = sp.ardl(df, "y", "x", lags=2, x_lags=1).poos(start="1990Q1")
    # the DGP's innovations have standard deviation 1
    assert 0.85 < out.attrs["rmsfe"] < 1.2
    assert abs(out.attrs["bias"]) < 3 * out.attrs["bias_se"]


def test_poos_bias_exposes_a_break_the_in_sample_fit_hides():
    rng = np.random.default_rng(5)
    n = 240
    y = np.zeros(n)
    for t in range(1, n):
        mean = 2.0 if t < 180 else 0.0  # the mean falls by 2 at t = 180
        y[t] = mean * 0.5 + 0.5 * y[t - 1] + rng.normal(scale=0.5)
    res = sp.ardl(pd.DataFrame({"y": y}), "y", lags=1)
    out = res.poos(start=180)
    assert out.attrs["bias"] < -3 * out.attrs["bias_se"]
    rolling = res.poos(start=180, window="rolling")
    assert len(rolling) == len(out)


# -------------------------------------------------------------- interface
def test_time_column_sorts_and_indexes(df):
    frame = df.reset_index(names="quarter").sample(frac=1, random_state=1)
    a = sp.ardl(frame, "y", "x", lags=1, time="quarter")
    b = sp.ardl(df, "y", "x", lags=1)
    np.testing.assert_array_equal(a.params.to_numpy(), b.params.to_numpy())
    assert "ADL(1, 1)" in a.summary()


def test_refusals(df):
    with pytest.raises(MethodIncompatibility, match="not in data"):
        sp.ardl(df, "nope")
    with pytest.raises(MethodIncompatibility, match="unknown trend"):
        sp.ardl(df, "y", trend="quadratic")
    with pytest.raises(MethodIncompatibility, match="unknown lag rule"):
        sp.ardl(df, "y", lags="hqic")
    gap = df.copy()
    gap.iloc[100, 0] = np.nan
    with pytest.raises(MethodIncompatibility, match="gap"):
        sp.ardl(gap, "y", lags=1)
    with pytest.raises(MethodIncompatibility, match="collide"):
        sp.ardl(df.assign(y_L1=df.x), "y", "y_L1", lags=1)
    # an unused column of that name is not a problem
    sp.ardl(df.assign(y_L1=1.0), "y", lags=1)
    with pytest.raises(DataInsufficient):
        sp.ardl(df.iloc[:6], "y", lags=4)
    with pytest.raises(MethodIncompatibility, match="x at date t"):
        sp.ardl(df, "y", "x", contemporaneous=True).forecast()
    with pytest.raises(MethodIncompatibility, match="no additional regressor"):
        sp.ardl(df, "y", lags=1).granger()
    with pytest.raises(DataInsufficient, match="too few observations before"):
        sp.ardl(df, "y", lags=1).poos(start=df.index[3])
    with pytest.raises(MethodIncompatibility, match="not a date"):
        sp.ardl(df, "y", lags=1).poos(start="2099Q1")


# ------------------------------------------------- partial, robust sup-F
@pytest.fixture(scope="module")
def lagged(df):
    out = df.assign(y1=df.y.shift(1), x1=df.x.shift(1)).dropna()
    return out.reset_index(drop=True)


def test_all_coefficients_nonrobust_is_the_existing_sup_f(lagged):
    old = sp.structural_break(lagged, y="y", x=["y1", "x1"], method="sup-f")
    new = sp.structural_break(
        lagged, y="y", x=["y1", "x1"], method="sup-f", break_vars=["const", "y1", "x1"]
    )
    np.testing.assert_allclose(new.f_stats, old.f_stats, rtol=1e-10)
    np.testing.assert_allclose(new.p_values, old.p_values, rtol=1e-8)
    assert new.sup_break == old.sup_break


def test_partial_robust_statistic_at_one_date_by_hand(lagged):
    res = sp.structural_break(
        lagged, y="y", x=["y1", "x1"], method="sup-f",
        break_vars=["const", "x1"], vce="hc1",
    )  # fmt: skip
    t = int(res.candidate_breaks[10])
    data = lagged.assign(d=(np.arange(len(lagged)) >= t).astype(float))
    data["dx1"] = data.d * data.x1
    fit = sp.regress("y ~ y1 + x1 + d + dx1", data=data, robust="hc1")
    want = sp.test(fit, "d dx1")["statistic"]
    np.testing.assert_allclose(res.f_path[10], want, rtol=1e-9)
    assert res.n_restrictions == 2 and res.vce == "hc1"
    assert res.f_stats == res.f_path.max()


def test_robust_sup_f_holds_size_under_heteroskedasticity():
    rng = np.random.default_rng(31)
    plain = robust = 0
    reps, n = 300, 200
    for _ in range(reps):
        x = rng.normal(size=n)
        y = 1 + 0.5 * x + rng.normal(size=n) * (0.3 + 1.5 * x**2)
        data = pd.DataFrame({"y": y, "x": x})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = sp.structural_break(data, y="y", x=["x"], method="sup-f")
            b = sp.structural_break(data, y="y", x=["x"], method="sup-f", vce="hc1")
        plain += a.p_values < 0.05
        robust += b.p_values < 0.05
    # no break in the DGP: the classical statistic over-rejects, the robust
    # one stays near 5% (binomial SE 0.013 with 300 draws)
    assert plain / reps > 0.15
    assert robust / reps < 0.10


def test_sup_f_finds_a_break_in_the_named_coefficient():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=n)
    slope = np.where(np.arange(n) < 180, 0.5, 1.5)
    data = pd.DataFrame({"x": x, "y": 1 + slope * x + rng.normal(size=n)})
    res = sp.structural_break(
        data, y="y", x=["x"], method="sup-f", break_vars=["x"], vce="hc1"
    )
    assert res.p_values < 0.001 and abs(res.sup_break - 180) <= 10


def test_p_value_agrees_with_the_textbook_critical_values():
    # Stock & Watson's QLR table, 15% trimming, q = 3 restrictions: 4.71 at
    # 5% and 6.02 at 1% (the values drawn on their Figure 15.5). Hansen's
    # approximation should put those statistics near those levels.
    lam = ((1 - 0.15) / 0.15) ** 2
    assert abs(hansen_supf_pvalue(3 * 4.71, 3, lam) - 0.05) < 0.01
    assert abs(hansen_supf_pvalue(3 * 6.02, 3, lam) - 0.01) < 0.003


def test_partial_options_are_refused_where_they_do_not_apply(lagged):
    with pytest.raises(MethodIncompatibility, match="sup-f"):
        sp.structural_break(lagged, y="y", x=["x1"], break_vars=["x1"])
    with pytest.raises(MethodIncompatibility, match="must be drawn from"):
        sp.structural_break(lagged, y="y", x=["x1"], method="sup-f", break_vars=["zz"])
    with pytest.raises(MethodIncompatibility, match="vce must be"):
        sp.structural_break(lagged, y="y", x=["x1"], method="sup-f", vce="hac")
