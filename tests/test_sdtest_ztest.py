"""Edge cases of ``sp.sdtest`` / ``sp.ztest`` and of the ``data=`` forms of
``sp.arima`` / ``sp.garch``. The numbers are checked against Stata in
``tests/reference_parity/test_textbook_syllabus_stata_parity.py``."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    out = pd.DataFrame({"g": np.repeat([0, 1, 2], 30)})
    out["y"] = rng.normal(5.0, 2.0, size=90)
    out["x"] = rng.normal(4.0, 3.0, size=90)
    out.loc[3, "y"] = np.nan
    return out


def test_sdtest_matches_its_definition(df):
    y = df["y"].dropna().to_numpy()
    res = sp.sdtest(df, "y", sd0=2.0)
    expected = (y.size - 1) * y.var(ddof=1) / 4.0
    assert np.isclose(res.statistic, expected)
    assert np.isclose(res.pvalue_less, stats.chi2.cdf(expected, y.size - 1))
    assert res.n_obs == 89 and res.df == 88.0
    assert res.ci[0] < res.estimate < res.ci[1]
    assert "chi2" in res.summary()


def test_sdtest_two_variables_use_their_own_rows(df):
    res = sp.sdtest(df, "y", other="x")
    assert res.df == (88.0, 89.0)
    assert np.isclose(res.statistic, df["y"].var() / df["x"].var())
    assert list(res.groups.index) == ["y", "x", "combined"]
    # a ratio is tested against one; the interval is for the ratio of sds
    assert res.null == 1.0 and res.ci[0] < res.estimate < res.ci[1]


def test_sdtest_summary_form_equals_the_data_form(df):
    two = df[df["g"] < 2]
    by = sp.sdtest(two, "y", by="g")
    a, b = two.loc[two.g == 0, "y"].dropna(), two.loc[two.g == 1, "y"]
    summ = sp.sdtest(n=(a.size, b.size), sd=(a.std(), b.std()))
    assert np.isclose(by.statistic, summ.statistic)
    assert np.isclose(by.pvalue, summ.pvalue)


def test_sdtest_refusals(df):
    with pytest.raises(MethodIncompatibility, match="needs sd0"):
        sp.sdtest(df, "y")
    with pytest.raises(MethodIncompatibility, match="exactly two groups"):
        sp.sdtest(df, "y", by="g")
    with pytest.raises(MethodIncompatibility, match="not both"):
        sp.sdtest(df, "y", by="g", other="x")
    with pytest.raises(MethodIncompatibility, match="ratio of one"):
        sp.sdtest(df[df.g < 2], "y", by="g", sd0=2)
    with pytest.raises(MethodIncompatibility, match="give data and y"):
        sp.sdtest()
    with pytest.raises(MethodIncompatibility, match="positive number"):
        sp.sdtest(n=10, sd=-1.0, sd0=2)
    with pytest.raises(MethodIncompatibility, match="at least 2"):
        sp.sdtest(n=1, sd=1.0, sd0=2)
    with pytest.raises(MethodIncompatibility, match="summary statistics"):
        sp.sdtest(df, "y", sd0=2, n=10)
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.sdtest(df, "y", sd0=2, alpha=1.5)
    with pytest.raises(MethodIncompatibility, match="no variation"):
        sp.sdtest(pd.DataFrame({"y": [1.0, 1.0, 1.0]}), "y", sd0=1)


def test_ztest_matches_its_definition(df):
    y = df["y"].dropna().to_numpy()
    res = sp.ztest(df, "y", mu=5.0, sd=2.0)
    z = (y.mean() - 5.0) / (2.0 / np.sqrt(y.size))
    assert np.isclose(res.statistic, z)
    assert np.isclose(res.pvalue, 2 * stats.norm.sf(abs(z)))
    assert res.df == float("inf") and res.statistic_name == "z"
    assert "z = " in res.summary() and "degrees of freedom" not in res.summary()
    # the known sd, not the sample one, is in the table
    assert res.groups.loc["y", "sd"] == 2.0


def test_ztest_two_independent_columns(df):
    res = sp.ztest(df, "y", other="x", sd=(2.0, 3.0))
    se = np.sqrt(4.0 / 89 + 9.0 / 90)
    assert np.isclose(res.se, se)
    assert np.isclose(res.estimate, df["y"].mean() - df["x"].mean())


def test_ztest_refusals(df):
    with pytest.raises(MethodIncompatibility, match="positive number"):
        sp.ztest(df, "y", sd=0.0)
    with pytest.raises(MethodIncompatibility, match="one per sample"):
        sp.ztest(df, "y", sd=(1.0, 2.0))
    with pytest.raises(MethodIncompatibility, match="mean= is required"):
        sp.ztest(n=10, sd=1.0)
    with pytest.raises(MethodIncompatibility, match="exactly two groups"):
        sp.ztest(df, "y", by="g", sd=1.0)


def test_a_t_test_still_reports_t(df):
    res = sp.ttest(df, "y", mu=5.0)
    assert res.statistic_name == "t" and "t = " in res.summary()


def test_arima_and_garch_take_a_column_of_data():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"y": 3.0 + rng.normal(size=150)})
    frame["dy"] = frame["y"].diff()  # a leading missing value
    by_name = sp.arima("dy", order=(1, 0, 0), data=frame)
    by_array = sp.arima(frame["dy"].dropna().to_numpy(), order=(1, 0, 0))
    assert np.allclose(by_name.params, by_array.params)
    assert by_name.n == 149
    vol = sp.garch("dy", data=frame)
    assert np.allclose(vol.params, sp.garch(frame["dy"].dropna()).params)
    with pytest.raises(MethodIncompatibility, match="pass data="):
        sp.arima("dy", order=(1, 0, 0))
    with pytest.raises(MethodIncompatibility, match="name of one of its columns"):
        sp.arima("nope", order=(1, 0, 0), data=frame)
    with pytest.raises(MethodIncompatibility, match="pass data="):
        sp.garch("dy")
    with pytest.raises(MethodIncompatibility, match="trend must be"):
        sp.arima(frame["y"], order=(1, 0, 0), trend="ct")


def test_heckman_needs_a_selection_equation():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)})
    with pytest.raises(MethodIncompatibility, match="selection equation"):
        sp.heckman(frame, y="y", x=["x"])
