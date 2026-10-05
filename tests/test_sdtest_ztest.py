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


# ------------------------------------------------- ttest from summary numbers
def test_ttest_summary_form_equals_the_data_form(df):
    two = df[df["g"] < 2]
    a, b = two.loc[two.g == 0, "y"].dropna(), two.loc[two.g == 1, "y"]
    for kw in ({}, {"unequal": True}, {"welch": True}):
        data = sp.ttest(two, "y", by="g", **kw)
        summ = sp.ttest(
            n=(a.size, b.size), mean=(a.mean(), b.mean()), sd=(a.std(), b.std()), **kw
        )
        assert np.isclose(data.statistic, summ.statistic)
        assert np.isclose(data.df, summ.df)
        assert np.isclose(data.pvalue, summ.pvalue)
    y = df["y"].dropna()
    one = sp.ttest(n=y.size, mean=y.mean(), sd=y.std(), mu=4.5)
    ref = sp.ttest(df, "y", mu=4.5)
    assert np.isclose(one.statistic, ref.statistic) and one.df == ref.df
    assert np.allclose(one.ci, ref.ci)


def test_ttesti_matches_stata():
    # Stata 18: ttesti 10 88 1.14 85 -> t = 8.3218, 9 df,
    # 95% CI [87.18449, 88.81551]
    res = sp.stata("ttesti 10 88 1.14 85")
    assert round(res.statistic, 4) == 8.3218 and res.df == 9
    assert np.allclose(res.ci, (87.18449, 88.81551), atol=5e-6)
    two = sp.from_stata("ttesti 52 19.8 4.7 22 24.8 6.6, unequal")
    assert two["arguments"] == {
        "n": (52, 22), "mean": (19.8, 24.8), "sd": (4.7, 6.6), "unequal": True,
    }  # fmt: skip


def test_ttest_summary_refusals(df):
    with pytest.raises(MethodIncompatibility, match="all of n="):
        sp.ttest(n=10, mean=1.0)
    with pytest.raises(MethodIncompatibility, match="used in place"):
        sp.ttest(df, "y", n=10, mean=1.0, sd=1.0)
    with pytest.raises(MethodIncompatibility, match="name columns of data"):
        sp.ttest(y="y", n=10, mean=1.0, sd=1.0)
    with pytest.raises(MethodIncompatibility, match="same one or two"):
        sp.ttest(n=(10, 12), mean=1.0, sd=1.0)
    with pytest.raises(MethodIncompatibility, match="at least 2"):
        sp.ttest(n=1, mean=1.0, sd=1.0)
    with pytest.raises(MethodIncompatibility, match="names the variable"):
        sp.ttest(df)


# ------------------------------------------------------ tobit: which sides
@pytest.fixture(scope="module")
def top_censored():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    latent = 1 + x + rng.normal(size=500)
    return pd.DataFrame({"x": x, "y": np.minimum(latent, 2.0)})


def test_tobit_censored_from_above_only(top_censored):
    """Stata 18, ``tobit y x, ul(2)`` iterated to 1e-12 on these bytes:
    b[x] = 0.998381062153 (0.045686846579), _cons = 0.925894762565,
    log likelihood -605.380110446110, 107 right-censored."""
    fit = sp.tobit(top_censored, y="y", x=["x"], ll=None, ul=2.0)
    table = fit.detail.set_index("variable")
    assert np.isclose(table.loc["x", "coefficient"], 0.998381062153, rtol=1e-8)
    assert np.isclose(table.loc["x", "se"], 0.045686846579, rtol=1e-6)
    assert np.isclose(table.loc["const", "coefficient"], 0.925894762565, rtol=1e-8)
    assert np.isclose(fit.model_info["log_likelihood"], -605.380110446110, rtol=1e-10)
    assert fit.model_info["n_censored"] == 107
    # the default lower limit of 0 would also censor every y <= 0
    both = sp.tobit(top_censored, y="y", x=["x"], ul=2.0)
    assert both.model_info["n_censored"] > 107


def test_tobit_translation_censors_only_the_sides_stata_does(top_censored):
    out = sp.from_stata("tobit y x, ul(2)")
    assert out["arguments"]["ll"] is None and out["arguments"]["ul"] == 2.0
    via = sp.stata("tobit y x, ul(2)", data=top_censored)
    assert via.model_info["n_censored"] == 107
    # no limit at all: Stata fits the uncensored normal model (0.797519891384)
    plain = sp.stata("tobit y x", data=top_censored)
    slope = plain.detail.set_index("variable").loc["x", "coefficient"]
    assert np.isclose(slope, 0.797519891384, rtol=1e-8)
    assert sp.from_stata("tobit y x, ll(0)")["arguments"] == {
        "y": "y", "x": ["x"], "ll": 0.0, "ul": None,
    }  # fmt: skip
    assert sp.from_stata("tobit y x, ll")["ok"] is False
    assert sp.from_stata("tobit y x, ll(lim)")["ok"] is False


# ------------------------------------------- abbreviated variable names
def test_variable_abbreviations_inside_options():
    rng = np.random.default_rng(3)
    n = 400
    frame = pd.DataFrame(
        {
            "education": rng.normal(12, 2, n),
            "married": rng.integers(0, 2, n).astype(float),
            "county": rng.integers(0, 10, n),
        }
    )
    u = rng.normal(size=n)
    work = 0.2 * frame["education"] - 2.4 + frame["married"] + u > 0
    frame["wage"] = np.where(
        work, 1 + 0.8 * frame["education"] + u + rng.normal(size=n), np.nan
    )
    cols = list(frame.columns)
    out = sp.from_stata("heckman wage educ, select(marr educ) twostep", columns=cols)
    assert out["arguments"]["x"] == ["education"]
    assert out["arguments"]["z"] == ["married", "education"]
    full = sp.stata(
        "heckman wage education, select(married education) twostep", data=frame
    )
    short = sp.stata("heckman wage educ, select(marr educ) twostep", data=frame)
    assert np.allclose(full.detail["coefficient"], short.detail["coefficient"])
    clustered = sp.from_stata("regress wage educ, vce(cluster coun)", columns=cols)
    assert clustered["arguments"]["cluster"] == "county"
    # an abbreviation that fits two variables is refused, as in Stata
    two = frame.assign(marital=1.0)
    bad = sp.from_stata(
        "heckman wage educ, select(mar educ)", columns=list(two.columns)
    )
    assert bad["ok"] is False and "ambiguous" in bad["error"]
    # without the columns nothing is guessed
    blind = sp.from_stata("heckman wage educ, select(marr educ) twostep")
    assert blind["arguments"]["z"] == ["marr", "educ"]
