"""What ``sp.stata`` does between estimation lines: data management, weights,
stored estimates, programs and ``simulate``, and the post-estimation commands
an introductory course uses.

Every test states the Stata rule it pins. Numbers against real Stata are in
``tests/reference_parity/test_textbook_methods_stata_parity.py``; here the
checks are identities (a weighted fit equals the fit on the expanded rows)
and known truths.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_run import StataSession
from statspai.exceptions import MethodIncompatibility


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def session(data=None):
    s = StataSession(data)
    run = s.run
    s.run = lambda line: quiet(run, line)  # type: ignore[method-assign]
    return s


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 200
    out = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "g": rng.integers(1, 5, n).astype(float),
            "n": rng.integers(1, 4, n).astype(float),
            "t": np.arange(1, n + 1, dtype=float),
        }
    )
    out["y"] = 1 + 0.5 * out.x - 0.3 * out.z + rng.normal(size=n)
    out["b"] = (out.y > 1).astype(float)
    return out


# --------------------------------------------------------------- data steps
def test_gsort_puts_missing_first_when_descending(df):
    data = df.head(6).copy()
    data.loc[2, "x"] = np.nan
    s = session(data)
    s.run("gsort -x")
    assert np.isnan(s.data["x"].iloc[0])  # missing is the largest value
    assert s.data["x"].iloc[1:].is_monotonic_decreasing
    s.run("gsort +g -x")
    assert s.data["g"].is_monotonic_increasing


def test_rename_and_tabulate_generate(df):
    s = session(df)
    s.run("rename z w")
    assert "w" in s.data.columns and "z" not in s.data.columns
    s.run("tabulate g, generate(gd)")
    table = s.output
    assert list(table.columns) == ["Freq.", "Percent", "Cum."]
    assert table["Freq."].sum() == len(df)
    assert np.isclose(table["Cum."].iloc[-1], 100.0)
    for k, level in enumerate(sorted(df.g.unique()), start=1):
        assert (s.data[f"gd{k}"] == (df.g == level)).all()
    # the indicators are in dataset order, so a varlist range reads them
    fit = sp.regress("y ~ gd2 + gd3 + gd4", data=s.data)
    s.run("reg y gd2-gd4")
    pd.testing.assert_series_equal(s.output.params, fit.params)


def test_collapse_is_a_groupby(df):
    s = session(df)
    s.run("preserve")
    s.run("collapse (mean) ybar=y (sd) x if t > 50, by(g)")
    rows = df[df.t > 50].groupby("g")
    assert np.allclose(s.data["ybar"], rows["y"].mean())
    assert np.allclose(s.data["x"], rows["x"].std())
    s.run("restore")
    assert len(s.data) == len(df)


def test_ipolate_fills_inside_the_observed_range_only():
    data = pd.DataFrame(
        {
            "x": np.arange(1.0, 9.0),
            "y": [np.nan, 2, np.nan, 4, 5, np.nan, np.nan, np.nan],
        }
    )
    s = session(data)
    s.run("ipolate y x, gen(yi)")
    assert np.allclose(s.data["yi"].iloc[1:5], [2, 3, 4, 5])
    assert np.isnan(s.data["yi"].iloc[0]) and np.isnan(s.data["yi"].iloc[5])
    s.run("ipolate y x, gen(ye) epolate")
    assert np.allclose(s.data["ye"], np.arange(1.0, 9.0))


def test_set_obs_and_expression_functions():
    s = session()
    s.run("clear")
    s.run("set obs 5")
    s.run("gen t = _n")
    s.run("gen c = sum(t)")  # running sum
    s.run("gen double a = asinh(t)")
    s.run("gen m = month(dofm(t))")  # month 1 is February 1960
    assert list(s.data["c"]) == [1, 3, 6, 10, 15]
    assert np.allclose(s.data["a"], np.arcsinh(np.arange(1, 6)))
    assert list(s.data["m"]) == [2, 3, 4, 5, 6]
    assert s.value("tq(1999q1)") == 156 and s.value("tm(1960m2)") == 1
    assert s.value("td(02jan1960)") == 1 and s.value("mdy(1, 2, 1960)") == 1
    assert np.isclose(s.value("normalden(1, 0, 2)"), 0.17603266338214976)
    assert np.isclose(s.value("invchi2tail(1, 0.05)"), 3.841458820694124)


def test_time_series_operators_in_generate():
    data = pd.DataFrame({"year": [2003.0, 2001, 2002, 2005], "v": [3.0, 1, 2, 5]})
    s = session(data)
    s.run("tsset year")  # sorts by time, as Stata does
    assert list(s.data["year"]) == [2001, 2002, 2003, 2005]
    s.run("gen d = d.v")
    s.run("gen l2 = L2.v")
    assert np.allclose(s.data["d"].iloc[1:3], [1, 1])
    assert np.isnan(s.data["d"].iloc[0]) and np.isnan(s.data["d"].iloc[3])  # gap
    # 2005 looks two years back to 2003, across the missing 2004
    assert s.data["l2"].iloc[2] == 1 and s.data["l2"].iloc[3] == 3


# ------------------------------------------------------------------ weights
def test_frequency_weights_are_expanded_rows(df):
    expanded = df.loc[df.index.repeat(df.n.astype(int))]
    a = quiet(sp.stata, "reg y x z [fweight=n], r", data=df)
    b = quiet(sp.regress, "y ~ x + z", data=expanded, robust="hc1")
    pd.testing.assert_series_equal(a.params, b.params)
    pd.testing.assert_series_equal(a.std_errors, b.std_errors)
    c = quiet(sp.stata, "logit b x z if g > 1 [fw=n]", data=df)
    d = quiet(sp.logit, "b ~ x + z", data=expanded[expanded.g > 1])
    assert np.allclose(c.params, d.params)
    table = quiet(sp.stata, "sum y if x > 0 [fw=n]", data=df)
    assert np.isclose(table.loc["y", "Mean"], expanded.loc[expanded.x > 0, "y"].mean())


def test_a_weight_expression_is_evaluated(df):
    data = df.assign(v=np.exp(df.x))
    a = quiet(sp.stata, "reg y x z [aw=1/v]", data=data)
    b = quiet(sp.regress, "y ~ x + z", data=data.assign(w=1 / data.v), weights="w")
    pd.testing.assert_series_equal(a.params, b.params)
    pd.testing.assert_series_equal(a.std_errors, b.std_errors)
    # one translated line cannot evaluate it, and says so
    out = sp.from_stata("reg y x [aw=1/v]")
    assert not out["ok"] and "expression" in out["error"]
    with pytest.raises(MethodIncompatibility, match="non-negative integers"):
        quiet(sp.stata, "reg y x [fw=x]", data=df)


# -------------------------------------------------------------- regression
def test_noconstant_and_stored_results(df):
    s = session(df)
    s.run("reg y x z, noconstant")
    fit = sp.regress("y ~ x + z - 1", data=df)
    pd.testing.assert_series_equal(s.output.params, fit.params)
    y = df.y.to_numpy()
    resid = np.asarray(fit.data_info["residuals"])
    assert np.isclose(fit.diagnostics["R-squared"], 1 - resid @ resid / (y @ y))
    assert np.isclose(s.value("e(rss)"), resid @ resid)
    assert np.isclose(s.value("e(rss) + e(mss)"), y @ y)
    assert s.value("e(df_m)") == 2 and s.value("e(df_r)") == len(df) - 2
    assert np.isclose(s.value("e(rmse)^2 * e(df_r)"), resid @ resid)


def test_predict_leverage_probability_and_if(df):
    s = session(df)
    s.run("reg y x z")
    s.run("predict h, leverage")
    assert np.isclose(s.data["h"].sum(), 3.0, atol=1e-5)  # trace of the hat matrix
    s.run("predict e if g == 1, residuals")
    assert s.data.loc[df.g != 1, "e"].isna().all()
    s.run("logit b x z")
    s.run("predict p")
    fit = sp.logit("b ~ x + z", data=df)
    assert np.allclose(s.data["p"], fit.data_info["fitted_values"], atol=1e-6)
    with pytest.raises(MethodIncompatibility, match="does not apply"):
        s.run("predict r2, residuals")


# ------------------------------------------------------- stored estimates
def test_estimates_store_table_and_hausman(df):
    rng = np.random.default_rng(2)
    n = 1500
    z, u = rng.normal(size=n), rng.normal(size=n)
    d = z + 0.8 * u + rng.normal(size=n)
    data = pd.DataFrame({"y": 1 + 0.5 * d + u, "d": d, "z": z})
    s = session(data)
    s.run("ivregress 2sls y (d = z)")
    s.run("estimates store iv")
    s.run("qui reg y d")
    s.run("est sto ols")
    s.run("hausman iv ols, sigmamore")
    assert s.output["df"] == 1 and s.output["pvalue"] < 0.01
    s.run("estimates table iv ols, b se")
    assert "iv" in str(s.output) and "ols" in str(s.output)
    s.run("esttab iv ols, se star(* 0.1 ** 0.05 *** 0.01)")
    assert s.output is not None
    s.run("estimates restore iv")
    assert np.isclose(s.value("_b[d]"), s.estimates["iv"][0].params["d"])
    with pytest.raises(MethodIncompatibility, match="not stored"):
        s.run("hausman iv nosuch")


def test_bysort_summarize_runs_group_by_group(df):
    out = quiet(sp.stata, "bysort g: sum y x", data=df)
    for level, rows in df.groupby("g"):
        assert np.isclose(out.loc[(level, "y"), "Mean"], rows["y"].mean())


# ---------------------------------------------------------- post-estimation
def test_estat_writes_statas_defaults_out(df):
    s = session(df)
    s.run("reg y x z")
    fit = s.last
    s.run("estat hettest")
    assert s.output == sp.estat(
        fit, "hettest", variables="fitted", version="normal", print_results=False
    )
    s.run("estat ovtest")
    assert s.output["df1"] == 3  # powers two to four
    s.run("estat bgodfrey, lags(2) nomiss0")
    assert s.output["fill"] == "drop" and s.output["lags"] == 2
    s.run("estat ic")
    assert np.isclose(s.output["AIC"], -2 * s.output["ll"] + 2 * 3)
    s.run("estat vif")
    assert set(s.output["vif_table"]["variable"]) == {"x", "z"}
    s.run("estat imtest")
    parts = s.output["table"]["chi2"]
    assert np.isclose(parts["total"], parts.iloc[:3].sum())
    assert (
        s.output["statistic"]
        == sp.estat(fit, "white", print_results=False)["statistic"]
    )
    assert not sp.from_stata("estat archlm")["ok"]


def test_var_family_and_forecast():
    rng = np.random.default_rng(3)
    n = 250
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = [0.5 * y[t - 1, 0] + 0.2 * y[t - 1, 1], 0.4 * y[t - 1, 1]] + rng.normal(
            size=2
        )
    data = pd.DataFrame(y, columns=["a", "b"]).assign(t=np.arange(n))
    s = session(data)
    s.run("tsset t")
    s.run("varsoc a b, maxlag(3)")
    assert s.output.attrs["selected"]["SBIC"] == 1
    s.run("var a b, lags(1/1)")
    fit = s.last
    s.run("varstable")
    assert s.output["stable"] and len(s.output["table"]) == 2
    s.run("vargranger")
    table = s.output["table"].set_index(["equation", "excluded"])
    assert table.loc[("a", "b"), "p"] < 0.05 < table.loc[("b", "a"), "p"]
    s.run("fcast compute f_, step(3)")
    assert np.allclose(s.output.loc[1, ["a", "b"]], fit.forecast(1).loc[1, ["a", "b"]])
    assert s.output["a_se"].is_monotonic_increasing
    # a lag list that skips a lag has no counterpart
    assert not sp.from_stata("var a b, lags(2)")["ok"]


# ------------------------------------------------------ programs, simulate
def test_program_and_simulate_reproduce_the_design():
    script = """
        program onesample, rclass
            drop _all
            set obs 30
            gen x = runiform()
            sum x
            return scalar m = r(mean)
        end
        simulate xbar = r(m), seed(101) reps(400) nodots: onesample
    """
    s = session()
    for line in script.strip().splitlines():
        s.run(line.strip())
    assert s.data.shape == (400, 1) and s.simulated
    # a mean of 30 uniforms: centre 0.5, standard deviation sqrt(1/12/30)
    assert abs(s.data["xbar"].mean() - 0.5) < 0.01
    assert abs(s.data["xbar"].std() - np.sqrt(1 / 360)) < 0.006
    # the same seed gives the same sample
    again = session()
    for line in script.strip().splitlines():
        again.run(line.strip())
    pd.testing.assert_frame_equal(s.data, again.data)


def test_random_draws_warn_and_unsupported_programs_are_refused():
    s = StataSession()
    s.run("clear")
    s.run("set obs 10")
    with pytest.warns(UserWarning, match="numpy, not from Stata"):
        s.run("gen e = rnormal()")
    s.run("program bad")
    with pytest.raises(MethodIncompatibility, match="arguments or macros"):
        s.run("syntax varlist")
    with pytest.raises(MethodIncompatibility, match="program defined above"):
        s.run("simulate b = r(b), reps(5): nosuch")


# ----------------------------------------------------------------- teffects
def test_teffects_generate_predict_and_balance():
    rng = np.random.default_rng(4)
    n = 600
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    treat = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2 - 1)))).astype(float)
    data = pd.DataFrame(
        {"y": 1 + treat + x1 + rng.normal(size=n), "d": treat, "x1": x1, "x2": x2}
    )
    s = session(data)
    s.run("teffects psmatch (y) (d x1 x2), atet nn(2) gen(m)")
    assert {"m1", "m2"} <= set(s.data.columns)
    treated = s.data["d"] == 1
    matched = s.data.loc[treated, "m1"].astype(int) - 1
    assert (s.data["d"].to_numpy()[matched] == 0).all()  # matches are controls
    s.run("predict ps, ps")
    assert s.data["ps"].between(0, 1).all()
    s.run("tebalance summarize")
    table = s.output
    assert (table["std_diff_matched"].abs() < table["std_diff_raw"].abs()).all()
    s.run("gen ps_match = ps[m1]")
    gap = (s.data["ps"] - s.data["ps_match"])[treated].abs().mean()
    assert gap < 0.02
    # too few matches within the caliper: the variable is created and the
    # command stops, as in Stata
    with pytest.raises(MethodIncompatibility, match="fewer than 2"):
        s.run("teffects psmatch (y) (d x1 x2), atet nn(2) caliper(0.0005) osample(out)")
    assert s.data["out"].isin([0, 1]).all() and s.data["out"].sum() > 0
