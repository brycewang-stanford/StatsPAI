"""Fixes from the Wooldridge 7e / Mixtape replication report (2026-11-28).

The report replayed every data example of Wooldridge's *Introductory
Econometrics* (7th ed.) and the Stata code of Cunningham's *Causal
Inference: The Mixtape* against Stata 18 MP. Each test below pins one
finding. The first part runs on simulated data and asserts identities (a
fit with a collinear column omitted equals the fit without it, a preset
equals the options it stands for). The second part checks the book
examples against the numbers Stata 18 MP printed on the original files; it
runs only when the data are available:

    STATSPAI_WOOLDRIDGE_DIR=/path/to/wooldridge/dta \\
    STATSPAI_MIXTAPE_DIR=/path/to/mixtape/dta \\
        pytest tests/test_textbook_parity_nov28.py
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning, MethodIncompatibility

# --------------------------------------------------------------------------
# Simulated data
# --------------------------------------------------------------------------


def _iv_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 4, n)  # four groups: dummies d0..d3 sum to one
    j = rng.integers(0, 5, n)  # five judges: z0..z4 sum to one
    df = pd.DataFrame({f"d{k}": (g == k).astype(float) for k in range(4)})
    for k in range(5):
        df[f"z{k}"] = (j == k).astype(float)
    df["w"] = rng.normal(size=n)
    u = rng.normal(size=n)
    df["x"] = 0.4 * j + 0.3 * df["w"] + u + rng.normal(size=n)
    df["y"] = 1.0 + 0.5 * df["x"] + 0.2 * g + 0.8 * u + rng.normal(size=n)
    return df


@pytest.mark.parametrize("fn", ["jive", "liml"])
def test_iv_collinear_columns_are_omitted_like_stata(fn):
    # Stata drops the last dummy of each set summing to the constant
    # ("note: t6 omitted because of collinearity"); the fit equals the one
    # with those columns removed by hand.
    df = _iv_data()
    full_x, full_z = ["w", "d0", "d1", "d2", "d3"], [f"z{k}" for k in range(5)]
    est = getattr(sp, fn)
    with pytest.warns(UserWarning, match="d3 omitted because of collinearity"):
        res = est(data=df, y="y", x_endog=["x"], x_exog=full_x, z=full_z)
    ref = est(data=df, y="y", x_endog=["x"], x_exog=full_x[:-1], z=full_z[:-1])
    np.testing.assert_allclose(res.params, ref.params, rtol=1e-10)
    np.testing.assert_allclose(res.std_errors, ref.std_errors, rtol=1e-10)
    assert [o["variable"] for o in res.model_info["omitted"]] == ["d3", "z4"]


def test_jive_refuses_endogenous_regressor_reproduced_by_controls():
    df = _iv_data()
    df["x"] = df["w"] * 2.0
    with pytest.raises(MethodIncompatibility, match="endogenous regressor"):
        sp.jive(df, y="y", x_endog=["x"], x_exog=["w"], z=["z0", "z1"])


def _dated(n=120, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "c": rng.poisson(2, n),
            "id": np.repeat(np.arange(12), 10),
            "t": np.tile(np.arange(10), 12),
        }
    )
    df["dt"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(
        rng.integers(0, 30, n), unit="D"
    )
    return df


@pytest.mark.parametrize(
    "call",
    [
        lambda d: sp.regress("y ~ x + dt", d),
        lambda d: sp.feols("y ~ x + dt | id", d),
        lambda d: sp.poisson("c ~ x + dt", d),
        lambda d: sp.qreg("y ~ x + dt", d),
        lambda d: sp.ivreg("y ~ (x ~ c) + dt", d),
        lambda d: sp.panel(d, "y ~ x + dt", entity="id", time="t", method="fe"),
    ],
)
def test_datetime_regressor_is_refused(call):
    # patsy used to make one dummy per date; the own-parser estimators read
    # nanoseconds or failed inside numpy.
    with pytest.raises(MethodIncompatibility, match="hold dates") as err:
        call(_dated())
    assert err.value.diagnostics["datetime_columns"] == ["dt"]


def test_datetime_as_label_is_allowed():
    d = _dated()
    n_dates = d["dt"].nunique()
    assert len(sp.regress("y ~ x + C(dt)", d).params) == 1 + 1 + (n_dates - 1)
    assert list(sp.feols("y ~ x | dt", d).params.index) == ["x"]


def _panel(seed=2, n_id=80, n_t=5):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_id), n_t)
    a = rng.normal(size=n_id)[ids]
    df = pd.DataFrame({"id": ids, "t": np.tile(np.arange(n_t), n_id)})
    df["z"] = rng.normal(size=n_id)[ids]  # time-invariant
    df["x"] = rng.normal(size=len(df)) + 0.5 * a
    df["y"] = 1.0 + 0.7 * df["x"] + 0.4 * df["z"] + a + rng.normal(size=len(df))
    return df


def test_hausman_method_takes_the_users_re_fit():
    df = _panel()
    fe = sp.panel(df, "y ~ x", entity="id", time="t", method="fe")
    re = sp.panel(df, "y ~ x + z", entity="id", time="t", method="re")
    ref = sp.hausman(fe, re)["statistic"]
    assert fe.hausman_test(efficient=re)["statistic"] == pytest.approx(ref, rel=1e-12)
    assert re.hausman_test(efficient=fe)["statistic"] == pytest.approx(ref, rel=1e-12)
    # without it the RE model is refitted on the FE regressors, and says so
    own = fe.hausman_test()
    assert "refitted on this result's regressors (x)" in own["re_specification"]
    re_x = sp.panel(df, "y ~ x", entity="id", time="t", method="re")
    assert own["statistic"] == pytest.approx(
        sp.hausman(fe, re_x)["statistic"], rel=1e-8
    )


def _heckman_data(n=800, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    e = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n)
    df["s"] = (0.3 + df["x"] + df["z"] + e[:, 0] > 0).astype(int)
    df["y"] = np.where(df["s"] == 1, 1 + 0.5 * df["x"] + e[:, 1], np.nan)
    return df


@pytest.mark.parametrize("method", ["twostep", "mle"])
def test_heckman_n_obs_is_the_full_sample(method):
    df = _heckman_data()
    res = sp.heckman(df, y="y", x=["x"], select="s", z=["x", "z"], method=method)
    assert res.n_obs == len(df)
    assert res.model_info["n_selected"] == int(df["s"].sum())
    assert res.model_info["n_censored"] == int((df["s"] == 0).sum())


def test_white_fstat_is_the_auxiliary_regression_f():
    rng = np.random.default_rng(4)
    n = 150
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = 1 + df["x1"] + rng.normal(size=n) * (1 + np.abs(df["x1"]))
    r = sp.regress("y ~ x1 + x2", df)
    out = sp.estat(r, "white", variables="fitted", version="fstat", print_results=False)
    e2, f = r.residuals() ** 2, r.fitted_values()
    aux = sp.regress("e2 ~ f + I(f**2)", pd.DataFrame({"e2": e2, "f": f}))
    assert out["statistic"] == pytest.approx(aux.diagnostics["F-statistic"], rel=1e-10)
    assert (out["df1"], out["df2"]) == (2, n - 3)
    chi2 = sp.estat(r, "white", variables="fitted", print_results=False)
    assert chi2["statistic"] == pytest.approx(n * aux.diagnostics["R-squared"])


def test_estat_refuses_version_a_test_does_not_have():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x": rng.normal(size=60)})
    df["y"] = df["x"] + rng.normal(size=60)
    r = sp.regress("y ~ x", df)
    with pytest.raises(MethodIncompatibility, match="not an option of this test"):
        sp.estat(r, "reset", version="fstat", print_results=False)
    with pytest.raises(MethodIncompatibility, match="not 'iid' or 'fstat'"):
        sp.estat(r, "white", version="normal", print_results=False)


def test_column_named_inf_works_inside_an_interaction():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"a": rng.normal(size=40), "inf": rng.normal(size=40)})
    df["y"] = df["a"] - df["a"] * df["inf"] + rng.normal(size=40)
    res = sp.regress("y ~ a + a:inf", df)
    ref = sp.regress("y ~ a + a:infl", df.rename(columns={"inf": "infl"}))
    assert list(res.params.index) == ["Intercept", "a", "a:inf"]
    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy())


def _design_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [
        str(w.message)
        for w in caught
        if getattr(w.message, "diagnostics", {}).get("test")
        == "treatment_cluster_level"
        or "is constant within" in str(w.message)
    ]


def test_year_dummies_are_not_flagged_as_group_level_treatments():
    rng = np.random.default_rng(7)
    n_t, per = 7, 60
    df = pd.DataFrame({"year": np.repeat(np.arange(n_t) + 81, per)})
    for t in range(1, n_t):
        df[f"d{81 + t}"] = (df["year"] == 81 + t).astype(int)
    df["x"] = rng.normal(size=len(df))
    df["y"] = df["x"] + 0.1 * df["year"] + rng.normal(size=len(df))
    rhs = " + ".join(f"d{81 + t}" for t in range(1, n_t))
    assert _design_warnings(lambda: sp.regress(f"y ~ x + {rhs}", df)) == []


def test_state_level_treatment_is_still_flagged():
    rng = np.random.default_rng(8)
    n_s, per = 30, 40
    df = pd.DataFrame({"state": np.repeat(np.arange(n_s), per)})
    df["treat"] = (df["state"] < 15).astype(int)
    df["y"] = 0.3 * df["treat"] + rng.normal(size=n_s)[df["state"]]
    df["y"] += rng.normal(size=len(df))
    hits = _design_warnings(lambda: sp.regress("y ~ treat", df))
    assert hits and "'state'" in hits[0]


def test_constant_outcome_without_intercept_does_not_warn():
    # the robust LM statistic regresses 1 on u1, u2 without a constant
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"u1": rng.normal(size=50), "u2": rng.normal(size=50)})
    df["one"] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        res = sp.regress("one ~ u1 + u2 - 1", df)
    assert list(res.params.index) == ["u1", "u2"]


def test_lag_edge_rows_are_noted_once_per_frame_interior_gaps_always():
    rng = np.random.default_rng(10)
    df = pd.DataFrame({"y": rng.normal(size=60), "x": rng.normal(size=60)})
    df["x1"] = df["x"].shift(1)
    df["x2"] = df["x"].shift(2)

    def notes(frame, formula):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sp.regress(formula, frame)
        return [
            w
            for w in caught
            if getattr(w.message, "diagnostics", {}).get("test") == "listwise_deletion"
        ]

    first = notes(df, "y ~ x1")
    assert len(first) == 1 and "start or end of the data" in str(first[0].message)
    assert notes(df, "y ~ x1 + x2") == []
    gap = df.copy()
    gap.loc[30, "x"] = np.nan
    assert len(notes(gap, "y ~ x")) == 1
    assert len(notes(gap, "y ~ x")) == 1


def test_zero_analytic_weights_drop_the_row_as_stata_does():
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"x": rng.normal(size=80), "w": rng.uniform(1, 3, 80)})
    df["y"] = df["x"] + rng.normal(size=80)
    df.loc[[3, 40], "w"] = 0.0
    with pytest.warns(AssumptionWarning, match="2 row\\(s\\) with zero weight"):
        res = sp.regress("y ~ x", df, weights="w")
    ref = sp.regress("y ~ x", df[df["w"] > 0], weights="w")
    np.testing.assert_allclose(res.params, ref.params, rtol=1e-12)
    np.testing.assert_allclose(res.std_errors, ref.std_errors, rtol=1e-12)
    assert res.nobs == 78 and res.data_info["n_zero_weight_dropped"] == 2
    df.loc[5, "w"] = -1.0
    with pytest.raises(MethodIncompatibility, match="strictly positive"):
        sp.regress("y ~ x", df, weights="w")


def test_test_can_skip_restrictions_on_omitted_terms():
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    df["c"] = df["a"] + df["b"]  # omitted as collinear
    df["y"] = df["a"] + rng.normal(size=100)
    r = sp.regress("y ~ a + b + c", df)
    with pytest.raises(MethodIncompatibility, match="omitted='skip'"):
        sp.test(r, "a b c")
    out = sp.test(r, "a b c", omitted="skip")
    ref = sp.test(r, "a b")
    assert out["statistic"] == pytest.approx(ref["statistic"], rel=1e-12)
    assert out["omitted_dropped"] == ["c"]
    assert sp.test(r, ["a = 0", "c = 0"], omitted="skip")["omitted_dropped"] == [
        "c = 0"
    ]
    assert "omitted_dropped" not in ref
    with pytest.raises(MethodIncompatibility, match="nothing is left to test"):
        sp.test(r, "c", omitted="skip")


def test_stata_test_translates_with_omitted_skip():
    out = sp.from_stata("test x1 x2")
    calls = out if isinstance(out, list) else [out]
    assert any("omitted='skip'" in str(getattr(c, "python", c)) for c in calls)


def _ts(n=80, seed=13):
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.6 * e[i - 1] + rng.normal()
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = 1 + 0.5 * df["x"] + e
    return df


def test_newey_preset_is_hac_with_small_sample_factor():
    df = _ts()
    a = sp.regress("y ~ x", df, robust="newey", hac_lags=2)
    b = sp.regress("y ~ x", df, vce="newey", hac_lags=2)
    ref = sp.regress("y ~ x", df, robust="hac", hac_lags=2, hac_small=True)
    np.testing.assert_allclose(a.std_errors, ref.std_errors, rtol=1e-14)
    np.testing.assert_allclose(b.std_errors, ref.std_errors, rtol=1e-14)
    bare = sp.regress("y ~ x", df, robust="hac", hac_lags=2)
    k = 2
    np.testing.assert_allclose(
        a.std_errors**2, bare.std_errors**2 * len(df) / (len(df) - k), rtol=1e-12
    )


def test_panel_accepts_robust_true():
    df = _panel()
    a = sp.panel(df, "y ~ x", entity="id", time="t", method="fe", robust=True)
    b = sp.panel(df, "y ~ x", entity="id", time="t", method="fe", robust="robust")
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-14)


def test_first_difference_constant_is_ols_on_the_differences():
    df = _panel().sort_values(["id", "t"])
    res = sp.panel(df, "y ~ x", entity="id", time="t", method="fd", constant=True)
    d = df.groupby("id")[["y", "x"]].diff().dropna()
    ref = sp.regress("y ~ x", d)
    assert list(res.params.index) == ["const", "x"]
    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy())
    np.testing.assert_allclose(res.std_errors.to_numpy(), ref.std_errors.to_numpy())
    with pytest.raises(MethodIncompatibility, match="method='fd' only"):
        sp.panel(df, "y ~ x", entity="id", time="t", method="fe", constant=True)


def test_random_effects_components_are_public():
    df = _panel()
    re = sp.panel(df, "y ~ x + z", entity="id", time="t", method="re")
    info = re.model_info
    su2, se2 = info["sigma_u"] ** 2, info["sigma_e"] ** 2
    assert info["rho"] == pytest.approx(su2 / (su2 + se2), rel=1e-12)
    # balanced panel: one theta = 1 - sqrt(se2 / (se2 + T su2))
    assert info["theta"] == pytest.approx(1 - np.sqrt(se2 / (se2 + 5 * su2)))


def test_system_results_have_a_labelled_coefficient_vector():
    rng = np.random.default_rng(14)
    n = 200
    df = pd.DataFrame({v: rng.normal(size=n) for v in ["p", "inc", "cost"]})
    df["q"] = 1 - 0.5 * df["p"] + 0.3 * df["inc"] + rng.normal(size=n)
    df["qs"] = 0.5 + 0.4 * df["p"] + 0.2 * df["cost"] + rng.normal(size=n)
    res = sp.sureg(
        equations={"demand": ("q", ["p", "inc"]), "supply": ("qs", ["p", "cost"])},
        data=df,
    )
    assert res.params["demand:p"] == res.equations["demand"]["params"]["p"]
    assert res.std_errors["supply:cost"] == res.equations["supply"]["se"]["cost"]
    tidy = res.tidy()
    assert list(tidy["term"])[:2] == list(res.params.index)[:2]
    assert "demand:inc" in str(sp.regtable(res))


def test_df_resid_attribute():
    df = _ts()
    r = sp.regress("y ~ x", df)
    assert r.df_resid == r.data_info["df_resid"] == len(df) - 2


def _synth_panel(seed=15, n_units=12, n_t=14):
    rng = np.random.default_rng(seed)
    rows = []
    f = rng.normal(size=n_t).cumsum()
    for u in range(n_units):
        load = rng.uniform(0.5, 1.5)
        for t in range(n_t):
            y = load * f[t] + rng.normal(scale=0.3) + (2.0 if u == 0 and t >= 10 else 0)
            rows.append((u, t, y, rng.normal() + load))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "cov"])


def test_synth_mspe_period_is_the_window_the_weights_fit():
    df = _synth_panel()
    preds = [("y", 3, "mean"), ("y", 6, "mean"), ("cov", slice(0, 8), "mean")]
    common = dict(
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=0,
        special_predictors=preds,
        v_method="regression",
        placebo=False,
    )
    # periods 0..9 in the MSPE with treatment at 9 = treatment at 10
    a = sp.synth(df, treatment_time=10, **common)
    b = sp.synth(df, treatment_time=9, mspe_period=list(range(10)), **common)
    wa = a.model_info["weights"].set_index("unit")["weight"]
    wb = b.model_info["weights"].set_index("unit")["weight"]
    np.testing.assert_allclose(wa.sort_index(), wb.sort_index(), atol=1e-12)
    # the default window is the pre-treatment periods
    c = sp.synth(df, treatment_time=10, mspe_period=list(range(10)), **common)
    wc = c.model_info["weights"].set_index("unit")["weight"]
    np.testing.assert_array_equal(wa.sort_index(), wc.sort_index())
    with pytest.raises(MethodIncompatibility, match="mspe_period"):
        sp.synth(df, treatment_time=10, mspe_period=[99], **common)


# --------------------------------------------------------------------------
# The book examples against Stata 18 MP
# --------------------------------------------------------------------------

WOO = os.environ.get("STATSPAI_WOOLDRIDGE_DIR")
MIX = os.environ.get("STATSPAI_MIXTAPE_DIR")
needs_woo = pytest.mark.skipif(
    not WOO or not Path(WOO).is_dir(),
    reason="set STATSPAI_WOOLDRIDGE_DIR to the Wooldridge .dta files",
)
needs_mix = pytest.mark.skipif(
    not MIX or not Path(MIX).is_dir(),
    reason="set STATSPAI_MIXTAPE_DIR to the Mixtape .dta files",
)


def woo(name):
    return pd.read_stata(Path(WOO) / f"{name}.dta")


def mix(name):
    return pd.read_stata(
        Path(MIX) / f"{name}.dta", convert_categoricals=False, convert_dates=False
    )


WAGEPAN_RE = (
    "lwage ~ educ + black + hisp + exper + expersq + married + union"
    " + d81 + d82 + d83 + d84 + d85 + d86 + d87"
)


@needs_woo
def test_example_14_4_hausman_and_re_components():
    wp = woo("wagepan")
    fe = sp.panel(
        wp,
        "lwage ~ expersq + married + union + d81 + d82 + d83 + d84 + d85 + d86 + d87",
        entity="nr",
        time="year",
        method="fe",
    )
    re = sp.panel(wp, WAGEPAN_RE, entity="nr", time="year", method="re")
    # hausman fe re -> chi2(10) = 26.360909420658
    assert fe.hausman_test(efficient=re)["statistic"] == pytest.approx(
        26.360909420658, rel=1e-6
    )
    # xtreg ..., re: e(sigma_u) e(sigma_e) e(rho) e(theta)
    info = re.model_info
    assert info["sigma_u"] == pytest.approx(0.324603143915704, rel=1e-6)
    assert info["sigma_e"] == pytest.approx(0.350990014333998, rel=1e-6)
    assert info["rho"] == pytest.approx(0.461002152140399, rel=1e-6)
    assert info["theta"] == pytest.approx(0.642910880264968, rel=1e-6)


@needs_woo
def test_example_17_5_heckman_two_step_n():
    m = woo("mroz")
    res = sp.heckman(
        m,
        y="lwage",
        x=["educ", "exper", "expersq"],
        select="inlf",
        z=["educ", "exper", "expersq", "nwifeinc", "age", "kidslt6", "kidsge6"],
    )
    # heckman ..., twostep: e(N) 753, e(N_selected) 428
    assert (res.n_obs, res.model_info["n_selected"]) == (753, 428)


@needs_woo
def test_example_12_7_newey():
    pr = woo("prminwge")
    res = sp.regress(
        "lprepop ~ lmincov + lprgnp + lusgnp + t", pr, robust="newey", hac_lags=2
    )
    # newey ..., lag(2): _se[lmincov]
    assert res.std_errors["lmincov"] == pytest.approx(0.045718806887546, rel=1e-6)


@needs_woo
def test_example_8_5_special_white_f():
    h = woo("hprice1")
    r = sp.regress("price ~ lotsize + sqrft + bdrms", h)
    out = sp.estat(r, "white", variables="fitted", version="fstat", print_results=False)
    # the book: F(2, 85) = 9.64 (Wooldridge 7e, example 8.5)
    assert round(out["statistic"], 2) == 9.64
    assert (out["df1"], out["df2"]) == (2, 85)


@needs_woo
def test_equation_13_20_first_difference_with_intercept():
    c = woo("crime2").reset_index(drop=True)
    c["cid"] = np.arange(len(c)) // 2  # two rows (1982, 1987) per city
    res = sp.panel(
        c, "crmrte ~ unem", entity="cid", time="year", method="fd", constant=True
    )
    # xtset cid year, delta(5); regress D.crmrte D.unem (the file's own
    # ccrmrte is stored in single precision, so it is not the reference)
    np.testing.assert_allclose(
        res.params[["const", "unem"]], [15.402193871769132, 2.217996493908312], 1e-6
    )
    np.testing.assert_allclose(
        res.std_errors[["const", "unem"]],
        [4.702115941854486, 0.877865658548391],
        1e-6,
    )


@needs_woo
def test_example_10_6_inf_interaction():
    res = sp.regress(
        "demvote ~ partyWH + incum + partyWH:gnews + partyWH:inf", woo("fair")
    )
    # regress demvote partyWH incum c.partyWH#c.gnews c.partyWH#c.inf
    assert res.params["partyWH:inf"] == pytest.approx(-0.008338255843226, rel=1e-6)
    assert res.nobs == 21


JUDGE_PRE = [f"judge_pre_{i}" for i in range(1, 9)]
CONTROL2 = ["day", "day2", "day3", "bailDate", "t1", "t2", "t3", "t4", "t5", "t6"]


@needs_mix
def test_mixtape_7_2_jive_and_first_stage_test():
    judge = mix("judge_fe")
    res = sp.jive(
        judge, y="guilt", x_endog=["jail3"], x_exog=CONTROL2, z=JUDGE_PRE, robust="hc1"
    )
    # jive guilt (jail3 = $judge_pre) $control2, robust
    assert res.params["jail3"] == pytest.approx(0.162435192679368, rel=1e-6)
    assert res.std_errors["jail3"] == pytest.approx(0.070140945568131, rel=1e-6)
    first = sp.regress("jail3 ~ " + " + ".join(JUDGE_PRE + CONTROL2), judge)
    # test $judge_pre -> F(7, 331954) = 35.358994199447 (constraint dropped)
    out = sp.test(first, JUDGE_PRE, omitted="skip")
    assert out["statistic"] == pytest.approx(35.358994199447, rel=1e-6)
    assert out["df"] == (7, 331954)


@needs_mix
def test_mixtape_9_4_zero_aweight_row():
    ab = mix("abortion")
    bf = ab[ab["bf15"] == 1]
    controls = "acc + ir + pi + alcohol + crack + poverty + income + ur"
    res = sp.regress(
        f"lnr ~ C(repeal)*C(year) + C(fip) + {controls}",
        bf,
        weights="totpop",
        cluster="fip",
    )
    # xi: reg lnr i.repeal*i.year i.fip ... if bf15==1 [aweight=totpop]: N 736
    assert res.nobs == 736
    # _b[acc], _se[acc], _b[ur] (xi codes the repeal x year cells differently)
    assert res.params["acc"] == pytest.approx(0.002877739805802, rel=1e-6)
    assert res.std_errors["acc"] == pytest.approx(0.001279839000409, rel=1e-6)
    assert res.params["ur"] == pytest.approx(-0.027823707469328, rel=1e-6)
