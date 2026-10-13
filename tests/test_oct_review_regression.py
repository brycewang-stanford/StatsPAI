"""Regression tests for the October 2026 review of the regression family.

Each test states an identity (the fit equals the one without the redundant
column, the same data under a RangeIndex, an independent computation) or a
specific exception or warning class.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)

OMITTED = "omitted because of collinearity"


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 400
    d = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "g": rng.integers(0, 40, n),
        }
    )
    d["y"] = 1 + 2 * d.x1 - d.x2 + rng.normal(size=n)
    d["x3"] = d.x1 + d.x2
    d["c"] = (rng.random(n) < 1 / (1 + np.exp(-d.x1))).astype(int)
    d["cnt"] = rng.poisson(np.exp(0.3 + 0.4 * d.x1))
    d["z1"] = rng.normal(size=n)
    d["e"] = 0.6 * d.z1 + 0.5 * rng.normal(size=n)
    d["tr"] = (d.z1 + rng.normal(size=n) > 0).astype(int)
    return d


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(1)
    G, T = 150, 6
    p = pd.DataFrame({"id": np.repeat(np.arange(G), T)})
    a = rng.normal(size=G)[p.id]
    p["t"] = np.tile(np.arange(T), G)
    p["x"] = rng.normal(size=len(p)) + 0.5 * a
    p["w"] = rng.normal(size=len(p))
    p["ti"] = rng.normal(size=G)[p.id]
    p["ti2"] = (p.id % 3).astype(float) + 0.3 * a
    p["y"] = 1 + p.x + 0.5 * p.w + 0.4 * p.ti + a + rng.normal(size=len(p))
    return p


def _quiet(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f()


def _same(a, b):
    return np.array_equal(a.params.to_numpy(), b.params.to_numpy()) and (
        np.array_equal(a.std_errors.to_numpy(), b.std_errors.to_numpy())
    )


def _duplicate_index(d):
    out = d.copy()
    out.index = np.arange(len(d)) % 7
    return out


# ---------------------------------------------------------------- 1 standardize


@pytest.mark.parametrize(
    "term", ["log(xp)", "np.log(xp)", "I(xp^2)", "C(xp)", "factor(xp)"]
)
def test_standardize_leaves_variables_inside_calls_alone(df, term):
    d = df.assign(xp=np.exp(df.x2).round(1) + 1.0)
    out = sp.standardize(d, formula=f"y ~ x1 + {term}")
    assert list(out.attrs["standardize"]) == ["x1"]
    pd.testing.assert_series_equal(out["xp"], d["xp"])


def test_standardize_skips_a_variable_that_is_also_transformed(df):
    out = sp.standardize(df, formula="y ~ x1 + x2 + I(x2^2)")
    assert list(out.attrs["standardize"]) == ["x1"]


def test_standardize_reads_backticked_names(df):
    d = df.rename(columns={"x2": "my col"})
    out = sp.standardize(d, formula="y ~ x1 + `my col`")
    assert list(out.attrs["standardize"]) == ["x1", "my col"]
    assert abs(out["my col"].mean()) < 1e-12
    assert out["my col"].std() == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------- 2 itcv


@pytest.fixture(scope="module")
def weighted():
    rng = np.random.default_rng(3)
    n = 300
    d = pd.DataFrame({"z": rng.normal(size=n), "w": rng.normal(size=n)})
    d["x"] = 0.5 * d.z + rng.normal(size=n)
    d["y"] = 0.4 * d.x + 0.3 * d.z + rng.normal(size=n)
    d["wt"] = np.where(d.z > 0, 5, 1)
    return d


def test_itcv_frequency_weights_equal_duplicated_rows(weighted):
    d = weighted
    a = sp.itcv(sp.regress("y ~ x + z + w", d, weights="wt"), "x")
    long = d.loc[d.index.repeat(d.wt)].reset_index(drop=True)
    b = sp.itcv(sp.regress("y ~ x + z + w", long), "x")
    cols = ["r_yz", "r_xz", "impact"]
    np.testing.assert_allclose(
        a["impacts"].loc[["z", "w"], cols],
        b["impacts"].loc[["z", "w"], cols],
        rtol=1e-10,
    )
    assert a["benchmark"] == pytest.approx(b["benchmark"], rel=1e-10)


def test_itcv_weighted_impacts_share_the_moments_of_r_obs(weighted):
    # the weighted partial correlation of y and x given the controls is
    # the one the weighted t statistic implies
    d = weighted
    out = sp.itcv(sp.regress("y ~ x + z + w", d, weights="wt"), "x")
    P = np.linalg.inv(
        np.cov(d[["y", "x", "z", "w"]].to_numpy(), rowvar=False, aweights=d.wt)
    )
    assert out["r_obs"] == pytest.approx(
        -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]), rel=1e-10
    )


def test_itcv_refuses_iv_and_the_constant(weighted):
    with pytest.raises(MethodIncompatibility, match="least squares"):
        sp.itcv(_quiet(lambda: sp.ivreg("y ~ w + (x ~ z)", weighted)), "x")
    with pytest.raises(MethodIncompatibility, match="constant"):
        sp.itcv(sp.regress("y ~ x + z", weighted), "Intercept")


# ---------------------------------------------------------------- 3 support


@pytest.mark.parametrize("family, shift", [("binomial", 1), ("poisson", -1)])
def test_gee_and_gam_refuse_an_outcome_outside_the_family_support(df, family, shift):
    d = df.assign(c2=df.c + shift)
    with pytest.raises(MethodIncompatibility, match="outside the support"):
        sp.gee("c2 ~ x1", d, id="g", family=family)
    with pytest.raises(MethodIncompatibility, match="outside the support"):
        sp.gam("c2 ~ s(x1)", d, family=family)
    with pytest.raises(MethodIncompatibility, match="outside the support"):
        sp.glm("c2 ~ x1", d, family=family)


# ---------------------------------------------------------------- 4 gee


@pytest.mark.parametrize(
    "outcome, family", [("y", "gaussian"), ("c", "binomial"), ("cnt", "poisson")]
)
def test_gee_omits_collinear_and_constant_regressors(df, outcome, family):
    d = df.assign(k=3.0)
    kw = dict(id="g", family=family, corstr="exchangeable")
    with pytest.warns(UserWarning, match=OMITTED):
        full = sp.gee(f"{outcome} ~ x1 + x2 + x3 + k", d, **kw)
    short = sp.gee(f"{outcome} ~ x1 + x2", d, **kw)
    assert _same(full, short)
    assert [o["variable"] for o in full.model_info["omitted"]] == ["x3", "k"]
    assert short.model_info["omitted"] == []


def test_gee_docstring_lists_the_families_it_accepts():
    from statspai.regression.glm import FAMILIES

    doc = sp.gee.__doc__
    assert all(f'"{name}"' in doc for name in FAMILIES)


# ---------------------------------------------------------------- 5 gam


def _same_gam(a, b):
    return (
        _same(a, b)
        and np.array_equal(a.fitted_values, b.fitted_values)
        and np.array_equal(a.lambdas, b.lambdas)
        and a.edf == b.edf
    )


@pytest.mark.parametrize(
    "full, short, gone",
    [
        ("y ~ s(x1) + x2 + z", "y ~ s(x1) + x2", "z"),
        ("y ~ s(x1) + kk", "y ~ s(x1)", "kk"),
        # the centred straight line is in the null space of the penalty of
        # s(x1): the parametric x1 is the one omitted
        ("y ~ x1 + s(x1) + x2", "y ~ s(x1) + x2", "x1"),
    ],
)
def test_gam_omits_unidentified_parametric_terms(df, full, short, gone):
    d = df.assign(z=2 * df.x2, kk=3.0)
    with pytest.warns(UserWarning, match=OMITTED):
        a = sp.gam(full, d)
    b = sp.gam(short, d)
    assert _same_gam(a, b)
    assert [o["variable"] for o in a.model_info["omitted"]] == [gone]
    assert b.model_info["omitted"] == []
    np.testing.assert_array_equal(a.predict(d.head(9)), b.predict(d.head(9)))


# ---------------------------------------------------------------- 6 ridge


def test_ridge_at_zero_on_a_deficient_design_is_the_minimum_norm_solution(df):
    with pytest.warns(UserWarning, match="minimum-norm"):
        fit = sp.ridge("y ~ x1 + x2 + x3", df, lambda_=0.0)
    X = df[["x1", "x2", "x3"]].to_numpy()
    Xc = X - X.mean(axis=0)
    scale = np.sqrt((Xc**2).mean(axis=0))
    slopes = np.linalg.pinv(Xc / scale) @ (df.y - df.y.mean()).to_numpy() / scale
    np.testing.assert_allclose(fit.params[["x1", "x2", "x3"]], slopes, rtol=1e-9)
    ols = sp.regress("y ~ x1 + x2", df)
    np.testing.assert_allclose(
        fit.predict(), ols.data_info["fitted_values"], rtol=1e-10, atol=1e-10
    )


def test_ridge_default_path_starts_at_the_same_minimum_norm_row(df):
    at_zero = _quiet(lambda: sp.ridge("y ~ x1 + x2 + x3", df, lambda_=0.0))
    with pytest.warns(UserWarning, match="minimum-norm"):
        path = sp.ridge("y ~ x1 + x2 + x3", df).path
    assert path["lambda"].iloc[0] == 0.0
    np.testing.assert_allclose(
        path.loc[0, ["Intercept", "x1", "x2", "x3"]].astype(float),
        at_zero.params[["Intercept", "x1", "x2", "x3"]],
        rtol=1e-12,
    )


@pytest.mark.parametrize("value", [1 / 3, np.log(7.0), 0.1])
def test_a_constant_regressor_is_refused_whatever_its_float_noise(df, value):
    d = df.assign(k=value)
    with pytest.raises(MethodIncompatibility, match="do not vary"):
        sp.ridge("y ~ x1 + k", d)
    with pytest.raises(MethodIncompatibility, match="do not vary"):
        sp.shrinkage(d, "y", ["x1", "k"])


def test_ridge_rule_names_as_lambda(df):
    d = df.assign(x4=df.x1 * df.x2)
    for rule in ("hkb", "lw", "gcv"):
        named = sp.ridge("y ~ x1 + x2 + x4", d, lambda_=rule)
        chosen = sp.ridge("y ~ x1 + x2 + x4", d, select=rule)
        assert named.selection == rule
        assert named.lambda_ == chosen.lambda_
        pd.testing.assert_series_equal(named.params, chosen.params)
    with pytest.raises(MethodIncompatibility, match="rules"):
        sp.ridge("y ~ x1 + x2", d, lambda_="cv")


# ---------------------------------------------------------------- 7 index


@pytest.mark.parametrize(
    "fit",
    [
        lambda d: sp.gam("y ~ s(x1) + x2", d),
        lambda d: sp.gee("y ~ x1 + x2", d, id="g", corstr="exchangeable"),
        lambda d: sp.threshold("y ~ x1", d, threshold="x2"),
        lambda d: sp.series("y ~ x2", d, "x1", order=2),
    ],
    ids=["gam", "gee", "threshold", "series"],
)
def test_a_non_unique_index_gives_the_range_index_fit(df, fit):
    d = df.copy()
    d.loc[5, "x2"] = np.nan  # a dropped row: positions and labels part ways
    assert _same(_quiet(lambda: fit(_duplicate_index(d))), _quiet(lambda: fit(d)))


def test_xthtaylor_with_a_non_unique_index(panel):
    def fit(d):
        return sp.xthtaylor("y ~ x + w + ti + ti2", d, id="id", endog=["x", "ti2"])

    assert _same(fit(_duplicate_index(panel)), fit(panel))


def test_sample_index_keeps_the_labels_of_the_data(df):
    d = df.set_index(pd.date_range("2020-01-01", periods=len(df)))
    out = sp.series("y ~ x2", d, "x1", order=2)
    assert out.data_info["sample_index"].equals(d.index)


# ---------------------------------------------------------------- 8 xtoverid


def test_xtoverid_names_a_dependent_regressor(panel):
    with pytest.raises(MethodIncompatibility, match="x2 .collinear with 'x'"):
        sp.xtoverid(panel.assign(x2=2 * panel.x), "y", ["x", "x2", "w"], id="id")
    with pytest.raises(MethodIncompatibility, match="k .collinear with '_cons'"):
        sp.xtoverid(panel.assign(k=3.0), "y", ["x", "k"], id="id")


def test_xtoverid_needs_more_clusters_than_restrictions(panel):
    d = panel.assign(cl=panel.id % 2)
    with pytest.raises(DataInsufficient, match="rank at most 1"):
        sp.xtoverid(d, "y", ["x", "w"], id="id", cluster="cl")
    # one restriction can be tested with two clusters
    assert sp.xtoverid(d, "y", ["x"], id="id", cluster="cl")["df"] == 1


# ---------------------------------------------------------------- 9 xtserial


def test_xtserial_omits_a_collinear_regressor(panel):
    with pytest.warns(UserWarning, match=OMITTED):
        full = sp.xtserial(
            panel.assign(x2=2 * panel.x), "y", ["x", "x2", "w"], id="id", time="t"
        )
    short = sp.xtserial(panel, "y", ["x", "w"], id="id", time="t")
    assert full["statistic"] == short["statistic"]
    assert full["pvalue"] == short["pvalue"]
    pd.testing.assert_series_equal(full["params"], short["params"])
    pd.testing.assert_series_equal(full["std_errors"], short["std_errors"])


# ---------------------------------------------------------------- 10 nls


@pytest.fixture(scope="module")
def curve():
    rng = np.random.default_rng(2)
    d = pd.DataFrame({"x": rng.uniform(1, 5, 800)})
    d["y"] = 2 + 3 * d.x**0.5 + rng.normal(size=800)
    return d


@pytest.mark.parametrize("formula", ["y ~ {a} + {b} + {c}*x", "y ~ {a}*{b}*x"])
def test_nls_refuses_parameters_that_are_not_identified(curve, formula):
    with pytest.raises(NumericalInstability, match="rank deficient"):
        _quiet(lambda: sp.nls(formula, curve))


def test_nls_predict_evaluates_the_regression_function(curve):
    fit = sp.nls("y ~ {a} + {b}*x^{c}", curve, start={"c": 1})
    new = pd.DataFrame({"x": [1.0, 2.5, 4.0]})
    a, b, c = fit.params[["a", "b", "c"]]
    np.testing.assert_allclose(fit.predict(new), a + b * new.x**c, rtol=1e-12)
    np.testing.assert_array_equal(fit.predict(curve), fit.predict())
    # a linear model written as nls: the delta-method interval is the
    # least-squares one
    lin = sp.nls("y ~ {a} + {b}*x", curve)
    ols = sp.regress("y ~ x", curve)
    for what in ("confidence", "prediction"):
        np.testing.assert_allclose(
            lin.predict(new, what=what), ols.predict(new, what=what), rtol=1e-6
        )


def test_nls_predict_with_a_python_function(curve):
    fit = sp.nls(
        lambda p, d: p["a"] + p["b"] * d["x"] ** p["c"],
        curve,
        start={"a": 1, "b": 1, "c": 1},
        y="y",
    )
    new = pd.DataFrame({"x": [1.0, 2.5]})
    a, b, c = fit.params[["a", "b", "c"]]
    np.testing.assert_allclose(fit.predict(new), a + b * new.x**c, rtol=1e-12)


@pytest.mark.parametrize(
    "kw",
    [{}, {"regime": ["x1", "z1"]}, {"kink": True}],
    ids=["all", "regime", "kink"],
)
def test_threshold_predict_reproduces_the_fitted_values(df, kw):
    fit = _quiet(lambda: sp.threshold("y ~ x1", df, threshold="x2", **kw))
    np.testing.assert_allclose(
        fit.predict(df), fit.data_info["fitted_values"], rtol=1e-10, atol=1e-10
    )


def test_threshold_predict_interval_is_the_least_squares_one(df):
    # given the threshold the model is linear in indicator interactions
    fit = _quiet(lambda: sp.threshold("y ~ x1", df, threshold="x2", vce="ols"))
    gamma = fit.model_info["threshold"]
    d = df.assign(above=(df.x2 > gamma).astype(float))
    ols = sp.regress("y ~ x1 + above + above:x1", d)
    np.testing.assert_allclose(
        fit.predict(df.head(6), what="confidence")[["yhat", "se"]],
        ols.predict(d.head(6), what="confidence")[["yhat", "se"]],
        rtol=1e-8,
    )


def test_threshold_predict_is_refused_with_absorbed_effects(df):
    fit = _quiet(lambda: sp.threshold("y ~ x1", df, threshold="x2", absorb="g"))
    with pytest.raises(MethodIncompatibility, match="absorb"):
        fit.predict(df)


# ---------------------------------------------------------------- 11 ctree


def test_ctree_docstring_example_splits_on_age():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"age": rng.uniform(1, 70, 500), "noise": rng.normal(size=500)})
    d["survived"] = (rng.uniform(size=500) < np.where(d["age"] < 20, 0.7, 0.35)).astype(
        int
    )
    fit = sp.ctree(d, y="survived", covariates=["age", "noise"], task="classification")
    assert fit.nodes.loc[1, "variable"] == "age"
    assert 'df["age"] < 20' in sp.ctree.__doc__


# ---------------------------------------------------------------- 12 ivpoisson


def test_ivpoisson_refuses_an_all_zero_outcome(df):
    with pytest.raises(DataInsufficient, match="zero on every"):
        sp.ivpoisson(df.assign(zero=0), "zero", ["x1"], ["e"], ["z1"])


# ---------------------------------------------------------------- 13 labels


@pytest.mark.parametrize(
    "call",
    [
        lambda d: sp.ivprobit(d, "c", ["x1", "lab"], ["e"], ["z1"]),
        lambda d: sp.ivpoisson(d, "cnt", ["x1", "lab"], ["e"], ["z1"]),
        lambda d: sp.etpoisson(d, "cnt", ["x1", "lab"], "tr", ["z1"]),
        lambda d: sp.xtsum(d, ["x1", "lab"], id="g"),
        lambda d: sp.xtoverid(d, "y", ["x1", "lab"], id="g"),
        lambda d: sp.best_subset(d, "y", ["x1", "x2", "lab"]),
        lambda d: sp.blp_instruments(d, ["x1", "lab"], market_id="g"),
        lambda d: sp.threshold("y ~ x1", d, threshold="lab"),
        lambda d: sp.series("y ~ x2", d, "lab"),
        lambda d: sp.shrinkage(d, "y", ["x1", "lab"]),
    ],
    ids=[
        "ivprobit",
        "ivpoisson",
        "etpoisson",
        "xtsum",
        "xtoverid",
        "best_subset",
        "blp_instruments",
        "threshold",
        "series",
        "shrinkage",
    ],
)
@pytest.mark.parametrize("kind", ["string", "category"])
def test_a_label_column_is_named_in_the_error(df, call, kind):
    lab = pd.Series(np.where(df.x2 > 0, "a", "b"), index=df.index)
    d = df.assign(lab=lab.astype("category") if kind == "category" else lab)
    with pytest.raises(MethodIncompatibility, match=r"\['lab'\] hold labels"):
        call(d)


def test_numbers_stored_as_text_still_pass_shrinkage(df):
    text = df.assign(x2=df.x2.round(3).astype(str))
    number = df.assign(x2=df.x2.round(3))
    a = sp.shrinkage(text, "y", ["x1", "x2"], seed=1)
    b = sp.shrinkage(number, "y", ["x1", "x2"], seed=1)
    pd.testing.assert_series_equal(a.params, b.params)


# ---------------------------------------------------------------- 14 smaller


@pytest.mark.parametrize("alpha", [1.5, 0.0, -0.1, "a"])
def test_boxcox_validates_alpha(df, alpha):
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.boxcox("pos ~ x1", df.assign(pos=np.exp(0.2 * df.y)), alpha=alpha)


@pytest.mark.parametrize("value", [2.0, 1 / 3])
def test_boxcox_refuses_a_constant_outcome(df, value):
    with pytest.raises(DataInsufficient, match="does not vary"):
        sp.boxcox("yy ~ x1", df.assign(yy=value))


def test_rolling_announces_a_globally_collinear_regressor(df):
    with pytest.warns(UserWarning, match=OMITTED):
        full = sp.rolling("y ~ x1 + x2 + x3", df, 60, step=20)
    short = sp.rolling("y ~ x1 + x2", df, 60, step=20)
    pd.testing.assert_frame_equal(full, short)
    assert [o["variable"] for o in full.attrs["omitted"]] == ["x3"]
    assert short.attrs["omitted"] == []


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "market_id": np.repeat(np.arange(20.0), 5),
            "firm_id": np.tile([0.0, 0.0, 1.0, 1.0, 2.0], 20),
            "size": rng.normal(size=100),
        }
    )


@pytest.mark.parametrize("column", ["market_id", "firm_id"])
def test_blp_instruments_refuses_a_missing_group(market, column):
    d = market.copy()
    d.loc[3, column] = np.nan
    with pytest.raises(DataInsufficient, match=column):
        sp.blp_instruments(d, ["size"], firm_id="firm_id")


def test_blp_instruments_reads_a_single_name(market):
    pd.testing.assert_frame_equal(
        sp.blp_instruments(market, "size", firm_id="firm_id"),
        sp.blp_instruments(market, ["size"], firm_id="firm_id"),
    )


def test_xthtaylor_requires_units_nested_in_clusters(panel):
    kw = dict(id="id", endog=["x", "ti2"], vce="robust")
    formula = "y ~ x + w + ti + ti2"
    with pytest.raises(MethodIncompatibility, match="nested"):
        sp.xthtaylor(formula, panel.assign(cl=panel.t), cluster="cl", **kw)
    # a relabelling of the units is the unit-clustered fit
    nested = sp.xthtaylor(
        formula, panel.assign(cl=panel.id * 7 + 1), cluster="cl", **kw
    )
    assert _same(nested, sp.xthtaylor(formula, panel, **kw))


@pytest.mark.parametrize("max_vars", [0, -1])
def test_best_subset_rejects_a_non_positive_size_limit(df, max_vars):
    with pytest.raises(MethodIncompatibility, match="max_vars"):
        sp.best_subset(df, "y", ["x1", "x2", "z1"], max_vars=max_vars)


def test_glmnet_predict_names_the_missing_columns(df):
    fit = sp.glmnet(df, "y", ["x1", "x2", "z1"], seed=1)
    with pytest.raises(MethodIncompatibility, match=r"\['z1'\]"):
        fit.predict(df[["x1", "x2"]])


def test_nlogit_says_when_every_free_parameter_is_fixed():
    rng = np.random.default_rng(0)
    n, modes = 200, ["air", "bus", "car", "train"]
    d = pd.DataFrame({"trip": np.repeat(np.arange(n), 4), "mode": modes * n})
    d["cost"] = rng.normal(size=len(d))
    utility = -0.8 * d["cost"].to_numpy() + rng.gumbel(size=len(d))
    best = utility.reshape(n, 4).argmax(axis=1)
    d["chosen"] = (np.tile(np.arange(4), n) == np.repeat(best, 4)).astype(int)
    kw = dict(y="chosen", x=["cost"], chid="trip", alt="mode")
    nests = {"ground": ["bus", "car", "train"], "air": ["air"]}
    with pytest.raises(MethodIncompatibility, match="none is left to estimate"):
        sp.nlogit(d, nests=nests, fixed_lambda={"ground": 1.0}, **kw)
    singles = {m: [m] for m in modes}
    with pytest.raises(MethodIncompatibility, match="single alternative"):
        sp.nlogit(d, nests=singles, **kw)


# --------------------------------------------------------------------- #
# predict() on a factor level the fit never saw: glm, poisson, nbreg
# --------------------------------------------------------------------- #


def _party_frame():
    rng = np.random.default_rng(19)
    n = 600
    d = pd.DataFrame({"pid": rng.choice(["R", "D", "I"], n, p=[0.5, 0.3, 0.2])})
    share = d.pid.map({"R": 0.8, "D": 0.1, "I": 0.4})
    d["vote"] = (rng.uniform(size=n) < share).astype(int)
    d["cnt"] = rng.poisson(d.pid.map({"R": 3.0, "D": 1.0, "I": 2.0}))
    return d


_LEVEL_FITS = {
    "glm": lambda d: sp.glm("vote ~ C(pid)", d, family="binomial"),
    "poisson": lambda d: sp.poisson("cnt ~ C(pid)", d),
    "nbreg": lambda d: sp.nbreg("cnt ~ C(pid)", d),
}


@pytest.mark.parametrize("name", sorted(_LEVEL_FITS))
def test_predict_refuses_a_level_the_fit_never_saw(name):
    d = _party_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _LEVEL_FITS[name](d)
    with pytest.raises(sp.MethodIncompatibility, match="pid"):
        fit.predict(pd.DataFrame({"pid": ["X"]}))


@pytest.mark.parametrize("name", sorted(_LEVEL_FITS))
def test_predict_on_known_levels_is_the_cell_mean(name):
    # a saturated model reproduces the cell means, reference level included
    d = _party_frame()
    y = "vote" if name == "glm" else "cnt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _LEVEL_FITS[name](d)
        got = np.asarray(fit.predict(pd.DataFrame({"pid": ["D", "I", "R"]})))
    want = d.groupby("pid")[y].mean().loc[["D", "I", "R"]].to_numpy()
    np.testing.assert_allclose(got, want, rtol=1e-6)
