"""Fixes from running the Python Polars edition of Gow and Lubberink,
*Empirical Research in Accounting* (https://iangow.github.io/era_pl_book/),
through StatsPAI chapter by chapter (2026-11-19).

References are the statsmodels / pyfixest calls the book itself makes.

* A column read only as category labels may hold ``inf``: the book codes a
  never-treated firm's event time as ``-inf`` inside ``C(t,
  Treatment(reference=-1))`` (chapter 21). Numeric uses of ``inf`` are
  still refused.
* A text regressor (Fama-French industry ``ff12`` as ``"1"`` ... ``"12"``,
  chapter 24) is expanded into indicators by the count, ordered and
  quantile models; it used to be cast to float and fitted as one number.
* ``sp.regress(cluster=[a, b], <unknown>=...)`` raises like the one-way
  path does.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

smf = pytest.importorskip("statsmodels.formula.api")
sm = pytest.importorskip("statsmodels.api")


# ---------------------------------------------------------------- inf labels
@pytest.fixture
def event_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame(
        {
            "t": rng.choice([-np.inf, -2.0, -1.0, 0.0, 1.0], n),
            "y": rng.normal(size=n),
            "g": rng.integers(0, 25, n),
        }
    )


def test_regress_accepts_inf_as_a_factor_level(event_panel):
    f = "y ~ C(t, Treatment(reference=-1))"
    ref = smf.ols(f, event_panel).fit()
    ours = sp.regress(f, event_panel)
    assert list(ours.params.index) == list(ref.params.index)
    np.testing.assert_allclose(ours.params, ref.params, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ours.std_errors, ref.bse, rtol=1e-10)


def test_feols_accepts_inf_as_a_factor_level(event_panel):
    pf = pytest.importorskip("pyfixest")
    for f in ("y ~ C(t, Treatment(reference=-1)) | g", "y ~ i(t, ref=-1) | g"):
        ref = pf.feols(f, data=event_panel, vcov="iid")
        ours = sp.feols(f, data=event_panel, vcov="iid")
        np.testing.assert_allclose(
            ours.params.reindex(ref.coef().index), ref.coef(), rtol=1e-10
        )
        np.testing.assert_allclose(
            ours.std_errors.reindex(ref.se().index), ref.se(), rtol=1e-10
        )


@pytest.mark.parametrize("formula", ["y ~ t", "y ~ C(t) + t", "t ~ C(g)"])
def test_inf_in_a_numeric_use_is_still_refused(event_panel, formula):
    with pytest.raises(DataInsufficient):
        sp.regress(formula, event_panel)


def test_inf_in_a_numeric_feols_regressor_is_still_refused(event_panel):
    with pytest.raises(DataInsufficient):
        sp.feols("y ~ t | g", event_panel)


def test_inf_label_column_used_as_weights_is_still_refused(event_panel):
    d = event_panel.assign(w=np.where(np.isinf(event_panel["t"]), np.inf, 1.0))
    with pytest.raises(DataInsufficient):
        sp.regress("y ~ C(t)", d, weights="w")


def test_label_only_columns_reads_the_formula():
    from statspai._input_validation import label_only_columns

    cols = ["y", "t", "g", "x", "z"]
    assert label_only_columns("y ~ C(t) + x | g", cols) == {"t", "g"}
    assert label_only_columns("y ~ C(t) + t", cols) == set()
    assert label_only_columns("y ~ x | g | x ~ z", cols) == {"g"}
    assert label_only_columns("t ~ C(t)", cols) == set()


# ------------------------------------------------------------- text regressor
@pytest.fixture(params=[["1", "2", "3", "10"], ["a", "b", "c", "d"]])
def labelled(request) -> pd.DataFrame:
    labels = request.param
    rng = np.random.default_rng(3)
    n = 800
    d = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(labels, n)})
    d["y"] = rng.poisson(np.exp(0.3 * d["x"] + (d["g"] == labels[1]) * 0.5))
    d["yc"] = d["y"] + rng.normal(size=n)
    return d


def _aligned(ours: pd.Series, ref: pd.Series) -> np.ndarray:
    ref = ref.rename({"Intercept": "_cons"})
    return (
        ours.rename({"const": "_cons", "Intercept": "_cons"})
        .reindex(ref.index)
        .to_numpy()
    )


def test_poisson_expands_a_text_regressor(labelled):
    ref = smf.glm("y ~ x + g", labelled, family=sm.families.Poisson()).fit(tol=1e-12)
    ours = sp.poisson("y ~ x + g", labelled)
    np.testing.assert_allclose(_aligned(ours.params, ref.params), ref.params, rtol=1e-7)
    np.testing.assert_allclose(_aligned(ours.std_errors, ref.bse), ref.bse, rtol=1e-6)


def test_nbreg_expands_a_text_regressor(labelled):
    ours = sp.nbreg("y ~ x + g", labelled)
    assert {"g[T.%s]" % lv for lv in sorted(labelled["g"].unique())[1:]} <= set(
        ours.params.index
    )
    assert "g" not in ours.params.index


def test_qreg_expands_a_text_regressor(labelled):
    ref = smf.quantreg("yc ~ x + g", labelled).fit(q=0.5)
    ours = sp.qreg("yc ~ x + g", labelled)
    np.testing.assert_allclose(_aligned(ours.params, ref.params), ref.params, atol=1e-5)


def test_ologit_expands_a_text_regressor(labelled):
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    d = labelled.assign(yo=np.clip(labelled["y"], 0, 3))
    levels = sorted(d["g"].unique())
    exog = pd.DataFrame({"x": d["x"]})
    for lv in levels[1:]:  # patsy's treatment coding: first sorted level is base
        exog[f"g[T.{lv}]"] = (d["g"] == lv).astype(float)
    ref = OrderedModel(d["yo"], exog, distr="logit").fit(
        method="bfgs", disp=0, maxiter=5000, gtol=1e-10
    )
    ours = sp.ologit("yo ~ x + g", d)
    slopes = list(exog.columns)
    np.testing.assert_allclose(
        ours.params.reindex(slopes), ref.params.reindex(slopes), rtol=1e-4
    )


@pytest.mark.parametrize("fn", ["zinb", "hurdle"])
def test_models_without_factor_expansion_refuse_text(labelled, fn):
    with pytest.raises(MethodIncompatibility, match="category labels"):
        getattr(sp, fn)("y ~ x + g", data=labelled)


def test_column_list_entry_refuses_text(labelled):
    with pytest.raises(MethodIncompatibility, match="category labels"):
        sp.poisson(data=labelled, y="y", x=["x", "g"])
    with pytest.raises(MethodIncompatibility, match="category labels"):
        sp.qreg(data=labelled, y="yc", x=["x", "g"])


# ------------------------------------------------------- regress **kwargs
def test_two_way_cluster_rejects_unknown_keywords():
    rng = np.random.default_rng(1)
    d = pd.DataFrame(
        {
            "y": rng.normal(size=60),
            "x": rng.normal(size=60),
            "a": np.repeat(range(12), 5),
            "b": np.tile(range(5), 12),
        }
    )
    with pytest.raises(TypeError, match="ssc"):
        sp.regress("y ~ x", d, cluster=["a", "b"], ssc={"G_adj": False})
    with pytest.raises(TypeError, match="foo"):
        sp.regress("y ~ x", d, cluster="a", vce="cr2", foo=1)
    with pytest.raises(TypeError, match="wild_reps"):
        sp.regress("y ~ x", d, cluster="a", wild_reps=99)


# ------------------------------------------------- second pass (2026-11-19)
def test_ivreg_expands_a_text_exogenous_regressor(labelled):
    pf = pytest.importorskip("pyfixest")
    d = labelled.assign(
        z=labelled["x"] + np.random.default_rng(9).normal(size=len(labelled))
    )
    ours = sp.ivreg("yc ~ g + (x ~ z)", d)
    ref = pf.feols("yc ~ C(g) | x ~ z", data=d)
    ref_coef = ref.coef().rename(lambda k: k.replace("C(g)", "g"))
    np.testing.assert_allclose(ours.params.reindex(ref_coef.index), ref_coef, rtol=1e-9)
    with pytest.raises(MethodIncompatibility, match="category labels"):
        sp.ivreg("yc ~ x + (g ~ z)", d)


@pytest.mark.parametrize(
    "call",
    [
        lambda d: sp.hdfe_ols("yc ~ x + g", d),
        lambda d: sp.rifreg("yc ~ x + g", d),
        lambda d: sp.heckman(d, y="yc", x=["x", "g"], z=["x", "g", "y"], select=None),
        lambda d: sp.fracreg(
            d.assign(f=d["yc"] / (1 + d["yc"].max())), y="f", x=["x", "g"]
        ),
        lambda d: sp.logit(
            data=d.assign(b=(d["y"] > 0).astype(int)), y="b", x=["x", "g"]
        ),
        lambda d: sp.tobit(data=d, y="yc", x=["x", "g"]),
        lambda d: sp.mixed(
            d.assign(grp=np.arange(len(d)) // 8),
            y="yc",
            x_fixed=["x", "g"],
            group="grp",
        ),
        lambda d: sp.liml(
            data=d.assign(z=d["x"]), y="yc", x_endog=["x"], x_exog=["g"], z=["z"]
        ),
    ],
    ids=[
        "hdfe_ols",
        "rifreg",
        "heckman",
        "fracreg",
        "logit_xy",
        "tobit_xy",
        "mixed",
        "liml",
    ],
)
def test_estimators_without_expansion_refuse_text(labelled, call):
    with pytest.raises(MethodIncompatibility, match="category labels"):
        call(labelled)


def test_cox_expands_a_text_covariate(labelled):
    rng = np.random.default_rng(4)
    d = labelled.assign(
        dur=rng.exponential(1.0, len(labelled)) * np.exp(-0.3 * labelled["x"]),
        ev=(rng.random(len(labelled)) < 0.8).astype(int),
    )
    ours = sp.cox("dur ~ x + g", d, event="ev")
    ref = smf.phreg("dur ~ x + g", d, status=d["ev"].to_numpy(), ties="efron").fit()
    ref_params = pd.Series(ref.params, index=ref.model.exog_names)
    np.testing.assert_allclose(
        ours.params.reindex(ref_params.index), ref_params, rtol=1e-6
    )


def test_regtable_keep_and_drop_take_patterns():
    rng = np.random.default_rng(2)
    d = pd.DataFrame({"x": rng.normal(size=200), "yr": rng.choice([1, 2, 3], 200)})
    d["y"] = d["x"] + rng.normal(size=200)
    fit = sp.regress("y ~ x * C(yr)", d)
    names = list(fit.params.index)
    from statspai.output.regression_table import _match_coef_names

    assert _match_coef_names(names, ["yr"], "drop") == {n for n in names if "yr" in n}
    assert _match_coef_names(names, ["x"], "keep") == {"x"}  # exact name wins
    assert _match_coef_names(names, ["x:*"], "keep") == {
        n for n in names if n.startswith("x:")
    }
    with pytest.warns(UserWarning, match="matched no coefficient"):
        _match_coef_names(names, ["nothere"], "keep")
    text = str(sp.regtable(fit, drop=["yr"]))
    assert "C(yr)" not in text and "x" in text


def test_rdplot_passes_masspoints():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(5)
    x = np.round(rng.uniform(-1, 1, 800), 1)  # heavy mass points
    d = pd.DataFrame({"x": x, "y": x + (x >= 0) + rng.normal(0, 0.3, 800)})
    from statspai.rd._rdplot_numbers import rdplot_numbers

    for mp in ("off", "adjust"):
        fig, _ = sp.rdplot(d, y="y", x="x", c=0, masspoints=mp)
        ref = rdplot_numbers(d["y"].to_numpy(), d["x"].to_numpy(), c=0, masspoints=mp)
        assert tuple(fig.rdplot_data["J"]) == tuple(ref["J"])


def _panel_text_frame():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame(
        {"id": np.repeat(np.arange(30), 10), "time": np.tile(np.arange(10), 30)}
    )
    df["x"] = rng.normal(size=n)
    df["g"] = rng.choice(["1", "2", "10"], n)
    df["h"] = np.where(df["id"] % 3 == 0, "7", "8")  # constant within unit
    df["y"] = df["x"] + (df["g"] == "10") + (df["h"] == "8") + rng.normal(size=n)
    for col, lev in (("g", "10"), ("g", "2"), ("h", "8")):
        df[f"{col}{lev}"] = (df[col] == lev).astype(float)
    return df


@pytest.mark.parametrize("method", ["re", "fe", "be", "pooled", "twoway"])
@pytest.mark.parametrize("cluster", [None, "id"])
def test_panel_text_regressor_equals_its_indicators(method, cluster):
    # method='re' used to raise TypeError (unit means of a text column), and
    # the ssc= covariance cast the labels to numbers
    pytest.importorskip("linearmodels")
    if method == "be" and cluster is not None:
        pytest.skip("between estimator has conventional SEs only")
    df = _panel_text_frame()
    kw = dict(entity="id", time="time", method=method, cluster=cluster)
    a = sp.panel(df, "y ~ x + g + h", **kw)
    b = sp.panel(df, "y ~ x + g10 + g2 + h8", **kw)
    assert [c.replace(".", "") for c in a.params.index] == list(b.params.index)
    np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(a.std_errors), np.asarray(b.std_errors), rtol=1e-10
    )


_COLUMN_LIST_CALLS = {
    "xtabond": lambda d: sp.xtabond(d, y="y", x=["x", "g"]),
    "xtdpdsys": lambda d: sp.xtdpdsys(d, y="y", x=["x", "g"]),
    "ardl": lambda d: sp.ardl(d, y="y", x=["x", "g"]),
    "xtpcse": lambda d: sp.xtpcse(d, y="y", x=["x", "g"], id="id", time="time"),
    "xtregar": lambda d: sp.xtregar(d, y="y", x=["x", "g"], id="id", time="time"),
    "mmqreg": lambda d: sp.mmqreg(d, y="y", x=["x", "g"]),
    "ridge": lambda d: sp.ridge(data=d, y="y", x=["x", "g"]),
}


@pytest.mark.parametrize("name", sorted(_COLUMN_LIST_CALLS))
def test_column_list_estimators_refuse_text_but_fit_numbers(name):
    # found by a sweep: each fitted the digit labels as one number, giving
    # exactly the estimates of the float-coded column
    df = _panel_text_frame()
    call = _COLUMN_LIST_CALLS[name]
    with pytest.raises(MethodIncompatibility, match="category labels"):
        call(df)
    call(df.assign(g=df["g"].astype(float)))  # a numeric column still fits


def test_ridge_formula_expands_text():
    df = _panel_text_frame()
    res = sp.ridge("y ~ x + g", data=df)
    assert {"g[T.10]", "g[T.2]"} <= set(res.params.index)


def _cov_frame():
    df = _panel_text_frame()
    rng = np.random.default_rng(2)
    df["d"] = (rng.normal(size=len(df)) + 0.3 * df["x"] > 0).astype(int)
    df["x2"] = rng.normal(size=len(df))
    df["z"] = df["x2"] + rng.normal(size=len(df))
    df["g[T.10]"] = (df["g"] == "10").astype(float)
    df["g[T.2]"] = (df["g"] == "2").astype(float)
    df["m"] = 0.5 * df["d"] + df["x"] + rng.normal(size=len(df))
    df["dur"] = rng.exponential(1.0, len(df))
    df["ev"] = rng.integers(0, 3, len(df))  # competing causes 1, 2
    df["ev01"] = (df["ev"] > 0).astype(int)
    df["ylo"] = df["y"] - 1
    df["yhi"] = df["y"] + 1
    df["prot"] = rng.integers(0, 2, len(df))
    df["worker"] = df["id"]
    df["firm"] = df["id"] % 6
    return df


_EXPANDING = {
    "didregress": lambda d, c: sp.didregress(
        data=d, y="y", treat="d", group="id", time="time", covariates=c
    ),
    "westfall_young": lambda d, c: sp.westfall_young(
        data=d, y="y", treat="d", covariates=c, seed=0
    ),
    "qte": lambda d, c: sp.qte(data=d, y="y", treatment="d", controls=c),
    "psm": lambda d, c: sp.psm(d, y="y", d="d", X=c),
    "conformal_cate": lambda d, c: sp.conformal_cate(
        data=d, y="y", treat="d", covariates=c, random_state=0
    ),
    "dml_did": lambda d, c: sp.dml_did(data=d, y="y", treat="d", covariates=c),
    "dml_panel": lambda d, c: sp.dml_panel(
        data=d, y="y", treat="d", unit="id", covariates=c
    ),
    "dynamic_dml": lambda d, c: sp.dynamic_dml(
        data=d, y="y", treat="d", id="id", time="time", covariates=c
    ),
    "mediate": lambda d, c: sp.mediate(
        data=d, y="y", treat="d", mediator="m", covariates=c, n_boot=50, seed=0
    ),
    "negative_control_outcome": lambda d, c: sp.negative_control_outcome(
        data=d, nco="x2", treat="d", covariates=c
    ),
    "stteffects": lambda d, c: sp.stteffects(
        data=d, duration="dur", event="ev01", treat="d", covariates=c
    ),
    "akm": lambda d, c: sp.akm(
        data=d, y="y", worker="worker", firm="firm", covariates=c, seed=0
    ),
}


def _numbers(res):
    if isinstance(res, pd.DataFrame):
        return res.select_dtypes("number").to_numpy(dtype=float).ravel()
    for attr in ("estimate", "params", "att", "ate"):
        v = getattr(res, attr, None)
        if v is not None:
            try:
                return np.atleast_1d(np.asarray(v, dtype=float))
            except (TypeError, ValueError):
                continue
    raise AssertionError(f"no numbers on {type(res).__name__}")


@pytest.mark.parametrize("name", sorted(_EXPANDING))
def test_covariate_list_text_is_expanded_like_explicit_indicators(name):
    # these cast the labels to one number before; covariate lists follow
    # core/_covariates.py: a categorical entry becomes g[T.level] columns
    df = _cov_frame()
    call = _EXPANDING[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = _numbers(call(df, ["x", "g"]))
        b = _numbers(call(df, ["x", "g[T.10]", "g[T.2]"]))
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


_REFUSING = {
    "glmnet": lambda d: sp.glmnet(d, y="y", x=["x", "g"], cv=False, lambda_=0.1),
    "ivqreg": lambda d: sp.ivqreg(
        data=d, y="y", endog="x2", instruments=["z"], exog=["x", "g"]
    ),
    "ivqreg_instruments": lambda d: sp.ivqreg(
        data=d, y="y", endog="x2", instruments=["z", "g"], exog=["x"]
    ),
    "rlasso_iv": lambda d: sp.rlasso_iv(data=d, y="y", d="d", z=["z"], x=["x", "g"]),
    "interactive_fe": lambda d: sp.interactive_fe(data=d, y="y", x=["x", "g"]),
    "etregress": lambda d: sp.etregress(
        data=d, y="y", x=["x", "g"], treatment="d", z=["z"]
    ),
    "truncreg": lambda d: sp.truncreg(data=d, y="y", x=["x", "g"]),
    "panel_fgls": lambda d: sp.panel_fgls(data=d, y="y", x=["x", "g"]),
    "frontier": lambda d: sp.frontier(data=d, y="y", x=["x", "g"]),
    "xtfrontier": lambda d: sp.xtfrontier(data=d, y="y", x=["x", "g"], id="id"),
    "zisf": lambda d: sp.zisf(data=d, y="y", x=["x", "g"]),
    "lcsf": lambda d: sp.lcsf(data=d, y="y", x=["x", "g"]),
    "arima": lambda d: sp.arima(data=d, y="y", exog=["x", "g"]),
    "noisy_covariates": lambda d: sp.noisy_covariates(
        data=d, y="y", treat="d", covariates=["x", "g"]
    ),
    "xtintreg": lambda d: sp.xtintreg(
        data=d, y_lower="ylo", y_upper="yhi", x=["x", "g"], id="id"
    ),
    "finegray": lambda d: sp.finegray(data=d, duration="dur", event="ev", x=["x", "g"]),
    "orthogonal_to_bias": lambda d: sp.orthogonal_to_bias(
        data=d, features=["x", "g"], protected="prot"
    ),
}


@pytest.mark.parametrize("name", sorted(_REFUSING))
def test_regressor_list_text_is_refused(name):
    with pytest.raises(MethodIncompatibility, match="category labels"):
        _REFUSING[name](_cov_frame())


def test_ddd_heterogeneous_and_mc_synth_expand_text_covariates():
    rng = np.random.default_rng(1)
    N, T = 120, 8
    df = pd.DataFrame(
        {"id": np.repeat(np.arange(N), T), "time": np.tile(np.arange(T), N)}
    )
    df["x"] = np.repeat(rng.normal(size=N), T)
    df["g"] = np.repeat(rng.choice(["1", "2", "10"], N), T)
    df["coh"] = np.repeat(
        np.where(np.arange(N) % 3 == 0, 0, 4 + 2 * (np.arange(N) % 3 == 2)), T
    )
    df["sg"] = np.repeat(rng.integers(0, 2, N), T)
    df["y"] = (
        1
        + df["x"]
        + 0.5 * ((df["coh"] > 0) & (df["time"] >= df["coh"]) & (df["sg"] == 1))
    )
    df["y"] += rng.normal(size=len(df))
    df["g[T.10]"] = (df["g"] == "10").astype(float)
    df["g[T.2]"] = (df["g"] == "2").astype(float)
    dummies = ["x", "g[T.10]", "g[T.2]"]

    def ddd(c):
        return sp.ddd_heterogeneous(
            df,
            y="y",
            unit="id",
            time="time",
            cohort="coh",
            subgroup="sg",
            x=c,
            seed=1,
            n_boot=50,
            est_method="reg",
        ).estimate

    def mc(c):
        return sp.mc_synth(
            df,
            outcome="y",
            unit="id",
            time="time",
            treated_unit=0,
            treatment_time=6,
            covariates=c,
            seed=3,
            placebo=False,
        ).estimate

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ddd(["x", "g"]) == pytest.approx(ddd(dummies), rel=1e-10)
        assert mc(["x", "g"]) == pytest.approx(mc(dummies), rel=1e-10)


def test_panel_fd_omits_a_regressor_the_differences_remove():
    pytest.importorskip("linearmodels")
    df = _panel_text_frame()
    df["h8"] = (df["h"] == "8").astype(float)  # constant within unit
    with pytest.warns(UserWarning, match="first differences remove"):
        a = sp.panel(df, "y ~ x + h8", entity="id", time="time", method="fd")
    b = sp.panel(df, "y ~ x", entity="id", time="time", method="fd")
    assert a.model_info["omitted"] == ["h8"]
    np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-12)
    with pytest.raises(MethodIncompatibility, match="every regressor"):
        sp.panel(df, "y ~ h8", entity="id", time="time", method="fd")


def test_panel_be_with_cluster_is_refused():
    pytest.importorskip("linearmodels")
    df = _panel_text_frame()
    with pytest.raises(MethodIncompatibility, match="no clustered standard errors"):
        sp.panel(df, "y ~ x", entity="id", time="time", method="be", cluster="id")


def test_bcf_longitudinal_needs_twenty_bootstrap_draws():
    df = _panel_text_frame()
    df["d"] = (df["x"] > 0).astype(int)
    with pytest.raises(MethodIncompatibility, match="at least 20"):
        sp.bcf_longitudinal(
            df,
            outcome="y",
            treatment="d",
            unit="id",
            time="time",
            covariates=["x"],
            n_trees_mu=10,
            n_trees_tau=5,
            n_bootstrap=5,
        )
