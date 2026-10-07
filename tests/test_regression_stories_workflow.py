"""Behaviour and edge cases of the regression-workflow additions.

Numerical agreement with R lives in
``tests/reference_parity/test_regression_stories_r_parity.py`` and the
samplers' exact-posterior checks in
``tests/reference_parity/test_bayes_workflow_exact_posterior.py``. This
file covers what those do not: argument validation, the loud failures,
and the formula spellings.
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
    StatsPAIWarning,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 240
    g = rng.integers(0, 3, n)
    out = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "a": (g == 0) * 1.0,
            "b": (g == 1) * 1.0,
            "c": (g == 2) * 1.0,
            "g": np.array(["p", "q", "r"])[g],
        }
    )
    eta = 0.3 + 0.8 * out["x"] + 0.5 * out["a"] - 0.4 * out["b"]
    out["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    out["cnt"] = rng.poisson(np.exp(0.2 + 0.3 * out["x"] + 0.3 * out["a"]))
    out["yc"] = eta + rng.normal(size=n)
    out["o"] = np.digitize(out["yc"], [-0.3, 0.8])
    return out


@pytest.fixture(scope="module")
def fit(df):
    return sp.bayes_regress(
        "yc ~ x + z", df, prior="weakly_informative", draws=1500, burnin=300, seed=1
    )


# ---------------------------------------------------------------------
# loo / waic / kfold / compare
# ---------------------------------------------------------------------


def test_loo_result_shape_and_export(fit):
    out = sp.loo(fit)
    assert out.kind == "loo" and out.n_obs == 240 and out.n_draws == 1500
    assert list(out.estimates.index) == ["elpd", "p", "ic"]
    assert out.ic == pytest.approx(-2 * out.elpd)
    # three coefficients and a variance
    assert 2.5 < out.p < 6.5
    assert out.pareto_k_table()["count"].sum() == 240
    assert "Pareto k" in out.summary()
    json.dumps(out.to_dict())
    assert out.cite()
    assert out.plot() is not None


def test_loo_accepts_a_matrix_and_checks_it(fit):
    ll = fit.log_lik()
    a, b = sp.loo(ll), sp.loo(pd.DataFrame(ll))
    assert a.elpd == pytest.approx(b.elpd)
    with pytest.raises(MethodIncompatibility, match="missing or infinite"):
        bad = ll.copy()
        bad[3, 4] = -np.inf
        sp.loo(bad)
    with pytest.raises(DataInsufficient):
        sp.loo(ll[:10])
    with pytest.raises(MethodIncompatibility, match="r_eff"):
        sp.loo(ll, r_eff=[0.5, 0.2])
    with pytest.raises(MethodIncompatibility):
        sp.loo("not a model")


def test_loo_warns_about_an_influential_observation():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.normal(size=40)})
    d["y"] = 1 + d["x"] + rng.normal(size=40)
    d.loc[0, ["x", "y"]] = [6.0, -20.0]
    f = sp.bayes_regress("y ~ x", d, prior="weakly_informative", draws=2000, seed=1)
    with pytest.warns(StatsPAIWarning, match="Pareto k values exceed"):
        out = sp.loo(f)
    assert 0 in out.bad_observations()
    assert np.isnan(out.mcse_elpd)
    with pytest.raises(MethodIncompatibility):
        sp.waic(f).pareto_k


def test_psis_weights_are_normalised_and_bounded():
    rng = np.random.default_rng(1)
    lr = rng.standard_t(3, size=(800, 4))
    out = sp.psis(lr)
    np.testing.assert_allclose(out.weights().sum(axis=0), 1.0)
    # smoothing never raises a weight above the largest raw one
    raw_max = np.exp(lr - lr.max(axis=0)).max(axis=0)
    smooth = out.weights() / out.weights().max(axis=0)
    assert np.all(smooth.max(axis=0) <= raw_max + 1e-12)
    assert out.n_eff.shape == (4,)
    assert sp.psis(lr[:, 0]).log_weights.shape == (800, 1)


def test_kfold_and_splits(df, fit):
    folds = sp.kfold_split(240, k=6, seed=3)
    assert np.bincount(folds).tolist() == [40] * 6
    grouped = sp.kfold_split(240, k=3, seed=3, groups=df["g"])
    assert (
        df.groupby("g")
        .apply(lambda s: grouped[s.index].min() == grouped[s.index].max())
        .all()
    )
    strat = sp.kfold_split(240, k=4, seed=3, stratify=df["y"])
    share = [df["y"].to_numpy()[strat == j].mean() for j in range(4)]
    assert max(share) - min(share) < 0.05
    with pytest.raises(MethodIncompatibility):
        sp.kfold_split(10, k=1)
    with pytest.raises(MethodIncompatibility):
        sp.kfold_split(10, k=3, groups=[1] * 10)

    out = sp.kfold(fit, k=5, seed=1)
    assert out.kind == "kfold" and out.model_info["k"] == 5
    loo = sp.loo(fit)
    # same quantity, estimated two ways
    assert out.elpd == pytest.approx(loo.elpd, abs=3 * loo.se_elpd / 4)
    with pytest.raises(MethodIncompatibility, match="folds has"):
        sp.kfold(fit, folds=[0, 1, 2])
    with pytest.raises(MethodIncompatibility, match="refitting recipe"):
        sp.kfold(object())


def test_kfold_with_a_user_refit(df, fit):
    def refit(train):
        return sp.bayes_regress("yc ~ x", train, draws=400, burnin=200, seed=2)

    small = sp.bayes_regress("yc ~ x", df, draws=400, burnin=200, seed=2)
    out = sp.kfold(small, k=4, seed=1, refit=refit, data=df)
    assert np.isfinite(out.elpd)
    with pytest.raises(MethodIncompatibility, match="needs data"):
        sp.kfold(small, refit=refit)


def test_loo_compare_orders_models_and_refuses_mismatched_data(df, fit):
    small = sp.bayes_regress("yc ~ 1", df, draws=1500, burnin=300, seed=1)
    table = sp.loo_compare({"full": fit, "null": small})
    assert list(table.index) == ["full", "null"]
    assert table.loc["full", "elpd_diff"] == 0.0
    assert table.loc["null", "elpd_diff"] < -20
    # paired differences: far tighter than the two standard errors combined
    assert (
        table.loc["null", "se_diff"]
        < table.loc["null", "se_elpd"] + table.loc["full", "se_elpd"]
    )
    other = sp.bayes_regress("yc ~ x", df.iloc[:200], draws=300, burnin=100, seed=1)
    with pytest.raises(
        MethodIncompatibility, match="different numbers of observations"
    ):
        sp.loo_compare(fit, other)
    with pytest.raises(MethodIncompatibility):
        sp.loo_compare(fit)
    with pytest.warns(StatsPAIWarning, match="different kinds"):
        sp.loo_compare(sp.loo(fit), sp.waic(small))


def test_loo_predict_and_r2(df, fit):
    y_loo = sp.loo_predict(fit)
    fitted = fit.predict()
    resid_fit = df["yc"].to_numpy() - fitted
    resid_loo = df["yc"].to_numpy() - y_loo
    # leaving the point out can only make its residual larger on average
    assert (resid_loo**2).mean() > (resid_fit**2).mean()
    r2 = sp.bayes_r2(fit)
    looed = sp.loo_r2(fit, seed=1)
    assert 0 < looed.estimate < r2.estimate < 1
    assert looed.estimate == pytest.approx(
        1 - resid_loo.var(ddof=1) / df["yc"].var(ddof=1)
    )
    assert sp.loo_r2(fit, seed=2).estimate == looed.estimate
    resid_kind = sp.bayes_r2(fit, kind="residual")
    assert resid_kind.estimate == pytest.approx(r2.estimate, abs=0.05)
    assert "R-squared" in r2.summary() and r2.draws.shape == (1500,)
    with pytest.raises(MethodIncompatibility):
        sp.bayes_r2(fit, kind="other")
    with pytest.raises(MethodIncompatibility):
        sp.loo_predict(fit, what="nonsense")


def test_r2_is_refused_where_it_means_nothing(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sp.bayes_regress(
            "o ~ x", df, model="oprobit", draws=300, burnin=200, seed=1
        )
    with pytest.raises(MethodIncompatibility, match="not defined"):
        sp.bayes_r2(f)
    with pytest.raises(MethodIncompatibility):
        sp.loo_r2(f)
    with pytest.raises(MethodIncompatibility):
        sp.bayes_r2(object())


def test_noise_regressors_lower_loo_r2_but_not_bayes_r2():
    rng = np.random.default_rng(3)
    n = 60
    d = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"n{i}" for i in range(8)])
    d["x"] = rng.normal(size=n)
    d["y"] = d["x"] + rng.normal(size=n)
    kw = dict(prior="weakly_informative", draws=3000, burnin=300, seed=1)
    small = sp.bayes_regress("y ~ x", d, **kw)
    big = sp.bayes_regress("y ~ x + " + " + ".join(f"n{i}" for i in range(8)), d, **kw)
    assert sp.bayes_r2(big).estimate > sp.bayes_r2(small).estimate
    assert sp.loo_r2(big, seed=1).estimate < sp.loo_r2(small, seed=1).estimate


# ---------------------------------------------------------------------
# posterior predictive checks
# ---------------------------------------------------------------------


def test_ppc_flags_overdispersion_and_passes_the_right_model():
    rng = np.random.default_rng(5)
    d = pd.DataFrame({"x": rng.normal(size=300)})
    d["y"] = rng.negative_binomial(0.6, 0.6 / (0.6 + np.exp(0.5 + 0.3 * d["x"])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pois = sp.bayes_regress(
            "y ~ x", d, model="poisson", draws=1500, burnin=500, seed=1
        )
        nb = sp.bayes_regress(
            "y ~ x", d, model="negbin", draws=1500, burnin=500, seed=1
        )
    assert sp.ppc(pois, "prop_zero", seed=1).p_value < 0.02
    assert sp.ppc(pois, "sd", seed=1).p_value < 0.02
    assert 0.05 < sp.ppc(nb, "prop_zero", seed=1).p_value < 0.95
    out = sp.ppc(nb, stat=lambda v: float(np.quantile(v, 0.9)), draws=200, seed=1)
    assert out.replicated.shape == (200,) and out.y_rep.shape == (200, 300)
    assert "Posterior predictive check" in out.summary()
    json.dumps(out.to_dict())
    assert out.plot() is not None and out.plot(kind="density") is not None
    with pytest.raises(MethodIncompatibility, match="Unknown statistic"):
        sp.ppc(nb, "kurtosis")
    with pytest.raises(MethodIncompatibility):
        out.plot(kind="pie")
    with pytest.raises(MethodIncompatibility):
        sp.ppc(object())


def test_posterior_predict_arguments(df, fit):
    assert fit.posterior_predict(draws=50, seed=1).shape == (50, 240)
    new = df.iloc[:7]
    assert fit.posterior_predict(new, seed=1).shape == (1500, 7)
    assert fit.posterior_epred(new).shape == (1500, 7)
    np.testing.assert_allclose(fit.posterior_epred(new).mean(axis=0), fit.predict(new))
    np.testing.assert_allclose(fit.posterior_linpred(new), fit.posterior_epred(new))
    np.testing.assert_array_equal(
        fit.posterior_predict(new, seed=4), fit.posterior_predict(new, seed=4)
    )
    with pytest.raises(MethodIncompatibility):
        fit.posterior_predict(draws=0)
    assert sp.mad_sd(fit.draws)["x"] == pytest.approx(fit.table.loc["x", "sd"], rel=0.1)
    assert sp.mad_sd(np.arange(5.0)) == pytest.approx(1.4826)


# ---------------------------------------------------------------------
# priors and offsets
# ---------------------------------------------------------------------


def test_weak_prior_does_not_depend_on_units(df):
    kw = dict(prior="weakly_informative", draws=4000, burnin=300, seed=1)
    base = sp.bayes_regress("yc ~ x", df, **kw)
    scaled = sp.bayes_regress("yc ~ x", df.assign(x=df["x"] * 1000 + 5e4), **kw)
    assert scaled.params["x"] * 1000 == pytest.approx(base.params["x"], rel=0.02)
    assert scaled.table.loc["x", "sd"] * 1000 == pytest.approx(
        base.table.loc["x", "sd"], rel=0.05
    )
    # while the fixed default prior pulls the shifted intercept toward zero
    with pytest.warns(StatsPAIWarning, match="prior='weakly_informative'"):
        vague = sp.bayes_regress(
            "yc ~ x", df.assign(x=df["x"] * 1000 + 5e4), draws=2000, seed=1
        )
    assert abs(vague.params["Intercept"]) < abs(scaled.params["Intercept"])


def test_weak_prior_argument_checks(df):
    with pytest.raises(MethodIncompatibility, match="leave prior_mean and prior_var"):
        sp.bayes_regress("yc ~ x", df, prior="weakly_informative", prior_var=4.0)
    with pytest.raises(MethodIncompatibility, match="is defined for models"):
        sp.bayes_regress("yc ~ x", df, model="quantile", prior="weakly_informative")
    with pytest.raises(MethodIncompatibility, match="prior must be"):
        sp.bayes_regress("yc ~ x", df, prior="flat")
    # an explicit inverse-gamma keeps the Gibbs sampler
    f = sp.bayes_regress(
        "yc ~ x", df, prior="weakly_informative", sigma2_prior=(2, 2), draws=300, seed=1
    )
    assert f.sampler == "Gibbs"


def test_weak_prior_keeps_separated_data_finite():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    d = pd.DataFrame({"x": x, "y": (x > 0).astype(int)})
    with pytest.warns(ConvergenceWarning, match="Complete separation"):
        mle = sp.logit("y ~ x", d)
    assert mle.params["x"] > 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sp.bayes_regress(
            "y ~ x", d, model="logit", prior="weakly_informative", draws=4000, seed=1
        )
    assert 2 < f.params["x"] < 20
    assert f.table.loc["x", "sd"] < 5


def test_offset_checks(df):
    d = df.assign(expo=np.linspace(0.5, 2.0, len(df)))
    with pytest.raises(MethodIncompatibility, match="not both"):
        sp.bayes_regress("cnt ~ x", d, model="poisson", offset="x", exposure="expo")
    with pytest.raises(MethodIncompatibility, match="offset / exposure are for"):
        sp.bayes_regress("yc ~ x", d, offset="x")
    with pytest.raises(MethodIncompatibility, match="must be positive"):
        sp.bayes_regress(
            "cnt ~ x", d.assign(expo=-1.0), model="poisson", exposure="expo"
        )
    with pytest.raises(MethodIncompatibility, match="not in data"):
        sp.bayes_regress("cnt ~ x", d, model="poisson", exposure="nope")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr = sp.bayes_regress(
            "cnt ~ x",
            d,
            model="poisson",
            offset=np.log(d["expo"].to_numpy()),
            draws=300,
            burnin=200,
            seed=1,
        )
        col = sp.bayes_regress(
            "cnt ~ x",
            d,
            model="poisson",
            exposure="expo",
            draws=300,
            burnin=200,
            seed=1,
        )
    np.testing.assert_allclose(arr.draws.to_numpy(), col.draws.to_numpy())
    # predictions for new rows scale with their exposure
    two = d.iloc[:2].assign(expo=[1.0, 2.0], x=[0.3, 0.3])
    rate = col.predict(two)
    assert rate[1] == pytest.approx(2 * rate[0])
    with pytest.raises(MethodIncompatibility, match="passed as an array"):
        arr.predict(two)


# ---------------------------------------------------------------------
# formula spellings
# ---------------------------------------------------------------------


def test_logical_outcome_is_one_indicator(df):
    ref = sp.logit("big ~ x", df.assign(big=(df["yc"] > 0.5).astype(int)))
    for formula in ("(yc > 0.5) ~ x", "I(yc > 0.5) ~ x"):
        np.testing.assert_allclose(sp.logit(formula, df).params, ref.params)
        np.testing.assert_allclose(sp.probit(formula, df).nobs, 240)
    lpm = sp.regress("(yc > 0.5) ~ x", df)
    assert 0 < lpm.params["Intercept"] < 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = sp.bayes_regress(
            "(yc > 0.5) ~ x", df, model="logit", draws=300, burnin=200, seed=1
        )
    assert b.n_obs == 240


def test_factor_wrapped_outcome_of_categorical_models(df):
    plain = sp.ologit("o ~ x", df)
    for lhs in ("factor(o)", "C(o)", "as.factor(o)", "ordered(o)"):
        np.testing.assert_allclose(sp.ologit(f"{lhs} ~ x", df).params, plain.params)
    np.testing.assert_allclose(
        sp.mlogit("factor(o) ~ x", df).params, sp.mlogit("o ~ x", df).params
    )
    np.testing.assert_allclose(
        sp.oprobit("factor(o) ~ x + C(g)", df).params,
        sp.oprobit("o ~ x + C(g)", df).params,
    )


def test_dot_stands_for_every_other_column(df):
    d = df[["yc", "x", "z", "a"]]
    np.testing.assert_allclose(
        sp.regress("yc ~ .", d).params, sp.regress("yc ~ x + z + a", d).params
    )
    np.testing.assert_allclose(
        sp.regress("yc ~ . - a", d).params, sp.regress("yc ~ x + z", d).params
    )
    odd = d.rename(columns={"z": "my z"})
    assert 'Q("my z")' in sp.regress("yc ~ .", odd).params.index
    from statspai.core.utils import expand_dot

    assert expand_dot("yc ~ x + 0.5", d) == "yc ~ x + 0.5"
    assert expand_dot("yc ~ np.log(x)", d) == "yc ~ np.log(x)"
    assert expand_dot("yc ~ x | z", d) == "yc ~ x | z"
    with pytest.raises(MethodIncompatibility, match="there is none"):
        sp.regress("yc ~ .", d[["yc"]])


# ---------------------------------------------------------------------
# collinear regressors and separation
# ---------------------------------------------------------------------

_FITTERS = {
    "logit": lambda f, d: sp.logit(f"y ~ {f}", d),
    "probit": lambda f, d: sp.probit(f"y ~ {f}", d),
    "cloglog": lambda f, d: sp.cloglog(f"y ~ {f}", d),
    "glm_binomial": lambda f, d: sp.glm(f"y ~ {f}", d, family="binomial"),
    "glm_poisson": lambda f, d: sp.glm(f"cnt ~ {f}", d, family="poisson"),
    "poisson": lambda f, d: sp.poisson(f"cnt ~ {f}", d),
    "nbreg": lambda f, d: sp.nbreg(f"cnt ~ {f}", d),
    "ologit": lambda f, d: sp.ologit(f"o ~ {f}", d),
    "oprobit": lambda f, d: sp.oprobit(f"o ~ {f}", d),
    "mlogit": lambda f, d: sp.mlogit(f"o ~ {f}", d),
    "tobit": lambda f, d: sp.tobit(d, formula=f"yc ~ {f}", ll=-1.0),
    "qreg": lambda f, d: sp.qreg(d, formula=f"yc ~ {f}"),
}


@pytest.mark.parametrize("name", sorted(_FITTERS))
@pytest.mark.parametrize(
    "full, reduced, dropped",
    [
        ("x + a + b + c", "x + a + b", "c"),
        ("x + x2 + a", "x + a", "x2"),
        ("x + a + s", "x + a", "s"),
    ],
)
def test_dependent_regressor_is_omitted_not_estimated(df, name, full, reduced, dropped):
    """The dummy-variable trap, a duplicated column and an exact linear
    combination: each estimator must return the fit without the dependent
    regressor and say which one it omitted. Before this the same calls
    returned coefficients of order 1e13, standard errors of zero or 1e6,
    often without a warning."""
    d = df.assign(x2=2 * df["x"], s=df["x"] + df["a"])
    fitter = _FITTERS[name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = fitter(full, d)
    notes = [
        str(w.message)
        for w in caught
        if "omitted because of collinearity" in str(w.message)
    ]
    assert notes and f"note: {dropped} omitted" in notes[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = fitter(reduced, d)
    assert len(got.params) == len(want.params)
    np.testing.assert_allclose(
        np.asarray(got.params, float),
        np.asarray(want.params, float),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(got.std_errors, float), np.asarray(want.std_errors, float), rtol=1e-5
    )
    assert np.all(np.isfinite(np.asarray(got.std_errors, float)))


def test_full_rank_designs_are_left_alone(df):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fit = sp.logit("y ~ x + a + b", df)
    assert fit.model_info["omitted"] == []
    from statspai.core._collinear import independent_columns

    # nearly, not exactly, dependent: kept
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(50), rng.normal(size=50)])
    X = np.column_stack([X, X[:, 1] + 1e-6 * rng.normal(size=50)])
    keep, omitted = independent_columns(X, ["Intercept", "u", "v"])
    assert keep == [0, 1, 2] and omitted == []
    # a large-mean, small-spread regressor is not "collinear with the constant"
    year = 2000.0 + np.arange(50) % 10
    keep, omitted = independent_columns(
        np.column_stack([np.ones(50), year]), ["Intercept", "year"]
    )
    assert keep == [0, 1]
    # the constant is never the one omitted, wherever it is written
    keep, omitted = independent_columns(
        np.column_stack([np.full(50, 2.0), np.ones(50)]), ["two", "Intercept"]
    )
    assert keep == [1] and omitted[0]["variable"] == "two"


def test_survey_regression_omits_dependent_regressors(df):
    d = df.assign(w=np.linspace(0.5, 2.0, len(df)))
    design = sp.svydesign(d, weights="w")
    with pytest.warns(UserWarning, match="c omitted because of collinearity"):
        got = design.glm("yc ~ x + a + b + c")
    want = design.glm("yc ~ x + a + b")
    np.testing.assert_allclose(got.estimate, want.estimate, rtol=1e-9)
    np.testing.assert_allclose(got.std_error, want.std_error, rtol=1e-9)
    assert np.all(np.isfinite(got.std_error))


def test_separation_is_announced_by_every_binary_fitter():
    """With 60 observations one point close to the separating threshold
    used to keep the warning silent while the slope ran past 1,000."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    d = pd.DataFrame({"x": x, "y": (x > 0).astype(int)})
    for fitter in (
        lambda: sp.logit("y ~ x", d),
        lambda: sp.probit("y ~ x", d),
        lambda: sp.glm("y ~ x", d, family="binomial"),
    ):
        with pytest.warns(ConvergenceWarning, match="Complete separation"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("always", ConvergenceWarning)
                fitter()


def test_no_separation_warning_on_an_ordinary_fit(df):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        sp.logit("y ~ x + a", df)
        sp.glm("y ~ x + a", df, family="binomial")
        sp.probit("y ~ x + a", df)


# ---------------------------------------------------------------------
# glm spellings
# ---------------------------------------------------------------------


def test_grouped_binomial_checks(df):
    d = df.assign(trials=6, succ=np.minimum(df["cnt"], 6))
    with pytest.raises(MethodIncompatibility, match="family='binomial'"):
        sp.glm("cbind(succ, trials - succ) ~ x", d, family="poisson")
    with pytest.raises(MethodIncompatibility, match="number of trials is the weight"):
        sp.glm("cbind(succ, trials - succ) ~ x", d, family="binomial", weights="trials")
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.glm("cbind(succ, 2 - succ) ~ x", d, family="binomial")
    with pytest.raises(MethodIncompatibility, match="Could not evaluate"):
        sp.glm("cbind(succ, nope) ~ x", d, family="binomial")
    # expanding each group into its trials gives the same fit
    long = pd.DataFrame(
        {
            "x": np.repeat(d["x"].to_numpy(), 6),
            "hit": np.concatenate(
                [np.r_[np.ones(int(s)), np.zeros(6 - int(s))] for s in d["succ"]]
            ),
        }
    )
    grouped = sp.glm("cbind(succ, trials - succ) ~ x", d, family="binomial")
    single = sp.glm("hit ~ x", long, family="binomial")
    np.testing.assert_allclose(grouped.params, single.params, rtol=1e-8)
    np.testing.assert_allclose(grouped.std_errors, single.std_errors, rtol=1e-6)


def test_robit_and_quasi_argument_checks(df):
    with pytest.raises(MethodIncompatibility, match="family='binomial'"):
        sp.glm("cnt ~ x", df, family="poisson", link="robit(4)")
    with pytest.raises(MethodIncompatibility, match="more than 2 degrees"):
        sp.glm("y ~ x", df, family="binomial", link="robit(2,unit)")
    default = sp.glm("y ~ x", df, family="binomial", link="robit")
    seven = sp.glm("y ~ x", df, family="binomial", link="robit(7)")
    np.testing.assert_allclose(default.params, seven.params)
    # a robust covariance replaces the quasi scaling rather than stacking on it
    robust = sp.glm("cnt ~ x", df, family="quasipoisson", robust="hc1")
    plain = sp.glm("cnt ~ x", df, family="poisson", robust="hc1")
    np.testing.assert_allclose(robust.std_errors, plain.std_errors)


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------


def test_binned_residuals_inputs_and_plot(df):
    fit = sp.logit("y ~ x + a", df)
    out = sp.binned_residuals(fit)
    assert len(out) == int(np.sqrt(240)) and out["n"].sum() == 240
    assert 0 <= out.attrs["share_outside"] <= 0.4
    assert sp.binned_residuals_plot(fit) is not None
    assert sp.binned_residuals_plot(fit, by=df["x"], xlabel="x") is not None
    count = sp.binned_residuals(sp.poisson("cnt ~ x", df), n_bins=8)
    assert len(count) == 8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = sp.bayes_regress("y ~ x", df, model="logit", draws=300, burnin=200, seed=1)
    assert len(sp.binned_residuals(b, n_bins=6)) == 6
    with pytest.raises(MethodIncompatibility):
        sp.binned_residuals(np.arange(10.0), np.arange(9.0))
    with pytest.raises(MethodIncompatibility):
        sp.binned_residuals(np.arange(10.0), np.arange(10.0), n_bins=1)
    with pytest.raises(DataInsufficient):
        sp.binned_residuals([1.0, 2.0], [0.1, 0.2])
    with pytest.raises(MethodIncompatibility):
        sp.binned_residuals(object())
    with pytest.raises(MethodIncompatibility):
        sp.binned_residuals(np.arange(10.0), np.arange(10.0), by=np.arange(10.0))


def test_a_misspecified_mean_shows_in_the_binned_residuals():
    rng = np.random.default_rng(2)
    # probabilities stay between 0.18 and 0.82: in a bin where the outcome
    # hardly varies the band 2 sd / sqrt(n) collapses and flags nothing real
    x = rng.uniform(-2.5, 2.5, 6000)
    d = pd.DataFrame({"x": x})
    d["y"] = (rng.uniform(size=6000) < 1 / (1 + np.exp(-(0.5 * x**2 - 1.5)))).astype(
        int
    )
    wrong = sp.binned_residuals(sp.logit("y ~ x", d), by=d["x"], n_bins=30)
    right = sp.binned_residuals(sp.logit("y ~ x + I(x**2)", d), by=d["x"], n_bins=30)
    assert wrong.attrs["share_outside"] > 0.5
    assert right.attrs["share_outside"] < 0.2


def test_standardize_frame_rules(df):
    out = sp.standardize(df)
    assert out["x"].std() == pytest.approx(0.5)
    assert abs(out["a"].mean()) < 1e-12 and out["a"].std() == pytest.approx(
        df["a"].std()
    )
    assert out["g"].equals(df["g"])
    rec = out.attrs["standardize"]
    np.testing.assert_allclose(
        (df["x"] - rec["x"]["center"]) / rec["x"]["scale"], out["x"]
    )
    kept = sp.standardize(df, exclude=["x"])
    assert kept["x"].equals(df["x"])
    z = sp.standardize(df, columns=["x"], divisor=1.0)
    assert z["x"].std() == pytest.approx(1.0) and z["z"].equals(df["z"])
    only = sp.standardize(df, formula="yc ~ x + a")
    assert only["yc"].equals(df["yc"]) and only["z"].equals(df["z"])
    const = sp.standardize(df.assign(k=3.0))
    assert (const["k"] == 3.0).all()
    assert isinstance(sp.standardize(df["x"].to_numpy()), np.ndarray)
    with pytest.raises(MethodIncompatibility):
        sp.standardize(df, binary="half")
    with pytest.raises(MethodIncompatibility):
        sp.standardize(df, columns=["g"])
    with pytest.raises(MethodIncompatibility):
        sp.standardize(df, columns=["nope"])
    with pytest.raises(MethodIncompatibility):
        sp.standardize(df, columns=["x"], formula="yc ~ x")
    with pytest.raises(MethodIncompatibility):
        sp.standardize(df, divisor=0)


def test_invlogit():
    assert sp.invlogit(0.0) == 0.5
    np.testing.assert_allclose(sp.invlogit(np.array([-800.0, 800.0])), [0.0, 1.0])
    s = sp.invlogit(pd.Series([0.0, np.log(3.0)]))
    assert isinstance(s, pd.Series) and s.iloc[1] == pytest.approx(0.75)


def test_retrodesign_properties_and_checks():
    low = sp.retrodesign(0.1, 1.0)
    high = sp.retrodesign(4.0, 1.0)
    assert low.power < 0.06 and low.type_s > 0.3 and low.type_m > 10
    assert high.power > 0.97 and high.type_s < 1e-8
    assert high.type_m == pytest.approx(1.0, abs=0.02)
    # the sign of the hypothesised effect does not matter
    assert sp.retrodesign(-0.1, 1.0).type_m == pytest.approx(low.type_m)
    # exaggeration falls and power rises with the effect
    grid = sp.retrodesign(np.linspace(0.2, 4, 20), 1.0).table
    assert (
        grid["type_m"].is_monotonic_decreasing and grid["power"].is_monotonic_increasing
    )
    assert np.all(grid["type_m"] >= 1.0)
    assert "type_s" in low.summary()
    json.dumps(low.to_dict())
    for bad in (
        dict(effect=0.0, se=1.0),
        dict(effect=1.0, se=0.0),
        dict(effect=1.0, se=1.0, alpha=1.5),
        dict(effect=1.0, se=1.0, dof=1.0),
        dict(effect=1.0, se=1.0, dof=5, method="other"),
        dict(effect=[1.0, 2.0], se=[1.0, 2.0, 3.0]),
    ):
        with pytest.raises(MethodIncompatibility):
            sp.retrodesign(**bad)


def test_poststratify_matches_the_weighted_average(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            "y ~ C(g)", df, model="logit", draws=1500, burnin=500, seed=1
        )
    cells = pd.DataFrame(
        {"g": ["p", "q", "r"], "N": [100, 300, 600], "region": ["n", "n", "s"]}
    )
    out = sp.poststratify(fit, cells)
    draws = fit.posterior_epred(cells) @ np.array([0.1, 0.3, 0.6])
    assert out.estimate == pytest.approx(draws.mean())
    assert out.sd == pytest.approx(draws.std(ddof=1))
    assert out.lower < out.estimate < out.upper
    # shares instead of counts: same answer
    assert sp.poststratify(
        fit, cells.assign(N=[0.1, 0.3, 0.6])
    ).estimate == pytest.approx(out.estimate)
    by = sp.poststratify(fit, cells, by="region")
    assert list(by.table.index) == ["overall", "region=n", "region=s"]
    assert by.table.loc["region=s", "estimate"] == pytest.approx(
        fit.posterior_epred(cells)[:, 2].mean()
    )
    assert by.draws.shape == (1500, 3)
    # the raw mean over-represents nobody here, but the reweighting moves it
    assert out.estimate != pytest.approx(df["y"].mean(), abs=1e-3)
    # a classical fit gives the point estimate only
    point = sp.poststratify(sp.logit("y ~ C(g)", df), cells)
    assert point.estimate == pytest.approx(out.estimate, abs=0.02) and np.isnan(
        point.sd
    )
    vec = sp.poststratify([0.2, 0.4, 0.6], cells)
    assert vec.estimate == pytest.approx(0.02 + 0.12 + 0.36)
    json.dumps(out.to_dict())
    assert "Poststratified" in out.summary()
    with pytest.raises(MethodIncompatibility, match="no column"):
        sp.poststratify(fit, cells, count="pop")
    with pytest.raises(MethodIncompatibility, match="for 2 cells"):
        sp.poststratify([0.2, 0.4], cells)
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.poststratify(fit, cells.assign(N=[-1, 2, 3]))
    with pytest.raises(MethodIncompatibility, match="by columns"):
        sp.poststratify(fit, cells, by="nope")
    with pytest.raises(MethodIncompatibility):
        sp.poststratify(fit, [1, 2, 3])


def test_shrinkage_argument_checks(df):
    with pytest.raises(MethodIncompatibility, match="belong to prior='horseshoe'"):
        sp.bayes_shrink("yc ~ x + z", df, prior="lasso", p0=1)
    with pytest.raises(MethodIncompatibility, match="not both"):
        sp.bayes_shrink("yc ~ x + z", df, prior="horseshoe", p0=1, global_scale=0.1)
    with pytest.raises(MethodIncompatibility, match="p0 must be between"):
        sp.bayes_shrink("yc ~ x + z", df, prior="horseshoe", p0=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hs = sp.bayes_shrink(
            "yc ~ x + z + a", df, prior="hs", draws=600, burnin=300, seed=1
        )
    assert hs.model == "horseshoe" and "tau" in hs.draws.columns
    assert hs.table["shrinkage"].notna().sum() == 3
    # the predictive side reads sigma2 from its own column, not the last one
    ll = hs.log_lik()
    d = hs.draws.to_numpy()
    X = np.column_stack([np.ones(len(df)), df[["x", "z", "a"]].to_numpy()])
    from scipy import stats

    want = stats.norm.logpdf(
        df["yc"].to_numpy(),
        d[:, :4] @ X.T,
        np.sqrt(hs.draws["sigma2"].to_numpy())[:, None],
    )
    np.testing.assert_allclose(ll, want, rtol=1e-10)
    assert np.isfinite(sp.loo(hs).elpd)


# ---------------------------------------------------------------------
# second round: default prior, grouped outcomes, ordered logit, slab
# ---------------------------------------------------------------------


def test_default_prior_change_is_announced_only_where_it_applies(df):
    with pytest.warns(DeprecationWarning, match="will change in StatsPAI 1.40"):
        sp.bayes_regress("yc ~ x", df, draws=300, seed=1)
    with pytest.warns(DeprecationWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StatsPAIWarning)
            warnings.simplefilter("always", DeprecationWarning)
            sp.bayes_regress("y ~ x", df, model="logit", draws=300, burnin=200, seed=1)
    quiet = [
        dict(prior="vague"),
        dict(prior="weakly_informative"),
        dict(prior_var=50.0),
        dict(prior_mean=[0.5, 0.0]),
        dict(model="conjugate"),
        dict(model="t"),
        dict(model="quantile"),
    ]
    for kw in quiet:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter("ignore", StatsPAIWarning)
            sp.bayes_regress("yc ~ x", df, draws=200, burnin=100, seed=1, **kw)
    # naming the prior reproduces the unnamed fit exactly, and a refit for
    # cross-validation does not repeat the announcement
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.bayes_regress("yc ~ x", df, draws=300, seed=1)
    b = sp.bayes_regress("yc ~ x", df, draws=300, seed=1, prior="vague")
    np.testing.assert_array_equal(a.draws.to_numpy(), b.draws.to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sp.kfold(a, k=3, seed=1)


def test_grouped_binomial_arguments(df):
    d = df.assign(m=6, hits=np.minimum(df["cnt"], 6))
    kw = dict(model="logit", prior="weakly_informative", draws=300, burnin=200, seed=1)
    with pytest.raises(MethodIncompatibility, match="trials= is for model='logit'"):
        sp.bayes_regress("hits ~ x", d, model="poisson", trials="m", prior="vague")
    with pytest.raises(MethodIncompatibility, match="must name a column"):
        sp.bayes_regress("hits ~ x", d, trials="nope", **kw)
    with pytest.raises(MethodIncompatibility, match="between 0 and trials"):
        sp.bayes_regress("hits ~ x", d.assign(m=2), trials="m", **kw)
    with pytest.raises(MethodIncompatibility, match="do not pass trials"):
        sp.bayes_regress("cbind(hits, m - hits) ~ x", d, trials="m", **kw)
    with pytest.raises(MethodIncompatibility, match="model='logit'"):
        sp.bayes_regress("cbind(hits, m - hits) ~ x", d, model="probit", prior="vague")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress("cbind(hits, m - hits) ~ x", d, **kw)
        mle = sp.glm("cbind(hits, m - hits) ~ x", d, family="binomial")
    assert fit.formula == "cbind(hits, m - hits) ~ x"
    assert fit.params["x"] == pytest.approx(
        mle.params["x"], abs=3 * mle.std_errors["x"]
    )
    assert fit.posterior_predict(d.iloc[:4], seed=1).shape == (300, 4)
    assert np.isfinite(sp.loo(fit).elpd)
    assert np.isfinite(sp.kfold(fit, k=3, seed=1).elpd)
    assert len(sp.binned_residuals(fit, n_bins=6)) == 6
    with pytest.raises(MethodIncompatibility, match="grouped binomial"):
        sp.bayes_r2(fit)
    with pytest.raises(MethodIncompatibility, match="grouped binomial"):
        sp.loo_r2(fit)


def test_ordered_logit_surface(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            "o ~ x + a", df, model="ologit", draws=3000, burnin=500, seed=1
        )
        same = sp.bayes_regress(
            "o ~ x + a", df, model="polr", draws=3000, burnin=500, seed=1
        )
        mle = sp.ologit("o ~ x + a", df)
    np.testing.assert_array_equal(fit.draws.to_numpy(), same.draws.to_numpy())
    assert list(fit.params.index) == ["x", "a", "cut1", "cut2"]
    # a diffuse prior and 240 observations: close to maximum likelihood
    for mine, theirs in zip(["x", "a", "cut1", "cut2"], ["x", "a", "/cut1", "/cut2"]):
        assert fit.params[mine] == pytest.approx(
            mle.params[theirs], abs=0.35 * mle.std_errors[theirs]
        )
    probs = fit.predict(df.iloc[:5], what="probabilities")
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert set(np.unique(fit.posterior_predict(draws=100, seed=1))) <= {0.0, 1.0, 2.0}
    assert np.isfinite(sp.loo(fit).elpd)
    with pytest.raises(MethodIncompatibility):
        sp.bayes_r2(fit)
    with pytest.raises(MethodIncompatibility, match="free cutpoints"):
        sp.bayes_regress("o ~ x - 1", df, model="ologit")


def test_metropolis_models_report_both_acceptance_rates(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            "cnt ~ x + a",
            df,
            model="poisson",
            prior="weakly_informative",
            draws=4000,
            burnin=500,
            seed=1,
        )
    assert 0.1 < fit.acceptance_rate < 0.7
    assert fit._extras["independence_accept"] > 0.5
    # nearly independent draws: far more than a random walk alone delivers
    assert fit.table["ess"].min() > 1200


def test_slab_arguments(df):
    with pytest.raises(MethodIncompatibility, match="belong to prior='horseshoe'"):
        sp.bayes_shrink("yc ~ x + z", df, prior="lasso", slab_scale=2.0)
    with pytest.raises(MethodIncompatibility, match="must be positive"):
        sp.bayes_shrink("yc ~ x + z", df, prior="horseshoe", slab_scale=-1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_shrink(
            "yc ~ x + z + a",
            df,
            prior="horseshoe",
            slab_scale=2.5,
            draws=800,
            burnin=500,
            seed=1,
        )
    assert list(fit.draws.columns)[-3:] == ["sigma2", "tau", "slab"]
    assert fit.model_info["slab_scale"] == 2.5
    assert len(fit.model_info["metropolis_steps"]) == 4
    assert np.isfinite(sp.loo(fit).elpd)


def test_model_band_does_not_collapse_where_the_outcome_is_constant():
    rng = np.random.default_rng(2)
    x = rng.uniform(-3, 3, 4000)
    d = pd.DataFrame({"x": x})
    d["y"] = (rng.uniform(size=4000) < 1 / (1 + np.exp(-(x**2 - 2)))).astype(int)
    fit = sp.logit("y ~ x + I(x**2)", d)
    arm = sp.binned_residuals(fit, by=d["x"], n_bins=30)
    model = sp.binned_residuals(fit, by=d["x"], n_bins=30, band="model")
    # the model is right, yet the empirical band flags the bins at the ends
    assert arm.attrs["share_outside"] > model.attrs["share_outside"]
    assert model.attrs["share_outside"] <= 0.15
    np.testing.assert_allclose(arm["ybar"], model["ybar"])
    assert sp.binned_residuals_plot(fit, band="model") is not None
    with pytest.raises(MethodIncompatibility, match="0 / 1 outcome"):
        counts = d.assign(c=rng.poisson(3.0, size=4000))
        sp.binned_residuals(sp.poisson("c ~ x", counts), band="model")
    with pytest.raises(MethodIncompatibility, match="needs the fitted model"):
        sp.binned_residuals(x, d["y"] - 0.5, band="model")
    with pytest.raises(MethodIncompatibility):
        sp.binned_residuals(fit, band="other")


def test_nbreg_counts_only_the_fixed_effects_it_estimates(df):
    d = df.assign(grp=np.arange(len(df)) % 4)
    d["dup"] = (d["grp"] == 3) * 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = sp.nbreg("cnt ~ x | grp", d)
        over = sp.nbreg("cnt ~ x + dup | grp", d)
    assert base.model_info["n_fe_params"] == 3
    # 'dup' takes the place of one indicator, which is then omitted
    assert over.model_info["n_fe_params"] == 2
    assert len(over.model_info["omitted"]) == 1
