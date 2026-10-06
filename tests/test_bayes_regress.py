"""API, edge cases and failure modes of the NumPy Bayesian toolkit:
``sp.bayes_regress``, ``sp.bma``, ``sp.bayes_factor``, ``sp.savage_dickey``,
``sp.bayes_bootstrap`` and the MCMC diagnostics.

Numerical correctness lives in ``tests/reference_parity/``
(``test_bayes_regress_exact_posterior.py`` and ``test_bayes_mcmc_parity.py``).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    d = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.uniform(size=n)})
    eta = 0.5 + 0.8 * d["x1"] - 1.2 * d["x2"]
    d["y"] = eta + rng.normal(size=n)
    d["d"] = (eta + rng.normal(size=n) > 0).astype(int)
    d["c"] = rng.poisson(np.exp(eta))
    d["o"] = np.digitize(eta + rng.normal(size=n), [-0.5, 0.3, 1.2])
    d["grade"] = pd.Categorical.from_codes(
        d["o"], ["low", "mid", "high", "top"], ordered=True
    )
    return d


FAST = dict(draws=1500, burnin=300, seed=1)


# -- results ---------------------------------------------------------------


def test_result_surface(df):
    fit = sp.bayes_regress("y ~ x1 + x2", df, **FAST)
    assert list(fit.params.index) == ["Intercept", "x1", "x2", "sigma2"]
    assert fit.draws.shape == (1500, 4)
    assert fit.n_obs == 400 and fit.chains == 1
    assert fit.conf_int().shape == (4, 2)
    hpd = fit.conf_int(kind="hpd", level=0.9)
    assert (hpd["upper"] > hpd["lower"]).all()
    assert fit.prob("x1 > 0") == 1.0
    assert 0.0 <= fit.prob("x1 > -x2") <= 1.0
    assert "Bayesian normal regression" in fit.summary()
    assert set(fit.tidy().columns) >= {
        "term",
        "estimate",
        "std_error",
        "lower",
        "upper",
    }
    json.dumps(fit.to_dict())
    assert fit.cite() == "gelfand1990sampling"
    assert isinstance(fit.log_marginal_likelihood(), float)
    diag = fit.diagnostics()
    assert set(diag) == {"geweke", "heidel", "raftery"}
    # 1,500 draws are too few for the default Raftery-Lewis accuracy
    assert isinstance(diag["raftery"], str) and "3746" in diag["raftery"]


def test_seed_reproduces_and_matters(df):
    a = sp.bayes_regress("d ~ x1 + x2", df, model="probit", **FAST)
    b = sp.bayes_regress("d ~ x1 + x2", df, model="probit", **FAST)
    c = sp.bayes_regress(
        "d ~ x1 + x2", df, model="probit", draws=1500, burnin=300, seed=2
    )
    assert a.draws.equals(b.draws)
    assert not a.draws.equals(c.draws)


def test_thinning_and_chains(df):
    fit = sp.bayes_regress(
        "d ~ x1 + x2",
        df,
        model="logit",
        draws=800,
        burnin=200,
        thin=3,
        chains=3,
        seed=4,
    )
    assert fit.draws.shape[0] == 2400 and fit.n_draws == 2400
    assert [len(c) for c in fit.chain_list()] == [800, 800, 800]
    gr = fit.diagnostics()["gelman_rubin"]
    assert gr.settings["chains"] == 3
    assert (gr.table["psrf"] < 1.1).all()
    assert 0.1 < fit.acceptance_rate < 0.6


def test_vague_prior_posterior_is_close_to_maximum_likelihood(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model, outcome, mle in (
            ("logit", "d", sp.logit),
            ("probit", "d", sp.probit),
            ("poisson", "c", sp.poisson),
        ):
            fit = sp.bayes_regress(
                f"{outcome} ~ x1 + x2",
                df,
                model=model,
                prior_var=1e4,
                draws=6000,
                burnin=1000,
                seed=3,
            )
            ref = mle(f"{outcome} ~ x1 + x2", df)
            names = ["Intercept", "x1", "x2"]
            const = next(k for k in ref.params.index if k not in ("x1", "x2"))
            theirs = [const, "x1", "x2"]
            est = np.asarray([ref.params[k] for k in theirs])
            se = np.asarray([ref.std_errors[k] for k in theirs])
            # the posterior mean sits within a fraction of a standard error
            # of the MLE, and the posterior sd is the asymptotic SE
            assert np.max(np.abs(fit.params[names].to_numpy() - est) / se) < 0.25
            assert np.max(np.abs(fit.std_errors[names].to_numpy() / se - 1)) < 0.12


def test_predict(df):
    fit = sp.bayes_regress("d ~ x1 + x2", df, model="probit", **FAST)
    p = fit.predict()
    assert p.shape == (400,) and ((p > 0) & (p < 1)).all()
    new = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.5, 0.5]})
    iv = fit.predict(new, what="interval")
    assert list(iv.columns) == ["mean", "lower", "upper"]
    assert (iv["lower"] < iv["mean"]).all() and (iv["mean"] < iv["upper"]).all()
    assert iv["mean"].iloc[1] > iv["mean"].iloc[0]
    assert fit.predict(new, what="draws").shape == (1500, 2)
    assert fit.predict(new, what="linear").shape == (2,)
    with pytest.raises(sp.MethodIncompatibility, match="oprobit"):
        fit.predict(new, what="probabilities")


def test_ordered_probit_levels_and_probabilities(df):
    num = sp.bayes_regress("o ~ x1 + x2", df, model="oprobit", **FAST)
    cat = sp.bayes_regress("grade ~ x1 + x2", df, model="oprobit", **FAST)
    assert list(num.params.index) == ["x1", "x2", "cut1", "cut2", "cut3"]
    assert num.draws.equals(cat.draws)  # same codes, same seed
    assert cat.model_info["levels"] == ["low", "mid", "high", "top"]
    assert (np.diff(num.draws[["cut1", "cut2", "cut3"]].to_numpy(), axis=1) > 0).all()
    probs = cat.predict(what="probabilities")
    assert list(probs.columns) == ["low", "mid", "high", "top"]
    assert np.allclose(probs.sum(axis=1), 1.0)
    # average predicted shares track the observed shares
    obs = df["grade"].value_counts(normalize=True).reindex(probs.columns).to_numpy()
    assert np.max(np.abs(probs.mean(axis=0).to_numpy() - obs)) < 0.03
    # and the posterior agrees with the frequentist ordered probit
    ref = sp.oprobit("o ~ x1 + x2", df)
    assert abs(num.params["x1"] - ref.params["x1"]) < 0.5 * ref.std_errors["x1"]
    with pytest.raises(sp.MethodIncompatibility, match="cutpoints"):
        sp.bayes_regress("o ~ x1 + x2 - 1", df, model="oprobit", **FAST)


def test_tobit_counts_censored_rows(df):
    d = df.assign(yc=np.clip(df["y"], 0.0, 2.0))
    fit = sp.bayes_regress(
        "yc ~ x1 + x2", d, model="tobit", lower=0.0, upper=2.0, **FAST
    )
    assert fit.model_info["n_left_censored"] == int((d["yc"] <= 0).sum())
    assert fit.model_info["n_right_censored"] == int((d["yc"] >= 2).sum())
    # censoring attenuates least squares; the tobit posterior does not
    assert abs(fit.params["x1"] - 0.8) < 0.15
    with pytest.raises(sp.MethodIncompatibility, match="censoring point"):
        sp.bayes_regress("yc ~ x1", d, model="tobit", **FAST)


def test_quantile_regression_tracks_the_conditional_quantiles(df):
    lo = sp.bayes_regress("y ~ x1 + x2", df, model="quantile", quantile=0.1, **FAST)
    hi = sp.bayes_regress("y ~ x1 + x2", df, model="quantile", quantile=0.9, **FAST)
    # homoskedastic normal errors: intercepts 2 * 1.2816 apart, equal slopes
    gap = hi.params["Intercept"] - lo.params["Intercept"]
    assert gap == pytest.approx(2 * 1.2816, abs=0.45)
    assert lo.params["x1"] == pytest.approx(hi.params["x1"], abs=0.3)
    ref = sp.qreg("y ~ x1 + x2", df, quantile=0.9)
    assert hi.params["x1"] == pytest.approx(ref.params["x1"], abs=0.1)
    fixed = sp.bayes_regress("y ~ x1 + x2", df, model="quantile", scale=1.0, **FAST)
    assert "sigma" not in fixed.params.index


def test_student_t_resists_outliers():
    rng = np.random.default_rng(2)
    n = 300
    d = pd.DataFrame({"x": rng.normal(size=n)})
    d["y"] = 1 + 0.5 * d["x"] + 0.3 * rng.normal(size=n)
    top = d["x"].nlargest(6).index
    d.loc[top, "y"] -= 25.0
    normal = sp.bayes_regress("y ~ x", d, **FAST)
    robust = sp.bayes_regress("y ~ x", d, model="t", dof=3, **FAST)
    assert abs(robust.params["x"] - 0.5) < 0.1
    assert abs(normal.params["x"] - 0.5) > 0.4


def test_model_aliases(df):
    a = sp.bayes_regress("y ~ x1", df, model="gaussian", **FAST)
    b = sp.bayes_regress("c ~ x1", df, model="nbreg", **FAST)
    assert a.model == "normal" and b.model == "negbin"
    assert "alpha" in b.params.index


# -- warnings --------------------------------------------------------------


def test_default_prior_that_is_not_vague_is_flagged(df):
    d = df.assign(big=df["x1"] / 500.0)  # coefficient of 400 against prior sd 31.6
    with pytest.warns(
        sp.exceptions.StatsPAIWarning, match="default prior is not vague"
    ):
        fit = sp.bayes_regress("y ~ big + x2", d, **FAST)
    assert any("big" in w for w in fit.diagnostics_info["warnings"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = sp.bayes_regress("y ~ big + x2", d, prior_var=1e7, **FAST)
    assert ok.params["big"] == pytest.approx(400.0, rel=0.2)


def test_slow_mixing_is_flagged(df):
    with pytest.warns(
        sp.exceptions.ConvergenceWarning, match="converged or mixes slowly"
    ):
        sp.bayes_regress(
            "d ~ x1 + x2", df, model="logit", draws=300, burnin=0, tune=0.02, seed=1
        )


# -- refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(formula="y ~ x1", model="spline"), "Unknown model"),
        (dict(formula="y ~ x1", model="logit"), "0/1 outcome"),
        (dict(formula="y ~ x1", model="poisson"), "non-negative integer"),
        (dict(formula="d ~ x1", model="oprobit"), "at least three"),
        (dict(formula="y ~ x1", draws=0), "draws must be"),
        (dict(formula="y ~ x1", thin=0), "thin must be"),
        (dict(formula="y ~ x1", prior_var=-1.0), "positive definite"),
        (
            dict(formula="y ~ x1", prior_mean=[0.0, 0.0, 0.0]),
            "one entry per coefficient",
        ),
        (dict(formula="y ~ x1", sigma2_prior=(0.0, 1.0)), "positive numbers"),
        (dict(formula="y ~ x1", model="quantile", quantile=1.2), "quantile must be"),
        (dict(formula="y ~ x1", level=1.5), "level must be"),
        (dict(formula="y x1"), "formula must look like"),
    ],
)
def test_bad_arguments_are_refused(df, kwargs, match):
    kwargs = {"draws": 200, "burnin": 50, **kwargs}
    formula = kwargs.pop("formula")
    with pytest.raises(sp.MethodIncompatibility, match=match):
        sp.bayes_regress(formula, df, **kwargs)


def test_degenerate_data_is_refused(df):
    d = df.assign(x3=2 * df["x1"])
    with pytest.raises(sp.MethodIncompatibility, match="collinear"):
        sp.bayes_regress("y ~ x1 + x3", d, **FAST)
    with pytest.raises(sp.DataInsufficient, match="constant"):
        sp.bayes_regress("one ~ x1", df.assign(one=1), model="probit", **FAST)
    with pytest.raises(sp.DataInsufficient, match="observations"):
        sp.bayes_regress("y ~ x1 + x2", df.head(3), **FAST)


def test_perfect_separation_is_handled_by_the_prior(df):
    d = df.assign(sep=(df["x1"] > 0).astype(int))
    # the likelihood has no maximum, the posterior does
    fit = sp.bayes_regress("sep ~ x1", d, model="logit", prior_var=25.0, **FAST)
    assert fit.params["x1"] > 3


def test_conjugate_has_no_chain_and_only_the_exact_marginal_likelihood(df):
    fit = sp.bayes_regress("y ~ x1 + x2", df, model="conjugate", draws=500, seed=1)
    assert (fit.table["mcse"] == 0).all()
    with pytest.raises(sp.MethodIncompatibility, match="independent draws"):
        fit.diagnostics()
    with pytest.raises(sp.MethodIncompatibility, match="closed-form"):
        fit.log_marginal_likelihood("chib")
    other = sp.bayes_regress("y ~ x1 + x2", df, **FAST)
    with pytest.raises(sp.MethodIncompatibility, match="Only model='conjugate'"):
        other.log_marginal_likelihood("exact")
    with pytest.raises(sp.MethodIncompatibility, match="normal and probit"):
        sp.bayes_regress("d ~ x1", df, model="logit", **FAST).log_marginal_likelihood(
            "chib"
        )


# -- model comparison ----------------------------------------------------------


def test_bayes_factor(df):
    small = sp.bayes_regress("y ~ x1", df, model="conjugate", seed=1)
    large = sp.bayes_regress("y ~ x1 + x2", df, model="conjugate", seed=1)
    out = sp.bayes_factor(small, large, names=["without x2", "with x2"])
    assert out.log_bf < -10  # x2 matters
    assert "very strong evidence for with x2" == out.evidence
    assert out.table["post_prob"].sum() == pytest.approx(1.0)
    assert out.table.loc["with x2", "post_prob"] > 0.999
    num = sp.bayes_factor(-100.0, -103.0, -101.0, prior_probs=[1, 1, 2])
    assert num.table["post_prob"].idxmax() == "model 1"
    assert num.log_bf == pytest.approx(3.0)
    with pytest.raises(sp.MethodIncompatibility, match="at least two"):
        sp.bayes_factor(small)
    half = sp.bayes_regress("y ~ x1", df.head(200), model="conjugate", seed=1)
    with pytest.raises(
        sp.MethodIncompatibility, match="different numbers of observations"
    ):
        sp.bayes_factor(small, half)


def test_savage_dickey_estimators_and_refusals(df):
    fit = sp.bayes_regress("d ~ x1 + x2", df, model="logit", **FAST)
    far = sp.savage_dickey(fit, "x1")  # zero is far in the tail
    assert far["estimator"].startswith("normal approximation") and far["bf01"] < 1e-6
    near = sp.savage_dickey(fit, "x1", value=float(fit.params["x1"]))
    assert near["estimator"] == "kernel" and near["bf01"] > 1
    with pytest.raises(sp.MethodIncompatibility, match="not a coefficient"):
        sp.savage_dickey(fit, "sigma2")
    op = sp.bayes_regress("o ~ x1", df, model="oprobit", **FAST)
    with pytest.raises(sp.MethodIncompatibility, match="oprobit"):
        sp.savage_dickey(op, "x1")


# -- Bayesian bootstrap ----------------------------------------------------------


def test_bayes_bootstrap(df):
    fit = sp.bayes_bootstrap("y ~ x1 + x2", df, draws=1500, seed=1)
    ols = sp.regress("y ~ x1 + x2", df, robust="hc0")
    for k in ("Intercept", "x1", "x2"):
        assert fit.params[k] == pytest.approx(ols.params[k], abs=0.02)
        assert fit.std_errors[k] == pytest.approx(ols.std_errors[k], rel=0.12)
    assert fit.predict(df.head(3)).shape == (3,)
    stat = sp.bayes_bootstrap(
        data=df,
        statistic=lambda d, w: {
            "mean": w @ d["y"].to_numpy(),
            "share": w @ d["d"].to_numpy(),
        },
        draws=500,
        seed=1,
    )
    assert list(stat.params.index) == ["mean", "share"]
    with pytest.raises(sp.MethodIncompatibility, match="independent draws"):
        fit.diagnostics()
    with pytest.raises(sp.MethodIncompatibility, match="no marginal likelihood"):
        fit.log_marginal_likelihood()
    with pytest.raises(sp.MethodIncompatibility, match="not both"):
        sp.bayes_bootstrap("y ~ x1", df, statistic=lambda d, w: 0.0)
    with pytest.raises(sp.MethodIncompatibility, match="user-supplied statistic"):
        stat.predict()


# -- diagnostics ----------------------------------------------------------------


def test_diagnostics_accept_arrays_frames_and_results(df):
    fit = sp.bayes_regress("y ~ x1", df, **FAST)
    for chain in (
        fit,
        fit.draws,
        fit.draws.to_numpy(),
        fit.draws["x1"],
        fit.draws["x1"].to_numpy(),
    ):
        assert sp.mcmc_ess(chain).iloc[0] > 0
    assert list(sp.mcmc_summary(fit, hpd=0.9).columns[-2:]) == [
        "hpd_lower",
        "hpd_upper",
    ]
    assert sp.geweke_diag(fit).table.shape == (3, 2)
    assert "Geweke" in sp.geweke_diag(fit).summary()
    out = sp.gelman_rubin(fit.draws, split=True)
    assert out.settings["chains"] == 2


def test_diagnostics_detect_a_chain_that_has_not_converged():
    rng = np.random.default_rng(1)
    drift = np.linspace(3, 0, 4000) + rng.normal(size=4000) * 0.3
    assert not sp.geweke_diag(drift).passed
    assert not sp.heidel_diag(drift).passed
    stuck = [rng.normal(size=1000), 4 + rng.normal(size=1000)]
    assert sp.gelman_rubin(stuck).table["psrf"].iloc[0] > 2
    sticky = np.repeat(rng.normal(size=400), 10)  # every state held ten steps
    assert sp.mcmc_ess(sticky).iloc[0] < 0.2 * 4000
    assert not sp.raftery_diag(sticky).passed


def test_diagnostics_refuse_bad_input():
    x = np.random.default_rng(0).normal(size=500)
    bad = x.copy()
    bad[3] = np.nan
    with pytest.raises(sp.MethodIncompatibility, match="missing or infinite"):
        sp.mcmc_ess(bad)
    with pytest.raises(sp.MethodIncompatibility, match="1-D or 2-D"):
        sp.mcmc_summary(np.zeros((2, 3, 4)))
    with pytest.raises(sp.MethodIncompatibility, match="frac1"):
        sp.geweke_diag(x, frac1=0.7, frac2=0.6)
    with pytest.raises(sp.MethodIncompatibility, match="at least two chains"):
        sp.gelman_rubin(x)
    with pytest.raises(sp.MethodIncompatibility, match="same length"):
        sp.gelman_rubin([x, x[:100]])
    with pytest.raises(sp.MethodIncompatibility, match="prob must be"):
        sp.hpd_interval(x, prob=1.0)
    const = np.ones(200)
    assert sp.mcmc_ess(const).iloc[0] == 0.0


# -- model averaging ---------------------------------------------------------------


def test_bma_surface_and_refusals(df):
    d = df.assign(n1=np.random.default_rng(5).normal(size=len(df)))
    out = sp.bma("y ~ x1 + x2 + n1", d)
    assert out.table.loc["x1", "pip"] == pytest.approx(1.0)
    assert out.table.loc["n1", "pip"] < 0.3
    assert out.models["post_prob"].sum() == pytest.approx(1.0)
    assert out.predict(d.head(4)).shape == (4,)
    assert "Bayesian model averaging" in out.summary()
    json.dumps(out.to_dict())
    lg = sp.bma("d ~ x1 + x2 + n1", d, family="logit")
    assert lg.family == "binomial"
    p = lg.predict()
    assert ((p > 0) & (p < 1)).all()
    with pytest.raises(sp.MethodIncompatibility, match="family='gaussian'"):
        sp.bma("d ~ x1 + x2", d, family="binomial", method="gprior")
    with pytest.raises(sp.MethodIncompatibility, match="intercept"):
        sp.bma("y ~ x1 + x2 - 1", d)
    with pytest.raises(sp.MethodIncompatibility, match="not in the formula"):
        sp.bma("y ~ x1 + x2", d, always=["zz"])
    with pytest.raises(sp.MethodIncompatibility, match="strictly between"):
        sp.bma("y ~ x1 + x2", d, prior_inclusion=1.0)
    with pytest.raises(sp.MethodIncompatibility, match="collinear"):
        sp.bma("y ~ x1 + x2 + x3", d.assign(x3=d["x1"] - d["x2"]))
    with pytest.raises(sp.NumericalInstability, match="max_nodes"):
        sp.bma("y ~ x1 + x2 + n1", d, occam_ratio=1e9, max_nodes=3)


def test_bayes_mixed_surface_and_prior_share():
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(40), 6)
    d = pd.DataFrame({"id": g, "x": rng.normal(size=240)})
    small = 0.05 * rng.normal(size=40)[g]  # state effects with variance 0.0025
    d["y"] = 1 + 0.5 * d["x"] + small + 0.1 * rng.normal(size=240)
    d["c"] = rng.poisson(np.exp(0.2 + 0.3 * d["x"] + 10 * small))
    kw = dict(group="id", draws=1500, burnin=500, seed=1)
    # the textbook / MCMCpack prior is centred on a variance of one: the
    # wrong scale here, and the fit says so
    with pytest.warns(sp.exceptions.StatsPAIWarning, match="driving the variance"):
        loose = sp.bayes_mixed("y ~ x", d, re_prior=(3.0, 1.0), **kw)
    assert loose.model_info["re_prior_share"]["Intercept"] > 0.5
    # the default prior adds 0.02 to a sum of squares of about 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("error", sp.exceptions.StatsPAIWarning)
        fit = sp.bayes_mixed("y ~ x", d, **kw)
    assert fit.model_info["re_prior_share"]["Intercept"] < 0.25
    scaled = sp.bayes_mixed("y ~ x", d, re_prior=(3.0, 0.003), **kw)
    # two weak priors, one answer (the truth is 0.0025)
    assert scaled.params["var(Intercept)"] == pytest.approx(0.0025, abs=0.0015)
    assert fit.params["var(Intercept)"] < 0.2 * loose.params["var(Intercept)"]
    assert fit.params["var(Intercept)"] == pytest.approx(0.0025, abs=0.002)
    assert list(fit.params.index) == ["Intercept", "x", "sigma2", "var(Intercept)"]
    assert fit.random_effects.shape == (40, 2) and fit.n_groups == 40
    assert 0 < fit.model_info["icc"]["mean"] < 1
    assert "Groups: 40" in fit.summary()
    assert fit.cite() == "chib1999mcmc"
    json.dumps(fit.to_dict())
    assert fit.predict(d.head(3)).shape == (3,)
    with pytest.raises(sp.MethodIncompatibility, match="marginal likelihood"):
        fit.log_marginal_likelihood()
    pois = sp.bayes_mixed("c ~ x", d, family="poisson", **kw)
    assert "sigma2" not in pois.params.index and 0 < pois.acceptance_rate < 1
    assert pois.params["x"] == pytest.approx(0.3, abs=0.12)
    slopes = sp.bayes_mixed("y ~ x", d, random=["x"], re_prior=(4.0, 0.003), **kw)
    assert list(slopes.re_cov.columns) == ["Intercept", "x"]
    assert "cov(Intercept,x)" in slopes.params.index
    # refusals
    with pytest.raises(sp.MethodIncompatibility, match="not in data"):
        sp.bayes_mixed("y ~ x", d, group="nope")
    with pytest.raises(sp.MethodIncompatibility, match="Unknown family"):
        sp.bayes_mixed("y ~ x", d, group="id", family="gamma")
    with pytest.raises(sp.DataInsufficient, match="single observation"):
        sp.bayes_mixed("y ~ x", d.assign(u=np.arange(240)), group="u")
    with pytest.raises(sp.MethodIncompatibility, match="0/1 outcome"):
        sp.bayes_mixed("y ~ x", d, group="id", family="logit")
    with pytest.raises(sp.MethodIncompatibility, match="no random effect"):
        sp.bayes_mixed("y ~ x", d, group="id", random_intercept=False)


def _iv_data(n, pi, corr, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    v = rng.normal(size=n)
    eps = corr * v + np.sqrt(1 - corr**2) * rng.normal(size=n)
    d = pi * z + 0.3 * x + v
    return pd.DataFrame({"y": 1 + 0.5 * d + 0.2 * x + eps, "d": d, "z": z, "x": x})


def test_bayes_ivreg_agrees_with_2sls_when_the_instrument_is_strong():
    df = _iv_data(800, pi=1.0, corr=0.9, seed=7)
    fit = sp.bayes_ivreg("y ~ x + (d ~ z)", df, draws=6000, burnin=1000, seed=1)
    tsls = sp.ivreg("y ~ x + (d ~ z)", df)
    assert list(fit.params.index) == [
        "Intercept",
        "x",
        "d",
        "rho",
        "sigma2_y",
        "sigma2_d",
        "fs:Intercept",
        "fs:x",
        "fs:z",
    ]
    for k in ("Intercept", "x", "d"):
        se = float(tsls.std_errors[k])
        assert abs(fit.params[k] - float(tsls.params[k])) < 0.25 * se, k
        # the joint posterior has the 2SLS spread, not the smaller spread
        # of a control function with first-stage residuals taken as data
        assert fit.std_errors[k] == pytest.approx(se, rel=0.12), k
    ci = fit.conf_int()
    assert ci.loc["d", "lower"] < 0.5 < ci.loc["d", "upper"]
    assert ci.loc["rho", "lower"] < 0.9 < ci.loc["rho", "upper"]
    assert fit.prob("rho > 0") == 1.0  # OLS would be biased upward
    ols = sp.regress("y ~ x + d", df)
    assert float(ols.params["d"]) > ci.loc["d", "upper"]
    assert fit.model_info["first_stage_F"] > 100
    assert fit.model_info["tsls"]["d"] == pytest.approx(
        float(tsls.params["d"]), rel=1e-8
    )
    assert fit.cite() == "rossi2005bayesian"
    assert fit.predict(df.head(3)).shape == (3,)
    assert set(fit.diagnostics()) == {"geweke", "heidel", "raftery"}
    with pytest.raises(sp.MethodIncompatibility, match="marginal likelihood"):
        fit.log_marginal_likelihood()


def test_bayes_ivreg_exogenous_regressor_and_weak_instrument():
    exo = _iv_data(600, pi=1.0, corr=0.0, seed=3)
    fit = sp.bayes_ivreg("y ~ x + (d ~ z)", exo, draws=3000, burnin=500, seed=1)
    ci = fit.conf_int()
    assert ci.loc["rho", "lower"] < 0 < ci.loc["rho", "upper"]
    weak = _iv_data(300, pi=0.05, corr=0.8, seed=5)
    with pytest.warns(sp.exceptions.AssumptionWarning, match="Weak instruments"):
        w = sp.bayes_ivreg("y ~ x + (d ~ z)", weak, draws=3000, burnin=500, seed=1)
    assert w.model_info["first_stage_F"] < 10
    assert w.std_errors["d"] > 5 * fit.std_errors["d"]


def test_bayes_ivreg_refusals():
    df = _iv_data(200, pi=1.0, corr=0.5, seed=1)
    with pytest.raises(sp.MethodIncompatibility, match="exactly one"):
        sp.bayes_ivreg("y ~ x + d", df)
    with pytest.raises(sp.MethodIncompatibility, match="one endogenous"):
        sp.bayes_ivreg("y ~ (d + x ~ z)", df)
    with pytest.raises(sp.MethodIncompatibility, match="No excluded instrument"):
        sp.bayes_ivreg("y ~ x + (d ~ x)", df)
    with pytest.raises(sp.MethodIncompatibility, match="collinear"):
        sp.bayes_ivreg("y ~ x + (d ~ z + z2)", df.assign(z2=2 * df["z"]))
    with pytest.raises(sp.MethodIncompatibility, match="sigma_prior"):
        sp.bayes_ivreg("y ~ (d ~ z)", df, sigma_prior=(0.5, 1.0))


def test_everything_is_registered():
    names = set(sp.list_functions())
    for fn in (
        "bayes_regress",
        "bayes_mixed",
        "bayes_ivreg",
        "bma",
        "bayes_factor",
        "savage_dickey",
        "bayes_bootstrap",
        "mcmc_summary",
        "mcmc_ess",
        "hpd_interval",
        "geweke_diag",
        "raftery_diag",
        "heidel_diag",
        "gelman_rubin",
    ):
        assert fn in names
        spec = sp.describe_function(fn)
        assert spec["category"] == "bayes" and spec["example"]
    # the reference keys resolve in paper.bib
    text = sp.bibtex(keys=["kozumi2011gibbs", "raftery1997bayesian", "plummer2006coda"])
    assert "Kozumi" in text and "Raftery" in text and "Plummer" in text


def test_sur_shrink_and_mlogit_surface(df):
    d = df.assign(y2=df["y"] * 0.5 + np.random.default_rng(4).normal(size=len(df)))
    sur = sp.bayes_sur(["y ~ x1", "y2 ~ x2"], d, draws=800, burnin=200, seed=1)
    assert sur.model == "sur" and "corr(y,y2)" in sur.params.index
    assert sur.cite().startswith("zellner1962efficient")
    with pytest.raises(sp.MethodIncompatibility, match="at least two"):
        sp.bayes_sur(["y ~ x1"], d)
    with pytest.raises(sp.MethodIncompatibility, match="different outcome"):
        sp.bayes_sur(["y ~ x1", "y ~ x2"], d)
    las = sp.bayes_shrink("y ~ x1 + x2", d, draws=800, burnin=200, seed=1)
    assert list(las.params.index) == ["Intercept", "x1", "x2", "sigma2", "lam"]
    assert las.predict(d.head(3)).shape == (3,)
    ss = sp.bayes_shrink("y ~ x1 + x2", d, prior="ssvs", draws=800, burnin=200, seed=1)
    assert ss.table.loc["x1", "pip"] > 0.9 and np.isnan(ss.table.loc["sigma2", "pip"])
    # standardising makes the prior invariant to the units of a regressor
    scaled = sp.bayes_shrink(
        "y ~ big + x2",
        d.assign(big=d["x1"] * 1000),
        prior="ssvs",
        draws=800,
        burnin=200,
        seed=1,
    )
    assert scaled.params["big"] * 1000 == pytest.approx(ss.params["x1"], rel=1e-6)
    with pytest.raises(sp.MethodIncompatibility, match="'lasso' or 'ssvs'"):
        sp.bayes_shrink("y ~ x1", d, prior="horseshoe")
    with pytest.raises(sp.MethodIncompatibility, match="intercept"):
        sp.bayes_shrink("y ~ x1 - 1", d)
    with pytest.raises(sp.MethodIncompatibility, match="at least three"):
        sp.bayes_regress("d ~ x1", d, model="mlogit", **FAST)
    ml = sp.bayes_regress(
        "grade ~ x1", d, model="mlogit", draws=600, burnin=200, seed=1
    )
    assert ml.model_info["base_level"] == "low" and "top:x1" in ml.params.index
    with pytest.raises(sp.MethodIncompatibility, match="probabilities"):
        ml.predict()


def test_stochvol_and_bayes_arima_surface():
    rng = np.random.default_rng(8)
    n = 250
    h = np.zeros(n)
    for t in range(1, n):
        h[t] = -1 + 0.9 * (h[t - 1] + 1) + 0.3 * rng.normal()
    r = pd.Series(np.exp(h / 2) * rng.normal(size=n), name="ret")
    frame = pd.DataFrame({"ret": r})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        sv = sp.stochvol("ret", frame, draws=400, burnin=200, seed=1)
        again = sp.stochvol(r.to_numpy(), draws=400, burnin=200, seed=1)
        zeros = sp.stochvol(r.where(r.abs() > 0.05, 0.0), demean=False, **FAST)
    assert list(sv.params.index) == ["mu", "phi", "sigma"]
    assert sv.params.equals(again.params)
    vol = sv.model_info["volatility"]
    assert vol.shape == (n, 3) and (vol["lower"] <= vol["upper"]).all()
    assert sv.cite().startswith("kastner2014ancillarity")
    assert any("zero returns" in w for w in zeros.diagnostics_info["warnings"])
    with pytest.raises(sp.DataInsufficient):
        sp.stochvol(r.to_numpy()[:20])
    with pytest.raises(sp.MethodIncompatibility, match="missing"):
        sp.stochvol(np.r_[r.to_numpy(), np.nan])
    with pytest.raises(sp.MethodIncompatibility, match="column"):
        sp.stochvol("nope", frame)

    y = np.cumsum(0.2 + rng.normal(size=200))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        rw = sp.bayes_arima(y, order=(0, 1, 1), constant=True, horizon=3, **FAST)
        ar = sp.bayes_arima(np.diff(y), order=(2, 0, 0), chains=2, **FAST)
        no_const = sp.bayes_arima(y, order=(1, 1, 0), **FAST)
    assert list(rw.params.index) == ["const", "ma.L1", "sigma2"]
    fc = rw.model_info["forecast"]
    # the forecast is on the scale of the series, not of its differences
    assert fc.shape == (3, 4) and abs(fc["mean"].iloc[0] - y[-1]) < 3
    assert list(ar.params.index) == ["const", "ar.L1", "ar.L2", "sigma2"]
    assert set(ar.chain) == {0, 1}
    assert list(no_const.params.index) == ["ar.L1", "sigma2"]
    assert np.isfinite(ar.log_marginal_likelihood())
    # every draw is stationary
    roots = [
        np.abs(np.roots(np.r_[1.0, -row])).max()
        for row in ar.draws[["ar.L1", "ar.L2"]].to_numpy()[::50]
    ]
    assert max(roots) < 1
    with pytest.raises(sp.MethodIncompatibility, match="order"):
        sp.bayes_arima(y, order=(1, 0))
    with pytest.raises(sp.MethodIncompatibility, match="non-negative"):
        sp.bayes_arima(y, order=(-1, 0, 0))
    with pytest.raises(sp.MethodIncompatibility, match="horizon"):
        sp.bayes_arima(y, horizon=-1)
    with pytest.raises(sp.DataInsufficient):
        sp.bayes_arima(y[:8], order=(2, 0, 2))


def test_probit_systems_and_mixture_surface():
    rng = np.random.default_rng(9)
    n = 300
    x = rng.normal(size=n)
    e = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n)
    u = np.column_stack([np.zeros(n), 0.8 * x, -0.5 * x]) + rng.normal(size=(n, 3))
    d = pd.DataFrame(
        {
            "x": x,
            "y1": (0.5 * x + e[:, 0] > 0) * 1,
            "y2": (-0.5 * x + e[:, 1] > 0) * 1,
            "choice": np.array(["a", "b", "c"])[u.argmax(axis=1)],
            "z": np.r_[rng.normal(-3, 0.5, n // 2), rng.normal(3, 0.5, n - n // 2)],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        mv = sp.bayes_mvprobit(["y1 ~ x", "y2 ~ x"], d, draws=600, burnin=300, seed=1)
        mn = sp.bayes_mnprobit("choice ~ x", d, draws=600, burnin=300, seed=1)
    assert list(mv.params.index)[-1] == "corr(y1,y2)" and mv.params.iloc[-1] > 0.2
    assert mv.params["y1:x"] > 0 > mv.params["y2:x"]
    assert mv.cite().startswith("rossi2005bayesian")
    assert mn.model_info["base_level"] == "a"
    assert list(mn.params.index) == [
        "b:Intercept",
        "b:x",
        "c:Intercept",
        "c:x",
        "var(c)",
        "cov(b,c)",
    ]
    assert mn.params["b:x"] > 0 > mn.params["c:x"]
    with pytest.raises(sp.MethodIncompatibility, match="at least two"):
        sp.bayes_mvprobit(["y1 ~ x"], d)
    with pytest.raises(sp.MethodIncompatibility, match="0 and 1"):
        sp.bayes_mvprobit(["y1 ~ x", "z ~ x"], d)
    with pytest.raises(sp.DataInsufficient, match="does not vary"):
        sp.bayes_mvprobit(["y1 ~ x", "one ~ x"], d.assign(one=1))
    with pytest.raises(sp.MethodIncompatibility, match="at least\n?\\s*three|three"):
        sp.bayes_mnprobit("y1 ~ x", d)
    with pytest.raises(sp.MethodIncompatibility, match="column"):
        sp.bayes_mnprobit("nope ~ x", d)

    mix = sp.bayes_mixture("z ~ 1", d, draws=400, burnin=200, seed=1)
    assert mix.params["comp1:Intercept"] < 0 < mix.params["comp2:Intercept"]
    assert mix.model_info["similarity"].shape == (n, n)
    assert set(mix.model_info["cluster"]) == {1, 2}
    assert mix.prior["data_dependent"] is True
    assert {"y", "density", "lower", "upper"} == set(mix.model_info["density"].columns)
    reg = sp.bayes_mixture(
        "z ~ x", d, components="dp", alpha_prior=(2, 2), draws=300, burnin=200, seed=1
    )
    assert "density" not in reg.model_info and "alpha" in reg.params.index
    with pytest.raises(sp.MethodIncompatibility, match="'dp'"):
        sp.bayes_mixture("z ~ 1", d, components="many")
    with pytest.raises(sp.MethodIncompatibility, match="at least 2"):
        sp.bayes_mixture("z ~ 1", d, components=1)
    with pytest.raises(sp.MethodIncompatibility, match="alpha_prior"):
        sp.bayes_mixture("z ~ 1", d, components=2, alpha_prior=(1, 1))
    with pytest.warns(sp.ConvergenceWarning, match="max_components"):
        sp.bayes_mixture(
            "z ~ 1", d, components="dp", max_components=2, draws=200, burnin=100, seed=1
        )
