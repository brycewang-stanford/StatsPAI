"""Regression tests for the October 2026 review of ``statspai.mcmc``.

Each test states a proposition that held false before the fix: a default
prior that crushed an estimate without a word, a prediction from a missing
regressor, a raw NumPy / patsy / SciPy error in place of a typed one, a
result that changed with the row labels of the data. Chains are short on
purpose: nothing here is about the posterior itself.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
    StatsPAIWarning,
)

K = dict(draws=80, burnin=40, seed=1)
PRIOR_NOTE = "The default prior is not vague"


def _notes(fit_call):
    """Run a fit, return (result, texts of the default-prior warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_call()
    texts = [
        str(w.message)
        for w in caught
        if issubclass(w.category, StatsPAIWarning) and PRIOR_NOTE in str(w.message)
    ]
    return res, texts


def _quiet(fit_call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_call()


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 200
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    u = rng.normal(size=n)
    d = 0.8 * z + 0.3 * x + 0.7 * u + rng.normal(size=n)
    out = pd.DataFrame({"x": x, "z": z, "d": d})
    out["y"] = 1 + 1.5 * d - 0.5 * x + u + rng.normal(size=n)
    out["b1"] = (0.2 + x + rng.normal(size=n) > 0).astype(int)
    out["b2"] = (-0.2 + 0.8 * z + rng.normal(size=n) > 0).astype(int)
    util = np.c_[
        np.zeros(n),
        0.5 * x + rng.normal(size=n),
        -0.5 * x + rng.normal(size=n),
    ]
    out["m"] = util.argmax(axis=1)
    out["g"] = np.repeat(np.arange(20), 10)
    out["w"] = 3 + 2 * x + rng.normal(size=20)[out["g"]] + rng.normal(size=n)
    out["c"] = rng.poisson(2.0, n)
    out["ex"] = rng.uniform(1.0, 2.0, n)
    out["nt"] = 5
    out["s"] = rng.binomial(5, 0.4, n)
    out["yo"] = np.digitize(x + rng.normal(size=n), [-0.5, 0.5])
    # the same data in other units: every coefficient is 1000 times larger
    out["d_k"] = out["d"] / 1000.0
    out["x_k"] = out["x"] / 1000.0
    out["z_k"] = out["z"] / 1000.0
    out["w_big"] = out["w"] * 1000.0 + 3e6
    return out


# ---------------------------------------------------------------------------
# 1. the default-prior check runs wherever a fixed default prior is in force
# ---------------------------------------------------------------------------

_PRIOR_CASES = {
    "ivreg": (
        lambda d, **kw: sp.bayes_ivreg("y ~ x + (d ~ z)", d, **K, **kw),
        lambda d, **kw: sp.bayes_ivreg("y ~ x + (d_k ~ z)", d, **K, **kw),
        {"prior_var": 1e9},
        "d_k",
    ),
    "sur": (
        lambda d, **kw: sp.bayes_sur(["y ~ x + d", "z ~ x"], d, **K, **kw),
        lambda d, **kw: sp.bayes_sur(["y ~ x + d_k", "z ~ x"], d, **K, **kw),
        {"prior_var": 1e9},
        "y:d_k",
    ),
    "mixed": (
        lambda d, **kw: sp.bayes_mixed("w ~ x", d, group="g", **K, **kw),
        lambda d, **kw: sp.bayes_mixed("w_big ~ x", d, group="g", **K, **kw),
        {"prior_var": 1e14},
        "Intercept",
    ),
    "mvprobit": (
        lambda d, **kw: sp.bayes_mvprobit(["b1 ~ x", "b2 ~ z"], d, **K, **kw),
        lambda d, **kw: sp.bayes_mvprobit(["b1 ~ x_k", "b2 ~ z_k"], d, **K, **kw),
        {"prior_var": 1e8},
        "b1:x_k",
    ),
    "mnprobit": (
        lambda d, **kw: sp.bayes_mnprobit("m ~ x", d, **K, **kw),
        lambda d, **kw: sp.bayes_mnprobit("m ~ x_k", d, **K, **kw),
        {"prior_var": 1e8},
        "1:x_k",
    ),
    "arima": (
        lambda d, **kw: sp.bayes_arima(d["y"].to_numpy(), **K, **kw),
        lambda d, **kw: sp.bayes_arima(d["y"].to_numpy() * 1000 + 5e4, **K, **kw),
        {"mean_prior": (5e4, 1e12)},
        "const",
    ),
}


@pytest.mark.parametrize("name", list(_PRIOR_CASES))
def test_default_prior_is_flagged_where_it_binds(df, name):
    well_scaled, rescaled, own_prior, param = _PRIOR_CASES[name]

    _, texts = _notes(lambda: well_scaled(df))
    assert texts == [], "a prior that is vague for these data was flagged"

    res, texts = _notes(lambda: rescaled(df))
    assert len(texts) >= 1
    assert any(param in t for t in texts)
    # the note is on the result too, as sp.bayes_regress records it
    assert any(PRIOR_NOTE in t for t in res.diagnostics_info["warnings"])

    res, texts = _notes(lambda: rescaled(df, **own_prior))
    assert texts == []
    assert not any(PRIOR_NOTE in t for t in res.diagnostics_info["warnings"])


def test_default_prior_note_does_not_touch_the_draws(df):
    # the check reads summaries only: the flagged fit is the fit obtained
    # when the warning is turned into nothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.bayes_ivreg("y ~ x + (d_k ~ z)", df, **K)
        b = sp.bayes_ivreg("y ~ x + (d_k ~ z)", df, prior_var=1000.0 + 1e-9, **K)
    # same seed, a prior that differs in the 12th digit: same random stream
    np.testing.assert_allclose(a.draws.to_numpy(), b.draws.to_numpy(), rtol=1e-6)


def test_bayes_regress_keeps_its_own_note(df):
    res, texts = _notes(lambda: sp.bayes_regress("y ~ x + d_k", df, prior="vague", **K))
    assert len(texts) == 1 and "prior='weakly_informative'" in texts[0]
    assert texts[0] in res.diagnostics_info["warnings"]


# ---------------------------------------------------------------------------
# 2. predictions refuse a missing regressor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def surface():
    rng = np.random.default_rng(3)
    out = pd.DataFrame(rng.uniform(-2, 2, (90, 2)), columns=["a", "b"])
    out["y"] = np.where(out["a"] > 0, 2.0, -1.0) + 0.5 * rng.normal(size=90)
    return out


def test_bart_predict_refuses_missing_regressor(surface):
    fit = sp.bart("y ~ a + b", surface, n_trees=8, draws=20, burnin=10, seed=1)
    bad = pd.DataFrame({"a": [np.nan, 0.5], "b": [0.0, 1.0]})
    with pytest.raises(MethodIncompatibility, match=r"values in a\b"):
        fit.predict(bad)
    # complete rows still predict: on the sample, the fitted values
    # (which are stored in single precision)
    good = fit.predict(surface.iloc[:4])
    np.testing.assert_allclose(
        good["mean"].to_numpy(), fit.fitted["mean"].to_numpy()[:4], rtol=1e-6
    )


def test_gp_predict_refuses_missing_regressor(surface):
    fit = sp.gp_regress(
        "y ~ a + b",
        surface,
        optimize_hyper=False,
        length_scale=[1.0, 1.0],
        signal_var=1.0,
        noise_var=0.3,
    )
    bad = pd.DataFrame({"a": [0.1, 0.5], "b": [np.inf, 1.0]})
    with pytest.raises(MethodIncompatibility, match=r"values in b\b"):
        fit.predict(bad)
    assert fit.predict(surface.iloc[:3]).shape == (3, 4)


def test_gp_refuses_what_it_would_ignore(surface):
    with pytest.raises(MethodIncompatibility, match="mean of the process"):
        sp.gp_regress("y ~ a + b - 1", surface)
    with pytest.raises(MethodIncompatibility, match="one per regressor"):
        sp.gp_regress("y ~ a + b", surface, length_scale=[1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# 3. model averaging on outcomes it cannot explain
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bma_df():
    rng = np.random.default_rng(5)
    out = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    out["k"] = 1.0
    out["zero"] = 0
    out["sep"] = (out["a"] > 0).astype(int)
    out["rare"] = 0
    out.loc[:5, "rare"] = 1
    noisy = out["b"] + rng.normal(size=200) > 0
    out["quasi"] = (noisy | (out["rare"] == 1)).astype(int)
    out["fine"] = noisy.astype(int)
    return out


def test_bma_refuses_constant_outcome(bma_df):
    with pytest.raises(DataInsufficient, match="constant"):
        sp.bma("k ~ a + b + c", bma_df)
    with pytest.raises(DataInsufficient, match="constant"):
        sp.bma("zero ~ a + b", bma_df, family="binomial")


def test_bma_complete_separation_is_a_typed_error(bma_df):
    with pytest.raises(NumericalInstability, match="separation"):
        _quiet(lambda: sp.bma("sep ~ a + b", bma_df, family="binomial"))


def test_bma_quasi_separation_warns_as_logit_does(bma_df):
    with pytest.warns(ConvergenceWarning, match="separation") as rec:
        out = sp.bma("quasi ~ b + rare", bma_df, family="binomial")
    assert any("inclusion probabilities" in str(w.message) for w in rec)
    assert any("separation" in note for note in out.settings["notes"])
    # a well-behaved binary outcome is left alone
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        sp.bma("fine ~ a + b", bma_df, family="binomial")


# ---------------------------------------------------------------------------
# 4. row labels do not matter
# ---------------------------------------------------------------------------

_INDEX_CASES = {
    "regress": lambda d: sp.bayes_regress("y ~ x", d, prior="vague", **K),
    "exposure": lambda d: sp.bayes_regress(
        "c ~ x", d, model="poisson", prior="vague", exposure="ex", **K
    ),
    "offset_array": lambda d: sp.bayes_regress(
        "c ~ x",
        d,
        model="poisson",
        prior="vague",
        offset=np.log(d["ex"].to_numpy()),
        **K,
    ),
    "trials": lambda d: sp.bayes_regress(
        "s ~ x", d, model="logit", prior="vague", trials="nt", **K
    ),
    "ordered": lambda d: sp.bayes_regress(
        "yo ~ x", d, model="oprobit", prior="vague", **K
    ),
    "ivreg": lambda d: sp.bayes_ivreg("y ~ x + (d ~ z)", d, **K),
    "sur": lambda d: sp.bayes_sur(["y ~ x + d", "z ~ x"], d, **K),
    "mixed": lambda d: sp.bayes_mixed("w ~ x", d, group="g", **K),
    "mvprobit": lambda d: sp.bayes_mvprobit(["b1 ~ x", "b2 ~ z"], d, **K),
    "mnprobit": lambda d: sp.bayes_mnprobit("m ~ x", d, **K),
    "shrink": lambda d: sp.bayes_shrink("y ~ x + z + d", d, **K),
}


@pytest.mark.parametrize("name", list(_INDEX_CASES))
def test_repeated_row_labels_change_nothing(df, name):
    dup = df.copy()
    dup.index = list(range(len(df) // 2)) * 2
    fit = _INDEX_CASES[name]
    a = _quiet(lambda: fit(dup))
    b = _quiet(lambda: fit(df.reset_index(drop=True)))
    assert a.n_obs == b.n_obs == len(df)
    np.testing.assert_array_equal(a.draws.to_numpy(), b.draws.to_numpy())


def test_kfold_and_log_lik_on_repeated_row_labels(df):
    dup = df.copy()
    dup.index = list(range(len(df) // 2)) * 2
    a = _quiet(lambda: sp.bayes_regress("y ~ x", dup, prior="vague", **K))
    b = _quiet(lambda: sp.bayes_regress("y ~ x", df, prior="vague", **K))
    ka = _quiet(lambda: sp.kfold(a, k=3, seed=2))
    kb = _quiet(lambda: sp.kfold(b, k=3, seed=2))
    # rtol: the refits are the same draws, the log-sum-exp over them goes
    # through threaded BLAS and can differ in the last bit
    np.testing.assert_allclose(
        ka.pointwise.to_numpy(), kb.pointwise.to_numpy(), rtol=1e-10
    )
    # scoring the estimation rows as new data is the in-sample matrix
    np.testing.assert_allclose(a.log_lik(dup), a.log_lik(), rtol=1e-12)


# ---------------------------------------------------------------------------
# 5. a training fold without a factor level
# ---------------------------------------------------------------------------


def test_kfold_names_the_missing_factor_level(df):
    d = df.copy()
    d["f"] = np.where(np.arange(len(d)) % 2 == 0, "u", "v")
    d.loc[0, "f"] = "rare"  # one row: its fold's training set has no 'rare'
    fit = _quiet(lambda: sp.bayes_regress("y ~ x + f", d, prior="vague", **K))
    with pytest.raises(MethodIncompatibility, match="kfold_split") as err:
        _quiet(lambda: sp.kfold(fit, k=4, seed=1))
    assert "stratify" in str(err.value)


# ---------------------------------------------------------------------------
# 6. checks on results that have no predictive distribution
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def other_fits(df, surface):
    return _quiet(
        lambda: {
            "mixed": sp.bayes_mixed("w ~ x", df, group="g", **K),
            "ivreg": sp.bayes_ivreg("y ~ x + (d ~ z)", df, **K),
            "sur": sp.bayes_sur(["y ~ x + d", "z ~ x"], df, chains=2, **K),
            "mnprobit": sp.bayes_mnprobit("m ~ x", df, **K),
            "gp": sp.gp_regress(
                "y ~ a",
                surface,
                optimize_hyper=False,
                length_scale=1.0,
                signal_var=1.0,
                noise_var=0.3,
            ),
            "bart": sp.bart(
                "y ~ a + b", surface, n_trees=8, draws=20, burnin=10, seed=1
            ),
        }
    )


@pytest.mark.parametrize("name", ["mixed", "ivreg", "sur", "mnprobit", "gp", "bart"])
@pytest.mark.parametrize("check", ["ppc", "bayes_r2", "loo_r2", "loo", "waic"])
def test_checks_refuse_other_results_cleanly(other_fits, name, check):
    with pytest.raises(MethodIncompatibility):
        getattr(sp, check)(other_fits[name])


@pytest.mark.parametrize("name", ["sur", "mnprobit"])
def test_predict_says_why_it_is_unavailable(other_fits, name):
    with pytest.raises(MethodIncompatibility) as err:
        other_fits[name].predict()
    assert "user-supplied statistic" not in str(err.value)
    assert f"model='{name}'" in str(err.value)


# ---------------------------------------------------------------------------
# 7, 8. chains are compared as chains
# ---------------------------------------------------------------------------


def test_gelman_rubin_reads_the_chains_of_a_fit(df):
    fit = _quiet(lambda: sp.bayes_regress("y ~ x", df, prior="vague", chains=3, **K))
    for split in (False, True):
        a = sp.gelman_rubin(fit, split=split)
        b = sp.gelman_rubin(fit.chain_list(), split=split)
        pd.testing.assert_frame_equal(a.table, b.table)
        assert a.settings["chains"] == b.settings["chains"] == (6 if split else 3)
    one = _quiet(lambda: sp.bayes_regress("y ~ x", df, prior="vague", **K))
    with pytest.raises(MethodIncompatibility, match="two chains"):
        sp.gelman_rubin(one)
    assert sp.gelman_rubin(one, split=True).settings["chains"] == 2


@pytest.mark.parametrize("name", ["sur", "mixed"])
def test_split_rhat_is_the_one_of_the_chains(df, name):
    if name == "sur":
        fit = _quiet(lambda: sp.bayes_sur(["y ~ x + d", "z ~ x"], df, chains=2, **K))
    else:
        fit = _quiet(lambda: sp.bayes_mixed("w ~ x", df, group="g", chains=2, **K))
    ref = sp.gelman_rubin(fit.chain_list(), split=True).table["psrf"].max()
    assert fit.diagnostics_info["max_split_rhat"] == pytest.approx(ref, rel=1e-12)


# ---------------------------------------------------------------------------
# 9. validation
# ---------------------------------------------------------------------------


def test_bayes_regress_validation(df):
    B = lambda f, d=df, **kw: _quiet(  # noqa: E731
        lambda: sp.bayes_regress(f, d, prior="vague", **K, **kw)
    )
    with pytest.raises(MethodIncompatibility, match="inference"):
        B("y ~ x", model="conjugate", inference="zzz")
    # a '- 1' inside a term is not a request to drop the intercept
    assert list(B("yo ~ I(x - 1)", model="oprobit").params.index) == [
        "I(x - 1)",
        "cut1",
        "cut2",
    ]
    with pytest.raises(MethodIncompatibility, match="cutpoints"):
        B("yo ~ x - 1", model="ologit")
    with pytest.raises(MethodIncompatibility, match="model='ologit'"):
        B("b1 ~ x", model="ologit")
    with pytest.raises(MethodIncompatibility, match="infinite"):
        B("y ~ x", df.assign(y=df["y"].where(df.index != 3, np.inf)))
    # a continuous outcome has one category per row
    for model in ("oprobit", "ologit", "mlogit"):
        with pytest.raises(DataInsufficient, match="distinct values"):
            B("y ~ x", model=model)


def test_intercept_parser():
    from statspai.mcmc._core import drops_intercept

    assert drops_intercept("x - 1") and drops_intercept("0 + x")
    assert drops_intercept("x1-1") and drops_intercept("x + 0")
    assert not drops_intercept("I(x1-1)")
    assert not drops_intercept('Q("a-1") + x')
    assert not drops_intercept("x - 10") and not drops_intercept("x1")


def test_mixture_stochvol_bootstrap_validation(df):
    with pytest.raises(MethodIncompatibility, match="max_components"):
        sp.bayes_mixture("y ~ 1", df, components=2, max_components=3, **K)
    rng = np.random.default_rng(2)
    with pytest.raises(MethodIncompatibility, match="one series"):
        sp.stochvol(rng.normal(size=(100, 2)), **K)
    with pytest.raises(MethodIncompatibility, match="constant"):
        sp.stochvol(np.ones(100), demean=False, **K)
    r = rng.normal(size=80)
    dated = pd.Series(r, index=pd.date_range("2020-01-01", periods=80))
    a = _quiet(lambda: sp.stochvol(dated, **K))
    b = _quiet(lambda: sp.stochvol(r[:, None], **K))
    assert a.model_info["volatility"].index.equals(dated.index)
    np.testing.assert_array_equal(a.draws.to_numpy(), b.draws.to_numpy())

    with pytest.raises(MethodIncompatibility, match="at least 2"):
        sp.bayes_bootstrap("y ~ x", df, draws=1, seed=1)
    with pytest.raises(MethodIncompatibility, match="same quantities"):
        sp.bayes_bootstrap(
            data=df,
            statistic=lambda d, w: np.ones(1 if w[0] > w[1] else 2),
            draws=30,
            seed=1,
        )
    with pytest.raises(DataInsufficient, match="constant"):
        sp.bayes_mixed("zero ~ x", df.assign(zero=0), group="g", family="logit", **K)


def test_abc_validation():
    def sim(theta, rng):
        return theta[:, :1] + rng.normal(size=(len(theta), 1))

    def broken(theta, rng):
        return np.full((len(theta), 1), np.nan)

    prior = [stats.norm(0, 3)]
    with pytest.raises(MethodIncompatibility, match="scale has 2 entries"):
        sp.abc(sim, [1.0], prior, vectorized=True, n_sim=500, scale=[1.0, 2.0])
    for method in ("rejection", "synthetic"):
        with pytest.raises(DataInsufficient, match="Every simulation"):
            sp.abc(
                broken,
                [1.0],
                prior,
                vectorized=True,
                n_sim=500,
                method=method,
                draws=100,
                seed=1,
            )


def test_summaries_and_checks_validation(df):
    fit = _quiet(lambda: sp.bayes_regress("y ~ x", df, prior="vague", **K))
    with pytest.raises(MethodIncompatibility, match="n_boot"):
        sp.loo_r2(fit, n_boot=0)
    with pytest.raises(MethodIncompatibility, match="level"):
        sp.bayes_r2(fit, level=3)
    with pytest.raises(MethodIncompatibility, match="quantiles"):
        sp.mcmc_summary(fit.draws, quantiles=(1.5,))
    with pytest.warns(StatsPAIWarning, match="not finite"):
        out = sp.ppc(fit, stat=lambda y: np.nan, seed=1)
    assert np.isnan(out.p_value)
    assert 0.0 <= sp.ppc(fit, seed=1).p_value <= 1.0


def test_savage_dickey_far_tail_is_finite_in_logs(df):
    fit = _quiet(lambda: sp.bayes_regress("y ~ x + d", df, prior="vague", **K))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        far = sp.savage_dickey(fit, "x", 50.0)
    assert np.isfinite(far["log_bf01"]) and far["log_bf01"] < -100
    assert far["bf01"] == 0.0
    # where nothing underflows the two routes are the same number
    near = sp.savage_dickey(fit, "x", -0.5)
    assert near["log_bf01"] == pytest.approx(
        np.log(near["posterior_density"]) - np.log(near["prior_density"]), abs=1e-12
    )
    assert near["bf01"] == pytest.approx(np.exp(near["log_bf01"]), rel=1e-12)
