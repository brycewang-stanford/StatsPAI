"""``inits=``, ``continue_from=``, ``verbose=`` and ``keep_burnin=`` on the
:mod:`statspai.mcmc` samplers other than ``sp.bayes_regress`` (rOpenSci
BS1.3a, BS2.7, BS2.8, BS2.12, BS5.0, BS6.2; audit gap #11). The
``bayes_regress`` contract is in ``tests/test_bayes_chains.py``; this
file holds every other sampler to the same one through the shared
``statspai.mcmc._chains.ChainRun``.

Seeded draws of calls without the new arguments were checked to be
byte-identical to the release before the change (on the machine that made
the change); that check is not repeated here because seeded draws are
not identical across BLAS builds.
"""

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 160
    out = pd.DataFrame(
        {"x": rng.normal(size=n), "z": rng.normal(size=n), "w": rng.normal(size=n)}
    )
    e = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n)
    out["y"] = 1 + 0.5 * out.x + e[:, 0]
    out["y2"] = -1 + 0.8 * out.z + e[:, 1]
    out["yb"] = (out.y > 1).astype(int)
    out["yb2"] = (out.y2 > -1).astype(int)
    u = np.column_stack([np.zeros(n), 0.8 * out.x, -0.5 * out.x])
    u = u + rng.normal(size=(n, 3))
    out["choice"] = np.array(["a", "b", "c"])[u.argmax(1)]
    v = rng.normal(size=n)
    out["d"] = out.z + v
    out["yiv"] = 1 + 0.5 * out.d + 0.8 * v + 0.6 * rng.normal(size=n)
    out["g"] = np.repeat(np.arange(32), 5)
    out["ym"] = 1 + 0.5 * out.x + rng.normal(size=32)[out.g] + rng.normal(size=n)
    out["ymb"] = (out.ym > 1).astype(int)
    half = n // 2
    out["ymix"] = np.r_[rng.normal(-2, 0.5, half), rng.normal(2, 0.5, n - half)]
    out["yt"] = 1 + out.x + rng.standard_t(4, size=n)
    out["ytr"] = (rng.random(n) < 1 / (1 + np.exp(-out.x))).astype(int)
    out["ycau"] = out.x + 1.0 * out.ytr + rng.normal(scale=0.5, size=n)
    return out


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(1)
    r = np.zeros(300)
    h = 0.0
    for t in range(300):
        h = -1 + 0.9 * (h + 1) + 0.3 * rng.normal()
        r[t] = np.exp(h / 2) * rng.normal()
    ar = np.zeros(200)
    for t in range(1, 200):
        ar[t] = 0.6 * ar[t - 1] + rng.normal()
    return {"returns": r, "ar": ar}


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def _abc_sim(theta, rng):
    return rng.normal(theta[:, 0], 1 / np.sqrt(50))[:, None]


_ABC_PRIOR = [stats.norm(0, 3)]
_SHRINK = "y ~ x + z + w"

# name -> (function, positional arguments from the fixtures, fixed keyword
#          arguments, chains used in the tests or None for one-chain
#          samplers without a chains argument)
CASES = {
    "sur": (sp.bayes_sur, lambda d, s: (["y ~ x", "y2 ~ z"], d), {}, 2),
    "iv": (sp.bayes_ivreg, lambda d, s: ("yiv ~ (d ~ z)", d), {}, 2),
    "mvprobit": (sp.bayes_mvprobit, lambda d, s: (["yb ~ x", "yb2 ~ x"], d), {}, 2),
    "mnprobit": (sp.bayes_mnprobit, lambda d, s: ("choice ~ x", d), {}, 2),
    "stochvol": (sp.stochvol, lambda d, s: (s["returns"],), {}, 2),
    "arima": (sp.bayes_arima, lambda d, s: (s["ar"],), {"order": (1, 0, 1)}, 2),
    "mixed": (sp.bayes_mixed, lambda d, s: ("ym ~ x", d), {"group": "g"}, 2),
    "glmm": (
        sp.bayes_mixed,
        lambda d, s: ("ymb ~ x", d),
        {"group": "g", "family": "logit"},
        2,
    ),
    "lasso": (sp.bayes_shrink, lambda d, s: (_SHRINK, d), {"prior": "lasso"}, None),
    "horseshoe": (
        sp.bayes_shrink,
        lambda d, s: (_SHRINK, d),
        {"prior": "horseshoe"},
        None,
    ),
    "rhs": (
        sp.bayes_shrink,
        lambda d, s: (_SHRINK, d),
        {"prior": "horseshoe", "slab_scale": 2.0},
        None,
    ),
    "ssvs": (sp.bayes_shrink, lambda d, s: (_SHRINK, d), {"prior": "ssvs"}, None),
    "hs_logit": (
        sp.bayes_shrink,
        lambda d, s: ("yb ~ x + z + w", d),
        {"prior": "horseshoe", "family": "logit"},
        None,
    ),
    "mixture": (sp.bayes_mixture, lambda d, s: ("ymix ~ 1", d), {}, 2),
    "mixture_dp": (
        sp.bayes_mixture,
        lambda d, s: ("ymix ~ 1", d),
        {"components": "dp", "alpha_prior": (1, 1)},
        2,
    ),
    "semipar": (sp.semiparametric_bayes, lambda d, s: ("yt ~ x", d), {}, 2),
    "abc": (
        sp.abc,
        lambda d, s: (_abc_sim, [0.8], _ABC_PRIOR),
        {"method": "synthetic", "n_synthetic": 20, "vectorized": True},
        None,
    ),
    "bart": (sp.bart, lambda d, s: ("y ~ x + z", d), {"n_trees": 20}, None),
    "bart_binary": (
        sp.bart,
        lambda d, s: ("yb ~ x + z", d),
        {"n_trees": 20, "family": "binary"},
        None,
    ),
    "bcf": (
        sp.bart_causal,
        lambda d, s: (d, "ycau", "ytr", ["x"]),
        {"method": "bcf", "n_trees_control": 20, "n_trees_moderate": 10},
        2,
    ),
    "bart_causal": (
        sp.bart_causal,
        lambda d, s: (d, "ycau", "ytr", ["x"]),
        {"method": "bart", "n_trees_control": 20},
        2,
    ),
}

#: continued run == one longer run, draw for draw
EXACT = sorted(set(CASES) - {"arima", "bart", "bart_binary", "bcf", "bart_causal"})
#: results that are BayesRegressResult (draws in a frame, chain labels)
FRAME = sorted(set(CASES) - {"bart", "bart_binary", "bcf", "bart_causal"})
#: the draws of every sampler need at least this many per chain
MIN_DRAWS = {"abc": 100, "bart": 10, "bart_binary": 10, "bcf": 10, "bart_causal": 10}


def _call(name, df, series, **kw):
    fn, args, fixed, chains = CASES[name]
    kw = {"seed": 7, **fixed, **kw}
    if chains is not None:
        kw.setdefault("chains", chains)
    return _quiet(fn, *args(df, series), **kw)


def _draws(res):
    """The posterior draws as an array, chains stacked."""
    if isinstance(res, sp.BARTResult):
        return res._state["train_draws"]
    if hasattr(res, "tau_draws"):
        return res.tau_draws
    return res.draws.to_numpy()


def _summaries(res):
    """Everything a fit reports that is computed from its draws."""
    out = {}
    if isinstance(res, sp.BARTResult):
        out["fitted"] = res.fitted.to_numpy()
        out["importance"] = res.variable_importance.to_numpy()
        out["params"] = res.params.to_numpy()
        return out
    if hasattr(res, "tau_draws"):
        out["cate"] = res.cate.to_numpy()
        out["estimate"] = np.array([res.estimate, res.se])
        return out
    out["table"] = res.table.to_numpy(dtype=float)
    mi = res.model_info
    for key in ("volatility", "forecast", "density"):
        if isinstance(mi.get(key), pd.DataFrame):
            out[key] = mi[key].to_numpy(dtype=float)
    if "similarity" in mi:
        out["similarity"] = np.asarray(mi["similarity"], dtype=float)
    if "m_eff" in mi:
        out["m_eff"] = np.array([mi["m_eff"]])
    if isinstance(getattr(res, "random_effects", None), pd.DataFrame):
        out["random_effects"] = res.random_effects.to_numpy(dtype=float)
    comp = res.diagnostics_info.get("comparison")
    if isinstance(comp, pd.DataFrame):
        # a continued run keeps the first run's baseline posterior
        out["comparison"] = comp.drop(
            columns=["baseline_mean", "baseline_sd"]
        ).to_numpy(dtype=float)
    if res.acceptance_rate is not None:
        out["acceptance"] = np.array([res.acceptance_rate])
    return out


# ------------------------------------------------------------ continuation


@pytest.mark.parametrize("thin", [1, 2])
@pytest.mark.parametrize("name", EXACT)
def test_continued_chain_equals_one_long_chain(df, series, name, thin):
    if name == "abc" and thin > 1:
        pytest.skip("the synthetic-likelihood chain has no thin argument")
    n = MIN_DRAWS.get(name, 30)
    common = dict(burnin=20) if name == "abc" else dict(burnin=20, thin=thin)
    long = _call(name, df, series, draws=2 * n, **common)
    first = _call(name, df, series, draws=n, **common)
    cont = _call(name, df, series, draws=n, continue_from=first)
    # draw for draw, every chain
    np.testing.assert_array_equal(_draws(cont), _draws(long))
    np.testing.assert_array_equal(cont.chain, long.chain)
    for key, val in _summaries(long).items():
        np.testing.assert_allclose(_summaries(cont)[key], val, rtol=1e-12, err_msg=key)
    assert cont.n_draws == long.n_draws and cont.burnin == long.burnin
    assert cont.model_info["continued"] == {
        "draws_before": n,
        "draws_added": n,
        "exact": True,
    }
    # a continuation of a continuation is the same chain again
    longer = _call(name, df, series, draws=3 * n, **common)
    cont2 = _call(name, df, series, draws=n, continue_from=cont)
    np.testing.assert_array_equal(_draws(cont2), _draws(longer))


@pytest.mark.parametrize("name", ["arima", "bart", "bart_binary", "bcf", "bart_causal"])
def test_continued_chain_keeps_the_draws_and_is_reproducible(df, series, name):
    """The Metropolis sampler of ``bayes_arima`` draws a run's proposals in
    one block and the tree samplers draw from numba's generator, which is
    seeded again when a run continues; so these continuations differ from
    one longer run. What holds: the earlier draws are kept as they were,
    the continuation is a deterministic function of the earlier result,
    and it is flagged as not exact."""
    n = MIN_DRAWS.get(name, 30)
    first = _call(name, df, series, draws=n, burnin=20)
    a = _call(name, df, series, draws=n, continue_from=first)
    b = _call(name, df, series, draws=n, continue_from=first)
    chains = CASES[name][3] or 1
    before, after = _draws(first), _draws(a)
    per_chain = len(before) // chains
    for c in range(chains):
        np.testing.assert_array_equal(
            after[c * 2 * per_chain : c * 2 * per_chain + per_chain],
            before[c * per_chain : (c + 1) * per_chain],
        )
    np.testing.assert_array_equal(_draws(a), _draws(b))
    assert len(after) == 2 * len(before)
    assert a.model_info["continued"] == {
        "draws_before": n,
        "draws_added": n,
        "exact": False,
    }


@pytest.mark.parametrize("name", ["sur", "stochvol", "bart", "abc"])
def test_continuation_refuses_another_posterior(df, series, name):
    n = MIN_DRAWS.get(name, 20)
    first = _call(name, df, series, draws=n, burnin=10)
    fn, args, fixed, chains = CASES[name]
    shifted_df = df.assign(y=df.y + 1, y2=df.y2 + 1)
    shifted_series = {k: 2.0 * v for k, v in series.items()}
    other = args(shifted_df, shifted_series)
    if name == "abc":
        other = (_abc_sim, [0.5], _ABC_PRIOR)
    with pytest.raises(sp.MethodIncompatibility, match="different data"):
        _quiet(fn, *other, draws=n, continue_from=first, **fixed)
    with pytest.raises(sp.MethodIncompatibility, match="burn-in"):
        _call(name, df, series, draws=n, burnin=50, continue_from=first)
    with pytest.raises(sp.MethodIncompatibility, match="seed="):
        _call(name, df, series, draws=n, seed=8, continue_from=first)
    if name in ("sur", "stochvol"):
        with pytest.raises(sp.MethodIncompatibility, match="thin=1"):
            _call(name, df, series, draws=n, thin=3, continue_from=first)
        with pytest.raises(sp.MethodIncompatibility, match="chains=2"):
            _call(name, df, series, draws=n, chains=3, continue_from=first)
        with pytest.raises(sp.MethodIncompatibility, match="inits"):
            _call(name, df, series, draws=n, continue_from=first, inits=first.params)


def test_continue_from_must_come_from_the_same_sampler(df, series):
    other = _call("iv", df, series, draws=20, burnin=10)
    with pytest.raises(sp.MethodIncompatibility, match="sp.bayes_sur"):
        _call("sur", df, series, draws=20, continue_from=other)
    reg = _quiet(sp.bayes_regress, "y ~ x", df, draws=20, burnin=10, seed=1)
    with pytest.raises(sp.MethodIncompatibility, match="sp.bayes_sur"):
        _call("sur", df, series, draws=20, continue_from=reg)
    # and bayes_regress refuses the others' results
    with pytest.raises(sp.MethodIncompatibility, match="model="):
        _quiet(sp.bayes_regress, "y ~ x", df, draws=20, continue_from=other)


# --------------------------------------------------------------- burn-in


@pytest.mark.parametrize("name", sorted(CASES))
def test_keeping_the_burnin_does_not_change_the_draws(df, series, name):
    n = MIN_DRAWS.get(name, 30)
    res = _call(name, df, series, draws=n, burnin=20, keep_burnin=True)
    plain = _call(name, df, series, draws=n, burnin=20)
    np.testing.assert_array_equal(_draws(res), _draws(plain))
    chains = CASES[name][3] or 1
    assert len(res.burnin_draws) == 20 * chains
    assert plain.burnin_draws is None


#: samplers whose burn-in draws nothing that a kept iteration does not,
#: and does not tune: the kept burn-in is the start of a chain without one
UNTUNED = ["arima", "hs_logit", "iv", "mixed", "mixture_dp", "mnprobit"]
UNTUNED += ["mvprobit", "stochvol", "sur"]


@pytest.mark.parametrize("name", UNTUNED)
def test_kept_burnin_is_the_start_of_the_chain(df, series, name):
    full = _call(name, df, series, draws=50, burnin=0)
    res = _call(name, df, series, draws=30, burnin=20, keep_burnin=True)
    for c in range(res.chains):
        chain = full.draws.to_numpy()[full.chain == c]
        burn = res.burnin_draws.to_numpy()[res.burnin_chain == c]
        np.testing.assert_array_equal(burn, chain[:20])
        np.testing.assert_array_equal(res.draws.to_numpy()[res.chain == c], chain[20:])


def test_kept_burnin_of_bart_is_the_start_of_the_chain(df, series):
    full = _call("bart", df, series, draws=40, burnin=0)
    res = _call("bart", df, series, draws=20, burnin=20, keep_burnin=True)
    np.testing.assert_array_equal(
        res.burnin_draws["sigma"].to_numpy(), full.sigma.to_numpy()[:20]
    )
    np.testing.assert_array_equal(res.sigma.to_numpy(), full.sigma.to_numpy()[20:])


def test_shrink_burnin_holds_the_conditional_mean_of_the_intercept(df, series):
    """The intercept is drawn only at kept iterations; its burn-in rows are
    ``ybar - xbar' b`` at the slopes of the same row."""
    res = _call("lasso", df, series, draws=20, burnin=15, keep_burnin=True)
    burn = res.burnin_draws
    xbar = df[["x", "z", "w"]].mean().to_numpy()
    np.testing.assert_allclose(
        burn["Intercept"].to_numpy(),
        df.y.mean() - burn[["x", "z", "w"]].to_numpy() @ xbar,
        rtol=1e-12,
    )


def test_mixture_burnin_has_no_component_draws(df, series):
    res = _call("mixture", df, series, draws=20, burnin=15, keep_burnin=True)
    burn = res.burnin_draws
    assert burn.drop(columns="n_clusters").isna().all().all()
    assert burn["n_clusters"].between(1, 2).all()


def test_trace_plot_shows_a_kept_burnin(df, series):
    res = _call("sur", df, series, draws=30, burnin=15, thin=2, keep_burnin=True)
    fig = res.plot(params=["y:x"], kind="trace")
    ax = fig.axes[0]
    kept, burn = ax.lines[0], ax.lines[1]
    np.testing.assert_array_equal(burn.get_xdata(), np.arange(15))
    np.testing.assert_array_equal(kept.get_xdata(), 15 + 2 * np.arange(30))


# ----------------------------------------------------------------- inits


def _used(name, fit):
    """The parameters whose starting value the first step uses, at the
    values of an earlier fit."""
    p = fit.params
    if name == "sur":
        return {k: v for k, v in p.items() if ":" in k}
    if name == "iv":
        keep = ("rho", "sigma2_y", "sigma2_d")
        return {k: v for k, v in p.items() if k in keep or k.startswith("fs:")}
    if name == "mixed":
        return {k: v for k, v in p.items() if k == "sigma2" or "(" in k}
    if name in ("lasso", "horseshoe", "rhs", "ssvs"):
        return {k: v for k, v in p.items() if k in ("sigma2", "lam", "tau", "slab")}
    return dict(p)


INITS = ["abc", "arima", "glmm", "horseshoe", "hs_logit", "iv", "lasso", "mixed"]
INITS += ["mnprobit", "mvprobit", "rhs", "ssvs", "stochvol", "sur"]


@pytest.mark.parametrize("name", INITS)
def test_inits_are_the_recorded_starting_state(df, series, name):
    n = MIN_DRAWS.get(name, 40)
    base = _call(name, df, series, draws=n, burnin=40)
    used = _used(name, base)
    moved = dict(used)
    last = list(moved)[-1]
    moved[last] *= 0.95  # inside the support, off the posterior mean
    one = CASES[name][3] is None
    given = [moved] if one else [moved, base.params]
    res = _call(name, df, series, draws=n, burnin=0, inits=given if not one else moved)
    for c, values in enumerate([moved] if one else [moved, used]):
        row = res.inits.iloc[c]
        np.testing.assert_allclose(
            row[list(values)].to_numpy(float), list(values.values()), rtol=1e-12
        )
        # the rest is drawn before it is used
        assert row.drop(list(values)).isna().all()
    with pytest.raises(
        sp.MethodIncompatibility, match="unknown parameters.*no value for"
    ):
        _call(name, df, series, draws=n, burnin=0, inits={"not_a_parameter": 1.0})


def test_semiparametric_inits_are_on_the_sampler_scale(df, series):
    base = _call("semipar", df, series, draws=30, burnin=40)
    p = base.params
    start = {
        "Intercept": p["Intercept"],
        "x": p["x"] + 0.1,
        "log_sigma": np.log(p["sigma"]),
        "log_nu_minus_2": np.log(p["nu"] - 2.0),
        "delta:Intercept": 0.0,
        "delta:x": 0.0,
    }
    res = _call("semipar", df, series, draws=10, burnin=0, chains=1, inits=start)
    rec = res.inits.iloc[0]
    assert rec["x"] == pytest.approx(p["x"] + 0.1, rel=1e-12)
    assert rec["sigma"] == pytest.approx(p["sigma"], rel=1e-12)
    assert rec["nu"] == pytest.approx(p["nu"], rel=1e-12)
    assert rec["delta:x"] == 0.0


@pytest.mark.parametrize(
    "name, bad",
    [
        ("iv", {"rho": 1.5}),
        ("stochvol", {"phi": 1.2}),
        ("arima", {"ar.L1": 1.5}),
        ("mvprobit", {"corr(yb,yb2)": 1.5}),
        ("mixed", {"sigma2": -1.0}),
        ("lasso", {"sigma2": -1.0}),
        ("abc", {"theta1": np.inf}),
    ],
)
def test_inits_outside_the_support_are_refused(df, series, name, bad):
    n = MIN_DRAWS.get(name, 20)
    base = _call(name, df, series, draws=n, burnin=20)
    start = {**_used(name, base), **bad}
    with pytest.raises(sp.MethodIncompatibility, match="support|finite"):
        _call(name, df, series, draws=n, burnin=0, inits=start)


@pytest.mark.parametrize("name", ["mixture", "bart", "bcf"])
def test_inits_are_refused_where_the_state_is_not_a_parameter_vector(df, series, name):
    with pytest.raises(sp.MethodIncompatibility, match="inits= is not available"):
        _call(name, df, series, draws=10, inits={"x": 0.0})


def test_samplers_without_a_chain_refuse_the_chain_arguments(df):
    for kw in ({"inits": {"x": 0.0}}, {"keep_burnin": True}):
        with pytest.raises(sp.MethodIncompatibility, match="Markov chains"):
            sp.bayes_bootstrap("y ~ x", df, draws=20, seed=1, **kw)
        with pytest.raises(sp.MethodIncompatibility, match="Markov chains"):
            sp.abc(
                _abc_sim, [0.8], _ABC_PRIOR, n_sim=500, vectorized=True, seed=1, **kw
            )
    boot = sp.bayes_bootstrap("y ~ x", df, draws=20, seed=1)
    with pytest.raises(sp.MethodIncompatibility, match="Markov chains"):
        sp.bayes_bootstrap("y ~ x", df, draws=20, continue_from=boot)


# --------------------------------------------------------------- verbose


@pytest.mark.parametrize("name", sorted(CASES))
def test_verbose_controls_output(df, series, name, capsys):
    n = MIN_DRAWS.get(name, 20)
    _call(name, df, series, draws=n, burnin=10)
    assert capsys.readouterr().out == ""
    _call(name, df, series, draws=n, burnin=10, verbose=True)
    lines = capsys.readouterr().out.strip().splitlines()
    fn = CASES[name][0].__name__
    assert (
        lines[0]
        == f"{fn}: chain 1/{CASES[name][3] or 1}: {10 + n} iterations (10 burn-in)"
    )
    assert lines[-1].startswith(f"{fn}: {CASES[name][3] or 1} chain(s) x {n} draws")
    with pytest.raises(sp.MethodIncompatibility, match="verbose"):
        _call(name, df, series, draws=n, verbose="yes")
    with pytest.raises(sp.MethodIncompatibility, match="keep_burnin"):
        _call(name, df, series, draws=n, keep_burnin=1)


def test_verbose_on_the_samplers_without_a_chain(df, capsys):
    sp.bayes_bootstrap("y ~ x", df, draws=20, seed=1, verbose=True)
    assert capsys.readouterr().out.splitlines() == [
        "bayes_bootstrap: 20 weight draws",
        "bayes_bootstrap: 20 draws done; the draws are independent",
    ]
