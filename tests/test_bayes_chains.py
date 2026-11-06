"""Starting values, continuation, verbosity and kept burn-in for
``sp.bayes_regress`` (rOpenSci BS1.3a, BS2.7, BS2.8, BS2.12, BS2.13, BS5.0,
BS6.2; audit gap #11).
"""

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.mcmc._core import spawn_rngs

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 150
    out = pd.DataFrame({"x": rng.normal(size=n)})
    out["y"] = 1 + 0.5 * out.x + rng.normal(size=n)
    out["yb"] = (out.y > 1).astype(int)
    out["yc"] = rng.poisson(np.exp(0.2 + 0.3 * out.x.to_numpy()))
    out["yo"] = pd.cut(out.y, 3, labels=False)
    out["ycens"] = np.maximum(out.y, 0.5)
    return out


def _fit(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.bayes_regress(*args, **kw)


GIBBS = {
    "normal": dict(formula="y ~ x"),
    "normal_vague": dict(formula="y ~ x", prior="vague"),
    "t": dict(formula="y ~ x", model="t"),
    "probit": dict(formula="yb ~ x", model="probit"),
    "tobit": dict(formula="ycens ~ x", model="tobit", lower=0.5),
    "quantile": dict(formula="y ~ x", model="quantile"),
    "oprobit": dict(formula="yo ~ x", model="oprobit"),
}
METROPOLIS = {
    "logit": dict(formula="yb ~ x", model="logit"),
    "poisson": dict(formula="yc ~ x", model="poisson"),
    "negbin": dict(formula="yc ~ x", model="negbin"),
    "ologit": dict(formula="yo ~ x", model="ologit"),
    "mlogit": dict(formula="yo ~ x", model="mlogit"),
}


@pytest.mark.parametrize("chains, thin", [(1, 1), (2, 3)])
@pytest.mark.parametrize("name", list(GIBBS))
def test_continued_gibbs_chain_equals_one_long_chain(df, name, chains, thin):
    kw = dict(GIBBS[name])
    f = kw.pop("formula")
    common = dict(burnin=20, thin=thin, chains=chains, seed=7, **kw)
    long = _fit(f, df, draws=60, **common)
    first = _fit(f, df, draws=30, **common)
    cont = _fit(f, df, draws=30, continue_from=first, **kw)
    # draw for draw, every chain
    np.testing.assert_array_equal(cont.draws.to_numpy(), long.draws.to_numpy())
    np.testing.assert_array_equal(cont.chain, long.chain)
    pd.testing.assert_frame_equal(cont.table, long.table)
    assert cont.n_draws == long.n_draws and cont.burnin == long.burnin
    assert cont.model_info["continued"] == {
        "draws_before": 30,
        "draws_added": 30,
        "exact": True,
    }
    if long.acceptance_rate is not None:  # oprobit's cutpoint step
        assert cont.acceptance_rate == pytest.approx(long.acceptance_rate, rel=1e-12)
    # and a continuation of a continuation is the same chain again
    longer = _fit(f, df, draws=90, **common)
    cont2 = _fit(f, df, draws=30, continue_from=cont, **kw)
    np.testing.assert_array_equal(cont2.draws.to_numpy(), longer.draws.to_numpy())


@pytest.mark.parametrize("name", list(METROPOLIS))
def test_continued_metropolis_chain_keeps_the_draws_and_is_reproducible(df, name):
    """The block-drawn proposals make these chains differ from one longer
    run (see ``statspai.mcmc._chains``); what holds is that the earlier
    draws are kept as they were and the continuation is a deterministic
    function of the earlier result."""
    kw = dict(METROPOLIS[name])
    f = kw.pop("formula")
    first = _fit(f, df, draws=30, burnin=20, chains=2, seed=7, **kw)
    a = _fit(f, df, draws=40, continue_from=first, **kw)
    b = _fit(f, df, draws=40, continue_from=first, **kw)
    for c in range(2):
        np.testing.assert_array_equal(
            a.draws.to_numpy()[a.chain == c][:30],
            first.draws.to_numpy()[first.chain == c],
        )
    np.testing.assert_array_equal(a.draws.to_numpy(), b.draws.to_numpy())
    assert a.n_draws == 140
    assert a.model_info["continued"]["exact"] is False


def test_continuation_refuses_another_posterior(df):
    first = _fit("y ~ x", df, draws=20, burnin=10, seed=1)
    with pytest.raises(sp.MethodIncompatibility, match="different data"):
        _fit("y ~ x", df.assign(y=df.y + 1), draws=20, continue_from=first)
    with pytest.raises(sp.MethodIncompatibility, match="model="):
        _fit("y ~ x", df, model="t", draws=20, continue_from=first)
    with pytest.raises(sp.MethodIncompatibility, match="thin=1"):
        _fit("y ~ x", df, draws=20, thin=2, continue_from=first)
    with pytest.raises(sp.MethodIncompatibility, match="burn-in"):
        _fit("y ~ x", df, draws=20, burnin=50, continue_from=first)
    with pytest.raises(sp.MethodIncompatibility, match="inits"):
        _fit("y ~ x", df, draws=20, continue_from=first, inits=[0, 0, 1])
    with pytest.raises(sp.MethodIncompatibility, match="Markov"):
        _fit("y ~ x", df, model="conjugate", draws=20, continue_from=first)


# ------------------------------------------------------------------ inits


def test_first_gibbs_update_conditions_on_the_inits(df):
    """Normal model: sigma2 is drawn from IG(an/2, (d0 + SSR(beta0))/2)
    and then beta from its normal full conditional; rebuilt here from the
    model's full conditionals with the chain's own generator."""
    beta0 = {"Intercept": 5.0, "x": -3.0, "sigma2": 1.0}
    res = _fit("y ~ x", df, prior="vague", draws=2, burnin=0, seed=5, inits=beta0)
    X = np.column_stack([np.ones(len(df)), df.x])
    y = df.y.to_numpy()
    b = np.array([5.0, -3.0])
    ssr = float((y - X @ b) @ (y - X @ b))
    a0 = d0 = 0.001
    rng = spawn_rngs(5, 1)[0]
    s2 = (d0 + ssr) / 2.0 / rng.gamma((a0 + len(y)) / 2.0)
    prec = np.eye(2) / 1000.0 + X.T @ X / s2
    mean = np.linalg.solve(prec, X.T @ y / s2)
    chol = np.linalg.cholesky(prec)
    beta = mean + np.linalg.solve(chol.T, rng.standard_normal(2))
    np.testing.assert_allclose(res.draws["sigma2"].iloc[0], s2, rtol=1e-12)
    np.testing.assert_allclose(res.draws[["Intercept", "x"]].iloc[0], beta, rtol=1e-10)
    # sigma2 is drawn first, so the record shows it was not used
    assert res.inits.loc[0, "Intercept"] == 5.0 and res.inits.loc[0, "x"] == -3.0
    assert np.isnan(res.inits.loc[0, "sigma2"])


@pytest.mark.parametrize(
    "name", ["t", "tobit", "quantile", "probit", "oprobit", "logit", "ologit"]
)
def test_inits_are_the_recorded_starting_state(df, name):
    kw = dict({**GIBBS, **METROPOLIS}[name])
    f = kw.pop("formula")
    base = _fit(f, df, draws=200, burnin=100, seed=2, **kw)
    start = base.params.copy()
    start.iloc[0] += 0.5  # a point inside the support, off the mode
    res = _fit(
        f, df, draws=10, burnin=0, chains=2, seed=2, inits=[start, base.params], **kw
    )
    np.testing.assert_allclose(res.inits.iloc[0].to_numpy(), start.to_numpy(), 1e-12)
    np.testing.assert_allclose(
        res.inits.iloc[1].to_numpy(), base.params.to_numpy(), 1e-12
    )


def test_default_start_is_least_squares(df):
    res = _fit("y ~ x", df, prior="vague", draws=5, burnin=0, seed=1)
    X = np.column_stack([np.ones(len(df)), df.x])
    ols = np.linalg.lstsq(X, df.y.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(res.inits[["Intercept", "x"]].iloc[0], ols, rtol=1e-12)


def test_bad_inits_are_refused(df):
    with pytest.raises(sp.MethodIncompatibility, match="no value for"):
        _fit("y ~ x", df, draws=5, burnin=0, inits={"Intercept": 0.0})
    with pytest.raises(sp.MethodIncompatibility, match="support"):
        _fit(
            "y ~ x",
            df,
            prior="vague",
            draws=5,
            burnin=0,
            inits={"Intercept": 0.0, "x": 0.0, "sigma2": -1.0},
        )
    with pytest.raises(sp.MethodIncompatibility, match="shape"):
        _fit("y ~ x", df, draws=5, burnin=0, chains=2, inits=np.zeros((3, 3)))
    with pytest.raises(sp.MethodIncompatibility, match="2 sets"):
        _fit(
            "y ~ x",
            df,
            draws=5,
            burnin=0,
            chains=3,
            inits=[{"Intercept": 0, "x": 0, "sigma2": 1}] * 2,
        )


# ---------------------------------------------------------------- verbose


def test_verbose_controls_output_but_not_warnings(df, capsys):
    _fit("y ~ x", df, draws=50, burnin=10, chains=2, seed=1)
    assert capsys.readouterr().out == ""
    with pytest.warns(sp.ConvergenceWarning):
        sp.bayes_regress("y ~ x", df, draws=20, burnin=0, seed=1, verbose=False)
    assert capsys.readouterr().out == ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp.bayes_regress(
            "y ~ x", df, draws=50, burnin=10, chains=2, seed=1, verbose=True
        )
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "bayes_regress: chain 1/2: 60 iterations (10 burn-in)"
    assert lines[2] == "bayes_regress: chain 2/2: 60 iterations (10 burn-in)"
    assert lines[-1].startswith("bayes_regress: 2 chain(s) x 50 draws; min ESS")
    with pytest.raises(sp.MethodIncompatibility, match="verbose"):
        sp.bayes_regress("y ~ x", df, draws=5, verbose="yes")


# ---------------------------------------------------------------- burn-in


def test_kept_burnin_is_the_start_of_the_chain(df):
    full = _fit("y ~ x", df, draws=70, burnin=0, chains=2, seed=9)
    res = _fit("y ~ x", df, draws=50, burnin=20, chains=2, seed=9, keep_burnin=True)
    for c in range(2):
        chain = full.draws.to_numpy()[full.chain == c]
        np.testing.assert_array_equal(
            res.burnin_draws.to_numpy()[res.burnin_chain == c], chain[:20]
        )
        np.testing.assert_array_equal(res.draws.to_numpy()[res.chain == c], chain[20:])
    # the summaries use the kept draws only
    pd.testing.assert_frame_equal(
        res.table,
        _fit("y ~ x", df, draws=50, burnin=20, chains=2, seed=9).table,
    )
    assert _fit("y ~ x", df, draws=5, burnin=5, seed=9).burnin_draws is None


def test_trace_plot_marks_the_burnin(df):
    res = _fit("y ~ x", df, draws=40, burnin=15, thin=2, seed=9, keep_burnin=True)
    fig = res.plot(params=["x"], kind="trace")
    ax = fig.axes[0]
    kept, burn = ax.lines[0], ax.lines[1]
    np.testing.assert_array_equal(burn.get_xdata(), np.arange(15))
    np.testing.assert_array_equal(kept.get_xdata(), 15 + 2 * np.arange(40))
    np.testing.assert_array_equal(kept.get_ydata(), res.draws["x"].to_numpy())
    # the dashed line sits between the last burn-in and the first kept draw
    assert ax.lines[2].get_xdata()[0] == pytest.approx(14.5)
