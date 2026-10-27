"""Option scan, round 6: ``statspai.mcmc`` and ``statspai.bayes``.

Each test states the correct behaviour of an option combination that had
no test before. The PyMC tests skip when the ``bayes`` extra is missing;
run them with ``PYTENSOR_FLAGS="cxx=,mode=NUMBA"`` on machines whose
pytensor C compiler fails.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import ConvergenceWarning, MethodIncompatibility


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(list("abc"), n)})
    df["y"] = 1.0 + 0.5 * df["x"] + 0.7 * (df["g"] == "b") + rng.normal(size=n)
    return df


# --------------------------------------------------------------------------
# Prediction on new data with a missing regressor
# --------------------------------------------------------------------------


@pytest.mark.parametrize("formula", ["y ~ x + C(g)", "y ~ x"])
def test_bayes_regress_predict_refuses_missing_regressor(frame, formula):
    """A row of new data with a missing regressor has no prediction.

    The patsy design dropped that row, so ``predict`` returned one value
    fewer than there were rows and every later value sat against the
    wrong row; the plain-column design returned NaN for it. Both now
    refuse, as ``sp.bart`` and ``sp.gp_regress`` already did.
    """
    fit = sp.bayes_regress(formula, frame, prior="vague", draws=200, burnin=50, seed=1)
    new = frame.head(5).copy()
    new.loc[new.index[1], "x"] = np.nan
    for call in (
        lambda: fit.predict(new),
        lambda: fit.posterior_epred(new),
        lambda: fit.posterior_linpred(new),
        lambda: fit.posterior_predict(new, seed=1),
    ):
        with pytest.raises(MethodIncompatibility, match="missing"):
            call()
    # complete rows predict one value each, in order
    full = frame.head(5)
    out = fit.predict(full)
    assert out.shape == (5,)
    beta = fit.draws.to_numpy()[:, : fit._model.k]
    X = fit._design(full)
    np.testing.assert_allclose(out, (beta @ X.T).mean(axis=0), rtol=1e-12)


def test_bma_predict_refuses_missing_regressor(frame):
    out = sp.bma("y ~ x + g", frame, method="gprior")
    new = frame.head(5).copy()
    new.loc[new.index[2], "g"] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing"):
        out.predict(new)
    assert out.predict(frame.head(5)).shape == (5,)


# --------------------------------------------------------------------------
# sp.bayes_dml
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dml_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["d"] = 0.5 * df["x1"] + rng.normal(size=n)
    df["y"] = df["d"] + df["x1"] - df["x2"] + rng.normal(size=n)
    return df


def test_bayes_dml_random_state_sets_the_cross_fitting_split(dml_frame):
    """``random_state`` reached only the Monte Carlo draws; the folds of
    the DML fit kept the default seed of ``sp.dml`` (42), so changing it
    never changed the posterior."""
    kw = dict(y="y", covariates=["x1", "x2"])
    for seed in (1, 42):
        post = sp.bayes_dml(dml_frame, treatment="d", random_state=seed, **kw)
        ref = sp.dml(dml_frame, treat="d", random_state=seed, **kw)
        assert post.dml_point == ref.estimate
        assert post.dml_se == ref.se
        # the Normal-Normal update of that point and standard error
        prec = 1.0 / 10.0**2 + 1.0 / ref.se**2
        assert post.posterior_mean == pytest.approx(
            (ref.estimate / ref.se**2) / prec, rel=1e-12
        )
        assert post.posterior_sd == pytest.approx(prec**-0.5, rel=1e-12)


def test_bayes_dml_full_mode_refuses_before_fitting(dml_frame, monkeypatch):
    """``mode='full'`` needs score arrays that ``sp.dml`` does not return;
    it used to fit the whole DML model first and then raise a bare
    RuntimeError."""
    import sys

    dml_pkg = sys.modules["statspai.dml"]

    def boom(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("sp.dml was fitted before the refusal")

    monkeypatch.setattr(dml_pkg, "dml", boom)
    with pytest.raises(MethodIncompatibility, match="mode='conjugate'"):
        sp.bayes_dml(
            dml_frame, y="y", treatment="d", covariates=["x1", "x2"], mode="full"
        )


# --------------------------------------------------------------------------
# sp.bayes_did / sp.bayes_synth input handling (no sampling needed)
# --------------------------------------------------------------------------


def test_bayes_did_warns_when_missing_cohort_drops_rows():
    """Never-treated units coded with a missing cohort were dropped
    without a word, taking the comparison group with them."""
    from statspai.bayes.did import _prepare_did_frame

    rng = np.random.default_rng(1)
    rows = []
    for unit in range(8):
        cohort = [2.0, 3.0, np.nan, np.nan][unit % 4]
        for t in range(1, 5):
            treated = not np.isnan(cohort)
            rows.append(
                {
                    "id": unit,
                    "t": t,
                    "y": rng.normal(),
                    "treat": int(treated),
                    "post": int(treated and t >= cohort),
                    "g": cohort,
                }
            )
    df = pd.DataFrame(rows)
    with pytest.warns(sp.exceptions.StatsPAIWarning, match="cohort"):
        prep = _prepare_did_frame(df, "y", "treat", "post", "id", "t", None, "g")
    assert prep["n"] == 16
    # a sentinel keeps every row and needs no warning
    df["g"] = df["g"].fillna(-1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", sp.exceptions.StatsPAIWarning)
        prep = _prepare_did_frame(df, "y", "treat", "post", "id", "t", None, "g")
    assert prep["n"] == 32


def test_bayes_synth_refuses_duplicate_unit_time_rows():
    """``pivot_table`` averaged duplicated (unit, time) rows silently."""
    from statspai.bayes.synth import _wide_panel

    df = pd.DataFrame(
        {
            "u": np.repeat(["A", "B", "C"], 4),
            "t": np.tile([1, 2, 3, 4], 3),
            "y": np.arange(12, dtype=float),
        }
    )
    times, treated, donors, names = _wide_panel(df, "y", "u", "t", "A", None)
    np.testing.assert_array_equal(treated, [0.0, 1.0, 2.0, 3.0])
    dup = pd.concat([df, df.iloc[[5]].assign(y=100.0)], ignore_index=True)
    with pytest.raises(MethodIncompatibility, match="duplicate"):
        _wide_panel(dup, "y", "u", "t", "A", None)


# --------------------------------------------------------------------------
# PyMC-based estimators
# --------------------------------------------------------------------------


def _pymc():
    return pytest.importorskip("pymc")


def test_bayes_iv_per_instrument_uses_the_pooled_sample():
    """The per-instrument sub-fits re-read ``data`` and dropped only the
    rows missing their own instrument, so they ran on a larger sample
    than the pooled fit they are reported next to."""
    _pymc()
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame({"z1": rng.normal(size=n), "z2": rng.normal(size=n)})
    v = rng.normal(size=n)
    df["d"] = 0.8 * df["z1"] + 0.6 * df["z2"] + v
    df["y"] = 1.0 + 0.5 * df["d"] + 0.5 * v + rng.normal(size=n)
    df.loc[:9, "z2"] = np.nan
    kw = dict(draws=150, tune=150, chains=1, random_state=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pooled = sp.bayes_iv(
            df, y="y", treat="d", instrument=["z1", "z2"], per_instrument=True, **kw
        )
        alone = sp.bayes_iv(df.iloc[10:], y="y", treat="d", instrument="z1", **kw)
    assert pooled.n_obs == n - 10
    got = pooled.instrument_summaries["z1"]
    assert got["posterior_mean"] == alone.posterior_mean
    assert got["posterior_sd"] == alone.posterior_sd


def test_bayes_fit_with_poor_diagnostics_raises_convergence_warning():
    """The module convention (rhat > 1.01 or ESS < 400 gives a
    ConvergenceWarning) was printed in ``summary()`` only."""
    _pymc()
    rng = np.random.default_rng(0)
    n = 150
    x = rng.uniform(-1, 1, n)
    df = pd.DataFrame({"x": x, "y": 0.5 + 0.8 * x + (x >= 0) + rng.normal(size=n)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sp.bayes_rd(
            df,
            y="y",
            running="x",
            bandwidth=0.6,
            draws=60,
            tune=60,
            chains=2,
            random_state=1,
        )
    ours = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert fit.ess < 400
    assert len(ours) == 1
    assert "effective sample size" in str(ours[0].message)


def test_convergence_warning_counts_divergent_transitions():
    pm = _pymc()
    from statspai.bayes._base import _sample_model, _warn_convergence

    # Neal's funnel: NUTS diverges at this step size
    with pm.Model() as funnel:
        v = pm.Normal("v", 0.0, 3.0)
        pm.Normal("x", 0.0, pm.math.exp(v / 2.0), shape=9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = _sample_model(
            funnel, draws=300, tune=100, chains=2, random_state=1, target_accept=0.6
        )
    n_div = int(np.asarray(trace.sample_stats["diverging"].values).sum())
    assert n_div > 0
    with pytest.warns(ConvergenceWarning, match=f"{n_div} divergent"):
        _warn_convergence(trace, rhat=1.0, ess=1000.0)
    # one warning per fit
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        _warn_convergence(trace, rhat=1.0, ess=1000.0)
