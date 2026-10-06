"""Bootstrap likelihood-ratio test of the number of Markov-switching regimes.

The Monte Carlo evidence on size and power is produced by
``tests/reference_parity/_fixtures/_generate_mswitch_lrtest_mc.py`` (hours of
CPU) and asserted here from its committed output.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.timeseries import _mswitch_core as core
from statspai.timeseries.mswitch_lrtest import (
    MarkovSwitchingLRTest,
    _one_state,
    _pair,
    _simulate,
    _split,
    mswitch_lrtest,
)

MC = Path(__file__).parent / "reference_parity" / "_fixtures" / "mswitch_lrtest_mc.json"


def _two_regimes(n: int, seed: int, gap: float = 1.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.uniform() < 0.95 else 1 - s[t - 1]
    return np.where(s == 0, -gap, gap) + rng.normal(size=n)


def _ar1(n: int, seed: int, phi: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal()
    return y


def _data(y: np.ndarray, p: int, x: np.ndarray | None = None) -> core.Data:
    empty = np.empty((len(y), 0))
    return core.Data(y, empty if x is None else x.reshape(len(y), -1), empty, p)


@pytest.fixture(scope="module")
def clear() -> tuple[np.ndarray, MarkovSwitchingLRTest]:
    y = _two_regimes(80, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return y, mswitch_lrtest(y, states=2, reps=9, starts=3, seed=11)


# --- the one-regime likelihood -------------------------------------------


def test_one_regime_loglik_is_gaussian_ols() -> None:
    y = _ar1(120, 0)
    x = np.random.default_rng(1).normal(size=120)
    # white noise around a mean, with a regressor: OLS of y on (1, x)
    spec = core.Spec(1, "dr", 0, 1, 0, "switch", False, False)
    theta, ll = _one_state(spec, _data(y, 0, x), 500, 1e-9)
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    # same maximiser in closed form: agreement to rounding
    assert ll == pytest.approx(ols.llf, rel=1e-12)
    assert theta[:2] == pytest.approx(ols.params, rel=1e-10)
    # two lags as regressors, conditional on the first two observations
    spec = core.Spec(1, "dr", 2, 0, 0, "switch", False, False)
    _, ll = _one_state(spec, _data(y, 2), 500, 1e-9)
    lags = sm.add_constant(np.column_stack([y[1:-1], y[:-2]]))
    assert ll == pytest.approx(sm.OLS(y[2:], lags).fit().llf, rel=1e-12)


def test_one_regime_ar_form_is_the_same_likelihood() -> None:
    # y_t - mu = phi (y_{t-1} - mu) + e_t is the AR(1) regression
    # re-parameterised (mu = c / (1 - phi)); Newton on the switching
    # likelihood expression must land on the OLS value: 1e-9 relative.
    y = _ar1(120, 0) + 2.0
    spec = core.Spec(1, "ar", 1, 0, 0, "switch", False, False)
    theta, ll = _one_state(spec, _data(y, 1), 500, 1e-9)
    ols = sm.OLS(y[1:], sm.add_constant(y[:-1])).fit()
    assert ll == pytest.approx(ols.llf, rel=1e-9)
    assert theta[0] == pytest.approx(ols.params[0] / (1 - ols.params[1]), rel=1e-6)
    assert theta[1] == pytest.approx(ols.params[1], rel=1e-6)


def test_null_is_a_point_of_the_alternative() -> None:
    # splitting a regime into identical copies leaves the likelihood
    # unchanged (lumpable chain): exact up to rounding in the filter.
    y = _ar1(100, 4)
    for model, p in (("dr", 0), ("ar", 1), ("dr", 2)):
        dat = _data(y, p)
        spec0 = core.Spec(1, model, p, 0, 0, "switch", False, False)
        theta0, ll0 = _one_state(spec0, dat, 500, 1e-9)
        for k in (2, 3):
            spec1 = core.Spec(k, model, p, 0, 0, "switch", False, False)
            for stay in (0.9, 0.5):
                th = _split(spec0, spec1, dat, theta0, stay, 0.0)
                assert core.loglik(spec1, dat, th) == pytest.approx(ll0, abs=1e-9)
    # two regimes inside three, with state-dependent variances
    spec2 = core.Spec(2, "dr", 0, 0, 0, "switch", False, True)
    spec3 = core.Spec(3, "dr", 0, 0, 0, "switch", False, True)
    dat = _data(y, 0)
    th2 = np.array([-0.4, 0.7, -0.1, 0.2, -2.0, 1.5])
    th3 = _split(spec2, spec3, dat, th2, 0.9, 0.0)
    assert core.loglik(spec3, dat, th3) == pytest.approx(
        core.loglik(spec2, dat, th2), abs=1e-9
    )


# --- simulation from the null --------------------------------------------


def test_simulated_series_have_the_moments_of_the_model() -> None:
    n = 40_000
    base = np.zeros(n)
    base[0] = 3.0
    rng = np.random.default_rng(5)
    # AR(1) around a mean of 3, coefficient 0.6, sd 2
    spec = core.Spec(1, "ar", 1, 0, 0, "switch", False, False)
    y = _simulate(spec, _data(base, 1), np.array([3.0, 0.6, np.log(2.0)]), rng)
    assert y[0] == 3.0  # the presample observation is kept
    # tolerances: 4 Monte Carlo standard errors at n = 40,000
    assert y.mean() == pytest.approx(3.0, abs=0.1)
    assert y.var() == pytest.approx(4.0 / (1 - 0.36), rel=0.05)
    assert np.corrcoef(y[1:], y[:-1])[0, 1] == pytest.approx(0.6, abs=0.02)
    # the same numbers in the dynamic-regression form: c = 3 (1 - 0.6)
    spec = core.Spec(1, "dr", 1, 0, 0, "switch", False, False)
    y = _simulate(spec, _data(base, 1), np.array([1.2, 0.6, np.log(2.0)]), rng)
    assert y.mean() == pytest.approx(3.0, abs=0.1)
    # two regimes: means -1 / 2, p11 = 0.9, p22 = 0.8, sd 0.5
    spec = core.Spec(2, "dr", 0, 0, 0, "switch", False, False)
    q = [-np.log(0.9 / 0.1), -np.log(0.2 / 0.8)]
    y = _simulate(spec, _data(base, 0), np.array([-1.0, 2.0, np.log(0.5)] + q), rng)
    share = float((y > 0.5).mean())
    assert share == pytest.approx(1.0 / 3.0, abs=0.02)  # ergodic 0.1 / 0.3
    assert y[y > 0.5].mean() == pytest.approx(2.0, abs=0.02)
    assert y[y <= 0.5].std() == pytest.approx(0.5, abs=0.02)


def test_simulation_keeps_regressors_fixed() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(size=5_000)
    spec = core.Spec(1, "dr", 0, 1, 0, "switch", False, False)
    dat = _data(np.zeros(5_000), 0, x)
    y = _simulate(spec, dat, np.array([1.0, 2.0, np.log(0.1)]), rng)
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    assert fit.params == pytest.approx([1.0, 2.0], abs=0.01)


# --- the test itself -----------------------------------------------------


def test_statistic_and_fits(clear: tuple[np.ndarray, MarkovSwitchingLRTest]) -> None:
    y, res = clear
    assert res.fit_null is None and res.null_states == 1 and res.states == 2
    ols = sm.OLS(y, np.ones(len(y))).fit()
    assert res.loglik_null == pytest.approx(ols.llf, rel=1e-12)
    assert res.loglik_alt == res.fit_alt.loglik
    assert res.statistic == pytest.approx(2 * (res.fit_alt.loglik - ols.llf), rel=1e-9)
    # the alternative fit is the maximum sp.mswitch finds from its own starts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sp.mswitch(y, states=2, starts=5, seed=0)
    assert res.loglik_alt == pytest.approx(plain.loglik, abs=1e-7)
    assert res.null_params["const"] == pytest.approx(y.mean(), rel=1e-10)
    assert res.naive_df == 3


def test_pvalue_and_replicates(
    clear: tuple[np.ndarray, MarkovSwitchingLRTest],
) -> None:
    _, res = clear
    rep = res.replicates
    assert len(rep) == 9 and res.n_valid == 9 and res.diagnostics["n_failed"] == 0
    assert (rep["lr"] >= 0.0).all()
    assert (2 * (rep["loglik_alt"] - rep["loglik_null"]) >= -1e-8)[
        ~rep["floored"]
    ].all()
    # clear regimes: the statistic exceeds every replicate
    assert res.pvalue == pytest.approx(1 / 10)
    assert res.pvalue == (1 + (rep["lr"] >= res.statistic).sum()) / 10
    assert not res.reject  # 9 replicates cannot reject at 5%
    assert any("cannot reject" in note for note in res.model_info["notes"])
    crit = res.critical_values
    assert list(crit.index) == [0.10, 0.05, 0.01]
    assert crit[0.10] == rep["lr"].max()  # ceil(10 * 0.9) = 9th of 9
    assert np.isinf(crit[0.05]) and np.isinf(crit[0.01])
    for key in ("n_floored", "n_multiple_maxima", "n_alt_not_converged"):
        assert 0 <= res.diagnostics[key] <= 9
    text = res.summary()
    assert "1 against 2" in text and "does not establish" in text
    assert "not valid" in text
    json.dumps(res.to_dict())


def test_observed_series_and_replicates_share_one_search(
    clear: tuple[np.ndarray, MarkovSwitchingLRTest],
) -> None:
    y, res = clear
    spec0 = core.Spec(1, "dr", 0, 0, 0, "switch", False, False)
    spec1 = core.Spec(2, "dr", 0, 0, 0, "switch", False, False)
    seq = np.random.SeedSequence(11).spawn(10)[0]
    pair = _pair(spec0, spec1, _data(y, 0), seq, 3, 500, 1e-9)
    # the routine applied to the replicates, run on the sample: same bits
    assert pair["loglik_alt"] == res.loglik_alt
    assert pair["loglik_null"] == res.loglik_null


def test_seed_reproducibility_and_processes() -> None:
    y = _ar1(50, 8)
    kw = dict(states=2, reps=2, starts=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = mswitch_lrtest(y, seed=5, **kw)
        b = mswitch_lrtest(y, seed=5, n_jobs=2, **kw)
        c = mswitch_lrtest(y, seed=6, **kw)
    pd.testing.assert_frame_equal(a.replicates, b.replicates)
    assert a.pvalue == b.pvalue and a.statistic == b.statistic
    assert not np.array_equal(a.replicates["lr"], c.replicates["lr"])


def test_two_against_three_regimes() -> None:
    y = _two_regimes(90, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mswitch_lrtest(y, states=3, reps=2, starts=3, seed=1)
    assert res.null_states == 2 and res.fit_null is not None
    assert res.loglik_null == res.fit_null.loglik
    assert res.statistic >= 0.0 and (res.replicates["lr"] >= 0.0).all()
    assert res.naive_df == 5  # one mean and four transition parameters
    assert list(res.null_params.index) == list(res.fit_null.theta.index)


def test_start_params_hook_of_mswitch() -> None:
    y = _two_regimes(80, 3)
    base = sp.mswitch(y, states=2, starts=1)
    again = sp.mswitch(y, states=2, starts=1, start_params=[base.theta.to_numpy()])
    assert len(again.starts) == 2 and np.isnan(again.starts["loglik_em"].iloc[1])
    assert again.starts["loglik"].iloc[1] == pytest.approx(base.loglik, abs=1e-9)
    assert again.loglik == pytest.approx(base.loglik, abs=1e-9)
    with pytest.raises(MethodIncompatibility, match="start_params"):
        sp.mswitch(y, states=2, start_params=[np.zeros(3)])


def test_bad_arguments() -> None:
    y = _two_regimes(80, 3)
    with pytest.raises(MethodIncompatibility, match="bootstrap"):
        mswitch_lrtest(y, method="chi2")
    with pytest.raises(MethodIncompatibility, match="null_states"):
        mswitch_lrtest(y, states=2, null_states=2)
    with pytest.raises(MethodIncompatibility, match="null_states"):
        mswitch_lrtest(y, states=2, null_states=0)
    for bad in (dict(reps=0), dict(starts=2), dict(n_jobs=0)):
        with pytest.raises(MethodIncompatibility, match="reps >= 1"):
            mswitch_lrtest(y, **bad)
    with pytest.raises(MethodIncompatibility, match="model must be"):
        mswitch_lrtest(y, model="var")


# --- Monte Carlo evidence (committed output of the generator) ------------


@pytest.fixture(scope="module")
def mc() -> dict:
    out = json.loads(MC.read_text(encoding="utf-8"))
    assert out["generator"] == "_generate_mswitch_lrtest_mc.py"
    assert out["scale"] == 1.0
    return out["designs"]


@pytest.mark.parametrize("design", ["size_wn", "size_ar1"])
def test_mc_size_is_nominal(mc: dict, design: str) -> None:
    d = mc[design]
    assert d["mc"] >= 200 and d["starts"] == 10
    for tag, level in (("10", 0.10), ("5", 0.05)):
        # within 3 Monte Carlo standard errors of the nominal level
        assert abs(d[f"reject_{tag}"] - level) <= 3 * d[f"reject_{tag}_mc_se"]
        # chi-squared with one degree of freedom (one restricted mean)
        # rejects several times too often
        assert d[f"chi2_df1_reject_{tag}"] > 3 * level
        # with the parameter-count difference it is not below the bootstrap
        assert d[f"chi2_df3_reject_{tag}"] >= d[f"reject_{tag}"]
    # no statistic below zero other than those counted as set to zero
    assert d["lr_raw_min"] > -1e-8 or d["floored"] > 0


def test_mc_power(mc: dict) -> None:
    # 5 of the 100 "clear" samples have at most 3 dates in the second
    # regime; the test rejects in all the others
    assert mc["power_clear"]["reject_5"] >= 0.95
    assert mc["power_moderate"]["reject_5"] >= 0.75


def test_mc_warp_speed_agrees_with_the_full_bootstrap(mc: dict) -> None:
    d = mc["double_wn"]
    for tag, level in (("10", 0.10), ("5", 0.05)):
        se = d[f"reject_{tag}_mc_se"]
        # full bootstrap in each of the samples: nominal within 3 MC se
        assert abs(d[f"reject_{tag}"] - level) <= 3 * se
        # and the one-replicate device on the same samples gives the same
        # rate up to the Monte Carlo error of both
        assert abs(d[f"reject_{tag}"] - d[f"warp_reject_{tag}"]) <= 3 * se
