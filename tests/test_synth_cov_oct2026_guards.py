"""Coverage round (Oct 2026): input guards and small exact helpers in
``statspai.synth`` (K-fold t-test, CausalImpact, staggered SCM, the
regression-control method and the shared simplex / placebo primitives).

Guard tests assert the exception class and message.  Helper tests assert
closed-form answers worked out by hand in the comments.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.synth import _core as core

bsts = importlib.import_module("statspai.synth.bsts")


def _panel(seed=0, J=6, T=14, T0=10, eff=2.0):
    """Units u0..uJ over periods 1..T; u0 treated from period T0 + 1."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=T).cumsum()
    lam = rng.uniform(0.5, 1.5, J + 1)
    rows = []
    for i in range(J + 1):
        for t in range(T):
            d = int(i == 0 and t >= T0)
            y = 5 + lam[i] * f[t] + 0.3 * i + rng.normal(0, 0.2) + eff * d
            rows.append({"unit": f"u{i}", "time": t + 1, "y": y, "d": d})
    return pd.DataFrame(rows)


KW = dict(outcome="y", unit="unit", time="time")


# --------------------------------------------------------------------- #
#  sp.synth_ttest
# --------------------------------------------------------------------- #


def _ttest(df, **kw):
    args = dict(KW, treated_unit="u0", treatment_time=11)
    args.update(kw)
    return sp.synth_ttest(df, **args)


def test_ttest_rejects_malformed_requests():
    df = _panel()
    with pytest.raises(MethodIncompatibility, match="n_folds must be at least 2"):
        _ttest(df, n_folds=1)
    with pytest.raises(MethodIncompatibility, match="columns not found"):
        _ttest(df, outcome="nope")
    with pytest.raises(MethodIncompatibility, match="one row per"):
        _ttest(pd.concat([df, df.iloc[:1]]))
    with pytest.raises(MethodIncompatibility, match="is not a value of"):
        _ttest(df, treated_unit="zz")


def test_ttest_reports_insufficient_data():
    df = _panel()
    with pytest.raises(DataInsufficient, match="no post-treatment period"):
        _ttest(df, treatment_time=99)
    with pytest.raises(DataInsufficient, match="too few for n_folds"):
        _ttest(df, treatment_time=3, n_folds=3)  # two pre-periods
    hole = df.copy()
    hole.loc[(hole.unit == "u0") & (hole.time == 2), "y"] = np.nan
    with pytest.raises(DataInsufficient, match="treated unit has missing"):
        _ttest(hole)
    empty = df.copy()
    empty.loc[(empty.unit != "u0") & (empty.time == 2), "y"] = np.nan
    with pytest.raises(DataInsufficient, match="no donor has complete"):
        _ttest(empty)


def test_ttest_drops_an_incomplete_donor_and_matches_the_fit_without_it():
    df = _panel()
    hole = df.copy()
    hole.loc[(hole.unit == "u3") & (hole.time == 4), "y"] = np.nan
    with pytest.warns(UserWarning, match="Dropped 1 donor"):
        res = _ttest(hole)
    ref = _ttest(df[df.unit != "u3"])
    # same donor matrix after the drop: identical computation
    assert res.estimate == pytest.approx(ref.estimate, rel=1e-12)
    assert res.se == pytest.approx(ref.se, rel=1e-12)


# --------------------------------------------------------------------- #
#  CausalImpact (statspai.synth.bsts)
# --------------------------------------------------------------------- #


def _ts(seed=0, n=30):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n).cumsum()
    y = 1.0 + 0.8 * x + rng.normal(0, 0.1, n)
    y[20:] += 2.0
    return pd.DataFrame({"y": y, "x": x})


def _ci(df=None, **kw):
    args = dict(
        pre_period=(0, 19), post_period=(20, 29), outcome="y", n_simulations=50, seed=0
    )
    args.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bsts.causal_impact(_ts() if df is None else df, **args)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(outcome=""), "`outcome` must be a non-empty column name"),
        (dict(model=3), "`model` must be a string option"),
        (dict(model="  "), "`model` must be a non-empty string option"),
        (dict(alpha=True), r"`alpha` must be a number in \(0, 1\)"),
        (dict(alpha="abc"), r"`alpha` must be a number in \(0, 1\)"),
        (dict(alpha=1.5), r"`alpha` must be in \(0, 1\)"),
        (dict(n_simulations=True), "`n_simulations` must be an integer >= 2"),
        (dict(n_simulations="many"), "`n_simulations` must be an integer >= 2"),
        (dict(n_simulations=1), "`n_simulations` must be an integer >= 2"),
        (dict(n_simulations=2.5), "`n_simulations` must be an integer >= 2"),
        (dict(covariates=5), "`covariates` must be a column name or list"),
        (dict(covariates=[""]), "only non-empty string column names"),
        (dict(pre_period=5), "`pre_period` must be a two-element period pair"),
        (dict(post_period=(20, 25, 29)), "`post_period` must be a two-element"),
        (dict(pre_period=("a", "b")), "must be comparable with the index"),
    ],
)
def test_causal_impact_rejects_malformed_arguments(kwargs, message):
    with pytest.raises(MethodIncompatibility, match=message):
        _ci(**kwargs)


def test_causal_impact_rejects_unusable_data():
    inf = _ts()
    inf.loc[3, "y"] = np.inf
    with pytest.raises(DataInsufficient, match="non-finite"):
        _ci(inf)
    nox = _ts()
    nox.loc[:19, "x"] = np.nan
    with pytest.raises(DataInsufficient, match="observed pre-period values"):
        _ci(nox, covariates=["x"])


def test_causal_impact_accepts_a_single_covariate_name():
    a = _ci(covariates="x")
    b = _ci(covariates=["x"])
    # same model, same seed
    assert a.estimate == b.estimate
    assert a.model_info["n_covariates"] == 1
    # the covariate explains the series, so the +2 level shift is found;
    # 0.5 is several times the simulation spread of this 20-point fit
    assert a.estimate == pytest.approx(2.0, abs=0.5)


def test_causal_impact_without_covariates_is_a_pure_local_level_forecast():
    res = _ci(covariates=[])
    mi = res.model_info
    assert mi["n_covariates"] == 0 and mi["regression_coefficients"] == {}
    # a local level forecasts a flat line: the counterfactual mean has no
    # trend beyond Monte Carlo noise of the 50 simulated paths
    cf = np.asarray(mi["counterfactual_mean"])
    assert cf.shape == (10,)
    assert np.ptp(cf) < 3 * mi["sigma_level"] * np.sqrt(10) + 1e-8


def test_kalman_filter_skips_the_update_at_a_missing_observation():
    y = np.array([1.0, 1.2, np.nan, 1.1, 0.9])
    X = np.zeros((5, 0))
    kf = bsts._kalman_filter(y, X, np.zeros(0), sigma_obs=0.5, sigma_level=0.3)
    state = np.asarray(kf["filtered_state"]).reshape(5, -1)[:, 0]
    cov = np.asarray(kf["filtered_cov"]).reshape(5, -1)[:, 0]
    # random-walk level: with no observation the filtered state is the
    # prediction (previous level) and its variance grows by sigma_level^2
    assert state[2] == state[1]
    assert cov[2] == pytest.approx(cov[1] + 0.3**2, rel=1e-12)
    assert cov[3] < cov[2]  # the next observation is informative again


# --------------------------------------------------------------------- #
#  sp.staggered_synth
# --------------------------------------------------------------------- #


def _stag(seed=0, J=8, T=12, starts=None):
    """``starts`` maps unit index -> first treated period (1-based)."""
    starts = {0: 8, 1: 10} if starts is None else starts
    rng = np.random.default_rng(seed)
    f = rng.normal(size=T).cumsum()
    lam = rng.uniform(0.5, 1.5, J)
    rows = []
    for i in range(J):
        for t in range(1, T + 1):
            d = int(i in starts and t >= starts[i])
            y = 5 + lam[i] * f[t - 1] + rng.normal(0, 0.2) + 2.0 * d
            rows.append({"unit": f"u{i}", "time": t, "y": y, "d": d})
    return pd.DataFrame(rows)


def _ss(df, **kw):
    return sp.staggered_synth(df, treatment="d", **KW, **kw)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(method="stacked"), "method must be 'separate' or 'pooled'"),
        (dict(se_method="bootstrap"), "se_method must be 'placebo' or 'jackknife'"),
        (dict(nu=1.5), "nu must be in"),
        (dict(penalization=-1.0), "penalization must be non-negative"),
        (dict(n_leads=0), "n_leads must be >= 1"),
        (dict(n_lags=0), "n_lags must be >= 1"),
    ],
)
def test_staggered_rejects_malformed_options(kwargs, message):
    with pytest.raises(MethodIncompatibility, match=message):
        _ss(_stag(), **kwargs)


def test_staggered_rejects_malformed_panels():
    df = _stag()
    with pytest.raises(MethodIncompatibility, match="column 'nope' not found"):
        sp.staggered_synth(df, outcome="nope", unit="unit", time="time", treatment="d")
    with pytest.raises(MethodIncompatibility, match="one row per"):
        _ss(pd.concat([df, df.iloc[:1]]))
    with pytest.raises(MethodIncompatibility, match="balanced panel"):
        _ss(df.iloc[1:])
    with pytest.raises(MethodIncompatibility, match="binary 0/1"):
        _ss(df.assign(d=df["d"] * 2))
    off = df.copy()
    off.loc[(off.unit == "u0") & (off.time == 12), "d"] = 0
    with pytest.raises(MethodIncompatibility, match="switches off"):
        _ss(off)
    with pytest.raises(DataInsufficient, match="treated in the first period"):
        _ss(_stag(starts={0: 1, 1: 10}))


def test_staggered_placebo_needs_three_never_treated_units():
    # two never-treated units: the in-space placebo distribution is empty,
    # so no standard error can be formed, while the point estimate is the
    # one the fit without placebos reports
    df = _stag(J=4, starts={0: 8, 1: 10})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_placebo = _ss(df, placebo=True)
        without = _ss(df, placebo=False)
    assert with_placebo.estimate == pytest.approx(without.estimate, rel=1e-12)
    assert with_placebo.model_info.get("n_placebos", 0) == 0


# --------------------------------------------------------------------- #
#  sp.synth(method="rcm")
# --------------------------------------------------------------------- #


def _rcm(df, **kw):
    args = dict(KW, treated_unit="u0", treatment_time=11, method="rcm")
    args.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.synth(df, **args)


def test_rcm_rejects_malformed_requests():
    df = _panel()
    with pytest.raises(MethodIncompatibility, match="selection must be one of"):
        _rcm(df, selection="ridge")
    with pytest.raises(MethodIncompatibility, match="is not a value of"):
        _rcm(df, treated_unit="zz")
    with pytest.raises(MethodIncompatibility, match="are not control units"):
        _rcm(df, donors=["u1", "zz"])
    with pytest.raises(MethodIncompatibility, match="placebo_units must list"):
        _rcm(df, placebo=True, placebo_units=["zz"])
    dup = df.copy()
    dup.loc[dup.unit == "u2", "y"] = dup.loc[dup.unit == "u1", "y"].to_numpy()
    with pytest.raises(MethodIncompatibility, match="collinear"):
        _rcm(dup)


def test_rcm_reports_insufficient_data():
    df = _panel()
    hole = df.copy()
    hole.loc[(hole.unit == "u2") & (hole.time == 3), "y"] = np.nan
    with pytest.raises(DataInsufficient, match="missing outcomes"):
        _rcm(hole)
    with pytest.raises(DataInsufficient, match="at least four pre-treatment"):
        _rcm(df, treatment_time=3)
    with pytest.raises(DataInsufficient, match="placebo_time leaves fewer than four"):
        _rcm(df, placebo_time=3)


def test_rcm_donors_argument_restricts_the_control_pool():
    df = _panel()
    pool = ["u1", "u2", "u3"]
    res = _rcm(df, donors=pool)
    ref = _rcm(df[df.unit.isin(["u0", *pool])])
    # restricting by argument or by subsetting is the same regression
    assert res.estimate == pytest.approx(ref.estimate, rel=1e-10)


# --------------------------------------------------------------------- #
#  Shared primitives (statspai.synth._core)
# --------------------------------------------------------------------- #


def test_least_norm_weights_solution_and_infeasible_starts():
    A = np.ones((1, 3))
    b = np.array([1.0])
    w = core._least_norm_weights(A, b, np.array([1.0, 0.0, 0.0]))
    # min ||w||^2 s.t. sum(w) = 1, w >= 0 is the uniform point
    np.testing.assert_allclose(w, np.full(3, 1 / 3), atol=1e-12)
    # a start with no positive weight is not a feasible point
    assert core._least_norm_weights(A, b, np.zeros(3)) is None
    # two parallel rows with different right-hand sides: no w satisfies both
    A2 = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert core._least_norm_weights(A2, np.array([1.0, 2.0]), np.full(3, 1 / 3)) is None


def test_polish_returns_the_input_when_it_cannot_certify_an_improvement():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 3))
    # (a) duplicated donors on the support: the restricted problem has no
    # unique solution, so the weights are handed back untouched
    Xd = np.column_stack([X[:, 0], X[:, 0], X[:, 2]])
    w = np.array([0.5, 0.5, 0.0])
    assert core.polish_simplex_weights(Xd @ w, Xd, w) is w
    # (b) the target is the excluded donor: the optimum on the support is
    # not optimal for the full problem, so the input is kept
    y = X[:, 2].copy()
    w = np.array([0.5, 0.5, 0.0])
    assert core.polish_simplex_weights(y, X, w) is w
    # (c) a support that contains the optimum is polished to it exactly
    y = X @ np.array([0.3, 0.7, 0.0])
    out = core.polish_simplex_weights(y, X, np.array([0.31, 0.69, 0.0]))
    np.testing.assert_allclose(out, [0.3, 0.7, 0.0], atol=1e-12)


def test_placebo_inversion_ci_linear_boundary_with_one_placebo():
    # Treated gap: pre (1, -1) so pre-MSPE = 1; post (1, 3).
    # One placebo: pre (1, -1), post (0, 0), weight 1 on the treated unit.
    # Under H0: effect = C the placebo's post gap is C and the treated one
    # is (1 - C, 3 - C); the placebo is at least as extreme iff
    #   C^2 >= ((1 - C)^2 + (3 - C)^2) / 2 = C^2 - 4C + 5  <=>  C >= 5/4.
    # With alpha = 0.6 and two units, p(C) >= alpha needs that one placebo.
    pre = np.array([True, True, False, False])
    lo, hi = core.placebo_inversion_ci(
        np.array([1.0, -1.0, 1.0, 3.0]),
        np.array([1.0, -1.0, 0.0, 0.0]),  # 1-D: a single placebo
        np.array([1.0]),
        pre,
        ~pre,
        alpha=0.6,
    )
    assert lo == pytest.approx(1.25, abs=1e-12)
    assert hi == np.inf


def test_placebo_inversion_ci_is_nan_when_no_constant_effect_is_accepted():
    # Same treated gap. Placebo A (post 3, 3; ratio^2 = 9) is at least as
    # extreme iff 1 + (2 - C)^2 <= 9; placebo B (post .5, .5; ratio^2 = .25)
    # never is, because the treated ratio^2 is at least 1 for every C.
    # alpha = 0.9 with three units needs both placebos: impossible.
    pre = np.array([True, True, False, False])
    g1 = np.array([1.0, -1.0, 1.0, 3.0])
    G = np.array([[1.0, 1.0], [-1.0, -1.0], [3.0, 0.5], [3.0, 0.5]])
    out = core.placebo_inversion_ci(g1, G, np.zeros(2), pre, ~pre, alpha=0.9)
    assert np.isnan(out[0]) and np.isnan(out[1])
    # needing one placebo (alpha = 0.5) gives placebo A's interval exactly
    lo, hi = core.placebo_inversion_ci(g1, G, np.zeros(2), pre, ~pre, alpha=0.5)
    assert lo == pytest.approx(2 - np.sqrt(8), abs=1e-12)
    assert hi == pytest.approx(2 + np.sqrt(8), abs=1e-12)


def test_predictor_weight_helpers_fall_back_to_equal_weights():
    # softmax of non-finite parameters is undefined -> equal weights
    np.testing.assert_array_equal(
        core._v_from_params(np.array([np.nan, np.nan]), 2), np.ones(2)
    )
    v = core._v_from_params(np.array([0.0, np.log(3.0)]), 2)
    np.testing.assert_allclose(v, [0.5, 1.5])  # tr(V) = K, ratio 1:3
    rng = np.random.default_rng(0)
    X1, X0 = rng.normal(size=2), rng.normal(size=(2, 5))
    Z1, Z0 = np.zeros(4), np.zeros((4, 5))
    # an outcome that is identically zero gives zero slopes: no information
    np.testing.assert_array_equal(core._regression_v_init(X1, X0, Z1, Z0), np.ones(2))
    with pytest.raises(DataInsufficient, match="no predictor explains"):
        core.regression_based_v(X1, X0, Z1, Z0)


def test_adh_solver_rejects_an_unknown_perfect_fit_rule():
    rng = np.random.default_rng(0)
    X1, X0 = rng.normal(size=2), rng.normal(size=(2, 5))
    Z1, Z0 = rng.normal(size=4), rng.normal(size=(4, 5))
    with pytest.raises(MethodIncompatibility, match="perfect_fit must be"):
        core.solve_synth_weights_adh(X1, X0, Z1, Z0, perfect_fit="exact")
