"""Coverage gaps in ``synth.bsts`` (argument checks, Kalman filter with
missing observations), ``synth.ttest`` (loud failures, degenerate variance)
and the shared helpers of ``synth._core`` (placebo-inversion interval,
predictor-weight fallbacks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.synth import _core
from statspai.synth.bsts import (
    _coerce_column_list,
    _kalman_filter,
    _neg_log_likelihood,
    causal_impact,
)

# ---------------------------------------------------------------------- #
#  causal_impact (wide-format BSTS)
# ---------------------------------------------------------------------- #


def _wide(seed=0, n=60, effect=10.0):
    rng = np.random.default_rng(seed)
    x1 = 100 + np.cumsum(rng.normal(0, 1, n))
    y = 1.2 * x1 + rng.normal(0, 1, n)
    y[40:] += effect
    return pd.DataFrame({"y": y, "x1": x1})


def _ci(data=None, **kw):
    args = dict(pre_period=(0, 39), post_period=(40, 59), n_simulations=50, seed=0)
    args.update(kw)
    return causal_impact(_wide() if data is None else data, **args)


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"outcome": 5}, "`outcome` must be a non-empty column name"),
        ({"model": 3}, "`model` must be a string option"),
        ({"model": "  "}, "`model` must be a non-empty string option"),
        ({"alpha": True}, "`alpha` must be a number in \\(0, 1\\)"),
        ({"alpha": "abc"}, "`alpha` must be a number in \\(0, 1\\)"),
        ({"alpha": 1.5}, "`alpha` must be in \\(0, 1\\)"),
        ({"n_simulations": True}, "`n_simulations` must be an integer >= 2"),
        ({"n_simulations": "x"}, "`n_simulations` must be an integer >= 2"),
        ({"n_simulations": 1}, "`n_simulations` must be an integer >= 2"),
        ({"covariates": 5}, "`covariates` must be a column name or list"),
        ({"covariates": ["x1", 3]}, "only non-empty string column names"),
        ({"pre_period": 5}, "`pre_period` must be a two-element period pair"),
        ({"pre_period": (1, 2, 3)}, "`pre_period` must be a two-element"),
        ({"pre_period": ("a", "b")}, "must be comparable with the index"),
    ],
)
def test_causal_impact_argument_checks(kwargs, fragment):
    with pytest.raises(MethodIncompatibility, match=fragment):
        _ci(**kwargs)


def test_required_column_list_cannot_be_empty():
    with pytest.raises(MethodIncompatibility, match="at least one column name"):
        _coerce_column_list([], "covariates")
    assert _coerce_column_list([], "covariates", allow_empty=True) == []


def test_covariates_as_a_single_name_equals_the_default_controls():
    by_default = _ci()
    by_name = _ci(covariates="x1")
    assert by_name.estimate == by_default.estimate
    assert by_name.model_info["n_covariates"] == 1
    assert by_default.estimate == pytest.approx(10.0, abs=1.5)


def test_no_covariates_is_a_pure_local_level_model():
    res = _ci(covariates=[])
    assert res.model_info["n_covariates"] == 0
    assert len(res.model_info["regression_coefficients"]) == 0
    # a local level forecasts no drift: the counterfactual stays near the
    # end of the pre-period (y[39] = 114.7 here), so the estimated effect is
    # the post-period mean minus that level
    cf = np.asarray(res.model_info["counterfactual_mean"])[-20:]
    y = _wide()["y"].to_numpy()
    assert abs(cf.mean() - y[35:40].mean()) < 2.0
    assert res.estimate == pytest.approx(y[40:].mean() - cf.mean(), abs=1e-8)


def test_causal_impact_refuses_infinite_outcomes_and_unobserved_covariates():
    bad = _wide()
    bad.loc[45, "y"] = np.inf
    with pytest.raises(DataInsufficient, match="non-finite values"):
        _ci(bad)
    empty = _wide()
    empty.loc[:39, "x1"] = np.nan
    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        with pytest.raises(DataInsufficient, match="observed pre-period values"):
            _ci(empty)


def test_kalman_filter_skips_the_update_at_a_missing_observation():
    y = np.array([1.0, 1.2, np.nan, 1.1, 0.9])
    X = np.zeros((5, 0))
    kf = _kalman_filter(y, X, np.zeros(0), sigma_obs=0.5, sigma_level=0.2)
    # no information at t = 2: filtered = predicted, level carried forward
    np.testing.assert_array_equal(kf["filtered_state"][2], kf["predicted_state"][2])
    np.testing.assert_array_equal(kf["filtered_cov"][2], kf["predicted_cov"][2])
    np.testing.assert_allclose(kf["filtered_state"][2], kf["filtered_state"][1])
    # the missing period adds nothing to the likelihood: dropping it and
    # doubling the level innovation variance over the gap gives the same value
    assert np.isfinite(kf["log_likelihood"])
    assert kf["filtered_cov"][2, 0, 0] == pytest.approx(
        kf["filtered_cov"][1, 0, 0] + 0.2**2
    )


def test_negative_log_likelihood_is_a_finite_penalty_on_overflow():
    y = np.array([1.0, 1.2, 0.8, 1.1])
    X = np.zeros((4, 0))
    with np.errstate(all="ignore"):  # squared innovations overflow
        degenerate = _neg_log_likelihood(
            np.log([0.5, 0.2]), y * 1e200, X, np.zeros(0), False
        )
    assert degenerate == 1e12
    regular = _neg_log_likelihood(np.log([0.5, 0.2]), y, X, np.zeros(0), False)
    assert np.isfinite(regular) and regular < 1e12


# ---------------------------------------------------------------------- #
#  synth_ttest
# ---------------------------------------------------------------------- #


def _long(seed=1, n_donors=5, n_t=30, t0=24, effect=3.0):
    rng = np.random.default_rng(seed)
    rows = [
        (f"d{j}", t, rng.normal() + 0.1 * t * j)
        for j in range(n_donors)
        for t in range(n_t)
    ]
    df = pd.DataFrame(rows, columns=["u", "t", "y"])
    tr = df[df["u"] == "d0"].copy()
    tr["u"] = "tr"
    tr["y"] = 0.5 * tr["y"].to_numpy() + 0.5 * df.loc[df["u"] == "d1", "y"].to_numpy()
    tr.loc[tr["t"] >= t0, "y"] += effect
    return pd.concat([df, tr], ignore_index=True)


def _tt(df, **kw):
    args = dict(outcome="y", unit="u", time="t", treated_unit="tr", treatment_time=24)
    args.update(kw)
    return sp.synth_ttest(df, **args)


def test_synth_ttest_recovers_the_effect_of_a_convex_combination():
    res = _tt(_long())
    assert res.estimate == pytest.approx(3.0, abs=1e-8)
    assert res.model_info["df"] == 2


def test_synth_ttest_argument_checks():
    df = _long()
    with pytest.raises(MethodIncompatibility, match="n_folds must be at least 2"):
        _tt(df, n_folds=1)
    with pytest.raises(MethodIncompatibility, match="columns not found.*'nope'"):
        _tt(df, outcome="nope")
    with pytest.raises(MethodIncompatibility, match="found duplicates"):
        _tt(pd.concat([df, df.iloc[[0]]], ignore_index=True))
    with pytest.raises(MethodIncompatibility, match="'zz' is not a value of 'u'"):
        _tt(df, treated_unit="zz")


def test_synth_ttest_data_requirements():
    df = _long()
    with pytest.raises(DataInsufficient, match="no post-treatment period"):
        _tt(df, treatment_time=99)
    with pytest.raises(DataInsufficient, match="2 pre-periods are too few"):
        _tt(df, treatment_time=2)
    gap = df.copy()
    gap.loc[(gap["u"] == "tr") & (gap["t"] == 4), "y"] = np.nan
    with pytest.raises(DataInsufficient, match="treated unit has missing outcomes"):
        _tt(gap)
    none = df.copy()
    none.loc[(none["u"] != "tr") & (none["t"] == 4), "y"] = np.nan
    with pytest.raises(DataInsufficient, match="no donor has complete outcomes"):
        _tt(none)


def test_synth_ttest_drops_incomplete_donors_with_a_warning():
    df = _long()
    full = _tt(df[df["u"] != "d4"])
    holed = df.copy()
    holed.loc[(holed["u"] == "d4") & (holed["t"] == 7), "y"] = np.nan
    with pytest.warns(UserWarning, match=r"Dropped 1 donor\(s\) with missing"):
        res = _tt(holed)
    # the same as never having had that donor
    assert res.estimate == pytest.approx(full.estimate, abs=1e-12)
    assert res.se == pytest.approx(full.se, abs=1e-12)


def test_synth_ttest_zero_variance_gives_undefined_test():
    # one donor, integer outcomes, treated = donor + 3 after treatment:
    # every fold estimates exactly 3, so the t statistic is undefined
    t = np.arange(30)
    donor = pd.DataFrame({"u": "d0", "t": t, "y": (t % 7).astype(float)})
    treated = donor.assign(u="tr")
    treated.loc[treated["t"] >= 24, "y"] += 3.0
    res = _tt(pd.concat([donor, treated], ignore_index=True))
    assert res.estimate == 3.0
    assert res.se == 0.0
    assert np.isnan(res.pvalue)
    assert res.ci == (3.0, 3.0)


# ---------------------------------------------------------------------- #
#  _core: placebo-inversion interval
# ---------------------------------------------------------------------- #


def _pvalue_at(c, g1, G, w, pre, post):
    """p(C) by definition: rank of the treated post/pre RMSPE ratio."""
    r1 = np.mean((g1[post] - c) ** 2) / np.mean(g1[pre] ** 2)
    rj = np.mean((G[post] + w * c) ** 2, axis=0) / np.mean(G[pre] ** 2, axis=0)
    return (1 + int(np.sum(rj >= r1))) / (G.shape[1] + 1)


def test_inversion_interval_is_unbounded_with_a_single_placebo():
    pre = np.array([True, True, False, False])
    g1 = np.array([1.0, -1.0, 4.0, 4.0])
    one = np.array([1.0, -1.0, 0.5, 0.5])  # 1-D: a single placebo unit
    ci = _core.placebo_inversion_ci(g1, one, np.zeros(1), pre, ~pre, alpha=0.05)
    assert ci == (-np.inf, np.inf)  # smallest attainable p is 1/2 > alpha


def test_inversion_interval_linear_boundary_matches_the_definition():
    # placebo with weight one on the treated unit and the same pre-fit:
    # the quadratic terms cancel and the boundary is a single (linear) root
    pre = np.array([True, True, False, False])
    post = ~pre
    g1 = np.array([1.0, -1.0, 1.0, 1.0])
    G = np.array([[1.0], [-1.0], [0.0], [0.0]])
    w = np.array([1.0])
    lo, hi = _core.placebo_inversion_ci(g1, G, w, pre, post, alpha=0.6)
    # |C| >= |1 - C|  <=>  C >= 1/2
    assert lo == pytest.approx(0.5) and hi == np.inf
    assert _pvalue_at(0.5 + 1e-6, g1, G, w, pre, post) >= 0.6
    assert _pvalue_at(0.5 - 1e-6, g1, G, w, pre, post) < 0.6


def test_inversion_interval_is_empty_when_no_constant_effect_fits():
    # two placebos accept disjoint sets of C ([0.5, 1.5] and C <= 0.25); at
    # alpha = 0.9 both must be at least as extreme as the treated unit
    pre = np.array([True, True, False, False])
    post = ~pre
    g1 = np.array([1.0, -1.0, 0.0, 2.0])
    G = np.array(
        [[1.0, 1.0], [-1.0, -1.0], [np.sqrt(1.25), -1.5], [np.sqrt(1.25), -1.5]]
    )
    w = np.array([0.0, 1.0])
    for c in np.linspace(-5, 5, 201):
        assert _pvalue_at(c, g1, G, w, pre, post) < 0.9
    ci = _core.placebo_inversion_ci(g1, G, w, pre, post, alpha=0.9)
    assert np.isnan(ci[0]) and np.isnan(ci[1])


# ---------------------------------------------------------------------- #
#  _core: predictor weights
# ---------------------------------------------------------------------- #


def test_v_from_params_falls_back_to_equal_weights_on_overflow():
    with np.errstate(invalid="ignore"):  # inf - inf inside the softmax
        v_bad = _core._v_from_params(np.array([np.inf, 0.0, 0.0]), 3)
    np.testing.assert_array_equal(v_bad, np.ones(3))
    v = _core._v_from_params(np.array([0.0, np.log(2.0), 0.0]), 3)
    np.testing.assert_allclose(v, [0.75, 1.5, 0.75])  # tr(V) = K


def test_regression_v_init_is_equal_when_outcomes_carry_no_signal():
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=(3, 6))
    X1 = rng.normal(size=3)
    zeros = np.zeros((4, 6))
    np.testing.assert_array_equal(
        _core._regression_v_init(X1, X0, np.zeros(4), zeros), np.ones(3)
    )


def test_regression_based_v_refuses_flat_outcomes():
    rng = np.random.default_rng(1)
    X0 = rng.normal(size=(2, 8))
    X1 = rng.normal(size=2)
    with pytest.raises(DataInsufficient, match="no predictor explains"):
        _core.regression_based_v(X1, X0, np.zeros(5), np.zeros((5, 8)))


def test_adh_solver_rejects_unknown_perfect_fit_rule():
    rng = np.random.default_rng(2)
    with pytest.raises(MethodIncompatibility, match="perfect_fit must be"):
        _core.solve_synth_weights_adh(
            rng.normal(size=2),
            rng.normal(size=(2, 5)),
            rng.normal(size=4),
            rng.normal(size=(4, 5)),
            perfect_fit="strict",
        )
