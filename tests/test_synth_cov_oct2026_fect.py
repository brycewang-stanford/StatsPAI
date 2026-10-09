"""Coverage round (Oct 2026): ``sp.fect`` and its cross-validation folds.

What is asserted:

* fixed-effect specifications other than two-way have closed forms
  (grand / unit / period means of the untreated cells), so the ATT is
  compared with that closed form;
* a noiseless one-factor panel is recovered exactly by ``method='ife'``;
* the relative-period index of treatment histories with reversals,
  late entry and gaps is compared with hand-derived answers;
* the fold builders are checked against the invariants their docstrings
  state (held-out cells are untreated cells, scored cells are a subset
  of the hidden ones, every unit keeps ``min_t0`` untreated periods);
* invalid inputs raise the documented exception.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient
from statspai.synth import _fect_cv as cv

# ``statspai.synth.fect`` is the function; the module sits behind it.
fx = importlib.import_module("statspai.synth.fect")


def _panel(seed=0, N=12, T=10, n_treated=4, onset=7, noise=0.3, factor=0.0):
    rng = np.random.default_rng(seed)
    alpha = rng.normal(size=N)
    xi = rng.normal(size=T)
    f = rng.normal(size=T)
    lam = rng.normal(size=N)
    rows = []
    for i in range(N):
        for t in range(1, T + 1):
            d = int(i < n_treated and t >= onset)
            y = 2.0 + alpha[i] + xi[t - 1] + factor * f[t - 1] * lam[i]
            y += 1.5 * d + noise * rng.normal()
            rows.append({"id": i, "time": t, "y": y, "d": d, "x": rng.normal()})
    return pd.DataFrame(rows)


def _fect(df, **kw):
    kw.setdefault("tol", 1e-12)
    return sp.fect(df, y="y", treat="d", unit="id", time="time", **kw)


# --------------------------------------------------------------------- #
#  Closed forms for the simpler fixed-effect specifications
# --------------------------------------------------------------------- #


def test_force_none_att_is_treated_mean_minus_untreated_grand_mean():
    df = _panel()
    res = _fect(df, method="fe", force="none")
    expected = df.loc[df.d == 1, "y"].mean() - df.loc[df.d == 0, "y"].mean()
    # the counterfactual is the grand mean of the untreated cells; the
    # only error is the fixed-point tolerance (1e-12), so 1e-9 is ample
    assert res.estimate == pytest.approx(expected, abs=1e-9)


def test_force_none_with_one_factor_recovers_the_effect_without_fixed_effects():
    # y = 2 + f_t * lambda_i + 1.5 d, no unit or period effects, no noise.
    # With force="none" the model is a grand mean plus a rank-r surface
    # extracted from the grand-mean-centred matrix, so the factor is
    # centred here to keep the DGP inside that model.
    rng = np.random.default_rng(9)
    N, T = 8, 12
    f, lam = rng.normal(size=T), rng.normal(size=N)
    f = f - f.mean()
    rows = [
        {
            "id": i,
            "time": t + 1,
            "d": int(i < 2 and t >= 8),
            "y": 2.0 + f[t] * lam[i] + 1.5 * int(i < 2 and t >= 8),
        }
        for i in range(N)
        for t in range(T)
    ]
    res = _fect(pd.DataFrame(rows), method="ife", r=1, force="none")
    # exact rank-one untreated surface: only EM convergence error remains
    assert res.estimate == pytest.approx(1.5, abs=1e-6)


def test_force_unit_att_is_the_mean_deviation_from_each_units_untreated_mean():
    df = _panel()
    res = _fect(df, method="fe", force="unit")
    unit_mean = df[df.d == 0].groupby("id")["y"].mean()
    tr = df[df.d == 1]
    expected = (tr["y"] - tr["id"].map(unit_mean)).mean()
    # closed form up to the fixed-point tolerance
    assert res.estimate == pytest.approx(expected, abs=1e-9)


def test_force_time_att_is_the_mean_deviation_from_each_periods_untreated_mean():
    df = _panel()
    res = _fect(df, method="fe", force="time")
    time_mean = df[df.d == 0].groupby("time")["y"].mean()
    tr = df[df.d == 1]
    expected = (tr["y"] - tr["time"].map(time_mean)).mean()
    # closed form up to the fixed-point tolerance
    assert res.estimate == pytest.approx(expected, abs=1e-9)


def test_ife_recovers_the_effect_exactly_on_a_noiseless_one_factor_panel():
    # N <= T exercises the "more periods than units" factor extraction.
    df = _panel(seed=4, N=8, T=12, n_treated=2, onset=9, noise=0.0, factor=1.0)
    res = _fect(df, method="ife", r=1)
    assert res.model_info["factors"].shape == (12, 1)
    # no noise and the right rank: the untreated cells determine the
    # factor model, so the ATT is the true 1.5 up to EM convergence
    assert res.estimate == pytest.approx(1.5, abs=1e-6)
    fe = _fect(df, method="fe")
    assert abs(fe.estimate - 1.5) > 1e-3  # two-way FE alone is biased here


def test_hard_thresholding_keeps_singular_values_above_the_cutoff_unshrunk():
    rng = np.random.default_rng(1)
    E = rng.normal(size=(6, 5))
    s = np.linalg.svd(E / E.size, compute_uv=False)
    cut = 0.5 * (s[1] + s[2])  # keeps exactly the two largest
    hard = fx._panel_fe_soft(E, cut, hard=1)
    soft = fx._panel_fe_soft(E, cut)
    s_hard = np.linalg.svd(hard / E.size, compute_uv=False)
    s_soft = np.linalg.svd(soft / E.size, compute_uv=False)
    # SVD round-off only
    np.testing.assert_allclose(s_hard[:2], s[:2], rtol=1e-10)
    np.testing.assert_allclose(s_hard[2:], 0.0, atol=1e-12)
    np.testing.assert_allclose(s_soft[:2], s[:2] - cut, rtol=1e-10)


def test_all_zero_covariate_gets_a_missing_coefficient_and_the_same_att():
    df = _panel(seed=2)
    df["zero"] = 0.0
    one = _fect(df, method="fe", covariates=["x"])
    two = _fect(df, method="fe", covariates=["x", "zero"])
    beta = np.asarray(two.model_info["beta"], dtype=float)
    assert beta.shape == (2,)
    assert np.isnan(beta[1])
    # the all-zero column is left out, so this is the one-covariate fit
    # (same iterations; differences are floating-point order only)
    assert beta[0] == pytest.approx(float(np.asarray(one.model_info["beta"])[0]))
    assert two.estimate == pytest.approx(one.estimate, abs=1e-10)


# --------------------------------------------------------------------- #
#  Relative-period index
# --------------------------------------------------------------------- #

NAN = np.nan


@pytest.mark.parametrize(
    "d, ii, expected",
    [
        # on, off, on again: each spell restarts the count
        ([0, 0, 1, 1, 0, 0, 1], [1] * 7, [-1, 0, 1, 2, -1, 0, 1]),
        # treated from the first period: no onset to count from
        ([1, 1, 0, 0, 1], [1] * 5, [NAN, NAN, -1, 0, 1]),
        # starts treated, three switches
        ([1, 0, 1, 1, 0, 0], [1] * 6, [NAN, 0, 1, 2, NAN, NAN]),
        # ends untreated after a spell
        ([0, 1, 1, 0], [1] * 4, [0, 1, 2, NAN]),
        # unit enters the panel late
        ([0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1], [NAN, NAN, -1, 0, 1, 2]),
        # a missing period inside the history carries the status forward
        ([0, 0, 0, 1, 1], [1, 1, 0, 1, 1], [-2, -1, 0, 1, 2]),
        # a single observed period
        ([0, 0, 1], [0, 0, 1], [NAN, NAN, NAN]),
    ],
)
def test_relative_period_index_matches_hand_derived_values(d, ii, expected):
    out = fx._get_term(np.array(d, dtype=float), np.array(ii))
    np.testing.assert_array_equal(out, np.array(expected, dtype=float))


# --------------------------------------------------------------------- #
#  Input validation and unit filtering
# --------------------------------------------------------------------- #


def test_invalid_force_vce_column_and_treatment_values_are_rejected():
    df = _panel()
    with pytest.raises(ValueError, match="force must be"):
        _fect(df, force="both")
    with pytest.raises(ValueError, match="vce must be"):
        _fect(df, vce="hc1")
    with pytest.raises(ValueError, match="'nope' not in data"):
        _fect(df, covariates=["nope"])
    bad = df.assign(d=df["d"] * 2)
    with pytest.raises(ValueError, match="must be a 0/1 indicator"):
        _fect(bad)


def test_all_treated_units_below_min_t0_is_data_insufficient():
    df = _panel(onset=3)  # two untreated periods per treated unit
    with pytest.raises(DataInsufficient, match="fewer untreated periods"):
        _fect(df, method="ife", r=1)  # min_t0 defaults to 5 for ife


def test_short_history_unit_is_dropped_with_its_covariate_rows():
    df = _panel(seed=5, N=12, T=10)
    # unit 0 becomes treated from period 2: one untreated period only
    df.loc[(df.id == 0) & (df.time >= 2), "d"] = 1
    with pytest.warns(UserWarning, match="1 unit.s. with fewer than min_t0=5"):
        res = _fect(df, method="ife", r=1, covariates=["x"], tol=1e-8)
    assert res.model_info["dropped_units"] == [0]
    assert res.model_info["n_units"] == 11
    ref = _fect(df[df.id != 0], method="ife", r=1, covariates=["x"], tol=1e-8)
    # dropping the unit beforehand is the same estimation problem
    assert res.estimate == pytest.approx(ref.estimate, abs=1e-10)


def test_jackknife_skips_the_resample_that_removes_the_only_treated_unit():
    df = _panel(seed=6, N=9, T=10, n_treated=1)
    res = _fect(df, method="fe", vce="jackknife", tol=1e-8)
    # nine leave-one-out samples, one of which has no treated unit
    assert res.model_info["n_boot_success"] == 8
    assert res.se > 0


def test_a_single_bootstrap_draw_gives_no_standard_error_and_says_so():
    df = _panel()
    with pytest.warns(RuntimeWarning, match="fewer than two successful resamples"):
        res = _fect(df, method="fe", vce="bootstrap", n_boot=1, seed=0, tol=1e-8)
    assert res.se is None and res.ci is None
    assert res.model_info["n_boot_success"] == 1


# --------------------------------------------------------------------- #
#  Cross-validation: argument checks
# --------------------------------------------------------------------- #


def _cv(df, **kw):
    kw.setdefault("method", "ife")
    kw.setdefault("k", 3)
    kw.setdefault("tol", 1e-3)
    kw.setdefault("random_state", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fect(df, cv=True, **kw)


def test_cv_argument_checks():
    df = _panel(N=14, T=12, onset=9)
    with pytest.raises(ValueError, match="cv_rule must be one of"):
        _cv(df, cv_rule="2se")
    with pytest.raises(ValueError, match="k must be >= 1"):
        _cv(df, k=0)
    with pytest.raises(ValueError, match="r_range must be"):
        _cv(df, r_range=(2, 1))
    with pytest.raises(ValueError, match="lambda_grid must be non-negative"):
        _cv(df, method="mc", lambda_grid=[0.1, -0.1])


def test_cv_r_range_given_as_a_list_of_candidates_spans_min_to_max():
    df = _panel(seed=7, N=14, T=12, onset=9, factor=1.0, noise=0.05)
    res = _cv(df, r_range=(0, 2, 1), cv_rule="1pct")
    info = res.model_info["cv"]
    assert info["grid"] == [0.0, 1.0, 2.0]
    table = info["table"]
    mspe = table["MSPE"].to_numpy()
    # the 1 % rule: the smallest r whose score is within 1 % of the best
    expected = int(np.flatnonzero(mspe <= 1.01 * mspe.min())[0])
    assert res.model_info["r_cv"] == expected
    # a strong factor makes the factor-free model clearly worse
    assert expected >= 1 and mspe[0] > 5 * mspe[expected]


def test_cv_over_treated_units_only_selects_from_the_requested_grid():
    df = _panel(seed=8, N=14, T=12, n_treated=6, onset=9, factor=1.0, noise=0.05)
    res = _cv(df, r_range=(0, 1), cv_method="treated_units", cv_rule="min")
    info = res.model_info["cv"]
    mspe = info["table"]["MSPE"].to_numpy()
    assert info["grid"] == [0.0, 1.0]
    # rule "min" returns the grid point with the smallest score
    assert res.model_info["r_cv"] == int(np.argmin(mspe))


# --------------------------------------------------------------------- #
#  Cross-validation: fold builders
# --------------------------------------------------------------------- #


def _masks(N=8, T=12, n_treated=3, onset=8):
    D = np.zeros((T, N))
    D[onset:, :n_treated] = 1.0
    II = (D == 0).astype(int)
    return II, D


def _check_block_folds(II, folds, count, min_t0):
    flat = II.ravel(order="F")
    for cv_id, est_id in folds:
        assert len(cv_id) == count
        assert flat[cv_id].all()  # only untreated cells are hidden
        assert set(est_id.tolist()) <= set(cv_id.tolist())
        kept = flat.copy()
        kept[cv_id] = 0
        kept = kept.reshape(II.shape, order="F")
        assert (kept.sum(axis=1) >= 1).all()
        assert (kept.sum(axis=0) >= min_t0).all()


@pytest.mark.parametrize("cv_nobs", [1, 2, 3])
def test_block_folds_hide_the_requested_share_of_untreated_cells(cv_nobs):
    II, D = _masks()
    rng = np.random.default_rng(0)
    folds = cv._block_folds(II, D, 4, 0.1, cv_nobs, False, 0, 2, rng)
    assert len(folds) == 4
    _check_block_folds(II, folds, int(np.floor(II.sum() * 0.1)), 2)


def test_block_folds_over_treated_units_hide_only_their_cells():
    II, D = _masks()
    rng = np.random.default_rng(1)
    folds = cv._block_folds(II, D, 3, 0.05, 2, True, 0, 2, rng)
    count = int(np.floor(II.sum() * 0.05))
    _check_block_folds(II, folds, count, 2)
    T = II.shape[0]
    for cv_id, _ in folds:
        assert set((cv_id // T).tolist()) <= {0, 1, 2}  # the treated units


def test_block_folds_over_treated_units_need_enough_pre_treatment_cells():
    II, D = _masks(n_treated=1, onset=2)  # two untreated cells to draw from
    with pytest.raises(DataInsufficient, match="Too few untreated cells"):
        cv._block_folds(II, D, 2, 0.1, 2, True, 0, 1, np.random.default_rng(0))


def test_block_folds_reject_a_share_that_holds_out_nothing():
    II, D = _masks()
    with pytest.raises(DataInsufficient, match="no cell to hold out"):
        cv._block_folds(II, D, 2, 1e-4, 2, False, 0, 2, np.random.default_rng(0))


def test_block_folds_restore_units_that_cannot_spare_a_cell():
    # Three units are treated after exactly min_t0 = 4 periods, so hiding
    # any of their cells violates the constraint; half the untreated cells
    # are requested, so every draw touches them.  After 200 failed draws
    # the offending units get their cells back and a warning is issued.
    T, N, min_t0 = 12, 6, 4
    D = np.zeros((T, N))
    D[min_t0:, :3] = 1.0
    II = (D == 0).astype(int)
    with pytest.warns(UserWarning, match="too few pre-treatment observations"):
        folds = cv._block_folds(
            II, D, 1, 0.5, 2, False, 0, min_t0, np.random.default_rng(0)
        )
    cv_id, est_id = folds[0]
    assert len(cv_id) > 0
    flat = II.ravel(order="F")
    kept = flat.copy()
    kept[cv_id] = 0
    kept = kept.reshape((T, N), order="F")
    # the restored design satisfies the per-unit constraint again
    assert (kept.sum(axis=0) >= min_t0).all()
    assert not set((cv_id // T).tolist()) & {0, 1, 2}
    assert set(est_id.tolist()) <= set(cv_id.tolist())


def test_rolling_folds_need_a_unit_with_min_t0_plus_cv_nobs_periods():
    II, D = _masks(N=4, T=6, n_treated=4, onset=3)
    with pytest.raises(DataInsufficient, match="rolling cross-validation"):
        cv._rolling_folds(II, D, 2, 3, 1, 0.5, 5, np.random.default_rng(0))


def test_score_and_selection_rule_edge_cases():
    with pytest.raises(DataInsufficient, match="No residuals"):
        cv._score(np.zeros(0), [], None)
    nan = np.full(3, np.nan)
    assert cv._apply_cv_rule(nan, nan, "1se") is None
    means = np.array([1.0, 0.504, 0.5, 0.7])
    ses = np.array([0.1, 0.001, 0.001, 0.1])
    assert cv._apply_cv_rule(means, ses, "min") == 2
    assert cv._apply_cv_rule(means, ses, "1pct") == 1  # 0.504 <= 0.505
    assert cv._apply_cv_rule(means, ses, "1se") == 2  # 0.504 > 0.501
