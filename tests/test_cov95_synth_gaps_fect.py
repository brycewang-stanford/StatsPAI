"""Coverage gaps in ``sp.fect`` and its cross-validation (``_fect_cv``).

One-way fixed-effect fits are checked against the group means they must
equal, the relative-period coding against hand-worked treatment histories
(reversals, late entry, gaps), and the argument checks against their
messages.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient

# ``statspai.synth.fect`` as an attribute is the function; import the module.
FECT = importlib.import_module("statspai.synth.fect")
CV = importlib.import_module("statspai.synth._fect_cv")

KW = dict(y="y", treat="d", unit="id", time="time")
NAN = np.nan


def _panel(seed=3, N=30, T=12):
    rng = np.random.default_rng(seed)
    first = np.zeros(N, dtype=int)
    first[10:20] = 6
    first[20:30] = 9
    alpha, xi = rng.normal(size=N), rng.normal(size=T)
    rows = []
    for i in range(N):
        for t in range(1, T + 1):
            d = int(first[i] > 0 and t >= first[i])
            y = 2.0 + alpha[i] + xi[t - 1] + 1.5 * d + rng.normal(scale=0.3)
            rows.append({"id": i + 1, "time": t, "y": y, "d": d})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def df():
    return _panel()


# ---------------------------------------------------------------------- #
#  One-way fixed effects
# ---------------------------------------------------------------------- #


def test_unit_only_fe_imputes_each_unit_by_its_untreated_mean(df):
    res = sp.fect(df, **KW, force="unit")
    tr, un = df[df["d"] == 1], df[df["d"] == 0]
    expected = (tr["y"] - tr["id"].map(un.groupby("id")["y"].mean())).mean()
    assert res.estimate == pytest.approx(expected, abs=1e-10)
    np.testing.assert_array_equal(res.model_info["xi"], 0.0)


def test_time_only_fe_imputes_each_period_by_its_untreated_mean(df):
    res = sp.fect(df, **KW, force="time")
    tr, un = df[df["d"] == 1], df[df["d"] == 0]
    expected = (tr["y"] - tr["time"].map(un.groupby("time")["y"].mean())).mean()
    assert res.estimate == pytest.approx(expected, abs=1e-10)
    np.testing.assert_array_equal(res.model_info["alpha"], 0.0)


@pytest.mark.parametrize("force", [0, 1, 2])
def test_y_demean_removes_exactly_the_requested_means(force):
    Y = np.random.default_rng(0).normal(size=(5, 4)) + 3.0
    YY, mu, alpha, xi = FECT._y_demean(Y, force)
    assert mu == pytest.approx(Y.mean())
    if force == 0:
        np.testing.assert_allclose(YY, Y - Y.mean())
        assert not alpha.any() and not xi.any()
    elif force == 1:
        np.testing.assert_allclose(YY.mean(axis=0), 0.0, atol=1e-14)
        np.testing.assert_allclose(alpha, Y.mean(axis=0))
        assert not xi.any()
    else:
        np.testing.assert_allclose(YY.mean(axis=1), 0.0, atol=1e-14)
        np.testing.assert_allclose(xi, Y.mean(axis=1))
        assert not alpha.any()


# ---------------------------------------------------------------------- #
#  Factor and thresholding primitives
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", [(4, 9), (9, 4)])
def test_panel_factor_is_the_rank_r_truncation_in_either_orientation(shape):
    E = np.random.default_rng(1).normal(size=shape)
    T, N = shape
    factor, lam, VNT = FECT._panel_factor(E, 2)
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    np.testing.assert_allclose(factor @ lam.T, (U[:, :2] * s[:2]) @ Vt[:2], atol=1e-10)
    np.testing.assert_allclose(np.diag(VNT), s[:2] ** 2 / (N * T), atol=1e-12)
    # fect's normalisation: F'F/T = I when T < N, L'L/N = I otherwise
    if T < N:
        np.testing.assert_allclose(factor.T @ factor / T, np.eye(2), atol=1e-10)
    else:
        np.testing.assert_allclose(lam.T @ lam / N, np.eye(2), atol=1e-10)


def test_hard_thresholding_keeps_large_singular_values_unshrunk():
    E = np.random.default_rng(2).normal(size=(6, 5))
    s = np.linalg.svd(E / 30, compute_uv=False)
    cut = float((s[1] + s[2]) / 2)  # keeps two components
    hard = FECT._panel_fe_soft(E, cut, hard=1)
    soft = FECT._panel_fe_soft(E, cut)
    s_hard = np.linalg.svd(hard / 30, compute_uv=False)
    s_soft = np.linalg.svd(soft / 30, compute_uv=False)
    np.testing.assert_allclose(s_hard, [s[0], s[1], 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(s_soft, [s[0] - cut, s[1] - cut, 0, 0, 0], atol=1e-12)


def test_iterative_initial_fit_gives_unidentified_effects_the_average():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(6, 4))
    II = np.ones((6, 4), dtype=int)
    II[:, 3] = 0  # unit 3 is never observed untreated
    II[5, :] = 0  # nor is anything in period 5
    Y0, beta0 = FECT._initial_fit_iterative(Y, None, II, 3)
    assert beta0.shape == (0,)
    # identified block: the two-way fit on the 5 x 3 untreated cells
    sub = Y[:5, :3]
    twoway = sub.mean(axis=1)[:, None] + sub.mean(axis=0)[None, :] - sub.mean()
    np.testing.assert_allclose(Y0[:5, :3], twoway, atol=1e-10)
    # the unidentified unit / period get the average unit / period effect
    np.testing.assert_allclose(Y0[:5, 3], twoway.mean(axis=1), atol=1e-10)
    np.testing.assert_allclose(Y0[5, :3], twoway.mean(axis=0), atol=1e-10)


# ---------------------------------------------------------------------- #
#  Relative-period coding (fect get_term, type = "on")
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "d, observed, expected",
    [
        # treated, switched off, treated again: each spell restarts at 1
        ([0, 0, 1, 1, 0, 0, 1], [1] * 7, [-1, 0, 1, 2, -1, 0, 1]),
        # trailing untreated periods lead to no onset
        ([0, 0, 1, 1, 0, 0, 1, 0], [1] * 8, [-1, 0, 1, 2, -1, 0, 1, NAN]),
        # treated from the first period: the opening spell has no onset
        ([1, 1, 0, 0, 1, 1, 0], [1] * 7, [NAN, NAN, -1, 0, 1, 2, NAN]),
        ([1, 1, 0, 0, 1, 1, 0, 1], [1] * 8, [NAN, NAN, -1, 0, 1, 2, 0, 1]),
        # unit enters the panel in period 3
        ([0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [NAN, NAN, 0, 1, 2]),
        # a gap carries the last observed status forward
        ([0, 0, 1, 1, 1], [1, 1, 0, 1, 1], [-2, -1, 0, 1, 2]),
        # a single observed period has no relative time
        ([0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [NAN] * 5),
        ([1], [1], [NAN]),
    ],
)
def test_relative_period_coding(d, observed, expected):
    out = FECT._get_term(np.array(d), np.array(observed))
    np.testing.assert_array_equal(out, np.array(expected, dtype=float))


# ---------------------------------------------------------------------- #
#  Argument checks and covariates
# ---------------------------------------------------------------------- #


def test_fect_argument_checks(df):
    with pytest.raises(ValueError, match="force must be 'none', 'unit', 'time'"):
        sp.fect(df, **KW, force="both")
    with pytest.raises(ValueError, match="vce must be None, 'bootstrap'"):
        sp.fect(df, **KW, vce="hc1")
    with pytest.raises(ValueError, match="column 'nope' not in data"):
        sp.fect(df, y="nope", treat="d", unit="id", time="time")
    bad = df.copy()
    bad.loc[0, "d"] = 2
    with pytest.raises(ValueError, match="treat='d' must be a 0/1 indicator"):
        sp.fect(bad, **KW)
    with pytest.raises(DataInsufficient, match="fewer untreated periods than min_t0"):
        sp.fect(df, **KW, min_t0=20)


def _with_covariate(df):
    out = df.copy()
    out["x"] = np.random.default_rng(0).normal(size=len(out))
    out["zero"] = 0.0
    out["y"] = out["y"] + 0.5 * out["x"]
    return out


def test_all_zero_covariate_is_dropped_with_a_missing_coefficient(df):
    dx = _with_covariate(df)
    one = sp.fect(dx, **KW, covariates=["x"])
    two = sp.fect(dx, **KW, covariates=["x", "zero"])
    beta = two.model_info["beta"]
    assert np.isnan(beta["zero"])
    assert beta["x"] == pytest.approx(one.model_info["beta"]["x"], abs=1e-12)
    assert beta["x"] == pytest.approx(0.5, abs=0.05)
    assert two.estimate == pytest.approx(one.estimate, abs=1e-12)


def test_min_t0_drops_units_together_with_their_covariates(df):
    dx = _with_covariate(df)
    with pytest.warns(UserWarning, match="10 unit.s. with fewer than min_t0=6"):
        res = sp.fect(dx, **KW, covariates=["x"], min_t0=6)
    assert res.model_info["n_units"] == 20
    # the same as fitting on the retained units only
    kept = dx[~dx["id"].between(11, 20)]
    direct = sp.fect(kept, **KW, covariates=["x"])
    assert res.estimate == pytest.approx(direct.estimate, abs=1e-12)


# ---------------------------------------------------------------------- #
#  Resampling edge cases
# ---------------------------------------------------------------------- #


def test_jackknife_skips_the_draw_that_leaves_no_treated_unit(df):
    small = df[df["id"].isin([1, 2, 11])]
    res = sp.fect(small, **KW, vce="jackknife")
    # three leave-one-out draws; dropping the only treated unit is unusable
    assert res.model_info["n_boot_success"] == 2
    assert np.isfinite(res.se) and res.se > 0


def test_fewer_than_two_resamples_reports_no_standard_error(df):
    pair = df[df["id"].isin([1, 11])]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = sp.fect(pair, **KW, vce="jackknife")
    assert any("fewer than two successful resamples" in str(w.message) for w in rec)
    assert res.se is None and res.ci is None


# ---------------------------------------------------------------------- #
#  Cross-validation
# ---------------------------------------------------------------------- #


def _cv(df, **kw):
    kw.setdefault("random_state", 0)
    kw.setdefault("k", 3)
    return sp.fect(df, **KW, method="ife", cv=True, **kw)


def test_cv_argument_checks(df):
    with pytest.raises(ValueError, match="cv_rule must be one of"):
        _cv(df, cv_rule="2se")
    with pytest.raises(ValueError, match="k must be >= 1"):
        _cv(df, k=0)
    with pytest.raises(ValueError, match="r_range must be .r_min, r_max."):
        _cv(df, r_range=(2, 1))
    with pytest.raises(ValueError, match="lambda_grid must be non-negative"):
        sp.fect(df, **KW, method="mc", cv=True, lambda_grid=[-1.0, 0.1], k=2)
    with pytest.raises(DataInsufficient, match="cv_prop leaves no cell"):
        _cv(df, cv_method="block", cv_prop=1e-4)
    with pytest.raises(DataInsufficient, match="Too few untreated cells of treated"):
        _cv(df, cv_method="treated_units", cv_prop=0.9)


def test_r_range_longer_than_two_is_read_as_its_extremes(df):
    res = _cv(df, r_range=(0, 1, 2))
    assert list(res.model_info["cv"]["grid"]) == [0, 1, 2]


def test_one_percent_rule_never_selects_more_factors_than_the_minimum(df):
    # no factors in the data: every rule should stay at r = 0, and the 1 %
    # rule cannot pick a larger r than the minimiser
    by_min = _cv(df, r_range=(0, 2), cv_rule="min")
    by_pct = _cv(df, r_range=(0, 2), cv_rule="1pct")
    assert by_pct.model_info["cv"]["cv_rule"] == "1pct"
    assert by_pct.model_info["r_cv"] <= by_min.model_info["r_cv"]
    assert by_pct.model_info["r_cv"] == 0
    direct = sp.fect(df, **KW, method="fe")
    assert by_pct.estimate == pytest.approx(direct.estimate, abs=1e-6)


def test_treated_units_holdout_scores_only_treated_units_cells(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _cv(df, r_range=(0, 1), cv_method="treated_units")
    cv = res.model_info["cv"]
    assert cv["cv_method"] == "treated_units"
    assert all(n > 0 for n in cv["n_scored_per_fold"])
    # 20 treated units x at most 8 untreated periods, a tenth held out
    assert max(cv["n_holdout_per_fold"]) <= 0.1 * 20 * 8 * 2


def test_apply_cv_rule_on_explicit_scores():
    means = np.array([1.000, 1.005, 0.999, np.nan])
    ses = np.array([0.1, 0.1, 0.0005, np.nan])
    assert CV._apply_cv_rule(means, ses, "min") == 2
    assert CV._apply_cv_rule(means, ses, "1pct") == 0  # 1.000 <= 0.999 * 1.01
    assert CV._apply_cv_rule(np.array([1.02, 1.005, 0.999]), ses[:3], "1pct") == 1
    assert CV._apply_cv_rule(np.full(3, np.nan), np.full(3, np.nan), "1se") is None


def test_score_refuses_an_empty_holdout():
    with pytest.raises(DataInsufficient, match="No residuals collected"):
        CV._score(np.zeros(0), [], None)
