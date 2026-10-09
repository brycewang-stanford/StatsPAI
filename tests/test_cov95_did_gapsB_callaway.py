"""Coverage gaps (batch B) for statspai.did.callaway_santanna.

Degenerate ATT(g, t) cells must come back as "not estimated"
(``att = 0, se = inf`` with a zero influence function) rather than as a
number; failed covariate regressions must fall back to the unadjusted
estimator with a ConvergenceWarning; and a handful of validation branches
on the repeated-cross-section / unbalanced-panel route.
"""

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
import statspai.did.callaway_santanna  # noqa: F401
from statspai.exceptions import ConvergenceWarning, DataInsufficient

C = sys.modules["statspai.did.callaway_santanna"]
K = dict(y="y", g="g", t="t", i="i")


def make_panel(seed=0, cohorts=(4, 6, 0), n_per=25, T=8):
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for g in cohorts:
        for _ in range(n_per):
            u_fe = rng.normal()
            xv = rng.normal()
            for t in range(1, T + 1):
                te = max(0, t - g + 1) if g > 0 else 0
                yv = u_fe + 0.3 * t + te + rng.normal() * 0.5 + 0.5 * xv
                rows.append(
                    {"i": uid, "t": t, "y": yv, "g": g, "x1": xv, "cl": uid % 10}
                )
            uid += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


def _not_estimated(out, n):
    att, se, inf = out
    return att == 0.0 and np.isinf(se) and inf.shape == (n,) and not inf.any()


# ----------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------
def test_allow_unbalanced_panel_is_inert_for_repeated_cross_sections(panel):
    with pytest.warns(UserWarning, match="has no effect when panel=False"):
        a = sp.callaway_santanna(panel, **K, panel=False, allow_unbalanced_panel=True)
    b = sp.callaway_santanna(panel, **K, panel=False)
    assert a.estimate == b.estimate and a.se == b.se


def test_single_cohort_treated_from_the_first_period_leaves_no_periods(panel):
    df = panel[panel["g"] == 4].assign(g=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(DataInsufficient, match="No periods remain before"):
            sp.callaway_santanna(df, **K, control_group="notyettreated")


def test_repeated_cross_section_weights_summing_to_zero(panel):
    with pytest.raises(DataInsufficient, match="weights column 'w' sums to zero"):
        sp.callaway_santanna(panel.assign(w=0.0), **K, panel=False, weights="w")


def test_unbalanced_panel_clustered_multiplier_bootstrap(panel):
    ub = panel.drop(index=[3, 77, 190]).reset_index(drop=True)
    kw = dict(allow_unbalanced_panel=True, bstrap=True, random_state=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clustered = sp.callaway_santanna(ub, **K, clustervars="cl", **kw)
        again = sp.callaway_santanna(ub, **K, clustervars="cl", **kw)
        plain = sp.callaway_santanna(ub, **K, **kw)
    assert clustered.model_info["se_method"] == "multiplier"
    # clustering changes the spread, never the point estimates
    np.testing.assert_array_equal(clustered.detail["att"], plain.detail["att"])
    assert not np.allclose(clustered.detail["se"], plain.detail["se"])
    assert (clustered.detail["se"] > 0).all()
    np.testing.assert_array_equal(clustered.detail["se"], again.detail["se"])


@pytest.mark.parametrize("cutoff", ["asinr", "cohort"])
def test_rcs_notyet_cutoff_only_moves_pre_treatment_cells(panel, cutoff):
    kw = dict(panel=False, control_group="notyettreated", base_period="universal")
    ref = sp.callaway_santanna(panel, **K, x=["x1"], **kw)
    alt = sp.callaway_santanna(panel, **K, x=["x1"], notyet_cutoff=cutoff, **kw)
    post = ref.detail["relative_time"] >= 0
    np.testing.assert_allclose(alt.detail.loc[post, "att"], ref.detail.loc[post, "att"])
    moved = ~np.isclose(alt.detail.loc[~post, "att"], ref.detail.loc[~post, "att"])
    # 'asinr' admits cohorts treated between t and the universal base period
    # as comparisons for the leads; 'cohort' coincides with R's rule here
    assert int(moved.sum()) == (3 if cutoff == "asinr" else 0)


# ----------------------------------------------------------------------
# comparison-arm rule on the observation-level paths
# ----------------------------------------------------------------------
def test_rcs_control_mask_rules():
    g = np.array([0, 3, 5, 7])
    args = dict(g_val=5, t_val=2, base_val=4, control_group="notyettreated")
    asinr = C._rcs_control_mask(g, notyet_cutoff="asinr", anticipation=0, **args)
    cohort = C._rcs_control_mask(g, notyet_cutoff="cohort", anticipation=0, **args)
    period = C._rcs_control_mask(g, notyet_cutoff="period", anticipation=0, **args)
    assert asinr.tolist() == [True, True, True, True]  # G > t
    assert cohort.tolist() == [True, False, False, True]  # G > max(t, g)
    assert period.tolist() == [True, False, True, True]  # G > max(t, base)


# ----------------------------------------------------------------------
# cells that cannot be estimated
# ----------------------------------------------------------------------
def test_drop_unidentified_cells_edge_cases():
    empty = pd.DataFrame(columns=["group", "time", "se"])
    detail, infs, dropped = C._drop_unidentified_cells(empty, [])
    assert detail is empty and infs == [] and dropped == []
    all_bad = pd.DataFrame({"group": [4, 4], "time": [5, 6], "se": [np.inf, np.inf]})
    with pytest.raises(DataInsufficient, match="No ATT\\(g, t\\) cell has both"):
        C._drop_unidentified_cells(all_bad, [np.zeros(3), np.zeros(3)])


def test_panel_cell_with_zero_weight_mass_is_not_estimated(panel):
    wide = panel.pivot(index="i", columns="t", values="y")
    info = panel.groupby("i")[["g"]].first()
    n = len(info)
    args = (wide, info, 4, 5, 3, "g", None, "dr", "nevertreated", n)
    assert _not_estimated(C._estimate_single_att(*args, unit_weights=np.zeros(n)), n)
    att, se, _ = C._estimate_single_att(*args, unit_weights=np.ones(n))
    assert abs(att - 2.0) < 4 * se  # true effect at t = g + 1 is 2


def test_two_by_two_engines_without_a_treated_unit():
    rng = np.random.default_rng(0)
    n = 40
    dy, x = rng.normal(size=n), rng.normal(size=(n, 1))
    d = np.zeros(n)
    assert _not_estimated(C._dr_imp_att(dy, d, x), n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels: perfect separation
        assert _not_estimated(C._ipw_abadie_att(dy, d, x), n)


def test_dr_estimation_effect_is_skipped_on_degenerate_weights():
    rng = np.random.default_rng(1)
    n = 40
    dy = rng.normal(size=n)
    base = dict(
        dy=dy,
        d=(np.arange(n) < 15).astype(float),
        x=rng.normal(size=(n, 1)),
        w=np.ones(n),
        keep=np.ones(n),
        pscore=np.full(n, 0.4),
        resid=dy,
        eta_c=0.0,
        p_d=0.375,
        ipw_denom=0.3,
    )
    ok = C._dr_estimation_effect(**base)
    assert ok.shape == (n,) and np.isfinite(ok).all()
    assert C._dr_estimation_effect(**{**base, "p_d": 0.0}) is None
    assert C._dr_estimation_effect(**{**base, "ipw_denom": 0.0}) is None
    # a zero propensity score gives the comparison arm no weight at all
    assert C._dr_estimation_effect(**{**base, "pscore": np.zeros(n)}) is None


def test_rcs_sz_cells_that_cannot_be_estimated():
    rng = np.random.default_rng(2)
    common = dict(g_val=4, t_val=4, base_val=3, control_group="nevertreated")
    # fewer than five rows in the cell
    g, t = np.array([4, 4, 0, 0]), np.array([3, 4, 3, 4])
    out = C._estimate_single_att_rcs_sz(
        np.arange(4.0), g, t, None, estimator="dr", n_obs=4, **common
    )
    assert _not_estimated(out, 4)
    # the cohort is never observed in the base period
    g = np.r_[np.full(10, 4), np.zeros(20)].astype(int)
    t = np.r_[np.full(10, 4), np.tile([3, 4], 10)]
    y = rng.normal(size=30)
    for est in ("dr", "drimp", "ipw", "reg"):
        out = C._estimate_single_att_rcs_sz(
            y, g, t, None, estimator=est, n_obs=30, **common
        )
        assert _not_estimated(out, 30), est
    # an infinite outcome makes the cell's estimate non-finite
    g = np.r_[np.full(20, 4), np.zeros(20)].astype(int)
    t = np.tile([3, 4], 20)
    y = rng.normal(size=40)
    y[0] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = C._estimate_single_att_rcs_sz(
            y, g, t, None, estimator="dr", n_obs=40, **common
        )
    assert _not_estimated(out, 40)


def test_rcs_cell_mean_did_with_a_zero_weight_cell():
    rng = np.random.default_rng(3)
    g = np.r_[np.full(20, 4), np.zeros(20)].astype(int)
    t = np.tile([3, 4], 20)
    y = rng.normal(size=40) + 2.0 * ((g == 4) & (t == 4))
    w = np.ones(40)
    att, se, inf = C._estimate_single_att_rcs(y, g, t, 4, 4, 3, 40, w_arr=w)
    cell = lambda gv, tv: y[(g == gv) & (t == tv)].mean()  # noqa: E731
    assert att == pytest.approx((cell(4, 4) - cell(4, 3)) - (cell(0, 4) - cell(0, 3)))
    assert np.isfinite(se) and se > 0
    w[(g == 4) & (t == 4)] = 0.0
    assert _not_estimated(C._estimate_single_att_rcs(y, g, t, 4, 4, 3, 40, w_arr=w), 40)


# ----------------------------------------------------------------------
# failed covariate regressions fall back loudly
# ----------------------------------------------------------------------
def test_reg_att_falls_back_to_unadjusted_when_the_regression_fails():
    rng = np.random.default_rng(4)
    n = 60
    dy = rng.normal(size=n)
    d = (np.arange(n) < 20).astype(float)
    x = rng.normal(size=(n, 1))
    x[30, 0] = np.nan  # a control row the regression cannot use
    with pytest.warns(ConvergenceWarning, match="falls back to the UNADJUSTED"):
        att, se, inf = C._reg_att(dy, d, x)
    att0, se0, inf0 = C._reg_att(dy, d, None)
    assert att == pytest.approx(att0) and se == pytest.approx(se0)
    assert att == pytest.approx(dy[d == 1].mean() - dy[d == 0].mean())
    np.testing.assert_allclose(inf, inf0)


def test_outcome_regression_falls_back_to_the_control_mean():
    rng = np.random.default_rng(5)
    n = 60
    dy = rng.normal(size=n)
    c = (np.arange(n) >= 20).astype(float)
    x = rng.normal(size=(n, 1))
    x[30, 0] = np.nan
    with pytest.warns(ConvergenceWarning, match="UNADJUSTED control mean"):
        m_hat = C._estimate_outcome_reg(dy, c, x, n)
    np.testing.assert_allclose(m_hat, dy[c == 1].mean())


def test_rcs_residualisation_is_skipped_when_least_squares_fails(panel):
    df = panel.copy()
    df.loc[df.index[-1], "x1"] = np.nan  # a never-treated row
    y = df["y"].to_numpy()
    out = C._rcs_residualise_on_controls(y, df, "g", "t", ["x1"])
    np.testing.assert_array_equal(out, y)
    # with clean covariates the control-arm slope is removed
    clean = C._rcs_residualise_on_controls(
        panel["y"].to_numpy(), panel, "g", "t", ["x1"]
    )
    ctrl = (panel["g"] == 0).to_numpy()
    assert abs(np.corrcoef(clean[ctrl], panel.loc[ctrl, "x1"])[0, 1]) < abs(
        np.corrcoef(y[ctrl], panel.loc[ctrl, "x1"])[0, 1]
    )


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def test_weighted_control_mean_with_no_control_mass():
    dy = np.array([1.0, 2.0, 3.0, 4.0])
    c_mask = np.array([True, True, False, False])
    assert C._weighted_control_mean(dy, c_mask, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    assert C._weighted_control_mean(
        dy, c_mask, np.array([3.0, 1.0, 1.0, 1.0])
    ) == pytest.approx(1.25)


def test_fit_converged_reports_unknown_without_a_flag():
    class NoFlag:
        mle_retvals = {"converged": "yes"}  # not a boolean
        converged = None

    assert C._fit_converged(NoFlag()) is None
    assert C._fit_converged(object()) is None


# ----------------------------------------------------------------------
# repeated-cross-section core called directly
# ----------------------------------------------------------------------
def test_rcs_core_with_no_usable_rows(panel):
    with pytest.raises(DataInsufficient, match="No observations after dropping NaNs"):
        C._callaway_santanna_rcs(
            panel.assign(y=np.nan), "y", "g", "t", "universal", 0, 0.05
        )


def test_rcs_core_unbalanced_route_needs_two_units(panel):
    with pytest.raises(DataInsufficient, match="needs at least 2 units, found 1"):
        C._callaway_santanna_rcs(
            panel[panel["i"] == 0], "y", "g", "t", "universal", 0, 0.05, unit_col="i"
        )
