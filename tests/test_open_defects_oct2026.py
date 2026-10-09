"""Degenerate inputs that used to crash or return noise (Oct 2026).

Each test is the smallest input that reached the defect. The assertions are
on what the function has to do there: refuse with a typed error, skip the
unusable rows, or report the quantity under its own name.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.synth import bsts


def _factor_panel(n_units=8, n_periods=7, t0=4, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n_periods).cumsum()
    rows = []
    for i in range(n_units):
        lam = rng.normal(1, 0.3)
        for t in range(n_periods):
            effect = 2.0 if (i == 0 and t >= t0) else 0.0
            rows.append((i, t, lam * f[t] + rng.normal(0, 0.2) + effect))
    return pd.DataFrame(rows, columns=["unit", "time", "y"])


SCPI = dict(outcome="y", unit="unit", time="time", treated_unit=0, treatment_time=4)


def test_scpi_ridge_needs_five_pre_periods_or_a_radius():
    df = _factor_panel()
    with pytest.raises(DataInsufficient, match="at least 5"):
        sp.scpi(df, **SCPI, w_constr="ridge", sims=20, seed=1)
    # the remedy named in the message works
    fit = sp.scpi(df, **SCPI, w_constr="ridge", Q=1.0, sims=20, seed=1)
    assert fit.model_info["w_constr_spec"]["Q"] == 1.0
    assert np.isfinite(fit.estimate)


def _impact_series(seed=0, n=60):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n).cumsum()
    y = 1 + 0.8 * x + rng.normal(0, 0.3, n)
    y[40:] += 2.0
    return pd.DataFrame({"y": y, "x": x})


def test_causal_impact_skips_a_missing_pre_period_outcome():
    df = _impact_series()
    full = bsts.causal_impact(df, (0, 39), (40, 59), seed=1)
    holed = bsts.causal_impact(
        df.assign(y=df["y"].where(df.index != 10)), (0, 39), (40, 59), seed=1
    )
    # one of forty pre-period points carries little information
    assert np.isfinite(holed.estimate) and np.isfinite(holed.se)
    assert abs(holed.estimate - full.estimate) < 0.25 * full.se
    assert holed.estimate == pytest.approx(2.0, abs=4 * holed.se)


def test_causal_impact_refuses_a_missing_post_period_outcome():
    df = _impact_series()
    with pytest.raises(DataInsufficient, match="1 post-period row"):
        bsts.causal_impact(
            df.assign(y=df["y"].where(df.index != 50)), (0, 39), (40, 59), seed=1
        )


def test_fect_refuses_collinear_covariates():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        a = rng.normal()
        for t in range(10):
            x = rng.normal()
            d = int(i < 6 and t >= 6)
            y = a + 0.1 * t + 0.5 * x + 2 * d + rng.normal(0, 0.3)
            rows.append((i, t, y, d, x, 2 * x, rng.normal()))
    df = pd.DataFrame(rows, columns=["id", "t", "y", "d", "x", "x2", "z"])
    kw = dict(y="y", treat="d", unit="id", time="t", method="fe")
    with pytest.raises(MethodIncompatibility, match="collinear"):
        sp.fect(df, **kw, covariates=["x", "x2"])
    ok = sp.fect(df, **kw, covariates=["x", "z"])
    assert ok.estimate == pytest.approx(2.0, abs=0.3)


def test_twowayfeweights_saturated_design_has_no_standard_error():
    # four cells, four fixed effects and a slope: the residuals are zero
    df = pd.DataFrame(
        {"g": [0, 0, 1, 1], "t": [0, 1, 0, 1], "d": [0, 0, 0, 1], "y": [1, 2, 1.5, 4]}
    )
    res = sp.twowayfeweights(df, y="y", group="g", time="t", treat="d")
    assert res.estimate == pytest.approx((4 - 1.5) - (2 - 1))
    assert np.isnan(res.se) and np.isnan(res.pvalue)


def _plr_data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = x + rng.normal(size=n)
    y = d + x + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x": x})


def _plr(df, folds):
    return sp.dml(
        df,
        y="y",
        treat="d",
        covariates=["x"],
        model="plr",
        ml_g=LinearRegression(),
        ml_m=LinearRegression(),
        n_folds=4,
        fold_indices=folds,
    )


def test_dml_refuses_a_missing_fold_label_on_a_complete_row():
    df = _plr_data()
    folds = (np.arange(len(df)) % 4).astype(float)
    folds[5] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing values on 1 row"):
        _plr(df, folds)


def test_dml_still_drops_an_incomplete_row_whatever_its_fold_label():
    df = _plr_data()
    folds = (np.arange(len(df)) % 4).astype(float)
    holed = df.copy()
    holed.loc[5, "y"] = np.nan
    unlabelled = folds.copy()
    unlabelled[5] = np.nan
    assert _plr(holed, unlabelled).estimate == _plr(holed, folds).estimate


def test_fuzzy_rdrandinf_reports_the_itt_interval_under_its_own_name():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.uniform(-1, 1, n)
    z = x >= 0
    d = np.where(z, rng.random(n) < 0.8, rng.random(n) < 0.1).astype(float)
    y = 2.0 * d + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x": x, "d": d})
    grid = np.linspace(0.5, 3.5, 31)
    res = sp.rdrandinf(
        df, y="y", x="x", fuzzy="d", wl=-0.5, wr=0.5, ci=grid, n_perms=200, seed=1
    )
    info = res.model_info
    lo, hi = info["itt_ci"]
    assert lo < info["itt"] < hi
    assert (lo + hi) / 2 == pytest.approx(info["itt"])
    # the grid inversion tests y - tau0 * D, so it bounds the complier effect
    ar = info["anderson_rubin_ci"]
    assert ar is not None and ar[0] <= 2.0 <= ar[1]
    assert not (lo <= 2.0 <= hi)


def test_synth_sensitivity_returns_the_estimate_its_plot_marks():
    df = sp.california_prop99()
    kw = dict(
        outcome="packspercapita",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
    )
    sens = sp.synth_sensitivity(df, **kw, n_donor_samples=5, seed=1)
    base = sp.synth(df, **kw, placebo=False)
    assert sens["original_att"] == pytest.approx(base.estimate, rel=1e-10)
    fig, axes = sp.synthplot(sens, type="sensitivity")
    marks = [ln for ln in axes[0, 0].lines if ln.get_linestyle() == "--"]
    assert len(marks) == 1
    np.testing.assert_allclose(marks[0].get_xdata(), base.estimate)
    matplotlib.pyplot.close(fig)


def test_did_multiplegt_dyn_names_a_missing_cluster_column():
    df = sp.dgp_did(n_units=40, n_periods=6, staggered=True, seed=1)
    with pytest.raises(MethodIncompatibility, match="cluster column 'nope'"):
        sp.did_multiplegt_dyn(
            df, y="y", group="unit", time="time", treatment="treated", cluster="nope"
        )


def test_did_multiplegt_dyn_analytic_se_says_nothing_about_a_bootstrap(recwarn):
    df = sp.dgp_did(n_units=60, n_periods=6, staggered=True, seed=1)
    res = sp.did_multiplegt_dyn(
        df,
        y="y",
        group="unit",
        time="time",
        treatment="treated",
        dynamic=2,
        se_method="analytic",
    )
    assert np.isfinite(res.se)
    assert not [w for w in recwarn.list if "bootstrap replicates" in str(w.message)]


def test_rd_optimized_fuzzy_refuses_a_treatment_with_no_jump():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(-1, 1, n)
    y = x + (x >= 0) + rng.normal(0, 0.3, n)
    for d in (np.ones(n), 0.5 + 0.2 * x):
        df = pd.DataFrame({"y": y, "x": x, "d": d})
        with pytest.raises(DataInsufficient, match="does not jump"):
            sp.rd_optimized(df, y="y", x="x", fuzzy="d", M=1.0)


def test_stacked_did_lists_only_the_cohorts_it_stacked():
    df = sp.dgp_did(n_units=90, n_periods=12, staggered=True, seed=3)
    res = sp.stacked_did(
        df, y="y", group="unit", time="time", first_treat="first_treat", window=(-2, 2)
    )
    info = res.model_info
    assert len(info["cohorts"]) == info["n_cohorts"]
    assert set(info["cohorts"]).isdisjoint(info["skipped_cohorts"])
