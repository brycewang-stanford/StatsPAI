"""Causal forests with fixed effects, and the split-sample tools built on
any causal forest.

A forest with ``fe="twoway"`` has no propensity score.  Everything it
reports about averages comes from *imputation scores*: unit and period
effects are fitted on the untreated cells and ``Y - alpha_i - gamma_t`` of
a treated cell is an unbiased signal of that cell's effect.  Those scores
do not depend on the forest at all, so the ATT, its covariate-adjusted
variants, the best linear projection, group effects, the calibration
regression and the RATE point estimate can each be rebuilt with a
dummy-variable least-squares fit in numpy and compared exactly.

The second half checks ``sp.rate_split`` / ``sp.forest_policy_tree`` /
``sp.cate_pretrend_test`` / ``sp.tune_causal_forest``: the pieces that are
deterministic given their inputs, and whether the forests they refit are
the forest they were handed.

``ATOL = 1e-8``: the package solves the untreated two-way model with a
sparse LU and the tests with a dense least-squares fit; the two agree to
about 1e-12 on these panels, and 1e-8 leaves room without admitting a
different estimator.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from statspai.forest.forest_heterogeneity import _refit_halves

ATOL = 1e-8
KW = dict(n_estimators=100, random_state=31)


# --------------------------------------------------------------------------- #
#  Hand-written imputation
# --------------------------------------------------------------------------- #


def _imputation_scores(y, d, unit, time, controls=None):
    """``Y - alpha_i - gamma_t [- C beta]`` with the model fitted on the
    untreated cells; NaN where the unit or the period has no untreated cell."""
    u_codes = pd.factorize(unit)[0]
    t_codes = pd.factorize(time)[0]
    blocks = [np.eye(u_codes.max() + 1)[u_codes], np.eye(t_codes.max() + 1)[t_codes]]
    if controls is not None:
        blocks.append(np.asarray(controls, dtype=float).reshape(len(y), -1))
    Z = np.column_stack(blocks)
    untreated = d == 0
    coef = np.linalg.lstsq(Z[untreated], y[untreated], rcond=None)[0]
    scores = y - Z @ coef
    ok = np.isin(unit, unit[untreated]) & np.isin(time, time[untreated])
    return np.where(ok, scores, np.nan)


def _rate(scores, prio, target):
    _, codes = np.unique(prio, return_inverse=True)
    s_avg = (np.bincount(codes, weights=scores) / np.bincount(codes))[codes]
    s = s_avg[np.argsort(-codes, kind="stable")]
    k = np.arange(1, len(s) + 1)
    toc = np.cumsum(s) / k - s.mean()
    return toc.mean() if target == "AUTOC" else np.mean(k / len(s) * toc)


def _panel(seed=401, n_units=70, n_periods=6, adopt_p=(0.25, 0.25, 0.2, 0.3)):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)
    n = unit.size
    z = rng.normal(size=n_units)[unit]
    xv = rng.normal(size=n)
    adopt = rng.choice([2, 3, 4, 99], size=n_units, p=adopt_p)[unit]
    d = (time >= adopt).astype(float)
    y = (
        rng.normal(size=n_units)[unit]
        + 0.3 * time
        + (1 + 0.5 * z) * d
        + 0.5 * xv
        + rng.normal(scale=0.5, size=n)
    )
    return pd.DataFrame(dict(y=y, d=d, z=z, xv=xv, unit=unit, time=time))


def _fit(df, **extra):
    args = dict(data=df, y="y", d="d", x=["z", "xv"], id="unit", time="time")
    args.update(KW)
    args.update(extra)
    args.setdefault("fe", "twoway")
    return sp.causal_forest(**args)


@pytest.fixture(scope="module")
def panel():
    df = _panel()
    cf = _fit(df)
    scores = _imputation_scores(
        df["y"].to_numpy(), df["d"].to_numpy(), df["unit"].to_numpy(), df["time"]
    )
    return cf, df, scores


# --------------------------------------------------------------------------- #
#  Fit-time checks
# --------------------------------------------------------------------------- #


def test_treatment_collinear_with_period_effects_is_refused():
    df = _panel(seed=402)
    d = (df["time"] >= 3).astype(float)
    # Premise: the two-way within transformation of this treatment is zero.
    within = (
        d
        - d.groupby(df["unit"]).transform("mean")
        - d.groupby(df["time"]).transform("mean")
        + d.mean()
    )
    assert np.max(np.abs(within)) < 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(DataInsufficient):
            _fit(df.assign(d=d))


def test_fixed_effects_fit_checks(panel):
    _, df, _ = panel
    with pytest.raises(MethodIncompatibility, match="requires unit= ids"):
        sp.causal_forest(data=df, y="y", d="d", x=["z"], fe="twoway", **KW)
    with pytest.raises(MethodIncompatibility, match="requires time= ids"):
        sp.causal_forest(data=df, y="y", d="d", x=["z"], id="unit", fe="twoway", **KW)
    with pytest.raises(MethodIncompatibility, match="pairs must be unique"):
        _fit(pd.concat([df, df.head(3)], ignore_index=True))
    with pytest.raises(DataInsufficient, match="does not vary within units"):
        _fit(df.assign(d=(df["z"] > 0).astype(float)))
    with pytest.raises(DataInsufficient, match="at least two units"):
        _fit(df.assign(unit=0, time=np.arange(len(df))))
    with pytest.raises(MethodIncompatibility, match="must nest units"):
        _fit(df, clusters="time")
    with pytest.raises(DataInsufficient, match="at least two clusters"):
        _fit(df, clusters=np.zeros(len(df)))
    bad_id = df["unit"].to_numpy().astype(float)
    bad_id[0] = np.nan
    with pytest.raises(MethodIncompatibility, match="unit contains missing"):
        _fit(df, id=bad_id)
    with pytest.raises(MethodIncompatibility, match="unit contains missing"):
        _fit(df, id=np.array([None] + list(df["unit"][1:]), dtype=object))
    with pytest.raises(MethodIncompatibility, match="one id per row"):
        _fit(df, id=np.arange(5))
    with pytest.raises(MethodIncompatibility, match="tau-heterogeneity"):
        sp.causal_forest(data=df, y="y", d="d", x=["z"], split_rule="cffe", **KW)


def test_fit_diagnostics_describe_the_panel(panel):
    cf, df, _ = panel
    diag = cf.diagnostics
    assert diag["fe"] == "twoway" and diag["n_units"] == 70
    assert diag["n_periods"] == 6 and diag["n_clusters"] == 70
    switching = df.groupby("unit")["d"].nunique().gt(1).mean()
    assert diag["share_units_switching_treatment"] == pytest.approx(switching)
    assert diag["nuisance_overlap"]["applicable"] is False
    assert "Fixed effects:            twoway" in cf.summary()
    # Coarser clusters that contain whole units are accepted and reported.
    coarse = _fit(df, clusters=df["unit"].to_numpy() // 2)
    assert coarse.diagnostics["n_clusters"] == 35
    assert coarse.average_treatment_effect("treated")["n_clusters"] == 35


def test_cffe_split_rule_warns_about_tree_size_and_keeps_the_imputation_att(panel):
    cf, df, _ = panel
    with pytest.warns(AssumptionWarning, match="tau-heterogeneity criterion"):
        cffe = _fit(df, split_rule="cffe")
    assert cffe.diagnostics["split_rule"] == "cffe"
    # The imputation ATT does not use the forest, so the split rule cannot
    # move it.
    assert cffe.average_treatment_effect("treated")["estimate"] == pytest.approx(
        cf.average_treatment_effect("treated")["estimate"], abs=ATOL
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        _fit(df, split_rule="cffe", min_samples_leaf=20, max_depth=4)


# --------------------------------------------------------------------------- #
#  Imputation averages
# --------------------------------------------------------------------------- #


def test_att_is_the_mean_imputation_score_of_the_treated_cells(panel):
    cf, df, scores = panel
    treated = df["d"].to_numpy() == 1
    out = cf.average_treatment_effect("treated")
    assert out["estimate"] == pytest.approx(scores[treated].mean(), abs=ATOL)
    assert out["method"] == "imputation" and out["estimand"] == "ATT"
    assert out["n_treated_cells"] == int(treated.sum()) and out["n"] == len(df)
    assert out["n_not_imputable"] == 0
    assert out["weighting"] == "equal per treated cell"
    z = stats.norm.ppf(0.975)
    assert out["ci_high"] == pytest.approx(out["estimate"] + z * out["se"])
    assert out["pvalue"] == pytest.approx(
        2 * stats.norm.sf(abs(out["estimate"] / out["se"]))
    )
    assert out["forest_plug_in"] == pytest.approx(cf.predict()[treated].mean())
    # The scalar wrapper carries the same number.
    assert float(cf.att()) == pytest.approx(out["estimate"], abs=ATOL)
    assert cf.att().se == pytest.approx(out["se"])


def test_variance_option_changes_the_centring_not_the_estimate(panel):
    cf, _, _ = panel
    forest = cf.average_treatment_effect("treated", variance="forest")
    bjs = cf.average_treatment_effect("treated", variance="bjs")
    assert forest["estimate"] == bjs["estimate"]
    assert forest["variance"] == "forest" and bjs["variance"] == "bjs"
    assert "OOB forest" in forest["method_detail"]
    assert "OOB forest" not in bjs["method_detail"]
    assert np.isfinite(forest["se"]) and np.isfinite(bjs["se"])
    with pytest.raises(MethodIncompatibility, match="variance must be one of"):
        cf.average_treatment_effect("treated", variance="hc1")


def test_covariates_enter_the_untreated_model(panel):
    cf, df, _ = panel
    y, d = df["y"].to_numpy(), df["d"].to_numpy()
    treated = d == 1
    adjusted = _imputation_scores(y, d, df["unit"].to_numpy(), df["time"], df["xv"])
    hand = adjusted[treated].mean()
    by_name = cf.average_treatment_effect("treated", covariates=["xv"])
    by_array = cf.average_treatment_effect("treated", covariates=df["xv"].to_numpy())
    auto = cf.average_treatment_effect("treated", covariates="auto")
    alias = cf.average_treatment_effect("treated", controls=["xv"])
    for out in (by_name, by_array, auto, alias):
        assert out["estimate"] == pytest.approx(hand, abs=ATOL)
    assert by_name["imputation_covariates"] == ["xv"]
    # 'auto' offers both effect modifiers; z is constant within units, so
    # the unit effects absorb it and only xv is kept.
    assert auto["imputation_covariates"] == ["xv"]
    assert by_array["imputation_covariates"] == ["control0"]
    only_z = cf.average_treatment_effect("treated", covariates=["z"])
    assert only_z["imputation_covariates"] == []
    assert only_z["estimate"] == pytest.approx(
        cf.average_treatment_effect("treated")["estimate"], abs=ATOL
    )
    assert abs(hand - only_z["estimate"]) > 1e-4  # the control does something


@pytest.mark.parametrize(
    "covariates, match",
    [
        (["nope"], "not effect modifiers or controls"),
        ("xv", "must be 'auto', 'none'"),
        (np.zeros(5), "one finite row per training row"),
        (np.full(420, np.nan), "one finite row per training row"),
    ],
)
def test_bad_imputation_covariates_are_refused(panel, covariates, match):
    with pytest.raises(MethodIncompatibility, match=match):
        panel[0].average_treatment_effect("treated", covariates=covariates)


def test_controls_passed_as_w_are_offered_by_auto(panel):
    _, df, _ = panel
    cf = sp.causal_forest(
        data=df, y="y", d="d", x=["z"], w=["xv"], id="unit", time="time",
        fe="twoway", **KW,
    )  # fmt: skip
    out = cf.average_treatment_effect("treated", covariates="auto")
    assert out["imputation_covariates"] == ["xv"]
    y, d = df["y"].to_numpy(), df["d"].to_numpy()
    adjusted = _imputation_scores(y, d, df["unit"].to_numpy(), df["time"], df["xv"])
    assert out["estimate"] == pytest.approx(adjusted[d == 1].mean(), abs=ATOL)


def test_targets_other_than_the_treated_cells_are_refused(panel):
    cf, df, _ = panel
    for target in ("all", "control", "overlap"):
        with pytest.raises(MethodIncompatibility, match="not identified"):
            cf.average_treatment_effect(target)
    with pytest.raises(MethodIncompatibility, match="subset= is available"):
        cf.average_treatment_effect("treated", subset=np.arange(10))
    with pytest.raises(MethodIncompatibility, match="training sample"):
        cf.average_treatment_effect(
            X=df[["z", "xv"]].to_numpy() + 1.0, target_sample="treated"
        )
    with pytest.raises(MethodIncompatibility, match="fixed effects"):
        sp.get_scores(cf)
    # ate() has no identified counterpart: plug-in value, reason attached.
    effect = cf.ate()
    assert effect.se is None and "not identified" in effect.inference_error
    assert float(effect) == pytest.approx(cf.predict().mean())


def test_cells_that_cannot_be_imputed_are_excluded_and_counted():
    # Adoption period 0 makes a fifth of the units always treated.
    rng = np.random.default_rng(403)
    df = _panel(seed=403)
    always = rng.random(70) < 0.2
    d = df["d"].to_numpy().copy()
    d[always[df["unit"].to_numpy()]] = 1.0
    df = df.assign(d=d, y=df["y"] + (d - df["d"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = _fit(df)
    scores = _imputation_scores(
        df["y"].to_numpy(), d, df["unit"].to_numpy(), df["time"]
    )
    usable = (d == 1) & np.isfinite(scores)
    lost = int(np.sum((d == 1) & ~np.isfinite(scores)))
    assert lost == 6 * int(always.sum()) > 0
    with pytest.warns(AssumptionWarning, match=f"{lost} treated cell"):
        out = cf.average_treatment_effect("treated")
    assert out["estimate"] == pytest.approx(scores[usable].mean(), abs=ATOL)
    assert out["n_not_imputable"] == lost
    assert out["n_treated_cells"] == int(usable.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = sp.forest_group_effects(cf)
    assert tab.loc["all", "n_rows"] == int(usable.sum())
    assert tab.attrs["n_not_imputable"] == lost


def test_a_treatment_that_switches_off_is_flagged():
    rng = np.random.default_rng(404)
    df = _panel(seed=404)
    d = rng.binomial(1, 0.4, len(df)).astype(float)
    df = df.assign(d=d, y=df["y"] - df["d"] * (1 + 0.5 * df["z"]) + d)
    cf = _fit(df)
    scores = _imputation_scores(
        df["y"].to_numpy(), d, df["unit"].to_numpy(), df["time"]
    )
    with pytest.warns(AssumptionWarning, match="switches off"):
        out = cf.average_treatment_effect("treated")
    assert out["estimate"] == pytest.approx(scores[d == 1].mean(), abs=ATOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(MethodIncompatibility, match="absorbing"):
            sp.cate_pretrend_test(cf)


def test_equalized_cluster_weights_reweight_the_treated_cells():
    """Units observed for different numbers of periods: with
    ``equalize_cluster_weights`` each treated cell counts ``1 / (rows of
    its unit)``, so the ATT moves away from the equal-cell average."""
    rng = np.random.default_rng(405)
    df = _panel(seed=405, n_units=90)
    # Drop the first one or two periods of a random third of the units.
    late = rng.integers(0, 3, 90)[df["unit"].to_numpy()]
    df = df[df["time"] >= late].reset_index(drop=True)
    sizes = df.groupby("unit")["y"].transform("size").to_numpy()
    assert set(sizes) == {4, 5, 6}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = _fit(df, equalize_cluster_weights=True)
        out = cf.average_treatment_effect("treated")
    d = df["d"].to_numpy()
    scores = _imputation_scores(
        df["y"].to_numpy(), d, df["unit"].to_numpy(), df["time"].to_numpy()
    )
    # Units whose untreated periods were all dropped cannot be imputed.
    usable = (d == 1) & np.isfinite(scores)
    assert out["n_not_imputable"] == int(np.sum((d == 1) & ~usable)) > 0
    w = 1.0 / sizes[usable]
    weighted = np.sum(w * scores[usable]) / np.sum(w)
    assert out["estimate"] == pytest.approx(weighted, abs=ATOL)
    assert out["weighting"].startswith("forest observation weights")
    assert out["effective_sample_size"] == pytest.approx(w.sum() ** 2 / np.sum(w**2))
    assert abs(weighted - scores[usable].mean()) > 1e-4


# --------------------------------------------------------------------------- #
#  Projections, groups, calibration, RATE
# --------------------------------------------------------------------------- #


def test_blp_is_ols_of_the_scores_over_treated_cells(panel):
    cf, df, scores = panel
    treated = df["d"].to_numpy() == 1
    D = np.column_stack([np.ones(treated.sum()), df[["z", "xv"]].to_numpy()[treated]])
    beta = np.linalg.lstsq(D, scores[treated], rcond=None)[0]
    tab = cf.best_linear_projection(alpha=0.1)
    assert list(tab.index) == ["Intercept", "z", "xv"]
    np.testing.assert_allclose(tab["coef"], beta, atol=ATOL)
    np.testing.assert_allclose(
        tab["ci_upper"], tab["coef"] + stats.norm.ppf(0.95) * tab["se"]
    )
    assert f"{int(treated.sum())} treated cells" in tab.attrs["method"]
    # Any aligned covariate may be projected on.
    one = sp.best_linear_projection(cf, A=df[["z"]])
    D1 = D[:, :2]
    np.testing.assert_allclose(
        one["coef"], np.linalg.lstsq(D1, scores[treated], rcond=None)[0], atol=ATOL
    )
    constant = np.ones(len(df))
    with pytest.raises(MethodIncompatibility, match="collinear on the treated"):
        sp.best_linear_projection(cf, A=constant)


def test_group_effects_are_group_means_of_the_scores(panel):
    cf, df, scores = panel
    treated = df["d"].to_numpy() == 1
    high = df["z"].to_numpy() > 0
    tab = sp.forest_group_effects(cf, by=np.where(high, "high", "low"))
    assert tab.attrs["estimand"] == "ATT (treated cells)"
    for label, rows in (("high", treated & high), ("low", treated & ~high)):
        assert tab.loc[label, "estimate"] == pytest.approx(
            scores[rows].mean(), abs=ATOL
        )
        assert tab.loc[label, "n_rows"] == rows.sum()
        assert tab.loc[label, "n_units"] == df["unit"][rows].nunique()
    tests = tab.attrs["tests"]
    assert tests["equality_df"] == 1
    assert tests["last_minus_first"]["estimate"] == pytest.approx(
        tab.loc["low", "estimate"] - tab.loc["high", "estimate"]
    )
    # One group difference: the Wald statistic is the squared z of it.
    lmf = tests["last_minus_first"]
    assert tests["equality_wald"] == pytest.approx(
        (lmf["estimate"] / lmf["se"]) ** 2, rel=1e-8
    )
    quartiles = sp.forest_group_effects(cf, by="cate_quantile", n_groups=4)
    assert quartiles["n_rows"].sum() == treated.sum()
    # Quantile groups of the out-of-bag prediction among treated cells:
    # the cell of 0-based rank r goes to group floor(4 r / m).
    m = int(treated.sum())
    rank = np.argsort(np.argsort(cf.predict()[treated], kind="stable"))
    bin_of = np.minimum(rank * 4 // m, 3)
    for g in range(4):
        cells = np.flatnonzero(treated)[bin_of == g]
        assert quartiles["n_rows"].iloc[g] == len(cells)
        assert quartiles["estimate"].iloc[g] == pytest.approx(
            scores[cells].mean(), abs=ATOL
        )
    # A custom cluster variable changes the variance, not the estimates.
    by_time = sp.forest_group_effects(cf, cluster=df["time"].to_numpy())
    assert by_time["estimate"].iloc[0] == pytest.approx(
        scores[treated].mean(), abs=ATOL
    )
    assert "supplied cluster ids" in by_time.attrs["method"]
    with pytest.raises(MethodIncompatibility, match="one value per training row"):
        sp.forest_group_effects(cf, cluster=np.arange(5))
    with pytest.raises(MethodIncompatibility, match="needs members"):
        sp.forest_group_effects(cf, cluster="dyadic")
    with pytest.raises(MethodIncompatibility, match=r"\(n, 2\) array"):
        sp.forest_group_effects(cf, members=np.arange(len(df)))


def test_imputation_calibration_regresses_scores_on_oob_predictions(panel):
    cf, df, scores = panel
    treated = df["d"].to_numpy() == 1
    tau = cf.predict()[treated]
    D = np.column_stack([np.full(tau.size, tau.mean()), tau - tau.mean()])
    beta = np.linalg.lstsq(D, scores[treated], rcond=None)[0]
    tab = sp.calibration_test(cf, alpha=0.1)
    np.testing.assert_allclose(tab["coef"], beta, atol=ATOL)
    np.testing.assert_allclose(tab["p"], stats.norm.sf(tab["coef"] / tab["se"]))
    np.testing.assert_allclose(
        tab["ci_high"], tab["coef"] + stats.norm.ppf(0.95) * tab["se"]
    )
    assert tab.attrs["tau_bar"] == pytest.approx(tau.mean())
    # beta_mean * tau_bar is the imputation ATT.
    assert beta[0] * tau.mean() == pytest.approx(scores[treated].mean(), abs=ATOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = sp.calibrate_cate(cf)
    assert cal["method"] == "blp_imputation"
    oob = cf.predict()
    np.testing.assert_allclose(
        cal["cate"], beta[0] * tau.mean() + beta[1] * (oob - tau.mean()), atol=1e-7
    )
    # The earlier within-transformed regression stays available.
    within = sp.calibration_test(cf, method="within")
    assert within.shape == (2, 8) and "robust SE" in within.attrs["method"]
    with pytest.warns(AssumptionWarning, match="within-transformed"):
        assert sp.calibrate_cate(cf, method="within")["method"] == "blp_oob"


@pytest.mark.parametrize("target", ["AUTOC", "QINI"])
def test_rate_of_a_fixed_effects_forest_ranks_the_treated_cells(panel, target):
    cf, df, scores = panel
    treated = df["d"].to_numpy() == 1
    prio = df["z"].to_numpy()
    hand = _rate(scores[treated], prio[treated], target)
    full = sp.rate(cf, target=target, priorities=prio)
    assert full["estimate"] == pytest.approx(hand, abs=ATOL)
    assert full["n"] == int(treated.sum())
    # Priorities may be given for the treated cells only.
    short = sp.rate(cf, target=target, priorities=prio[treated])
    assert short["estimate"] == pytest.approx(hand, abs=ATOL)
    infl = sp.rate(cf, target=target, priorities=prio, se_method="influence")
    assert infl["estimate"] == pytest.approx(hand, abs=ATOL) and infl["se"] > 0
    with pytest.raises(MethodIncompatibility, match="se_method must be"):
        sp.rate(cf, priorities=prio, se_method="half_sample")
    with pytest.raises(MethodIncompatibility):
        sp.rate(cf, priorities=prio[:7])
    with pytest.raises(MethodIncompatibility, match="one value per training row"):
        sp.rate(cf, priorities=prio, cluster=np.arange(3))
    bad_cluster = df["unit"].to_numpy().astype(float)
    bad_cluster[0] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing values"):
        sp.rate(cf, priorities=prio, cluster=bad_cluster)


def test_rate_with_the_forests_own_ranking_warns_that_it_is_not_a_test(panel):
    cf, df, scores = panel
    treated = df["d"].to_numpy() == 1
    with pytest.warns(AssumptionWarning, match="does not give a valid test"):
        out = sp.rate(cf)
    assert out["estimate"] == pytest.approx(
        _rate(scores[treated], cf.predict()[treated], "AUTOC"), abs=ATOL
    )


# --------------------------------------------------------------------------- #
#  Unit fixed effects only
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def unit_fe(panel):
    _, df, _ = panel
    return _fit(df, fe="unit", time=None)


def test_unit_fixed_effects_have_no_imputation(unit_fe):
    cf = unit_fe
    assert cf.diagnostics["n_periods"] is None
    with pytest.raises(MethodIncompatibility, match="fe='twoway'"):
        cf.average_treatment_effect("treated")
    with pytest.raises(MethodIncompatibility, match="fe='twoway'"):
        sp.calibration_test(cf, method="imputation")
    # The default calibration falls back to the within regression.
    tab = sp.calibration_test(cf)
    assert "robust SE" in tab.attrs["method"]
    np.testing.assert_allclose(tab["p"], stats.t.sf(tab["t"], len(cf._Y_original) - 2))


def test_support_is_measured_against_units_whose_treatment_varies(panel):
    cf, df, _ = panel
    switching = df.groupby("unit")["d"].transform("nunique").gt(1).to_numpy()
    out = sp.forest_support(cf, df[["z", "xv"]].head(8), k=3)
    summary = out.attrs["summary"]
    assert summary["n_reference_rows"] == int(switching.sum())
    assert summary["reference"] == "rows of units whose treatment varies"
    # The benchmark excludes a unit's own rows: by brute force.
    X = df[["z", "xv"]].to_numpy()[switching]
    units = df["unit"].to_numpy()[switching]
    ref = (X - X.mean(axis=0)) / X.std(axis=0)
    dist = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=2)
    kth = np.array([np.sort(dist[i][units != units[i]])[2] for i in range(len(ref))])
    assert summary["knn_cutoff"] == pytest.approx(np.quantile(kth, 0.95), rel=1e-9)
    with pytest.raises(DataInsufficient, match="k or fewer units"):
        sp.forest_support(cf, df[["z", "xv"]].head(2), k=int(len(np.unique(units))))


# --------------------------------------------------------------------------- #
#  Pre-trends by predicted-effect group
# --------------------------------------------------------------------------- #


def _pretrend_by_hand(df, groups_of_row, leads, clusters):
    """Dummy-variable regression on untreated cells: unit and period
    effects plus group x lead indicators; CR1 covariance."""
    d = df["d"].to_numpy()
    unit, time = df["unit"].to_numpy(), df["time"].to_numpy()
    first = df["time"].where(df["d"] == 1).groupby(df["unit"]).transform("min")
    rel = (df["time"] - first).to_numpy()  # NaN for never-treated units
    rows = d == 0
    labels = sorted(set(groups_of_row))
    L, index = [], []
    for g in labels:
        for k in range(1, leads + 1):
            col = ((groups_of_row == g) & (rel == -k))[rows].astype(float)
            if col.sum():
                L.append(col)
                index.append((g, -k))
    L = np.column_stack(L)
    u = pd.factorize(unit[rows])[0]
    t = pd.factorize(time[rows])[0]
    FE = np.column_stack([np.eye(u.max() + 1)[u], np.eye(t.max() + 1)[t][:, 1:]])
    y = df["y"].to_numpy()[rows]
    # Frisch-Waugh: partial the fixed effects out of y and the indicators.
    Lt = L - FE @ np.linalg.lstsq(FE, L, rcond=None)[0]
    yt = y - FE @ np.linalg.lstsq(FE, y, rcond=None)[0]
    bread = np.linalg.inv(Lt.T @ Lt)
    beta = bread @ Lt.T @ yt
    resid = yt - Lt @ beta
    cl = pd.factorize(clusters[rows])[0]
    S = np.vstack(
        [(Lt * resid[:, None])[cl == g].sum(axis=0) for g in range(cl.max() + 1)]
    )
    n_obs, G = rows.sum(), cl.max() + 1
    K = L.shape[1] + t.max()  # lead columns + period dummies (units nested)
    V = bread @ S.T @ S @ bread * G / (G - 1) * (n_obs - 1) / (n_obs - K)
    return beta, np.sqrt(np.diag(V)), index, V


def test_pretrend_test_is_the_documented_regression(panel):
    cf, df, _ = panel
    groups = np.where(df["z"].to_numpy() > 0, "hi", "lo")
    out = sp.cate_pretrend_test(cf, groups=groups, leads=2, alpha=0.1)
    beta, se, index, V = _pretrend_by_hand(df, groups, 2, df["unit"].to_numpy())
    coef = out["coefficients"]
    assert list(coef.index) == index
    np.testing.assert_allclose(coef["coef"], beta, atol=1e-7)
    np.testing.assert_allclose(coef["se"], se, rtol=1e-6)
    np.testing.assert_allclose(
        coef["ci_high"], coef["coef"] + stats.norm.ppf(0.95) * coef["se"]
    )
    stat = float(beta @ np.linalg.solve(V, beta))
    assert out["joint_zero"]["stat"] == pytest.approx(stat, rel=1e-6)
    assert out["joint_zero"]["df"] == len(beta)
    assert out["joint_zero"]["p"] == pytest.approx(
        stats.chi2.sf(stat, len(beta)), rel=1e-6
    )
    assert out["equal_across_groups"]["df"] == 2
    assert out["leads"] == 2 and out["n_obs"] == int((df["d"] == 0).sum())
    assert out["n_clusters"] == df["unit"][df["d"] == 0].nunique()
    assert set(out["group_of_unit"]) == {"hi", "lo"}


def test_pretrend_test_default_groups_and_options(panel):
    cf, df, _ = panel
    out = sp.cate_pretrend_test(cf)
    assert out["time_effects"] == "common"
    assert set(out["group_of_unit"].dropna()) == {"Q1", "Q2"}
    # Units are split at the median of their mean out-of-bag prediction.
    unit_tau = pd.Series(cf.predict()).groupby(df["unit"]).mean()
    assert (out["group_of_unit"] == "Q2").sum() == len(unit_tau) // 2
    top = unit_tau.sort_values().index[-1]
    assert out["group_of_unit"].iloc[top] == "Q2"
    by_group = sp.cate_pretrend_test(cf, time_effects="by_group", leads=1)
    assert "group-specific" in by_group["method"]
    with_ctrl = sp.cate_pretrend_test(cf, covariates=["xv"], leads=1)
    assert with_ctrl["controls"] == ["xv"]
    with pytest.raises(MethodIncompatibility, match="time_effects"):
        sp.cate_pretrend_test(cf, time_effects="none")
    with pytest.raises(MethodIncompatibility, match="n_groups"):
        sp.cate_pretrend_test(cf, n_groups=1)
    with pytest.raises(DataInsufficient, match="leads must be between"):
        sp.cate_pretrend_test(cf, leads=40)
    with pytest.raises(MethodIncompatibility, match="one non-missing label"):
        sp.cate_pretrend_test(cf, groups=np.ones(3))
    with pytest.raises(MethodIncompatibility, match="constant within units"):
        sp.cate_pretrend_test(cf, groups=df["time"].to_numpy())
    with pytest.raises(DataInsufficient, match="at least two groups"):
        sp.cate_pretrend_test(cf, groups=np.zeros(len(df)))


def test_pretrend_test_needs_a_fixed_effects_forest():
    rng = np.random.default_rng(406)
    X = rng.normal(size=(200, 2))
    T = rng.binomial(1, 0.5, 200).astype(float)
    cf = sp.causal_forest(Y=X[:, 0] + T + rng.normal(size=200), T=T, X=X, **KW)
    with pytest.raises(MethodIncompatibility, match="fixed effects"):
        sp.cate_pretrend_test(cf)
    with pytest.raises(MethodIncompatibility, match="fixed effects"):
        sp.forest_policy_tree(cf)


def test_pretrend_test_rejects_an_alpha_outside_the_unit_interval(panel):
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.cate_pretrend_test(panel[0], alpha=1.5)


# --------------------------------------------------------------------------- #
#  Split-sample evaluation
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def confounded():
    """A pooled design whose only confounder is a control passed as W."""
    rng = np.random.default_rng(407)
    n = 600
    X = rng.normal(size=(n, 2))
    w0 = rng.normal(size=n)
    T = rng.binomial(1, 1 / (1 + np.exp(-2 * w0))).astype(float)
    Y = 3 * w0 + (1 + X[:, 0]) * T + rng.normal(size=n)
    cf = sp.causal_forest(Y=Y, T=T, X=X, W=w0, **KW)
    return cf, w0


def test_split_refits_keep_the_controls_of_the_forest(confounded):
    cf, _ = confounded
    assert cf.data_info["n_controls"] == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        halves = _refit_halves(cf, None, 0.5, 0, "rate_split()")
    assert halves.train.data_info["n_controls"] == 1
    assert halves.evaluate.data_info["n_controls"] == 1


def test_split_refits_estimate_nuisances_the_way_the_forest_did(confounded):
    cf, _ = confounded
    source = cf.get_nuisances()["source"]
    assert source == {
        "Y_hat": "grf regression forest (OOB)",
        "W_hat": "grf regression forest (OOB)",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        halves = _refit_halves(cf, None, 0.5, 0, "rate_split()")
    assert halves.train.get_nuisances()["source"] == source
    assert halves.evaluate.get_nuisances()["source"] == source


def test_split_refits_of_a_panel_forest_do_not_use_class_labels_as_w_hat(panel):
    cf, _, _ = panel
    assert len(np.unique(cf.get_nuisances()["W_hat"])) > 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        halves = _refit_halves(cf, None, 0.5, 0, "rate_split()")
    assert len(np.unique(halves.train.get_nuisances()["W_hat"])) > 2


def test_split_refits_carry_the_rows_of_user_supplied_nuisances(confounded):
    cf, w0 = confounded
    nu = cf.get_nuisances()
    supplied = sp.causal_forest(
        Y=cf._Y_original,
        T=cf._T_original,
        X=cf._X_original,
        W=w0,
        Y_hat=nu["Y_hat"],
        W_hat=nu["W_hat"],
        **KW,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        halves = _refit_halves(supplied, None, 0.5, 0, "rate_split()")
    for half, rows in (
        (halves.train, halves.in_train),
        (halves.evaluate, halves.in_eval),
    ):
        got = half.get_nuisances()
        assert got["source"] == {"Y_hat": "user-supplied", "W_hat": "user-supplied"}
        np.testing.assert_array_equal(got["Y_hat"], nu["Y_hat"][rows])
        np.testing.assert_array_equal(got["W_hat"], nu["W_hat"][rows])
        # The confounder is adjusted for on each side (naive difference: 5).
        assert abs(half.average_treatment_effect()["estimate"] - 1.0) < 0.6


def test_split_halves_partition_units_and_keep_feature_names(panel):
    cf, df, _ = panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        halves = _refit_halves(cf, None, 0.4, 3, "rate_split()")
    unit = df["unit"].to_numpy()
    assert not np.any(halves.in_train & halves.in_eval)
    assert np.all(halves.in_train | halves.in_eval) and halves.dropped == 0
    # No unit is on both sides, and the requested share of units trains.
    assert set(unit[halves.in_train]).isdisjoint(unit[halves.in_eval])
    assert len(set(unit[halves.in_train])) == round(0.4 * 70)
    assert halves.split_by == "units"
    assert halves.train._feature_names == ["z", "xv"]
    assert halves.n_groups(halves.in_eval, cf, None) == 70 - round(0.4 * 70)
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(MethodIncompatibility, match="train_frac"):
            _refit_halves(cf, None, bad, 0, "rate_split()")


def test_rate_split_scores_a_rule_fitted_on_the_other_half(panel):
    cf, df, _ = panel
    X = df[["z", "xv"]].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        halves = _refit_halves(cf, None, 0.5, 5, "rate_split()")
        prio = halves.train.effect(X[halves.in_eval])
        direct = sp.rate(halves.evaluate, priorities=prio, variance="bjs")
    with pytest.warns(AssumptionWarning, match="n_splits=1 reports one draw"):
        one = sp.rate_split(cf, n_splits=1, random_state=5)
    assert one["estimate"] == pytest.approx(direct["estimate"], abs=ATOL)
    assert one["se"] == pytest.approx(direct["se"], abs=ATOL)
    assert one["priority_source"] == "held_out_forest"
    assert one["n_train_units"] + one["n_eval_units"] == 70
    assert one["split_by"] == "units" and one["n_splits"] == 1
    assert one["method"].endswith("split-sample")


def test_rate_split_aggregates_splits_by_the_median_rule(panel):
    cf, _, _ = panel
    alpha = 0.10
    singles = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(3):
            singles.append(
                sp.rate_split(cf, n_splits=1, random_state=20 + b, alpha=alpha / 2)
            )
        pooled = sp.rate_split(cf, n_splits=3, random_state=20, alpha=alpha)
    est = np.array([r["estimate"] for r in singles])
    se = np.array([r["se"] for r in singles])
    assert pooled["estimate"] == pytest.approx(np.median(est), abs=ATOL)
    assert pooled["se"] == pytest.approx(np.median(se), abs=ATOL)
    # Conditional intervals at 1 - alpha / 2, then their medians.
    assert pooled["ci_low"] == pytest.approx(
        np.median([r["ci_low"] for r in singles]), abs=ATOL
    )
    assert pooled["ci_high"] == pytest.approx(
        np.median([r["ci_high"] for r in singles]), abs=ATOL
    )
    p_cond = 2 * stats.norm.sf(np.abs(est / se))
    assert pooled["p"] == pytest.approx(min(1.0, 2 * np.median(p_cond)), rel=1e-8)
    assert pooled["n_splits"] == 3 and pooled["n_splits_skipped"] == 0
    assert pooled["estimate_min"] == pytest.approx(est.min(), abs=ATOL)
    assert pooled["estimate_max"] == pytest.approx(est.max(), abs=ATOL)
    assert pooled["alpha"] == alpha and "VEIN over 3 splits" in pooled["method"]
    for bad in (0, 1.5, True):
        with pytest.raises(MethodIncompatibility, match="n_splits"):
            sp.rate_split(cf, n_splits=bad)


def test_policy_tree_prices_the_rule_on_the_held_out_half(panel):
    cf, df, _ = panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        free = sp.forest_policy_tree(cf, depth=1, n_splits=1, random_state=2)
        costly = sp.forest_policy_tree(
            cf, depth=1, n_splits=1, random_state=2, cost=0.4
        )
        halves = _refit_halves(cf, None, 0.5, 2, "forest_policy_tree()")
    # Pricing "treat every cell" is the held-out half's imputation ATT.
    sub = df[halves.in_eval]
    scores = _imputation_scores(
        sub["y"].to_numpy(),
        sub["d"].to_numpy(),
        sub["unit"].to_numpy(),
        sub["time"].to_numpy(),
    )
    att = np.nanmean(scores[sub["d"].to_numpy() == 1])
    assert free["value_treat_all"]["estimate"] == pytest.approx(att, abs=ATOL)
    # A known cost shifts that value one for one, and not its variance.
    assert costly["value_treat_all"]["estimate"] == pytest.approx(att - 0.4, abs=ATOL)
    assert costly["value_treat_all"]["se"] == pytest.approx(
        free["value_treat_all"]["se"], rel=1e-9
    )
    for out in (free, costly):
        # The gain's estimate is exactly the difference of the other two.
        assert out["gain_over_treat_all"]["estimate"] == pytest.approx(
            out["value"]["estimate"] - out["value_treat_all"]["estimate"], abs=1e-10
        )
        # The value of the rule is the mean of (score - cost) * policy.
        rows = sub["d"].to_numpy() == 1
        pi = out["policy"]
        assert pi.shape == (int(np.isfinite(scores[rows]).sum()),)
        assert out["value"]["estimate"] == pytest.approx(
            np.mean((scores[rows] - out["cost"]) * pi), abs=ATOL
        )
        assert out["share_treated"] == pytest.approx(pi.mean())
        assert out["policy_covariates"] == ["z", "xv"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for bad in (0, 1.5, True):
            with pytest.raises(MethodIncompatibility, match="depth"):
                sp.forest_policy_tree(cf, depth=bad, n_splits=1)
        with pytest.raises(MethodIncompatibility, match="not in the forest's"):
            sp.forest_policy_tree(cf, x=["nope"], n_splits=1)
        with pytest.raises(MethodIncompatibility, match="one row per training"):
            sp.forest_policy_tree(cf, x=np.zeros((5, 1)), n_splits=1)
        named = sp.forest_policy_tree(cf, x=["z"], depth=1, n_splits=1)
    assert named["policy_covariates"] == ["z"]


# --------------------------------------------------------------------------- #
#  Tuning
# --------------------------------------------------------------------------- #


def test_tuning_scores_settings_by_the_r_loss_on_fixed_nuisances():
    rng = np.random.default_rng(408)
    n = 240
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, n).astype(float)
    Y = X[:, 0] + T * (1 + 2 * (X[:, 1] > 0)) + rng.normal(size=n)
    names = (
        "min_samples_leaf",
        "max_samples",
        "mtry",
        "honesty_fraction",
        "honesty_prune_leaves",
        "imbalance_penalty",
    )
    out = sp.tune_causal_forest(
        Y=Y, T=T, X=X, parameters=names, n_draws=4, tune_trees=20, tune_reps=2,
        n_estimators=40, random_state=6,
    )  # fmt: skip
    trials = out["trials"]
    assert len(trials) == 5 and (trials["setting"] == "default").sum() == 1
    draws = trials[trials["setting"] == "draw"]
    assert draws["max_samples"].between(0.05, 0.5).all()
    assert draws["mtry"].between(1, 3).all()
    assert draws["honesty_fraction"].between(0.5, 0.8).all()
    assert draws["imbalance_penalty"].between(0.0, 2.0).all()
    assert draws["min_samples_leaf"].between(1, n / 16).all()
    assert list(trials["error"]) == sorted(trials["error"])
    # The default setting's error, rebuilt from the documented recipe.
    seed0 = int(np.random.default_rng(6).integers(0, 2**31 - 1))
    base = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=40, random_state=seed0)
    nu = base.get_nuisances()
    losses = []
    for rep in (1, 2):
        trial = sp.causal_forest(
            Y=Y, T=T, X=X, n_estimators=20, random_state=seed0 + 7919 * rep,
            Y_hat=nu["Y_hat"], W_hat=nu["W_hat"],
        )  # fmt: skip
        tau = trial.predict()
        ok = np.isfinite(tau)
        resid = (Y - nu["Y_hat"])[ok] - (T - nu["W_hat"])[ok] * tau[ok]
        losses.append(np.mean(resid**2))
    assert out["default_error"] == pytest.approx(np.mean(losses), rel=1e-10)
    assert out["tuned_error"] <= out["default_error"]
    if out["tuned"]:
        assert set(out["best_params"]) == set(names)
        assert isinstance(out["best_params"]["honesty_prune_leaves"], bool)
    else:
        assert out["best_params"] == {}
        np.testing.assert_array_equal(out["forest"].predict(), base.predict())


def test_tuning_argument_rules():
    rng = np.random.default_rng(409)
    X = rng.normal(size=(60, 2))
    T = rng.binomial(1, 0.5, 60).astype(float)
    Y = rng.normal(size=60)
    args = dict(Y=Y, T=T, X=X)
    with pytest.raises(MethodIncompatibility, match="cannot tune"):
        sp.tune_causal_forest(parameters=("depth",), **args)
    with pytest.raises(MethodIncompatibility, match="cannot tune nothing"):
        sp.tune_causal_forest(parameters=(), **args)
    with pytest.raises(MethodIncompatibility, match="both tuned and fixed"):
        sp.tune_causal_forest(parameters=("mtry",), mtry=1, **args)
    for bad in (dict(n_draws=0), dict(tune_trees=5), dict(tune_reps=0)):
        with pytest.raises(MethodIncompatibility, match="are needed"):
            sp.tune_causal_forest(**bad, **args)
