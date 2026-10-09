"""Coverage campaign (did, Oct 2026) -- stacked DiD and the calibrated simulation.

Two modules whose remaining gaps are input guards and estimator adapters:

* ``sp.stacked_did``: every refusal is checked for its exception type and
  message, and the options that must reproduce another call (zero weights =
  dropped rows, unit weights = unweighted) are checked as identities.
* ``sp.did_calibrated_simulation``: the five estimator adapters that the
  default line-up does not include are run on a nearly noise-free panel with
  a constant injected effect, which every one of them must recover.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did import calibrated_simulation as cal
from statspai.exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
)

# ``statspai.did.stacked_did`` the attribute is the function; the module of
# the same name has to be fetched from the import system.
sd_mod = importlib.import_module("statspai.did.stacked_did")


def _staggered(seed=0, n_units=30, T=8, cohorts=(4, 6, 0), effect=2.0, noise=0.05):
    """Balanced staggered panel, unit and period effects, constant effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        g = cohorts[i % len(cohorts)]
        a = rng.normal()
        for t in range(1, T + 1):
            d = float(g > 0 and t >= g)
            rows.append(
                {
                    "unit": i,
                    "t": t,
                    "g": g,
                    "d": d,
                    "y": a + 0.3 * t + effect * d + noise * rng.normal(),
                }
            )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
#  sp.stacked_did
# ══════════════════════════════════════════════════════════════════════


def test_stacked_recovers_constant_effect():
    df = _staggered()
    r = sp.stacked_did(
        df, y="y", group="unit", time="t", first_treat="g", window=(-2, 2)
    )
    # truth 2.0, noise sd 0.05, 20 treated units: the ATT's sampling sd is
    # about 0.02, so 0.1 is five standard errors.
    assert r.estimate == pytest.approx(2.0, abs=0.1)
    assert r.model_info["n_cohorts"] == 2


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(family="binomial"), "family must be"),
        (dict(spec="dynamic"), "spec must be"),
        (dict(control_group="clean"), "control_group must be"),
        (dict(weights="nope"), "Weight column 'nope' not found"),
        (dict(cluster="nope"), "Cluster column 'nope' not found"),
    ],
)
def test_stacked_rejects_unknown_options(kwargs, match):
    df = _staggered()
    with pytest.raises(MethodIncompatibility, match=match):
        sp.stacked_did(
            df, y="y", group="unit", time="t", first_treat="g", window=(-2, 2), **kwargs
        )


def test_stacked_rejects_negative_or_missing_weights():
    df = _staggered()
    df["w"] = 1.0
    df.loc[3, "w"] = -1.0
    with pytest.raises(MethodIncompatibility, match="finite and non-negative"):
        sp.stacked_did(
            df,
            y="y",
            group="unit",
            time="t",
            first_treat="g",
            window=(-2, 2),
            weights="w",
        )


def test_stacked_needs_first_treat_or_a_stack():
    df = _staggered()
    # first_treat=None is caught by the generic column check first
    with pytest.raises(ValueError, match="not found in data"):
        sp.stacked_did(df, y="y", group="unit", time="t")


def test_stacked_events_argument_guards():
    df = _staggered()
    df["ev"] = ((df["g"] > 0) & (df["t"] == df["g"])).astype(int)
    base = dict(y="y", group="unit", time="t", window=(-2, 2))
    with pytest.raises(MethodIncompatibility, match="own_overlap must be"):
        sp.stacked_did(df, events="ev", own_overlap="maybe", **base)
    with pytest.raises(MethodIncompatibility, match="Column 'nope' not found"):
        sp.stacked_did(df, events="nope", **base)
    with pytest.raises(MethodIncompatibility, match="not several"):
        sp.stacked_did(df, events="ev", first_treat="g", **base)


def test_stacked_prebuilt_stack_guards():
    df = _staggered()
    df["stack"] = 1
    df["tr"] = (df["g"] == 4).astype(int)
    df["rel"] = df["t"] - 4
    base = dict(y="y", group="unit", time="t", window=(-2, 2), event_id="stack")
    with pytest.raises(MethodIncompatibility, match="missing \\['event_time'\\]"):
        sp.stacked_did(df, treated="tr", **base)
    with pytest.raises(MethodIncompatibility, match="Column 'nope' not found"):
        sp.stacked_did(df, treated="tr", event_time="nope", **base)
    bad = df.assign(tr=2)
    with pytest.raises(MethodIncompatibility, match="must be 0/1"):
        sp.stacked_did(bad, treated="tr", event_time="rel", **base)
    far = df.assign(rel=50)
    with pytest.raises(DataInsufficient, match="fall inside window"):
        sp.stacked_did(far, treated="tr", event_time="rel", **base)


def test_stacked_prebuilt_stack_equals_builtin_stack_for_one_cohort():
    # One cohort against never-treated units: the stack is the panel itself.
    df = _staggered(cohorts=(4, 0))
    builtin = sp.stacked_did(
        df, y="y", group="unit", time="t", first_treat="g", window=(-2, 2)
    )
    pre = df.assign(stack=1, tr=(df["g"] == 4).astype(int), rel=df["t"] - 4)
    mine = sp.stacked_did(
        pre,
        y="y",
        group="unit",
        time="t",
        window=(-2, 2),
        event_id="stack",
        treated="tr",
        event_time="rel",
    )
    # same rows, same regression: equal up to floating point
    assert mine.estimate == pytest.approx(builtin.estimate, abs=1e-10)
    assert mine.se == pytest.approx(builtin.se, rel=1e-8)


def test_stacked_no_controls_means_no_sub_experiment():
    df = _staggered(cohorts=(4, 6))  # nobody is never treated
    with pytest.raises(ValueError, match="No valid sub-experiments"):
        sp.stacked_did(
            df, y="y", group="unit", time="t", first_treat="g", window=(-2, 2)
        )
    one = _staggered(cohorts=(4,))
    with pytest.raises(ValueError, match="No valid sub-experiments"):
        sp.stacked_did(
            one,
            y="y",
            group="unit",
            time="t",
            first_treat="g",
            window=(-2, 2),
            control_group="notyettreated_rows",
        )


def test_stacked_cohort_outside_the_panel_is_skipped():
    df = _staggered(cohorts=(4, 0, 40))  # cohort 40 adopts long after T = 8
    r = sp.stacked_did(
        df, y="y", group="unit", time="t", first_treat="g", window=(-2, 2)
    )
    assert r.model_info["n_cohorts"] == 1
    ref = sp.stacked_did(
        df[df["g"] != 40],
        y="y",
        group="unit",
        time="t",
        first_treat="g",
        window=(-2, 2),
    )
    # the skipped cohort contributes no rows, so its units are simply absent
    assert r.estimate == pytest.approx(ref.estimate, abs=1e-10)


def test_stacked_window_without_post_period_data():
    df = _staggered(cohorts=(9, 0))  # adoption one period after the panel ends
    with pytest.raises(DataInsufficient, match="no post-treatment"):
        sp.stacked_did(
            df, y="y", group="unit", time="t", first_treat="g", window=(-3, 1)
        )
    with pytest.raises(ValueError, match="Not enough relative time"):
        sp.stacked_did(
            df, y="y", group="unit", time="t", first_treat="g", window=(-1, 1)
        )


def test_stacked_zero_weight_and_missing_rows_equal_dropping_them():
    df = _staggered()
    df["w"] = 1.0
    drop_w = df["unit"].isin([2, 5])  # two never-treated units
    df.loc[drop_w, "w"] = 0.0
    miss = (df["unit"] == 8) & (df["t"] == 3)
    df.loc[miss, "y"] = np.nan
    kw = dict(y="y", group="unit", time="t", first_treat="g", window=(-2, 2))
    weighted = sp.stacked_did(df, weights="w", **kw)
    ref = sp.stacked_did(df.loc[~drop_w & ~miss], **kw)
    # identical estimating sample and unit weights: floating-point equality
    assert weighted.estimate == pytest.approx(ref.estimate, abs=1e-10)
    assert weighted.se == pytest.approx(ref.se, rel=1e-8)


def test_stacked_poisson_unit_weights_equal_unweighted():
    df = _staggered(effect=0.4)
    df["y"] = np.exp(df["y"] / 4.0)
    df["w"] = 1.0
    kw = dict(
        y="y",
        group="unit",
        time="t",
        first_treat="g",
        window=(-2, 2),
        family="poisson",
        spec="pooled",
    )
    a = sp.stacked_did(df, **kw)
    b = sp.stacked_did(df, weights="w", **kw)
    # tolerance: both are IRLS fits stopped by the same rule
    assert b.estimate == pytest.approx(a.estimate, abs=1e-8)
    assert b.se == pytest.approx(a.se, rel=1e-6)
    # log-linear effect of 0.4 on y/4 is 0.1 in the exponent
    assert a.estimate == pytest.approx(0.1, abs=0.02)


def test_stacked_events_every_event_dropped_for_own_overlap():
    df = _staggered(cohorts=(0,), n_units=6)
    # unit 0 has two events two periods apart: each sits in the other's window
    df["ev"] = ((df["unit"] == 0) & df["t"].isin([4, 6])).astype(int)
    kw = dict(y="y", group="unit", time="t", window=(-2, 2), events="ev")
    with pytest.raises(DataInsufficient, match="no event has a usable window"):
        sp.stacked_did(df, **kw)
    kept = sp.stacked_did(df, own_overlap="keep", **kw)
    assert kept.model_info["n_cohorts"] == 2


def test_stacked_legacy_helpers_reproduce_the_pooled_regression():
    # The module keeps an alternating-projection demeaner and a small OLS /
    # cluster-variance pair. On a single-cohort stack they must give the
    # coefficient of the pooled specification, and the Liang-Zeger formula.
    df = _staggered(cohorts=(4, 0))
    ref = sp.stacked_did(
        df,
        y="y",
        group="unit",
        time="t",
        first_treat="g",
        window=(-3, 4),
        spec="pooled",
    )
    y = df["y"].to_numpy()
    X = df[["d"]].to_numpy()
    y_dm, X_dm = sd_mod._twoway_demean(y, X, df["unit"].to_numpy(), df["t"].to_numpy())
    # balanced panel: one sweep is exact, and both margins have mean zero
    assert abs(pd.Series(y_dm).groupby(df["unit"]).mean()).max() < 1e-10
    assert abs(pd.Series(y_dm).groupby(df["t"]).mean()).max() < 1e-10
    beta, resid = sd_mod._ols(X_dm, y_dm)
    assert beta[0] == pytest.approx(ref.estimate, abs=1e-10)

    cl = df["unit"].to_numpy()
    se = sd_mod._cluster_robust_se(X_dm, resid, cl)
    n, k = X_dm.shape
    G = len(np.unique(cl))
    scores = pd.Series(X_dm[:, 0] * resid).groupby(cl).sum().to_numpy()
    manual = np.sqrt((G / (G - 1)) * ((n - 1) / (n - k)) * np.sum(scores**2)) / float(
        X_dm[:, 0] @ X_dm[:, 0]
    )
    assert se[0] == pytest.approx(manual, rel=1e-10)

    # degenerate shapes
    b0, r0 = sd_mod._ols(np.empty((n, 0)), y_dm)
    assert b0.size == 0 and np.array_equal(r0, y_dm)
    assert sd_mod._cluster_robust_se(np.empty((n, 0)), y_dm, cl).size == 0
    # a duplicated column makes X'X singular: the pseudo-inverse bread still
    # returns a finite, symmetric matrix
    X2 = np.column_stack([X_dm[:, 0], X_dm[:, 0]])
    V = sd_mod._cluster_robust_vcov(X2, resid, cl)
    assert V.shape == (2, 2) and np.all(np.isfinite(V))
    np.testing.assert_allclose(V, V.T, atol=1e-12)


# ══════════════════════════════════════════════════════════════════════
#  sp.did_calibrated_simulation
# ══════════════════════════════════════════════════════════════════════


def _study(df, estimators, **kw):
    opts = dict(
        y="y",
        id="unit",
        time="t",
        cohort="g",
        effect=1.0,
        assignment="observed",
        calibrate="none",
        resample="wild",
        n_sims=2,
        seed=0,
    )
    opts.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_calibrated_simulation(df, estimators=estimators, **opts)


@pytest.mark.parametrize(
    "alias,name",
    [
        ("sa", "sun_abraham"),
        ("etwfe", "etwfe"),
        ("dcdh", "did_multiplegt_dyn"),
        ("stacked", "stacked_did"),
        ("lpdid", "lp_did"),
    ],
)
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_adapter_recovers_injected_constant_effect(alias, name, control_group):
    if alias == "sa" and control_group == "notyettreated":
        pytest.skip("sp.sun_abraham has no not-yet-treated control group")
    df = _staggered(effect=0.0, noise=0.01, n_units=24, T=6, cohorts=(3, 5, 0))
    study = _study(df, alias, control_group=control_group)
    assert study.model_info["estimators"] == [name]
    assert study.failures.empty
    row = study.table.iloc[0]
    assert row["estimator"] == name and row["n_ok"] == 2
    # A constant effect of 1 is added to a panel whose only departure from
    # exact two-way effects is noise of sd 0.01; each estimate is then within
    # a few hundredths of 1 (0.05 is at least five of its standard errors).
    assert study.draws["estimate"].to_numpy() == pytest.approx(1.0, abs=0.05)
    assert study.model_info["truth_mean"] == pytest.approx(1.0)


def test_sun_abraham_and_etwfe_adapters_coincide_on_a_balanced_panel():
    # Both are the cohort x period saturated regression against never-treated
    # units, aggregated with treated-observation weights.
    df = _staggered(effect=0.0, noise=0.3, n_units=24, T=6, cohorts=(3, 5, 0))
    study = _study(df, ["sa", "etwfe"])
    wide = study.draws.pivot(index="sim", columns="estimator", values="estimate")
    np.testing.assert_allclose(wide["sun_abraham"], wide["etwfe"], atol=1e-8)


def test_noise_free_panel_is_recorded_as_unusable_not_scored():
    df = _staggered(effect=0.0, noise=0.0, n_units=12, T=5, cohorts=(3, 0))
    study = _study(df, ["etwfe"])
    # exact fit: the estimate is the injected effect but the SE is zero
    assert int(study.table["n_ok"].iloc[0]) == 0
    assert len(study.failures) == 2
    assert study.failures["failure"].str.contains("unusable fit").all()
    with pytest.raises(DataInsufficient, match="Every estimator failed"):
        study.best()


def test_best_by_absolute_bias_and_heterogeneity_note():
    df = _staggered(effect=0.0, noise=0.2, n_units=24, T=6, cohorts=(3, 5, 0))
    study = _study(
        df, ["twfe", "did_imputation"], effect=lambda g, t: 1.0 + 0.5 * (t - g)
    )
    assert study.model_info["heterogeneous_effect"] is True
    assert "estimand difference" in study.summary()
    pick = study.best("abs_bias")
    tab = study.table.set_index("estimator")
    assert abs(tab.loc[pick, "bias"]) == tab["bias"].abs().min()


def test_cohort_codes_outside_the_panel_and_sentinels():
    df = _staggered(effect=0.0, noise=0.2, n_units=24, T=6, cohorts=(3, 5, 0))
    alt = df.copy()
    alt["g"] = alt["g"].astype(object)
    units = alt["unit"]
    alt.loc[units == 2, "g"] = None  # never treated, spelled as missing
    alt.loc[units == 5, "g"] = 99  # adopts after the panel ends
    alt.loc[units == 8, "g"] = np.inf  # never treated, spelled as inf
    panel, _ = _prepared(alt)
    ref, _ = _prepared(df)
    np.testing.assert_array_equal(panel.cohort, ref.cohort)

    early = df.copy()
    early.loc[early["unit"] == 0, "g"] = -3  # treated before the panel starts
    # ... which leaves it without a pre-period, so it is dropped, loudly
    with pytest.warns(AssumptionWarning, match="treated in the first period"):
        p_early, _ = _prepared(early)
    assert p_early.n_units == ref.n_units - 1
    np.testing.assert_array_equal(p_early.cohort, ref.cohort[1:])

    odd = df.copy()
    odd["g"] = odd["g"].astype(object)
    odd.loc[odd["unit"] == 0, "g"] = "soon"
    with pytest.raises(MethodIncompatibility, match="not periods of the panel"):
        _prepared(odd)


def _prepared(df):
    out = cal._prepare(df, "y", "unit", "t", "g")
    return out if isinstance(out, tuple) else (out, None)


def test_prepare_rejects_missing_outcomes_and_tiny_panels():
    df = _staggered(effect=0.0, noise=0.2, n_units=12, T=5, cohorts=(3, 0))
    holes = df.copy()
    holes.loc[4, "y"] = np.nan
    with pytest.raises(MethodIncompatibility, match="must be complete"):
        _study(holes, ["twfe"])
    with pytest.raises(DataInsufficient, match="Only 3 usable units"):
        _study(df[df["unit"] < 3], ["twfe"])


def test_assignment_that_cannot_be_rerandomised_is_refused():
    # A single treated unit: every permutation of the cohorts still has one
    # treated unit, and a design needs two.
    df = _staggered(effect=0.0, noise=0.2, n_units=8, T=5, cohorts=(0,))
    df.loc[df["unit"] == 0, "g"] = 3
    with pytest.raises(DataInsufficient, match="200 draws"):
        _study(df, ["twfe"], assignment="resample_cohorts")
    assert cal._estimable(np.array([0, 3, 3, 3]), 5) is False
    assert cal._estimable(np.array([0, 0, 3, 3]), 5) is True
    # a cohort treated from the first period has no pre-period
    assert cal._estimable(np.array([0, 0, 1, 1]), 5) is False


def test_n_jobs_and_estimator_resolution():
    assert cal._resolve_n_jobs(None) == 1
    assert cal._resolve_n_jobs(3) == 3
    assert cal._resolve_n_jobs(-1) >= 1
    with pytest.raises(MethodIncompatibility):
        cal._resolve_n_jobs(True)
    assert cal._resolve_estimators("bjs") == ["did_imputation"]
    assert cal._as_float("soon") is None and cal._as_float("2.5") == 2.5
