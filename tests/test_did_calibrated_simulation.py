"""Calibrated placebo simulation for DiD estimator selection.

The harness has no reference implementation to align against, so the
evidence is T1: properties that hold by construction and can be checked
exactly (the calibration zeroes the imputation event study; the injected
effect is recovered), plus Monte Carlo checks of size and of the known
negative-weighting failure of TWFE under staggered dynamic effects.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did import calibrated_simulation as cal
from statspai.workflow._degradation import WorkflowDegradedWarning

FAST = ["twfe", "callaway_santanna", "did_imputation", "gardner_did"]


def _panel(
    rng,
    *,
    n_units=40,
    T=6,
    cohorts=(3, 4, 5),
    effect=0.0,
    dynamic=0.0,
    rho=0.5,
):
    """Staggered panel with unit and period effects and AR(1) errors.

    ``effect`` is the impact effect and ``dynamic`` its growth per period
    since adoption, so ``dynamic > 0`` makes the effect heterogeneous across
    cohorts at any calendar date -- the case TWFE mis-weights.
    """
    rows = []
    labels = list(cohorts) + [0]
    for i in range(n_units):
        g = labels[i % len(labels)]
        alpha = rng.normal(0, 1)
        e = np.empty(T)
        e[0] = rng.normal()
        for k in range(1, T):
            e[k] = rho * e[k - 1] + rng.normal(0, np.sqrt(1 - rho**2))
        for k in range(T):
            t = k + 1
            d = 1.0 if (g > 0 and t >= g) else 0.0
            tau = (effect + dynamic * (t - g)) if d else 0.0
            rows.append(
                {
                    "unit": i,
                    "year": 2000 + k,
                    "g": 0 if g == 0 else 2000 + g - 1,
                    "y": alpha + 0.3 * k + tau + 0.5 * e[k],
                }
            )
    return pd.DataFrame(rows)


def _frame_from(panel):
    """The calibrated panel as a frame the estimators can read."""
    return pd.DataFrame(
        {
            "y": panel.y0,
            "id": panel.uid,
            "t": panel.tid,
            "g": panel.cohort[panel.uid],
        }
    )


# ----------------------------------------------------------------------
# The calibration step
# ----------------------------------------------------------------------


def test_calibration_zeroes_the_imputation_event_study():
    """By construction the BJS fit on the calibrated panel returns zero."""
    rng = np.random.default_rng(0)
    df = _panel(rng, effect=1.5, dynamic=0.4)
    panel, _ = cal._prepare(df, "y", "unit", "year", "g")
    info = cal._calibrate(panel, "imputation")
    fit = sp.did_imputation(
        _frame_from(panel), y="y", group="id", time="t", first_treat="g", cluster="id"
    )
    assert abs(fit.estimate) < 1e-10
    # ... and the effect it removed is the one that was planted.
    assert info["removed_effect_by_horizon"][0] == pytest.approx(1.5, abs=0.25)
    assert info["removed_effect_by_horizon"][2] == pytest.approx(2.3, abs=0.35)


def test_calibrate_none_leaves_the_outcome_alone():
    rng = np.random.default_rng(1)
    df = _panel(rng, effect=1.0)
    panel, _ = cal._prepare(df, "y", "unit", "year", "g")
    before = panel.y0.copy()
    info = cal._calibrate(panel, "none")
    assert np.array_equal(panel.y0, before)
    assert info["removed_effect_by_horizon"] == {}


def test_two_way_fit_matches_a_dummy_regression():
    rng = np.random.default_rng(2)
    df = _panel(rng)
    panel, _ = cal._prepare(df, "y", "unit", "year", "g")
    fitted = cal._two_way_fit(panel.y0, panel.uid, panel.tid)
    X = np.column_stack(
        [
            np.ones(panel.uid.size),
            pd.get_dummies(panel.uid, drop_first=True).astype(float).to_numpy(),
            pd.get_dummies(panel.tid, drop_first=True).astype(float).to_numpy(),
        ]
    )
    dense = X @ np.linalg.lstsq(X, panel.y0, rcond=None)[0]
    assert np.allclose(fitted, dense, atol=1e-9)


# ----------------------------------------------------------------------
# Recovery, size and the TWFE failure
# ----------------------------------------------------------------------


def test_constant_effect_is_recovered_by_every_estimator():
    pytest.importorskip("pyfixest")  # the TWFE estimator runs on sp.feols
    rng = np.random.default_rng(3)
    df = _panel(rng)
    study = sp.did_calibrated_simulation(
        df,
        y="y",
        id="unit",
        time="year",
        cohort="g",
        estimators=FAST,
        effect=0.8,
        n_sims=40,
        seed=5,
    )
    assert study.model_info["truth_mean"] == pytest.approx(0.8, abs=1e-12)
    assert not study.model_info["heterogeneous_effect"]
    for row in study.table.itertuples():
        assert row.n_ok == 40, row.estimator
        assert abs(row.bias) < 3.0 * row.mc_se_bias + 0.02, row.estimator
        # The reported SEs track the dispersion they claim to measure.
        assert 0.7 < row.se_ratio < 1.4, row.estimator


def test_size_and_coverage_under_a_true_null():
    rng = np.random.default_rng(4)
    df = _panel(rng, n_units=48)
    study = sp.did_calibrated_simulation(
        df,
        y="y",
        id="unit",
        time="year",
        cohort="g",
        estimators=["callaway_santanna", "did_imputation"],
        effect=0.0,
        n_sims=60,
        seed=7,
    )
    assert study.model_info["truth_mean"] == 0.0
    for row in study.table.itertuples():
        # 60 draws put the Monte Carlo SE of a 5% rate at 2.8 points, so the
        # assertion is against gross size failure, not a sharp equality.
        assert row.reject_rate < 0.20, row.estimator
        assert row.coverage > 0.85, row.estimator


def test_twfe_is_pulled_below_the_truth_by_dynamic_effects():
    """The Goodman-Bacon bad comparison, measured rather than asserted."""
    pytest.importorskip("pyfixest")  # the TWFE estimator runs on sp.feols
    rng = np.random.default_rng(6)
    df = _panel(rng, n_units=48)

    def growing(horizon):
        return 0.5 + 0.5 * horizon

    study = sp.did_calibrated_simulation(
        df,
        y="y",
        id="unit",
        time="year",
        cohort="g",
        estimators=["twfe", "did_imputation"],
        effect=growing,
        n_sims=40,
        seed=9,
    )
    assert study.model_info["heterogeneous_effect"]
    table = study.table.set_index("estimator")
    twfe, bjs = table.loc["twfe"], table.loc["did_imputation"]
    assert abs(bjs["bias"]) < 3.0 * bjs["mc_se_bias"] + 0.02
    assert twfe["bias"] < bjs["bias"] - 3.0 * twfe["mc_se_bias"]
    assert study.best("rmse") == "did_imputation"


def test_seed_is_reproducible_and_shifts_with_the_seed():
    rng = np.random.default_rng(7)
    df = _panel(rng, n_units=24)
    kw = dict(
        y="y",
        id="unit",
        time="year",
        cohort="g",
        estimators=["did_imputation"],
        effect=0.3,
        n_sims=6,
    )
    a = sp.did_calibrated_simulation(df, seed=3, **kw)
    b = sp.did_calibrated_simulation(df, seed=3, **kw)
    c = sp.did_calibrated_simulation(df, seed=4, **kw)
    fixed = [c for c in a.draws.columns if c != "seconds"]  # wall clock varies
    pd.testing.assert_frame_equal(a.draws[fixed], b.draws[fixed])
    assert not np.allclose(a.draws["estimate"], c.draws["estimate"])


@pytest.mark.parametrize("assignment", ["resample_cohorts", "random_timing"])
def test_assignment_schemes_keep_the_design_estimable(assignment):
    rng = np.random.default_rng(8)
    df = _panel(rng, n_units=32)
    study = sp.did_calibrated_simulation(
        df,
        y="y",
        id="unit",
        time="year",
        cohort="g",
        estimators=["did_imputation"],
        assignment=assignment,
        n_sims=8,
        seed=2,
    )
    assert study.table.loc[0, "n_ok"] == 8


@pytest.mark.parametrize("resample", ["units", "wild", "none"])
def test_resample_schemes(resample):
    rng = np.random.default_rng(9)
    df = _panel(rng, n_units=32)
    study = sp.did_calibrated_simulation(
        df,
        y="y",
        id="unit",
        time="year",
        cohort="g",
        estimators=["did_imputation"],
        resample=resample,
        n_sims=6,
        seed=2,
    )
    assert study.table.loc[0, "n_ok"] == 6


def test_wild_resampling_works_on_an_unbalanced_panel_where_units_cannot():
    rng = np.random.default_rng(10)
    df = _panel(rng, n_units=32)
    unbalanced = df.drop(df.index[[5, 17, 40]])
    with pytest.raises(sp.MethodIncompatibility, match="balanced"):
        sp.did_calibrated_simulation(
            unbalanced,
            y="y",
            id="unit",
            time="year",
            cohort="g",
            estimators=["did_imputation"],
            n_sims=4,
        )
    study = sp.did_calibrated_simulation(
        unbalanced,
        y="y",
        id="unit",
        time="year",
        cohort="g",
        estimators=["did_imputation"],
        resample="wild",
        n_sims=4,
        seed=1,
    )
    assert study.model_info["balanced"] is False
    assert study.table.loc[0, "n_ok"] == 4


# ----------------------------------------------------------------------
# Loud failure
# ----------------------------------------------------------------------


def test_failed_draws_are_recorded_and_degrade_loudly(monkeypatch):
    rng = np.random.default_rng(11)
    df = _panel(rng, n_units=24)

    def boom(frame, opt):
        raise RuntimeError("synthetic estimator failure")

    monkeypatch.setitem(cal._ADAPTERS, "twfe", boom)
    with pytest.warns(WorkflowDegradedWarning, match="synthetic estimator failure"):
        study = sp.did_calibrated_simulation(
            df,
            y="y",
            id="unit",
            time="year",
            cohort="g",
            estimators=["twfe", "did_imputation"],
            n_sims=4,
            seed=1,
        )
    table = study.table.set_index("estimator")
    assert table.loc["twfe", "n_ok"] == 0
    assert table.loc["twfe", "n_failed"] == 4
    assert np.isnan(table.loc["twfe", "rmse"])
    assert len(study.failures) == 4
    assert study.degradations and "estimator:twfe" in study.degradations[0]["section"]
    # A dead estimator never wins the ranking.
    assert study.best("rmse") == "did_imputation"
    assert "failed draws" in study.summary()


def test_always_treated_units_are_dropped_with_a_warning():
    rng = np.random.default_rng(12)
    df = _panel(rng, n_units=32)
    df.loc[df["unit"] < 4, "g"] = 2000  # treated in the first period
    with pytest.warns(sp.AssumptionWarning, match="first period"):
        study = sp.did_calibrated_simulation(
            df,
            y="y",
            id="unit",
            time="year",
            cohort="g",
            estimators=["did_imputation"],
            n_sims=4,
            seed=1,
        )
    assert study.model_info["n_always_treated_dropped"] == 4
    assert study.model_info["n_units"] == 28


def test_no_never_treated_unit_raises():
    rng = np.random.default_rng(13)
    df = _panel(rng, n_units=30, cohorts=(3, 4, 5))
    df = df[df["g"] != 0]
    with pytest.raises(sp.DataInsufficient, match="never-treated"):
        sp.did_calibrated_simulation(
            df, y="y", id="unit", time="year", cohort="g", n_sims=4
        )


def test_no_treated_unit_raises():
    rng = np.random.default_rng(14)
    df = _panel(rng, n_units=30)
    df["g"] = 0
    with pytest.raises(sp.DataInsufficient, match="No treated unit"):
        sp.did_calibrated_simulation(
            df, y="y", id="unit", time="year", cohort="g", n_sims=4
        )


def test_time_varying_cohort_raises():
    rng = np.random.default_rng(15)
    df = _panel(rng, n_units=30)
    df.loc[df.index[:3], "g"] = 2005  # unit 0 adopts in 2002 elsewhere
    with pytest.raises(sp.MethodIncompatibility, match="constant within a unit"):
        sp.did_calibrated_simulation(
            df, y="y", id="unit", time="year", cohort="g", n_sims=4
        )


def test_duplicate_unit_period_rows_raise():
    rng = np.random.default_rng(16)
    df = pd.concat([_panel(rng, n_units=30)] * 2, ignore_index=True)
    with pytest.raises(sp.MethodIncompatibility, match="More than one row"):
        sp.did_calibrated_simulation(
            df, y="y", id="unit", time="year", cohort="g", n_sims=4
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"estimators": ["nope"]}, "Unknown estimator"),
        ({"estimators": []}, "empty"),
        ({"assignment": "shuffle"}, "not one of"),
        ({"calibrate": "twfe"}, "not one of"),
        ({"resample": "bayes"}, "not one of"),
        ({"control_group": "anyone"}, "control_group"),
        ({"alpha": 0.0}, "alpha"),
        ({"n_sims": 1}, "n_sims"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"assignment": "observed", "resample": "none"}, "redraws nothing"),
        ({"effect": lambda g, t, extra: 0.0}, "one argument"),
        ({"y": "missing"}, "Columns not found"),
    ],
)
def test_argument_errors(kwargs, match):
    rng = np.random.default_rng(17)
    df = _panel(rng, n_units=24)
    call = dict(y="y", id="unit", time="year", cohort="g", n_sims=4)
    call.update(kwargs)
    with pytest.raises(sp.MethodIncompatibility, match=match):
        sp.did_calibrated_simulation(df, **call)


def test_too_few_periods_raises():
    rng = np.random.default_rng(18)
    df = _panel(rng, n_units=24, T=2, cohorts=(2,))
    with pytest.raises(sp.DataInsufficient, match="at least 3 periods"):
        sp.did_calibrated_simulation(
            df, y="y", id="unit", time="year", cohort="g", n_sims=4
        )


# ----------------------------------------------------------------------
# Result surface
# ----------------------------------------------------------------------


def test_result_surface_and_aliases():
    rng = np.random.default_rng(19)
    df = _panel(rng, n_units=24).rename(columns={"g": "first_treat"})
    study = sp.did_calibrated_simulation(
        df,
        y="y",
        unit="unit",  # alias for id=
        time="year",
        first_treat="first_treat",  # alias for cohort=
        estimators=["bjs", "did2s"],  # short spellings
        effect=0.4,
        n_sims=5,
        seed=1,
    )
    assert list(study.table["estimator"]) == ["did_imputation", "gardner_did"]
    assert len(study.draws) == 10
    assert study.failures.empty
    assert study.n_sims == 5
    assert "Calibrated placebo simulation" in study.summary()
    assert study.cite()  # the harness carries its own reference
    with pytest.raises(sp.MethodIncompatibility, match="criterion"):
        study.best("elegance")


def test_estimators_are_deduplicated_and_ordered():
    assert cal._resolve_estimators(["cs", "callaway_santanna", "bjs"]) == [
        "callaway_santanna",
        "did_imputation",
    ]
