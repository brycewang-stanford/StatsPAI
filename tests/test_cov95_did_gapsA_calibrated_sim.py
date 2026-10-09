"""Coverage gaps in ``sp.did_calibrated_simulation``: the five estimator
adapters the default list leaves out, cohort-coding edge cases, the
assignment redraw loop, the process pool and the result helpers.
"""

import importlib
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from statspai.workflow._degradation import WorkflowDegradedWarning

cal = importlib.import_module("statspai.did.calibrated_simulation")

K = dict(y="y", id="i", time="t", cohort="g")


def _panel(seed=0, n=40, T=6, cohorts=(3, 4, 5), noise=0.3):
    rng = np.random.default_rng(seed)
    labels = list(cohorts) + [0]
    rows = []
    for i in range(n):
        g = labels[i % len(labels)]
        a = rng.normal()
        for t in range(1, T + 1):
            rows.append(
                {"i": i, "t": t, "g": g, "y": a + 0.3 * t + noise * rng.normal()}
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def df():
    return _panel()


# ---------------------------------------------------------------------------
#  Estimator adapters
# ---------------------------------------------------------------------------


def test_remaining_adapters_recover_a_constant_effect(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = sp.did_calibrated_simulation(
            df,
            **K,
            estimators=["sa", "etwfe", "dcdh", "stacked", "lpdid"],
            effect=0.5,
            n_sims=4,
            seed=1,
        )
    assert list(study.table["estimator"]) == [
        "sun_abraham",
        "etwfe",
        "did_multiplegt_dyn",
        "stacked_did",
        "lp_did",
    ]
    assert study.model_info["truth_mean"] == pytest.approx(0.5, abs=1e-12)
    assert study.failures.empty
    table = study.table.set_index("estimator")
    assert (table["n_ok"] == 4).all()
    # Noise sd 0.3 on 40 units: every estimator is within 0.15 of the truth.
    assert (table["bias"].abs() < 0.15).all()
    assert (table["rmse"] < 0.3).all()
    # Saturated cohort x period regressions with never-treated controls:
    # Sun-Abraham's cell-weighted ATT and ETWFE's simple aggregate coincide.
    assert table.loc["sun_abraham", "bias"] == pytest.approx(
        table.loc["etwfe", "bias"], abs=1e-8
    )


def test_not_yet_treated_controls_reach_the_adapters(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = sp.did_calibrated_simulation(
            df,
            **K,
            estimators=["etwfe", "stacked", "lpdid"],
            effect=0.5,
            control_group="notyettreated",
            n_sims=3,
            seed=1,
        )
    assert study.model_info["control_group"] == "notyettreated"
    table = study.table.set_index("estimator")
    assert (table["n_ok"] == 3).all()
    assert (table["bias"].abs() < 0.2).all()


def test_string_estimator_heterogeneous_effect_and_ranking(df):
    study = sp.did_calibrated_simulation(
        df,
        **K,
        estimators="sa",
        effect=lambda h: 0.2 * h,
        assignment="observed",
        resample="units",
        n_sims=3,
        n_jobs=None,
    )
    assert list(study.table["estimator"]) == ["sun_abraham"]
    assert study.model_info["heterogeneous_effect"] is True
    assert "estimand difference, not an error" in study.summary()
    assert study.best("abs_bias") == "sun_abraham"
    # assignment='observed' keeps the real adoption pattern, so the injected
    # truth is the same in every replication.
    assert study.draws["truth"].nunique() == 1


# ---------------------------------------------------------------------------
#  Cohort coding
# ---------------------------------------------------------------------------


def test_cohort_labels_that_are_not_periods_are_refused(df):
    labelled = df.assign(g=df["g"].map({0: "never", 3: "3", 4: "4", 5: "5"}))
    with pytest.raises(MethodIncompatibility, match="not periods of the panel") as e:
        sp.did_calibrated_simulation(labelled, **K, n_sims=2)
    assert e.value.diagnostics["n_unknown"] == 40

    between = df.astype({"g": float})
    between.loc[between["i"] == 0, "g"] = 3.5
    with pytest.raises(MethodIncompatibility, match=r"\['3.5'\]"):
        sp.did_calibrated_simulation(between, **K, n_sims=2)


def test_missing_early_and_late_cohorts_are_recoded(df):
    coded = df.astype({"g": float})
    coded.loc[coded["g"] == 0, "g"] = np.nan  # never treated, as NaN
    coded.loc[coded["i"] == 0, "g"] = -2.0  # adopted before the panel starts
    coded.loc[coded["i"] == 1, "g"] = 99.0  # adopts after it ends
    with pytest.warns(AssumptionWarning, match="treated in the first period"):
        study = sp.did_calibrated_simulation(
            coded, **K, estimators=["bjs"], n_sims=2, seed=0
        )
    # The early adopter is always treated and leaves; the late one stays as
    # a never-treated unit.
    assert study.model_info["n_always_treated_dropped"] == 1
    assert study.model_info["n_units"] == 39
    assert (study.table["n_ok"] == 2).all()


def test_incomplete_and_tiny_panels(df):
    holed = df.copy()
    holed.loc[2, "y"] = np.nan
    with pytest.raises(MethodIncompatibility, match="must be complete") as e:
        sp.did_calibrated_simulation(holed, **K, n_sims=2)
    assert e.value.diagnostics == {"n_missing": 1}
    with pytest.raises(DataInsufficient, match="Only 3 usable units"):
        sp.did_calibrated_simulation(df[df["i"].isin([3, 5, 7])], **K, n_sims=2)


# ---------------------------------------------------------------------------
#  Redrawing the adoption pattern
# ---------------------------------------------------------------------------


def test_a_single_treated_unit_cannot_be_rerandomised(df):
    lonely = df.copy()
    lonely.loc[(lonely["g"] > 0) & (lonely["i"] != 0), "g"] = 0
    with pytest.raises(DataInsufficient, match="200 draws of the adoption pattern"):
        sp.did_calibrated_simulation(lonely, **K, estimators=["bjs"], n_sims=2)


def test_estimable_requires_two_never_treated_units():
    assert cal._estimable(np.array([0, 0, 3, 3]), n_times=6)
    assert not cal._estimable(np.array([0, 3, 3, 4]), n_times=6)  # one control
    assert not cal._estimable(np.array([0, 0, 3]), n_times=6)  # one treated
    assert not cal._estimable(np.array([0, 0, 1, 1]), n_times=6)  # no pre-period


def test_random_timing_redraws_until_the_design_is_estimable():
    # Two never-treated units out of twelve: an i.i.d. redraw of the cohorts
    # leaves fewer than two about a third of the time and has to be repeated.
    thin = _panel(n=12, cohorts=(3, 4, 5, 3, 4))
    assert thin.groupby("g")["i"].nunique().to_dict()[0] == 2
    study = sp.did_calibrated_simulation(
        thin, **K, estimators=["bjs"], assignment="random_timing", n_sims=8, seed=0
    )
    assert (study.table["n_ok"] == 8).all()
    assert study.model_info["assignment"] == "random_timing"


# ---------------------------------------------------------------------------
#  Unusable fits
# ---------------------------------------------------------------------------


def test_a_fit_without_a_standard_error_is_a_failed_draw(df, monkeypatch):
    monkeypatch.setitem(cal._ADAPTERS, "twfe", lambda frame, opt: (1.0, float("nan")))
    with pytest.warns(WorkflowDegradedWarning, match="unusable fit"):
        study = sp.did_calibrated_simulation(
            df, **K, estimators=["twfe"], n_sims=2, seed=0
        )
    assert int(study.table.loc[0, "n_ok"]) == 0
    assert len(study.failures) == 2
    assert study.failures["failure"].str.startswith("NumericalInstability").all()
    with pytest.raises(DataInsufficient, match="Every estimator failed"):
        study.best("rmse")


# ---------------------------------------------------------------------------
#  Parallel execution
# ---------------------------------------------------------------------------


def test_n_jobs_resolution():
    assert cal._resolve_n_jobs(None) == 1
    assert cal._resolve_n_jobs(3) == 3
    assert cal._resolve_n_jobs(-1) == max(1, os.cpu_count() or 1)
    with pytest.raises(MethodIncompatibility, match="positive integer or -1"):
        cal._resolve_n_jobs(0)


def test_process_pool_reproduces_the_serial_draws(df):
    kw = dict(estimators=["bjs"], effect=0.5, n_sims=2, seed=3)
    serial = sp.did_calibrated_simulation(df, **K, **kw)
    pooled = sp.did_calibrated_simulation(df, **K, n_jobs=2, **kw)
    cols = ["sim", "estimator", "estimate", "se", "truth"]
    pd.testing.assert_frame_equal(serial.draws[cols], pooled.draws[cols])
