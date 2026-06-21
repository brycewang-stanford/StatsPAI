"""Coverage round-4 — option branches of the staggered-DiD estimators.

Exercises the under-covered configuration branches (covariates, cluster,
event windows, event-study mode, alternate control sets / methods, and
input-validation raises) of:

- ``sp.sun_abraham``          (interaction-weighted event study)
- ``sp.cohort_anchored_event_study``
- ``sp.lp_did``               (local-projections DiD)
- ``sp.gardner_did``          (two-stage imputation)
- ``sp.continuous_did``       (twfe / dose_response / att_gt / cgs)
- ``sp.did_multiplegt``       (de Chaisemartin-D'Haultfœuille, placebos +
                               dynamics)
- ``sp.did_analysis``         (the full DiD pipeline, several methods)

All DGPs carry a constant +2 ATT switched on at each cohort's adoption
date; assertions check the recovered overall ATT is near +2 (or, for the
continuous dose design, that estimates are finite with valid CIs/SEs),
p-values in [0, 1], CI ordering, and that bad inputs raise.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402

TRUE_ATT = 2.0


def _stag_panel(seed=0, n_units=120, n_periods=10, att=TRUE_ATT):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 5, 7][u % 3]
        fe = rng.normal()
        x = rng.normal()
        st = u % 5  # a coarser cluster than unit
        for t in range(1, n_periods + 1):
            treated_now = (g > 0) and (t >= g)
            treat = 1 if treated_now else 0
            y = (
                fe
                + 0.3 * t
                + 0.4 * x
                + (att if treated_now else 0.0)
                + rng.normal(0, 0.4)
            )
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": y,
                    "g": g,
                    "treat": treat,
                    "first_treat": g,
                    "x": x,
                    "state": st,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _stag_panel()


def _ci_ok(res):
    lo, hi = res.ci
    assert lo <= res.estimate <= hi
    assert 0.0 <= res.pvalue <= 1.0
    assert np.isfinite(res.se) and res.se > 0


# ── Sun-Abraham ──────────────────────────────────────────────────── #


def test_sun_abraham_covariates_cluster_window(panel):
    r = sp.sun_abraham(
        panel,
        y="y",
        g="g",
        t="time",
        i="unit",
        covariates=["x"],
        cluster="state",
        event_window=(-3, 3),
    )
    _ci_ok(r)
    assert "event_study" in (r.model_info or {})


def test_sun_abraham_lastcohort_control(panel):
    r = sp.sun_abraham(
        panel, y="y", g="g", t="time", i="unit", control_group="lastcohort"
    )
    assert np.isfinite(r.estimate)


def test_sun_abraham_bad_control_raises(panel):
    with pytest.raises(ValueError):
        sp.sun_abraham(
            panel, y="y", g="g", t="time", i="unit", control_group="notyettreated"
        )


def test_sun_abraham_missing_col_raises(panel):
    with pytest.raises(ValueError):
        sp.sun_abraham(panel, y="nope", g="g", t="time", i="unit")


def test_sun_abraham_missing_covariate_raises(panel):
    with pytest.raises(ValueError):
        sp.sun_abraham(panel, y="y", g="g", t="time", i="unit", covariates=["nope"])


# ── Cohort-anchored event study ──────────────────────────────────── #


def test_cohort_anchored_cluster(panel):
    r = sp.cohort_anchored_event_study(
        panel, y="y", treat="first_treat", time="time", id="unit", cluster="state"
    )
    assert abs(r.estimate - TRUE_ATT) < 0.8
    _ci_ok(r)
    assert "event_study" in (r.model_info or {})


def test_cohort_anchored_no_cohorts_raises(panel):
    df = panel.assign(first_treat=0)
    with pytest.raises(ValueError):
        sp.cohort_anchored_event_study(
            df, y="y", treat="first_treat", time="time", id="unit"
        )


# ── Local-projections DiD ────────────────────────────────────────── #


def test_lp_did_controls_nevertreated(panel):
    r = sp.lp_did(
        panel,
        y="y",
        unit="unit",
        time="time",
        treatment="treat",
        controls=["x"],
        clean_controls="never_treated",
    )
    assert abs(r.estimate - TRUE_ATT) < 1.0
    _ci_ok(r)


def test_lp_did_horizons(panel):
    r = sp.lp_did(
        panel, y="y", unit="unit", time="time", treatment="treat", horizons=(-2, 4)
    )
    assert "event_study" in (r.model_info or {}) or r.detail is not None


# ── Gardner two-stage ────────────────────────────────────────────── #


def test_gardner_event_study_controls(panel):
    r = sp.gardner_did(
        panel,
        y="y",
        group="unit",
        time="time",
        first_treat="first_treat",
        controls=["x"],
        event_study=True,
    )
    assert abs(r.estimate - TRUE_ATT) < 1.0
    _ci_ok(r)
    assert "event_study" in (r.model_info or {})


# ── Continuous-treatment DiD ─────────────────────────────────────── #


def _dose_panel(seed=3, n_units=120, n_periods=10):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 5, 7][u % 3]
        fe = rng.normal()
        x = rng.normal()
        dose = abs(x) if g > 0 else 0.0
        for t in range(1, n_periods + 1):
            treated_now = (g > 0) and (t >= g)
            treat = 1 if treated_now else 0
            y = fe + 0.3 * t + dose * treat + rng.normal(0, 0.4)
            rows.append(
                {"unit": u, "time": t, "y": y, "dose": dose, "x": x, "treat": treat}
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", ["att_gt", "twfe", "dose_response", "cgs"])
def test_continuous_did_methods(method):
    df = _dose_panel()
    r = sp.continuous_did(
        df,
        y="y",
        dose="dose",
        time="time",
        id="unit",
        method=method,
        t_pre=4,
        t_post=6,
        controls=["x"],
        n_boot=40,
        seed=0,
    )
    assert np.isfinite(r.estimate)
    lo, hi = r.ci
    assert lo <= hi


# ── de Chaisemartin-D'Haultfœuille ───────────────────────────────── #


def test_did_multiplegt_placebo_dynamic(panel):
    r = sp.did_multiplegt(
        panel,
        y="y",
        group="unit",
        time="time",
        treatment="treat",
        placebo=2,
        dynamic=3,
        n_boot=50,
        seed=0,
    )
    assert abs(r.estimate - TRUE_ATT) < 1.0
    assert "event_study" in (r.model_info or {})


# ── did_analysis pipeline ────────────────────────────────────────── #


def test_did_analysis_cs_pipeline(panel):
    a = sp.did_analysis(
        panel, y="y", treat="first_treat", time="time", id="unit", method="cs"
    )
    assert np.isfinite(a.main_result.estimate)
    assert isinstance(a.summary(), str)
    ax = a.plot()
    assert ax is not None
    plt.close("all")


def test_did_analysis_sun_abraham_pipeline(panel):
    a = sp.did_analysis(
        panel, y="y", treat="first_treat", time="time", id="unit", method="sa"
    )
    assert np.isfinite(a.main_result.estimate)
    assert isinstance(a.summary(), str)


def test_did_analysis_bjs_pipeline(panel):
    a = sp.did_analysis(
        panel, y="y", treat="first_treat", time="time", id="unit", method="bjs"
    )
    assert np.isfinite(a.main_result.estimate)
    assert isinstance(a.summary(), str)
