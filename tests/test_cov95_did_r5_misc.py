"""Coverage round-5 for the smaller statspai.did modules.

Covers the still-missing branches in:
- pretrends      (F-test, single-pre slope, singular VCV, no-pre errors,
                  power on no-pre, summary moderate/low-power lines, plot)
- aggte          (no-detail / no-inf-bstrap / no-cells / cohort-weights None)
- analysis       (did_analysis steps: bacon warning, event study, sensitivity)
- cohort_anchored, timevarying_covariates, did_multiplegt,
  did_multiplegt_dyn, sun_abraham, wooldridge_did / etwfe / etwfe_emfx
  edge paths and validation errors.

All panels are real synthetic staggered/switching DiD data.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult
from statspai.did.analysis import did_analysis


def make_panel(seed=0, cohorts=(4, 6, 0), n_per=22, T=8):
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for g in cohorts:
        for _ in range(n_per):
            ufe = rng.normal()
            for t in range(1, T + 1):
                te = max(0, t - g + 1) if g > 0 else 0
                rows.append(
                    {
                        "i": uid,
                        "t": t,
                        "y": ufe + 0.3 * t + te + rng.normal() * 0.5,
                        "g": g,
                    }
                )
            uid += 1
    return pd.DataFrame(rows)


def _mk_es_result(rt, att, se, model_info=None, est=1.0):
    det = pd.DataFrame({"relative_time": rt, "att": att, "estimate": att, "se": se})
    return CausalResult(
        method="event_study",
        estimand="ATT",
        estimate=est,
        se=0.2,
        pvalue=0.01,
        ci=(0.6, 1.4),
        alpha=0.05,
        n_obs=200,
        detail=det,
        model_info=model_info or {},
    )


# ======================================================================
# pretrends
# ======================================================================


def test_pretrends_f_test():
    from statspai.did.pretrends import pretrends_test

    df = make_panel()
    es = sp.event_study(df, y="y", treat_time="g", time="t", unit="i")
    out = pretrends_test(es, type="f")
    assert out["type"] == "f"
    assert "F(" in out["stat_label"]


def test_pretrends_bad_type():
    from statspai.did.pretrends import pretrends_test

    es = _mk_es_result([-2, -1, 1], [0.05, 0.03, 1.0], [0.1, 0.1, 0.2])
    with pytest.raises(ValueError, match="must be 'wald' or 'f'"):
        pretrends_test(es, type="zzz")


def test_pretrends_test_singular_vcv():
    from statspai.did.pretrends import pretrends_test

    es = _mk_es_result(
        [-2, -1, 1],
        [0.05, 0.03, 1.0],
        [0.1, 0.1, 0.2],
        model_info={"vcv_pre": np.array([[1.0, 1.0], [1.0, 1.0]])},
    )
    with pytest.raises(ValueError, match="singular"):
        pretrends_test(es)


def test_pretrends_power_singular_vcv():
    from statspai.did.pretrends import pretrends_power

    es = _mk_es_result(
        [-2, -1, 1],
        [0.05, 0.03, 1.0],
        [0.1, 0.1, 0.2],
        model_info={"vcv_pre": np.array([[1.0, 1.0], [1.0, 1.0]])},
    )
    with pytest.raises(ValueError, match="singular"):
        pretrends_power(es)


def test_pretrends_power_no_pre_raises():
    from statspai.did.pretrends import pretrends_power

    es = _mk_es_result([0, 1, 2], [1.0, 1.1, 1.2], [0.2, 0.2, 0.2])
    with pytest.raises(ValueError, match="No pre-treatment periods"):
        pretrends_power(es)


def test_pretrends_power_all_zero_se():
    from statspai.did.pretrends import pretrends_power

    es = _mk_es_result([-1, 0, 1], [0.0, 1.0, 1.2], [0.0, 0.2, 0.2])
    with pytest.raises(ValueError, match="zero standard error"):
        pretrends_power(es)


def test_sensitivity_rr_single_pre_slope():
    from statspai.did.pretrends import sensitivity_rr

    es = _mk_es_result([-1, 1, 2], [0.05, 1.0, 1.2], [0.1, 0.2, 0.2])
    s = sensitivity_rr(es, Mbar=[0, 0.05, 0.1])
    assert len(s.mbar_grid) == 3


def test_sensitivity_rr_no_pre_raises():
    from statspai.did.pretrends import sensitivity_rr

    es = _mk_es_result([0, 1, 2], [1.0, 1.1, 1.2], [0.2, 0.2, 0.2])
    with pytest.raises(ValueError, match="No pre-treatment periods"):
        sensitivity_rr(es)


def test_sensitivity_rr_bad_method():
    from statspai.did.pretrends import sensitivity_rr

    es = _mk_es_result([-1, 1], [0.05, 1.0], [0.1, 0.2])
    with pytest.raises(NotImplementedError, match="C-LF"):
        sensitivity_rr(es, method="other")


def test_sensitivity_rr_plot():
    from statspai.did.pretrends import sensitivity_rr

    es = _mk_es_result([-2, -1, 1, 2], [0.05, 0.03, 1.0, 1.2], [0.1, 0.1, 0.2, 0.2])
    s = sensitivity_rr(es, Mbar=[0, 0.01, 0.05])
    ax = s.plot()
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def test_pretrends_summary_prints():
    from statspai.did.pretrends import pretrends_summary

    df = make_panel()
    es = sp.event_study(df, y="y", treat_time="g", time="t", unit="i")
    out = pretrends_summary(es)
    assert isinstance(out, str) and "Pre-Trends" in out


# ======================================================================
# aggte
# ======================================================================


def test_aggte_no_detail_raises():
    import copy
    from statspai.did.aggte import aggte
    from statspai.did.callaway_santanna import callaway_santanna

    cs = callaway_santanna(make_panel(), y="y", g="g", t="t", i="i")
    bad = copy.copy(cs)
    bad.detail = None
    with pytest.raises(ValueError, match="no ATT.*detail"):
        aggte(bad, type="simple")


def test_aggte_bstrap_falls_back_when_no_inf():
    import copy
    from statspai.did.aggte import aggte
    from statspai.did.callaway_santanna import callaway_santanna

    cs = callaway_santanna(make_panel(), y="y", g="g", t="t", i="i")
    noinf = copy.copy(cs)
    noinf._influence_funcs = None
    r = aggte(noinf, type="simple", bstrap=True)
    assert np.isfinite(r.estimate)


def test_aggte_no_cells_after_filter():
    from statspai.did.aggte import aggte
    from statspai.did.callaway_santanna import callaway_santanna

    cs = callaway_santanna(make_panel(), y="y", g="g", t="t", i="i")
    with pytest.raises(ValueError, match="no aggregation cells"):
        aggte(cs, type="dynamic", min_e=100, max_e=200)


def test_cohort_weight_series_none_and_zero():
    from statspai.did.aggte import _cohort_weight_series

    detail = pd.DataFrame({"group": [4, 6], "relative_time": [0, 0]})
    s_none = _cohort_weight_series(detail, None)
    assert np.allclose(s_none.values, 0.5)
    # All-zero sizes -> equal-weight fallback (lines 335-337)
    s_zero = _cohort_weight_series(detail, pd.Series({4: 0.0, 6: 0.0}))
    assert np.allclose(s_zero.values, 0.5)


# ======================================================================
# analysis: did_analysis
# ======================================================================


def test_did_analysis_cs_full_pipeline():
    df = make_panel()
    r = did_analysis(df, y="y", treat="g", time="t", id="i", method="cs")
    txt = r.summary()
    assert "ATT" in txt
    assert np.isfinite(r.main_result.estimate)


def test_did_analysis_bjs_method():
    df = make_panel()
    r = did_analysis(df, y="y", treat="g", time="t", id="i", method="bjs")
    assert np.isfinite(r.main_result.estimate)
    assert "Borusyak" in r.method_used


def test_did_analysis_sa_method():
    df = make_panel()
    r = did_analysis(df, y="y", treat="g", time="t", id="i", method="sa")
    assert "Sun-Abraham" in r.method_used


def test_did_analysis_cs_requires_id():
    from statspai.exceptions import MethodIncompatibility

    df = make_panel()
    with pytest.raises(MethodIncompatibility):
        did_analysis(df, y="y", treat="g", time="t", method="cs")


def test_did_analysis_unknown_method():
    df = make_panel()
    with pytest.raises(ValueError, match="Unknown method"):
        did_analysis(
            df,
            y="y",
            treat="g",
            time="t",
            id="i",
            method="bogus",
            run_bacon=False,
            run_event_study=False,
            run_sensitivity=False,
        )


# ======================================================================
# cohort_anchored
# ======================================================================


def test_cohort_anchored_basic():
    from statspai.did.cohort_anchored import cohort_anchored_event_study

    df = make_panel()
    r = cohort_anchored_event_study(
        df, y="y", treat="g", time="t", id="i", leads=2, lags=2
    )
    assert np.isfinite(r.estimate)
    assert "event_study" in r.model_info


def test_cohort_anchored_no_cohorts_raises():
    from statspai.did.cohort_anchored import cohort_anchored_event_study

    df = make_panel(cohorts=(0, 0))
    with pytest.raises(ValueError, match="No treated cohorts"):
        cohort_anchored_event_study(df, y="y", treat="g", time="t", id="i")


# ======================================================================
# timevarying_covariates
# ======================================================================


def make_tvc(seed=0, cohorts=(4, 6, 0), n_per=20, T=8):
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for g in cohorts:
        for _ in range(n_per):
            ufe = rng.normal()
            for t in range(1, T + 1):
                te = max(0, t - g + 1) if g > 0 else 0
                rows.append(
                    {
                        "i": uid,
                        "year": t,
                        "earn": ufe + 0.2 * t + te + rng.normal() * 0.4,
                        "g": g,
                        "age": 20 + t + rng.normal(),
                    }
                )
            uid += 1
    return pd.DataFrame(rows)


def test_timevarying_basic():
    from statspai.did.timevarying_covariates import did_timevarying_covariates

    df = make_tvc()
    r = did_timevarying_covariates(
        df,
        y="earn",
        unit="i",
        time="year",
        cohort="g",
        covariates=["age"],
        n_boot=30,
        seed=1,
    )
    assert np.isfinite(r.estimate)
    assert r.model_info["n_cells"] > 0


def test_timevarying_missing_column():
    from statspai.did.timevarying_covariates import did_timevarying_covariates

    df = make_tvc()
    with pytest.raises(ValueError, match="not in data"):
        did_timevarying_covariates(
            df, y="nope", unit="i", time="year", cohort="g", covariates=["age"]
        )


def test_timevarying_no_treated_cohorts():
    from statspai.did.timevarying_covariates import did_timevarying_covariates

    df = make_tvc(cohorts=(0, 0))
    with pytest.raises(ValueError, match="No treated cohorts"):
        did_timevarying_covariates(
            df, y="earn", unit="i", time="year", cohort="g", covariates=["age"]
        )


def test_timevarying_no_never_treated():
    from statspai.did.timevarying_covariates import did_timevarying_covariates

    df = make_tvc(cohorts=(4, 6))
    with pytest.raises(ValueError, match="No never-treated"):
        did_timevarying_covariates(
            df, y="earn", unit="i", time="year", cohort="g", covariates=["age"]
        )


def test_timevarying_degenerate_se_one_boot():
    from statspai.did.timevarying_covariates import did_timevarying_covariates

    df = make_tvc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = did_timevarying_covariates(
            df,
            y="earn",
            unit="i",
            time="year",
            cohort="g",
            covariates=["age"],
            n_boot=1,
            seed=1,
        )
    assert np.isnan(r.pvalue)


# ======================================================================
# did_multiplegt (de Chaisemartin-D'Haultfoeuille)
# ======================================================================


def make_switch(seed=0, n=35, T=6):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n):
        sw = int(rng.integers(2, T + 1)) if rng.random() < 0.6 else 999
        ufe = rng.normal()
        for t in range(1, T + 1):
            d = 1 if t >= sw else 0
            rows.append(
                {
                    "g": u,
                    "t": t,
                    "y": ufe + 0.3 * t + 2 * d + rng.normal() * 0.3,
                    "d": d,
                }
            )
    return pd.DataFrame(rows)


def test_did_multiplegt_placebo_dynamic():
    from statspai.did.did_multiplegt import did_multiplegt

    df = make_switch()
    r = did_multiplegt(
        df,
        y="y",
        group="g",
        time="t",
        treatment="d",
        placebo=2,
        dynamic=2,
        n_boot=30,
        seed=1,
    )
    assert np.isfinite(r.estimate)
    mi = r.model_info or {}
    assert "placebo" in mi and "dynamic" in mi


def test_did_multiplegt_dyn_basic():
    from statspai.did.did_multiplegt_dyn import did_multiplegt_dyn

    df = make_switch()
    r = did_multiplegt_dyn(
        df,
        y="y",
        group="g",
        time="t",
        treatment="d",
        dynamic=2,
        placebo=1,
        n_boot=30,
        seed=1,
    )
    assert r is not None
    assert "event_study" in (r.model_info or {}) or np.isfinite(r.estimate)


# ======================================================================
# sun_abraham
# ======================================================================


def test_sun_abraham_basic():
    from statspai.did.sun_abraham import sun_abraham

    df = make_panel()
    r = sun_abraham(df, y="y", g="g", t="t", i="i")
    assert np.isfinite(r.estimate)


def test_sun_abraham_no_cohorts_raises():
    from statspai.did.sun_abraham import sun_abraham

    df = make_panel(cohorts=(0, 0))
    with pytest.raises(ValueError, match="No treated cohorts"):
        sun_abraham(df, y="y", g="g", t="t", i="i")


# ======================================================================
# wooldridge_did / etwfe / etwfe_emfx
# ======================================================================


def test_wooldridge_did_basic():
    from statspai.did.wooldridge_did import wooldridge_did

    df = make_panel()
    r = wooldridge_did(df, y="y", group="i", time="t", first_treat="g")
    assert np.isfinite(r.estimate)


def test_wooldridge_did_no_cohorts_raises():
    from statspai.did.wooldridge_did import wooldridge_did

    df = make_panel(cohorts=(0, 0))
    with pytest.raises(ValueError, match="No treated cohorts"):
        wooldridge_did(df, y="y", group="i", time="t", first_treat="g")


def test_etwfe_no_cohorts_raises():
    from statspai.did.wooldridge_did import etwfe

    df = make_panel(cohorts=(0, 0))
    with pytest.raises(ValueError, match="No treated cohorts"):
        etwfe(df, y="y", group="i", time="t", first_treat="g")


@pytest.mark.parametrize("typ", ["simple", "event", "group", "calendar"])
def test_etwfe_emfx_aggregations(typ):
    from statspai.did.wooldridge_did import etwfe, etwfe_emfx

    df = make_panel()
    r = etwfe(df, y="y", group="i", time="t", first_treat="g")
    m = etwfe_emfx(r, type=typ)
    assert np.isfinite(m.estimate)


def test_etwfe_emfx_treated_weighting():
    from statspai.did.wooldridge_did import etwfe, etwfe_emfx

    df = make_panel()
    r = etwfe(df, y="y", group="i", time="t", first_treat="g")
    m = etwfe_emfx(r, type="group", weighting="treated")
    assert np.isfinite(m.estimate)
