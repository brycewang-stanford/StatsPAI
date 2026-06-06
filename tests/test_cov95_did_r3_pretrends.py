"""Coverage round-3 for statspai.did.pretrends (non-plot paths).

Covers pretrends_test (Wald + F + bad-type), pretrends_power (default and
explicit delta), pretrends_summary, sensitivity_rr, and the full-VCV
alignment branch via a hand-built event-study result carrying ``vcv_pre``.

Plot methods (SensitivityResult.plot) are intentionally NOT exercised here —
they live in the excluded plotting surface and need a matplotlib backend.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


@pytest.fixture(scope="module")
def es_result():
    df = sp.dgp_did(n_units=140, n_periods=10, staggered=False, seed=5).copy()
    df["first_treat"] = df["first_treat"].fillna(0)
    return sp.event_study(df, y="y", treat_time="first_treat", time="time",
                          unit="unit", window=(-4, 4))


def test_pretrends_test_wald(es_result):
    out = sp.pretrends_test(es_result, type="wald")
    assert 0.0 <= out["pvalue"] <= 1.0
    assert out["type"] == "wald"
    assert out["df"] >= 1
    assert "interpretation" in out


def test_pretrends_test_f(es_result):
    out = sp.pretrends_test(es_result, type="f")
    assert 0.0 <= out["pvalue"] <= 1.0
    assert out["type"] == "f"
    assert out["stat_label"].startswith("F(")


def test_pretrends_test_bad_type(es_result):
    with pytest.raises(ValueError, match="type must be"):
        sp.pretrends_test(es_result, type="bogus")


def test_pretrends_power_default(es_result):
    out = sp.pretrends_power(es_result)
    assert 0.0 <= out["power"] <= 1.0
    assert "noncentrality" in out
    assert out["df"] >= 1


def test_pretrends_power_explicit_delta(es_result):
    # delta length must match number of estimated pre-periods (SE>0).
    es = sp.pretrends_test(es_result)  # ensures pre-periods exist
    k = es["df"]
    delta = np.linspace(0.1, 0.3, k)
    out = sp.pretrends_power(es_result, delta=delta)
    assert 0.0 <= out["power"] <= 1.0


def test_pretrends_power_moderate_warning(es_result):
    # A mid-size delta pushes power into the 0.5-0.8 "moderate" band, which
    # triggers the moderate-power warning branch.
    k = sp.pretrends_test(es_result)["df"]
    out = sp.pretrends_power(es_result, delta=np.full(k, 5 * 0.05))
    assert 0.5 <= out["power"] < 0.8
    assert out["warning"] is not None
    assert "MODERATE" in out["warning"]


def test_pretrends_summary(es_result):
    s = sp.pretrends_summary(es_result)
    assert "Pre-Trends Analysis" in s
    assert "Power against linear violation" in s


def test_sensitivity_rr(es_result):
    sr = sp.sensitivity_rr(es_result)
    assert hasattr(sr, "summary")
    txt = sr.summary()
    assert isinstance(txt, str) and len(txt) > 0
    assert hasattr(sr, "mbar_grid")
    assert len(sr.mbar_grid) == len(sr.ci_lower) == len(sr.ci_upper)


# ----------------------------------------------------------------------
# Full-VCV alignment branch: build an event-study-shaped CausalResult with
# a vcv_pre matrix sized to ALL pre-periods (reference included). This
# triggers the np.ix_ alignment to the estimated (SE>0) subset.
# ----------------------------------------------------------------------
def _hand_event_study_result():
    # Pre periods -3..-1 (with -1 the reference, SE=0) plus post 0..2.
    rel = [-3, -2, -1, 0, 1, 2]
    est = [0.02, -0.01, 0.0, 0.4, 0.5, 0.55]
    se = [0.05, 0.05, 0.0, 0.06, 0.06, 0.07]
    detail = pd.DataFrame({
        "relative_time": rel,
        "estimate": est,
        "se": se,
    })
    # vcv_pre over ALL pre-periods (3 rows incl. the reference), diagonal.
    pre_se = np.array([0.05, 0.05, 0.0])
    vcv_pre = np.diag(pre_se ** 2)
    return CausalResult(
        method="event_study",
        estimand="ATT(k)",
        estimate=0.48,
        se=0.05,
        pvalue=0.0,
        ci=(0.38, 0.58),
        alpha=0.05,
        n_obs=500,
        detail=detail,
        model_info={"event_study": detail, "vcv_pre": vcv_pre},
    )


def test_pretrends_test_with_full_vcv_pre():
    res = _hand_event_study_result()
    out = sp.pretrends_test(res, type="wald")
    assert 0.0 <= out["pvalue"] <= 1.0
    # only the two SE>0 pre-periods enter the test
    assert out["df"] == 2


def test_pretrends_power_with_full_vcv_pre():
    res = _hand_event_study_result()
    out = sp.pretrends_power(res)
    assert 0.0 <= out["power"] <= 1.0
    assert out["df"] == 2
