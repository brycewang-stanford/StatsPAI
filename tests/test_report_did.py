"""``result.report()`` for event-study DiD.

The reported interval is the Rambachan-Roth (2023) smoothness interval with
M = 0 (the pre-period differential trend continues linearly) whenever the
pre-trend test rejects or is too weak to rule out a trend that would move
the truth out of the conventional interval (Roth 2022).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TAU = 1.2


def _panel(seed, slope, n=500, T=8, g=4):
    rng = np.random.default_rng(seed)
    tr = np.arange(n) < n // 2
    a = rng.normal(size=n)
    rows = []
    for t in range(T):
        y = a + 0.2 * t + tr * slope * t + TAU * tr * (t >= g) + rng.normal(size=n)
        rows.append(
            pd.DataFrame(
                {
                    "unit": np.arange(n),
                    "time": t,
                    "y": y,
                    "first_treat": np.where(tr, g, np.nan),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _es(df, window=(-4, 3)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.event_study(
            df, y="y", treat_time="first_treat", time="time", unit="unit", window=window
        )


def test_reported_interval_is_the_smoothness_interval_at_m_zero():
    es = _es(_panel(0, 0.06))
    rep = es.report()
    assert rep["status"] == "adjusted"
    assert rep["ci_method"] == "honest_did_linear_trend"
    ref = sp.honest_did(
        es, l_vec=np.full(4, 0.25), method="smoothness", m_grid=[0.0]
    ).iloc[0]
    assert rep["ci"] == pytest.approx([ref["ci_lower"], ref["ci_upper"]])
    # the fit's own numbers are repeated, not changed
    assert rep["conventional"]["estimate"] == pytest.approx(es.estimate)
    assert rep["conventional"]["se"] == pytest.approx(es.se)
    assert rep["alternatives"]["relative_magnitude"]["Mbar"] == 1.0


def test_power_is_computed_against_the_trend_that_breaks_the_interval():
    es = _es(_panel(1, 0.0))
    rep = es.report()
    # bias of the average of post effects 0..3 under slope s is 2.5 s
    assert rep["pretrend_power_slope"] == pytest.approx(
        1.959964 * es.se / 2.5, rel=1e-6
    )
    ref = sp.pretrends_power(es, slope=rep["pretrend_power_slope"])
    assert rep["pretrend_power"] == pytest.approx(ref["power_joint"])


def test_rejected_pretrend_moves_the_point_estimate_to_the_trend_adjusted_one():
    es = _es(_panel(2, 0.15))
    rep = es.report()
    assert rep["pretrend_pvalue"] < 0.05
    assert rep["estimator"] == "linear_trend_adjusted"
    assert rep["estimate"] == pytest.approx(sum(rep["ci"]) / 2)
    # the linear trend biases the conventional estimate by 2.5 * 0.15
    assert abs(rep["estimate"] - TAU) < abs(es.estimate - TAU)


def test_coverage_under_a_linear_differential_trend():
    conv = cov = 0
    reps = 60
    for s in range(reps):
        rep = _es(_panel(200 + s, 0.06)).report()
        lo, hi = rep["conventional"]["ci"]
        conv += lo <= TAU <= hi
        cov += rep["ci"][0] <= TAU <= rep["ci"][1]
    assert conv / reps < 0.85
    assert cov / reps > 0.95 - 3 * np.sqrt(0.95 * 0.05 / reps)


def test_one_pre_period_cannot_be_trend_adjusted():
    df = _panel(3, 0.15)
    rep = _es(df[df.time >= 2], window=(-2, 3)).report()
    assert rep["status"] in ("caution", "conventional")
    if rep["status"] == "caution":
        assert rep["ci"] == rep["conventional"]["ci"]


def test_agent_payload_carries_the_report():
    d = _es(_panel(0, 0.06)).to_dict(detail="agent")
    assert d["report"]["status"] == "adjusted"


def test_audit_verdict_is_not_a_tick_for_having_run_honest_did():
    # A rejected pre-trend test and an honest-DiD run on record used to read
    # "parallel_trends failed, honest_did passed". The verdict says what to
    # report; the existence check says only that the analysis was run.
    es = _es(_panel(2, 0.15))
    es.model_info["honest_did"] = {"has_run": True, "max_rejecting_M": 2.0}
    rep = sp.audit(es)
    assert rep["verdict"]["action"] == "report_adjusted"
    assert rep["verdict"]["ci"] == es.report()["ci"]
    hd = next(c for c in rep["checks"] if c["name"] == "honest_did")
    assert hd["status"] == "passed" and "not a finding" in hd["note"]
