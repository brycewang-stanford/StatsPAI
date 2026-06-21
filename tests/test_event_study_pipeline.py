"""End-to-end numerical-coherence test for the event-study workflow.

Beyond "the consumers don't crash" (see test_event_study_consumers.py), this
checks that the *science* flows through correctly on a DGP with a known
post-treatment effect of 2.0 and NO pre-trend: the overall ATT and the
post-period coefficients recover ~2.0, the pre-period coefficients sit near
zero, the joint pre-trend test cannot reject parallel trends, and the
Rambachan-Roth honest CIs widen with the smoothness bound M while bracketing
the estimate at M = 0.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TRUE_EFFECT = 2.0


@pytest.fixture
def es_result():
    rng = np.random.RandomState(42)
    rows = []
    for u in range(60):
        tt = np.nan if u % 3 == 0 else rng.choice([5, 7])
        fe = rng.normal() * 2
        for t in range(1, 13):
            post = (not np.isnan(tt)) and t >= tt
            y = fe + 0.3 * t + (TRUE_EFFECT if post else 0.0) + rng.normal()
            rows.append((u, t, y, tt))
    df = pd.DataFrame(rows, columns=["unit", "time", "y", "treat_time"])
    return sp.event_study(
        df,
        y="y",
        treat_time="treat_time",
        time="time",
        unit="unit",
        window=(-4, 4),
    )


def test_overall_att_recovers_effect(es_result):
    assert float(es_result.estimate) == pytest.approx(TRUE_EFFECT, abs=0.4)


def test_pre_periods_flat_post_periods_lifted(es_result):
    table = es_result.model_info["event_study"]
    estimated = table[table["se"] > 0]
    pre = estimated[estimated["relative_time"] < 0]
    post = estimated[estimated["relative_time"] >= 0]
    # No pre-trend: pre-period coefficients hug zero.
    assert pre["att"].abs().max() < 0.5
    # Post-period coefficients recover the planted effect on average.
    assert post["att"].mean() == pytest.approx(TRUE_EFFECT, abs=0.5)


def test_pretrend_test_cannot_reject_parallel_trends(es_result):
    pt = sp.pretrends_test(es_result)
    # DGP has parallel pre-trends, so the joint test should not reject.
    assert pt["pvalue"] > 0.10


def test_honest_did_widens_with_smoothness_bound(es_result):
    hd = sp.honest_did(es_result)
    assert (hd["ci_lower"] <= hd["ci_upper"]).all()
    width = hd["ci_upper"] - hd["ci_lower"]
    # Allowing more pre-trend curvature (larger M) cannot tighten the CI.
    assert width.iloc[-1] >= width.iloc[0] - 1e-9
    # The M = 0 (parallel-trends) interval brackets the point estimate.
    row0 = hd.iloc[0]
    assert row0["ci_lower"] <= float(es_result.estimate) <= row0["ci_upper"]


def test_tidy_carries_main_and_event_rows(es_result):
    tidy = es_result.tidy()
    assert (tidy["type"] == "event_study").sum() > 0
    assert "main" in set(tidy["type"])
    assert np.isfinite(tidy["estimate"].dropna()).all()
