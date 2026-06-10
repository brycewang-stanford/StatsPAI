"""Tests for epidemiological power / sample-size calculations.

Validation uses Monte-Carlo agreement (the analytic power matches the
empirical rejection rate of the test it approximates) and the closed-form
Schoenfeld events requirement.
"""
import numpy as np
import pytest

import statspai as sp


def _mc_two_proportions(n, p1, p2, ratio=1.0, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    za = 1.959963985
    n1 = int(round(n / (1 + ratio)))
    n2 = int(round(n * ratio / (1 + ratio)))
    rej = 0
    for _ in range(reps):
        ph1 = rng.binomial(n1, p1) / n1
        ph2 = rng.binomial(n2, p2) / n2
        se = np.sqrt(ph1 * (1 - ph1) / n1 + ph2 * (1 - ph2) / n2)
        if se > 0 and abs(ph2 - ph1) / se > za:
            rej += 1
    return rej / reps


def test_two_proportions_matches_monte_carlo():
    for p1, p2, n in [(0.3, 0.5, 200), (0.1, 0.2, 400)]:
        analytic = sp.power_two_proportions(n=n, p1=p1, p2=p2).power
        mc = _mc_two_proportions(n, p1, p2)
        assert analytic == pytest.approx(mc, abs=0.04)


def test_two_proportions_sample_size_achieves_target():
    res = sp.power_two_proportions(p1=0.3, p2=0.5, power_target=0.8)
    assert res.power >= 0.8
    # one fewer subject should drop below target
    below = sp.power_two_proportions(n=res.n - 1, p1=0.3, p2=0.5).power
    assert below < 0.8


def test_two_proportions_monotone_and_array():
    p = sp.power_two_proportions(n=[100, 200, 400], p1=0.3, p2=0.5).power
    assert np.all(np.diff(p) > 0)
    assert sp.power_two_proportions(n=100, p1=0.3, p2=0.3).power == \
        pytest.approx(0.025, abs=0.01)   # no effect -> power ~ alpha/2 side


def test_logrank_schoenfeld_events():
    # HR=0.5, equal allocation, 80% power -> ~65-66 events (Schoenfeld).
    res = sp.power_logrank(hazard_ratio=0.5, prob_event=1.0, power_target=0.8)
    assert 60 <= res.params["n_events"] <= 70
    assert res.power >= 0.8
    # power at exactly 65 events should be ~0.80
    p65 = sp.power_logrank(n=65, hazard_ratio=0.5, prob_event=1.0).power
    assert p65 == pytest.approx(0.80, abs=0.03)


def test_logrank_rejects_bad_hr():
    with pytest.raises(ValueError, match="hazard_ratio"):
        sp.power_logrank(n=100, hazard_ratio=1.0)


def _mc_case_control(nc, OR, p0, ratio=1.0, reps=4000, seed=1):
    rng = np.random.default_rng(seed)
    za = 1.959963985
    p1 = (OR * p0) / (1 + p0 * (OR - 1))
    nco = int(round(nc * ratio))
    rej = 0
    for _ in range(reps):
        pc = rng.binomial(nc, p1) / nc
        po = rng.binomial(nco, p0) / nco
        se = np.sqrt(pc * (1 - pc) / nc + po * (1 - po) / nco)
        if se > 0 and abs(pc - po) / se > za:
            rej += 1
    return rej / reps


def test_case_control_matches_monte_carlo():
    for OR, p0, nc in [(2.0, 0.3, 150), (3.0, 0.2, 100)]:
        analytic = sp.power_case_control(
            n_cases=nc, odds_ratio=OR, exposure_prevalence=p0
        ).power
        mc = _mc_case_control(nc, OR, p0)
        assert analytic == pytest.approx(mc, abs=0.04)


def test_case_control_validates_inputs():
    with pytest.raises(ValueError, match="odds_ratio"):
        sp.power_case_control(n_cases=100, odds_ratio=1.0, exposure_prevalence=0.3)
    with pytest.raises(ValueError, match="exposure_prevalence"):
        sp.power_case_control(n_cases=100, odds_ratio=2.0, exposure_prevalence=1.5)


def test_power_results_have_summary():
    r = sp.power_two_proportions(n=200, p1=0.3, p2=0.5)
    assert "two_proportions" in r.summary().lower() or "TWO_PROPORTIONS" in r.summary()
    assert 0 <= r.power <= 1
