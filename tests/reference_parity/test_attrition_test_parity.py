"""Analytical parity: sp.attrition_test treatment-vs-control attrition comparison.

Under random attrition, the treatment and control attrition rates are equal
up to sampling noise. The estimator reports a chi-square test statistic and
p-value for their equality. Analytical evidence tier.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def test_balanced_attrition_no_significant_difference():
    """Random attrition on both arms yields a non-significant test."""
    rng = np.random.default_rng(0)
    n = 5000
    d = rng.integers(0, 2, n)
    # Equal ~20% missing on both arms.
    o = np.where(rng.random(n) < 0.20, 1, 0)
    df = pd.DataFrame({"d": d, "o": o})
    r = sp.attrition_test(df, treatment="d", observed="o")
    # Both rates near 0.20 -> diff ~ 0; test statistic small.
    assert abs(float(r.treat_rate) - float(r.control_rate)) < 0.05
    assert float(r.diff_p_value) > 0.05


def test_exact_attrition_rates_match_hand():
    """`o = 1` means observed; the reported rate is the ATTRITION (= 1 - observed) rate.

    With observation probabilities p_treat_obs and p_ctrl_obs, attrition is
    1 - p_treat_obs and 1 - p_ctrl_obs, so attrition diff = p_ctrl_obs - p_treat_obs.
    """
    rng = np.random.default_rng(1)
    n = 1000
    d = rng.integers(0, 2, n)
    p_treat_obs, p_ctrl_obs = 0.90, 0.80
    o = np.where(d == 0, rng.random(n) < p_ctrl_obs, rng.random(n) < p_treat_obs)
    df = pd.DataFrame({"d": d, "o": o})
    r = sp.attrition_test(df, treatment="d", observed="o")
    assert float(r.treat_rate) == pytest.approx(1 - p_treat_obs, abs=0.04)
    assert float(r.control_rate) == pytest.approx(1 - p_ctrl_obs, abs=0.03)
    # More attrition in control (lower observation rate).
    assert float(r.control_rate) - float(r.treat_rate) == pytest.approx(
        p_treat_obs - p_ctrl_obs, abs=0.05
    )
