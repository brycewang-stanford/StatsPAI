"""Tests for ``sp.longitudinal`` — unified longitudinal analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.longitudinal import regime, always_treat, never_treat


# ---------------------------------------------------------------------------
# Regime DSL
# ---------------------------------------------------------------------------


def test_always_treat_regime():
    r = sp.always_treat(K=3)
    assert r.kind == "static"
    assert r.rule == [1.0, 1.0, 1.0]
    assert r.treatment({}, 0) == 1.0
    assert r.treatment({}, 5) == 1.0  # clamps to last


def test_never_treat_regime():
    r = sp.never_treat(K=3)
    assert r.rule == [0.0, 0.0, 0.0]


def test_regime_from_list():
    r = sp.regime([1, 0, 1])
    assert r.kind == "static"
    assert r.rule == [1.0, 0.0, 1.0]


def test_regime_from_string_if_then_else():
    r = sp.regime("if cd4 < 200 then 1 else 0")
    assert r.kind == "dynamic"
    assert r.treatment({"cd4": 150}, 0) == 1
    assert r.treatment({"cd4": 250}, 0) == 0


def test_regime_from_string_always():
    r = sp.regime("always_treat")
    assert r.kind == "static"
    assert r.rule == [1.0]


def test_regime_from_callable():
    def my_regime(h, t):
        return 1 if h.get("x", 0) > 0 else 0
    r = sp.regime(my_regime, name="x_positive")
    assert r.kind == "dynamic"
    assert r.treatment({"x": 5}, 0) == 1
    assert r.treatment({"x": -1}, 0) == 0


def test_regime_string_rejects_unsafe_syntax():
    import pytest
    # Function calls, attribute access, etc. should be rejected
    with pytest.raises(ValueError):
        sp.regime("__import__('os').system('ls')")


def test_regime_string_rejects_unknown_var():
    r = sp.regime("if nonexistent > 0 then 1 else 0")
    with pytest.raises(NameError):
        r.treatment({}, 0)


# ---------------------------------------------------------------------------
# analyze() — IPW path (no time-varying confounders)
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_cross_sectional_panel():
    """One row per unit, simulating simple cross-sectional treatment."""
    rng = np.random.default_rng(42)
    n = 500
    age = rng.normal(50, 10, n)
    treated = rng.binomial(1, 1 / (1 + np.exp(-(age - 50) / 10)))
    # True ATE = 2.0
    y = 10 + 2.0 * treated + 0.1 * age + rng.normal(0, 1, n)
    return pd.DataFrame({
        "pid": range(n),
        "visit": 0,
        "treat": treated,
        "y": y,
        "age": age,
    })


def test_analyze_ipw_without_time_varying(simple_cross_sectional_panel):
    df = simple_cross_sectional_panel
    r = sp.longitudinal_analyze(
        data=df, id="pid", time="visit",
        treatment="treat", outcome="y",
        baseline=["age"], regime=always_treat(),
    )
    assert r.method == "ipw"
    assert np.isfinite(r.estimate)
    assert r.n == df["pid"].nunique()


def test_analyze_contrast_a_minus_b(simple_cross_sectional_panel):
    df = simple_cross_sectional_panel
    diff = sp.longitudinal_contrast(
        data=df, id="pid", time="visit",
        treatment="treat", outcome="y",
        regime_a=always_treat(), regime_b=never_treat(),
        baseline=["age"],
    )
    assert "contrast" in diff
    assert "ci" in diff
    # Contrast should be positive (true ATE = 2)
    assert diff["contrast"] > 0


# ---------------------------------------------------------------------------
# analyze() — MSM path (time-varying)
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_with_tv_confounders():
    """Simulate a longitudinal panel with time-varying confounder."""
    rng = np.random.default_rng(99)
    rows = []
    for pid in range(100):
        bp_prev = rng.normal(120, 10)
        for t in range(3):
            bp_lag = bp_prev
            p_treat = 1 / (1 + np.exp(-(bp_lag - 120) / 20))
            treat = rng.binomial(1, p_treat)
            bp = bp_prev - 2 * treat + rng.normal(0, 3)
            rows.append({
                "pid": pid, "visit": t,
                "treat": treat, "bp": bp, "bp_lag": bp_lag,
            })
            bp_prev = bp
    df = pd.DataFrame(rows)
    # End-of-follow-up outcome
    df["y"] = df.groupby("pid")["bp"].transform("last")
    return df


def test_analyze_msm_path(panel_with_tv_confounders):
    df = panel_with_tv_confounders
    r = sp.longitudinal_analyze(
        data=df, id="pid", time="visit",
        treatment="treat", outcome="y",
        time_varying=["bp_lag"],
        regime="if bp_lag > 120 then 1 else 0",
    )
    assert r.method == "msm"
    assert np.isfinite(r.estimate)
    assert "weight_mean" in r.diagnostics or len(r.diagnostics) == 0


def test_analyze_gformula_path_with_static_regime(panel_with_tv_confounders):
    df = panel_with_tv_confounders
    r = sp.longitudinal_analyze(
        data=df, id="pid", time="visit",
        treatment="treat", outcome="y",
        time_varying=["bp_lag"],
        regime=always_treat(K=3),
        method="g-formula",
    )
    assert r.method == "g-formula"
    assert np.isfinite(r.estimate)
