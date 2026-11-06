"""Tests for ``sp.longitudinal`` — unified longitudinal analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.longitudinal import always_treat, never_treat, regime

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
    return pd.DataFrame(
        {
            "pid": range(n),
            "visit": 0,
            "treat": treated,
            "y": y,
            "age": age,
        }
    )


def test_analyze_ipw_without_time_varying(simple_cross_sectional_panel):
    df = simple_cross_sectional_panel
    r = sp.longitudinal_analyze(
        data=df,
        id="pid",
        time="visit",
        treatment="treat",
        outcome="y",
        baseline=["age"],
        regime=always_treat(),
    )
    assert r.method == "ipw"
    assert np.isfinite(r.estimate)
    assert r.n == df["pid"].nunique()


def test_longitudinal_analyze_ipw_recovers_balanced_static_regime_means():
    rows = []
    pid = 0
    for age in (30.0, 40.0, 50.0, 60.0):
        for _ in range(15):
            for treat in (0.0, 1.0):
                rows.append(
                    {
                        "pid": pid,
                        "visit": 0,
                        "age": age,
                        "treat": treat,
                        "y": 5.0 + 0.1 * age + 2.0 * treat,
                    }
                )
                pid += 1
    df = pd.DataFrame(rows)

    always = sp.longitudinal_analyze(
        data=df,
        id="pid",
        time="visit",
        treatment="treat",
        outcome="y",
        baseline=["age"],
        regime=always_treat(),
    )
    never = sp.longitudinal_analyze(
        data=df,
        id="pid",
        time="visit",
        treatment="treat",
        outcome="y",
        baseline=["age"],
        regime=never_treat(),
    )

    np.testing.assert_allclose(always.estimate, 11.5, atol=1e-12)
    np.testing.assert_allclose(never.estimate, 9.5, atol=1e-12)
    np.testing.assert_allclose(always.estimate - never.estimate, 2.0, atol=1e-12)


def test_analyze_contrast_a_minus_b(simple_cross_sectional_panel):
    df = simple_cross_sectional_panel
    diff = sp.longitudinal_contrast(
        data=df,
        id="pid",
        time="visit",
        treatment="treat",
        outcome="y",
        regime_a=always_treat(),
        regime_b=never_treat(),
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
            rows.append(
                {
                    "pid": pid,
                    "visit": t,
                    "treat": treat,
                    "bp": bp,
                    "bp_lag": bp_lag,
                }
            )
            bp_prev = bp
    df = pd.DataFrame(rows)
    # End-of-follow-up outcome
    df["y"] = df.groupby("pid")["bp"].transform("last")
    return df


def test_analyze_msm_path(panel_with_tv_confounders):
    df = panel_with_tv_confounders
    with pytest.warns(sp.AssumptionWarning, match="current exposure"):
        r = sp.longitudinal_analyze(
            data=df,
            id="pid",
            time="visit",
            treatment="treat",
            outcome="y",
            time_varying=["bp_lag"],
            regime="if bp_lag > 120 then 1 else 0",
        )
    assert r.method == "msm"
    assert r.estimate == r.underlying_result.estimate
    assert r.diagnostics["estimand"].startswith("MSM coefficient on current")


def test_analyze_gformula_path_with_static_regime(panel_with_tv_confounders):
    df = panel_with_tv_confounders
    r = sp.longitudinal_analyze(
        data=df,
        id="pid",
        time="visit",
        treatment="treat",
        outcome="y",
        time_varying=["bp_lag"],
        regime=always_treat(K=3),
        method="g-formula",
    )
    assert r.method == "g-formula"
    assert np.isfinite(r.estimate)


# ---------------------------------------------------------------------------
# Baseline-IPW path: inputs it cannot handle are refused (1.43)
# ---------------------------------------------------------------------------


def _two_period(treat_rows):
    rows = []
    for pid, (a0, a1) in enumerate(treat_rows):
        rows += [
            {"pid": pid, "visit": 0, "treat": a0, "y": 1.0 + pid, "age": 40.0 + pid},
            {"pid": pid, "visit": 1, "treat": a1, "y": 2.0 + pid, "age": 40.0 + pid},
        ]
    return pd.DataFrame(rows)


def test_ipw_path_refuses_treatment_that_varies_within_id():
    """Before 1.43 the path read only the last-period treatment, so
    'always treat' meant 'treated at the end'."""
    from statspai.exceptions import MethodIncompatibility

    df = _two_period([(0, 1), (1, 1), (0, 0), (1, 0), (1, 1), (0, 0)])
    with pytest.raises(MethodIncompatibility, match="2 ids change treatment"):
        sp.longitudinal_analyze(
            df,
            id="pid",
            time="visit",
            treatment="treat",
            outcome="y",
            baseline=["age"],
            regime=always_treat(),
        )


def test_ipw_path_refuses_dynamic_and_time_varying_regimes():
    """Before 1.43 a dynamic regime returned the unweighted mean of y and a
    static [1, 0] regime was read as 'always treat'."""
    from statspai.exceptions import MethodIncompatibility

    df = _two_period([(1, 1), (0, 0), (1, 1), (0, 0)])
    with pytest.raises(MethodIncompatibility, match="dynamic regime"):
        sp.longitudinal_analyze(
            df,
            id="pid",
            time="visit",
            treatment="treat",
            outcome="y",
            regime=regime("if age > 41 then 1 else 0"),
        )
    with pytest.raises(MethodIncompatibility, match="changes over time"):
        sp.longitudinal_analyze(
            df,
            id="pid",
            time="visit",
            treatment="treat",
            outcome="y",
            regime=[1, 0],
        )


def test_ipw_path_refuses_a_regime_nobody_follows():
    """Before 1.43 this returned estimate = NaN."""
    from statspai.exceptions import DataInsufficient

    df = _two_period([(0, 0), (0, 0), (0, 0)])
    with pytest.raises(DataInsufficient, match="no id has treatment 1.0"):
        sp.longitudinal_analyze(
            df,
            id="pid",
            time="visit",
            treatment="treat",
            outcome="y",
            regime=always_treat(),
        )


# --- regime contrasts: paired g-formula bootstrap, MSM refusal (2026-11-06) ---


def _tv_panel(n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        L, a_prev = rng.normal(), 0
        for t in range(3):
            L = 0.5 * L + 0.4 * a_prev + rng.normal()
            a = int(rng.random() < 1 / (1 + np.exp(-0.6 * L)))
            rows.append(
                dict(id=i, time=t, A=a, L=L, Y=1.0 * a + 0.5 * L + rng.normal())
            )
            a_prev = a
    return pd.DataFrame(rows)


_KW = dict(id="id", time="time", treatment="A", outcome="Y", time_varying=["L"])


def test_gformula_contrast_se_is_the_paired_bootstrap_of_the_difference():
    df = _tv_panel()
    res = sp.longitudinal_contrast(
        df, regime_a="always_treat", regime_b="never_treat", seed=7, **_KW
    )
    a, b = res["a_result"], res["b_result"]
    assert a.method == b.method == "g-formula"
    assert a.diagnostics["seed"] == b.diagnostics["seed"] == res["seed"] == 7
    # Same seed and data: both regimes were refitted on the same resamples.
    da = a.underlying_result.bootstrap_draws
    db = b.underlying_result.bootstrap_draws
    assert len(da) == len(db) == 200
    np.testing.assert_allclose(res["se"], np.std(da - db, ddof=1), rtol=1e-12)
    # The marginal SEs are the replicate SDs, so the identity
    # var(a - b) = var(a) + var(b) - 2 cov(a, b) holds draw for draw.
    cov = np.cov(da, db, ddof=1)[0, 1]
    np.testing.assert_allclose(res["se"] ** 2, a.se**2 + b.se**2 - 2 * cov, rtol=1e-10)
    # Reproducible: the same seed gives the same contrast and SE.
    again = sp.longitudinal_contrast(
        df, regime_a="always_treat", regime_b="never_treat", seed=7, **_KW
    )
    assert again["se"] == res["se"] and again["contrast"] == res["contrast"]


def test_gformula_contrast_without_seed_reports_the_seed_it_drew():
    df = _tv_panel(n=120)
    res = sp.longitudinal_contrast(
        df, regime_a="always_treat", regime_b="never_treat", **_KW
    )
    rerun = sp.longitudinal_contrast(
        df, regime_a="always_treat", regime_b="never_treat", seed=res["seed"], **_KW
    )
    assert rerun["se"] == res["se"]


def test_msm_path_says_its_estimate_ignores_the_regime_and_contrast_refuses():
    df = _tv_panel(n=150)
    with pytest.warns(sp.AssumptionWarning, match="does not depend on the regime"):
        a = sp.longitudinal_analyze(df, regime="always_treat", method="msm", **_KW)
    with pytest.warns(sp.AssumptionWarning):
        b = sp.longitudinal_analyze(df, regime="never_treat", method="msm", **_KW)
    assert a.estimate == b.estimate  # the reason the contrast is refused
    assert "regime-independent" in a.diagnostics["estimand"]
    with (
        pytest.warns(sp.AssumptionWarning),
        pytest.raises(sp.MethodIncompatibility, match="0 by construction"),
    ):
        sp.longitudinal_contrast(
            df, regime_a="always_treat", regime_b="never_treat", method="msm", **_KW
        )
