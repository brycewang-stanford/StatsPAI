"""Tests for the unified .sensitivity() dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import statspai as sp


@dataclass
class _FakeResult:
    estimate: float
    se: float
    ci: tuple[float, float]


def test_unified_sensitivity_from_plain_result():
    res = _FakeResult(estimate=0.3, se=0.1, ci=(0.1, 0.5))
    dash = sp.unified_sensitivity(res)
    assert np.isfinite(dash.e_value_point)
    assert dash.breakdown is not None
    assert "bias_to_flip" in dash.breakdown
    assert dash.rr_observed > 0


def test_unified_sensitivity_zero_effect():
    res = _FakeResult(estimate=0.0, se=0.1, ci=(-0.2, 0.2))
    dash = sp.unified_sensitivity(res)
    # breakdown undefined when sign is 0
    assert (
        np.isnan(dash.breakdown["bias_to_flip"]) or dash.breakdown["bias_to_flip"] == 0
    )


def test_sensitivity_method_on_causal_result_smoke():
    """Smoke test: DID result should expose a .sensitivity() method."""
    import pandas as pd

    rng = np.random.default_rng(5)
    n_unit = 40
    rows = []
    for u in range(n_unit):
        for t in range(2):
            post = int(t == 1)
            treated = int(u < n_unit // 2)
            y = 1.0 + 0.2 * post + 0.8 * (post * treated) + rng.normal(0, 0.5)
            rows.append({"unit": u, "time": t, "y": y, "treat": treated, "post": post})
    df = pd.DataFrame(rows)
    res = sp.did_2x2(df, y="y", treat="treat", time="post")
    # Should have a sensitivity method
    assert hasattr(res, "sensitivity")
    dash = res.sensitivity()
    assert np.isfinite(dash.e_value_point)
    assert dash.rr_observed > 0


def test_unified_sensitivity_summary_runs():
    res = _FakeResult(estimate=0.3, se=0.1, ci=(0.1, 0.5))
    dash = sp.unified_sensitivity(res)
    text = dash.summary()
    assert "Sensitivity" in text
    assert "E-value" in text


def test_unified_sensitivity_oster_with_r2():
    res = _FakeResult(estimate=0.3, se=0.1, ci=(0.1, 0.5))
    dash = sp.unified_sensitivity(
        res,
        r2_treated=0.25,
        r2_controlled=0.40,
    )
    # Oster may fail if the underlying function has a different signature;
    # in that case we expect a note but no exception.
    # The result should still be constructed.
    assert np.isfinite(dash.e_value_point)


def test_unified_sensitivity_oster_matches_oster_bounds():
    """The Oster component must report the breakdown delta (delta_for_zero),
    with r2_treated routed to r2_short and r2_controlled to r2_long."""
    res = _FakeResult(estimate=0.3, se=0.1, ci=(0.1, 0.5))
    dash = sp.unified_sensitivity(
        res,
        r2_treated=0.15,
        r2_controlled=0.45,
        beta_uncontrolled=0.5,
    )
    direct = sp.oster_bounds(
        beta_short=0.5,
        r2_short=0.15,
        beta_long=0.3,
        r2_long=0.45,
        r_max=1.0,
        delta=1.0,
    )
    assert dash.oster is not None
    assert dash.oster["delta"] == pytest.approx(direct["delta_for_zero"])
    assert dash.oster["beta_star"] == pytest.approx(direct["beta_adjusted"])
    # The breakdown delta is not the input proportionality delta (1.0).
    assert dash.oster["delta"] != pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Sensemakr component: requires raw data — dead-code fix lock
# (previously called sensemakr(result, treatment=...) which always raised
#  TypeError and was silently swallowed into notes)
# ---------------------------------------------------------------------------


def _toy_regression_df(n: int = 300, seed: int = 42):
    import pandas as pd

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = (0.5 * x1 - 0.3 * x2 + rng.normal(size=n) > 0).astype(float)
    y = 1.0 + 0.6 * d + 0.8 * x1 + 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def test_unified_sensitivity_sensemakr_matches_direct_call():
    df = _toy_regression_df()
    res = _FakeResult(estimate=0.6, se=0.1, ci=(0.4, 0.8))
    dash = sp.unified_sensitivity(
        res,
        data=df,
        y="y",
        treat="d",
        controls=["x1", "x2"],
    )
    direct = sp.sensemakr(df, y="y", treat="d", controls=["x1", "x2"])
    assert dash.sensemakr is not None
    assert np.isfinite(dash.sensemakr["rv_q1"])
    assert dash.sensemakr["rv_q1"] == pytest.approx(direct["rv_q"])
    assert dash.sensemakr["rv_qa"] == pytest.approx(direct["rv_qa"])
    assert not any("Sensemakr skipped" in n for n in dash.notes)
    # And the summary should render the RV line
    assert "Sensemakr RV(q=1)" in dash.summary()


def test_unified_sensitivity_sensemakr_without_data_notes_guidance():
    """Without raw data the component is skipped with an actionable note —
    never a swallowed TypeError."""
    res = _FakeResult(estimate=0.3, se=0.1, ci=(0.1, 0.5))
    dash = sp.unified_sensitivity(res)
    assert dash.sensemakr is None
    assert any("sp.sensemakr(data, y, treat, controls)" in n for n in dash.notes)
    assert not any("TypeError" in n for n in dash.notes)


def test_unified_sensitivity_sensemakr_partial_args_note():
    df = _toy_regression_df(n=50)
    res = _FakeResult(estimate=0.6, se=0.1, ci=(0.4, 0.8))
    dash = sp.unified_sensitivity(res, data=df, y="y")
    assert dash.sensemakr is None
    note = next(n for n in dash.notes if "Sensemakr skipped" in n)
    assert "treat" in note and "controls" in note


def test_sensitivity_method_forwards_sensemakr_kwargs():
    """EconometricResults.sensitivity() must forward data/y/treat/controls."""
    df = _toy_regression_df()
    res = sp.regress("y ~ d + x1 + x2", df)
    dash = res.sensitivity(data=df, y="y", treat="d", controls=["x1", "x2"])
    direct = sp.sensemakr(df, y="y", treat="d", controls=["x1", "x2"])
    assert dash.sensemakr is not None
    assert dash.sensemakr["rv_q1"] == pytest.approx(direct["rv_q"])


# ---------------------------------------------------------------------------
# Rosenbaum component: matched_pairs coercion — dead-code fix lock
# (previously imported a nonexistent module and passed the result object
#  where outcome arrays were expected)
# ---------------------------------------------------------------------------


def test_unified_sensitivity_rosenbaum_matches_direct_call():
    from dataclasses import dataclass, field

    rng = np.random.default_rng(7)
    control = rng.normal(0.0, 1.0, 60)
    treated = control + 0.5 + rng.normal(0.0, 1.0, 60)

    @dataclass
    class R:
        estimate: float
        se: float
        ci: tuple
        matched_pairs: tuple = field(default=None)

    res = R(estimate=0.5, se=0.15, ci=(0.2, 0.8), matched_pairs=(treated, control))
    dash = sp.unified_sensitivity(res)
    direct = sp.rosenbaum_bounds(treated, control, alternative="two-sided")
    assert dash.rosenbaum is not None
    assert dash.rosenbaum["gamma_critical"] == float(direct.gamma_critical)
    assert not any("Rosenbaum Gamma skipped" in n for n in dash.notes)
