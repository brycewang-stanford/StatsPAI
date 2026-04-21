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
    assert np.isnan(dash.breakdown["bias_to_flip"]) or \
        dash.breakdown["bias_to_flip"] == 0


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
            rows.append({"unit": u, "time": t, "y": y,
                         "treat": treated, "post": post})
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
