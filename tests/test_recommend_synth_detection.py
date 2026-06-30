"""Regression guard for FINDINGS F-001 (recommend_hit_rate benchmark).

A single-treated-unit comparative case study must route to synthetic control,
NOT to staggered DiD (Callaway-Sant'Anna is degenerate with one treated unit).
Conversely, a genuine staggered design with many treated units must keep
routing to DiD. These two invariants are what the hit-rate benchmark drove.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _single_treated_panel(n_donors: int = 20, n_periods: int = 30, treat_at: int = 20):
    """One treated unit + a donor pool over a long pre/post panel (SCM shape)."""
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(n_donors + 1):
        base = rng.normal(10, 1)
        for t in range(n_periods):
            treated = 1 if (unit == 0 and t >= treat_at) else 0
            y = base + 0.1 * t + rng.normal(0, 0.2) + (-3.0 if treated else 0.0)
            rows.append({"unit": unit, "year": t, "y": y, "treated": treated})
    return pd.DataFrame(rows)


def _staggered_panel(n_units: int = 60, n_periods: int = 6):
    """Many treated units adopting at different times (staggered DiD shape)."""
    rng = np.random.default_rng(1)
    rows = []
    for unit in range(n_units):
        cohort = rng.choice([2, 3, 4, 0])  # 0 == never treated
        for t in range(n_periods):
            treated = 1 if (cohort > 0 and t >= cohort) else 0
            y = unit * 0.0 + 0.1 * t + rng.normal(0, 0.2) + (1.0 if treated else 0.0)
            rows.append({"unit": unit, "year": t, "y": y, "treat": treated})
    return pd.DataFrame(rows)


def test_single_treated_unit_routes_to_synth():
    df = _single_treated_panel()
    rec = sp.recommend(df, y="y", treatment="treated", id="unit", time="year")
    assert rec.design == "synth", f"expected synth, got {rec.design}"
    top1 = rec.recommendations[0]["method"].lower()
    assert "synthetic control" in top1, top1
    # And it must NOT lead with a staggered-DiD estimator.
    assert "callaway" not in top1 and "sun-abraham" not in top1


def test_synth_recommendation_is_runnable():
    df = _single_treated_panel()
    rec = sp.recommend(df, y="y", treatment="treated", id="unit", time="year")
    res = rec.run(which=0)
    est = getattr(res, "estimate", getattr(res, "att", None))
    assert est is not None
    assert np.isfinite(float(est))


def test_staggered_design_still_routes_to_did():
    """No-regression guard: many treated units must stay DiD (Callaway top)."""
    df = _staggered_panel()
    rec = sp.recommend(df, y="y", treatment="treat", id="unit", time="year")
    assert rec.design == "did", f"expected did, got {rec.design}"
    top1 = rec.recommendations[0]["method"].lower()
    assert "callaway" in top1 or "sun-abraham" in top1, top1


@pytest.mark.parametrize(
    "dataset,kwargs,expected",
    [
        (
            "california_prop99",
            dict(y="cigsale", treatment="treated", id="state", time="year"),
            "synth",
        ),
        (
            "mpdta",
            dict(y="lemp", treatment="treat", id="countyreal", time="year"),
            "did",
        ),
    ],
)
def test_real_data_design_routing(dataset, kwargs, expected):
    """Real bundled data: Prop 99 → synth, mpdta (staggered) → did."""
    loader = getattr(sp.datasets, dataset)
    try:
        df = loader(simulated=False)
    except TypeError:
        df = loader()
    rec = sp.recommend(df, **kwargs)
    assert rec.design == expected, f"{dataset}: expected {expected}, got {rec.design}"
