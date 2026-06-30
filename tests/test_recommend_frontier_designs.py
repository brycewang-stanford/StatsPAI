"""Regression guard for FINDINGS F-004 (recommend_hit_rate benchmark).

The frontier design families that recommend was blind to — bunching, RKD,
triple-difference (DDD), shift-share/Bartik IV, and decomposition — now route
to the already-shipping estimators when the design is declared. Each must
produce the right top-1 recommendation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture()
def df():
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "treat": rng.integers(0, 2, n),
            "time": rng.integers(0, 5, n),
            "c1": rng.normal(size=n),
        }
    )


@pytest.mark.parametrize(
    "design,kwargs,needle",
    [
        ("bunching", dict(y="x", running_var="x", cutoff=0.0), "bunching"),
        ("rkd", dict(y="y", running_var="x", cutoff=0.0), "regression kink"),
        ("ddd", dict(y="y", treatment="treat", time="time"), "triple difference"),
        ("bartik", dict(y="y", treatment="treat"), "bartik"),
        (
            "decomposition",
            dict(y="y", treatment="treat", covariates=["c1"]),
            "oaxaca",
        ),
    ],
)
def test_frontier_design_routes_to_estimator(df, design, kwargs, needle):
    rec = sp.recommend(df, design=design, **kwargs)
    assert rec.design == design, f"expected {design}, got {rec.design}"
    assert rec.recommendations, f"{design}: no recommendation produced"
    top1 = rec.recommendations[0]["method"].lower()
    assert needle in top1, f"{design}: top-1 was {top1!r}"


def test_rd_sharp_vs_fuzzy_autodetection():
    """High-confidence RD refinement: a deterministic step at the cutoff is
    sharp; a treatment-probability jump is fuzzy. Only refines a detected RD."""
    fuzzy = sp.dgp_rd(n=1500, fuzzy=True, cutoff=0.0, seed=13)
    rf = sp.recommend(fuzzy, y="y", running_var="x", cutoff=0.0, treatment="treatment")
    assert "fuzzy" in rf.recommendations[0]["method"].lower()
    sharp = sp.dgp_rd(n=1500, fuzzy=False, cutoff=0.0, seed=9)
    rs = sp.recommend(sharp, y="y", running_var="x", cutoff=0.0, treatment="treatment")
    assert "fuzzy" not in rs.recommendations[0]["method"].lower()
    assert "rd" in rs.recommendations[0]["method"].lower()


def test_frontier_does_not_regress_core_designs(df):
    """The new branches must not change the core dispatch (a DiD panel still
    routes to DiD, not a frontier design)."""
    rng = np.random.default_rng(1)
    rows = []
    for u in range(60):
        cohort = rng.choice([2, 3, 0])
        for t in range(5):
            tr = 1 if (cohort > 0 and t >= cohort) else 0
            rows.append(
                {
                    "unit": u,
                    "year": t,
                    "y": 0.1 * t + tr + rng.normal(0, 0.2),
                    "treat": tr,
                }
            )
    panel = pd.DataFrame(rows)
    rec = sp.recommend(panel, y="y", treatment="treat", id="unit", time="year")
    assert rec.design == "did"
