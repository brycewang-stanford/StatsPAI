"""
Tests for sp.sequential_sdid — Arkhangelsky & Samkov (2024).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_staggered_panel(
    *, n_units: int = 40, n_periods: int = 20,
    cohorts: dict = None, true_ate: float = 2.0, seed: int = 17,
) -> pd.DataFrame:
    """Staggered-adoption panel.

    ``cohorts`` maps unit → first-treated period (0 = never-treated).
    """
    rng = np.random.default_rng(seed)
    if cohorts is None:
        # Default: roughly equal split across 3 cohorts + never-treated
        cohorts = {}
        for u in range(n_units):
            if u < n_units // 4:
                cohorts[u] = 0  # never-treated
            elif u < n_units // 2:
                cohorts[u] = 8
            elif u < 3 * n_units // 4:
                cohorts[u] = 12
            else:
                cohorts[u] = 15
    unit_fe = rng.normal(0, 1, size=n_units)
    time_fe = np.linspace(0, 2, n_periods)
    rows = []
    for u in range(n_units):
        g = cohorts[u]
        for t in range(n_periods):
            treated = (g != 0) and (t >= g)
            y = (
                unit_fe[u] + time_fe[t]
                + (true_ate if treated else 0.0)
                + rng.normal(0, 0.3)
            )
            rows.append({
                "unit": u, "time": t, "y": y,
                "cohort": g,
            })
    return pd.DataFrame(rows)


def test_sequential_sdid_recovers_true_ate():
    df = _make_staggered_panel(true_ate=2.0, seed=19)
    res = sp.sequential_sdid(
        df, outcome="y", unit="unit", time="time", cohort="cohort",
        se_method="placebo", n_reps=30, seed=19,
    )
    assert res.method == "sequential_sdid"
    assert res.estimand == "ATT"
    # 3 cohorts, all recoverable.
    assert res.model_info["n_valid_cohorts"] >= 2
    assert abs(res.estimate - 2.0) < 0.6, res.estimate
    assert isinstance(res.detail, pd.DataFrame)
    assert {"cohort", "att", "se", "n_treated"} <= set(res.detail.columns)


def test_sequential_sdid_single_cohort_matches_sdid():
    """With only one cohort, sequential SDID should agree with plain SDID."""
    cohorts = {u: (0 if u < 20 else 12) for u in range(30)}
    df = _make_staggered_panel(
        n_units=30, n_periods=20, cohorts=cohorts,
        true_ate=1.5, seed=23,
    )
    res_seq = sp.sequential_sdid(
        df, outcome="y", unit="unit", time="time", cohort="cohort",
        se_method="placebo", n_reps=30, seed=23,
    )
    # Single-cohort → one row in detail
    assert len(res_seq.detail) == 1
    treated = [u for u, g in cohorts.items() if g != 0]
    res_plain = sp.sdid(
        df, y="y", unit="unit", time="time",
        treat_unit=treated, treat_time=12, method="sdid",
        se_method="placebo", n_reps=30, seed=23,
    )
    # Point estimates should be within numerical noise.
    assert abs(res_seq.estimate - res_plain.estimate) < 0.1


def test_sequential_sdid_no_treated_errors():
    df = pd.DataFrame({
        "unit": [1, 1, 2, 2], "time": [0, 1, 0, 1],
        "y": [1.0, 2.0, 3.0, 4.0], "cohort": [0, 0, 0, 0],
    })
    with pytest.raises(ValueError, match="No treated cohorts"):
        sp.sequential_sdid(
            df, outcome="y", unit="unit", time="time", cohort="cohort",
        )


def test_sequential_sdid_invalid_cohort_weights():
    df = _make_staggered_panel(seed=29)
    with pytest.raises(ValueError, match="cohort_weights"):
        sp.sequential_sdid(
            df, outcome="y", unit="unit", time="time", cohort="cohort",
            cohort_weights="bogus",
        )


def test_sequential_sdid_in_registry():
    fns = set(sp.list_functions())
    assert "sequential_sdid" in fns
