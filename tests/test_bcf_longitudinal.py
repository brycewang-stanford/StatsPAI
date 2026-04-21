"""
Tests for sp.bcf_longitudinal (BCFLong, arXiv:2508.08418).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_longitudinal(
    n_units: int = 60, T: int = 4, tau: float = 1.5, seed: int = 53,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        x1 = rng.normal()
        x2 = rng.normal()
        u_eff = rng.normal(0, 0.3)
        for t in range(T):
            d = int(rng.uniform() < 0.5)
            y = 0.5 * x1 + 0.3 * x2 + tau * d + u_eff + 0.2 * t + rng.normal(0, 0.3)
            rows.append({
                "unit": u, "time": t, "y": y, "d": d,
                "x1": x1, "x2": x2,
            })
    return pd.DataFrame(rows)


def test_bcf_longitudinal_recovers_constant_ate():
    df = _make_longitudinal(n_units=60, T=4, tau=1.5, seed=53)
    res = sp.bcf_longitudinal(
        df, outcome="y", treatment="d", unit="unit", time="time",
        covariates=["x1", "x2"], n_trees_mu=80, n_trees_tau=30,
        n_bootstrap=30, random_state=53,
    )
    assert isinstance(res, sp.BCFLongResult)
    # Average should recover tau ≈ 1.5 within 3 SE
    assert abs(res.average_ate - 1.5) < 3 * max(res.average_se, 0.2)
    # Per-time table should have T rows
    assert len(res.per_time_ate) == 4
    assert {"time", "ate_point", "se", "ci_low", "ci_high"} <= set(res.per_time_ate.columns)
    # Individual CATE should have n_units * T rows
    assert len(res.individual_cate) == 60 * 4


def test_bcf_longitudinal_requires_multi_time():
    df = pd.DataFrame({
        "unit": [1, 2, 3, 4], "time": [0, 0, 0, 0],
        "y": [1.0, 2.0, 3.0, 4.0], "d": [0, 0, 1, 1],
        "x1": [0.1, 0.2, 0.3, 0.4], "x2": [0.0, 0.0, 0.0, 0.0],
    })
    with pytest.raises(ValueError, match="distinct time points"):
        sp.bcf_longitudinal(
            df, outcome="y", treatment="d", unit="unit", time="time",
            covariates=["x1", "x2"], n_bootstrap=5,
        )


def test_bcf_longitudinal_duplicate_unit_time_errors():
    df = pd.DataFrame({
        "unit": [1, 1, 2], "time": [0, 0, 0],
        "y": [1.0, 2.0, 3.0], "d": [0, 1, 0],
        "x1": [0.1, 0.2, 0.3], "x2": [0.0, 0.0, 0.0],
    })
    with pytest.raises(ValueError, match="unique per row"):
        sp.bcf_longitudinal(
            df, outcome="y", treatment="d", unit="unit", time="time",
            covariates=["x1", "x2"], n_bootstrap=5,
        )


def test_bcf_longitudinal_summary_and_registry():
    df = _make_longitudinal(n_units=40, T=3, tau=1.0, seed=59)
    res = sp.bcf_longitudinal(
        df, outcome="y", treatment="d", unit="unit", time="time",
        covariates=["x1", "x2"], n_trees_mu=40, n_trees_tau=20,
        n_bootstrap=25, random_state=59,
    )
    s = res.summary()
    assert "BCFLong" in s
    assert "bcf_longitudinal" in set(sp.list_functions())
