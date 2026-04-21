"""
Tests for sp.overlap_weighted_did + sp.dl_propensity_score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_2x2_panel(n: int = 600, tau: float = 1.0, seed: int = 139) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n):
        x = rng.normal(0, 1)
        # Treatment prob depends on X
        t = int(rng.uniform() < 1 / (1 + np.exp(-0.5 * x)))
        unit_fe = rng.normal(0, 0.5)
        for time in (0, 1):
            y = unit_fe + 0.3 * x + 0.2 * time + (tau * t if time == 1 else 0.0) + rng.normal(0, 0.3)
            rows.append({"unit": u, "time": time, "t": t, "x": x, "y": y})
    return pd.DataFrame(rows)


def test_overlap_weighted_did_recovers_tau():
    df = _make_2x2_panel(n=400, tau=1.0, seed=139)
    res = sp.overlap_weighted_did(
        df, y="y", treat="t", time="time", covariates=["x"],
    )
    assert res.method == "overlap_weighted_did"
    assert abs(res.estimate - 1.0) < 0.3, res.estimate
    assert res.se > 0
    lo, hi = res.ci
    assert lo < res.estimate < hi


def test_overlap_weighted_did_without_covariates_matches_unweighted():
    df = _make_2x2_panel(n=300, seed=149)
    res = sp.overlap_weighted_did(
        df, y="y", treat="t", time="time",
    )
    # Without covariates weights = 1, so this is standard DID.
    means = (df.groupby(["t", "time"])["y"].mean())
    expected = (means[(1, 1)] - means[(1, 0)]) - (means[(0, 1)] - means[(0, 0)])
    assert abs(res.estimate - expected) < 1e-10


def test_overlap_weighted_did_with_gbm_ps():
    df = _make_2x2_panel(n=400, tau=1.5, seed=151)
    res = sp.overlap_weighted_did(
        df, y="y", treat="t", time="time", covariates=["x"],
        ps_model="gbm",
    )
    assert abs(res.estimate - 1.5) < 0.5, res.estimate


def test_overlap_weighted_did_validates_binary():
    df = _make_2x2_panel(n=100)
    df["t"] = df["t"] * 2.0  # 0, 2
    with pytest.raises(ValueError, match="must be binary"):
        sp.overlap_weighted_did(df, y="y", treat="t", time="time")


def test_dl_propensity_score_returns_valid_probs():
    df = _make_2x2_panel(n=300)
    probs = sp.dl_propensity_score(
        df.query("time == 0"), treatment="t", covariates=["x"],
    )
    assert probs.ndim == 1
    assert (probs > 0).all()
    assert (probs < 1).all()


def test_overlap_did_in_registry():
    fns = set(sp.list_functions())
    assert "overlap_weighted_did" in fns
    assert "dl_propensity_score" in fns
