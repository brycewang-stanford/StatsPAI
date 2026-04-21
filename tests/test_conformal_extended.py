"""
Tests for conformal_continuous + conformal_interference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_continuous_data(n: int = 400, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    t = rng.uniform(0, 2, size=n)
    y = 0.5 + 2.0 * t + 0.3 * x1 + 0.2 * x2 + rng.normal(0, 0.5, size=n)
    return pd.DataFrame({"t": t, "x1": x1, "x2": x2, "y": y})


def test_conformal_continuous_coverage():
    """Empirical coverage on a held-out test set should be near 1 - alpha."""
    df_train = _make_continuous_data(n=800, seed=101)
    df_test = _make_continuous_data(n=400, seed=103)
    res = sp.conformal_continuous(
        df_train, y="y", treatment="t", covariates=["x1", "x2"],
        test_data=df_test, alpha=0.1, random_state=101,
    )
    y_test = df_test["y"].to_numpy()
    covered = ((y_test >= res.predictions["lo"].to_numpy())
               & (y_test <= res.predictions["hi"].to_numpy()))
    coverage = covered.mean()
    # Allow slack for finite-sample noise.
    assert 0.80 < coverage < 0.98, coverage


def test_conformal_continuous_dose_grid():
    df = _make_continuous_data(n=400)
    test = df.head(5)
    res = sp.conformal_continuous(
        df, y="y", treatment="t", covariates=["x1", "x2"],
        test_data=test,
        dose_grid=np.linspace(0, 2, 11),
        alpha=0.1, random_state=107,
    )
    assert res.dose_curves is not None
    assert len(res.dose_curves) == 5 * 11
    assert set(res.dose_curves.columns) >= {"test_idx", "dose", "prediction", "lo", "hi"}


def test_conformal_continuous_missing_columns():
    df = _make_continuous_data(n=100)
    with pytest.raises(ValueError, match="missing"):
        sp.conformal_continuous(
            df, y="bogus", treatment="t", covariates=["x1"],
            test_data=df,
        )


def test_conformal_interference_cluster_level():
    rng = np.random.default_rng(109)
    rows = []
    for c in range(20):
        cluster_effect = rng.normal(0, 0.5)
        for u in range(10):
            x1 = rng.normal(0, 1)
            t = rng.binomial(1, 0.5)
            y = 0.5 * t + 0.3 * x1 + cluster_effect + rng.normal(0, 0.3)
            rows.append({"cluster": c, "t": t, "x1": x1, "y": y})
    df = pd.DataFrame(rows)
    res = sp.conformal_interference(
        df, y="y", treatment="t", cluster="cluster",
        covariates=["x1"],
        test_clusters=[0, 1, 2], alpha=0.1, random_state=109,
    )
    assert len(res.predictions) == 3
    assert res.quantile > 0
    assert set(res.predictions.columns) == {"cluster", "prediction", "lo", "hi"}
    # Interval widths must be strictly positive.
    widths = res.predictions["hi"] - res.predictions["lo"]
    assert (widths > 0).all()


def test_conformal_interference_requires_enough_clusters():
    df = pd.DataFrame({
        "cluster": [0, 0, 1, 1],
        "t": [0, 1, 0, 1],
        "y": [0.1, 0.2, 0.3, 0.4],
        "x1": [0.0, 0.1, 0.2, 0.3],
    })
    with pytest.raises(ValueError, match=">= 4 non-test"):
        sp.conformal_interference(
            df, y="y", treatment="t", cluster="cluster", covariates=["x1"],
            test_clusters=[0],
        )
