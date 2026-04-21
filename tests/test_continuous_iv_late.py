"""Tests for continuous-instrument LATE (Xie et al. 2025)."""

import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp


def _continuous_late_data(n=600, true_late=1.5, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    # Heterogeneous compliance: "responsive" units have stronger Z sensitivity
    pi = 0.5 + 0.3 * (rng.uniform(size=n) > 0.3)
    D = pi * Z + rng.normal(0, 0.3, n)
    Y = true_late * D + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": Y, "d": D, "z": Z})


def test_continuous_late_recovers_truth():
    df = _continuous_late_data(n=800, true_late=1.5, seed=0)
    r = sp.continuous_iv_late(df, y="y", treat="d", instrument="z",
                                n_quantiles=5, n_boot=50, seed=0)
    assert abs(r.estimate - 1.5) < 0.5, f"Estimate {r.estimate}"
    assert r.se > 0
    assert r.ci[0] < r.estimate < r.ci[1]
    assert 0 < r.complier_share <= 1


def test_continuous_late_summary_not_degenerate():
    df = _continuous_late_data(n=300, seed=1)
    r = sp.continuous_iv_late(df, y="y", treat="d", instrument="z",
                                n_quantiles=4, n_boot=20, seed=1)
    s = r.summary()
    # Bug regression: summary should NOT contain the header repeated dozens of times
    assert s.count("Continuous-Instrument LATE") == 1
    assert "=" in s
    assert "N" in s


def test_continuous_late_registry():
    assert "continuous_iv_late" in sp.list_functions()


def test_continuous_late_handles_few_quantiles():
    df = _continuous_late_data(n=200, seed=2)
    # Even with 2 quantiles it should produce a finite estimate
    r = sp.continuous_iv_late(df, y="y", treat="d", instrument="z",
                                n_quantiles=2, n_boot=10, seed=2)
    assert np.isfinite(r.estimate)
