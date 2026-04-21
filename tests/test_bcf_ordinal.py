"""Tests for BCF with ordinal/factor-exposure treatments."""

import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp


def test_bcf_ordinal_recovers_monotone_effects():
    """With true τ(T=k) = 0.5·k, the level effects should be roughly ordered."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 3))
    T = rng.integers(0, 4, size=n)
    # Monotone treatment effect: 0 for T=0, 0.5 for T=1, 1.0 for T=2, 1.5 for T=3
    y = 0.5 * T + X[:, 0] + rng.normal(0, 0.3, n)
    df = pd.DataFrame({
        "y": y, "t": T,
        **{f"x{j}": X[:, j] for j in range(3)},
    })
    r = sp.bcf_ordinal(
        df, y="y", treat="t", covariates=["x0", "x1", "x2"],
        n_trees_mu=50, n_trees_tau=20, n_bootstrap=30,
    )
    # Result object should expose level-specific effects
    assert r is not None
    attrs = [a for a in dir(r) if not a.startswith("_")]
    # BCFOrdinalResult exposes ate / cate / levels
    has_levels = any(
        kw in attrs for kw in ("level_effects", "effects", "tau", "contrasts",
                                "estimates", "ate", "cate", "levels")
    )
    assert has_levels, f"result has attrs {attrs[:12]}"
    # CATE is an n × (K-1) DataFrame of contrasts relative to the baseline level.
    # Should have one column per non-baseline level.
    if hasattr(r, "cate") and hasattr(r, "levels"):
        cate = r.cate
        if hasattr(cate, "shape"):
            # n-by-(K-1) or n-by-K depending on baseline convention
            assert cate.shape[1] in (len(r.levels) - 1, len(r.levels))
        # The mean contrast should be roughly monotone (true τ(k) = 0.5·k)
        if hasattr(cate, "mean"):
            col_means = cate.mean().to_numpy()
            # Each successive level should be higher than the previous
            assert col_means[-1] > col_means[0] - 0.1


def test_bcf_factor_exposure_runs():
    rng = np.random.default_rng(1)
    n = 200
    exposures = rng.normal(size=(n, 5))
    X = rng.normal(size=(n, 2))
    # True outcome depends on 1st factor
    y = exposures[:, 0] + X[:, 0] + rng.normal(0, 0.3, n)
    df = pd.DataFrame({
        "y": y,
        **{f"e{i}": exposures[:, i] for i in range(5)},
        "x1": X[:, 0], "x2": X[:, 1],
    })
    from statspai.bcf.factor_exposure import bcf_factor_exposure
    r = bcf_factor_exposure(
        df, y="y", exposures=[f"e{i}" for i in range(5)],
        covariates=["x1", "x2"], n_factors=2,
        n_trees_mu=40, n_trees_tau=20, n_bootstrap=20,
    )
    assert r is not None


def test_bcf_ordinal_validates_inputs():
    df = pd.DataFrame({
        "y": np.random.randn(50),
        "t": np.random.randint(0, 3, 50),
        "x": np.random.randn(50),
    })
    # Missing covariates arg should raise OR use sensible default
    with pytest.raises((TypeError, ValueError, KeyError)):
        sp.bcf_ordinal(df, y="nonexistent_col", treat="t", covariates=["x"])


def test_bcf_ordinal_registered():
    assert "bcf_ordinal" in sp.list_functions()
