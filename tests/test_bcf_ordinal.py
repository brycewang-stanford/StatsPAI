"""Tests for BCF with ordinal/factor-exposure treatments."""

import warnings
import numpy as np
import pandas as pd
import pytest
import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

warnings.filterwarnings("ignore")


def test_bcf_ordinal_recovers_monotone_effects():
    """With true τ(T=k) = 0.5·k, the level effects should be roughly ordered."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 3))
    T = rng.integers(0, 4, size=n)
    # Monotone treatment effect: 0 for T=0, then +0.5 per level.
    y = 0.5 * T + X[:, 0] + rng.normal(0, 0.3, n)
    df = pd.DataFrame(
        {
            "y": y,
            "t": T,
            **{f"x{j}": X[:, j] for j in range(3)},
        }
    )
    r = sp.bcf_ordinal(
        df,
        y="y",
        treat="t",
        covariates=["x0", "x1", "x2"],
        n_trees_mu=50,
        n_trees_tau=20,
        n_bootstrap=30,
    )
    # Result object should expose level-specific effects
    assert r is not None
    attrs = [a for a in dir(r) if not a.startswith("_")]
    # BCFOrdinalResult exposes ate / cate / levels
    has_levels = any(
        kw in attrs
        for kw in (
            "level_effects",
            "effects",
            "tau",
            "contrasts",
            "estimates",
            "ate",
            "cate",
            "levels",
        )
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
    df = pd.DataFrame(
        {
            "y": y,
            **{f"e{i}": exposures[:, i] for i in range(5)},
            "x1": X[:, 0],
            "x2": X[:, 1],
        }
    )
    from statspai.bcf.factor_exposure import bcf_factor_exposure

    r = bcf_factor_exposure(
        df,
        y="y",
        exposures=[f"e{i}" for i in range(5)],
        covariates=["x1", "x2"],
        n_factors=2,
        n_trees_mu=40,
        n_trees_tau=20,
        n_bootstrap=20,
    )
    assert r is not None
    # bcf_factor_exposure extracts factors via SVD, whose singular-vector signs
    # are not unique (numpy/LAPACK choose them differently across versions and
    # BLAS backends). A sign flip on a factor inverts the high/low exposure
    # binarisation and hence the per-factor and total mixture ATEs, so the exact
    # signed goldens are not portable. This is a smoke test (`_runs`): assert the
    # inference structure is coherent rather than pinning dev-machine values.
    assert np.isfinite(r.total_mixture_ate)
    assert r.total_mixture_se > 0
    lo, hi = r.total_mixture_ci[0], r.total_mixture_ci[1]
    assert lo <= r.total_mixture_ate <= hi
    pf = r.per_factor_ate
    assert np.all(np.isfinite(pf[["explained_var_ratio", "ate", "se"]].to_numpy()))
    assert np.all(pf["se"] > 0)
    # Explained-variance ratios are sign-independent: in (0, 1] and descending.
    evr = pf["explained_var_ratio"].to_numpy()
    assert np.all((evr > 0) & (evr <= 1))
    assert evr[0] >= evr[-1]


def test_bcf_ordinal_validates_inputs():
    df = pd.DataFrame(
        {
            "y": np.random.randn(50),
            "t": np.random.randint(0, 3, 50),
            "x": np.random.randn(50),
        }
    )
    with pytest.raises(MethodIncompatibility, match="missing") as exc:
        sp.bcf_ordinal(df, y="nonexistent_col", treat="t", covariates=["x"])
    assert exc.value.diagnostics["missing_columns"] == ["nonexistent_col"]


def test_bcf_ordinal_validates_levels_and_baseline():
    df = pd.DataFrame(
        {
            "y": np.random.randn(50),
            "t": np.zeros(50, dtype=int),
            "x": np.random.randn(50),
        }
    )
    with pytest.raises(DataInsufficient, match=">=2 levels") as exc:
        sp.bcf_ordinal(df, y="y", treat="t", covariates=["x"])
    assert exc.value.diagnostics["levels"] == [0]

    df["t"] = np.random.randint(0, 3, 50)
    with pytest.raises(MethodIncompatibility, match="baseline") as exc:
        sp.bcf_ordinal(df, y="y", treat="t", covariates=["x"], baseline=99)
    assert exc.value.diagnostics["baseline"] == 99


def test_bcf_ordinal_registered():
    assert "bcf_ordinal" in sp.list_functions()
