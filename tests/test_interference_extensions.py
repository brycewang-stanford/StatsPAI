"""
Tests for sp.network_hte + sp.inward_outward_spillover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_network_data(
    n: int = 400, tau_d: float = 0.5, tau_s: float = 0.3, seed: int = 113,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    # Unit-level propensity depends on X
    d = rng.binomial(1, 1.0 / (1 + np.exp(-(0.2 * x1 + 0.1 * x2))), size=n).astype(float)
    # Simulate neighbourhood exposure roughly correlated with D
    e = np.clip(0.3 * d + 0.4 * rng.uniform(size=n), 0, 1)
    y = 0.2 * x1 - 0.1 * x2 + tau_d * d + tau_s * e + rng.normal(0, 0.3, size=n)
    return pd.DataFrame({"y": y, "d": d, "e": e, "x1": x1, "x2": x2})


def test_network_hte_recovers_direct_and_spillover():
    df = _make_network_data(n=500, tau_d=0.5, tau_s=0.3, seed=113)
    res = sp.network_hte(
        df, y="y", treatment="d", neighbor_exposure="e",
        covariates=["x1", "x2"], n_folds=5, random_state=113,
    )
    assert abs(res.direct_effect - 0.5) < 0.12, res.direct_effect
    assert abs(res.spillover_effect - 0.3) < 0.12, res.spillover_effect
    assert res.direct_se > 0
    assert res.spillover_se > 0
    assert "Orthogonal Network HTE" in res.summary()


def test_network_hte_missing_columns_errors():
    df = _make_network_data(n=200)
    with pytest.raises(ValueError, match="Missing columns"):
        sp.network_hte(
            df, y="bogus", treatment="d", neighbor_exposure="e",
            covariates=["x1"],
        )


def test_inward_outward_spillover_recovers_direction():
    rng = np.random.default_rng(127)
    n = 500
    x1 = rng.normal(0, 1, size=n)
    d = rng.binomial(1, 0.5, size=n).astype(float)
    ein = rng.uniform(size=n)
    eout = rng.uniform(size=n)
    # True inward > outward
    y = 0.2 + 0.4 * d + 0.6 * ein + 0.2 * eout + 0.1 * x1 + rng.normal(0, 0.3, size=n)
    df = pd.DataFrame({"y": y, "d": d, "ein": ein, "eout": eout, "x1": x1})
    res = sp.inward_outward_spillover(
        df, y="y", treatment="d",
        inward_exposure="ein", outward_exposure="eout",
        covariates=["x1"],
    )
    assert abs(res.inward_effect - 0.6) < 0.1, res.inward_effect
    assert abs(res.outward_effect - 0.2) < 0.1, res.outward_effect
    assert res.inward_effect > res.outward_effect
    assert res.ratio_in_out > 1.0


def test_interference_extensions_in_registry():
    # network_hte is auto-registered via __all__
    fns = set(sp.list_functions())
    assert "network_hte" in fns or any(
        "network" in f and "hte" in f for f in fns
    ), fns
    assert "inward_outward_spillover" in fns or any(
        "inward" in f for f in fns
    ), fns
