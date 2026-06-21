"""Coverage campaign — HDFE absorber dispatch paths in ``statspai.panel.hdfe``.

Exercises the *coverable* dispatch lines of the high-dimensional fixed-effects
absorber (the numba kernel bodies themselves are JIT-compiled machine code and
are untraceable by coverage.py, so they are out of scope here):

- ``demean`` on a 1-D column, both unweighted and weighted (the ``x.ndim == 1``
  fast path);
- ``demean`` with the FE passed as a DataFrame vs. as a raw ndarray;
- ``absorb_ols`` with frequency weights (the weighted normal-equations branch)
  and with cluster-robust SEs.

Assertions check the defining property of within-transformation (group means
collapse to ~0 after absorbing that FE) and that ``absorb_ols`` recovers the
known DGP slopes — not fabricated numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _twoway_panel(n_id=40, T=8, seed=9):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_id):
        a = rng.normal(0, 2)
        for t in range(T):
            tau = 0.5 * t
            x1 = rng.normal() + 0.3 * a
            x2 = rng.normal()
            y = a + tau + 2.0 * x1 - 1.0 * x2 + rng.normal(0, 0.5)
            rows.append({"id": i, "time": t, "y": y, "x1": x1, "x2": x2})
    return pd.DataFrame(rows)


def test_demean_1d_unweighted_and_weighted():
    df = _twoway_panel()
    fe = df[["id", "time"]]
    x = df["x1"].to_numpy(dtype=float)

    dm = np.asarray(sp.demean(x.copy(), fe)[0])  # 1-D unweighted path
    # After absorbing entity FE, the within-entity means are ~0.
    by_id = pd.Series(dm).groupby(df["id"].values).mean()
    assert np.abs(by_id).max() < 1e-6

    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 1.5, len(df))
    dmw = np.asarray(sp.demean(x.copy(), fe, weights=w)[0])  # 1-D weighted path
    assert np.all(np.isfinite(dmw))
    # Weighted within-entity weighted means are ~0.
    num = pd.Series(dmw * w).groupby(df["id"].values).sum()
    den = pd.Series(w).groupby(df["id"].values).sum()
    assert np.abs(num / den).max() < 1e-6


def test_demean_fe_dataframe_matches_ndarray():
    df = _twoway_panel()
    x = df["x1"].to_numpy(dtype=float)
    dm_df = np.asarray(sp.demean(x.copy(), df[["id", "time"]])[0])
    dm_arr = np.asarray(sp.demean(x.copy(), df[["id", "time"]].to_numpy())[0])
    assert np.allclose(dm_df, dm_arr, atol=1e-8)


def test_absorb_ols_weighted_recovers_slopes():
    df = _twoway_panel()
    y = df["y"].to_numpy(dtype=float)
    X = df[["x1", "x2"]].to_numpy(dtype=float)
    fe = df[["id", "time"]]
    rng = np.random.default_rng(1)
    w = rng.uniform(0.5, 1.5, len(df))

    res = sp.absorb_ols(y, X, fe, weights=w)
    coef = np.asarray(res["coef"], dtype=float).ravel()
    assert coef.shape[0] == 2
    # True slopes are (2.0, -1.0) net of the two-way FE.
    assert abs(coef[0] - 2.0) < 0.15
    assert abs(coef[1] + 1.0) < 0.15
    assert np.all(np.asarray(res["se"]) > 0)


def test_absorb_ols_cluster_robust_se():
    df = _twoway_panel()
    y = df["y"].to_numpy(dtype=float)
    X = df[["x1", "x2"]].to_numpy(dtype=float)
    fe = df[["id", "time"]]
    cluster = df["id"].to_numpy()

    res = sp.absorb_ols(y, X, fe, cluster=cluster)
    coef = np.asarray(res["coef"], dtype=float).ravel()
    se = np.asarray(res["se"], dtype=float).ravel()
    assert abs(coef[0] - 2.0) < 0.2
    assert np.all(np.isfinite(se)) and np.all(se > 0)
