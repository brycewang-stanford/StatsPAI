"""Coverage campaign — ``statspai.rd.rdml`` and ``statspai.rd.hte``.

These two RD sub-modules have no dedicated cov95 file and were the two
largest remaining rd gaps. Covered here:

- ``rd_forest``       (Athey-Wager causal-forest RD CATE);
- ``rd_boost``        (gradient-boosting RD CATE);
- ``rd_lasso``        (post-double-selection LASSO RD);
- ``rd_cate_summary`` (multi-method CATE comparison bundle);
- ``rdhte``           (heterogeneous RD effects by a moderator);
- ``rdbwhte``         (HTE-optimal bandwidth);
- ``rdhte_lincom``    (linear combination of HTE evaluation points).

DGP: a sharp RD with a moderator ``z`` so the jump is +2 for ``z=0`` and
+5 for ``z=1`` (average ≈3.5). Assertions check recovery of that structure
(level at z=0, presence of heterogeneity, positive bandwidth), never
fabricated numbers. Forests/boosting use small ensembles for test speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

JUMP_Z0 = 2.0
JUMP_Z1 = 5.0
ATE = 0.5 * JUMP_Z0 + 0.5 * JUMP_Z1  # ≈ 3.5 with balanced z


def _rd_hte_data(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    treat = (x >= 0).astype(float)
    z = rng.integers(0, 2, n).astype(float)
    cov1 = rng.normal(size=n)
    eff = JUMP_Z0 + (JUMP_Z1 - JUMP_Z0) * z
    y = 0.5 * x + eff * treat + 0.3 * cov1 + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "cov1": cov1})


@pytest.fixture(scope="module")
def rd_df():
    return _rd_hte_data()


def _avg(result):
    v = getattr(result, "estimate", result)
    arr = np.atleast_1d(np.asarray(v, dtype=float))
    arr = arr[np.isfinite(arr)]
    assert arr.size
    return float(arr.ravel()[0])


# ── rdml: ML-based RD CATE estimators ───────────────────────────────────


def test_rd_forest(rd_df):
    r = sp.rd_forest(rd_df, y="y", x="x", c=0, covs=["cov1"], n_trees=60)
    assert abs(_avg(r) - ATE) < 1.5
    assert r.se > 0


def test_rd_boost(rd_df):
    r = sp.rd_boost(rd_df, y="y", x="x", c=0, covs=["cov1"], n_estimators=60)
    assert abs(_avg(r) - ATE) < 1.5


def test_rd_lasso(rd_df):
    r = sp.rd_lasso(rd_df, y="y", x="x", c=0, covs=["cov1"])
    assert abs(_avg(r) - ATE) < 1.5


def test_rd_cate_summary_bundle(rd_df):
    out = sp.rd_cate_summary(rd_df, y="y", x="x", c=0, covs=["cov1"])
    assert isinstance(out, dict)
    for key in ("forest", "boost", "lasso", "comparison"):
        assert key in out
    comp = out["comparison"]
    assert hasattr(comp, "columns") and "estimate" in comp.columns
    assert len(comp) >= 3  # one row per ML method


# ── hte: heterogeneous RD effects ───────────────────────────────────────


def test_rdhte_recovers_heterogeneity(rd_df):
    r = sp.rdhte(rd_df, y="y", x="x", z="z", c=0)
    det = r.detail
    assert {"z_value", "cate"}.issubset(det.columns)
    by_z = det.groupby("z_value")["cate"].first()
    # at z=0 the jump is ~2; the moderator induces a clear spread toward +5
    lo_z = by_z.index.min()
    assert abs(by_z.loc[lo_z] - JUMP_Z0) < 0.8
    assert (by_z.max() - by_z.min()) > 1.5  # heterogeneity recovered


def test_rdbwhte_positive(rd_df):
    h = sp.rdbwhte(rd_df, y="y", x="x", z="z", c=0)
    assert np.isfinite(h) and h > 0


def test_rdhte_lincom(rd_df):
    r = sp.rdhte(rd_df, y="y", x="x", z="z", c=0)
    n_pts = len(r.detail)
    w = np.ones(n_pts) / n_pts  # uniform average across eval points
    out = sp.rdhte_lincom(r, weights=w)
    assert isinstance(out, dict)
    est = out.get("estimate", out.get("value"))
    assert est is not None and np.isfinite(float(est))


def test_rdhte_continuous_moderator_and_eval_points():
    rng = np.random.default_rng(2)
    n = 1500
    x = rng.uniform(-1, 1, n)
    treat = (x >= 0).astype(float)
    zc = rng.uniform(0, 1, n)  # continuous moderator
    y = 0.5 * x + (2.0 + 3.0 * zc) * treat + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"y": y, "x": x, "z": zc})
    # explicit evaluation points exercise the eval_points branch
    r = sp.rdhte(df, y="y", x="x", z="z", c=0, eval_points=np.array([0.2, 0.5, 0.8]))
    det = r.detail
    assert len(det) >= 3
    # CATE should increase with the moderator (true slope +3)
    cate_by_z = det.sort_values("z_value")["cate"].to_numpy()
    assert cate_by_z[-1] > cate_by_z[0]


def test_rdhte_quadratic_and_multivariate_z():
    rng = np.random.default_rng(4)
    n = 2000
    x = rng.uniform(-1, 1, n)
    treat = (x >= 0).astype(float)
    z1 = rng.integers(0, 2, n).astype(float)
    z2 = rng.normal(size=n)
    y = 0.5 * x + (2.0 + 1.5 * z1) * treat + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})
    # p=2 (quadratic) + multivariate moderator grid
    r = sp.rdhte(df, y="y", x="x", z=["z1", "z2"], c=0, p=2, n_eval=9)
    assert np.all(np.isfinite(r.detail["cate"].to_numpy()))


def test_rdhte_plot(rd_df):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    r = sp.rdhte(rd_df, y="y", x="x", z="z", c=0)
    ax = r.plot()
    assert ax is not None
