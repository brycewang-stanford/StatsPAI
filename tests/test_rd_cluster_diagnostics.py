"""``cluster=`` in the RD diagnostics: rdbwsensitivity, rdplacebo, rdbalance,
rdsummary and rdplot.

The first four refit ``sp.rdrobust``, whose clustered inference is checked
against R/Stata elsewhere; here each row must be exactly the clustered
``rdrobust`` fit it stands for. ``rdplot`` computes its own intervals: each
bin's clustered interval must equal the intercept-only OLS with CR1 on that
bin (statsmodels, t(G - 1)), and the shaded band the clustered weighted
polynomial fit.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

import statspai as sp


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(7)
    G, per = 120, 10
    cl = np.repeat(np.arange(G), per)
    xc = rng.uniform(-1, 1, G)
    x = np.clip(xc[cl] + rng.normal(0, 0.1, G * per), -1, 1)
    y = (
        0.4 * x
        + 0.8 * (x >= 0)
        + rng.normal(0, 0.5, G)[cl]
        + rng.normal(0, 0.4, G * per)
    )
    return pd.DataFrame({"y": y, "x": x, "cl": cl, "z": rng.normal(size=G)[cl]})


def _quiet(f, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


def test_bwsensitivity_rows_are_clustered_rdrobust(df):
    tab = _quiet(sp.rdbwsensitivity, df, y="y", x="x", bw_grid=[0.3, 0.5], cluster="cl")
    for _, row in tab.iterrows():
        r = _quiet(sp.rdrobust, df, y="y", x="x", h=row["bandwidth"], cluster="cl")
        assert row["estimate"] == pytest.approx(r.estimate, rel=1e-12)
        assert row["se"] == pytest.approx(r.se, rel=1e-12)
    un = _quiet(sp.rdbwsensitivity, df, y="y", x="x", bw_grid=[0.3], ax=None)
    assert not np.isclose(un["se"].iloc[0], tab["se"].iloc[0])


def test_placebo_rows_are_clustered_rdrobust(df):
    tab = _quiet(
        sp.rdplacebo, df, y="y", x="x", placebo_cutoffs=[-0.5, 0.5], cluster="cl"
    )
    for _, row in tab.iterrows():
        c0 = row["cutoff"]
        sub = df if c0 == 0 else (df[df.x < 0] if c0 < 0 else df[df.x >= 0])
        r = _quiet(sp.rdrobust, sub, y="y", x="x", c=c0, cluster="cl")
        assert row["estimate"] == pytest.approx(r.estimate, rel=1e-12)
        assert row["se"] == pytest.approx(r.se, rel=1e-12)


def test_balance_and_summary_pass_cluster(df):
    bal = _quiet(sp.rdbalance, df, x="x", covs=["z"], h=0.4, cluster="cl")
    r = _quiet(sp.rdrobust, df, y="z", x="x", h=0.4, cluster="cl")
    assert bal["se"].iloc[0] == pytest.approx(r.se, rel=1e-12)
    out = _quiet(sp.rdsummary, df, y="y", x="x", cluster="cl", verbose=False)
    ref = _quiet(sp.rdrobust, df, y="y", x="x", cluster="cl")
    assert out["estimate"].se == pytest.approx(ref.se, rel=1e-12)


def test_rdplot_bin_intervals_are_cluster_robust(df):
    plt.switch_backend("Agg")
    fig, _ = _quiet(sp.rdplot, df, y="y", x="x", cluster="cl", nbins=(8, 8))
    vb = fig.rdplot_data["vars_bins"]
    lo, hi = vb["rdplot_min_bin"], vb["rdplot_max_bin"]
    checked = 0
    for j in range(len(lo)):
        right = lo[j] >= 0
        side = df[df.x >= 0] if right else df[df.x < 0]
        last = j == len(lo) - 1 or (not right and j == int(np.sum(lo < 0)) - 1)
        m = (side.x >= lo[j]) & ((side.x <= hi[j]) if last else (side.x < hi[j]))
        if right and j == int(np.sum(lo < 0)):
            m = (side.x >= lo[j]) & (side.x < hi[j])
        b = side[m]
        if b.cl.nunique() < 2 or len(b) != int(vb["rdplot_N"][j]):
            continue
        ols = sm.OLS(b.y.to_numpy(), np.ones(len(b))).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(b.cl)[0]}
        )
        G = b.cl.nunique()
        q = stats.t.ppf(0.975, G - 1)
        assert vb["rdplot_se_y"][j] == pytest.approx(float(ols.bse[0]), rel=1e-10)
        assert vb["rdplot_ci_r"][j] == pytest.approx(
            b.y.mean() + q * ols.bse[0], rel=1e-10
        )
        checked += 1
    assert checked >= 12
    plt.close(fig)


def test_rdplot_band_is_cr1():
    from statspai.rd.rdrobust import _weighted_poly_fit_ci

    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 300)
    cl = rng.integers(0, 30, 300)
    y = x + rng.normal(0, 0.3, 30)[cl] + rng.normal(0, 0.2, 300)
    w = 1 - x
    grid = np.array([0.0, 0.5])
    fit, lo, hi = _weighted_poly_fit_ci(x, y, 2, grid, 0.95, w, cl)
    X = np.column_stack([x**2, x, np.ones_like(x)])
    ref = sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": cl})
    G = np.column_stack([grid**2, grid, np.ones_like(grid)])
    se = np.sqrt(np.einsum("ij,jk,ik->i", G, ref.cov_params(), G))
    z = stats.norm.ppf(0.975)
    np.testing.assert_allclose(hi - fit, z * se, rtol=1e-10)
