"""``sp.cgs_continuous_did(cluster=)`` and the overall ACRT standard error.

The overall ACRT standard error treated units as independent (Web of Power,
QJE 2023: 0.015 against a clustered TWFE SE of 0.058). With ``cluster`` the
unit-level influence functions are summed within clusters. Checks: with no
cluster shocks the default SE matches the Monte Carlo spread (it was half of
it before the cell influence was put on the whole-cell scale); with the unit
as its own cluster the SE is unchanged; with cluster-level doses and shocks
the clustered SE tracks the Monte Carlo spread while the unclustered one
understates it several-fold.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _panel(seed, shock_sd=0.5, n_cl=30, per=12):
    """Even clusters dosed at a cluster-level dose from period 3."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cl):
        shock = rng.normal(scale=shock_sd, size=4)  # cluster x period shocks
        treated_cl = c % 2 == 0
        d_cl = rng.uniform(0.2, 1.0)
        for j in range(per):
            u = c * per + j
            g = 3 if treated_cl else 0
            d = d_cl if g else 0.0
            a = rng.normal()
            for t in range(1, 5):
                eff = 1.0 * d if (g and t >= g) else 0.0
                rows.append(
                    (u, c, t, g, d, a + shock[t - 1] + eff + rng.normal(scale=0.3))
                )
    return pd.DataFrame(rows, columns=["id", "cl", "t", "g", "dose", "y"])


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.cgs_continuous_did(
            df, y="y", dose="dose", time="t", unit="id", cohort="g", degree=1, **kw
        )


def test_unit_cluster_reproduces_default():
    df = _panel(0)
    a = _fit(df)
    b = _fit(df.assign(uid=df["id"]), cluster="uid")
    assert b.overall_acrt_se == pytest.approx(a.overall_acrt_se, rel=1e-12)


def _mc(shock_sd, reps=100):
    est, se_cl, se_un = [], [], []
    for s in range(reps):
        df = _panel(100 + s, shock_sd=shock_sd)
        r_cl = _fit(df, cluster="cl")
        est.append(r_cl.overall_acrt)
        se_cl.append(r_cl.overall_acrt_se)
        se_un.append(_fit(df).overall_acrt_se)
    return float(np.std(est, ddof=1)), float(np.mean(se_un)), float(np.mean(se_cl))


def test_default_se_tracks_the_monte_carlo_spread_without_shocks():
    sd, se_un, _ = _mc(0.0)
    assert 0.85 * sd < se_un < 1.15 * sd


def test_clustered_se_tracks_the_monte_carlo_spread():
    sd, se_un, se_cl = _mc(0.5)
    assert se_un < 0.4 * sd  # unclustered understates
    assert 0.7 * sd < se_cl < 1.3 * sd


def test_cluster_must_be_constant_within_unit():
    df = _panel(1)
    with pytest.raises(sp.MethodIncompatibility, match="varies within units"):
        _fit(df.assign(bad=df["t"]), cluster="bad")
