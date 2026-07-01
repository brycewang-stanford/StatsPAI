"""Analytical parity: sp.cluster_cross_interference DGP recovery.

A cluster-randomised trial with explicit interference adjustment regresses the
individual outcome on the cluster's own treatment (direct effect) and the share
of treated neighbours (spillover effect). On a known linear DGP

    y = direct * treat + spillover * neighbour_share + noise

the estimator recovers both structural coefficients. Analytical evidence tier
(known-truth recovery on a deterministic DGP; no cross-package reference).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

DIRECT = 1.0
SPILL = 0.8


def _simulate(seed, n_clusters=200):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        d = int(rng.integers(0, 2))
        nshare = float(rng.random())
        base = DIRECT * d + SPILL * nshare
        for _ in range(int(rng.integers(15, 30))):
            rows.append(
                {
                    "cluster": c,
                    "d": d,
                    "nshare": nshare,
                    "y": base + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


def _fit(df):
    return sp.cluster_cross_interference(
        df, y="y", cluster="cluster", treat="d", neighbour_treat_share="nshare"
    )


def test_recovers_direct_and_spillover_effects():
    df = _simulate(0)
    r = _fit(df)
    assert float(r.direct_effect) == pytest.approx(DIRECT, abs=0.1)
    assert float(r.spillover_effect) == pytest.approx(SPILL, abs=0.1)


def test_standard_errors_positive_and_finite():
    df = _simulate(1)
    r = _fit(df)
    assert float(r.direct_se) > 0 and np.isfinite(r.direct_se)
    assert float(r.spillover_se) > 0 and np.isfinite(r.spillover_se)
    assert int(r.n_clusters) == 200
