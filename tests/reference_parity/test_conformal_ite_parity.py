"""Analytical parity: sp.conformal_ite marginal-coverage guarantee (MC).

Conformal ITE intervals (Lei & Candes 2021, "Conformal inference of
counterfactuals and individual treatment effects") carry a *distribution-free
finite-sample* marginal-coverage guarantee: the interval covers the individual
treatment effect with probability at least ``1 - alpha``. This is the honest
evidence tier for a conformal method — a probabilistic lower bound, not a
cross-package bit-for-bit number.

We verify it by Monte-Carlo on a fully-known DGP where *both* potential
outcomes are simulated (so the realized ITE Y(1) - Y(0) is observable) and
check empirical coverage stays at or above the nominal level. We also assert
the exact structural identities of the returned interval object.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

ALPHA = 0.1
N = 1500
X_COLS = ["x0", "x1", "x2"]


def _simulate(seed):
    """Draw both potential outcomes so the true ITE is known."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, 3))
    base = X[:, 0] + 0.3 * X[:, 1]
    tau = 1.0 + 0.5 * X[:, 0]
    y0 = base + rng.normal(0, 0.5, N)
    y1 = base + tau + rng.normal(0, 0.5, N)
    d = rng.integers(0, 2, N)
    y = np.where(d == 1, y1, y0)
    ite = y1 - y0
    df = pd.DataFrame({"y": y, "d": d, "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2]})
    return df, ite


def _fit(df):
    return sp.conformal_ite(df, y="y", d="d", X=X_COLS, alpha=ALPHA)


def test_marginal_coverage_at_least_nominal():
    """Empirical ITE coverage >= 1 - alpha across independent replications."""
    covs = []
    for seed in range(10):
        df, ite = _simulate(seed)
        r = _fit(df)
        lo = np.asarray(r.diagnostics["cate_lower"])
        hi = np.asarray(r.diagnostics["cate_upper"])
        covs.append(float(np.mean((ite >= lo) & (ite <= hi))))
    # Conformal guarantee is a lower bound; allow small MC slack per replicate.
    assert min(covs) >= (1 - ALPHA) - 0.05
    assert float(np.mean(covs)) >= (1 - ALPHA)


def test_interval_structural_identities():
    df, _ = _simulate(0)
    r = _fit(df)
    lo = np.asarray(r.diagnostics["cate_lower"])
    hi = np.asarray(r.diagnostics["cate_upper"])
    cate = np.asarray(r.diagnostics["cate"])
    # Valid, ordered intervals that bracket the point CATE everywhere.
    assert np.all(lo <= hi)
    assert np.all((lo <= cate) & (cate <= hi))
    # Config + summary identities.
    assert float(r.diagnostics["coverage_level"]) == pytest.approx(1 - ALPHA)
    assert float(r.diagnostics["cate_mean"]) == pytest.approx(
        float(cate.mean()), abs=1e-12
    )
    # Since 1.41 the headline is the cross-fitted AIPW average (equal to
    # sp.metalearner's, tests/test_conformal_causal_scan_oct2026.py), not the
    # mean of the fitted CATEs, which stays in diagnostics["cate_mean"].
    assert np.isfinite(float(r.estimate)) and float(r.se) > 0
