"""Analytical parity: Borusyak-Hull-Jaravel shock-level aggregation.

The BHJ (2022) equivalence states that a location-level shift-share (Bartik) IV
estimate equals an exposure-weighted shock-level estimate. ``sp.ssaggregate``
computes the shock-level object, so its coefficient must reproduce the
location-level just-identified Bartik IV ``cov(B, y) / cov(B, x)`` (with
``B = shares @ shocks``) numerically. With a valid instrument the estimate is
also consistent for the structural coefficient as ``n`` grows. Analytical
evidence tier (numerical identity + known-truth recovery on a deterministic
DGP).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def _shift_share_dgp(n, K, beta=1.5, seed=0):
    rng = np.random.default_rng(seed)
    shares = rng.dirichlet(np.ones(K), size=n)  # exposure shares, rows sum to 1
    shocks = rng.normal(0, 1, K)                 # industry shocks
    bartik = shares @ shocks                      # shift-share instrument
    u = rng.normal(0, 1, n)                        # confounder
    x = 0.8 * bartik + 0.5 * u + rng.normal(0, 0.3, n)  # endogenous regressor
    y = beta * x + u
    df = pd.DataFrame({"y": y, "x": x})
    return df, shares, shocks, bartik


def test_ssaggregate_matches_location_level_bartik_iv():
    df, shares, shocks, bartik = _shift_share_dgp(n=2000, K=20)
    res = sp.ssaggregate(df, y="y", x="x", shares=shares, shocks=shocks)
    # Location-level just-identified Bartik IV (single instrument, with const):
    loc = np.cov(bartik, df["y"])[0, 1] / np.cov(bartik, df["x"])[0, 1]
    assert float(res.params["x"]) == pytest.approx(loc, rel=1e-6)


def test_ssaggregate_consistent_for_structural_coefficient():
    df, shares, shocks, _ = _shift_share_dgp(n=15000, K=20, beta=1.5)
    res = sp.ssaggregate(df, y="y", x="x", shares=shares, shocks=shocks)
    assert float(res.params["x"]) == pytest.approx(1.5, abs=0.2)


def test_ssaggregate_standard_error_positive_and_finite():
    df, shares, shocks, _ = _shift_share_dgp(n=2000, K=20)
    res = sp.ssaggregate(df, y="y", x="x", shares=shares, shocks=shocks)
    se = float(res.std_errors["x"])
    assert np.isfinite(se) and se > 0
