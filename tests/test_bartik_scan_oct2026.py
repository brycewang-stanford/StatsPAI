"""Option scan of ``statspai.bartik`` (2026-10-27).

Each test states the correct behaviour; every one failed on 1.40.2 and
pins the fix (see the CHANGELOG entries of the same date).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import StatsPAIError


def _ss_data(n: int = 80, K: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    regions = [f"r{i:03d}" for i in range(n)]
    inds = [f"ind{k}" for k in range(K)]
    shares = pd.DataFrame(
        rng.dirichlet(np.ones(K), size=n), index=regions, columns=inds
    )
    shocks = pd.Series(rng.normal(size=K), index=inds)
    B = shares.to_numpy() @ shocks.to_numpy()
    d = 0.8 * B + rng.normal(scale=0.5, size=n)
    y = 1.5 * d + rng.normal(size=n)
    data = pd.DataFrame({"y": y, "d": d}, index=regions)
    return data, shares, shocks


# ---------------------------------------------------------------------- #
#  S5  sp.bartik matches shares rows to data rows by position only
# ---------------------------------------------------------------------- #


def test_bartik_aligns_shares_by_index_label():
    data, shares, shocks = _ss_data()
    ref = sp.bartik(data, y="y", endog="d", shares=shares, shocks=shocks)
    perm = np.random.default_rng(1).permutation(len(data))
    shuffled = data.iloc[perm]
    try:
        res = sp.bartik(shuffled, y="y", endog="d", shares=shares, shocks=shocks)
    except StatsPAIError:
        return  # refusing loudly is also correct
    assert res.params["d"] == pytest.approx(ref.params["d"], rel=1e-12)


# ---------------------------------------------------------------------- #
#  S6  sp.ssaggregate matches shocks to share columns by position only
# ---------------------------------------------------------------------- #


def test_ssaggregate_aligns_shock_series_by_label():
    data, shares, shocks = _ss_data()
    data = data.reset_index(drop=True)
    shares = shares.reset_index(drop=True)
    ref = sp.ssaggregate(data, y="y", x="d", shares=shares, shocks=shocks)
    rev = shocks.iloc[::-1]
    try:
        res = sp.ssaggregate(data, y="y", x="d", shares=shares, shocks=rev)
    except StatsPAIError:
        return
    assert res.params["d"] == pytest.approx(ref.params["d"], rel=1e-12)
    assert res.std_errors["d"] == pytest.approx(ref.std_errors["d"], rel=1e-10)


# ---------------------------------------------------------------------- #
#  S7  shift_share_political long differences skip NaN endpoints
# ---------------------------------------------------------------------- #


def test_political_long_difference_does_not_shift_window():
    rng = np.random.default_rng(0)
    n, K = 60, 5
    units = list(range(n))
    inds = [f"k{k}" for k in range(K)]
    shares = pd.DataFrame(rng.dirichlet(np.ones(K), size=n), index=units, columns=inds)
    shocks = pd.Series(rng.normal(size=K), index=inds)
    B = shares.to_numpy() @ shocks.to_numpy()
    rows = []
    for i in units:
        x0 = rng.normal()
        dx = 0.8 * B[i] + rng.normal(scale=0.3)
        for t in (0, 1, 2):
            x = x0 + dx * t / 2
            rows.append(dict(u=i, t=t, x=x, yv=2.0 * x + rng.normal(scale=0.2)))
    df = pd.DataFrame(rows)
    df.loc[(df.u == 0) & (df.t == 0), "yv"] = np.nan
    kw = dict(unit="u", time="t", outcome="yv", endog="x", shares=shares, shocks=shocks)
    try:
        res = sp.shift_share_political(df, **kw)
    except StatsPAIError:
        return  # refusing loudly is correct
    ref = sp.shift_share_political(df[df.u != 0], **kw)
    assert res.estimate == pytest.approx(ref.estimate, rel=1e-12)


# ---------------------------------------------------------------------- #
#  V2  missing values flow into the 2SLS as NaN
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("fn", ["bartik", "ssaggregate"])
def test_missing_outcome_is_not_silent(fn):
    data, shares, shocks = _ss_data()
    data = data.reset_index(drop=True)
    shares = shares.reset_index(drop=True)
    data.loc[3, "y"] = np.nan
    try:
        if fn == "bartik":
            res = sp.bartik(data, y="y", endog="d", shares=shares, shocks=shocks)
            est = res.params["d"]
        else:
            res = sp.ssaggregate(data, y="y", x="d", shares=shares, shocks=shocks)
            est = res.params["d"]
    except StatsPAIError:
        return
    assert np.isfinite(est)
