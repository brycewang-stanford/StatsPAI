"""``sp.westfall_young``: permutation stepdown maxT across outcomes.

The MHT module advertised Westfall-Young but shipped only the bootstrap
Romano-Wolf; a stratified experiment (UCT, QJE 2016) needed a stepdown built
on its own re-randomizations. No Stata ``wyoung`` / R ``multtest`` is part of
the reference toolchain here, so the evidence is analytical:

* on a design small enough to enumerate, the adjusted p-values equal a
  from-scratch implementation of Westfall & Young's (1993) stepdown maxT
  (statsmodels HC1 t-statistics, explicit loops);
* a single outcome's permutation p-value equals ``sp.ri_test(stat='ols_t')``;
* under the global null the family-wise error rate stays at the level.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import statspai as sp


def _small():
    rng = np.random.default_rng(11)
    rows = []
    for s in range(2):
        z = rng.permutation([1, 1, 1, 0, 0, 0])
        for u in range(6):
            x = rng.normal()
            rows.append(
                dict(
                    block=s,
                    Z=z[u],
                    x=x,
                    y1=0.9 * z[u] + x + rng.normal(),
                    y2=0.2 * z[u] + rng.normal(),
                    y3=rng.normal(),
                )
            )
    return pd.DataFrame(rows)


def _t(y, d, x, block):
    X = np.column_stack([np.ones(len(y)), d, x, (block == 1).astype(float)])
    return sm.OLS(y, X).fit(cov_type="HC1").tvalues[1]


def test_equals_textbook_stepdown_on_enumerated_design():
    d = _small()
    ys = ["y1", "y2", "y3"]
    out = sp.westfall_young(
        d, y=ys, treat="Z", covariates=["x"], strata="block", n_perms=10_000
    )
    assert out.attrs["exact"] and out.attrs["n_perms"] == 400  # C(6,3)^2
    blk = d["block"].to_numpy()
    t_obs = np.array(
        [_t(d[o].to_numpy(), d.Z.to_numpy(), d.x.to_numpy(), blk) for o in ys]
    )
    idx = [np.flatnonzero(blk == s) for s in (0, 1)]
    T = []
    for c0 in itertools.combinations(idx[0], 3):
        for c1 in itertools.combinations(idx[1], 3):
            z = np.zeros(len(d))
            z[list(c0) + list(c1)] = 1
            T.append([_t(d[o].to_numpy(), z, d.x.to_numpy(), blk) for o in ys])
    T = np.abs(np.array(T))
    a = np.abs(t_obs)
    order = list(np.argsort(-a))
    want = {}
    prev = 0.0
    for step, h in enumerate(order):
        rest = order[step:]
        p = np.mean(T[:, rest].max(axis=1) >= a[h] - 1e-12)
        prev = max(prev, p)
        want[h] = prev
    np.testing.assert_allclose(out["t"], t_obs, rtol=1e-10)
    np.testing.assert_allclose(out["p_wy"], [want[j] for j in range(3)], atol=1e-12)


def test_single_outcome_matches_ri_test():
    d = _small()
    wy = sp.westfall_young(d, y=["y1"], treat="Z", covariates=["x"], strata="block")
    ri = sp.ri_test(
        d,
        y="y1",
        treat="Z",
        strata="block",
        stat="ols_t",
        covariates=["x"],
        n_perms=10_000,
    )
    assert wy["p_perm"].iloc[0] == pytest.approx(ri["p_value"], abs=1e-12)
    assert wy["p_wy"].iloc[0] == pytest.approx(ri["p_value"], abs=1e-12)


def test_fwer_under_the_global_null():
    rng = np.random.default_rng(5)
    rejections = 0
    n_sim = 400
    for _ in range(n_sim):
        n = 60
        common = rng.normal(size=n)
        d = pd.DataFrame({"Z": rng.permutation([0, 1] * (n // 2))})
        for k in range(4):
            d[f"y{k}"] = 0.7 * common + rng.normal(size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = sp.westfall_young(
                d,
                y=[f"y{k}" for k in range(4)],
                treat="Z",
                n_perms=199,
                seed=int(rng.integers(1e9)),
            )
        rejections += bool((out["p_wy"] <= 0.05).any())
    assert rejections / n_sim < 0.08  # nominal 0.05; MC SE ~0.011


def test_each_outcome_uses_its_own_rows():
    d = _small()
    d.loc[[0, 5], "y2"] = np.nan
    out = sp.westfall_young(d, y=["y1", "y2"], treat="Z", strata="block").set_index(
        "outcome"
    )
    assert out.loc["y1", "n_obs"] == 12 and out.loc["y2", "n_obs"] == 10
