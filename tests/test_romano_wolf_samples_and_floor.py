"""``sp.romano_wolf``: per-outcome samples, no p-value floor, studentised
recentring.

Three problems found by the top-5 replications:

* **Listwise deletion across outcomes** (Kinship, QJE 2019): rows missing
  *any* outcome were dropped from every regression, shrinking six country
  samples of 66-79 to 15. Each outcome now uses its own non-missing rows, as
  Stata ``rwolf``'s per-outcome regressions do, and ``table['n_obs']``
  reports them.
* **p-value floor** (UCT, QJE 2016): a bootstrap sample in which a control
  dummy is all zero made the regression singular; the draw was scored as
  ``t* = 0`` and so exceeded every observed ``|t|`` -- ``p_rw = 0.027`` at
  ``|t| = 10`` with 119 village dummies. Aliased controls are now omitted in
  each draw (Stata ``regress``) and a draw is discarded only when the
  treatment itself is unidentified.
* **Recentring**: the bootstrap statistic is ``|b* - b| / se*`` (Romano &
  Wolf 2005; Clarke, Romano & Wolf 2020), not ``|b*/se* - b/se|``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import statspai as sp
from statspai.mht.romano_wolf import _build_design, _ols_fit, _resample_indices


def _panel(seed=0, n=400):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"x": rng.normal(size=n), "c": rng.normal(size=n)})
    for k, b in enumerate([0.5, 0.2, 0.0]):
        d[f"y{k}"] = b * d.x + 0.3 * d.c + rng.normal(size=n)
    d.loc[rng.choice(n, 120, replace=False), "y0"] = np.nan
    d.loc[rng.choice(n, 150, replace=False), "y1"] = np.nan
    return d


def test_each_outcome_uses_its_own_sample():
    d = _panel()
    r = sp.romano_wolf(
        d, y=["y0", "y1", "y2"], x="x", controls=["c"], n_boot=50, seed=1
    )
    tab = r.table.set_index("outcome")
    for o in ("y0", "y1", "y2"):
        sub = d.dropna(subset=[o])
        fit = sm.OLS(sub[o], sm.add_constant(sub[["x", "c"]])).fit(cov_type="HC1")
        assert tab.loc[o, "n_obs"] == len(sub)
        assert tab.loc[o, "coef"] == pytest.approx(fit.params["x"], rel=1e-10)
        assert tab.loc[o, "se"] == pytest.approx(fit.bse["x"], rel=1e-10)


def test_no_p_value_floor_with_sparse_dummies():
    rng = np.random.default_rng(3)
    n, G = 600, 60
    g = rng.integers(0, G, n)
    d = pd.DataFrame({"x": rng.normal(size=n), "cl": g})
    dummies = pd.get_dummies(g, prefix="g", drop_first=True, dtype=float)
    d = pd.concat([d, dummies], axis=1)
    d["y0"] = 1.0 * d.x + rng.normal(size=n)  # |t| ~ 25
    d["y1"] = 0.0 * d.x + rng.normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.romano_wolf(
            d,
            y=["y0", "y1"],
            x="x",
            controls=list(dummies.columns),
            cluster="cl",
            n_boot=300,
            seed=1,
        )
    p = r.table.set_index("outcome")["p_rw"]
    assert p["y0"] == 0.0  # used to sit at the share of singular draws
    assert p["y1"] > 0.05


def test_bootstrap_statistic_is_studentised_recentred():
    d = _panel(seed=5).dropna()
    ys = ["y0", "y1"]
    r = sp.romano_wolf(d, y=ys, x="x", controls=["c"], n_boot=200, seed=7)
    # Rebuild the draws with the same generator and the textbook statistic.
    rng = np.random.default_rng(7)
    b0 = np.array([_ols_fit(*_build_design(d, o, ["x"], ["c"]))[0] for o in ys])
    t0 = np.array([_ols_fit(*_build_design(d, o, ["x"], ["c"]))[2] for o in ys])
    boot = np.empty((200, 2))
    for b in range(200):
        db = d.iloc[_resample_indices(len(d), None, rng)].reset_index(drop=True)
        for s, o in enumerate(ys):
            c, se, _, _ = _ols_fit(*_build_design(db, o, ["x"], ["c"]))
            boot[b, s] = abs(c - b0[s]) / se
    order = np.argsort(-np.abs(t0))
    want = np.empty(2)
    prev = 0.0
    active = list(range(2))
    for h in order:
        pv = max(prev, float(np.mean(boot[:, active].max(axis=1) >= abs(t0[h]))))
        want[h] = prev = pv
        active.remove(h)
    np.testing.assert_allclose(r.table["p_rw"].to_numpy(), want, rtol=0, atol=0)
