"""The weak-instrument-robust confidence set attached to ``sp.dml(model='pliv')``.

``model_info['anderson_rubin']`` inverts the test ``C(theta) = n *
mean(m)^2 / var(m)``, ``m = (ry - theta*rd) * rz``, on the cross-fitted
residuals. The weak-instrument chapter of *Applied Causal Inference
Powered by ML and AI* does this by scanning a grid, which truncates the
set at the grid's ends (its AJR example reports an upper end of 1.99 on a
grid that stops at 2; the set extends to 3.67). Here the set is the exact
solution of the quadratic inequality.

Evidence, all on simulated data:

* the exact set agrees with a fine grid scan (tolerance: the grid step);
* its ends solve ``C(theta) = q`` to 1e-9;
* under a weak instrument the set covers the truth at the nominal rate
  while the Wald interval does not (400 replications; binomial 3-sd band
  around 0.95 is +/- 0.033).

References
----------
[@anderson1949estimation], [@chernozhukov2018double]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression

import statspai as sp


def _simulate(n: int, strength: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    u = rng.normal(size=n)
    z = 0.5 * x[:, 0] + rng.normal(size=n)
    d = strength * z + 0.5 * x[:, 1] + u + 0.5 * rng.normal(size=n)
    y = 1.0 * d + x[:, 0] + 2.0 * u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "z": z, "x1": x[:, 0], "x2": x[:, 1]})


def _fit(df: pd.DataFrame):
    lin = LinearRegression()
    return sp.dml(
        df,
        y="y",
        treat="d",
        instrument="z",
        covariates=["x1", "x2"],
        model="pliv",
        ml_g=lin,
        ml_m=lin,
        ml_r=lin,
        n_folds=2,
        fold_indices=np.arange(len(df)) % 2,
    )


def _residuals(df: pd.DataFrame):
    folds = np.arange(len(df)) % 2
    X = df[["x1", "x2"]].to_numpy()
    out = []
    for col in ("y", "d", "z"):
        v = df[col].to_numpy()
        r = np.zeros(len(df))
        for f in (0, 1):
            tr, te = folds != f, folds == f
            r[te] = v[te] - LinearRegression().fit(X[tr], v[tr]).predict(X[te])
        out.append(r)
    return out


def _statistic(theta, ry, rd, rz):
    m = (ry - theta * rd) * rz
    return len(m) * np.mean(m) ** 2 / np.var(m)


def test_exact_set_agrees_with_a_grid_scan_and_solves_the_equation():
    df = _simulate(500, 0.5, seed=1)
    ar = _fit(df).model_info["anderson_rubin"]
    assert ar["kind"] == "interval"
    (lo, hi), q = ar["intervals"][0], ar["critical_value"]
    ry, rd, rz = _residuals(df)
    assert _statistic(lo, ry, rd, rz) == pytest.approx(q, rel=1e-9)
    assert _statistic(hi, ry, rd, rz) == pytest.approx(q, rel=1e-9)
    grid = np.arange(lo - 1.0, hi + 1.0, 1e-3)
    inside = grid[[_statistic(t, ry, rd, rz) <= q for t in grid]]
    assert inside.min() == pytest.approx(lo, abs=1e-3)
    assert inside.max() == pytest.approx(hi, abs=1e-3)
    assert ar["statistic_at_zero"] == pytest.approx(_statistic(0.0, ry, rd, rz))
    assert ar["p_value_at_zero"] == pytest.approx(
        stats.chi2.sf(ar["statistic_at_zero"], 1)
    )


def test_irrelevant_instrument_gives_an_unbounded_set():
    """No first stage: the data cannot bound theta, and the set says so."""
    kinds = set()
    for seed in range(20):
        ar = _fit(_simulate(300, 0.0, seed=100 + seed)).model_info["anderson_rubin"]
        kinds.add(ar["kind"])
        if ar["kind"] != "interval":
            assert ar["first_stage_statistic"] < ar["critical_value"]
            assert np.isinf(ar["intervals"][0][0])
    assert kinds & {"unbounded", "disjoint"}


def test_bounded_exactly_when_the_first_stage_is_significant():
    for seed in range(30):
        ar = _fit(_simulate(300, 0.12, seed=200 + seed)).model_info["anderson_rubin"]
        significant = ar["first_stage_statistic"] > ar["critical_value"]
        assert (ar["kind"] in ("interval", "empty")) == significant


def test_coverage_under_a_weak_instrument():
    truth = 1.0
    reps = ar_hits = wald_hits = 0
    for r in range(400):
        try:
            fit = _fit(_simulate(300, 0.12, seed=1000 + r))
        except RuntimeError:
            # sp.dml refuses a first stage with |partial corr| < 1e-3; the
            # ratio is not computed, so there is no interval to score.
            continue
        reps += 1
        ar = fit.model_info["anderson_rubin"]
        ar_hits += any(lo <= truth <= hi for lo, hi in ar["intervals"])
        wald_hits += fit.ci[0] <= truth <= fit.ci[1]
    assert reps >= 390
    assert abs(ar_hits / reps - 0.95) <= 0.033
    # The Wald interval is not centred on the truth with a weak instrument
    # and endogeneity this strong; it over- or under-covers by more.
    assert abs(wald_hits / reps - 0.95) > abs(ar_hits / reps - 0.95)


def test_not_attached_to_weighted_or_clustered_fits():
    df = _simulate(300, 0.5, seed=5).assign(g=np.arange(300) // 3, w=1.0)
    lin = LinearRegression()
    kw = dict(y="y", treat="d", instrument="z", covariates=["x1", "x2"],
              model="pliv", ml_g=lin, ml_m=lin, ml_r=lin, n_folds=2)  # fmt: skip
    assert "anderson_rubin" not in sp.dml(df, cluster="g", **kw).model_info
    assert "anderson_rubin" not in sp.dml(df, sample_weight="w", **kw).model_info
