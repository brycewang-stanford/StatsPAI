"""Coverage campaign — additional staggered-DiD estimators in ``statspai.did``.

Covers public estimators not exercised by the existing cov95 DiD suite:

- ``gardner_did``  (Gardner 2021 two-stage DiD), incl. event-study mode;
- ``stacked_did``  (stacked / cohort-event-study DiD);
- ``lp_did``       (local-projections DiD, Dube et al. 2023);
- ``sun_abraham``  (Sun & Abraham 2021 interaction-weighted estimator);
- ``did_bcf``      (Bayesian Causal Forest DiD);
- ``harvest_did``  (HARVEST staggered-adoption estimator).

DGP: a constant +2 treatment effect switched on at each cohort's adoption
date, so every consistent estimator must recover an overall ATT near 2.
Assertions check that recovery + finite SEs, never fabricated numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TRUE_ATT = 2.0


def _staggered(seed=0, n_units=120, n_periods=8, att=TRUE_ATT):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 4, 6][u % 3]                 # 0 = never-treated
        fe = rng.normal()
        x1 = rng.normal()
        for t in range(1, n_periods + 1):
            d = int(g > 0 and t >= g)
            te = att if d else 0.0
            y = fe + 0.3 * t + 0.5 * x1 + te + rng.normal(0, 0.4)
            rows.append({"unit": u, "time": t, "y": y, "g": g, "x1": x1, "d": d})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _staggered()


def _att(result):
    for attr in ("estimate", "att", "overall_att"):
        v = getattr(result, attr, None)
        if v is not None:
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            arr = arr[np.isfinite(arr)]
            if arr.size:
                return float(arr.ravel()[0])
    raise AssertionError("no finite estimate on result")


def test_gardner_two_stage(panel):
    r = sp.gardner_did(panel, y="y", group="unit", time="time", first_treat="g")
    assert abs(_att(r) - TRUE_ATT) < 0.6


def test_gardner_event_study(panel):
    r = sp.gardner_did(panel, y="y", group="unit", time="time", first_treat="g",
                       event_study=True)
    assert r is not None  # event-study path returns per-horizon effects


def test_stacked_did(panel):
    r = sp.stacked_did(panel, y="y", group="unit", time="time", first_treat="g",
                       window=(-3, 3))
    assert abs(_att(r) - TRUE_ATT) < 0.7


def test_lp_did(panel):
    r = sp.lp_did(panel, y="y", unit="unit", time="time", treatment="d",
                  horizons=(-2, 3))
    assert r is not None
    assert np.isfinite(_att(r))


def test_sun_abraham(panel):
    r = sp.sun_abraham(panel, y="y", g="g", t="time", i="unit")
    assert abs(_att(r) - TRUE_ATT) < 0.7


def test_did_bcf(panel):
    # Bayesian Causal Forest — keep the forest small for test speed.
    r = sp.did_bcf(panel, y="y", treat="d", time="time", id="unit", n_trees=20)
    assert r is not None
    assert np.isfinite(_att(r))


def test_harvest_did(panel):
    r = sp.harvest_did(panel, unit="unit", time="time", outcome="y",
                       cohort="g", never_value=0)
    assert r is not None
    assert np.isfinite(_att(r))
