"""Analytical parity: sp.geolift recovers a known geo-experiment lift.

``sp.geolift`` aggregates the treated markets into one series and fits a
synthetic control from the untreated markets (Abadie-Diamond-Hainmueller
2010). When the treated market is an *exact convex combination* of two
donor markets in the pre-period, a perfect synthetic control exists — the
optimal donor weights reproduce the treated pre-path exactly, so the
estimated post-period ATT equals the injected additive lift to numerical
precision. This is an analytical-only record: known-DGP recovery, no
cross-package reference.

The panel is deterministic (fixed RNG seed) so the recovery is reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import statspai as sp

_T_PERIODS = 30
_TREAT_TIME = 20


def _panel(lift):
    """Treated market = 0.5*A + 0.5*B exactly in pre-period, + `lift` post."""
    rng = np.random.default_rng(0)
    t = np.arange(_T_PERIODS)
    controls = {
        "A": 10 + 0.5 * t + rng.normal(0, 0.2, _T_PERIODS),
        "B": 5 + 0.2 * t + rng.normal(0, 0.2, _T_PERIODS),
        "C": 8 - 0.1 * t + rng.normal(0, 0.2, _T_PERIODS),
        "D": 12 + 0.3 * t + rng.normal(0, 0.2, _T_PERIODS),
    }
    treated = 0.5 * controls["A"] + 0.5 * controls["B"]
    treated_obs = treated.copy()
    treated_obs[t >= _TREAT_TIME] += lift

    rows = []
    for g, series in controls.items():
        for i in range(_T_PERIODS):
            rows.append({"geo": g, "week": i, "sales": series[i]})
    for i in range(_T_PERIODS):
        rows.append({"geo": "NYC", "week": i, "sales": treated_obs[i]})
    return pd.DataFrame(rows)


def test_geolift_recovers_known_lift():
    lift = 4.0
    res = sp.geolift(
        _panel(lift),
        outcome="sales",
        geo="geo",
        time="week",
        treated_geos="NYC",
        treatment_time=_TREAT_TIME,
    )
    # Perfect donor combo exists ⇒ ATT recovers the injected lift exactly.
    assert abs(float(res.estimate) - lift) <= 1e-6


def test_geolift_zero_lift_gives_null_att():
    res = sp.geolift(
        _panel(0.0),
        outcome="sales",
        geo="geo",
        time="week",
        treated_geos="NYC",
        treatment_time=_TREAT_TIME,
    )
    # No intervention ⇒ synthetic control reproduces the treated path, ATT ~ 0.
    assert abs(float(res.estimate)) <= 1e-6
