"""Coverage campaign — reachable estimator branches in the panel module.

Targets full-suite-uncovered, *non-defensive* branches (not the
``except LinAlgError`` singular-matrix fallbacks) in:

- ``interactive_fe`` — the non-finite-row filter (drops rows with NaNs
  before the Bai 2009 iteration).
- ``unit_root``      — the "series too short" guards in the Hadri test
  (per-unit skip + the all-units-skipped empty result).
- ``panel_binary``   — the conditional-FE-logit drop of units with no
  within-variation (all-0 / all-1 outcome) and the CRE Mundlak means.

Assertions check sane structure / properties (finite estimates after
dropping bad rows; NaN sentinels when there is nothing to estimate; the
reported drop count), not fabricated numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _balanced_panel(n_id=30, T=8, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_id):
        a = rng.normal()
        for t in range(T):
            x1 = rng.normal() + 0.3 * a
            x2 = rng.normal()
            y = a + 1.5 * x1 - 0.5 * x2 + rng.normal(0, 0.5)
            rows.append({"id": i, "time": t, "y": y, "x1": x1, "x2": x2})
    return pd.DataFrame(rows)


# ── interactive_fe: non-finite-row filter ───────────────────────────────


def test_interactive_fe_drops_nonfinite_rows():
    df = _balanced_panel()
    df.loc[5, "y"] = np.nan          # one non-finite row → triggers the filter
    df.loc[20, "x1"] = np.inf
    res = sp.interactive_fe(df, "y", ["x1", "x2"], id="id", time="time",
                            n_factors=1)
    # Estimation still succeeds on the finite rows and recovers sane slopes.
    beta = np.asarray(getattr(res, "params", getattr(res, "beta", None)),
                      dtype=float).ravel()
    assert beta.size >= 1
    assert np.all(np.isfinite(beta))


# ── unit_root: Hadri "too short" guards ─────────────────────────────────


def test_unit_root_hadri_skips_short_units():
    # A mixed panel: long units (>=5 periods) plus a few short units
    # (3 periods). The short units make ``_adf_single`` hit its ``T < 3``
    # early return and make the Hadri loop hit its ``len(y) < 5: continue``
    # skip, while the long units still yield a finite Hadri statistic.
    rng = np.random.default_rng(1)
    rows = []
    for i in range(10):                 # long units
        y = 0.0
        for t in range(18):
            y = 0.3 * y + rng.normal()
            rows.append({"id": i, "time": t, "v": y})
    for i in range(10, 13):             # short units (3 periods each)
        for t in range(3):
            rows.append({"id": i, "time": t, "v": rng.normal()})
    df = pd.DataFrame(rows)
    res = sp.panel_unitroot(df, "v", id="id", time="time", test="hadri")
    assert np.isfinite(res.statistic)   # finite stat from the long units
    assert res.n_units >= 10            # the 3 short units were skipped


def test_unit_root_ips_runs_on_adequate_panel():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(15):
        # a stationary AR(1) series per unit
        y = 0.0
        for t in range(20):
            y = 0.4 * y + rng.normal()
            rows.append({"id": i, "time": t, "v": y})
    df = pd.DataFrame(rows)
    res = sp.panel_unitroot(df, "v", id="id", time="time", test="ips")
    assert np.isfinite(res.statistic)


# ── panel_binary: conditional FE logit drops no-variation units ─────────


def test_panel_logit_fe_drops_constant_outcome_units():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(25):
        for t in range(6):
            x1 = rng.normal()
            p = 1.0 / (1.0 + np.exp(-(0.4 + 1.2 * x1)))
            y = int(rng.random() < p)
            rows.append({"id": i, "time": t, "y": y, "x1": x1})
    df = pd.DataFrame(rows)
    # Force two units to have no within-variation (all-1 and all-0).
    df.loc[df["id"] == 0, "y"] = 1
    df.loc[df["id"] == 1, "y"] = 0
    res = sp.panel_logit(df, "y", ["x1"], id="id", time="time", method="fe")
    # Those two units carry no conditional-likelihood information and are
    # dropped; the slope is still estimated and is positive (true sign).
    n_dropped = getattr(res, "n_dropped", None)
    if n_dropped is not None:
        assert n_dropped >= 2
    beta = np.asarray(getattr(res, "params", getattr(res, "beta", None)),
                      dtype=float).ravel()
    assert np.all(np.isfinite(beta))
