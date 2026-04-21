"""End-to-end smoke + correctness tests for all 6 bridging theorems.

Each ``sp.bridge(kind=...)`` dispatcher runs two independent identification
paths and reports a doubly-robust combined estimate plus an agreement
test.  These tests ensure every kind is dispatchable, produces finite
numbers, and — on a DGP that satisfies both identifying assumptions —
the two paths agree within normal sampling error.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp


# ═══════════════════════════════════════════════════════════════════════
#  did_sc — Shi-Athey 2025
# ═══════════════════════════════════════════════════════════════════════

def _did_sc_panel(seed=0, units=11, T=15, treat_time=10, tau=2.0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(units):
        u_fe = rng.normal(0, 0.3)
        for t in range(T):
            y = u_fe + rng.normal(0, 0.1)
            if u == 0 and t >= treat_time:
                y += tau
            rows.append({"unit": u, "year": t, "y": y})
    return pd.DataFrame(rows)


def test_bridge_did_sc_recovers_truth():
    df = _did_sc_panel(seed=0)
    r = sp.bridge(kind="did_sc", data=df, y="y", unit="unit", time="year",
                   treated_unit=0, treatment_time=10)
    assert abs(r.estimate_dr - 2.0) < 0.5
    assert r.kind == "did_sc"
    assert r.path_a_name and r.path_b_name
    assert r.se_a >= 0 and r.se_b >= 0


# ═══════════════════════════════════════════════════════════════════════
#  cb_ipw — Zhao-Percival 2025
# ═══════════════════════════════════════════════════════════════════════

def _binary_treatment_data(n=400, tau=1.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-X[:, 0]))
    D = (rng.uniform(size=n) < p).astype(float)
    Y = tau * D + X[:, 0] + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": Y, "d": D,
                          **{f"x{j}": X[:, j] for j in range(3)}})


def test_bridge_cb_ipw_recovers_truth():
    df = _binary_treatment_data(seed=0, tau=1.5)
    r = sp.bridge(kind="cb_ipw", data=df, y="y", treat="d",
                   covariates=["x0", "x1", "x2"])
    # CB and IPW should both recover ~1.5
    assert abs(r.estimate_a - 1.5) < 0.4
    assert abs(r.estimate_b - 1.5) < 0.4
    # Under correct specification the two paths should agree
    assert r.diff_p > 0.05, f"paths disagree (p={r.diff_p}) on correctly-specified DGP"


def test_bridge_cb_ipw_dr_combination_finite():
    df = _binary_treatment_data(seed=1)
    r = sp.bridge(kind="cb_ipw", data=df, y="y", treat="d",
                   covariates=["x0", "x1", "x2"])
    assert np.isfinite(r.estimate_dr) and np.isfinite(r.se_dr)


# ═══════════════════════════════════════════════════════════════════════
#  dr_calib — Zhang et al. 2025
# ═══════════════════════════════════════════════════════════════════════

def test_bridge_dr_calib_agreement():
    df = _binary_treatment_data(seed=2, tau=1.2)
    r = sp.bridge(kind="dr_calib", data=df, y="y", treat="d",
                   covariates=["x0", "x1", "x2"])
    # Two calibration variants should agree very tightly
    assert abs(r.estimate_a - r.estimate_b) < 0.3


# ═══════════════════════════════════════════════════════════════════════
#  ewm_cate — Ferman et al. 2025
# ═══════════════════════════════════════════════════════════════════════

def test_bridge_ewm_cate_dispatches():
    df = _binary_treatment_data(seed=3)
    r = sp.bridge(kind="ewm_cate", data=df, y="y", treat="d",
                   covariates=["x0", "x1", "x2"])
    assert np.isfinite(r.estimate_a)
    assert np.isfinite(r.estimate_b)
    assert r.kind == "ewm_cate"


# ═══════════════════════════════════════════════════════════════════════
#  kink_rdd — Lu-Wang-Xie 2025
# ═══════════════════════════════════════════════════════════════════════

def test_bridge_kink_rdd_dispatches():
    rng = np.random.default_rng(0)
    n = 500
    R = rng.uniform(-2, 2, n)
    # Outcome with a kink at 0 (slope changes)
    Y = 0.5 * R + 0.3 * np.maximum(R, 0) ** 2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": Y, "r": R})
    r = sp.bridge(kind="kink_rdd", data=df, y="y", running="r", cutoff=0.0)
    assert np.isfinite(r.estimate_a)
    assert np.isfinite(r.estimate_b)


# ═══════════════════════════════════════════════════════════════════════
#  surrogate_pci — Kallus-Mao 2026
# ═══════════════════════════════════════════════════════════════════════

def test_bridge_surrogate_pci_dispatches():
    rng = np.random.default_rng(0)
    n = 400
    D = rng.integers(0, 2, n).astype(float)
    S = 0.5 * D + rng.normal(0, 0.3, n)
    Y_long = 2.0 * D + 0.5 * S + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y_long": Y_long, "s": S, "d": D,
                       "x": rng.normal(0, 1, n)})
    r = sp.bridge(kind="surrogate_pci", data=df, long_term="y_long",
                   short_term=["s"], treat="d", covariates=["x"])
    assert np.isfinite(r.estimate_a)
    assert np.isfinite(r.estimate_b)


# ═══════════════════════════════════════════════════════════════════════
#  Dispatcher hygiene
# ═══════════════════════════════════════════════════════════════════════

def test_bridge_rejects_unknown_kind():
    df = _binary_treatment_data(seed=4)
    with pytest.raises((KeyError, ValueError)):
        sp.bridge(kind="bogus_bridge", data=df)


def test_bridge_all_kinds_registered_and_exposed():
    expected = {"did_sc", "ewm_cate", "cb_ipw", "kink_rdd",
                "dr_calib", "surrogate_pci"}
    from statspai.bridge.core import _BRIDGES
    assert expected.issubset(_BRIDGES.keys())
    # bridge itself is top-level
    assert "bridge" in sp.list_functions()


def test_bridge_result_has_reference():
    df = _binary_treatment_data(seed=5)
    r = sp.bridge(kind="cb_ipw", data=df, y="y", treat="d",
                   covariates=["x0", "x1", "x2"])
    assert r.reference  # non-empty citation string
