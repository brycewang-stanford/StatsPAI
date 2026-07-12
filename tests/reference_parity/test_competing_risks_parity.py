"""Analytical parity: competing-risks and frailty survival estimators.

With two competing causes at *constant* cause-specific hazards ``l1`` and
``l2``, the cumulative incidence of cause 1 has the closed form
``CIF_1(t) = l1/(l1+l2) * (1 - exp(-(l1+l2) t))``, so ``CIF_1(inf) =
l1/(l1+l2)``. This pins ``sp.cuminc`` exactly. Gray's test then separates equal
vs unequal incidence curves. ``sp.finegray`` recovers the sign/significance of
a covariate's effect on the subdistribution hazard (and a null under no
effect), and ``sp.cox_frailty`` recovers the log hazard ratio under shared
gamma frailty. Analytical evidence tier (known-truth recovery on deterministic
DGPs).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def _simulate_competing_risks(seed, l1, l2, n=6000, admin_censor=20.0):
    """Two constant cause-specific hazards; light administrative censoring."""
    rng = np.random.default_rng(seed)
    t1 = rng.exponential(1.0 / l1, n)
    t2 = rng.exponential(1.0 / l2, n)
    t = np.minimum(t1, t2)
    cause = np.where(t1 < t2, 1, 2)
    event = np.where(t <= admin_censor, cause, 0)
    dur = np.minimum(t, admin_censor)
    return pd.DataFrame({"dur": dur, "event": event})


# --------------------------------------------------------------------------
# cumulative incidence closed form
# --------------------------------------------------------------------------
def test_cuminc_matches_closed_form_plateau():
    l1, l2 = 0.3, 0.7
    df = _simulate_competing_risks(0, l1, l2)
    res = sp.cuminc(df, duration="dur", event="event")
    cif1 = float(res.cif_at(15, cause=1)["cif"].iloc[0])
    cif2 = float(res.cif_at(15, cause=1 + 1)["cif"].iloc[0])
    # CIF_k(inf) = l_k / (l1 + l2); by t=15 the process has effectively plateaued.
    assert cif1 == pytest.approx(l1 / (l1 + l2), abs=0.02)
    assert cif2 == pytest.approx(l2 / (l1 + l2), abs=0.02)


def test_cuminc_causes_sum_to_one_at_plateau():
    df = _simulate_competing_risks(1, 0.4, 0.6)
    res = sp.cuminc(df, duration="dur", event="event")
    c1 = float(res.cif_at(18, cause=1)["cif"].iloc[0])
    c2 = float(res.cif_at(18, cause=2)["cif"].iloc[0])
    # No survivors remain -> incidences of the two causes sum to 1.
    assert c1 + c2 == pytest.approx(1.0, abs=0.02)


# --------------------------------------------------------------------------
# Gray's test
# --------------------------------------------------------------------------
def test_gray_test_null_when_curves_equal():
    a = _simulate_competing_risks(1, 0.3, 0.7)
    a["g"] = 0
    b = _simulate_competing_risks(2, 0.3, 0.7)
    b["g"] = 1
    res = sp.cuminc(pd.concat([a, b]), duration="dur", event="event", group="g")
    assert float(res.gray_test[1]["p_value"]) > 0.05


def test_gray_test_rejects_when_curves_differ():
    a = _simulate_competing_risks(1, 0.3, 0.7)
    a["g"] = 0
    b = _simulate_competing_risks(3, 0.7, 0.3)  # swapped hazards
    b["g"] = 1
    res = sp.cuminc(pd.concat([a, b]), duration="dur", event="event", group="g")
    assert float(res.gray_test[1]["p_value"]) < 0.01


# --------------------------------------------------------------------------
# Fine-Gray subdistribution hazard
# --------------------------------------------------------------------------
def test_finegray_detects_positive_subdistribution_effect():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.standard_normal(n)
    l1 = 0.3 * np.exp(0.8 * x)  # x raises cause-1 hazard
    l2 = np.full(n, 0.5)
    t1 = rng.exponential(1.0 / l1)
    t2 = rng.exponential(1.0 / l2)
    t = np.minimum(t1, t2)
    cause = np.where(t1 < t2, 1, 2)
    event = np.where(t <= 15.0, cause, 0)
    dur = np.minimum(t, 15.0)
    df = pd.DataFrame({"dur": dur, "event": event, "x": x})
    res = sp.finegray(df, duration="dur", event="event", x=["x"], cause=1)
    assert float(res.params[0]) > 0.3
    assert float(res.pvalues[0]) < 0.01


def test_finegray_null_effect_is_insignificant():
    df = _simulate_competing_risks(5, 0.3, 0.5, n=4000, admin_censor=15.0)
    rng = np.random.default_rng(99)
    df = df.copy()
    df["x"] = rng.standard_normal(len(df))  # covariate unrelated to any cause
    res = sp.finegray(df, duration="dur", event="event", x=["x"], cause=1)
    assert abs(float(res.params[0])) < 0.1
    assert float(res.pvalues[0]) > 0.05


# --------------------------------------------------------------------------
# shared gamma-frailty Cox
# --------------------------------------------------------------------------
def test_cox_frailty_recovers_log_hazard_ratio():
    rng = np.random.default_rng(0)
    K, m, beta, theta = 120, 10, 0.7, 0.4
    rows = []
    for k in range(K):
        w = rng.gamma(1.0 / theta, theta)  # gamma frailty, mean 1, var theta
        for _ in range(m):
            x = rng.standard_normal()
            lam = 0.2 * w * np.exp(beta * x)
            t = rng.exponential(1.0 / lam)
            c = rng.exponential(6.0)
            rows.append((k, min(t, c), int(t <= c), x))
    df = pd.DataFrame(rows, columns=["grp", "time", "event", "x"])
    res = sp.cox_frailty("time + event ~ x", df, cluster="grp")
    assert float(res.params["x"]) == pytest.approx(0.7, abs=0.2)
    assert float(res.theta) > 0.0
