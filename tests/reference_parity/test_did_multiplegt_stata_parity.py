"""Reference parity: ``sp.did_multiplegt`` DID_M vs Stata ``did_multiplegt_old``.

The dCDH (2020) static DID_M conditions switchers/stayers on the baseline
treatment ``d_{t-1}`` and sign-flips switch-off cells. Prior to the
baseline-conditioning fix, ``sp.did_multiplegt`` pooled all switchers/stayers in
each period under a single majority sign, which (i) contaminated the control
trend with already-treated stayers and (ii) mixed switch-on/off effects --
giving 0.396 on the on/off DGP below (true effect 0.6) and -0.0170 on mpdta.

The Stata reference values were produced by ``did_multiplegt (old) ...`` on the
identical panels (Stata 18 MP, ssc ``did_multiplegt`` old mode); see
``Paper-DiD-JAE/replication/did_multiplegt/``. StatsPAI now matches them to ~1e-8.
"""
import numpy as np
import pandas as pd
import pytest

from statspai.did import did_multiplegt

# Stata did_multiplegt_old effect_0 on the two panels (breps=0 point estimate).
STATA_ONOFF = 0.6964177538040197
STATA_MPDTA = -0.01892218940086894


def _onoff_panel(n_units=200, n_periods=8, seed=12345):
    """On/off switching DGP (treatment turns on AND off). True switch effect 0.6.

    Identical to Paper-DiD-JAE/replication/did_multiplegt/build_panel.py so the
    Stata reference value applies.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        on = int(rng.integers(3, 7))
        off = int(rng.integers(on + 1, n_periods + 2))
        fx = rng.normal(scale=0.3)
        for t in range(1, n_periods + 1):
            d = int(on <= t < off)
            y = fx + 0.15 * t + 0.6 * d + rng.normal()
            rows.append({"g": u, "t": t, "d": d, "y": y})
    return pd.DataFrame(rows)


def test_did_m_onoff_matches_stata():
    """On/off panel: DID_M must match Stata did_multiplegt_old, not the old
    baseline-pooled value (0.396)."""
    df = _onoff_panel()
    r = did_multiplegt(
        df,
        y="y",
        group="g",
        time="t",
        treatment="d",
        placebo=0,
        dynamic=0,
        n_boot=0,
        seed=1,
    )
    assert r.estimate == pytest.approx(
        STATA_ONOFF, rel=1e-4
    ), f"DID_M={r.estimate:.8f} != Stata {STATA_ONOFF:.8f}; baseline conditioning?"
    # the bug returned ~0.396 (well outside tolerance); guard against regression
    assert abs(r.estimate - 0.3963) > 0.05
    assert (r.model_info or {}).get("n_switchers") == 346


def test_did_m_excludes_already_treated_stayers():
    """Constructed absorbing panel with a hand-computable answer that isolates
    the control-contamination bug.

    Controls A,B (never treated, dy=0); C is already-treated (1->1 stayer) with a
    strong +10 trend; D,E switch 0->1 at t=2 with a true effect of 5 vs the
    controls' 0 trend. Correct DID_M conditions on baseline 0, so C (baseline 1)
    is NOT in the comparison and the answer is exactly 5.0. The old code pooled C
    into the stayers, contaminating the control trend and returning ~1.667.
    """
    rows = [
        ("A", 1, 0, 0.0),
        ("A", 2, 0, 0.0),
        ("B", 1, 0, 1.0),
        ("B", 2, 0, 1.0),
        ("C", 1, 1, 0.0),
        ("C", 2, 1, 10.0),  # already-treated, strong trend
        ("D", 1, 0, 0.0),
        ("D", 2, 1, 5.0),  # switcher 0->1, effect 5
        ("E", 1, 0, 2.0),
        ("E", 2, 1, 7.0),  # switcher 0->1, effect 5
    ]
    df = pd.DataFrame(rows, columns=["g", "t", "d", "y"])
    r = did_multiplegt(
        df,
        y="y",
        group="g",
        time="t",
        treatment="d",
        placebo=0,
        dynamic=0,
        n_boot=0,
        seed=1,
    )
    assert r.estimate == pytest.approx(5.0, abs=1e-9), (
        f"DID_M={r.estimate}; already-treated stayer C contaminating the control "
        f"trend? (buggy pooled value ~1.667)"
    )
