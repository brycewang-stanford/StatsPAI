"""Reference parity: ``sp.did_multiplegt`` dynamic + placebo effects vs Stata
``did_multiplegt_old`` (robust_dynamic).

The dCDH (2020) dynamic-robust estimator (i) conditions on baseline treatment
d_{t-1}, (ii) for horizon l compares the long difference Y_{t+l}-Y_{t-1} of
switchers to *robust* stayers — units that keep the baseline treatment over the
whole window [t, t+l] — and (iii) sign-flips switch-off cells. The placebo uses
the mirror sign convention. Before this fix, ``_estimate_dynamic`` /
``_estimate_placebo`` pooled all switchers/stayers (no baseline conditioning, no
robust-stayer window), which contaminated the control trend and even flipped the
placebo sign.

On real ``did::mpdta`` (see Paper-DiD-JAE/replication/), Stata
``did_multiplegt (old) ..., dynamic(2) placebo(1) robust_dynamic`` returns
effect_0 -0.018922, effect_1 -0.053589, effect_2 -0.136274, placebo_1 +0.024269;
StatsPAI now reproduces all four to ~1e-6. The two tests below use small
hand-computable panels so they are self-contained.
"""
import pandas as pd
import pytest

from statspai.did import did_multiplegt


def test_dynamic_robust_stayers_exclude_already_treated():
    """Horizon-l long differences must use stayers that remain untreated through
    t+l. Constructed absorbing panel: controls A,B flat; C already-treated with a
    strong +10/period trend (must be EXCLUDED as a robust stayer); switcher D
    enters at t=2 with effect 5 at h=0 and 10 at h=1 vs the controls' flat trend.
    """
    rows = [
        ("A", 1, 0, 0.0),
        ("A", 2, 0, 0.0),
        ("A", 3, 0, 0.0),
        ("B", 1, 0, 1.0),
        ("B", 2, 0, 1.0),
        ("B", 3, 0, 1.0),
        ("C", 1, 1, 0.0),
        ("C", 2, 1, 10.0),
        ("C", 3, 1, 20.0),  # already-treated
        ("D", 1, 0, 0.0),
        ("D", 2, 1, 5.0),
        ("D", 3, 1, 10.0),  # switcher 0->1 at t=2
    ]
    df = pd.DataFrame(rows, columns=["g", "t", "d", "y"])
    r = did_multiplegt(
        df,
        y="y",
        group="g",
        time="t",
        treatment="d",
        placebo=0,
        dynamic=1,
        n_boot=0,
        seed=1,
    )
    dyn = {d["horizon"]: d["estimate"] for d in r.model_info["dynamic"]}
    assert dyn[0] == pytest.approx(5.0, abs=1e-9), f"h0={dyn[0]} (C contaminating?)"
    assert dyn[1] == pytest.approx(10.0, abs=1e-9), f"h1={dyn[1]} (robust stayer?)"


def test_placebo_sign_and_baseline_conditioning():
    """Placebo first-difference + Stata sign convention. Switcher D enters at t=3
    with a pre-trend of +2 over (t=1,t=2); controls A,B flat. The dCDH/Stata
    placebo_1 = -(switcher pre-change - stayer pre-change) = -2.0 (the old pooled
    code returned the wrong magnitude/sign).
    """
    rows = [
        ("A", 1, 0, 0.0),
        ("A", 2, 0, 0.0),
        ("A", 3, 0, 0.0),
        ("B", 1, 0, 1.0),
        ("B", 2, 0, 1.0),
        ("B", 3, 0, 1.0),
        ("D", 1, 0, 0.0),
        ("D", 2, 0, 2.0),
        ("D", 3, 1, 9.0),  # pre-trend +2, switch at t=3
    ]
    df = pd.DataFrame(rows, columns=["g", "t", "d", "y"])
    r = did_multiplegt(
        df,
        y="y",
        group="g",
        time="t",
        treatment="d",
        placebo=1,
        dynamic=0,
        n_boot=0,
        seed=1,
    )
    plac = {p["lag"]: p["estimate"] for p in r.model_info["placebo"]}
    assert plac[-1] == pytest.approx(
        -2.0, abs=1e-9
    ), f"placebo_1={plac[-1]} (sign/baseline?)"
