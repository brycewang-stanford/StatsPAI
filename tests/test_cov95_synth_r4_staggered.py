"""Round-4 coverage margin: synth.staggered (staggered_synth).

Real staggered-adoption panel with two treated cohorts plus several
never-treated controls. Exercises both the 'separate' and 'pooled'
estimation paths, the penalized weight solver, placebo inference, and
the input-validation guards.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.synth import staggered as _staggered

staggered_synth = _staggered.staggered_synth


def _staggered_panel(seed=0, n_control=4, effect=6.0):
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2018)
    rows = []
    # Never-treated controls.
    controls = {}
    for c in range(n_control):
        s = 20 + c + np.cumsum(rng.normal(0, 0.5, len(years)))
        controls[f"c{c}"] = s
        for i, yr in enumerate(years):
            rows.append((f"c{c}", yr, s[i], 0))
    # Two treated cohorts adopting in 2008 and 2012.
    cohort_map = {"t_early": 2008, "t_early2": 2008, "t_late": 2012}
    for u, g in cohort_map.items():
        base = (
            0.5 * controls["c0"] + 0.5 * controls["c1"] + rng.normal(0, 0.3, len(years))
        )
        for i, yr in enumerate(years):
            treated_flag = 1 if yr >= g else 0
            val = base[i] + (effect if treated_flag else 0.0)
            rows.append((u, yr, val, treated_flag))
    df = pd.DataFrame(rows, columns=["unit", "year", "y", "treat"])
    return df


def test_staggered_synth_separate():
    df = _staggered_panel(seed=1)
    res = staggered_synth(
        data=df,
        outcome="y",
        unit="unit",
        time="year",
        treatment="treat",
        method="separate",
        placebo=True,
    )
    assert np.isfinite(res.estimate)
    assert res.estimate > 0


def test_staggered_synth_pooled_penalized():
    df = _staggered_panel(seed=2)
    res = staggered_synth(
        data=df,
        outcome="y",
        unit="unit",
        time="year",
        treatment="treat",
        method="pooled",
        penalization=0.1,
        placebo=True,
    )
    assert np.isfinite(res.estimate)
    assert res.estimate > 0


def test_staggered_synth_no_placebo():
    df = _staggered_panel(seed=3)
    res = staggered_synth(
        data=df,
        outcome="y",
        unit="unit",
        time="year",
        treatment="treat",
        method="separate",
        placebo=False,
    )
    assert np.isfinite(res.estimate)


def test_staggered_synth_no_treated_raises():
    df = _staggered_panel(seed=4)
    df = df.copy()
    df["treat"] = 0  # nobody ever treated
    with pytest.raises(ValueError, match="No treated"):
        staggered_synth(
            data=df,
            outcome="y",
            unit="unit",
            time="year",
            treatment="treat",
        )


def test_staggered_synth_no_never_treated_raises():
    # Every unit is treated at some point -> no pure controls.
    rng = np.random.default_rng(5)
    years = np.arange(2000, 2012)
    rows = []
    for u, g in {"a": 2005, "b": 2006, "c": 2007}.items():
        s = 20 + np.cumsum(rng.normal(0, 0.5, len(years)))
        for i, yr in enumerate(years):
            rows.append((u, yr, s[i], 1 if yr >= g else 0))
    df = pd.DataFrame(rows, columns=["unit", "year", "y", "treat"])
    with pytest.raises(ValueError, match="never-treated"):
        staggered_synth(
            data=df,
            outcome="y",
            unit="unit",
            time="year",
            treatment="treat",
        )


def test_solve_weights_no_donors_raises():
    with pytest.raises(ValueError, match="No donor"):
        _staggered._solve_weights(np.zeros(5), np.zeros((5, 0)))


def test_solve_weights_penalized_runs():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(8, 3))
    y = X @ np.array([0.5, 0.3, 0.2]) + rng.normal(0, 0.01, 8)
    w = _staggered._solve_weights(y, X, penalization=0.5)
    assert w.shape == (3,)
    assert np.all(w >= -1e-8)
