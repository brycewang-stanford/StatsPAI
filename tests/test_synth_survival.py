"""Tests for Synthetic Survival Control (Agarwal-Shah 2025)."""

import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp


def _survival_panel(seed=0, T=24, n_donors=6, hazard_reduction=0.5, treat_time=6):
    rng = np.random.default_rng(seed)
    t_grid = np.arange(1, T + 1)
    h_t = 0.04 * np.ones(T)
    h_t[treat_time:] *= hazard_reduction
    rows = []
    s_t = np.exp(-np.cumsum(h_t))
    for i, t in enumerate(t_grid):
        rows.append({"unit": "treated", "time": t, "km": s_t[i], "tr": 1})
    for j in range(n_donors):
        h = 0.04 * (1 + 0.1 * rng.normal())
        s = np.exp(-np.cumsum(np.full(T, h) + rng.normal(0, 0.003, T)))
        for i, t in enumerate(t_grid):
            rows.append({"unit": f"donor{j}", "time": t, "km": s[i], "tr": 0})
    return pd.DataFrame(rows), treat_time


def test_synth_survival_matches_pretreat():
    df, tt = _survival_panel(seed=0)
    r = sp.synth_survival(df, unit="unit", time="time", survival="km",
                           treated="tr", treat_time=tt, n_placebos=5)
    assert r.pre_rmse is not None and r.pre_rmse < 0.05


def test_synth_survival_positive_gap_when_treated_benefits():
    df, tt = _survival_panel(seed=1, hazard_reduction=0.3)  # big benefit
    r = sp.synth_survival(df, unit="unit", time="time", survival="km",
                           treated="tr", treat_time=tt, n_placebos=5)
    # Treated has lower hazard -> higher survival -> positive gap
    post = r.time_grid >= tt
    assert np.mean(r.gap[post]) > 0


def test_synth_survival_weights_on_simplex():
    df, tt = _survival_panel(seed=2)
    r = sp.synth_survival(df, unit="unit", time="time", survival="km",
                           treated="tr", treat_time=tt, n_placebos=5)
    w = list(r.weights.values())
    assert abs(sum(w) - 1.0) < 1e-4
    assert all(wi >= -1e-9 for wi in w)


def test_synth_survival_summary():
    df, tt = _survival_panel(seed=3)
    r = sp.synth_survival(df, unit="unit", time="time", survival="km",
                           treated="tr", treat_time=tt, n_placebos=5)
    s = r.summary()
    assert "Synthetic Survival Control" in s
    assert "treated" in s


def test_synth_survival_rejects_two_treated():
    df, tt = _survival_panel(seed=4)
    df.loc[df["unit"] == "donor0", "tr"] = 1  # add a second 'treated'
    with pytest.raises(ValueError):
        sp.synth_survival(df, unit="unit", time="time", survival="km",
                           treated="tr", treat_time=tt)


def test_synth_survival_registered():
    assert "synth_survival" in sp.list_functions()
