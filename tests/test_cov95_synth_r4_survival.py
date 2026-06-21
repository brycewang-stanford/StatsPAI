"""Round-4 coverage margin: synth.survival (synth_survival).

Synthetic-control on survival curves (cloglog scale). Real survival
panels with a treated arm and several donor arms. Exercises both the
boolean-flag and explicit-name treated-unit detection, the donor /
pre-period guards, and the no-CI fallback.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.synth import survival as _survival

synth_survival = _survival.synth_survival


def _survival_panel(
    n_donor=5, n_time=12, treat_time=6, effect=-0.1, seed=0, treated_flag=True
):
    """Survival curves: monotone-decreasing in (0,1), treated drops faster."""
    rng = np.random.default_rng(seed)
    months = np.arange(n_time)
    rows = []
    donor_curves = {}
    for d in range(n_donor):
        # Decreasing survival from ~0.95 down.
        hazard = 0.03 + 0.01 * rng.random()
        s = np.clip(
            0.95 * np.exp(-hazard * months) + rng.normal(0, 0.005, n_time), 0.01, 0.99
        )
        donor_curves[d] = s
        for i, m in enumerate(months):
            rows.append((f"d{d}", m, s[i], 0))
    base = np.mean([donor_curves[d] for d in range(n_donor)], axis=0)
    treated = base.copy()
    treated[treat_time:] = np.clip(treated[treat_time:] + effect, 0.01, 0.99)
    for i, m in enumerate(months):
        flag = 1 if treated_flag else 0
        rows.append(("treated_arm", m, treated[i], flag))
    df = pd.DataFrame(rows, columns=["arm", "month", "surv", "is_treated"])
    return df


def test_synth_survival_boolean_flag():
    df = _survival_panel(seed=1)
    df = df.copy()
    df["is_treated"] = df["is_treated"].astype(bool)
    res = synth_survival(
        data=df,
        unit="arm",
        time="month",
        survival="surv",
        treated="is_treated",
        treat_time=6,
        n_placebos=4,
    )
    assert res.treated_unit == "treated_arm"
    assert res.s_treated.shape[0] == 12


def test_synth_survival_int_flag_and_summary():
    # Integer (non-bool) treatment flag -> the astype(bool) branch.
    df = _survival_panel(seed=11)
    res = synth_survival(
        data=df,
        unit="arm",
        time="month",
        survival="surv",
        treated="is_treated",
        treat_time=6,
        n_placebos=4,
    )
    s = res.summary()
    assert "Synthetic Survival Control" in s
    assert "treated_arm" in s


def test_synth_survival_multiple_treated_raises():
    # Two arms flagged treated -> "exactly one treated unit" guard.
    df = _survival_panel(seed=12)
    df = df.copy()
    df.loc[df["arm"] == "d0", "is_treated"] = 1
    with pytest.raises(ValueError, match="one treated"):
        synth_survival(
            data=df,
            unit="arm",
            time="month",
            survival="surv",
            treated="is_treated",
            treat_time=6,
        )


def test_synth_survival_explicit_unit_name():
    # 'treated' is not a column -> interpreted as the unit name directly.
    df = _survival_panel(seed=2)
    res = synth_survival(
        data=df,
        unit="arm",
        time="month",
        survival="surv",
        treated="treated_arm",
        treat_time=6,
        n_placebos=3,
    )
    assert res.treated_unit == "treated_arm"


def test_synth_survival_missing_column_raises():
    df = _survival_panel(seed=3)
    with pytest.raises(ValueError, match="not found"):
        synth_survival(
            data=df,
            unit="arm",
            time="month",
            survival="nope",
            treated="treated_arm",
            treat_time=6,
        )


def test_synth_survival_treated_not_found_raises():
    df = _survival_panel(seed=4)
    with pytest.raises(ValueError, match="not found among units"):
        synth_survival(
            data=df,
            unit="arm",
            time="month",
            survival="surv",
            treated="ghost_arm",
            treat_time=6,
        )


def test_synth_survival_no_donors_raises():
    # Only the treated arm present.
    months = np.arange(10)
    s = np.clip(0.9 * np.exp(-0.05 * months), 0.01, 0.99)
    rows = [("treated_arm", m, s[i], 1) for i, m in enumerate(months)]
    df = pd.DataFrame(rows, columns=["arm", "month", "surv", "is_treated"])
    with pytest.raises(Exception):
        synth_survival(
            data=df,
            unit="arm",
            time="month",
            survival="surv",
            treated="treated_arm",
            treat_time=5,
        )


def test_synth_survival_too_few_pre_periods_raises():
    df = _survival_panel(seed=6, treat_time=1)
    with pytest.raises(ValueError, match="pre-treatment"):
        synth_survival(
            data=df,
            unit="arm",
            time="month",
            survival="surv",
            treated="treated_arm",
            treat_time=1,
        )


def test_synth_survival_no_placebos_no_ci():
    # n_placebos=0 -> placebo_gaps empty -> CI fallback to None.
    df = _survival_panel(seed=7)
    res = synth_survival(
        data=df,
        unit="arm",
        time="month",
        survival="surv",
        treated="treated_arm",
        treat_time=6,
        n_placebos=0,
    )
    assert res is not None
