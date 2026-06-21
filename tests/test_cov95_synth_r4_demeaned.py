"""Round-4 coverage margin: synth.demeaned (demeaned_synth).

De-meaned and de-trended synthetic control on a real donor panel.
Exercises both variants with placebo inference, plus the period /
donor / variant validation guards and the weight-solver guard.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.synth import demeaned as _demeaned

demeaned_synth = _demeaned.demeaned_synth


def _panel(n_donor=5, n_pre=15, n_post=8, effect=6.0, seed=0):
    rng = np.random.default_rng(seed)
    T = n_pre + n_post
    years = np.arange(2000, 2000 + T)
    rows = []
    donor_series = []
    for d in range(n_donor):
        s = 20 + d * 2 + 0.3 * np.arange(T) + np.cumsum(rng.normal(0, 0.4, T))
        donor_series.append(s)
        for i, yr in enumerate(years):
            rows.append((f"d{d}", yr, s[i]))
    treated = 0.5 * donor_series[0] + 0.5 * donor_series[1] + rng.normal(0, 0.2, T)
    treated[n_pre:] += effect
    for i, yr in enumerate(years):
        rows.append(("T", yr, treated[i]))
    df = pd.DataFrame(rows, columns=["unit", "year", "y"])
    return df, years[n_pre]


@pytest.mark.parametrize("variant", ["demeaned", "detrended"])
def test_demeaned_synth_variants(variant):
    df, tt = _panel(seed=1)
    res = demeaned_synth(
        data=df,
        outcome="y",
        unit="unit",
        time="year",
        treated_unit="T",
        treatment_time=tt,
        variant=variant,
        placebo=True,
    )
    assert np.isfinite(res.estimate)
    assert res.estimate > 0


def test_demeaned_synth_no_placebo():
    df, tt = _panel(seed=2)
    res = demeaned_synth(
        data=df,
        outcome="y",
        unit="unit",
        time="year",
        treated_unit="T",
        treatment_time=tt,
        variant="detrended",
        placebo=False,
    )
    assert np.isfinite(res.estimate)


def test_demeaned_synth_too_few_pre_raises():
    df, tt = _panel(n_pre=1, n_post=5, seed=3)
    with pytest.raises(ValueError, match="2 pre-treatment"):
        demeaned_synth(
            data=df,
            outcome="y",
            unit="unit",
            time="year",
            treated_unit="T",
            treatment_time=tt,
        )


def test_demeaned_synth_invalid_variant_raises():
    df, tt = _panel(seed=4)
    with pytest.raises(ValueError, match="variant must be"):
        demeaned_synth(
            data=df,
            outcome="y",
            unit="unit",
            time="year",
            treated_unit="T",
            treatment_time=tt,
            variant="bogus",
        )


def test_solve_weights_no_donors_raises():
    with pytest.raises(ValueError, match="No donor"):
        _demeaned._solve_weights(np.zeros(5), np.zeros((5, 0)))
