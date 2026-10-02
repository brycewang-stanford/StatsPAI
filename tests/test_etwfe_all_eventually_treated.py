"""sp.etwfe on a panel in which every unit is eventually treated.

With no never-treated unit, nothing is untreated from the last cohort's
adoption date on, so the period effects of those dates are not identified.
Before the fix the saturated design was solved by ``pinv`` regardless and
returned arbitrary numbers (an overall ATT of 4.7e11 on a panel whose true
ATT is 68). R ``etwfe`` takes the last cohort as the reference and drops the
periods from its adoption onwards; ``sp.etwfe`` now does the same and says
so. Checked against R ``etwfe`` 0.6.2 on the Baker, Larcker & Wang (2022)
simulated panel: overall 68.3290597638, cohorts 94.9877509885 /
52.0066074672 / 20.9978906826, all reproduced to 1e-11.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient

COHORTS = (4, 7, 10)
N_PERIODS = 12
SLOPE = {4: 3.0, 7: 2.0, 10: 1.0}


def _panel(noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Staggered adoption, no never-treated unit, effects growing with
    exposure at a cohort-specific rate."""
    rng = np.random.default_rng(seed)
    rows = []
    unit = 0
    for g in COHORTS:
        for _ in range(40):
            alpha = rng.normal()
            for t in range(1, N_PERIODS + 1):
                effect = SLOPE[g] * (t - g + 1) if t >= g else 0.0
                rows.append(
                    {
                        "id": unit,
                        "t": t,
                        "g": g,
                        "y": alpha + 0.5 * t + effect + noise * rng.normal(),
                    }
                )
            unit += 1
    return pd.DataFrame(rows)


def _fit(df: pd.DataFrame, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.etwfe(df, "y", "id", "t", "g", cluster="id", **kwargs)


def test_warns_and_records_the_reference_cohort():
    df = _panel()
    with pytest.warns(UserWarning, match="every unit is eventually treated"):
        res = sp.etwfe(df, "y", "id", "t", "g", cluster="id")
    ref = res.model_info["last_cohort_reference"]
    assert ref["reference_cohort"] == 10
    assert ref["last_period_kept"] == 9
    assert ref["n_obs_dropped"] == int((df["t"] >= 10).sum())
    assert res.n_obs == int((df["t"] < 10).sum())


def test_recovers_the_known_effects_exactly():
    # No noise: every identified cell equals its true effect.
    res = _fit(_panel())
    cohorts = sp.etwfe_emfx(res, type="group").detail.set_index("cohort")
    assert set(cohorts.index) == {4, 7}  # the reference cohort has no ATT
    for g in (4, 7):
        truth = np.mean([SLOPE[g] * (t - g + 1) for t in range(g, 10)])
        # atol: exact recovery up to the conditioning of a 240-column design.
        assert cohorts.loc[g, "estimate"] == pytest.approx(truth, abs=1e-8)
    event = sp.etwfe_emfx(res, type="event").detail.set_index("event_time")
    # Exposure 0 is observed for both identified cohorts, equally sized.
    assert event.loc[0, "estimate"] == pytest.approx(
        (SLOPE[4] + SLOPE[7]) / 2, abs=1e-8
    )


def test_equals_trimming_by_hand():
    df = _panel(noise=1.0, seed=3)
    auto = _fit(df)
    by_hand = df[df["t"] < 10].copy()
    by_hand.loc[by_hand["g"] == 10, "g"] = 0  # never treated in the window
    manual = _fit(by_hand)
    assert "last_cohort_reference" not in manual.model_info
    assert auto.estimate == pytest.approx(manual.estimate, rel=1e-12)
    assert auto.se == pytest.approx(manual.se, rel=1e-12)


@pytest.mark.parametrize("kwargs", [{"fe": "unit"}, {"panel": False}])
def test_other_branches_use_the_same_sample(kwargs):
    df = _panel(noise=1.0, seed=5)
    res = _fit(df, **kwargs)
    assert res.n_obs == int((df["t"] < 10).sum())
    # Balanced panel: the cell coefficients do not depend on the FE basis.
    assert res.estimate == pytest.approx(_fit(df).estimate, rel=1e-9)


def test_untouched_when_a_clean_control_exists():
    df = _panel(noise=1.0, seed=7)
    df.loc[df["g"] == 10, "g"] = 0  # the last cohort is never treated instead
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sp.etwfe(df, "y", "id", "t", "g", cluster="id")
    assert "last_cohort_reference" not in res.model_info
    assert res.n_obs == len(df)


def test_single_adoption_date_without_controls_raises():
    df = _panel()
    one_cohort = df[df["g"] == 4]
    with pytest.raises(DataInsufficient, match="eventually treated"):
        sp.etwfe(one_cohort, "y", "id", "t", "g")
