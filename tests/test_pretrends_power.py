"""Tests for ``sp.pretrends_power`` (Roth 2022 pre-trend power).

Regression guard: the function used to raise an opaque
``numpy.linalg.LinAlgError: Singular matrix`` on *every* standard
``sp.event_study`` result, because the omitted reference period (relative
time -1, standard error exactly 0) was kept in the pre-period set and made
the diagonal VCV singular. It now drops zero-SE reference periods first.
"""

import numpy as np
import pytest

import statspai as sp


@pytest.fixture
def event_study_result():
    df = sp.dgp_panel(n_units=120, n_periods=12, seed=0)
    # Even units treated at t=7; odd units never treated (NaN, per the
    # event_study contract).
    df["treat_time"] = np.where(df["unit"] % 2 == 0, 7.0, np.nan)
    df["post"] = ((df["treat_time"].notna()) & (df["time"] >= df["treat_time"])).astype(
        int
    )
    df["y"] = df["y"] + 1.5 * df["post"]
    return sp.event_study(
        df,
        y="y",
        treat_time="treat_time",
        time="time",
        unit="unit",
        window=(-4, 4),
        ref_period=-1,
    )


def test_runs_on_standard_event_study(event_study_result):
    # The headline regression: this previously crashed with LinAlgError.
    out = sp.pretrends_power(event_study_result)
    assert 0.0 <= out["power"] <= 1.0


def test_reference_period_excluded_from_df(event_study_result):
    out = sp.pretrends_power(event_study_result)
    # Window (-4, 4) with ref -1 leaves three estimated pre-periods (-4..-2);
    # the SE = 0 reference period must not inflate the test's degrees of freedom.
    assert out["df"] == 3
    assert len(out["delta"]) == out["df"]


def test_power_increases_with_violation_size(event_study_result):
    small = sp.pretrends_power(event_study_result, delta=np.array([0.2, 0.2, 0.2]))
    large = sp.pretrends_power(event_study_result, delta=np.array([2.0, 2.0, 2.0]))
    assert large["power"] > small["power"]


def test_full_length_delta_is_aligned(event_study_result):
    # A delta with one entry per pre-period (reference included) is accepted
    # and aligned down to the estimated periods.
    out = sp.pretrends_power(event_study_result, delta=np.array([0.5, 0.4, 0.3, 0.0]))
    assert out["df"] == 3


def test_wrong_length_delta_raises(event_study_result):
    with pytest.raises(ValueError, match="delta has length"):
        sp.pretrends_power(event_study_result, delta=np.array([0.5, 0.4]))
