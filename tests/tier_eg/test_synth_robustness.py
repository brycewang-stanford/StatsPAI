"""Tier G — robustness tests for ``sp.synth`` (classic SCM).

Observed contract (StatsPAI 1.17.0), locked here — every degenerate design
raises a clear, specific ``ValueError`` rather than returning a silent number:

* **treated unit absent** -> "Treated unit '...' not found".
* **fewer than 2 pre-treatment periods** -> "Need at least 2 pre-treatment
  periods".
* **no post-treatment period** -> "Need at least 1 post-treatment period".
* **missing outcome column** -> "Column '...' not found".
* **NaN outcomes** -> finite ATT (the optimiser still solves on observed data).
* **single donor** -> finite ATT (the synthetic control collapses to that one
  donor; well-defined, if uninformative).
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp

from ._helpers import assert_raises_clean, coef, make_synth

_BASE = dict(
    outcome="y",
    unit="unit",
    time="time",
    method="classic",
    placebo=False,
)


@pytest.fixture(scope="module")
def base_df():
    return make_synth(n_donors=18, n_periods=20, treat_period=15, seed=1)


# --------------------------------------------------------------------------- #
# G — design / schema errors                                                  #
# --------------------------------------------------------------------------- #
def test_synth_treated_unit_absent_raises(base_df):
    assert_raises_clean(
        lambda: sp.synth(base_df, treated_unit=999, treatment_time=15, **_BASE),
        ValueError,
        match="treated unit|not found",
    )


def test_synth_no_pre_period_raises(base_df):
    assert_raises_clean(
        lambda: sp.synth(base_df, treated_unit=0, treatment_time=1, **_BASE),
        ValueError,
        match="pre-treatment|pre period|period",
    )


def test_synth_no_post_period_raises(base_df):
    assert_raises_clean(
        lambda: sp.synth(base_df, treated_unit=0, treatment_time=999, **_BASE),
        ValueError,
        match="post-treatment|post period|period",
    )


def test_synth_missing_outcome_raises(base_df):
    assert_raises_clean(
        lambda: sp.synth(
            base_df,
            treated_unit=0,
            treatment_time=15,
            outcome="qqq",
            unit="unit",
            time="time",
            method="classic",
            placebo=False,
        ),
        ValueError,
        KeyError,
        match="qqq|not found",
    )


# --------------------------------------------------------------------------- #
# G1 — NaN outcomes: finite ATT                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore")
def test_synth_nan_outcome_treated_refused_donor_dropped(base_df):
    """A NaN in the treated unit's outcome is refused (it used to give a
    finite but wrong ATT); a NaN donor is dropped with a warning, which is
    the fit without that donor."""
    treated = base_df.copy()
    treated.loc[(treated["unit"] == 0) & (treated["time"] < 5), "y"] = np.nan
    with pytest.raises(sp.DataInsufficient, match="treated unit 0"):
        sp.synth(treated, treated_unit=0, treatment_time=15, **_BASE)
    donor = base_df.copy()
    donor.loc[(donor["unit"] == 3) & (donor["time"] < 5), "y"] = np.nan
    got = sp.synth(donor, treated_unit=0, treatment_time=15, **_BASE)
    ref = sp.synth(
        base_df[base_df["unit"] != 3], treated_unit=0, treatment_time=15, **_BASE
    )
    assert np.isfinite(coef(got))
    np.testing.assert_allclose(coef(got), coef(ref), rtol=1e-10)
    assert got.model_info["dropped_donors"] == [3]


# --------------------------------------------------------------------------- #
# G — single donor: collapses to that donor, finite                          #
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore")
def test_synth_single_donor_finite(base_df):
    pair = base_df[base_df["unit"].isin([0, 1])]
    got = sp.synth(pair, treated_unit=0, treatment_time=15, **_BASE)
    assert np.isfinite(coef(got)), "single-donor synth returned a non-finite ATT"
