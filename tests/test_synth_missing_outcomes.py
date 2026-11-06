"""Synthetic control on an incomplete or non-finite outcome panel.

Until 2026-11 one NaN in the treated unit's pre-period outcome made the
classic solver's loss NaN, left the donor weights at equal 1/J with
``converged=True`` and moved the estimate; a NaN in a donor dropped it
without a word (rOpenSci audit probes P1, P2). The rules now, for every
method behind ``sp.synth``: inf in the outcome and a missing treated outcome
are refused, a donor with a missing period is dropped with an
AssumptionWarning and recorded in ``model_info`` (refused by the
balanced-panel methods, kept by matrix completion).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._input_validation import INFINITE_VALUES_HINT
from statspai.exceptions import AssumptionWarning, DataInsufficient
from statspai.synth._panel_checks import outcome_panel_gaps, screen_synth_panel

T0 = 14


def _panel(seed: int = 0, N: int = 12, T: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(T, 2)).cumsum(0)
    lam = rng.normal(size=(N, 2))
    Y = f @ lam.T + rng.normal(scale=0.3, size=(T, N))
    Y[T0:, 0] += 2.0
    return pd.DataFrame(
        [(i, t, Y[t, i]) for i in range(N) for t in range(T)],
        columns=["unit", "time", "y"],
    )


def _fit(df: pd.DataFrame, method: str = "classic", **kw):
    return sp.synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=0,
        treatment_time=T0,
        method=method,
        placebo=False,
        **kw,
    )


def _with(df: pd.DataFrame, unit: int, time: int, value: float) -> pd.DataFrame:
    out = df.copy()
    out.loc[(out.unit == unit) & (out.time == time), "y"] = value
    return out


# --------------------------------------------------------------------------- #
#  The classic estimator (SyntheticControl)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("period", [3, T0 + 2], ids=["pre", "post"])
def test_classic_refuses_missing_treated_outcome(period):
    with pytest.raises(DataInsufficient) as err:
        _fit(_with(_panel(), 0, period, np.nan))
    assert err.value.diagnostics["missing_periods"] == [period]
    assert err.value.diagnostics["treated_unit"] == 0
    assert err.value.recovery_hint


def test_classic_refuses_absent_treated_row():
    df = _panel()
    df = df[~((df.unit == 0) & (df.time == 5))]
    with pytest.raises(DataInsufficient, match="treated unit 0"):
        _fit(df)


@pytest.mark.parametrize("unit", [0, 3], ids=["treated", "donor"])
def test_classic_refuses_inf_outcome(unit):
    with pytest.raises(DataInsufficient) as err:
        _fit(_with(_panel(), unit, 3, np.inf))
    assert err.value.diagnostics["infinite_by_column"] == {"y": 1}
    assert err.value.recovery_hint == INFINITE_VALUES_HINT


@pytest.mark.parametrize("period", [3, T0 + 2], ids=["pre", "post"])
def test_classic_drops_incomplete_donor_with_warning(period):
    df = _panel()
    with pytest.warns(AssumptionWarning, match="1 of 11 donor"):
        res = _fit(_with(df, 3, period, np.nan))
    # dropping the donor is exactly the fit without that unit
    ref = _fit(df[df.unit != 3])
    assert res.estimate == pytest.approx(ref.estimate, rel=1e-12)
    assert res.model_info["n_donors"] == 10
    assert res.model_info["dropped_donors"] == [3]
    assert res.model_info["dropped_donor_missing_periods"] == {"3": 1}
    assert res.model_info["n_donors_before_drop"] == 11
    assert np.isfinite(res.estimate)


def test_classic_complete_panel_has_no_drop_record():
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = _fit(_panel())
    assert "dropped_donors" not in res.model_info
    assert res.model_info["converged"] is True


def test_classic_non_finite_loss_is_not_converged():
    sc = sp.SyntheticControl(
        _panel(),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=0,
        treatment_time=T0,
    )
    # a NaN reaching the solver (the path the input check now closes)
    sc.Y_treated = sc.Y_treated.copy()
    sc.Y_treated[2] = np.nan
    sc.X_treated = sc.Y_treated[sc.pre_mask].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sc.fit(placebo=False)
    assert res.model_info["converged"] is False


# --------------------------------------------------------------------------- #
#  The other methods behind sp.synth
# --------------------------------------------------------------------------- #

FAST_DROP = ["demeaned", "detrended", "unconstrained", "sparse", "penscm", "fdid"]


@pytest.mark.parametrize("method", FAST_DROP)
def test_drop_rule_matches_fit_without_the_donor(method):
    df = _panel()
    with pytest.warns(AssumptionWarning, match="donor unit"):
        res = _fit(_with(df, 3, 5, np.nan), method=method)
    ref = _fit(df[df.unit != 3], method=method)
    assert res.estimate == pytest.approx(ref.estimate, rel=1e-10)
    assert res.model_info["dropped_donors"] == [3]


@pytest.mark.parametrize(
    "method",
    FAST_DROP + ["mc", "rcm", "cluster", "kernel", "scpi", "augmented"],
)
@pytest.mark.parametrize("period", [3, T0 + 2], ids=["pre", "post"])
def test_every_method_refuses_missing_treated_outcome(method, period):
    with pytest.raises(DataInsufficient, match="treated unit 0"):
        _fit(_with(_panel(), 0, period, np.nan), method=method)


@pytest.mark.parametrize("method", ["bsts", "discos", "mc", "rcm", "demeaned"])
def test_every_method_refuses_inf(method):
    with pytest.raises(DataInsufficient, match="infinite"):
        _fit(_with(_panel(), 4, 3, -np.inf), method=method)


def test_matrix_completion_keeps_incomplete_donor():
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = _fit(_with(_panel(), 3, 5, np.nan), method="mc")
    assert res.model_info["n_donors"] == 11
    assert "dropped_donors" not in res.model_info


def test_all_donors_incomplete_is_refused():
    df = _panel(N=4)
    for u in (1, 2, 3):
        df = _with(df, u, 2, np.nan)
    with pytest.raises(DataInsufficient, match="every donor"):
        _fit(df, method="demeaned")


@pytest.mark.parametrize("entry", ["sdid", "gsynth"])
def test_balanced_panel_methods_refuse_incomplete_donor(entry):
    df = _with(_panel(), 3, 5, np.nan)
    with pytest.raises(DataInsufficient) as err:
        _fit(df, method=entry)
    assert err.value.diagnostics["units_with_missing_outcome"] == {"3": 1}


@pytest.mark.parametrize("entry", ["sdid", "gsynth"])
def test_balanced_panel_methods_refuse_missing_treated(entry):
    with pytest.raises(DataInsufficient, match="treated unit 0"):
        _fit(_with(_panel(), 0, 4, np.nan), method=entry)


def test_sdid_direct_entry_checks_the_panel():
    df = _with(_panel(), 0, 4, np.nan)
    with pytest.raises(DataInsufficient, match="sp.sdid"):
        sp.sdid(
            df, outcome="y", unit="unit", time="time", treated_unit=0, treatment_time=T0
        )


# --------------------------------------------------------------------------- #
#  The helpers
# --------------------------------------------------------------------------- #


def test_outcome_panel_gaps_counts_nan_and_absent_rows():
    df = _with(_panel(), 3, 5, np.nan)
    df = df[~((df.unit == 7) & df.time.isin([1, 2]))]
    gaps = outcome_panel_gaps(
        df, outcome="y", unit="unit", time="time", treated_units=[0], function="t"
    )
    assert gaps == {3: 1, 7: 2}


def test_screen_returns_same_object_when_complete():
    df = _panel()
    out, record = screen_synth_panel(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=0,
        method="demeaned",
        inference=None,
    )
    assert out is df
    assert record == {}


def test_screen_skips_unknown_columns():
    df = _panel()
    out, record = screen_synth_panel(
        df,
        outcome="nope",
        unit="unit",
        time="time",
        treated_unit=0,
        method="demeaned",
        inference=None,
    )
    assert out is df and record == {}
