"""Coverage gap-fill round 2 — ``statspai.synth`` remaining branches.

Targets (from the uncovered-line report):

- ``scm.py`` argument-validation helpers (blank method, bool/str alpha and
  penalization, non-list covariates, fractional n_random_starts), the
  ``SyntheticControl`` option checks (v_method, standardize_predictors),
  the treated-unit-dropped-by-pivot and non-comparable treatment_time
  errors, the equal-V unstandardized path, and the R ``Synth`` reference
  backend up to its Rscript guard;
- ``sdid.py``: the no-post-period guard, the R ``synthdid`` backend panel
  construction up to its Rscript guard, the jackknife SE degenerate case
  (single control -> empty leave-one-out estimates -> se=0), and the
  ``synthdid_plot`` / ``units_plot`` / ``rmse_plot`` caller-supplied-axes
  branches;
- ``augsynth.py``: the no-post-period guard and the R ``augsynth`` backend
  (numeric-time check, panel construction, Rscript guard);
- ``gsynth.py``: the R backend numeric-time check.

The Rscript-guard tests are skipped when an actual R installation is
present (they assert the loud, actionable error raised without R).
"""

from __future__ import annotations

import shutil
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import (  # noqa: E402
    ConvergenceFailure,
    DataInsufficient,
    MethodIncompatibility,
)

_HAS_RSCRIPT = shutil.which("Rscript") is not None
_skip_if_r = pytest.mark.skipif(
    _HAS_RSCRIPT, reason="Rscript installed; the no-R guard path is untestable"
)

EFFECT = 4.0
TREAT_TIME = 10


def _panel(n_units=8, n_periods=14, effect=EFFECT, seed=3):
    """Small long-format panel: unit u0 treated from TREAT_TIME with `effect`."""
    rng = np.random.default_rng(seed)
    alphas = rng.normal(10, 2, n_units)
    common = rng.normal(0, 0.4, n_periods)
    recs = []
    for i in range(n_units):
        for ti, t in enumerate(range(1, n_periods + 1)):
            y = alphas[i] + 0.5 * t + common[ti] + rng.normal(0, 0.2)
            if i == 0 and t >= TREAT_TIME:
                y += effect
            recs.append({"unit": f"u{i}", "time": t, "outcome": y})
    return pd.DataFrame(recs)


COMMON = dict(
    outcome="outcome",
    unit="unit",
    time="time",
    treated_unit="u0",
    treatment_time=TREAT_TIME,
)


# ── sp.synth argument validation (scm.py helpers) ─────────────────────────


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(method="   "), "non-empty string option"),
        (dict(alpha=True), r"`alpha` must be a number in \(0, 1\)"),
        (dict(alpha="x"), r"`alpha` must be a number in \(0, 1\)"),
        (dict(penalization=True), "non-negative finite number"),
        (dict(penalization="x"), "non-negative finite number"),
        (dict(covariates=123), "column name or list of column names"),
        (dict(n_random_starts=True), "integer >= 0"),
        (dict(n_random_starts="x"), "integer >= 0"),
        (dict(n_random_starts=1.5), "integer >= 0"),
        (dict(v_method="bogus"), "v_method must be one of"),
        (dict(standardize_predictors="yes"), "must be True or False"),
    ],
)
def test_synth_argument_validation(kwargs, match):
    df = _panel(n_units=6, n_periods=12)
    call = dict(data=df, placebo=False)
    call.update(COMMON)
    call.setdefault("method", "classic")
    call.update(kwargs)
    with pytest.raises(MethodIncompatibility, match=match):
        sp.synth(**call)


def test_synth_treated_unit_all_nan_outcome():
    # pivot_table drops all-NaN columns -> treated unit vanishes after pivot.
    df = _panel(n_units=6, n_periods=12)
    df.loc[df["unit"] == "u0", "outcome"] = np.nan
    with pytest.raises(DataInsufficient, match="missing after pivot"):
        sp.synth(df, **COMMON, method="classic", placebo=False)


def test_synth_treatment_time_not_comparable():
    df = _panel(n_units=6, n_periods=12)
    call = dict(COMMON, treatment_time="ten")
    with pytest.raises(MethodIncompatibility, match="comparable with the time column"):
        sp.synth(df, **call, method="classic", placebo=False)


def test_synth_equal_v_unstandardized_predictors():
    df = _panel()
    res = sp.synth(
        df,
        **COMMON,
        method="classic",
        v_method="equal",
        standardize_predictors=False,
        placebo=False,
    )
    assert np.isfinite(res.estimate)
    w = res.model_info["weights"]["weight"].values
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all()  # classic SCM weights stay on the simplex


# ── sp.synth R (Synth) reference backend ──────────────────────────────────


@_skip_if_r
def test_synth_r_backend_requires_rscript():
    df = _panel()
    with pytest.raises(ConvergenceFailure, match="requires Rscript"):
        sp.synth(df, **COMMON, method="classic", backend="synth", placebo=False)


def test_synth_r_backend_missing_treated_unit():
    df = _panel(n_units=6, n_periods=12)
    with pytest.raises(
        MethodIncompatibility, match="treated_unit and treatment_time are required"
    ):
        sp.synth(
            df,
            outcome="outcome",
            unit="unit",
            time="time",
            method="classic",
            backend="synth",
            placebo=False,
        )


def test_synth_r_backend_rejects_string_time():
    df = _panel(n_units=6, n_periods=12)
    df["time"] = df["time"].astype(str)
    call = dict(COMMON, treatment_time=str(TREAT_TIME))
    with pytest.raises(MethodIncompatibility, match="numeric time column"):
        sp.synth(df, **call, method="classic", backend="synth", placebo=False)


# ── sdid ──────────────────────────────────────────────────────────────────


def test_sdid_no_post_period_raises():
    df = _panel(n_units=6, n_periods=12)
    with pytest.raises(DataInsufficient, match="post-treatment period"):
        sp.sdid(df, **dict(COMMON, treatment_time=999))


@_skip_if_r
def test_sdid_r_backend_requires_rscript():
    # Valid args must survive method/se_method validation and the panel
    # build, then fail loudly at the Rscript guard.
    df = _panel()
    with pytest.raises(RuntimeError, match="requires Rscript"):
        sp.sdid(df, **COMMON, backend="r")


def test_sdid_jackknife_single_control_degenerates_to_zero_se():
    # With one control, every leave-one-out re-fit has zero donors and
    # fails, so the jackknife returns an empty tau set -> se must be 0,
    # not NaN, and the point estimate is still the two-unit SDID contrast.
    df = _panel()
    d2 = df[df["unit"].isin(["u0", "u1"])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = sp.sdid(d2, **COMMON, se_method="jackknife")
    assert res.se == 0.0
    assert abs(res.estimate - EFFECT) < 1.5


def test_synthdid_plots_reuse_caller_axes():
    df = _panel()
    res = sp.sdid(df, **COMMON, seed=1)
    assert abs(res.estimate - EFFECT) < 1.0

    fig0, ax0 = plt.subplots()
    _, ax = sp.synthdid_plot(res, ax=ax0)
    assert ax is ax0

    fig1, ax1 = plt.subplots()
    _, ax = sp.synthdid_units_plot(res, ax=ax1)
    assert ax is ax1

    fig2, ax2 = plt.subplots()
    _, ax = sp.synthdid_rmse_plot(res, ax=ax2)
    assert ax is ax2
    plt.close("all")


# ── augsynth ──────────────────────────────────────────────────────────────


def test_augsynth_no_post_period_raises():
    df = _panel(n_units=6, n_periods=12)
    with pytest.raises(DataInsufficient, match="post-treatment period"):
        sp.augsynth(df, **dict(COMMON, treatment_time=999))


@_skip_if_r
def test_augsynth_r_backend_requires_rscript():
    df = _panel()
    with pytest.raises(RuntimeError, match="requires Rscript"):
        sp.augsynth(df, **COMMON, backend="r")


def test_augsynth_r_backend_rejects_string_time():
    df = _panel(n_units=6, n_periods=12)
    df["time"] = df["time"].astype(str)
    call = dict(COMMON, treatment_time=str(TREAT_TIME))
    with pytest.raises(TypeError, match="numeric time column"):
        sp.augsynth(df, **call, backend="r")


# ── gsynth ────────────────────────────────────────────────────────────────


@_skip_if_r
def test_gsynth_r_backend_requires_rscript():
    df = _panel()
    with pytest.raises(RuntimeError, match="requires Rscript"):
        sp.gsynth(df, **COMMON, backend="r")


def test_gsynth_r_backend_rejects_string_time():
    df = _panel(n_units=6, n_periods=12)
    df["time"] = df["time"].astype(str)
    call = dict(COMMON, treatment_time=str(TREAT_TIME))
    with pytest.raises(TypeError, match="numeric time column"):
        sp.gsynth(df, **call, backend="r")
