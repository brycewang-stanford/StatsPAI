"""Coverage round-4 — triple-diff heterogeneity, time-varying covariates,
and misclassified-treatment DiD.

- ``sp.ddd_heterogeneous`` — Olden-Møen heterogeneity-robust DDD across
  cohort × time, with the cluster bootstrap SE + placebo joint Wald.
- ``sp.did_timevarying_covariates`` — ATT(g,t) with baseline-anchored
  time-varying covariates.
- ``sp.did_misclassified`` — naive CS ATT with the symmetric
  misclassification correction and anticipation leads.

The DGPs encode a known affected-vs-unaffected gap (DDD true = 1.5) and a
constant +2 ATT respectively; assertions check sign / magnitude window,
SE finiteness, p-values in [0, 1], and that bad inputs raise.  No
fabricated numbers, no mocking of the numeric path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# ── DDD heterogeneity ────────────────────────────────────────────── #


def _ddd_panel(seed=0, n_units=150, n_periods=8):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 4, 6][u % 3]
        sub = u % 2  # affected subgroup indicator
        fe = rng.normal()
        for t in range(1, n_periods + 1):
            treated_now = (g > 0) and (t >= g)
            te = (2.0 if sub == 1 else 0.5) if treated_now else 0.0
            y = fe + 0.2 * t + te + rng.normal(0, 0.4)
            rows.append(
                {"unit": u, "time": t, "y": y, "first_treat": g, "affected": sub}
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def ddd_panel():
    return _ddd_panel()


def test_ddd_heterogeneous_recovers(ddd_panel):
    r = sp.ddd_heterogeneous(
        ddd_panel,
        y="y",
        unit="unit",
        time="time",
        cohort="first_treat",
        subgroup="affected",
        n_boot=120,
        seed=0,
    )
    # true DDD = affected (2.0) - unaffected (0.5) = 1.5
    assert abs(r.estimate - 1.5) < 0.6
    assert np.isfinite(r.se) and r.se > 0
    assert 0.0 <= r.pvalue <= 1.0
    lo, hi = r.ci
    assert lo <= r.estimate <= hi
    assert len(r.detail) >= 1
    assert r.model_info["placebo_joint_test"] is not None


def test_ddd_missing_column_raises(ddd_panel):
    with pytest.raises(ValueError):
        sp.ddd_heterogeneous(
            ddd_panel,
            y="y",
            unit="unit",
            time="time",
            cohort="first_treat",
            subgroup="missing",
        )


def test_ddd_nonbinary_subgroup_raises(ddd_panel):
    bad = ddd_panel.copy()
    bad["affected"] = bad["affected"] + 2  # now {2,3}, not binary
    with pytest.raises(ValueError):
        sp.ddd_heterogeneous(
            bad,
            y="y",
            unit="unit",
            time="time",
            cohort="first_treat",
            subgroup="affected",
        )


def test_ddd_no_never_treated_raises(ddd_panel):
    # Drop the never-treated cohort -> must raise (no clean comparison group).
    treated_only = ddd_panel[ddd_panel["first_treat"] != 0].copy()
    with pytest.raises(ValueError):
        sp.ddd_heterogeneous(
            treated_only,
            y="y",
            unit="unit",
            time="time",
            cohort="first_treat",
            subgroup="affected",
            n_boot=20,
            seed=0,
        )


# ── Time-varying covariates ──────────────────────────────────────── #


def _tv_panel(seed=1, n_units=120, n_periods=10):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 5, 7][u % 3]
        fe = rng.normal()
        x0 = rng.normal()
        for t in range(1, n_periods + 1):
            treated_now = (g > 0) and (t >= g)
            te = 2.0 if treated_now else 0.0
            y = fe + 0.3 * t + 0.5 * x0 + te + rng.normal(0, 0.4)
            rows.append(
                {"unit": u, "time": t, "y": y, "first_treat": g, "x": x0 + 0.05 * t}
            )
    return pd.DataFrame(rows)


def test_timevarying_recovers():
    df = _tv_panel()
    r = sp.did_timevarying_covariates(
        df,
        y="y",
        unit="unit",
        time="time",
        cohort="first_treat",
        covariates=["x"],
        n_boot=120,
        seed=0,
    )
    assert abs(r.estimate - 2.0) < 0.6
    assert np.isfinite(r.se) and r.se > 0
    lo, hi = r.ci
    assert lo <= r.estimate <= hi


def test_timevarying_missing_covariate_raises():
    df = _tv_panel()
    with pytest.raises(ValueError):
        sp.did_timevarying_covariates(
            df,
            y="y",
            unit="unit",
            time="time",
            cohort="first_treat",
            covariates=["nope"],
        )


def test_timevarying_no_never_treated_raises():
    df = _tv_panel()
    treated_only = df[df["first_treat"] != 0].copy()
    with pytest.raises(ValueError):
        sp.did_timevarying_covariates(
            treated_only,
            y="y",
            unit="unit",
            time="time",
            cohort="first_treat",
            covariates=["x"],
            n_boot=20,
            seed=0,
        )


# ── Misclassified treatment ──────────────────────────────────────── #


def _mc_panel(seed=2, n_units=120, n_periods=10):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 5, 7][u % 3]
        fe = rng.normal()
        for t in range(1, n_periods + 1):
            treated_now = (g > 0) and (t >= g)
            te = 2.0 if treated_now else 0.0
            y = fe + 0.3 * t + te + rng.normal(0, 0.4)
            rows.append({"unit": u, "time": t, "y": y, "first_treat": g})
    return pd.DataFrame(rows)


def test_misclassified_correction_and_anticipation():
    df = _mc_panel()
    r = sp.did_misclassified(
        df,
        y="y",
        treat="first_treat",
        time="time",
        id="unit",
        pi_misclass=0.1,
        anticipation_periods=1,
    )
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.se) and r.se > 0
    # correction inflates the naive ATT; should stay positive & near +2-ish
    assert r.estimate > 1.0


def test_misclassified_bad_pi_raises():
    df = _mc_panel()
    with pytest.raises(ValueError):
        sp.did_misclassified(
            df, y="y", treat="first_treat", time="time", id="unit", pi_misclass=0.6
        )


def test_misclassified_bad_anticipation_raises():
    df = _mc_panel()
    with pytest.raises(ValueError):
        sp.did_misclassified(
            df,
            y="y",
            treat="first_treat",
            time="time",
            id="unit",
            anticipation_periods=-1,
        )


def test_misclassified_no_cohorts_raises():
    # All-zero treat encoding -> no treated cohorts.
    df = _mc_panel()
    df = df.assign(first_treat=0)
    with pytest.raises(Exception):
        sp.did_misclassified(df, y="y", treat="first_treat", time="time", id="unit")


def test_misclassified_no_controls_raises():
    # Drop never-treated -> no control units.
    df = _mc_panel()
    df = df[df["first_treat"] != 0].copy()
    with pytest.raises(Exception):
        sp.did_misclassified(df, y="y", treat="first_treat", time="time", id="unit")
