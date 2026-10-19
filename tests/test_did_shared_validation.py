"""Shared staggered-panel checks across the DiD estimators (2026-10-19 audit).

Each test asserts a true proposition: an exact identity between two fits,
or a refusal / warning that the input calls for. The reference-number
checks against R ``fixest`` and Stata ``did_imputation`` live in
``tests/reference_parity/test_did_validation_parity.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did._validate import RepeatedUnitPeriodWarning
from statspai.exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
)


def _panel(n=300, T=10, seed=1, cohorts=(0, 4, 6, 8)):
    rng = np.random.default_rng(seed)
    g = rng.choice(list(cohorts), size=n)
    a = rng.normal(size=n)
    ids = np.repeat(np.arange(n), T)
    tt = np.tile(np.arange(1, T + 1), n)
    gg = np.repeat(g, T)
    d = ((gg > 0) & (tt >= gg)).astype(int)
    eff = np.where(d == 1, 2.0 * (1 + 0.2 * (tt - gg)), 0.0)
    y = np.repeat(a, T) + 0.3 * tt + eff + rng.normal(size=n * T)
    return pd.DataFrame(dict(id=ids, t=tt, g=gg, y=y))


FITS = {
    "callaway_santanna": lambda d: sp.callaway_santanna(d, "y", "g", "t", "id"),
    "sun_abraham": lambda d: sp.sun_abraham(d, "y", "g", "t", "id"),
    "did_imputation": lambda d: sp.did_imputation(d, "y", "id", "t", "g"),
    "gardner_did": lambda d: sp.gardner_did(d, "y", "id", "t", "g"),
    "stacked_did": lambda d: sp.stacked_did(d, "y", "id", "t", "g"),
    "etwfe": lambda d: sp.etwfe(d, "y", "id", "t", "g"),
}


def _quiet(fn, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(d)


# ---------------------------------------------------------------- cohort


@pytest.mark.parametrize("name", sorted(FITS))
def test_cohort_varying_within_unit_is_refused(name):
    df = _panel()
    u = df.loc[df.g == 0, "id"].iloc[0]
    df.loc[(df.id == u) & (df.t == 1), "g"] = 4  # one row now says cohort 4
    with pytest.raises(MethodIncompatibility, match="more than one value"):
        _quiet(FITS[name], df)


@pytest.mark.parametrize("name", sorted(FITS))
def test_never_treated_codes_do_not_count_as_varying(name):
    # NaN in one row and 0 in another are both "never treated".
    df = _panel()
    u = df.loc[df.g == 0, "id"].iloc[0]
    df["g"] = df["g"].astype(float)
    df.loc[(df.id == u) & (df.t <= 3), "g"] = np.nan
    a = _quiet(FITS[name], df)
    b = _quiet(FITS[name], df.fillna({"g": 0}))
    assert a.estimate == pytest.approx(b.estimate, rel=1e-12, abs=1e-12)


# ------------------------------------------------------------ duplicates


@pytest.mark.parametrize(
    "name", ["sun_abraham", "did_imputation", "gardner_did", "stacked_did", "etwfe"]
)
def test_repeated_unit_period_rows_warn(name):
    df = _panel()
    dd = pd.concat([df, df[df.id < 5]], ignore_index=True)
    with pytest.warns(RepeatedUnitPeriodWarning, match="rows share"):
        FITS[name](dd)


def test_repeated_unit_period_rows_refused_by_cs():
    df = _panel()
    dd = pd.concat([df, df[df.id < 5]], ignore_index=True)
    with pytest.raises(MethodIncompatibility, match="share a"):
        _quiet(FITS["callaway_santanna"], dd)


# --------------------------------------------------------- always treated


def _with_always_treated():
    df = _panel(400, seed=1)
    ids = df.loc[df.g == 4, "id"].unique()[:20]
    df.loc[df.id.isin(ids), "g"] = 1  # treated from the first period
    return df


@pytest.mark.parametrize("name", ["sun_abraham", "stacked_did", "etwfe"])
def test_always_treated_cohort_dropped_equals_fit_without_it(name):
    df = _with_always_treated()
    with pytest.warns(UserWarning, match=r"dropped cohort\(s\) \[1\]"):
        r_full = FITS[name](df)
    r_drop = _quiet(FITS[name], df[df.g != 1])
    assert r_full.estimate == r_drop.estimate
    assert r_full.se == r_drop.se


def test_sun_abraham_always_treated_records_drop_and_finite_ses():
    df = _with_always_treated()
    r = _quiet(FITS["sun_abraham"], df)
    mi = r.model_info
    assert mi["dropped_unidentified_cohorts"] == [1]
    assert mi["n_units_dropped_unidentified"] == 20
    assert mi["n_rows_dropped_unidentified"] == 200
    # Before the fix every post-period SE was exactly 0 with p = 1.
    post = r.detail[r.detail.relative_time >= 0]
    assert (post["se"] > 1e-3 * post["att"].abs().max()).all()


def test_sun_abraham_unidentified_cohort_under_ref_p():
    # Cohort 4 is observed from t=1, so relative period -5 does not exist for
    # it: with ref_p=-5 it has no reference cell and is dropped.
    df = _panel()
    with pytest.warns(UserWarning, match=r"dropped cohort\(s\) \[4\]"):
        r = sp.sun_abraham(df, "y", "g", "t", "id", ref_p=-5)
    assert 4 not in r.model_info["cohorts"]


def test_sun_abraham_all_cohorts_unidentified_raises():
    df = _panel(cohorts=(0, 1))
    with pytest.raises(DataInsufficient, match="reference"):
        _quiet(FITS["sun_abraham"], df)


# ----------------------------------------------------------- few treated


@pytest.mark.parametrize(
    "name", ["sun_abraham", "did_imputation", "gardner_did", "stacked_did", "etwfe"]
)
def test_one_treated_cluster_warns(name):
    df = _panel()
    one = df.loc[df.g == 4, "id"].iloc[0]
    d1 = df[(df.g == 0) | (df.id == one)]
    with pytest.warns(AssumptionWarning, match="only 1 cluster"):
        FITS[name](d1)


# ------------------------------------------------------------ sun_abraham


def test_sun_abraham_no_never_treated_points_to_lastcohort():
    df = _panel(cohorts=(4, 6, 8))
    with pytest.raises(DataInsufficient) as exc:
        _quiet(FITS["sun_abraham"], df)
    assert "lastcohort" in exc.value.recovery_hint


def test_sun_abraham_headline_names_both_aggregations():
    df = _panel()
    r = _quiet(FITS["sun_abraham"], df)
    mi = r.model_info
    assert r.estimate == mi["att_event_time"]
    assert "equal-weight" in mi["headline"]
    assert f"{mi['att_fixest_att']:.6g}" in mi["other_aggregation"]
    r2 = _quiet(
        lambda d: sp.sun_abraham(d, "y", "g", "t", "id", aggregation="fixest_att"),
        df,
    )
    assert r2.estimate == mi["att_fixest_att"]
    assert "cell-weighted" in r2.model_info["headline"]
    assert "Other Aggregation" in str(r.summary())


def test_sun_abraham_anticipation_is_ref_p():
    df = _panel()
    a = _quiet(lambda d: sp.sun_abraham(d, "y", "g", "t", "id", anticipation=1), df)
    b = _quiet(lambda d: sp.sun_abraham(d, "y", "g", "t", "id", ref_p=-2), df)
    pd.testing.assert_frame_equal(a.detail, b.detail)
    # relative time -1 is an estimated anticipation effect, -2 the reference
    assert -1 in set(a.detail.relative_time) and -2 not in set(a.detail.relative_time)
    # ... and is not part of the pre-trend test
    assert a.model_info["pretrend_test"]["relative_times"] == [-7, -6, -5, -4, -3]


def test_sun_abraham_ref_p_inside_anticipation_window_refused():
    with pytest.raises(MethodIncompatibility, match="ref_p"):
        sp.sun_abraham(_panel(), "y", "g", "t", "id", anticipation=1, ref_p=-1)


def test_sun_abraham_rcs_equals_panel_with_cohort_level_units():
    # On repeated cross sections the cohort plays the unit: i=None equals a
    # fit whose "unit" is the cohort, clustered by row.
    df = _panel()
    df["row"] = np.arange(len(df))
    a = _quiet(lambda d: sp.sun_abraham(d, "y", "g", "t"), df)
    b = _quiet(lambda d: sp.sun_abraham(d, "y", "g", "t", "g", cluster="row"), df)
    np.testing.assert_allclose(a.detail["att"], b.detail["att"], rtol=1e-12)
    np.testing.assert_allclose(a.detail["se"], b.detail["se"], rtol=1e-10)
    assert a.model_info["se_type"].startswith("heteroskedasticity-robust")


# ---------------------------------------------------------- did_imputation


def test_did_imputation_shift_equals_shifted_cohort():
    df = _panel()
    a = _quiet(
        lambda d: sp.did_imputation(d, "y", "id", "t", "g", horizon=[0, 1, 2], shift=2),
        df,
    )
    d2 = df.assign(g=np.where(df.g > 0, df.g - 2, 0))
    b = _quiet(
        lambda d: sp.did_imputation(d, "y", "id", "t", "g", horizon=[0, 1, 2]), d2
    )
    assert a.estimate == pytest.approx(b.estimate, rel=1e-12)
    assert a.se == pytest.approx(b.se, rel=1e-12)
    assert a.model_info["shift"] == 2


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_did_imputation_shift_rejects_bad_values(bad):
    with pytest.raises(MethodIncompatibility, match="shift"):
        sp.did_imputation(_panel(), "y", "id", "t", "g", shift=bad)


# -------------------------------------------------------------------- fect


def test_fect_n_boot_without_bootstrap_warns():
    df = _panel(60, T=8, seed=3)
    df["d"] = ((df.g > 0) & (df.t >= df.g)).astype(int)
    with pytest.warns(UserWarning, match="n_boot=20 has no effect"):
        r = sp.fect(df, "y", "d", "id", "t", n_boot=20)
    assert r.se is None or not np.isfinite(r.se)


# ---------------------------------------------------------------------- CS


def test_cs_result_says_how_intervals_differ_from_r_did():
    df = _panel()
    r = _quiet(FITS["callaway_santanna"], df)
    assert "bstrap=TRUE, cband=TRUE" in r.model_info["inference_note"]
    r2 = _quiet(
        lambda d: sp.callaway_santanna(
            d,
            "y",
            "g",
            "t",
            "id",
            bstrap=True,
            cband=True,
            biters=50,
            random_state=0,
        ),
        df,
    )
    assert r2.model_info["inference_note"] is None
    agg = _quiet(lambda r_: sp.aggte(r_, type="dynamic"), r)
    assert "bstrap=TRUE" in agg.model_info["inference_note"]
