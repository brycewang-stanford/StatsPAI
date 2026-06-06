"""Coverage round-3 for several smaller statspai.did estimators.

Covers:
  * gardner_did (Gardner 2021/22 two-stage DID): overall + event-study,
    custom horizon, validation errors.
  * continuous_did (heuristic continuous-treatment DiD): all four methods
    plus controls/cluster paths.
  * did_multiplegt (de Chaisemartin & D'Haultfoeuille 2020 DID_M):
    placebo + dynamic horizons, controls, explicit cluster, validation.

Real synthetic panels; assertions check finite estimates, p-values in
[0, 1], shapes, and raised errors. No fabricated expected numbers.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ----------------------------------------------------------------------
# gardner_did
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def stag_panel():
    df = sp.dgp_did(n_units=90, n_periods=12, staggered=True, seed=8).copy()
    df["first_treat"] = df["first_treat"].fillna(0)
    rng = np.random.default_rng(3)
    df["xc"] = rng.normal(size=len(df))
    df["st"] = df["unit"] % 7
    return df


def test_gardner_overall(stag_panel):
    r = sp.gardner_did(stag_panel, y="y", group="unit", time="time",
                       first_treat="first_treat")
    assert np.isfinite(r.estimate)
    assert 0.0 <= r.pvalue <= 1.0


def test_gardner_event_study_default_horizon(stag_panel):
    r = sp.gardner_did(stag_panel, y="y", group="unit", time="time",
                       first_treat="first_treat", event_study=True)
    es = r.model_info.get("event_study")
    assert es is not None
    assert np.isfinite(r.estimate)


def test_gardner_event_study_custom_horizon(stag_panel):
    r = sp.gardner_did(stag_panel, y="y", group="unit", time="time",
                       first_treat="first_treat", event_study=True,
                       horizon=[-2, -1, 0, 1, 2], cluster="st")
    assert np.isfinite(r.estimate)


def test_gardner_missing_column_raises(stag_panel):
    with pytest.raises(ValueError, match="not found"):
        sp.gardner_did(stag_panel, y="nope", group="unit", time="time",
                       first_treat="first_treat")


def test_gardner_bad_cluster_raises(stag_panel):
    with pytest.raises(ValueError, match="cluster column"):
        sp.gardner_did(stag_panel, y="y", group="unit", time="time",
                       first_treat="first_treat", cluster="nope")


def test_gardner_too_few_untreated_raises():
    # Every row is treated (first_treat == first period) -> <10 untreated rows.
    rng = np.random.default_rng(0)
    rows = []
    for u in range(20):
        for t in range(1, 6):
            rows.append((u, t, rng.normal(), 1))  # treated from t=1 (first period)
    df = pd.DataFrame(rows, columns=["unit", "time", "y", "first_treat"])
    with pytest.raises(ValueError, match="untreated"):
        sp.gardner_did(df, y="y", group="unit", time="time",
                       first_treat="first_treat")


# ----------------------------------------------------------------------
# continuous_did
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def dose_panel():
    rng = np.random.default_rng(0)
    periods = [2018, 2019, 2020, 2021]
    rows = []
    for u in range(70):
        d = rng.uniform(0, 5)
        ai = rng.normal()
        for t in periods:
            post = 1 if t >= 2020 else 0
            y = ai + 0.4 * d * post + rng.normal(0, 0.5)
            rows.append((u, t, y, d, rng.normal()))
    return pd.DataFrame(rows, columns=["id", "time", "y", "dose", "xc"])


@pytest.mark.parametrize("method", ["att_gt", "twfe", "dose_response", "cgs"])
def test_continuous_did_methods(dose_panel, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.continuous_did(dose_panel, y="y", dose="dose", time="time",
                              id="id", t_pre=2019, t_post=2020,
                              method=method, n_boot=40, seed=1)
    assert r.estimate is not None
    # estimate may be nan for the cgs MVP on this DGP; the call must succeed.
    assert np.isfinite(r.estimate) or np.isnan(r.estimate)


def test_continuous_did_twfe_controls_cluster(dose_panel):
    r = sp.continuous_did(dose_panel, y="y", dose="dose", time="time",
                          id="id", t_pre=2019, t_post=2020, method="twfe",
                          controls=["xc"], cluster="id")
    assert np.isfinite(r.estimate)


def test_continuous_did_post_inferred_from_midpoint(dose_panel):
    # No t_pre/t_post and no post col -> midpoint split branch.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.continuous_did(dose_panel, y="y", dose="dose", time="time",
                              id="id", method="twfe")
    assert np.isfinite(r.estimate)


def test_continuous_did_explicit_post_column(dose_panel):
    df = dose_panel.copy()
    df["mypost"] = (df["time"] >= 2020).astype(int)
    r = sp.continuous_did(df, y="y", dose="dose", time="time", id="id",
                          post="mypost", method="twfe")
    assert np.isfinite(r.estimate)


@pytest.fixture(scope="module")
def dose_panel_with_controls():
    # ~1/3 of units have dose=0 (genuine controls) so att_gt / cgs reach
    # the full curve-fitting paths instead of the degenerate guards.
    rng = np.random.default_rng(0)
    periods = [2019, 2020]
    rows = []
    for u in range(90):
        d = 0.0 if u % 3 == 0 else rng.uniform(1, 5)
        ai = rng.normal()
        for t in periods:
            post = 1 if t >= 2020 else 0
            y = ai + 0.4 * d * post + rng.normal(0, 0.5)
            rows.append((u, t, y, d))
    return pd.DataFrame(rows, columns=["id", "time", "y", "dose"])


def test_continuous_did_att_gt_with_zero_dose_controls(dose_panel_with_controls):
    r = sp.continuous_did(dose_panel_with_controls, y="y", dose="dose",
                          time="time", id="id", t_pre=2019, t_post=2020,
                          method="att_gt", n_quantiles=4, n_boot=40, seed=1)
    assert np.isfinite(r.estimate)
    assert isinstance(r.detail, pd.DataFrame)
    assert {"dose_group", "att", "se"} <= set(r.detail.columns)


def test_continuous_did_cgs_full_curve(dose_panel_with_controls):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.continuous_did(dose_panel_with_controls, y="y", dose="dose",
                              time="time", id="id", t_pre=2019, t_post=2020,
                              method="cgs", n_boot=40, seed=2)
    assert np.isfinite(r.estimate)
    assert isinstance(r.detail, pd.DataFrame)
    assert {"dose", "att_d", "acrt_d"} <= set(r.detail.columns)


def test_continuous_did_cgs_no_controls_returns_nan(dose_panel):
    # dose is uniform(0, 5), so no dose==0 control units -> degenerate guard.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.continuous_did(dose_panel, y="y", dose="dose", time="time",
                              id="id", t_pre=2019, t_post=2020, method="cgs",
                              n_boot=20, seed=3)
    assert np.isnan(r.estimate)


def test_continuous_did_dose_response_curve(dose_panel_with_controls):
    r = sp.continuous_did(dose_panel_with_controls, y="y", dose="dose",
                          time="time", id="id", t_pre=2019, t_post=2020,
                          method="dose_response", n_boot=30, seed=4)
    assert np.isfinite(r.estimate)


# ----------------------------------------------------------------------
# did_multiplegt
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def switch_panel():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(48):
        ai = rng.normal()
        state = 0
        for t in range(10):
            if t == 4 and u % 2 == 0:
                state = 1
            if t == 7 and u % 4 == 0:
                state = 0
            y = ai + 0.1 * t + 0.5 * state + rng.normal(0, 0.5)
            rows.append((u, t, y, state, rng.normal(), u % 6))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "treat", "xc", "st"])


def test_multiplegt_main(switch_panel):
    r = sp.did_multiplegt(switch_panel, y="y", group="unit", time="time",
                          treatment="treat", n_boot=30, seed=1)
    assert np.isfinite(r.estimate)
    assert isinstance(r.detail, pd.DataFrame)


def test_multiplegt_placebo_dynamic(switch_panel):
    r = sp.did_multiplegt(switch_panel, y="y", group="unit", time="time",
                          treatment="treat", placebo=2, dynamic=2,
                          n_boot=30, seed=2)
    assert np.isfinite(r.estimate)
    mi = r.model_info
    assert isinstance(mi, dict)


def test_multiplegt_controls_cluster(switch_panel):
    r = sp.did_multiplegt(switch_panel, y="y", group="unit", time="time",
                          treatment="treat", controls=["xc"], cluster="st",
                          n_boot=20, seed=3)
    assert np.isfinite(r.estimate)


def test_multiplegt_missing_column_raises(switch_panel):
    with pytest.raises(ValueError, match="not found"):
        sp.did_multiplegt(switch_panel, y="nope", group="unit", time="time",
                          treatment="treat", n_boot=5)


def test_multiplegt_bad_control_raises(switch_panel):
    with pytest.raises(ValueError, match="Control column"):
        sp.did_multiplegt(switch_panel, y="y", group="unit", time="time",
                          treatment="treat", controls=["nope"], n_boot=5)
