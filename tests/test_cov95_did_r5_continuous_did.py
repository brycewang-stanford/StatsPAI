"""Coverage round-5 for statspai.did.continuous_did.

Heuristic continuous-treatment DiD.  Exercises all four method dispatch
branches (twfe / att_gt / dose_response / cgs), the TWFE controls +
clustered-SE path, the dose-quantile fallback when quantile edges
collapse, the dose-response local-linear fallback to linregress, and the
CGS MVP guards (auto pre/post periods, no controls, too few treated).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from statspai.did.continuous_did import continuous_did


def make_dose(seed=0, n=80, T=2, zero_frac=0.4, const_dose=False):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n):
        if const_dose:
            d = 2.0
        else:
            d = 0.0 if rng.random() < zero_frac else rng.uniform(0.5, 3)
        ufe = rng.normal()
        for t in range(1, T + 1):
            post = 1 if t == T else 0
            yv = ufe + 0.3 * t + 1.5 * d * post + rng.normal() * 0.3
            rows.append({"i": u, "t": t, "wage": yv, "dose": d})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", ["twfe", "att_gt", "dose_response", "cgs"])
def test_method_dispatch(method):
    df = make_dose()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = continuous_did(
            df,
            y="wage",
            dose="dose",
            time="t",
            id="i",
            t_pre=1,
            t_post=2,
            method=method,
            n_boot=30,
            seed=1,
        )
    assert r is not None
    assert np.isfinite(r.estimate)


def test_twfe_with_controls_and_cluster():
    df = make_dose()
    df["c1"] = np.random.default_rng(2).normal(size=len(df))
    r = continuous_did(
        df,
        y="wage",
        dose="dose",
        time="t",
        id="i",
        method="twfe",
        controls=["c1"],
        cluster="i",
    )
    assert np.isfinite(r.estimate)
    assert r.se >= 0


def test_post_inferred_from_midpoint():
    # post=None and no t_pre/t_post -> midpoint split (line 128-130)
    df = make_dose(T=4)
    r = continuous_did(df, y="wage", dose="dose", time="t", id="i", method="twfe")
    assert np.isfinite(r.estimate)


def test_att_gt_quantile_edges_collapse():
    # constant positive dose -> quantile edges collapse to <2 -> fallback
    # (lines 252-253). Mix in a few dose=0 controls so the DID is defined.
    df = make_dose(const_dose=True, zero_frac=0.0)
    ctrl = make_dose(n=20, zero_frac=1.0, seed=9)
    ctrl["i"] += 1000
    df = pd.concat([df, ctrl], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = continuous_did(
            df,
            y="wage",
            dose="dose",
            time="t",
            id="i",
            method="att_gt",
            n_quantiles=5,
            n_boot=20,
            seed=3,
        )
    assert r is not None


def test_dose_response_fallback_to_linregress():
    # Very few units -> lpoly likely raises -> linregress fallback (376-391).
    rng = np.random.default_rng(0)
    rows = []
    for u in range(6):
        d = rng.uniform(0, 2)
        ufe = rng.normal()
        for t in (1, 2):
            rows.append(
                {
                    "i": u,
                    "t": t,
                    "wage": ufe + (1.5 * d if t == 2 else 0) + rng.normal() * 0.1,
                    "dose": d,
                }
            )
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = continuous_did(
            df,
            y="wage",
            dose="dose",
            time="t",
            id="i",
            method="dose_response",
            t_pre=1,
            t_post=2,
            n_boot=10,
            seed=1,
        )
    assert r is not None
    assert "Dose-Response" in r.method or "Continuous DID" in r.method


def test_cgs_auto_periods():
    # t_pre / t_post None -> times[0] / times[-1] (lines 468, 472)
    df = make_dose()
    r = continuous_did(
        df, y="wage", dose="dose", time="t", id="i", method="cgs", n_boot=10, seed=1
    )
    assert np.isfinite(r.estimate)


def test_cgs_no_controls_warns():
    # All dose>0 -> no untreated control arm (lines 494-503)
    df = make_dose(zero_frac=0.0)
    r = continuous_did(
        df,
        y="wage",
        dose="dose",
        time="t",
        id="i",
        method="cgs",
        t_pre=1,
        t_post=2,
        n_boot=10,
        seed=1,
    )
    assert np.isnan(r.estimate)
    assert "control" in r.model_info.get("warning", "").lower()


def test_cgs_too_few_treated_warns():
    # Many controls, only 2 treated -> len(treated) < 3 (lines 504-513)
    rng = np.random.default_rng(0)
    rows = []
    for u in range(40):
        d = rng.uniform(0.5, 2) if u < 2 else 0.0
        ufe = rng.normal()
        for t in (1, 2):
            rows.append(
                {
                    "i": u,
                    "t": t,
                    "wage": ufe + (d if t == 2 else 0) + rng.normal() * 0.1,
                    "dose": d,
                }
            )
    df = pd.DataFrame(rows)
    r = continuous_did(
        df,
        y="wage",
        dose="dose",
        time="t",
        id="i",
        method="cgs",
        t_pre=1,
        t_post=2,
        n_boot=10,
        seed=1,
    )
    assert np.isnan(r.estimate)
    assert "treated" in r.model_info.get("warning", "").lower()


def test_cgs_empty_merge_returns_nan():
    # Disjoint pre/post unit sets -> merged empty (lines 482-488)
    rows = []
    for u in range(5):
        rows.append({"i": u, "t": 1, "wage": 1.0, "dose": 1.0})
    for u in range(5, 10):
        rows.append({"i": u, "t": 2, "wage": 2.0, "dose": 1.0})
    df = pd.DataFrame(rows)
    r = continuous_did(
        df,
        y="wage",
        dose="dose",
        time="t",
        id="i",
        method="cgs",
        t_pre=1,
        t_post=2,
        n_boot=5,
        seed=1,
    )
    assert np.isnan(r.estimate)
    assert r.n_obs == 0
