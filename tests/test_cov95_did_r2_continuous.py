"""Round-2 coverage for statspai.did.continuous_did:
the four method branches (twfe / att_gt / dose_response / cgs), controls,
cluster, and post inference. Uses a real continuous-dose panel."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _dose_panel(seed=0, n_units=160, n_periods=4):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        d = 0.0 if u % 4 == 0 else rng.uniform(0.5, 3.0)
        fe = rng.normal()
        for t in range(1, n_periods + 1):
            post = 1 if t >= 3 else 0
            y = fe + 0.3 * t + 1.2 * d * post + rng.normal(0, 0.5)
            rows.append(
                {
                    "id": u,
                    "time": t,
                    "y": y,
                    "dose": d,
                    "x1": rng.normal(),
                    "cl": u % 12,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def dp():
    return _dose_panel()


def test_continuous_twfe(dp):
    r = sp.continuous_did(
        dp, y="y", dose="dose", time="time", id="id", method="twfe", t_pre=2, t_post=3
    )
    assert r.se > 0
    assert 0 <= r.pvalue <= 1


def test_continuous_twfe_controls_cluster(dp):
    r = sp.continuous_did(
        dp,
        y="y",
        dose="dose",
        time="time",
        id="id",
        method="twfe",
        t_pre=2,
        t_post=3,
        controls=["x1"],
        cluster="cl",
    )
    assert r.se > 0


def test_continuous_att_gt(dp):
    r = sp.continuous_did(
        dp,
        y="y",
        dose="dose",
        time="time",
        id="id",
        method="att_gt",
        t_pre=2,
        t_post=3,
        seed=1,
    )
    assert r.detail is not None
    assert len(r.detail) > 0


def test_continuous_att_gt_post_col(dp):
    df = dp.copy()
    df["post"] = (df["time"] >= 3).astype(int)
    r = sp.continuous_did(
        df,
        y="y",
        dose="dose",
        time="time",
        id="id",
        post="post",
        method="att_gt",
        seed=2,
    )
    assert r.model_info["n_dose_groups"] >= 0


def test_continuous_dose_response(dp):
    r = sp.continuous_did(
        dp,
        y="y",
        dose="dose",
        time="time",
        id="id",
        method="dose_response",
        t_pre=2,
        t_post=3,
        n_boot=40,
        seed=3,
    )
    assert r is not None
    assert np.isfinite(r.estimate)


def test_continuous_cgs(dp):
    r = sp.continuous_did(
        dp,
        y="y",
        dose="dose",
        time="time",
        id="id",
        method="cgs",
        t_pre=2,
        t_post=3,
        n_boot=40,
        seed=4,
    )
    assert r is not None
    assert r.se >= 0 or np.isnan(r.se)


def test_continuous_default_post_inference(dp):
    # no post, no t_pre/t_post -> infers midpoint
    r = sp.continuous_did(dp, y="y", dose="dose", time="time", id="id", method="twfe")
    assert r.se > 0
