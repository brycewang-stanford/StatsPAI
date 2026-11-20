"""``result.report()`` under weak overlap, and data recovered from the call."""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _obs(seed, sep, n=4000):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    p = 1 / (1 + np.exp(-sep * x1))
    t = (rng.random(n) < p).astype(int)
    y = 1.2 * t + x1 + x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


def _aipw(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.aipw(df, y="y", treat="t", covariates=["x1", "x2"])


def test_good_overlap_reports_the_fit():
    r = _aipw(_obs(0, 1.0))
    rep = r.report()
    assert rep["status"] == "conventional"
    assert rep["estimate"] == pytest.approx(r.estimate)


def test_weak_overlap_is_a_caution_with_the_trimmed_effect_alongside():
    df = _obs(0, 6.0)
    r = _aipw(df)
    rep = r.report()
    assert rep["status"] == "caution"
    assert rep["extreme_propensity_share"] > 0.05
    alt = rep["alternatives"]["overlap_trimmed"]
    ps = np.asarray(r.model_info["_pscore"])
    keep = (ps >= 0.1) & (ps <= 0.9)
    ref = _aipw(df.loc[keep])
    assert alt["n_kept"] == int(keep.sum())
    assert alt["estimate"] == pytest.approx(ref.estimate, rel=1e-10)
    # the headline is still the fit's own number
    assert rep["estimate"] == pytest.approx(r.estimate)


def test_few_cluster_report_finds_the_frame_of_the_call():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(8), 60)
    df = pd.DataFrame(
        {
            "school": g,
            "treat": (g < 4).astype(int),
            "y": rng.normal(size=8)[g] + rng.normal(size=480),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.regress("y ~ treat", data=df, cluster="school")
    # no data= needed while the frame of the call is alive
    assert r.report()["ci_method"] == "wild_cluster_bootstrap"
    assert r.report()["ci"] == pytest.approx(r.report(data=df)["ci"])
