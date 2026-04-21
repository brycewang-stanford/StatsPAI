"""Tests for Gardner (2021) two-stage DID estimator."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _synthetic_staggered_panel(seed=0, N=200, T=10, tau=2.0):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)
    cohort = rng.choice([3, 5, 7, np.inf], size=N, p=[0.25, 0.25, 0.25, 0.25])
    first_treat = np.repeat(cohort, T)
    ue = rng.normal(0, 0.3, N)
    te = rng.normal(0, 0.3, T)
    unit_fe = np.repeat(ue, T)
    time_fe = np.tile(te, N)
    treated_now = (time >= first_treat) & np.isfinite(first_treat)
    y = 1.0 + unit_fe + time_fe + tau * treated_now + rng.normal(0, 0.3, N * T)
    return pd.DataFrame({
        "id": unit, "t": time, "y": y,
        "first_treat": np.where(np.isfinite(first_treat), first_treat, 0.0),
    })


def test_gardner_att_recovers_truth():
    df = _synthetic_staggered_panel(seed=0, N=300)
    r = sp.gardner_did(df, y="y", group="id", time="t", first_treat="first_treat")
    assert abs(r.estimate - 2.0) < 0.15, f"ATT {r.estimate} off from true 2.0"
    assert r.se > 0
    assert r.ci[0] < r.estimate < r.ci[1]
    assert r.estimand == "ATT"
    assert r.n_obs > 0


def test_gardner_agrees_with_bjs_imputation():
    df = _synthetic_staggered_panel(seed=1, N=300)
    r_g = sp.gardner_did(df, y="y", group="id", time="t", first_treat="first_treat")
    r_b = sp.did_imputation(df, y="y", group="id", time="t", first_treat="first_treat")
    # Gardner and BJS target the same estimand → should be close
    assert abs(r_g.estimate - r_b.estimate) / max(abs(r_b.estimate), 1e-6) < 0.10


def test_gardner_event_study_support():
    df = _synthetic_staggered_panel(seed=2, N=250)
    r = sp.gardner_did(
        df, y="y", group="id", time="t", first_treat="first_treat",
        event_study=True, horizon=[-2, -1, 0, 1, 2],
    )
    es = r.model_info["event_study"]
    assert set(es["horizon"]) == {"D_k-2", "D_k-1", "D_k+0", "D_k+1", "D_k+2"}
    # Post-treatment coefs should be ≈ 2.0, pre-treatment ≈ 0
    post = [es["coef"][k] for k in ("D_k+0", "D_k+1", "D_k+2")]
    # After the v1.5.1 reference-category fix, pre-trend should be ~0 and
    # post-treatment should closely track the truth.
    assert np.mean(post) > 1.6
    for k in ("D_k-2", "D_k-1"):
        assert abs(es["coef"][k]) < 0.3  # pre-trend tight relative to ATT


def test_gardner_alias_did_2stage():
    df = _synthetic_staggered_panel(seed=3, N=150)
    r1 = sp.gardner_did(df, y="y", group="id", time="t", first_treat="first_treat")
    r2 = sp.did_2stage(df, y="y", group="id", time="t", first_treat="first_treat")
    assert r1.estimate == pytest.approx(r2.estimate, abs=1e-10)


def test_gardner_in_registry():
    fns = sp.list_functions()
    assert "gardner_did" in fns
    assert "did_2stage" in fns


def test_gardner_raises_on_missing_column():
    df = _synthetic_staggered_panel(seed=4, N=100)
    with pytest.raises(ValueError):
        sp.gardner_did(df, y="nonexistent", group="id", time="t",
                        first_treat="first_treat")


def test_gardner_cluster_parameter():
    df = _synthetic_staggered_panel(seed=5, N=200)
    # Explicit cluster=group should match default behaviour
    r_default = sp.gardner_did(df, y="y", group="id", time="t",
                                first_treat="first_treat")
    r_cluster = sp.gardner_did(df, y="y", group="id", time="t",
                                first_treat="first_treat", cluster="id")
    assert r_default.estimate == pytest.approx(r_cluster.estimate)
    assert r_default.se == pytest.approx(r_cluster.se)
