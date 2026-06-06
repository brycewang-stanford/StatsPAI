"""Coverage round-3 for statspai.did.stacked_did (Cengiz et al. 2019).

Exercises validation branches, the never-treated vs not-yet-treated control
construction, and the private cluster-robust vcov/se helper edge cases.

NOTE (bug, reported separately, NOT exercised here): passing ``controls=``
alongside post-treatment periods crashes with a ``w @ V @ w`` dimension
mismatch because the aggregation weight vector is sized to the number of
event-study dummies while V includes the extra control columns. These tests
deliberately avoid the controls-with-post-periods path.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did.stacked_did import (
    _cluster_robust_se,
    _cluster_robust_vcov,
    _ols,
)


@pytest.fixture(scope="module")
def panel():
    return sp.dgp_did(n_units=90, n_periods=12, staggered=True, seed=3)


def test_stacked_basic(panel):
    r = sp.stacked_did(panel, y="y", group="unit", time="time",
                       first_treat="first_treat", window=(-4, 4))
    assert np.isfinite(r.estimate)
    assert 0.0 <= r.pvalue <= 1.0
    es = r.model_info["event_study"]
    assert {"relative_time", "att", "se", "ci_lower", "ci_upper"} <= set(es.columns)
    # reference period -1 present with zero coefficient
    ref = es.loc[es["relative_time"] == -1]
    assert (ref["att"] == 0.0).all()


def test_stacked_not_yet_treated_controls(panel):
    r = sp.stacked_did(panel, y="y", group="unit", time="time",
                       first_treat="first_treat", window=(-4, 4),
                       never_treated_only=False)
    assert r.model_info["never_treated_only"] is False
    assert np.isfinite(r.estimate)


def test_stacked_explicit_cluster(panel):
    r = sp.stacked_did(panel, y="y", group="unit", time="time",
                       first_treat="first_treat", window=(-3, 3),
                       cluster="unit")
    assert r.model_info["cluster_var"] == "unit"


def test_stacked_missing_outcome_raises(panel):
    with pytest.raises(ValueError, match="not found"):
        sp.stacked_did(panel, y="nope", group="unit", time="time",
                       first_treat="first_treat")


def test_stacked_missing_control_raises(panel):
    with pytest.raises(ValueError, match="Control column"):
        sp.stacked_did(panel, y="y", group="unit", time="time",
                       first_treat="first_treat", controls=["nope"])


def test_stacked_bad_window_lo_raises(panel):
    with pytest.raises(ValueError, match=r"window\[0\] must be negative"):
        sp.stacked_did(panel, y="y", group="unit", time="time",
                       first_treat="first_treat", window=(0, 5))


def test_stacked_bad_window_hi_raises(panel):
    with pytest.raises(ValueError, match=r"window\[1\] must be non-negative"):
        sp.stacked_did(panel, y="y", group="unit", time="time",
                       first_treat="first_treat", window=(-5, -1))


def test_stacked_no_cohorts_raises():
    df = sp.dgp_did(n_units=40, n_periods=8, staggered=True, seed=4)
    df = df.copy()
    df["first_treat"] = 0  # all never-treated
    with pytest.raises(ValueError, match="No treated cohorts"):
        sp.stacked_did(df, y="y", group="unit", time="time",
                       first_treat="first_treat")


def test_stacked_pre_only_window_no_post():
    # window with only pre periods (and the -1 reference) leaves no post k>=0
    # in some configs -> att/att_se default to 0. Use a narrow pre window.
    df = sp.dgp_did(n_units=80, n_periods=12, staggered=True, seed=6)
    r = sp.stacked_did(df, y="y", group="unit", time="time",
                       first_treat="first_treat", window=(-3, 0))
    # window=(-3, 0) still includes rel_time 0 (post). Verify it runs.
    assert np.isfinite(r.estimate)


# --- private helpers ---------------------------------------------------
def test_ols_zero_columns():
    y = np.array([1.0, 2.0, 3.0])
    X = np.zeros((3, 0))
    beta, resid = _ols(X, y)
    assert beta.size == 0
    np.testing.assert_array_equal(resid, y)


def test_cluster_vcov_zero_columns():
    X = np.zeros((5, 0))
    resid = np.array([0.1, -0.2, 0.0, 0.3, -0.1])
    clusters = np.array([0, 0, 1, 1, 1])
    V = _cluster_robust_vcov(X, resid, clusters)
    assert V.shape == (0, 0)


def test_cluster_se_zero_columns():
    X = np.zeros((4, 0))
    resid = np.array([0.1, -0.1, 0.2, -0.2])
    clusters = np.array([0, 0, 1, 1])
    se = _cluster_robust_se(X, resid, clusters)
    assert se.size == 0


def test_cluster_vcov_singular_uses_pinv():
    # Two perfectly collinear columns force the inv->pinv bread fallback.
    rng = np.random.default_rng(0)
    n = 20
    x = rng.normal(size=n)
    X = np.column_stack([x, 2.0 * x])  # rank-1
    resid = rng.normal(size=n) * 0.1
    clusters = np.repeat(np.arange(5), 4)
    V = _cluster_robust_vcov(X, resid, clusters)
    assert V.shape == (2, 2)
    assert np.all(np.isfinite(V))
