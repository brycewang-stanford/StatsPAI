"""Coverage round-3 for statspai.did.did_bcf (Forests for Differences).

Covers the no-covariate degenerate path (mean-difference ATT + cluster
bootstrap SE + per-cohort CATT) and the BCF-backed covariate path. Real
staggered panel; assertions check finite ATT, p-value in [0, 1], and the
per-cohort CATT dictionary structure.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def panel():
    df = sp.dgp_did(n_units=70, n_periods=8, staggered=True, seed=2).copy()
    df["first_treat"] = df["first_treat"].fillna(0)
    rng = np.random.default_rng(1)
    df["x1"] = rng.normal(size=len(df))
    df["x2"] = rng.normal(size=len(df))
    return df


def test_did_bcf_no_covariates(panel):
    r = sp.did_bcf(panel, y="y", treat="first_treat", time="time", id="unit")
    assert np.isfinite(r.estimate)
    assert 0.0 <= r.pvalue <= 1.0
    assert r.se > 0
    catt = r.model_info["catt_by_cohort"]
    assert isinstance(catt, dict) and len(catt) > 0
    # cohort keys are floats; values finite
    assert all(np.isfinite(v) for v in catt.values())
    assert r.model_info["n_covariates"] == 0


def test_did_bcf_with_covariates(panel):
    r = sp.did_bcf(
        panel,
        y="y",
        treat="first_treat",
        time="time",
        id="unit",
        covariates=["x1", "x2"],
        n_trees=20,
        seed=3,
    )
    assert np.isfinite(r.estimate)
    assert 0.0 <= r.pvalue <= 1.0
    assert r.model_info["n_covariates"] == 2


def test_did_bcf_ci_brackets_estimate(panel):
    r = sp.did_bcf(panel, y="y", treat="first_treat", time="time", id="unit")
    lo, hi = r.ci
    assert lo <= r.estimate <= hi
