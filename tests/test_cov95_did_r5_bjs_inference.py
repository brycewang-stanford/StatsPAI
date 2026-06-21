"""Coverage round-5 for statspai.did.bjs_inference (bjs_pretrend_joint).

Cluster-bootstrap joint pre-trend Wald test for the BJS imputation
estimator.  Covers horizon inference, the K=1 scalar-covariance branch,
and the up-front validation errors (missing event study, no pre-period
horizons, unknown cluster column).

The "not enough reps succeeded" RuntimeError (lines 156-167) and the
"unexpected exception in a bootstrap rep" path (140-144) are defensive
guards against pathological resamples / genuine BJS bugs; they cannot be
forced reliably without mocking the numeric refit, so they are left to a
dedicated parity harness.  np.linalg.LinAlgError pinv fallback (177-178)
is likewise an unreachable defensive double-guard after 1e-10 ridge.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from statspai.did.did_imputation import did_imputation
from statspai.did.bjs_inference import bjs_pretrend_joint


def make_panel(seed=0, cohorts=(4, 6, 0), n_per=20, T=8):
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for g in cohorts:
        for _ in range(n_per):
            ufe = rng.normal()
            for t in range(1, T + 1):
                te = max(0, t - g + 1) if g > 0 else 0
                rows.append(
                    {
                        "i": uid,
                        "t": t,
                        "y": ufe + 0.3 * t + te + rng.normal() * 0.5,
                        "g": g,
                    }
                )
            uid += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    df = make_panel()
    r = did_imputation(
        df, y="y", group="i", time="t", first_treat="g", horizon=list(range(-3, 4))
    )
    return df, r


def test_joint_pretrend_basic(fitted):
    df, r = fitted
    out = bjs_pretrend_joint(
        r, df, y="y", group="i", time="t", first_treat="g", n_boot=40, seed=1
    )
    assert set(out) == {"statistic", "df", "pvalue", "method", "n_boot", "pre_cov"}
    assert out["df"] >= 1
    assert 0.0 <= out["pvalue"] <= 1.0
    assert np.array(out["pre_cov"]).shape == (out["df"], out["df"])


def test_joint_pretrend_horizon_inferred(fitted):
    df, r = fitted
    # horizon=None -> inferred from the event-study table (line 94)
    out = bjs_pretrend_joint(
        r,
        df,
        y="y",
        group="i",
        time="t",
        first_treat="g",
        horizon=None,
        n_boot=30,
        seed=2,
    )
    assert out["df"] >= 1


def test_joint_pretrend_single_pre_period_scalar_cov():
    # One pre-period -> K=1 -> np.cov returns a 0-d scalar -> reshape
    # branch (line 172).
    df = make_panel()
    r = did_imputation(
        df, y="y", group="i", time="t", first_treat="g", horizon=[-1, 0, 1]
    )
    out = bjs_pretrend_joint(
        r, df, y="y", group="i", time="t", first_treat="g", n_boot=30, seed=3
    )
    assert out["df"] == 1
    assert np.array(out["pre_cov"]).shape == (1, 1)


def test_missing_event_study_raises(fitted):
    df, _ = fitted

    class _NoES:
        model_info = {}

    with pytest.raises(ValueError, match="missing an event-study"):
        bjs_pretrend_joint(_NoES(), df, y="y", group="i", time="t", first_treat="g")


def test_no_pre_horizons_raises():
    df = make_panel()
    # post-only horizon -> no pre rows in the event study (line 97-98)
    r = did_imputation(
        df, y="y", group="i", time="t", first_treat="g", horizon=[0, 1, 2]
    )
    with pytest.raises(ValueError, match="No pre-treatment horizons"):
        bjs_pretrend_joint(
            r, df, y="y", group="i", time="t", first_treat="g", n_boot=10, seed=1
        )


def test_unknown_cluster_column_raises(fitted):
    df, r = fitted
    with pytest.raises(ValueError, match="Cluster column"):
        bjs_pretrend_joint(
            r,
            df,
            y="y",
            group="i",
            time="t",
            first_treat="g",
            cluster="nonexistent",
            n_boot=10,
        )
