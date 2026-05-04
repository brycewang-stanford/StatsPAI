"""Branch coverage for ``statspai.did.did_imputation``.

Closes the validation / event-study / control-demeaning gaps surfaced by
the v1.12.x coverage audit. Existing files (``test_bjs_joint``,
``test_did_summary``, ``test_estimator_provenance_round2``) cover the
default ATT path; this file adds:

- All four ``ValueError`` guards (missing y/group/time/first_treat,
  missing control column, no treated obs, no untreated obs).
- The ``controls != []`` demeaning + beta-recovery branch.
- The horizon / event-study aggregation branch with the joint
  pre-trend chi-squared test.
- The internal ``_ols_coef`` helper's empty-X early return.
- The horizon SE helper's ``N_k == 0`` early return.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did.did_imputation import (
    _cluster_se_horizon,
    _ols_coef,
    did_imputation,
)


# --------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def staggered_panel():
    df = sp.dgp_did(n_units=80, n_periods=8, staggered=True, seed=11)
    rng = np.random.default_rng(0)
    df = df.copy()
    df["x"] = rng.normal(size=len(df))
    return df


# --------------------------------------------------------------------- #
#  Validation guards
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "missing_arg, kwargs",
    [
        ("y", {"y": "no_such_col"}),
        ("group", {"group": "no_such_col"}),
        ("time", {"time": "no_such_col"}),
        ("first_treat", {"first_treat": "no_such_col"}),
    ],
)
def test_did_imputation_validates_columns(staggered_panel, missing_arg, kwargs):
    base = dict(y="y", group="unit", time="time", first_treat="first_treat")
    base.update(kwargs)
    with pytest.raises(ValueError, match="not found in data"):
        did_imputation(staggered_panel, **base)


def test_did_imputation_validates_control_columns(staggered_panel):
    with pytest.raises(ValueError, match="Control column"):
        did_imputation(
            staggered_panel,
            y="y", group="unit", time="time", first_treat="first_treat",
            controls=["bogus"],
        )


def test_did_imputation_no_treated_raises(staggered_panel):
    df = staggered_panel.copy()
    df["first_treat"] = np.inf  # everyone never-treated
    with pytest.raises(ValueError, match="No treated observations"):
        did_imputation(
            df, y="y", group="unit", time="time", first_treat="first_treat",
        )


def test_did_imputation_no_untreated_raises():
    """Every obs is post-treatment (first_treat = -1) → no untreated."""
    rng = np.random.default_rng(0)
    n_units, n_per = 5, 4
    rows = []
    for u in range(n_units):
        for t in range(n_per):
            rows.append({
                "unit": u, "time": t, "y": rng.normal(),
                # first_treat smaller than every time observed → all treated
                "first_treat": -1.0,
            })
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="No untreated observations"):
        did_imputation(
            df, y="y", group="unit", time="time", first_treat="first_treat",
        )


# --------------------------------------------------------------------- #
#  Controls + horizon path
# --------------------------------------------------------------------- #


def test_did_imputation_with_controls_and_horizon(staggered_panel):
    res = did_imputation(
        staggered_panel,
        y="y", group="unit", time="time", first_treat="first_treat",
        controls=["x"], horizon=list(range(-3, 4)),
    )
    mi = res.model_info
    # Coefficient on the control surfaces in model_info
    assert "beta_controls" in mi and "x" in mi["beta_controls"]
    # Event study has rows for the requested horizons that exist in data
    assert "event_study" in mi and len(mi["event_study"]) > 0
    es = mi["event_study"]
    assert {"relative_time", "att", "se", "ci_lower", "ci_upper",
            "pvalue", "n_obs"} <= set(es.columns)
    # Pre-trend joint chi^2 test computed
    assert "pretrend_test" in mi
    pre = mi["pretrend_test"]
    assert {"statistic", "df", "pvalue"} <= set(pre.keys())


def test_did_imputation_explicit_cluster_overrides_default(staggered_panel):
    """Pass an explicit cluster column (must exist) and check it lands in
    model_info."""
    df = staggered_panel.copy()
    df["state"] = df["unit"] // 5  # group units into pseudo-states
    res = did_imputation(
        df, y="y", group="unit", time="time", first_treat="first_treat",
        cluster="state",
    )
    assert res.model_info["cluster_var"] == "state"


def test_did_imputation_horizon_event_study_orders_by_relative_time(
    staggered_panel,
):
    res = did_imputation(
        staggered_panel,
        y="y", group="unit", time="time", first_treat="first_treat",
        horizon=[-2, 0, 2, 4],
    )
    es = res.model_info["event_study"]
    # We requested horizons that exist in the panel; check the rows are
    # sorted ascending in relative_time (sort happens upstream).
    assert list(es["relative_time"]) == sorted(es["relative_time"])
    # Every row's CI brackets the point estimate
    assert (es["ci_lower"] <= es["att"]).all()
    assert (es["att"] <= es["ci_upper"]).all()


# --------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------- #


def test_ols_coef_with_empty_design_returns_empty():
    out = _ols_coef(np.empty((10, 0)), np.zeros(10))
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_ols_coef_recovers_known_beta():
    rng = np.random.default_rng(0)
    n, k = 200, 3
    X = rng.normal(size=(n, k))
    beta = np.array([0.5, -1.0, 2.0])
    y = X @ beta + 0.001 * rng.normal(size=n)
    out = _ols_coef(X, y)
    np.testing.assert_allclose(out, beta, atol=1e-2)


def test_cluster_se_horizon_zero_mask_returns_inf():
    """``mask_k.sum() == 0`` short-circuits to ``np.inf`` (no obs at horizon)."""
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "cluster": rng.integers(0, 5, size=n),
        "_uid": np.arange(n),
        "_tid": np.arange(n),
    })
    mask_k = np.zeros(n, dtype=bool)  # nothing at this horizon
    treated_mask = np.zeros(n, dtype=bool)
    out = _cluster_se_horizon(
        df=df,
        tau_hat=np.zeros(n),
        mask_k=mask_k,
        treated_mask=treated_mask,
        resid_untreated=np.zeros(n),
        cluster_col="cluster",
        uid_col="_uid",
        tid_col="_tid",
        alpha_hat=np.zeros(n),
        lambda_hat=np.zeros(n),
        unit_adj_count=np.ones(n),
        time_resid_count=np.ones(n),
        n_units=n,
        n_times=n,
    )
    assert np.isinf(out)


# --------------------------------------------------------------------- #
#  Citation registration is idempotent
# --------------------------------------------------------------------- #


def test_did_imputation_cite_returns_registered_bibtex(staggered_panel):
    res = did_imputation(
        staggered_panel,
        y="y", group="unit", time="time", first_treat="first_treat",
    )
    cite = res.cite()
    assert "borusyak2024revisiting" in cite
    assert "Review of Economic Studies" in cite
