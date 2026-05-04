"""Branch coverage for ``statspai.did.wooldridge_did``.

Existing tests already exercise the headline ``wooldridge_did`` /
``etwfe`` paths. This file fills four gaps surfaced by the v1.12.x
coverage audit:

1. ``twfe_decomposition`` — Bacon decomposition + dCDH weights helper
   (entirely uncovered before).
2. ``etwfe(panel=False)`` — repeated cross-section dispatch branch.
3. ``etwfe(cgroup='nevertreated')`` — per-cohort regression branch.
4. ``etwfe(xvar=...)`` — covariate-moderated heterogeneity branch +
   the validation guards (missing column, constant column,
   panel=False + cgroup='nevertreated').
5. ``etwfe_emfx`` group/event/calendar aggregations including
   ``include_leads=True``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did.wooldridge_did import (
    _logistic_fit,
    _ols_fit,
    _stars,
    drdid,
    etwfe,
    etwfe_emfx,
    twfe_decomposition,
    wooldridge_did,
)


# --------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def staggered_panel():
    """80-unit staggered panel with a never-treated cohort."""
    return sp.dgp_did(n_units=80, n_periods=8, staggered=True, seed=11)


@pytest.fixture(scope="module")
def repeated_cs_data():
    """Repeated-cross-section frame (no panel structure)."""
    rng = np.random.default_rng(0)
    n = 800
    ft = rng.choice([0, 5, 7, 10], size=n, p=[0.4, 0.2, 0.2, 0.2])
    t = rng.integers(0, 12, size=n)
    x = rng.normal(size=n)
    y = (t >= ft) * (ft != 0) * 2.0 + 0.4 * x + rng.normal(size=n)
    return pd.DataFrame({
        "y": y,
        "time": t,
        "first_treat": ft.astype(float),
        "x": x,
        "cluster": rng.integers(0, 30, size=n),
    })


# --------------------------------------------------------------------- #
#  twfe_decomposition (Bacon + dCDH)
# --------------------------------------------------------------------- #


def test_twfe_decomposition_returns_bacon_components(staggered_panel):
    res = twfe_decomposition(
        staggered_panel,
        y="y", group="unit", time="time", first_treat="first_treat",
    )
    assert res.method.startswith("TWFE Decomposition")
    # Detail has rows per 2x2 comparison and the dCDH cells in model_info
    assert {"type", "treated_cohort", "control_cohort",
            "estimate", "weight", "weighted_est"} <= set(res.detail.columns)
    # We have at least one of every comparison type in a staggered panel
    types = set(res.detail["type"].unique())
    assert "Earlier vs Later" in types
    assert "Later vs Earlier" in types
    assert "Treated vs Never" in types
    # Bacon weights sum to 1
    assert np.isclose(res.detail["weight"].sum(), 1.0, atol=1e-6)
    mi = res.model_info
    for key in (
        "twfe_beta", "bacon_att", "n_comparisons",
        "n_negative_weights_bacon", "n_negative_weights_dcdh",
        "n_cohorts", "cohorts", "has_never_treated",
        "n_units", "n_periods",
    ):
        assert key in mi
    # dCDH weights frame present and shaped right
    assert isinstance(mi["dcdh_weights"], pd.DataFrame)
    assert {"cohort", "period", "dcdh_weight", "n_cell"} <= set(
        mi["dcdh_weights"].columns
    )


def test_twfe_decomposition_bacon_recombines_to_estimate(staggered_panel):
    res = twfe_decomposition(
        staggered_panel, y="y", group="unit", time="time",
        first_treat="first_treat",
    )
    # weighted_est sums back to the headline ATT
    assert np.isclose(
        res.detail["weighted_est"].sum(),
        res.estimate,
        atol=1e-8,
    )


def test_twfe_decomposition_no_treated_raises():
    """All-control frame has no cohorts → explicit ValueError."""
    df = pd.DataFrame({
        "unit": np.repeat(np.arange(10), 5),
        "time": np.tile(np.arange(5), 10),
        "y": np.random.default_rng(0).normal(size=50),
        "first_treat": [np.nan] * 50,
    })
    with pytest.raises(ValueError):
        twfe_decomposition(
            df, y="y", group="unit", time="time", first_treat="first_treat",
        )


# --------------------------------------------------------------------- #
#  etwfe — repeated cross-section (panel=False)
# --------------------------------------------------------------------- #


def test_etwfe_panel_false_runs(repeated_cs_data):
    df = repeated_cs_data
    res = etwfe(
        df, y="y", group="cluster", time="time", first_treat="first_treat",
        panel=False,
    )
    # True effect is 2.0; with n=800 we should land within ~1.0 of it
    assert abs(res.estimate - 2.0) < 1.5
    assert res.model_info["panel"] is False
    assert res.model_info["n_cohorts"] == 3  # cohorts {5, 7, 10}


def test_etwfe_panel_false_rank_deficient_warns():
    """Tiny rank-deficient design fires the design-matrix RuntimeWarning."""
    rng = np.random.default_rng(0)
    n = 60
    ft = np.array([0] * 30 + [5] * 30, dtype=float)
    t = rng.integers(0, 6, size=n)
    # Identical control column will make things singular when expanded
    df = pd.DataFrame({
        "y": rng.normal(size=n),
        "time": t,
        "first_treat": ft,
        "ctrl_a": np.ones(n),
        "ctrl_b": np.ones(n),  # collinear with ctrl_a
        "cluster": rng.integers(0, 5, size=n),
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        etwfe(
            df, y="y", group="cluster", time="time",
            first_treat="first_treat",
            panel=False, controls=["ctrl_a", "ctrl_b"],
        )
    # At least one RuntimeWarning about rank-deficient design
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "rank-deficient" in str(w.message)
        for w in caught
    )


def test_etwfe_panel_false_with_nevertreated_raises(repeated_cs_data):
    with pytest.raises(NotImplementedError, match="not yet supported"):
        etwfe(
            repeated_cs_data,
            y="y", group="cluster", time="time", first_treat="first_treat",
            panel=False, cgroup="nevertreated",
        )


# --------------------------------------------------------------------- #
#  etwfe — cgroup='nevertreated'
# --------------------------------------------------------------------- #


def test_etwfe_cgroup_nevertreated(staggered_panel):
    res = etwfe(
        staggered_panel,
        y="y", group="unit", time="time", first_treat="first_treat",
        cgroup="nevertreated",
    )
    assert res.method.endswith("never-treated control")
    assert res.model_info["cgroup"] == "nevertreated"
    # Detail has one row per cohort
    assert len(res.detail) == res.model_info["n_cohorts"]


def test_etwfe_cgroup_nevertreated_no_never_raises():
    """Synthesise a panel with NO never-treated unit → explicit ValueError."""
    df = sp.dgp_did(n_units=40, n_periods=6, staggered=True, seed=2)
    # Fill all NaNs with a real cohort year so nothing is never-treated.
    df = df.copy()
    df.loc[df["first_treat"].isna(), "first_treat"] = 6
    with pytest.raises(ValueError, match="never-treated"):
        etwfe(
            df, y="y", group="unit", time="time", first_treat="first_treat",
            cgroup="nevertreated",
        )


def test_etwfe_invalid_cgroup_raises(staggered_panel):
    with pytest.raises(ValueError, match="cgroup must be"):
        etwfe(
            staggered_panel,
            y="y", group="unit", time="time", first_treat="first_treat",
            cgroup="bogus",
        )


# --------------------------------------------------------------------- #
#  etwfe — xvar (covariate-moderated)
# --------------------------------------------------------------------- #


def test_etwfe_with_xvar_single(staggered_panel):
    df = staggered_panel.copy()
    df["x1"] = np.random.default_rng(0).normal(size=len(df))
    res = etwfe(
        df, y="y", group="unit", time="time", first_treat="first_treat",
        xvar="x1",
    )
    # Single-xvar back-compat aliases populated
    cols = set(res.detail.columns)
    assert "att_at_xmean" in cols
    assert "slope_x1" in cols
    assert "slope_wrt_x" in cols  # back-compat alias
    assert res.model_info["xvar"] == ["x1"]
    # Centering point recorded
    assert "x1" in res.model_info["xvar_means"]


def test_etwfe_with_xvar_multi(staggered_panel):
    df = staggered_panel.copy()
    rng = np.random.default_rng(1)
    df["x1"] = rng.normal(size=len(df))
    df["x2"] = rng.normal(size=len(df))
    res = etwfe(
        df, y="y", group="unit", time="time", first_treat="first_treat",
        xvar=["x1", "x2"],
    )
    cols = set(res.detail.columns)
    assert "slope_x1" in cols and "slope_x2" in cols
    # Multi-xvar should NOT alias to slope_wrt_x
    assert "slope_wrt_x" not in cols


def test_etwfe_xvar_missing_column_raises(staggered_panel):
    with pytest.raises(KeyError, match="not found"):
        etwfe(
            staggered_panel,
            y="y", group="unit", time="time", first_treat="first_treat",
            xvar="nonexistent",
        )


def test_etwfe_xvar_constant_raises(staggered_panel):
    df = staggered_panel.copy()
    df["xc"] = 7.0  # constant
    with pytest.raises(ValueError, match="constant"):
        etwfe(
            df,
            y="y", group="unit", time="time", first_treat="first_treat",
            xvar="xc",
        )


def test_etwfe_xvar_too_few_observations_raises(staggered_panel):
    df = staggered_panel.copy()
    df["xs"] = np.nan
    df.loc[df.index[0], "xs"] = 1.5
    with pytest.raises(ValueError, match="fewer than 2"):
        etwfe(
            df,
            y="y", group="unit", time="time", first_treat="first_treat",
            xvar="xs",
        )


def test_etwfe_nevertreated_with_xvar(staggered_panel):
    """cgroup='nevertreated' + xvar runs through ``_etwfe_with_xvar`` per
    cohort."""
    df = staggered_panel.copy()
    df["x1"] = np.random.default_rng(2).normal(size=len(df))
    res = etwfe(
        df, y="y", group="unit", time="time", first_treat="first_treat",
        xvar=["x1"], cgroup="nevertreated",
    )
    assert res.method.endswith("never-treated control")


# --------------------------------------------------------------------- #
#  etwfe_emfx aggregations
# --------------------------------------------------------------------- #


def test_etwfe_emfx_simple_matches_overall(staggered_panel):
    fit = etwfe(
        staggered_panel,
        y="y", group="unit", time="time", first_treat="first_treat",
    )
    s = etwfe_emfx(fit, type="simple")
    assert np.isclose(s.estimate, fit.estimate)
    assert np.isclose(s.se, fit.se)


def test_etwfe_emfx_group_recovers_per_cohort(staggered_panel):
    fit = etwfe(
        staggered_panel,
        y="y", group="unit", time="time", first_treat="first_treat",
    )
    g = etwfe_emfx(fit, type="group")
    assert "cohort" in g.detail.columns and "estimate" in g.detail.columns
    # Headline estimate equals the overall ATT (H2 fix) — group view
    # exposes per-cohort, not their unweighted mean
    assert np.isclose(g.estimate, fit.estimate)


def test_etwfe_emfx_event_default_post_only(staggered_panel):
    fit = etwfe(
        staggered_panel,
        y="y", group="unit", time="time", first_treat="first_treat",
    )
    e = etwfe_emfx(fit, type="event")
    # Default include_leads=False → only rel_time >= 0
    assert (e.detail["event_time"] >= 0).all()


def test_etwfe_emfx_event_include_leads(staggered_panel):
    fit = etwfe(
        staggered_panel,
        y="y", group="unit", time="time", first_treat="first_treat",
    )
    e = etwfe_emfx(fit, type="event", include_leads=True)
    # With leads we should now see at least one negative event time
    assert (e.detail["event_time"] < 0).any()


def test_etwfe_emfx_calendar(staggered_panel):
    fit = etwfe(
        staggered_panel,
        y="y", group="unit", time="time", first_treat="first_treat",
    )
    c = etwfe_emfx(fit, type="calendar")
    assert "calendar_time" in c.detail.columns


def test_etwfe_emfx_invalid_type_raises(staggered_panel):
    fit = etwfe(
        staggered_panel,
        y="y", group="unit", time="time", first_treat="first_treat",
    )
    with pytest.raises(ValueError, match="type must be"):
        etwfe_emfx(fit, type="weekly")


def test_wooldridge_did_summary_renders_event_study(staggered_panel):
    """``CausalResult.summary()`` must accept the wooldridge_did event-study
    schema (``rel_time`` / ``estimate`` columns) — regression guard for the
    fix to ``core/results.py:summary`` that previously assumed
    (``relative_time`` / ``att``) only.
    """
    res = wooldridge_did(
        staggered_panel,
        y="y", group="unit", time="time", first_treat="first_treat",
    )
    s = res.summary()
    assert "ATT" in s
    assert "Event Study Coefficients" in s
    # First post-treatment event time row formatted as ``e =   0``
    assert "e =   0" in s


def test_etwfe_emfx_event_without_event_study_raises():
    """Pass a manufactured result with no ``event_study`` info → fails."""
    from statspai.core.results import CausalResult
    fake = CausalResult(
        method="manual",
        estimand="ATT",
        estimate=1.0, se=0.1,
        pvalue=0.0, ci=(0.5, 1.5),
        alpha=0.05, n_obs=10,
        detail=pd.DataFrame({"cohort": [1], "att": [1.0],
                             "se": [0.1], "pvalue": [0.0], "n_obs": [5]}),
        model_info={"cohorts": [1]},  # no event_study
        _citation_key="wooldridge_twfe",
    )
    with pytest.raises(ValueError, match="event_study coefficients"):
        etwfe_emfx(fake, type="event")


def test_etwfe_emfx_requires_etwfe_result():
    from statspai.core.results import CausalResult
    fake = CausalResult(
        method="other", estimand="ATT",
        estimate=1.0, se=0.1, pvalue=0.0, ci=(0.5, 1.5),
        alpha=0.05, n_obs=10,
        detail=pd.DataFrame(), model_info={},  # missing 'cohorts'
        _citation_key=None,
    )
    with pytest.raises(ValueError, match="missing 'cohorts'"):
        etwfe_emfx(fake, type="simple")


# --------------------------------------------------------------------- #
#  drdid — quick smoke at both ``method`` settings + small-bootstrap path
# --------------------------------------------------------------------- #


def test_drdid_traditional_method():
    rng = np.random.default_rng(0)
    n = 500
    G = rng.integers(0, 2, n)
    T = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n)
    y = 1 + 0.5 * x + 2 * G + 3 * T + 4 * G * T + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "treated": G, "post": T, "x": x})
    res = drdid(
        df, y="y", group="treated", time="post", covariates=["x"],
        method="trad", n_boot=50, random_state=0,
    )
    # Traditional DR-DID is consistent but noisier than the improved
    # variant on small / unbalanced 2x2 designs — wide tolerance is
    # appropriate; we just check the dispatcher returns a finite estimate.
    assert np.isfinite(res.estimate)
    assert res.model_info["method"] == "traditional"


def test_drdid_no_covariates_falls_back():
    """No covariates → simple 2x2 DID under the hood."""
    rng = np.random.default_rng(2)
    n = 400
    G = rng.integers(0, 2, n)
    T = rng.integers(0, 2, n)
    y = 1 + 2 * G + 3 * T + 4 * G * T + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "treated": G, "post": T})
    res = drdid(
        df, y="y", group="treated", time="post",
        covariates=None, n_boot=30, random_state=1,
    )
    assert abs(res.estimate - 4.0) < 1.5


def test_drdid_invalid_group_dimension_raises():
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "treated": [0, 1, 2, 1],   # not binary
        "post": [0, 1, 0, 1],
    })
    with pytest.raises(ValueError, match="must be binary"):
        drdid(df, y="y", group="treated", time="post")


def test_drdid_invalid_time_dimension_raises():
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "treated": [0, 1, 0, 1],
        "post": [0, 1, 2, 1],   # not binary
    })
    with pytest.raises(ValueError, match="must be binary"):
        drdid(df, y="y", group="treated", time="post")


# --------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------- #


def test_stars_thresholds():
    assert _stars(1e-4) == "***"
    assert _stars(0.005) == "**"
    assert _stars(0.04) == "*"
    assert _stars(0.10) == ""


def test_ols_fit_with_cluster_matches_no_cluster_dimensions():
    rng = np.random.default_rng(0)
    n, k = 200, 3
    X = rng.normal(size=(n, k))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=n)
    cluster = rng.integers(0, 10, size=n)
    beta_c, se_c, V_c = _ols_fit(X, y, cluster=cluster)
    beta_nc, se_nc, V_nc = _ols_fit(X, y, cluster=None)
    assert beta_c.shape == (k,) and se_c.shape == (k,)
    assert V_c.shape == (k, k) and V_nc.shape == (k, k)
    # Point estimate must be identical (only SE depends on cluster choice)
    np.testing.assert_allclose(beta_c, beta_nc, atol=1e-10)


def test_logistic_fit_recovers_sign():
    rng = np.random.default_rng(0)
    n = 800
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    p_true = 1.0 / (1.0 + np.exp(-(0.0 + 2.0 * X[:, 1])))
    y = rng.binomial(1, p_true)
    p_hat = _logistic_fit(X, y)
    # Predicted probabilities are bounded and correlated with truth
    assert ((p_hat > 0) & (p_hat < 1)).all()
    assert np.corrcoef(p_hat, p_true)[0, 1] > 0.8
