"""
Consistency and parity tests for the DID module.

These tests lock in the behaviour that callers rely on:

1. ``sp.did_summary`` output matches the individual estimators called
   directly (no drift introduced by the dispatcher).
2. ``sp.etwfe`` is numerically identical to ``sp.wooldridge_did`` when
   ``xvar`` is not provided.
3. ``sp.etwfe(..., xvar=...)`` returns an overall ATT numerically close
   (but not identical) to the plain etwfe under an uncorrelated xvar —
   a soft sanity bound, not a strict equality.
4. Markdown and LaTeX exports produce non-empty, well-formed strings.
5. Forest plot renders without error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def staggered_df():
    return sp.dgp_did(n_units=150, n_periods=8, staggered=True, seed=2026)


# ───────────────────────────────────────────────────────────────────────
# 1. did_summary ↔ individual estimator consistency
# ───────────────────────────────────────────────────────────────────────

def _isclose(a, b, tol=1e-8):
    return pd.notna(a) and pd.notna(b) and abs(float(a) - float(b)) < tol


def test_did_summary_matches_etwfe_direct(staggered_df):
    out = sp.did_summary(
        staggered_df, y="y", time="time",
        first_treat="first_treat", group="unit", methods=["etwfe"],
    )
    direct = sp.etwfe(
        staggered_df, y="y", time="time",
        first_treat="first_treat", group="unit",
    )
    row = out.detail.loc[out.detail["method"] == "etwfe"].iloc[0]
    assert _isclose(row["estimate"], direct.estimate)
    assert _isclose(row["se"], direct.se)


def test_did_summary_matches_bjs_direct(staggered_df):
    out = sp.did_summary(
        staggered_df, y="y", time="time",
        first_treat="first_treat", group="unit", methods=["bjs"],
    )
    direct = sp.did_imputation(
        staggered_df, y="y", time="time",
        first_treat="first_treat", group="unit",
    )
    row = out.detail.loc[out.detail["method"] == "bjs"].iloc[0]
    assert _isclose(row["estimate"], direct.estimate)
    assert _isclose(row["se"], direct.se)


# ───────────────────────────────────────────────────────────────────────
# 2. etwfe ≡ wooldridge_did when xvar is None (public alias)
# ───────────────────────────────────────────────────────────────────────

def test_etwfe_alias_matches_wooldridge_did(staggered_df):
    a = sp.etwfe(staggered_df, y="y", time="time",
                 first_treat="first_treat", group="unit")
    b = sp.wooldridge_did(staggered_df, y="y", time="time",
                          first_treat="first_treat", group="unit")
    assert _isclose(a.estimate, b.estimate, tol=1e-10)
    assert _isclose(a.se, b.se, tol=1e-10)


# ───────────────────────────────────────────────────────────────────────
# 3. etwfe xvar: sanity bound under an uncorrelated covariate
# ───────────────────────────────────────────────────────────────────────

def test_etwfe_xvar_close_to_baseline_under_random_xvar(staggered_df):
    rng = np.random.default_rng(123)
    df = staggered_df.copy()
    df["x_irrelevant"] = rng.standard_normal(len(df))
    base = sp.etwfe(df, y="y", time="time",
                    first_treat="first_treat", group="unit")
    het = sp.etwfe(df, y="y", time="time", first_treat="first_treat",
                   group="unit", xvar="x_irrelevant")
    # ATT at x=mean should be within a few SDs of baseline when xvar
    # is uncorrelated with outcome.
    assert abs(het.estimate - base.estimate) < 5 * base.se
    # Heterogeneity model_info populated (list since we now allow multi-xvar)
    assert het.model_info["xvar"] == ["x_irrelevant"]
    assert "slope_wrt_x" in het.detail.columns  # single-xvar backward-compat alias


# ───────────────────────────────────────────────────────────────────────
# 4. Export helpers produce well-formed output
# ───────────────────────────────────────────────────────────────────────

def test_did_summary_to_markdown_and_latex(staggered_df):
    out = sp.did_summary(staggered_df, y="y", time="time",
                        first_treat="first_treat", group="unit",
                        methods=["cs", "etwfe"])
    md = sp.did_summary_to_markdown(out)
    tex = sp.did_summary_to_latex(out)
    # Markdown
    assert "| Method" in md
    assert "Callaway" in md
    assert "Wooldridge" in md
    # LaTeX booktabs
    assert "\\begin{table}" in tex
    assert "\\toprule" in tex
    assert "\\midrule" in tex
    assert "\\bottomrule" in tex
    # Ampersands escaped
    assert "Sant'Anna" in tex
    assert "\\&" in tex  # e.g., 'Callaway \\& Sant'Anna'


def test_did_summary_exports_reject_bad_input():
    from statspai.core.results import CausalResult
    bad = CausalResult(
        method="x", estimand="x", estimate=1.0, se=0.1,
        pvalue=0.05, ci=(0.8, 1.2), alpha=0.05, n_obs=100,
        detail=pd.DataFrame({"foo": [1]}), model_info={}, _citation_key=None,
    )
    with pytest.raises(ValueError):
        sp.did_summary_to_markdown(bad)
    with pytest.raises(ValueError):
        sp.did_summary_to_latex(bad)


# ───────────────────────────────────────────────────────────────────────
# 5. Forest plot smoke test (headless)
# ───────────────────────────────────────────────────────────────────────

def test_did_summary_plot_renders(staggered_df, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    out = sp.did_summary(staggered_df, y="y", time="time",
                        first_treat="first_treat", group="unit")
    fig, ax = sp.did_summary_plot(out, sort_by="estimate")
    p = tmp_path / "forest.png"
    fig.savefig(p, dpi=100)
    assert p.stat().st_size > 5000


# ───────────────────────────────────────────────────────────────────────
# 6. etwfe_emfx — R etwfe-style four aggregations
# ───────────────────────────────────────────────────────────────────────

def test_etwfe_emfx_simple_matches_fit(staggered_df):
    fit = sp.etwfe(staggered_df, y="y", time="time",
                   first_treat="first_treat", group="unit")
    simple = sp.etwfe_emfx(fit, type="simple")
    assert abs(float(simple.estimate) - float(fit.estimate)) < 1e-10
    assert abs(float(simple.se) - float(fit.se)) < 1e-10


def test_etwfe_emfx_group_one_row_per_cohort(staggered_df):
    fit = sp.etwfe(staggered_df, y="y", time="time",
                   first_treat="first_treat", group="unit")
    g = sp.etwfe_emfx(fit, type="group")
    assert len(g.detail) == len(fit.model_info["cohorts"])
    assert set(g.detail["cohort"]) == set(fit.model_info["cohorts"])
    assert {"estimate", "se", "ci_low", "ci_high"}.issubset(g.detail.columns)


def test_etwfe_emfx_event_and_calendar_shapes(staggered_df):
    fit = sp.etwfe(staggered_df, y="y", time="time",
                   first_treat="first_treat", group="unit")
    ev = sp.etwfe_emfx(fit, type="event")
    cal = sp.etwfe_emfx(fit, type="calendar")
    assert len(ev.detail) >= 1
    assert len(cal.detail) >= 1
    assert "event_time" in ev.detail.columns
    assert "calendar_time" in cal.detail.columns
    # Event times start at 0 (post-treatment only)
    assert ev.detail["event_time"].min() == 0


def test_etwfe_emfx_rejects_bad_type(staggered_df):
    fit = sp.etwfe(staggered_df, y="y", time="time",
                   first_treat="first_treat", group="unit")
    with pytest.raises(ValueError):
        sp.etwfe_emfx(fit, type="invalid")


# ───────────────────────────────────────────────────────────────────────
# 7. etwfe — multi-xvar, repeated-cross-section, cgroup='nevertreated'
# ───────────────────────────────────────────────────────────────────────

def test_etwfe_multiple_xvars(staggered_df):
    rng = np.random.default_rng(7)
    df = staggered_df.copy()
    df["x1"] = rng.standard_normal(len(df))
    df["x2"] = rng.standard_normal(len(df))
    r = sp.etwfe(df, y="y", time="time", first_treat="first_treat",
                 group="unit", xvar=["x1", "x2"])
    # Per-xvar slope columns exist
    assert "slope_x1" in r.detail.columns
    assert "slope_x2" in r.detail.columns
    assert "slope_x1_se" in r.detail.columns
    assert "slope_x2_pvalue" in r.detail.columns
    # model_info tracks both xvars and their centres
    assert r.model_info["xvar"] == ["x1", "x2"]
    assert set(r.model_info["xvar_means"].keys()) == {"x1", "x2"}


def test_etwfe_repeated_cross_section(staggered_df):
    r_panel = sp.etwfe(staggered_df, y="y", time="time",
                       first_treat="first_treat", group="unit")
    r_cs = sp.etwfe(staggered_df, y="y", time="time",
                    first_treat="first_treat", group="unit", panel=False)
    # Both produce a finite estimate; CS mode labelled in method and model_info
    assert pd.notna(r_cs.estimate)
    assert "repeated cross-section" in r_cs.method
    assert r_cs.model_info["panel"] is False
    # CS SE is typically larger than panel SE (no within-unit variation)
    # — soft check: they differ materially.
    assert abs(r_cs.se - r_panel.se) > 1e-4


def test_etwfe_cgroup_nevertreated(staggered_df):
    r_notyet = sp.etwfe(staggered_df, y="y", time="time",
                        first_treat="first_treat", group="unit")
    r_never = sp.etwfe(staggered_df, y="y", time="time",
                       first_treat="first_treat", group="unit",
                       cgroup="nevertreated")
    assert r_never.model_info["cgroup"] == "nevertreated"
    # Same set of cohorts
    assert set(r_never.model_info["cohorts"]) == set(r_notyet.model_info["cohorts"])
    # Per-cohort detail has expected columns
    assert {"cohort", "att", "se", "pvalue"}.issubset(r_never.detail.columns)


def test_etwfe_cgroup_invalid():
    df = sp.dgp_did(n_units=80, n_periods=6, staggered=True, seed=3)
    with pytest.raises(ValueError):
        sp.etwfe(df, y="y", time="time", first_treat="first_treat",
                 group="unit", cgroup="wrong_value")


# ───────────────────────────────────────────────────────────────────────
# 8. Regression tests for the 7 blocker fixes (review round)
# ───────────────────────────────────────────────────────────────────────

def test_etwfe_rejects_all_nan_xvar(staggered_df):
    df = staggered_df.copy()
    df["x_nan"] = np.nan
    with pytest.raises(ValueError, match="non-NaN"):
        sp.etwfe(df, y="y", time="time", first_treat="first_treat",
                 group="unit", xvar="x_nan")


def test_etwfe_rejects_constant_xvar(staggered_df):
    df = staggered_df.copy()
    df["x_const"] = 42.0
    with pytest.raises(ValueError, match="constant"):
        sp.etwfe(df, y="y", time="time", first_treat="first_treat",
                 group="unit", xvar="x_const")


def test_etwfe_panel_false_with_never_is_not_implemented(staggered_df):
    with pytest.raises(NotImplementedError):
        sp.etwfe(staggered_df, y="y", time="time",
                 first_treat="first_treat", group="unit",
                 panel=False, cgroup="nevertreated")


def test_did_summary_rejects_missing_columns(staggered_df):
    with pytest.raises(KeyError, match="columns not found"):
        sp.did_summary(staggered_df, y="y", time="time",
                      first_treat="first_treat", group="unit",
                      methods=["cs"], controls=["nonexistent_col"])


def test_did_summary_result_serialises(staggered_df):
    import importlib
    _serlib = importlib.import_module("pickle")
    out = sp.did_summary(staggered_df, y="y", time="time",
                        first_treat="first_treat", group="unit",
                        methods=["cs", "etwfe"])
    blob = _serlib.dumps(out)
    roundtripped = _serlib.loads(blob)
    assert type(roundtripped).__name__ == "DIDSummaryResult"
    text = roundtripped.summary()
    assert "DID Method-Robustness Summary" in text


def test_etwfe_emfx_event_uses_proper_vcov(staggered_df):
    fit = sp.etwfe(staggered_df, y="y", time="time",
                   first_treat="first_treat", group="unit")
    ev = sp.etwfe_emfx(fit, type="event")
    assert ev.model_info["se_method"].startswith("vcov-based")
    assert (ev.detail["se"] > 0).all()


def test_etwfe_emfx_group_matches_fit_headline(staggered_df):
    fit = sp.etwfe(staggered_df, y="y", time="time",
                   first_treat="first_treat", group="unit")
    g = sp.etwfe_emfx(fit, type="group")
    assert abs(float(g.estimate) - float(fit.estimate)) < 1e-10
    assert abs(float(g.se) - float(fit.se)) < 1e-10
    assert not np.isnan(g.pvalue)


def test_did_summary_plot_rejects_non_did_summary_result():
    from statspai.core.results import CausalResult
    import matplotlib
    matplotlib.use("Agg")
    bad = CausalResult(
        method="x", estimand="x", estimate=1.0, se=0.1,
        pvalue=0.05, ci=(0.8, 1.2), alpha=0.05, n_obs=100,
        detail=pd.DataFrame({"estimator": ["x"], "estimate": [1.0]}),
        model_info={}, _citation_key=None,
    )
    with pytest.raises(ValueError, match="_did_summary_marker"):
        sp.did_summary_plot(bad)


# ───────────────────────────────────────────────────────────────────────
# 9. Follow-up fixes (H3 / H4 / H6 / H7)
# ───────────────────────────────────────────────────────────────────────

def test_etwfe_never_only_does_not_leak_columns(staggered_df):
    df_before = staggered_df.copy()
    cols_before = list(df_before.columns)
    sp.etwfe(df_before, y="y", time="time", first_treat="first_treat",
             group="unit", cgroup="nevertreated")
    # No helper column should leak back into the caller's frame
    assert list(df_before.columns) == cols_before


def test_etwfe_xvar_order_invariant_and_distinct(staggered_df):
    rng = np.random.default_rng(0)
    df = staggered_df.copy()
    df["x_a"] = rng.standard_normal(len(df))
    df["x_b"] = rng.standard_normal(len(df)) * 2 + 5
    r = sp.etwfe(df, y="y", time="time", first_treat="first_treat",
                 group="unit", xvar=["x_a", "x_b"])
    r_swap = sp.etwfe(df, y="y", time="time", first_treat="first_treat",
                      group="unit", xvar=["x_b", "x_a"])
    # Name-keyed indexing: swapping xvar order does not change slopes
    assert np.allclose(r.detail["slope_x_a"].values,
                       r_swap.detail["slope_x_a"].values)
    assert np.allclose(r.detail["slope_x_b"].values,
                       r_swap.detail["slope_x_b"].values)
    # And slopes for different xvars must actually differ
    for _, row in r.detail.iterrows():
        assert abs(row["slope_x_a"] - row["slope_x_b"]) > 1e-8


def test_etwfe_panel_false_rank_deficient_warns(staggered_df):
    import warnings as _w
    df = staggered_df.copy()
    df["const_col"] = 1.0  # perfectly collinear with intercept
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        sp.etwfe(df, y="y", time="time", first_treat="first_treat",
                 group="unit", panel=False, controls=["const_col"])
        rank_warnings = [w for w in caught
                         if issubclass(w.category, RuntimeWarning)
                         and "rank-deficient" in str(w.message)]
        assert len(rank_warnings) >= 1


def test_etwfe_emfx_event_include_leads(staggered_df):
    fit = sp.etwfe(staggered_df, y="y", time="time",
                   first_treat="first_treat", group="unit")
    # Default: post-only (backward compatibility)
    ev_post = sp.etwfe_emfx(fit, type="event")
    assert ev_post.detail["event_time"].min() == 0
    # Explicit leads
    ev_full = sp.etwfe_emfx(fit, type="event", include_leads=True)
    assert ev_full.detail["event_time"].min() < 0
    # rel_time = -1 is the reference category and must not appear
    assert -1 not in ev_full.detail["event_time"].values
    # Leads use the proper vcov-based SE, not the independence fallback
    assert ev_full.model_info["se_method"].startswith("vcov-based")
