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
