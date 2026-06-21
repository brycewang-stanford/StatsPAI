"""Coverage campaign — DataFrame-input DiD plot renderers (``did/plots.py``).

Part of the core-module ≥95% coverage initiative (see
``.coverage_campaign/CAMPAIGN.md``). ``test_cov95_did_plots.py`` covers the
result-based plotters; this file closes the two large *DataFrame-input*
renderers that need no estimator fit:

* ``sp.did_plot``                — the classic 2×2 DiD diagram with the dashed
  counterfactual line and the ATT annotation arrow (all branches: inferred vs
  explicit ``treat_time``, binary vs non-0/1 ``treat`` coding, counterfactual
  on/off, custom labels/colors, caller-supplied ``ax``).
* ``sp.treatment_rollout_plot``  — the unit×time adoption tile chart (binary
  treatment vs first-treat-period cohort coding; ``sort_by`` variants; cohort
  boundary labels).

Assertions are structural-but-real: each call must return a Matplotlib
``Figure`` whose axes carry the lines / image / annotations the renderer draws.
Rendering contracts, no estimator numerics.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import statspai as sp


# --------------------------------------------------------------------------
# Panel builders
# --------------------------------------------------------------------------
def _did_panel(seed=0, n_per_group=6, n_t=10, treat_time=5, effect=3.0):
    """Two-group (treated/control) panel with a common pre-trend."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, is_t in (("control", 0), ("treated", 1)):
        for u in range(n_per_group):
            base = rng.normal(0, 1) + (0.5 if is_t else 0.0)
            for t in range(1, n_t + 1):
                post = 1 if t >= treat_time else 0
                y = base + 0.4 * t + (effect * is_t * post) + rng.normal(0, 0.2)
                rows.append({"unit": f"{g}{u}", "time": t, "y": y, "treated": is_t})
    return pd.DataFrame(rows)


def _rollout_panel(seed=0, n_t=12):
    """Staggered-adoption panel: never-treated + three adopting cohorts."""
    rng = np.random.default_rng(seed)
    rows = []
    cohorts = {"n0": 0, "n1": 0, "a4": 4, "a4b": 4, "a7": 7, "a10": 10}
    for u, g in cohorts.items():
        for t in range(1, n_t + 1):
            d = 1 if (g != 0 and t >= g) else 0
            rows.append(
                {"unit": u, "time": t, "y": rng.normal(0, 1) + d, "d": d, "cohort": g}
            )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _is_fig(obj):
    return isinstance(obj, plt.Figure)


# --------------------------------------------------------------------------
# sp.did_plot
# --------------------------------------------------------------------------
def test_did_plot_counterfactual_and_annotation():
    df = _did_panel(seed=1)
    fig, ax = sp.did_plot(df, y="y", time="time", treat="treated", treat_time=5)
    assert _is_fig(fig)
    # control, treatment, and counterfactual lines all drawn
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert {"Treatment", "Control", "Counterfactual"} <= labels
    # ATT annotation text present
    assert any("ATT" in t.get_text() for t in ax.texts)


def test_did_plot_no_counterfactual():
    df = _did_panel(seed=2)
    fig, ax = sp.did_plot(
        df, y="y", time="time", treat="treated", treat_time=5, show_counterfactual=False
    )
    assert _is_fig(fig)
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert "Counterfactual" not in labels


def test_did_plot_infers_treat_time_midpoint():
    df = _did_panel(seed=3)
    # treat_time=None → inferred as the middle time value
    fig, ax = sp.did_plot(df, y="y", time="time", treat="treated")
    assert _is_fig(fig)
    assert len(ax.get_lines()) >= 2


def test_did_plot_nonbinary_treat_is_binarized():
    df = _did_panel(seed=4)
    # encode treatment as 0/2 instead of 0/1 → triggers the binarize branch
    df["grp"] = df["treated"] * 2
    fig, ax = sp.did_plot(df, y="y", time="time", treat="grp", treat_time=5)
    assert _is_fig(fig)


def test_did_plot_custom_labels_colors_and_ax():
    df = _did_panel(seed=5)
    fig0, ax0 = plt.subplots()
    fig, ax = sp.did_plot(
        df,
        y="y",
        time="time",
        treat="treated",
        treat_time=5,
        labels={
            "treat": "Adopters",
            "control": "Non-adopters",
            "counterfactual": "No-policy",
        },
        colors=("#111111", "#222222", "#333333"),
        ax=ax0,
        title="Custom DiD",
        annotate_effect=False,
    )
    assert ax is ax0
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert {"Adopters", "Non-adopters", "No-policy"} <= labels
    assert ax.get_title() == "Custom DiD"


# --------------------------------------------------------------------------
# sp.treatment_rollout_plot
# --------------------------------------------------------------------------
def test_treatment_rollout_binary_treat():
    df = _rollout_panel(seed=0)
    fig, ax = sp.treatment_rollout_plot(df, time="time", treat="d", id="unit")
    assert _is_fig(fig)
    # the adoption tile image is drawn
    assert len(ax.get_images()) == 1


def test_treatment_rollout_cohort_coded_treat():
    df = _rollout_panel(seed=1)
    # 'cohort' holds the first-treat period (0 = never) → non-binary branch
    fig, ax = sp.treatment_rollout_plot(
        df, time="time", treat="cohort", id="unit", sort_by="treat_time"
    )
    assert _is_fig(fig)
    assert len(ax.get_images()) == 1
    # cohort boundary labels rendered (g=4 / g=7 / g=10 / Never)
    yt = {t.get_text() for t in ax.get_yticklabels()}
    assert any(s.startswith("g=") or s == "Never" for s in yt)


def test_treatment_rollout_sort_by_id():
    df = _rollout_panel(seed=2)
    fig, ax = sp.treatment_rollout_plot(
        df, time="time", treat="d", id="unit", sort_by="id"
    )
    assert _is_fig(fig)


def test_treatment_rollout_long_horizon_and_supplied_ax():
    # >20 time periods → the down-sampled x-tick branch; caller-supplied ax.
    df = _rollout_panel(seed=3, n_t=24)
    fig0, ax0 = plt.subplots()
    fig, ax = sp.treatment_rollout_plot(df, time="time", treat="d", id="unit", ax=ax0)
    assert ax is ax0
    assert len(ax.get_images()) == 1


def test_treatment_rollout_no_cohort_labels():
    # show_cohort_labels=False with a small (≤30-unit) panel → per-unit yticks.
    df = _rollout_panel(seed=4)
    fig, ax = sp.treatment_rollout_plot(
        df, time="time", treat="d", id="unit", show_cohort_labels=False
    )
    assert _is_fig(fig)
    yt = {t.get_text() for t in ax.get_yticklabels()}
    assert "a4" in yt  # individual unit labels, not cohort labels


# --------------------------------------------------------------------------
# sp.parallel_trends_plot
# --------------------------------------------------------------------------
def test_parallel_trends_binary_with_treatment_line():
    df = _did_panel(seed=6)
    fig, ax = sp.parallel_trends_plot(
        df,
        y="y",
        time="time",
        treat="treated",
        treat_time=5,
        ci=True,
    )
    assert _is_fig(fig)
    # treated + control lines + treatment-onset axvline → 'Treatment' legend
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert {"Treatment"} <= labels or len(ax.get_lines()) >= 3


def test_parallel_trends_median_aggregation_with_ax():
    df = _did_panel(seed=7)
    fig0, ax0 = plt.subplots()
    fig, ax = sp.parallel_trends_plot(
        df,
        y="y",
        time="time",
        treat="treated",
        agg="median",
        ci=True,
        ax=ax0,
        title="PT median",
    )
    assert ax is ax0
    assert ax.get_title() == "PT median"


# --------------------------------------------------------------------------
# sp.bacon_plot — caller-supplied ax + empty-decomposition guard
# --------------------------------------------------------------------------
def test_bacon_plot_supplied_ax_and_custom_colors():
    bacon = sp.bacon_decomposition(
        _rollout_panel(seed=0), y="y", treat="d", time="time", id="unit"
    )
    fig0, ax0 = plt.subplots()
    fig, ax = sp.bacon_plot(
        bacon,
        ax=ax0,
        colors={
            "Treated vs Never-treated": "#123456",
            "Earlier vs Later treated": "#234567",
            "Later vs Already-treated": "#345678",
        },
    )
    assert ax is ax0
    assert _is_fig(fig)


def test_bacon_plot_empty_decomposition_raises():
    empty = {
        "decomposition": pd.DataFrame(columns=["type", "weight", "estimate"]),
        "beta_twfe": 0.0,
    }
    with pytest.raises(ValueError, match="no sub-comparisons"):
        sp.bacon_plot(empty)


# --------------------------------------------------------------------------
# sp.did_summary_plot — forest plot with supplied ax + the marker guard
# --------------------------------------------------------------------------
def test_did_summary_plot_forest_with_ax():
    df = sp.dgp_did(n_units=80, n_periods=8, staggered=True, seed=2026)
    out = sp.did_summary(
        df,
        y="y",
        time="time",
        first_treat="first_treat",
        group="unit",
        methods=["cs", "etwfe"],
    )
    fig0, ax0 = plt.subplots()
    fig, ax = sp.did_summary_plot(out, ax=ax0, reference=0.0)
    assert ax is ax0
    # one error-bar row per fitted method + the reference / mean lines
    assert len(ax.collections) >= 1 or len(ax.get_lines()) >= 1


def test_did_summary_plot_rejects_non_summary_result():
    import types

    bogus = types.SimpleNamespace(model_info={}, detail=None, estimate=1.0)
    with pytest.raises(ValueError, match="_did_summary_marker"):
        sp.did_summary_plot(bogus)


def test_did_summary_plot_rejects_malformed_detail():
    import types

    # Has the marker but the detail table lacks the required 'estimate' column.
    bad = types.SimpleNamespace(
        model_info={"_did_summary_marker": True},
        detail=pd.DataFrame({"method": ["cs"], "ci_low": [0.0], "ci_high": [1.0]}),
        estimate=0.5,
    )
    with pytest.raises(ValueError, match="malformed detail"):
        sp.did_summary_plot(bad)


def test_did_summary_plot_rejects_all_nan_estimates():
    import types

    # Marker + estimate column present, but every method failed to fit (all NaN).
    nan_detail = pd.DataFrame(
        {
            "method": ["cs", "etwfe"],
            "estimate": [np.nan, np.nan],
            "ci_low": [np.nan, np.nan],
            "ci_high": [np.nan, np.nan],
        }
    )
    res = types.SimpleNamespace(
        model_info={"_did_summary_marker": True},
        detail=nan_detail,
        estimate=np.nan,
    )
    with pytest.raises(ValueError, match="No successfully-fit methods"):
        sp.did_summary_plot(res)
