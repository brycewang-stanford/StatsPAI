"""Coverage campaign — specialised ``sp.synthplot`` renderers (``synth/plots.py``).

Part of the core-module ≥95% coverage initiative (see
``.coverage_campaign/CAMPAIGN.md``). ``test_synth_cov_plots.py`` already covers
the common plot types (trajectory / gap / both / weights / placebo / placebo_gap
/ rmspe / conformal / compare). This file closes the *specialised* renderers
that need a matching estimator fit before the ``synthplot(type=...)`` dispatch
reaches them:

* ``type='staggered'``         → ``_plot_staggered``  (cohort ATTs from staggered_synth)
* ``type='factors'``           → ``_plot_factors``    (latent factor paths from gsynth)
* ``type='distributional'``    → ``_plot_distributional`` (quantile effects from discos)
* ``type='multi_outcome'``     → ``_plot_multi_outcome``  (per-outcome ATTs)
* ``type='prediction_interval'`` → ``_plot_prediction_interval`` (scpi PI bands)
* ``type='sensitivity'``       → ``_plot_sensitivity`` (2×2 LOO/placebo/donor/RMSPE panel)

Assertions are structural-but-real: each renderer must return a Matplotlib
``Figure`` whose axes actually carry the data the renderer claims to draw (bars,
lines, the overall-ATT reference line, the 2×2 panel grid). They are rendering
contracts, not pixel pins — no estimator numerics are exercised for correctness
here (that is the parity suite's job), only that every dispatch branch and its
renderer run end-to-end on a real fitted result.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import statspai as sp

T_TREAT = 11


# --------------------------------------------------------------------------
# Panel builders
# --------------------------------------------------------------------------
def _panel(seed=0, n_donors=8, n_t=20, effect=4.0):
    """Standard single-treated SCM panel ('treated' unit + donor pool)."""
    rng = np.random.default_rng(seed)
    units = [f"u{i}" for i in range(n_donors)] + ["treated"]
    rows = []
    for u in units:
        base = rng.normal(0, 1)
        fe = rng.normal(0, 0.5)
        for t in range(1, n_t + 1):
            eff = effect if (u == "treated" and t >= T_TREAT) else 0.0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": base + 0.2 * t + fe + eff + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


def _staggered_panel(seed=0, n_control=4, effect=6.0):
    """Two adopting cohorts + never-treated controls, with a 0/1 treat column."""
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2018)
    rows = []
    controls = {}
    for c in range(n_control):
        s = 20 + c + np.cumsum(rng.normal(0, 0.5, len(years)))
        controls[f"c{c}"] = s
        for i, yr in enumerate(years):
            rows.append((f"c{c}", yr, s[i], 0))
    cohort_map = {"t_early": 2008, "t_early2": 2008, "t_late": 2012}
    for u, g in cohort_map.items():
        base = (
            0.5 * controls["c0"] + 0.5 * controls["c1"] + rng.normal(0, 0.3, len(years))
        )
        for i, yr in enumerate(years):
            tr = 1 if yr >= g else 0
            rows.append((u, yr, base[i] + (effect if tr else 0.0), tr))
    return pd.DataFrame(rows, columns=["unit", "year", "y", "treat"])


def _multi_outcome_panel(seed=0, n_donors=8, n_t=20, effect=4.0):
    """Two correlated outcomes y1, y2 sharing the donor structure."""
    rng = np.random.default_rng(seed)
    units = [f"u{i}" for i in range(n_donors)] + ["treated"]
    rows = []
    for u in units:
        base = rng.normal(0, 1)
        for t in range(1, n_t + 1):
            tr = u == "treated" and t >= T_TREAT
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y1": base + 0.2 * t + (effect if tr else 0.0) + rng.normal(0, 0.3),
                    "y2": base
                    + 0.1 * t
                    + (0.5 * effect if tr else 0.0)
                    + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


def _dist_panel(seed=0, n_donors=6, n_t=8, n_per=40):
    """Repeated cross-section panel for distributional SCM (discos)."""
    rng = np.random.default_rng(seed)
    units = [f"u{i}" for i in range(n_donors)] + ["treated"]
    rows = []
    for u in units:
        loc = rng.normal(0, 1)
        for t in range(1, n_t + 1):
            shift = 1.5 if (u == "treated" and t >= 5) else 0.0
            for _ in range(n_per):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": loc + 0.1 * t + shift + rng.normal(0, 1),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _is_fig(obj):
    return isinstance(obj, plt.Figure)


# --------------------------------------------------------------------------
# type='staggered' → _plot_staggered
# --------------------------------------------------------------------------
def test_synthplot_staggered_cohort_bars():
    res = sp.staggered_synth(
        data=_staggered_panel(seed=1),
        outcome="y",
        unit="unit",
        time="year",
        treatment="treat",
        method="separate",
        placebo=False,
    )
    fig, ax = sp.synthplot(res, type="staggered")
    assert _is_fig(fig)
    # cohort bar chart drawn + overall-ATT reference line present
    assert len(ax.patches) >= 1  # cohort bars
    assert len(ax.get_lines()) >= 1  # axhline(s) incl. overall ATT
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Overall ATT" in s for s in labels)


# --------------------------------------------------------------------------
# type='factors' → _plot_factors  (gsynth latent factors)
# --------------------------------------------------------------------------
def test_synthplot_factors_from_gsynth():
    res = sp.gsynth(
        _panel(seed=2),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
    )
    mi = res.model_info
    if mi.get("factors_pre") is None:
        pytest.skip("gsynth fit produced no latent factors on this DGP")
    fig, ax = sp.synthplot(res, type="factors")
    assert _is_fig(fig)
    # one line per latent factor
    n_factors = mi["factors_pre"].shape[1]
    assert len(ax.get_lines()) >= n_factors
    assert ax.get_ylabel() == "Factor Value"


# --------------------------------------------------------------------------
# type='distributional' → _plot_distributional  (discos quantile effects)
# --------------------------------------------------------------------------
def test_synthplot_distributional_from_discos():
    res = sp.discos(
        _dist_panel(seed=0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=5,
        method="mixture",
        n_quantiles=40,
        placebo=False,
        seed=1,
    )
    fig, ax = sp.synthplot(res, type="distributional")
    assert _is_fig(fig)
    # the quantile-effect curve spans the [0, 1] tau axis
    line = ax.get_lines()[0]
    xs = line.get_xdata()
    assert xs.min() >= 0.0 and xs.max() <= 1.0
    assert ax.get_xlabel().startswith("Quantile")


# --------------------------------------------------------------------------
# type='multi_outcome' → _plot_multi_outcome
# --------------------------------------------------------------------------
def test_synthplot_multi_outcome_per_outcome_bars():
    res = sp.multi_outcome_synth(
        _multi_outcome_panel(seed=3),
        outcomes=["y1", "y2"],
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        placebo=False,
    )
    fig, ax = sp.synthplot(res, type="multi_outcome")
    assert _is_fig(fig)
    # horizontal bars: one per outcome (2)
    yt = [t.get_text() for t in ax.get_yticklabels()]
    assert {"y1", "y2"}.issubset(set(yt))
    assert ax.get_xlabel() == "ATT"


# --------------------------------------------------------------------------
# type='prediction_interval' → _plot_prediction_interval  (scpi)
# --------------------------------------------------------------------------
def test_synthplot_prediction_interval_from_scpi():
    res = sp.scpi(
        _panel(seed=0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        seed=1,
    )
    fig, ax = sp.synthplot(res, type="prediction_interval")
    assert _is_fig(fig)
    # period-effect line drawn across time
    assert len(ax.get_lines()) >= 1
    assert ax.get_ylabel() == "Treatment Effect"


def test_synthplot_pi_alias():
    """'pi' is the short alias of 'prediction_interval'."""
    res = sp.scpi(
        _panel(seed=4),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        seed=2,
    )
    fig, ax = sp.synthplot(res, type="pi")
    assert _is_fig(fig)


# --------------------------------------------------------------------------
# _plot_sensitivity  (2×2 multi-panel)
# --------------------------------------------------------------------------
# ``synthplot`` reads ``result.model_info`` before dispatching, so the
# 2×2 sensitivity panel renderer is reached either with a CausalResult whose
# ``model_info['sensitivity']`` carries the dict, or by passing the
# ``synth_sensitivity()`` dict straight to the renderer (its documented dict
# entry point). We exercise the dict path directly.
def test_plot_sensitivity_2x2_panel_from_dict():
    from statspai.synth.plots import _plot_sensitivity

    sens = sp.synth_sensitivity(
        _panel(seed=0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        n_donor_samples=15,
        seed=0,
    )
    assert isinstance(sens, dict)
    fig, axes = _plot_sensitivity(sens)
    assert _is_fig(fig)
    # 2×2 grid of axes
    assert np.asarray(axes).shape == (2, 2)
    # default suptitle
    assert "Sensitivity" in fig._suptitle.get_text()


def test_plot_sensitivity_via_result_model_info():
    """The renderer's CausalResult branch: model_info['sensitivity'] holds the
    dict (the path ``synthplot(type='sensitivity')`` takes for a result)."""
    from statspai.synth.plots import _plot_sensitivity

    sens = sp.synth_sensitivity(
        _panel(seed=1),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        n_donor_samples=12,
        seed=1,
    )
    base = sp.synth(
        _panel(seed=1),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
    )
    base.model_info["sensitivity"] = sens
    fig, axes = _plot_sensitivity(base)
    assert _is_fig(fig)
    assert np.asarray(axes).shape == (2, 2)
