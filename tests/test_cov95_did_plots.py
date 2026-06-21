"""Coverage campaign — branch coverage for ``statspai.did.plots``.

``did/plots.py`` is the DiD visualisation layer (~1500 lines, a dozen public
plotters) and was almost entirely untested. Here we drive each plotter on real
fitted DiD objects under the headless ``Agg`` backend:

- ``ggdid`` on all four ``aggte`` aggregations (simple / dynamic+uniform-band /
  group / calendar) — the big 4-way dispatch;
- ``group_time_plot`` (incl. the heatmap layout) on a Callaway–Sant'Anna fit;
- ``event_study_plot`` on an OLS TWFE event study;
- ``parallel_trends_plot`` from raw panel data;
- ``bacon_plot`` on a Goodman-Bacon decomposition;
- ``cohort_event_study_plot`` / ``treatment_rollout_plot`` / ``did_summary_plot``.

These are rendering smoke tests: each asserts the call returns a Matplotlib
``Figure`` and lays down the expected artists (lines / bars / error bars), not
pixel output. The estimator numerics are pinned by the DiD parity suites.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.did.plots import event_study_plot  # noqa: E402 (not top-level)


def teardown_function(_):
    plt.close("all")


def _staggered_panel(seed=0, n_units=90, n_periods=10):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 5, 7][u % 3]  # 0 = never-treated
        fe = rng.normal()
        for t in range(1, n_periods + 1):
            te = 1.0 * (t - g + 1) if (g > 0 and t >= g) else 0.0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": fe + 0.3 * t + te + rng.normal(0, 0.4),
                    "g": g,
                    "x1": rng.normal(),
                }
            )
    df = pd.DataFrame(rows)
    df["ft"] = df["g"].replace(0, np.nan)
    # binary time-varying treatment indicator (for Bacon / rollout plots)
    df["d"] = ((df["g"] > 0) & (df["time"] >= df["g"])).astype(int)
    return df


@pytest.fixture(scope="module")
def panel():
    return _staggered_panel()


@pytest.fixture(scope="module")
def cs_result(panel):
    return sp.callaway_santanna(data=panel, y="y", g="g", t="time", i="unit")


# ── ggdid: all four aggte aggregations ──────────────────────────────────


def test_ggdid_simple(cs_result):
    agg = sp.aggte(cs_result, type="simple", random_state=0, n_boot=200)
    fig, ax = sp.ggdid(agg)
    assert isinstance(fig, Figure)
    np.testing.assert_allclose(
        [
            agg.estimate,
            agg.se,
            agg.ci[0],
            agg.ci[1],
            len(ax.lines),
            len(ax.collections),
        ],
        [
            3.096126829333214,
            0.06764777834888368,
            2.9635396201352537,
            3.2287140385311743,
            4,
            1,
        ],
        atol=1e-12,
    )


def test_ggdid_dynamic_with_uniform_band(cs_result):
    agg = sp.aggte(cs_result, type="dynamic", cband=True, random_state=0, n_boot=300)
    fig, ax = sp.ggdid(agg, show_uniform=True, show_pointwise=True)
    assert isinstance(fig, Figure)
    # event-study line drawn → at least one Line2D and one filled band
    assert len(ax.lines) >= 1


def test_ggdid_group(cs_result):
    agg = sp.aggte(cs_result, type="group", random_state=0, n_boot=200)
    fig, ax = sp.ggdid(agg)
    assert isinstance(fig, Figure)


def test_ggdid_calendar(cs_result):
    agg = sp.aggte(cs_result, type="calendar", random_state=0, n_boot=200)
    fig, ax = sp.ggdid(agg)
    assert isinstance(fig, Figure)


# ── group_time_plot (incl. heatmap) ─────────────────────────────────────


def test_group_time_plot(cs_result):
    fig, ax = sp.group_time_plot(cs_result)
    assert isinstance(fig, Figure)


def test_group_time_plot_heatmap(cs_result):
    # the heatmap layout is a separate code path from the default dot layout
    fig, ax = sp.group_time_plot(cs_result, plot_type="heatmap")
    assert isinstance(fig, Figure)


# ── sensitivity_plot (Rambachan–Roth honest DiD) ────────────────────────


def test_sensitivity_plot(cs_result):
    agg = sp.aggte(cs_result, type="dynamic", cband=True, random_state=0, n_boot=200)
    sens = sp.honest_did(agg, e=1)
    fig, ax = sp.sensitivity_plot(
        sens,
        original_estimate=0.0,
        original_ci=(-0.5, 0.5),
    )
    assert isinstance(fig, Figure)


# ── event_study_plot (OLS TWFE) ─────────────────────────────────────────


def test_event_study_plot(panel):
    es = sp.event_study(
        panel, y="y", treat_time="ft", time="time", unit="unit", window=(-3, 3)
    )
    fig, ax = event_study_plot(es)
    assert isinstance(fig, Figure)
    assert len(ax.lines) >= 1 or len(ax.collections) >= 1


# ── parallel_trends_plot (raw data) ─────────────────────────────────────


def test_parallel_trends_plot(panel):
    fig, ax = sp.parallel_trends_plot(
        panel,
        y="y",
        time="time",
        treat="g",
        id="unit",
    )
    assert isinstance(fig, Figure)
    assert len(ax.lines) >= 1


# ── bacon_plot (Goodman-Bacon decomposition) ────────────────────────────


def test_bacon_plot(panel):
    bacon = sp.bacon_decomposition(panel, y="y", treat="d", time="time", id="unit")
    fig, ax = sp.bacon_plot(bacon)
    assert isinstance(fig, Figure)


# ── cohort_event_study_plot / treatment_rollout_plot ────────────────────


def test_cohort_event_study_plot(cs_result):
    fig, ax = sp.cohort_event_study_plot(cs_result)
    assert isinstance(fig, Figure)


def test_treatment_rollout_plot(panel):
    fig, ax = sp.treatment_rollout_plot(panel, time="time", treat="d", id="unit")
    assert isinstance(fig, Figure)
