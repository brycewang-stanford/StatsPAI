"""Coverage gaps in ``synth.plots``: drawing onto a caller-supplied axes,
the estimator-specific overlays (SDID time-weight shading, prediction /
conformal bands, pre-RMSPE-filtered placebo gaps), the fallbacks when a
result carries less than the classic estimator does, and the short method
labels. Assertions read the artists back from the axes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  headless figure rendering

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.core.results import CausalResult  # noqa: E402
from statspai.synth.plots import _extract_weights, _method_short_label  # noqa: E402

T_TREAT = 18
KW = dict(
    outcome="y", unit="unit", time="time", treated_unit="tr", treatment_time=T_TREAT
)


def _panel(seed=0, n_donors=6, n_t=24, t0=17):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=n_t))
    rows = []
    for j in range(n_donors + 1):
        lam, a = rng.uniform(0.5, 1.5), rng.normal()
        u = "tr" if j == n_donors else f"d{j}"
        for t in range(n_t):
            eff = 2.0 if (u == "tr" and t >= t0) else 0.0
            y = a + lam * f[t] + rng.normal(0, 0.3) + eff
            rows.append((u, t + 1, y, 2 * y + rng.normal(0, 0.3)))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "y2"])


@pytest.fixture(scope="module")
def df():
    return _panel()


@pytest.fixture(scope="module")
def classic(df):
    return sp.synth(df, **KW, placebo=True)


@pytest.fixture(scope="module")
def scpi_res(df):
    return sp.scpi(df, "y", "unit", "time", "tr", T_TREAT, sims=20, seed=1)


@pytest.fixture(scope="module")
def conformal(df):
    # 17 pre-periods: the smallest attainable p-value is 1/18, so only a
    # level above that gives finite intervals
    return sp.conformal_synth(df, **KW, grid_size=41, alpha=0.2)


@pytest.fixture(scope="module")
def gsynth_res(df):
    return sp.synth(df, **KW, method="gsynth", placebo=False)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _own_axes():
    fig, ax = plt.subplots()
    return fig, ax


def _band_range(ax, label):
    band = next(
        c
        for c in ax.collections
        if isinstance(c, PolyCollection) and c.get_label() == label
    )
    ys = band.get_paths()[0].vertices[:, 1]
    return ys.min(), ys.max()


def _span_y(patch):
    """(bottom, top) of an ``axhspan`` (a Rectangle or a Polygon by version)."""
    if isinstance(patch, Rectangle):
        return patch.get_y(), patch.get_y() + patch.get_height()
    ys = patch.get_xy()[:, 1]
    return ys.min(), ys.max()


# ---------------------------------------------------------------------- #
#  Caller-supplied axes
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["weights", "placebo", "placebo_gap", "rmspe"])
def test_classic_plots_draw_on_the_given_axes(classic, kind):
    fig, ax = _own_axes()
    out_fig, out_ax = sp.synthplot(classic, type=kind, ax=ax)
    assert out_ax is ax and out_fig is fig
    assert len(fig.axes) == 1
    assert ax.get_title() != ""
    assert len(ax.patches) + len(ax.lines) > 0


def test_a_list_of_results_plots_its_first_element(classic, gsynth_res):
    _, ax_list = sp.synthplot([classic, gsynth_res], type="gap")
    _, ax_one = sp.synthplot(classic, type="gap")
    np.testing.assert_array_equal(
        ax_list.lines[0].get_ydata(), ax_one.lines[0].get_ydata()
    )
    assert ax_list.get_title() == ax_one.get_title()


def test_top_n_truncates_the_weight_bars(classic):
    names, _ = _extract_weights(classic)
    assert len(names) > 2
    _, ax = sp.synthplot(classic, type="weights", top_n=2)
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(bars) == 2
    shown = {t.get_text() for t in ax.get_yticklabels()}
    assert shown == set(map(str, names[:2]))


def test_factor_and_compare_plots_on_given_axes(classic, gsynth_res):
    fig, ax = _own_axes()
    _, out = sp.synthplot(gsynth_res, type="factors", ax=ax)
    assert out is ax
    # one line per latent factor
    assert len(ax.lines) >= gsynth_res.model_info["n_factors"]

    fig2, ax2 = _own_axes()
    _, out2 = sp.synthplot(
        [classic, gsynth_res], type="compare", ax=ax2, labels=["A", "B"]
    )
    assert out2 is ax2
    legend = [t.get_text() for t in ax2.get_legend().get_texts()]
    assert f"A (ATT={classic.estimate:.2f})" in legend
    assert f"B (ATT={gsynth_res.estimate:.2f})" in legend


def test_conformal_plot_on_given_axes_and_gap_band(conformal):
    fig, ax = _own_axes()
    _, out = sp.synthplot(conformal, type="conformal", ax=ax)
    assert out is ax and len(fig.axes) == 1

    _, gap_ax = sp.synthplot(conformal, type="gap")
    labels = [t.get_text() for t in gap_ax.get_legend().get_texts()]
    assert "80% Conformal CI" in labels
    pr = conformal.model_info["period_results"]
    assert np.isfinite(pr[["ci_lower", "ci_upper"]].to_numpy()).all()
    lo, hi = _band_range(gap_ax, "80% Conformal CI")
    assert lo == pytest.approx(pr["ci_lower"].min())
    assert hi == pytest.approx(pr["ci_upper"].max())


# ---------------------------------------------------------------------- #
#  SCPI: prediction-interval band, dict weights, gap_table fallback
# ---------------------------------------------------------------------- #


def test_scpi_gap_plot_shades_the_prediction_interval(scpi_res):
    _, ax = sp.synthplot(scpi_res, type="gap")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    # u_alpha = e_alpha = 0.05 give a nominal 90% interval
    assert scpi_res.alpha == pytest.approx(0.1)
    assert labels == ["90% Prediction interval"]
    pr = scpi_res.model_info["period_results"]
    lo, hi = _band_range(ax, "90% Prediction interval")
    assert lo == pytest.approx(pr["pi_lower"].min())
    assert hi == pytest.approx(pr["pi_upper"].max())


def test_weights_are_read_from_a_dict_sorted_by_magnitude(scpi_res):
    w = scpi_res.model_info["weights"]
    assert isinstance(w, dict)
    names, vals = _extract_weights(scpi_res)
    kept = {k: v for k, v in w.items() if abs(v) > 1e-6}
    assert set(names) == set(kept)
    assert list(vals) == sorted(kept.values(), key=abs, reverse=True)
    assert vals.sum() == pytest.approx(1.0, abs=1e-5)


def test_prediction_interval_plot_on_given_axes_and_gap_table_fallback(scpi_res):
    fig, ax = _own_axes()
    _, out = sp.synthplot(scpi_res, type="pi", ax=ax)
    assert out is ax
    assert f"{scpi_res.estimate:.3f}" in ax.get_title()

    # a result without period_results is drawn from its gap table
    info = {k: v for k, v in scpi_res.model_info.items() if k != "period_results"}
    bare = CausalResult(
        method="Synthetic Control with Prediction Interval",
        estimand="ATT",
        estimate=scpi_res.estimate,
        se=np.nan,
        pvalue=np.nan,
        ci=(np.nan, np.nan),
        alpha=0.05,
        n_obs=scpi_res.n_obs,
        model_info=info,
    )
    _, ax2 = sp.synthplot(bare, type="prediction_interval")
    assert len(ax2.lines) + len(ax2.collections) + len(ax2.patches) > 0
    assert ax2.get_xlabel() == "Time"


# ---------------------------------------------------------------------- #
#  SDID shading, placebo-gap filter and fallback
# ---------------------------------------------------------------------- #


def test_sdid_trajectory_shades_pre_periods_by_time_weight(df):
    res = sp.sdid(df, **KW, n_reps=3, seed=0)
    tw = res.model_info["time_weights"]
    assert len(tw) == T_TREAT - 1
    _, ax = sp.synthplot(res, type="trajectory")
    spans = [
        p
        for p in ax.patches
        if isinstance(p, Rectangle) and p.get_width() == pytest.approx(0.8)
    ]
    assert len(spans) == len(tw)
    alphas = np.array([p.get_alpha() for p in spans])
    np.testing.assert_allclose(alphas, 0.06 * tw.values / tw.max(), atol=1e-12)


def test_placebo_gap_filter_drops_badly_fitting_placebos(classic):
    mi = classic.model_info
    ratio = np.sqrt(np.array(mi["placebo_pre_mspes"]) / mi["pre_treatment_mspe"])
    thr = float(np.sort(ratio)[2]) + 1e-9  # keep the three best-fitting placebos
    _, ax = sp.synthplot(classic, type="placebo_gap", rmspe_threshold=thr)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Placebos (n=3)" in labels
    assert f"(pre-RMSPE ≤ {thr}×)" in ax.get_title()
    _, ax_all = sp.synthplot(classic, type="placebo_gap")
    all_labels = [t.get_text() for t in ax_all.get_legend().get_texts()]
    assert f"Placebos (n={len(ratio)})" in all_labels


def test_placebo_gap_falls_back_to_the_iqr_of_scalar_placebo_effects(classic):
    info = dict(classic.model_info)
    info["placebo_gaps"] = None
    atts = [-1.0, 0.0, 1.0, 2.0, 3.0]
    info["placebo_atts"] = atts
    bare = CausalResult(
        method=classic.method,
        estimand="ATT",
        estimate=classic.estimate,
        se=classic.se,
        pvalue=classic.pvalue,
        ci=classic.ci,
        alpha=0.05,
        n_obs=classic.n_obs,
        model_info=info,
    )
    _, ax = sp.synthplot(bare, type="placebo_gap")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Placebo IQR" in labels
    span = next(p for p in ax.patches if p.get_label() == "Placebo IQR")
    assert _span_y(span) == (0.0, 2.0)  # quartiles of the five placebo effects


# ---------------------------------------------------------------------- #
#  Staggered, distributional, multi-outcome
# ---------------------------------------------------------------------- #


def _staggered_panel(seed=3):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=16))
    rows = []
    adopt = {"a": 9, "b": 9, "c": 12}
    for u in ["a", "b", "c"] + [f"d{j}" for j in range(6)]:
        lam, base = rng.uniform(0.5, 1.5), rng.normal()
        for t in range(1, 17):
            d = int(u in adopt and t >= adopt[u])
            rows.append((u, t, base + lam * f[t - 1] + rng.normal(0, 0.2) + 2 * d, d))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "d"])


def test_staggered_plot_on_given_axes_and_unit_level_fallback():
    res = sp.staggered_synth(
        _staggered_panel(),
        outcome="y",
        unit="unit",
        time="time",
        treatment="d",
        placebo=False,
    )
    fig, ax = _own_axes()
    _, out = sp.synthplot(res, type="staggered", ax=ax)
    assert out is ax
    assert ax.get_xlabel() == "Cohort (adoption time)"

    unit_df = res.model_info["unit_effects"]
    info = {k: v for k, v in res.model_info.items() if k != "cohort_effects"}
    bare = CausalResult(
        method=res.method,
        estimand="ATT",
        estimate=res.estimate,
        se=res.se,
        pvalue=res.pvalue,
        ci=res.ci,
        alpha=0.05,
        n_obs=res.n_obs,
        model_info=info,
    )
    _, ax2 = sp.synthplot(bare, type="staggered")
    bars = [p for p in ax2.patches if isinstance(p, Rectangle)]
    assert len(bars) == len(unit_df)
    np.testing.assert_allclose(
        sorted(b.get_width() for b in bars), sorted(unit_df["att"])
    )
    assert ax2.get_xlabel() == "ATT"
    # the dashed line marks the overall ATT
    assert any(np.allclose(ln.get_xdata(), res.estimate) for ln in ax2.lines)


def test_multi_outcome_plot_on_given_axes(df):
    res = sp.multi_outcome_synth(
        df,
        outcomes=["y", "y2"],
        unit="unit",
        time="time",
        treated_unit="tr",
        treatment_time=T_TREAT,
        placebo=False,
    )
    fig, ax = _own_axes()
    _, out = sp.synthplot(res, type="multi_outcome", ax=ax)
    assert out is ax
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(bars) == 2
    assert _method_short_label(res) == "Multi-Outcome SCM"


def test_distributional_plot_on_given_axes():
    rng = np.random.default_rng(0)
    rows = []
    for u in [f"u{i}" for i in range(5)] + ["treated"]:
        loc = rng.normal()
        for t in range(1, 7):
            shift = 1.5 if (u == "treated" and t >= 4) else 0.0
            rows += [(u, t, v) for v in loc + shift + rng.normal(size=30)]
    data = pd.DataFrame(rows, columns=["unit", "time", "y"])
    res = sp.discos(
        data,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=4,
        placebo=False,
    )
    fig, ax = _own_axes()
    _, out = sp.synthplot(res, type="distributional", ax=ax)
    assert out is ax and len(ax.lines) >= 1
    assert _method_short_label(res) == "DiSCo"


# ---------------------------------------------------------------------- #
#  Sensitivity panel
# ---------------------------------------------------------------------- #


# ---------------------------------------------------------------------- #
#  Method labels
# ---------------------------------------------------------------------- #


def _stub(method, **info):
    return CausalResult(
        method=method,
        estimand="ATT",
        estimate=0.0,
        se=np.nan,
        pvalue=np.nan,
        ci=(np.nan, np.nan),
        alpha=0.05,
        n_obs=1,
        model_info=info,
    )


@pytest.mark.parametrize(
    "method, info, label",
    [
        ("x", {"inference_method": "conformal"}, "Conformal SCM"),
        ("x", {"n_cohorts": 2}, "Staggered SCM"),
        ("Demeaned Synthetic Control", {}, "De-meaned SCM"),
        ("De-trended SCM", {}, "De-trended SCM"),
        ("Unconstrained Synthetic Control", {}, "Unconstrained SCM"),
        ("Augmented Synthetic Control", {}, "Augmented SCM"),
        ("Generalized Synthetic Control", {}, "GSynth"),
        ("Staggered Synthetic Control", {}, "Staggered SCM"),
        ("Conformal Synthetic Control", {}, "Conformal SCM"),
        ("Synthetic Difference-in-Differences", {}, "SDID"),
        ("Multiple Outcomes SCM", {}, "Multi-Outcome SCM"),
    ],
)
def test_short_method_labels(method, info, label):
    assert _method_short_label(_stub(method, **info)) == label
