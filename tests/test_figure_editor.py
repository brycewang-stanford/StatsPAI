"""Figure editor (``sp.interactive`` / ``FigureEditor``).

The editor promises three things; each is checked here, pixel for pixel, on
every registered operation and on random sequences of operations:

1. ``generate_code()`` run on a freshly drawn copy of the figure gives the
   edited figure.
2. ``undo()`` / ``reset()`` give back the original figure.
3. A saved edit list replayed on a fresh copy gives the edited figure.

Plus the data-protection rules (locked content is never hidden, clipped or
made invisible), the artist classification on real StatsPAI plots, and that
the editor never changes global matplotlib settings.
"""

from __future__ import annotations

import io
import json
import random
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.plots._editor_ops import OPS  # noqa: E402
from statspai.plots.interactive import ArtistRole, FigureEditor  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    rc = mpl.rcParams.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield
    plt.close("all")
    mpl.rcParams.update(rc)


# ----------------------------------------------------------------------
# Figure factories (each call draws an identical, fresh figure)
# ----------------------------------------------------------------------


def lines_fig():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(10)
    ax.plot(x, x**1.1, "o-", label="Treated")
    ax.plot(x, 0.5 * x, label="Control")
    ax.fill_between(x, 0.5 * x - 1, 0.5 * x + 1, alpha=0.2, color="C1")
    ax.axhline(0, ls="--", color="k")
    ax.text(1, 6, "note")
    ax.set_title("Outcome")
    ax.set_xlabel("Year")
    ax.set_ylabel("y")
    ax.legend()
    return fig


def errorbar_fig():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(
        [-2, -1, 0, 1, 2],
        [0.1, 0.0, 0.5, 0.8, 1.0],
        yerr=0.3,
        fmt="o",
        capsize=3,
        color="C0",
        label="ATT",
    )
    ax.axvline(-0.5, ls=":", color="gray")
    ax.legend()
    return fig


def bars_fig():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["a", "b", "c"], [1, 5, 2], color="C2")
    return fig


def two_axes_fig():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7, 3))
    a0.plot([1, 2, 3], [1, 4, 9], label="left")
    a1.scatter([1, 2, 3], [3, 2, 1], color="C3", label="right")
    a0.legend()
    a1.set_title("Right")
    return fig


@pytest.fixture(scope="module")
def cs_dynamic():
    df = sp.datasets.mpdta()
    res = sp.callaway_santanna(df, y="lemp", g="first_treat", t="year", i="countyreal")
    return sp.aggte(res, type="dynamic")


def pixels(fig) -> np.ndarray:
    fig.set_dpi(fig.dpi)  # no-op; keeps the figure's own dpi
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def same(a, b) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def run_code(factory, code):
    fig = factory()
    exec(code, {"fig": fig})  # noqa: S102 - the code under test
    return fig


# ----------------------------------------------------------------------
# One case per operation: (factory, op name, params)
# ----------------------------------------------------------------------

CASES = [
    (lines_fig, "set_title", {"text": "New title"}),
    (lines_fig, "set_title", {"text": "Styled", "kwargs": {"color": "red"}}),
    (lines_fig, "set_xlabel", {"text": "Calendar year"}),
    (lines_fig, "set_ylabel", {"text": "Log employment"}),
    (lines_fig, "set_suptitle", {"text": "Figure 1"}),
    (lines_fig, "set_xlim", {"left": -1, "right": 12}),
    (lines_fig, "set_ylim", {"bottom": -3, "top": 15}),
    (lines_fig, "set_fontsize", {"target": "title", "size": 18}),
    (lines_fig, "set_fontsize", {"target": "ticks", "size": 14}),
    (lines_fig, "set_fontsize", {"target": "legend", "size": 13}),
    (lines_fig, "set_font", {"font_family": "serif", "font_name": "DejaVu Serif"}),
    (lines_fig, "apply_font_preset", {"preset": "Helvetica"}),
    (lines_fig, "apply_size_preset", {"preset": "Slides"}),
    (lines_fig, "set_title_weight", {"weight": "bold"}),
    (lines_fig, "set_color", {"target": "title", "color": "#aa0000"}),
    (lines_fig, "set_color", {"target": "line1", "color": "purple"}),
    (lines_fig, "set_series_color", {"group": 1, "color": "#009E73"}),
    (lines_fig, "apply_palette", {"palette": "okabe_ito"}),
    (
        two_axes_fig,
        "set_scatter_color",
        {"scatter_index": 0, "color": "green", "ax_index": 1},
    ),
    (lines_fig, "set_background_color", {"color": "#f4f4f4", "target": "axes"}),
    (lines_fig, "set_background_color", {"color": "#fffff0", "target": "figure"}),
    (lines_fig, "set_linewidth", {"line_index": 0, "width": 3.0}),
    (lines_fig, "set_linestyle", {"line_index": 1, "style": ":"}),
    (lines_fig, "set_marker", {"line_index": 0, "marker": "s"}),
    (lines_fig, "set_markersize", {"line_index": 0, "size": 11}),
    (lines_fig, "set_alpha", {"target": "ci0", "alpha": 0.5}),
    (lines_fig, "set_spine_visible", {"spine": "top", "visible": False}),
    (lines_fig, "set_grid", {"visible": True}),
    (lines_fig, "set_grid_style", {"color": "red", "alpha": 0.4, "linestyle": "--"}),
    (lines_fig, "set_tick_rotation", {"axis": "x", "angle": 45}),
    (lines_fig, "set_figsize", {"width": 6.5, "height": 4.5}),
    (lines_fig, "set_dpi", {"dpi": 120}),
    (lines_fig, "tight_layout", {}),
    (lines_fig, "set_legend", {"loc": "lower right", "fontsize": 12, "frameon": False}),
    (lines_fig, "set_legend_visible", {"visible": False}),
    (lines_fig, "set_legend_labels", {"labels": ["Treated (CA)", "Control"]}),
    (lines_fig, "add_annotation", {"text": "policy", "xy": [3, 4]}),
    (lines_fig, "set_text", {"index": 0, "text": "edited note"}),
    (lines_fig, "set_text_visible", {"index": 0, "visible": False}),
    (lines_fig, "apply_theme", {"theme": "ggplot"}),
    (lines_fig, "apply_theme", {"theme": "academic"}),
    (lines_fig, "apply_theme", {"theme": "dark_background", "recolor": False}),
    (errorbar_fig, "set_series_color", {"group": 0, "color": "#D55E00"}),
    (errorbar_fig, "set_ylim", {"bottom": 0.2, "top": 0.6}),
    (two_axes_fig, "set_title", {"text": "Left", "ax_index": 0}),
    (two_axes_fig, "set_xlabel", {"text": "x right", "ax_index": 1}),
]


def _legend_from_scratch():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([1, 2, 3], label="series")
    return fig


CASES.append((_legend_from_scratch, "set_legend_visible", {"visible": True}))


def test_every_operation_has_a_case():
    covered = {name for _, name, _ in CASES}
    assert covered == set(OPS), sorted(set(OPS) - covered)


def _case_id(case):
    factory, name, params = case
    return f"{name}-{factory.__name__}-{'-'.join(map(str, params.values()))[:30]}"


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_operation_invariants(case):
    factory, name, params = case
    rc_before = {k: str(v) for k, v in mpl.rcParams.items()}

    fig = factory()
    original = pixels(fig)
    ed = FigureEditor(fig)
    ed.run(name, **params)
    edited = pixels(fig)
    assert not same(original, edited), f"{name} changed nothing"

    # 1. code reproduces the edit on a fresh figure
    code = ed.generate_code()
    assert same(pixels(run_code(factory, code)), edited), code

    # 3. JSON replay on a fresh figure
    fresh = factory()
    FigureEditor(fresh).apply_edits_json(ed.to_edits_json())
    assert same(pixels(fresh), edited)

    # 2. undo is exact; redo returns the edit
    ed.undo()
    assert same(pixels(fig), original)
    ed.redo()
    assert same(pixels(fig), edited)
    ed.reset()
    assert same(pixels(fig), original)
    assert ed.edits == []

    # global matplotlib settings untouched
    assert {k: str(v) for k, v in mpl.rcParams.items()} == rc_before


SEQUENCE_POOL = [
    ("set_title", lambda r: {"text": f"T{r.randint(0, 9)}"}),
    (
        "set_fontsize",
        lambda r: {
            "target": r.choice(["title", "xlabel", "ticks", "legend"]),
            "size": float(r.randint(6, 20)),
        },
    ),
    ("set_grid", lambda r: {"visible": r.random() < 0.5}),
    ("set_grid_style", lambda r: {"color": r.choice(["red", "#999999"]), "alpha": 0.5}),
    (
        "set_series_color",
        lambda r: {"group": r.randint(0, 1), "color": r.choice(["#0072B2", "#E69F00"])},
    ),
    ("set_color", lambda r: {"target": "line0", "color": r.choice(["k", "#CC79A7"])}),
    ("set_ylim", lambda r: {"bottom": r.uniform(-5, 0), "top": r.uniform(5, 20)}),
    ("set_xlim", lambda r: {"left": r.uniform(-3, 1), "right": r.uniform(6, 14)}),
    (
        "set_spine_visible",
        lambda r: {"spine": r.choice(["top", "right"]), "visible": r.random() < 0.5},
    ),
    ("set_legend", lambda r: {"loc": r.choice(["upper left", "lower right"])}),
    ("set_legend_labels", lambda r: {"labels": {"0": f"S{r.randint(0, 9)}"}}),
    ("apply_theme", lambda r: {"theme": r.choice(["ggplot", "aea", "minimal"])}),
    ("set_font", lambda r: {"font_family": r.choice(["serif", "sans-serif"])}),
    ("apply_size_preset", lambda r: {"preset": r.choice(["Thesis", "Slides"])}),
    (
        "set_tick_rotation",
        lambda r: {"axis": "x", "angle": float(r.choice([0, 30, 90]))},
    ),
    (
        "add_annotation",
        lambda r: {"text": "a", "xy": [r.uniform(0, 9), r.uniform(0, 9)]},
    ),
    (
        "set_figsize",
        lambda r: {"width": r.choice([4.0, 6.0]), "height": r.choice([3.0, 4.0])},
    ),
    ("tight_layout", lambda r: {}),
    ("set_alpha", lambda r: {"target": "ci0", "alpha": r.uniform(0.05, 1)}),
    ("set_markersize", lambda r: {"line_index": 0, "size": float(r.randint(2, 12))}),
]


@pytest.mark.parametrize("seed", range(12))
def test_random_sequences_keep_all_three_guarantees(seed):
    r = random.Random(seed)
    fig = lines_fig()
    original = pixels(fig)
    ed = FigureEditor(fig)
    for _ in range(r.randint(3, 8)):
        name, gen = r.choice(SEQUENCE_POOL)
        ed.run(name, **gen(r))
    edited = pixels(fig)
    assert same(
        pixels(run_code(lines_fig, ed.generate_code())), edited
    ), ed.generate_code()
    fresh = lines_fig()
    FigureEditor(fresh).apply_edits_json(ed.to_edits_json())
    assert same(pixels(fresh), edited)
    # undo one step at a time back to the start
    for _ in range(len(ed.edits)):
        ed.undo()
    assert same(pixels(fig), original)


# ----------------------------------------------------------------------
# Code generation details
# ----------------------------------------------------------------------


def test_later_edit_of_same_slot_wins_in_order():
    ed = FigureEditor(lines_fig())
    ed.set_grid(True)
    ed.set_grid_style(color="red")
    ed.set_grid(False)
    code = ed.generate_code(include_comment=False)
    assert code.splitlines()[-1] == "ax.grid(False)"
    assert "red" not in code


def test_two_legend_edits_both_survive():
    ed = FigureEditor(lines_fig())
    ed.set_legend(loc="lower right")
    ed.set_legend(fontsize=14)
    code = ed.generate_code()
    assert "set_loc('lower right')" in code and "set_fontsize(14.0)" in code


def test_code_defines_its_own_axes_variables():
    ed = FigureEditor(two_axes_fig())
    ed.set_title("R", ax_index=1)
    code = ed.generate_code(include_comment=False)
    assert code.splitlines()[0] == "ax1 = fig.axes[1]"
    # runs with nothing but `fig` in scope
    exec(code, {"fig": two_axes_fig()})  # noqa: S102


def test_no_unrequested_tight_layout():
    ed = FigureEditor(lines_fig())
    ed.set_title("x")
    assert "tight_layout" not in ed.generate_code()


def test_title_edit_keeps_title_size():
    fig = lines_fig()
    ed = FigureEditor(fig)
    ed.set_fontsize("title", 20)
    ed.set_title("New")
    assert fig.axes[0].title.get_fontsize() == 20


def test_json_code_field_is_never_trusted():
    fig = lines_fig()
    ed = FigureEditor(fig)
    payload = [
        {
            "op": "set_title",
            "params": {"text": "x"},
            "code": "__import__('os').system('echo pwned')",
        }
    ]
    ed.apply_edits_json(json.dumps(payload))
    assert "os" not in ed.generate_code()
    legacy = [
        {
            "target": "ax.title",
            "property": "text",
            "old": "",
            "new": "y",
            "code": "__import__('os').system('echo pwned')",
        }
    ]
    ed.apply_edits_list(legacy)
    assert "system" not in ed.generate_code()
    assert fig.axes[0].get_title() == "y"


def test_version1_records_replay():
    v1 = [
        {
            "target": "ax.title",
            "property": "text",
            "old": "",
            "new": "Old format",
            "code": "",
        },
        {
            "target": "ax.xlim",
            "property": "range",
            "old": [0, 1],
            "new": [-1, 12],
            "code": "",
        },
        {
            "target": "ax.spines.top",
            "property": "visible",
            "old": True,
            "new": False,
            "code": "",
        },
        {
            "target": "ax.line0",
            "property": "linewidth",
            "old": 1.5,
            "new": 3.0,
            "code": "",
        },
        {"target": "theme", "property": "name", "old": None, "new": "aea", "code": ""},
        {"target": "fig", "property": "save", "old": None, "new": "x.png", "code": ""},
    ]
    fig = lines_fig()
    ed = FigureEditor(fig)
    ed.import_state({"version": 1, "edits": v1, "figure": {}})
    ax = fig.axes[0]
    assert ax.get_title() == "Old format"
    assert not ax.spines["top"].get_visible()
    assert ax.get_lines()[0].get_linewidth() == 3.0
    assert [e.op for e in ed.edits] == [
        "set_title",
        "set_xlim",
        "set_spine_visible",
        "set_linewidth",
        "apply_theme",
    ]


def test_state_roundtrip_v2():
    fig = lines_fig()
    ed = FigureEditor(fig)
    ed.set_series_color(0, "#D55E00")
    ed.set_legend_labels(["A", "B"])
    state = json.loads(json.dumps(ed.export_state()))
    assert state["version"] == 2
    fresh = lines_fig()
    FigureEditor(fresh).import_state(state)
    assert same(pixels(fresh), pixels(fig))


def test_style_transfers_to_another_figure():
    """A saved style (Stata's .grec) applies to a different figure; edits
    that do not fit it are skipped with a warning."""
    ed = FigureEditor(lines_fig())
    ed.apply_theme("aea")
    ed.set_fontsize("title", 16)
    ed.set_linewidth(1, 2.5)
    other = bars_fig()  # no lines at all
    with pytest.warns(UserWarning, match="Skipped edit set_linewidth"):
        FigureEditor(other).apply_edits_json(ed.to_edits_json())
    assert other.axes[0].title.get_fontsize() == 16


# ----------------------------------------------------------------------
# Data protection
# ----------------------------------------------------------------------


def test_ylim_cannot_clip_error_bars():
    fig = errorbar_fig()
    ed = FigureEditor(fig)
    with pytest.warns(UserWarning, match="would hide data"):
        ed.set_ylim(0.2, 0.6)
    lo, hi = fig.axes[0].get_ylim()
    assert lo <= -0.3 and hi >= 1.3  # error bars span [-0.3, 1.3]


def test_ylim_cannot_clip_confidence_band():
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 20)
    ax.plot(x, x)
    ax.fill_between(x, x - 3, x + 3, alpha=0.2)
    FigureEditor(fig).set_ylim(-0.5, 1.5)
    lo, hi = ax.get_ylim()
    assert lo <= -3 and hi >= 4


def test_ylim_cannot_clip_bars():
    fig = bars_fig()
    FigureEditor(fig).set_ylim(0, 2)
    assert fig.axes[0].get_ylim()[1] >= 5


def test_limits_respect_inverted_and_log_axes():
    fig, ax = plt.subplots()
    ax.plot([1, 10, 100], [1, 2, 3])
    ax.set_xscale("log")
    FigureEditor(fig).set_xlim(5, 50)
    lo, hi = ax.get_xlim()
    assert lo < 1 and hi > 100 and lo > 0
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 5, 10])
    FigureEditor(fig).set_ylim(8, 2)  # inverted request
    lo, hi = ax.get_ylim()
    assert lo > hi and lo >= 10 and hi <= 0


def test_limits_wider_than_data_are_kept_exactly():
    fig = lines_fig()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FigureEditor(fig).set_xlim(-5, 20)
    assert fig.axes[0].get_xlim() == (-5.0, 20.0)


def test_locked_alpha_has_a_floor():
    fig = lines_fig()
    ed = FigureEditor(fig)
    with pytest.warns(UserWarning, match="nearly invisible"):
        ed.set_alpha("line0", 0.0)
    assert fig.axes[0].get_lines()[0].get_alpha() == 0.2
    ed.set_alpha("ci0", 0.0)
    assert fig.axes[0].collections[0].get_alpha() == 0.05


def test_data_cannot_be_made_invisible():
    fig = errorbar_fig()  # data line: markers only
    ed = FigureEditor(fig)
    with pytest.raises(ValueError, match="invisible"):
        ed.set_marker(0, "None")
    with pytest.raises(ValueError, match="invisible"):
        ed.set_markersize(0, 0)
    fig = lines_fig()  # line 1: no markers
    with pytest.raises(ValueError, match="invisible"):
        FigureEditor(fig).set_linestyle(1, "None")
    # the figure is unchanged after a refused edit
    assert fig.axes[0].get_lines()[1].get_linestyle() == "-"


def test_unprotected_editor_allows_it():
    fig = lines_fig()
    FigureEditor(fig, protect_data=False).set_linestyle(1, "None")
    assert fig.axes[0].get_lines()[1].get_linestyle() == "None"


def test_colormapped_colours_are_locked():
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 2, 3], c=[0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="colour-mapped"):
        FigureEditor(fig).set_scatter_color(0, "red")


def test_numeric_result_labels_keep_their_wording():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.text(0.5, 0.5, "0.124***")
    ed = FigureEditor(fig)
    with pytest.raises(ValueError, match="locked"):
        ed.set_text(0, "0.500***")
    ed.set_text_visible(0, False)  # hiding is allowed
    assert not ax.texts[0].get_visible()


def test_failed_edit_leaves_figure_untouched():
    fig = lines_fig()
    before = pixels(fig)
    ed = FigureEditor(fig)
    with pytest.raises(ValueError):
        ed.set_legend_labels(["a", "b", "c"])  # three labels, two entries
    assert same(pixels(fig), before) and ed.edits == []


def test_unknown_parameters_and_actions_fail_loudly():
    ed = FigureEditor(lines_fig())
    with pytest.raises(ValueError, match="unknown parameter"):
        ed.run("set_title", text="x", colour="red")
    with pytest.raises(ValueError, match="Unknown action"):
        ed.apply_actions([{"action": "delete_data"}])
    with pytest.raises(ValueError, match="must be one of"):
        ed.run("set_spine_visible", spine="middle", visible=False)
    with pytest.raises(ValueError, match="not a matplotlib colour"):
        ed.set_color("title", "notacolour")
    with pytest.raises(ValueError, match="out of range"):
        ed.set_title("x", ax_index=3)


# ----------------------------------------------------------------------
# Classification on real StatsPAI figures
# ----------------------------------------------------------------------


def _roles(fig):
    ed = FigureEditor(fig)
    ax = fig.axes[0]
    return ed, ax


def test_event_study_classification(cs_dynamic):
    fig, ax = cs_dynamic.plot()
    ed = FigureEditor(fig)
    from matplotlib.container import ErrorbarContainer

    containers = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    for c in containers:
        data_line, caps, bars = c.lines
        if data_line is not None:
            assert ed.artist_roles[id(data_line)] is ArtistRole.DATA
        assert all(ed.artist_roles[id(cap)] is ArtistRole.CI for cap in caps)
        assert all(ed.artist_roles[id(b)] is ArtistRole.CI for b in bars)
    # the protected range covers every confidence interval drawn
    lo, hi = ed._get_data_range(ax, "y")
    ci_lo = float(np.min(cs_dynamic.detail["ci_lower"]))
    ci_hi = float(np.max(cs_dynamic.detail["ci_upper"]))
    assert lo <= ci_lo + 1e-12 and hi >= ci_hi - 1e-12


def test_event_study_ylim_keeps_every_interval(cs_dynamic):
    fig, ax = cs_dynamic.plot()
    FigureEditor(fig).set_ylim(-0.01, 0.01)
    lo, hi = ax.get_ylim()
    assert lo <= float(np.min(cs_dynamic.detail["ci_lower"]))
    assert hi >= float(np.max(cs_dynamic.detail["ci_upper"]))


def test_event_study_series_recolour_reproduces(cs_dynamic):
    def factory():
        return cs_dynamic.plot()[0]

    fig = factory()
    ed = FigureEditor(fig)
    assert ed.color_groups(0), "event study has at least one series"
    ed.apply_palette("okabe_ito")
    ed.apply_theme("aea")
    edited = pixels(fig)
    assert same(pixels(run_code(factory, ed.generate_code())), edited)
    ed.reset()
    assert same(pixels(fig), pixels(factory()))


def test_constant_data_series_stays_locked():
    fig, ax = plt.subplots()
    ax.plot(range(10), np.arange(10) ** 1.2, label="Treatment group")
    ax.plot(range(10), np.full(10, 3.0), label="Baseline mean")  # flat but data
    ax.axhline(0, color="k")
    ax.plot([0, 9], [5, 5], "--")  # drawn reference segment
    ed = FigureEditor(fig)
    roles = [ed.artist_roles[id(ln)] for ln in ax.get_lines()]
    assert roles[0] is ArtistRole.DATA
    assert roles[1] is ArtistRole.REFERENCE  # flat *and* labelled baseline
    assert roles[2] is ArtistRole.REFERENCE
    assert roles[3] is ArtistRole.REFERENCE
    fig, ax = plt.subplots()
    ax.plot(range(10), np.full(10, 3.0), label="Group mean")
    assert FigureEditor(fig).artist_roles[id(ax.get_lines()[0])] is ArtistRole.DATA


@pytest.mark.parametrize("maker", ["binscatter", "rdplot", "coefplot"])
def test_statspai_plots_reproduce(maker):
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": 1 + 0.5 * x + rng.normal(size=n)})

    def factory():
        if maker == "binscatter":
            return sp.binscatter(df, y="y", x="x")[0]
        if maker == "rdplot":
            xr = np.random.default_rng(1).uniform(-1, 1, n)
            yr = (
                0.5 * (xr > 0) + xr + np.random.default_rng(2).normal(scale=0.3, size=n)
            )
            return sp.rdplot(pd.DataFrame({"y": yr, "x": xr}), y="y", x="x", c=0)[0]
        return sp.coefplot(sp.regress("y ~ x", data=df))[0]

    fig = factory()
    ed = FigureEditor(fig)
    ed.set_title("Edited")
    ed.apply_font_preset("Times New Roman")
    ed.apply_palette("grayscale")
    ed.set_ylim(*fig.axes[0].get_ylim())
    edited = pixels(fig)
    assert same(pixels(run_code(factory, ed.generate_code())), edited)
    ed.reset()
    assert same(pixels(fig), pixels(factory()))


# ----------------------------------------------------------------------
# Editor behaviour
# ----------------------------------------------------------------------


def test_coalesce_merges_slider_drags_into_one_undo_step():
    fig = lines_fig()
    original = pixels(fig)
    ed = FigureEditor(fig, coalesce=True)
    for size in (12, 14, 16, 18):
        ed.set_fontsize("title", size)
    ed.set_title("x")
    assert len(ed.edits) == 2
    ed.undo()
    ed.undo()
    assert same(pixels(fig), original)


def test_batch_refreshes_once():
    ed = FigureEditor(lines_fig())
    calls = []
    ed.on_refresh(lambda f: calls.append(1))
    ed.apply_actions(
        [
            {"action": "set_title", "text": "a"},
            {"action": "set_line_color", "line_index": 0, "color": "red"},
            {"action": "apply_theme", "theme": "aea"},
        ]
    )
    assert len(calls) == 1 and len(ed.edits) == 3


def test_new_edit_clears_redo():
    ed = FigureEditor(lines_fig())
    ed.set_title("a")
    ed.undo()
    ed.set_title("b")
    ed.redo()  # nothing to redo
    assert [e.params["text"] for e in ed.edits] == ["b"]


def test_schema_lists_operations_and_groups():
    ed = FigureEditor(lines_fig())
    schema = json.loads(json.dumps(ed.get_editable_schema(), default=str))
    assert {o["name"] for o in schema["operations"]} == set(OPS)
    ax0 = schema["axes"][0]
    assert len(ax0["color_groups"]) == 2
    assert ax0["texts"][0]["wording_locked"] is False


def test_interactive_headless_leaves_figure_alone():
    fig = lines_fig()
    before = pixels(fig)
    n_axes = len(fig.axes)
    ed = sp.interactive(fig)
    assert len(fig.axes) == n_axes and same(pixels(fig), before)
    assert sp.interactive(fig) is ed  # same editor, same history


def test_interactive_actions_and_get_code():
    fig = lines_fig()
    ed = sp.interactive(fig, actions=[{"action": "set_title", "text": "Agent"}])
    assert fig.axes[0].get_title() == "Agent"
    assert sp.get_code(fig) == ed.generate_code()
    with pytest.raises(ValueError, match="ui must be"):
        sp.interactive(fig, ui="browser")


def test_interactive_jupyter_without_ipywidgets_warns(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "ipywidgets":
            raise ImportError("no ipywidgets")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.warns(UserWarning, match="needs ipywidgets"):
        sp.interactive(lines_fig(), ui="jupyter")


def test_save_embeds_truetype_fonts(tmp_path):
    ed = FigureEditor(lines_fig())
    path = tmp_path / "fig.pdf"
    ed.save(str(path))
    pdf = path.read_bytes()
    assert b"/FontFile2" in pdf and b"/Type3" not in pdf
    paths = ed.export(str(tmp_path / "fig"), formats=("pdf", "png", "svg"))
    assert [p.rsplit(".", 1)[1] for p in paths] == ["pdf", "png", "svg"]


def test_grayscale_preview_is_gray():
    import matplotlib.image as mimage

    data = FigureEditor(lines_fig()).preview(grayscale=True)
    img = mimage.imread(io.BytesIO(data))
    assert np.allclose(img[..., 0], img[..., 1]) and np.allclose(
        img[..., 1], img[..., 2]
    )


def test_check_reports_real_problems_only():
    fig = lines_fig()
    ed = FigureEditor(fig)
    assert ed.check() == []
    ed.set_fontsize("ticks", 4)
    ed.set_series_color(1, "#000000")  # same luminance band as series 0?
    ed.set_series_color(0, "#0a0a0a")
    ed.set_linestyle(1, "-")
    ed.set_marker(1, "o")
    codes = {f["code"] for f in ed.check()}
    assert "font_too_small" in codes
    assert "grayscale_collision" in codes
    ed2 = FigureEditor(lines_fig())
    ed2.set_background_color("#1f77b4", target="axes")
    assert any(f["code"] == "low_contrast" for f in ed2.check())


def test_repr_is_short():
    ed = FigureEditor(lines_fig())
    ed.set_title("x")
    assert repr(ed) == (
        "FigureEditor(1 axes, 1 edit(s), protect_data=True; see .summary(), "
        ".generate_code())"
    )


def test_summary_and_legacy_fields():
    ed = FigureEditor(lines_fig())
    ed.set_title("New")
    rec = ed.to_edits_list()[0]
    assert rec["op"] == "set_title" and rec["target"] == "ax.title"
    assert rec["property"] == "text" and rec["new"] == "New"
    assert "set_title" in ed.summary()


# ----------------------------------------------------------------------
# User interfaces
# ----------------------------------------------------------------------


@pytest.fixture
def panel_factory():
    pytest.importorskip("ipywidgets")
    from statspai.plots._jupyter_editor import create_jupyter_panel

    def make(fig):
        ed = FigureEditor(fig, coalesce=True)
        return ed, create_jupyter_panel(ed, show=False)

    return make


def test_panel_controls_edit_and_resync(panel_factory):
    fig = lines_fig()
    original = pixels(fig)
    ed, p = panel_factory(fig)
    c = p.controls
    assert c["title"].value == "Outcome"
    assert c["tick_size"].value == pytest.approx(
        fig.axes[0].get_xticklabels()[0].get_fontsize()
    )
    c["title"].value = "From the panel"
    c["title_size"].value = 17.0
    c["grid"].value = True
    c["legend_labels"].value = "A\nB"
    assert fig.axes[0].get_title() == "From the panel"
    assert fig.axes[0].title.get_fontsize() == 17.0
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["A", "B"]
    import html

    assert "ax.title.set_text('From the panel')" in html.unescape(c["code"].value)
    # undo / reset bring the widgets back with the figure
    c["undo"].click()
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [
        "Treated",
        "Control",
    ]
    assert c["legend_labels"].value == "Treated\nControl"
    c["reset"].click()
    assert c["title"].value == "Outcome" and c["grid"].value is False
    assert c["title_size"].value == pytest.approx(fig.axes[0].title.get_fontsize())
    assert same(pixels(fig), original)


def test_panel_shows_protection_warnings(panel_factory):
    fig = errorbar_fig()
    ed, p = panel_factory(fig)
    p.controls["ylim"].value = (0.2, 0.6)
    assert "would hide data" in p.status.value
    lo, hi = fig.axes[0].get_ylim()
    assert lo <= -0.3 and hi >= 1.3
    # the slider shows the limits actually set
    assert p.controls["ylim"].value[0] == pytest.approx(lo)


def test_panel_refused_edit_is_reported(panel_factory):
    fig = errorbar_fig()
    ed, p = panel_factory(fig)
    p.controls["line0_marker"].value = "None"
    assert "Not applied" in p.status.value and "invisible" in p.status.value
    assert p.controls["line0_marker"].value == "o"  # widget snapped back
    assert ed.edits == []


def test_panel_axis_switch_and_series_colours(panel_factory):
    fig = two_axes_fig()
    ed, p = panel_factory(fig)
    c = p.controls
    c["axis"].value = 1
    assert c["title"].value == "Right"
    c["title"].value = "Changed right"
    assert fig.axes[1].get_title() == "Changed right" and fig.axes[0].get_title() == ""
    c["group0"].value = "#0072b2"  # the scatter series on axes 1
    assert _hex(fig.axes[1].collections[0].get_facecolor()[0]) == "#0072b2"


def _hex(c):
    import matplotlib.colors as mcolors

    return mcolors.to_hex(c, keep_alpha=False)


def test_panel_theme_and_style_json(panel_factory):
    fig = lines_fig()
    ed, p = panel_factory(fig)
    p.controls["theme"].value = "ggplot"
    assert p.controls["theme"].value == ""  # an action, not a state
    p.controls["copy_style"].click()
    style = p.controls["style_json"].value
    other = lines_fig()
    ed2, p2 = panel_factory(other)
    p2.controls["style_json"].value = style
    p2.controls["apply_style"].click()
    assert same(pixels(other), pixels(fig))


def test_interactive_jupyter_mode_builds_panel(monkeypatch):
    pytest.importorskip("ipywidgets")
    import IPython.display

    shown = []
    monkeypatch.setattr(IPython.display, "display", lambda obj: shown.append(obj))
    ed = sp.interactive(lines_fig(), ui="jupyter")
    assert len(shown) == 1 and ed.coalesce is True


def test_script_editor_uses_its_own_window():
    from types import SimpleNamespace

    from statspai.plots._script_editor import create_script_editor

    def PickEvent(name, canvas, mouse, artist):  # the callback only reads .artist
        return SimpleNamespace(name=name, canvas=canvas, artist=artist)

    fig = lines_fig()
    before = pixels(fig)
    n_axes = len(fig.axes)
    ed = FigureEditor(fig)
    controls = create_script_editor(ed, show=False)
    assert controls is not fig and len(fig.axes) == n_axes
    assert same(pixels(fig), before)  # no buttons, no new margins
    widgets = controls._statspai_widgets
    title = fig.axes[0].title
    widgets["on_pick"](PickEvent("pick_event", fig.canvas, None, title))
    assert widgets["textbox"].text == "Outcome"
    assert ed.edits == []  # selecting is not editing
    widgets["textbox"].set_val("Typed")  # fires submit, as pressing Enter does
    assert title.get_text() == "Typed" and len(ed.edits) == 1
    undo = next(b for b in widgets["buttons"] if b.label.get_text() == "Undo")
    undo._observers.process("clicked", None)
    assert title.get_text() == "Outcome"
    # numeric result labels cannot be selected for rewording
    fig2, ax2 = plt.subplots()
    ax2.text(0.5, 0.5, "0.12***")
    ed2 = FigureEditor(fig2)
    c2 = create_script_editor(ed2, show=False)
    c2._statspai_widgets["on_pick"](
        PickEvent("pick_event", fig2.canvas, None, ax2.texts[0])
    )
    assert "locked" in c2._statspai_widgets["info"].get_text()


# ----------------------------------------------------------------------
# Legends follow restyled series
# ----------------------------------------------------------------------


def _series_fig(color, errorbar=False):
    fig, ax = plt.subplots(figsize=(4, 3))
    if errorbar:
        ax.errorbar(
            [0, 1, 2], [1, 2, 3], yerr=0.3, fmt="o", color=color, label="ATT", capsize=3
        )
    else:
        ax.plot([0, 1, 2], [1, 2, 3], "o-", color=color, label="estimate")
    ax.fill_between(
        [0, 1, 2], [0, 1, 2], [2, 3, 4], color=color, alpha=0.2, label="95% CI"
    )
    ax.legend()
    return fig


@pytest.mark.parametrize("errorbar", [False, True])
def test_legend_entries_follow_a_recoloured_series(errorbar):
    """Recolouring blue -> red must give the figure that was drawn red,
    legend included (matplotlib's legend keeps copies of the old look)."""
    reference = _series_fig("red", errorbar)
    fig = _series_fig("blue", errorbar)
    ed = FigureEditor(fig)
    ed.set_series_color(0, "red")
    assert same(pixels(fig), pixels(reference))
    code = ed.generate_code(include_comment=False)
    assert "import statspai as sp" in code and "sp.refresh_legend(ax, [0, 1])" in code
    assert same(
        pixels(run_code(lambda: _series_fig("blue", errorbar), code)), pixels(reference)
    )
    ed.undo()
    assert same(pixels(fig), pixels(_series_fig("blue", errorbar)))


def test_legend_follows_line_style_edits():
    def drawn(**kw):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1, 2], [1, 2, 3], label="a", **kw)
        ax.legend()
        return fig

    fig = drawn()
    ed = FigureEditor(fig)
    ed.set_linestyle(0, "--")
    ed.set_marker(0, "s")
    ed.set_linewidth(0, 3.0)
    assert same(pixels(fig), pixels(drawn(linestyle="--", marker="s", linewidth=3.0)))


def test_refresh_legend_public_function():
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1], color="blue", label="estimate")
    ax.legend()
    line.set_color("red")
    sp.refresh_legend(ax)
    handles = ax.get_legend().legend_handles
    assert handles[0].get_color() == "red"
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="sources"):
        sp.refresh_legend(ax, [0, 1])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not an index"):
        sp.refresh_legend(ax, [5])
    fig2, ax2 = plt.subplots()
    sp.refresh_legend(ax2)  # no legend: nothing to do


def test_explicit_marker_colours_follow_repeated_recolours():
    def drawn(color):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(
            [0, 1, 2],
            [1, 2, 3],
            "o-",
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            label="s",
        )
        ax.legend()
        return fig

    fig = drawn("blue")
    ed = FigureEditor(fig)
    ed.set_series_color(0, "green")
    ed.set_series_color(0, "red")
    assert same(pixels(fig), pixels(drawn("red")))
    assert same(
        pixels(run_code(lambda: drawn("blue"), ed.generate_code())), pixels(fig)
    )


# ----------------------------------------------------------------------
# Found by running every sp.*plot example through the editor
# ----------------------------------------------------------------------


def test_categorical_axis_lines_are_classified():
    """sp.synth_sensitivity_plot draws lines on a categorical x axis; the
    flat-line test must use converted coordinates, not the strings."""

    def make():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(["1x", "2x", "4x"], [0.5, 0.4, 0.45], "o-", label="RMSPE ratio")
        ax.plot(["1x", "4x"], [0.3, 0.3], "--", color="gray")
        return fig

    fig = make()
    ed = FigureEditor(fig)
    roles = [ed.artist_roles[id(ln)] for ln in fig.axes[0].get_lines()]
    assert roles == [ArtistRole.DATA, ArtistRole.REFERENCE]
    ed.apply_palette("okabe_ito")
    ed.set_title("x")
    assert same(pixels(run_code(make, ed.generate_code())), pixels(fig))


def test_recolour_leaves_transparent_edges_invisible():
    """Bars with alpha and no edge (a histogram): recolouring twice, the
    second time to black, must not turn the invisible edges into black
    frames (matplotlib applies the bar's alpha over the colour's)."""

    def make():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(
            np.random.default_rng(0).normal(size=200),
            bins=12,
            alpha=0.2,
            color="C0",
            edgecolor="none",
        )
        return fig

    fig = make()
    original = pixels(fig)
    ed = FigureEditor(fig)
    ed.apply_palette("tab10")
    ed.apply_palette("okabe_ito")  # first okabe_ito colour is black
    edges = {tuple(p.get_edgecolor()) for p in fig.axes[0].patches}
    assert all(e[3] == 0 for e in edges)
    assert same(pixels(run_code(make, ed.generate_code())), pixels(fig))
    ed.reset()
    assert same(pixels(fig), original)


def test_alpha_coded_spans_are_locked():
    """sp.synthdid_plot shades each period with alpha = its time weight;
    the transparency is data. Plain shading stays a reference element."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2, 3], [1, 2, 3, 4])
    for i, w in enumerate([0.02, 0.07, 0.0]):
        ax.axvspan(i - 0.4, i + 0.4, color="blue", alpha=w)
    ax.axvspan(2.5, 3.5, color="gray", alpha=0.3)
    ed = FigureEditor(fig)
    roles = [ed.artist_roles[id(p)] for p in ax.patches]
    assert roles == [ArtistRole.DATA] * 3 + [ArtistRole.REFERENCE]
    with pytest.raises(ValueError, match="encodes a value"):
        ed.set_alpha("patch0", 0.5)
    ed.set_alpha("patch3", 0.5)  # plain shading stays editable
    ed.apply_palette("okabe_ito")  # recolouring keeps each span's alpha
    assert [round(p.get_facecolor()[3], 6) for p in ax.patches[:3]] == [0.02, 0.07, 0.0]
