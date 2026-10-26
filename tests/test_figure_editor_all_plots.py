"""The figure editor's guarantees on every StatsPAI plotting function.

Each ``sp.*plot*`` function's docstring example is run, and on every figure
it draws a fixed set of edits is made (title, palette, theme, tighter y
limits, tick size). The generated code, run on a freshly drawn copy, must
give the edited pixels, and ``reset()`` must give back the original pixels.

The first run of this sweep (2026-10) found three defects that the unit
tests in ``test_figure_editor.py`` had missed (categorical axes, transparent
histogram edges, alpha-coded shading). New plotting functions are picked up
from the registry automatically. Marked slow: it draws every figure twice.
"""

from __future__ import annotations

import doctest
import inspect
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.plots.interactive import FigureEditor  # noqa: E402

pytestmark = pytest.mark.slow


def _plot_functions():
    return sorted(
        n
        for n in sp.list_functions(kind="function")
        if "plot" in n.lower() or n in ("ggdid", "binscatter")
    )


def _run_example(name):
    plt.close("all")
    doc = inspect.getdoc(getattr(sp, name)) or ""
    examples = [
        ex
        for ex in doctest.DocTestParser().get_examples(doc)
        if not ex.options.get(doctest.SKIP)
    ]
    if not examples:
        pytest.skip(f"{name}: example is marked +SKIP")
    ns = {"sp": sp, "np": np, "plt": plt}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ex in examples:
            try:
                exec(compile(ex.source, f"<{name} example>", "exec"), ns)  # noqa: S102
            except ImportError as exc:  # optional extra (torch, networkx, ...)
                pytest.skip(f"{name}: {exc}")
    return [plt.figure(n) for n in plt.get_fignums()]


def _pixels(fig):
    # Drawing the fresh copy runs plt.close("all"), which also closes ``fig``;
    # from matplotlib 3.11 a closed figure is left with a bare
    # FigureCanvasBase, so give it an Agg canvas again (same renderer).
    canvas = fig.canvas
    if not hasattr(canvas, "buffer_rgba"):
        canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba()).copy()


def _edit(ed, fig):
    ed.set_title("Edited title")
    ed.apply_palette("okabe_ito")
    ed.apply_theme("aea")
    for i, ax in enumerate(fig.get_axes()):
        lo, hi = ax.get_ylim()
        if ax.name == "rectilinear" and np.isfinite([lo, hi]).all() and hi != lo:
            mid, half = (lo + hi) / 2, (hi - lo) * 0.3
            ed.set_ylim(mid - half, mid + half, ax_index=i)
    ed.set_fontsize("ticks", 7)


@pytest.mark.parametrize("name", _plot_functions())
def test_editor_guarantees_hold_on_plot(name):
    figures = _run_example(name)
    if not figures:
        pytest.skip(f"{name}: example draws no matplotlib figure")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, fig in enumerate(figures):
            original = _pixels(fig)
            ed = FigureEditor(fig)
            _edit(ed, fig)
            edited = _pixels(fig)
            fresh = _run_example(name)[k]
            exec(ed.generate_code(), {"fig": fresh})  # noqa: S102
            assert np.array_equal(_pixels(fresh), edited), f"{name} figure {k}: code"
            ed.reset()
            assert np.array_equal(_pixels(fig), original), f"{name} figure {k}: reset"
            plt.close("all")
