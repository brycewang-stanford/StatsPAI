"""Print-size Matplotlib settings shared by every figure in the JSS paper.

Figures were previously drawn at ``figsize=(7.4, 4.2)`` with Matplotlib's
default 10 pt type and then included at ``width=0.64\\textwidth``.  The
scale factor is not free: 0.64 x 6.13 in / 7.4 in = 0.53, so the printed
axis labels came out near 5 pt -- half the body text, and below what JSS
accepts as legible.

The fix is to draw at the size the figure is printed and include it at
``width=\\textwidth``, so a point specified here is a point on the page.
``TEXT_WIDTH_IN`` is the JSS ``\\textwidth`` (442.65 pt); re-measure it
with ``\\typeout{\\the\\textwidth}`` if the class ever changes.

Imported by both ``tests/perf/compare_perf.py`` (Track C) and
``Paper-JSS/replication/scripts/generate_figures.py`` (worked examples)
so the two figure families cannot drift apart typographically.
"""

from __future__ import annotations

#: JSS \textwidth in inches (442.65 pt / 72.27 pt per inch).
TEXT_WIDTH_IN = 6.125

#: Body text is 10 pt; figure type one step down reads as a caption-level
#: element without disappearing.
RC_PARAMS = {
    "font.size": 9.0,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "figure.titlesize": 10.0,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.borderpad": 0.35,
    "legend.handlelength": 1.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # ASCII hyphen-minus in tick labels: the Unicode minus glyph is
    # dropped by some PDF text extractors, which makes negative ticks
    # unreadable in review tooling.
    "axes.unicode_minus": False,
}


def apply(plt) -> None:
    """Install the shared parameters on a live ``pyplot`` module."""
    plt.rcParams.update(RC_PARAMS)
