"""Pre-submission checks behind ``FigureEditor.check()``.

Each finding is a dict ``{"code", "severity", "message", "ax_index"}``.
The checks look at the figure as it will be printed: font sizes are in
points at the current figure size, colours as drawn.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from .interactive import FigureEditor


def _luminance(color: str) -> float:
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _visible_texts(ax: Any) -> List[Any]:
    texts = [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
    texts += ax.get_xticklabels() + ax.get_yticklabels()
    legend = ax.get_legend()
    if legend is not None and legend.get_visible():
        texts += legend.get_texts()
    return [t for t in texts if t.get_visible() and (t.get_text() or "").strip()]


def _series_style(editor: "FigureEditor", ax: Any, group: Dict[str, Any]) -> tuple:
    """How a series can be told apart without colour: line style + marker."""
    styles = set()
    for kind, j in group["members"]:
        if kind == "line":
            line = ax.get_lines()[j]
            styles.add((line.get_linestyle(), line.get_marker()))
    return tuple(sorted(styles, key=str))


def run_checks(
    editor: "FigureEditor", min_fontsize: float = 6.0
) -> List[Dict[str, Any]]:
    from ._editor_roles import single_color

    findings: List[Dict[str, Any]] = []
    for i, ax in enumerate(editor.fig.get_axes()):
        small = sorted(
            {
                round(float(t.get_fontsize()), 1)
                for t in _visible_texts(ax)
                if float(t.get_fontsize()) < min_fontsize
            }
        )
        if small:
            findings.append(
                {
                    "code": "font_too_small",
                    "severity": "warning",
                    "message": (
                        f"Text at {', '.join(str(s) for s in small)} pt is below "
                        f"{min_fontsize} pt at the current figure size; journals "
                        "usually ask for 6-8 pt at print size."
                    ),
                    "ax_index": i,
                }
            )

        groups = []
        for g in editor.color_groups(i):
            members = [
                {"line": ax.get_lines(), "coll": ax.collections, "patch": ax.patches}[
                    k
                ][j]
                for k, j in g["members"]
            ]
            current = {
                c for c in (single_color(m) for m in members if m.get_visible()) if c
            }
            if len(current) == 1:
                groups.append((current.pop(), g))
        for (c1, g1), (c2, g2) in combinations(groups, 2):
            if c1 == c2:
                continue
            close = abs(_luminance(c1) - _luminance(c2)) < 0.1
            if close and _series_style(editor, ax, g1) == _series_style(editor, ax, g2):
                findings.append(
                    {
                        "code": "grayscale_collision",
                        "severity": "warning",
                        "message": (
                            f"Series drawn in {c1} and {c2} look the same in "
                            "black and white (and share line style and marker); "
                            "change one colour, line style or marker."
                        ),
                        "ax_index": i,
                    }
                )

        bg = _luminance(_hex(ax.get_facecolor()))
        for color, g in groups:
            if abs(_luminance(color) - bg) < 0.08:
                findings.append(
                    {
                        "code": "low_contrast",
                        "severity": "warning",
                        "message": (
                            f"A series drawn in {color} is barely distinguishable "
                            f"from the axes background {_hex(ax.get_facecolor())}."
                        ),
                        "ax_index": i,
                    }
                )
    return findings


def _hex(c: Any) -> str:
    import matplotlib.colors as mcolors

    return str(mcolors.to_hex(c, keep_alpha=False))
