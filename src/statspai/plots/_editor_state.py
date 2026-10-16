"""Snapshots of everything the figure editor can change, for exact undo.

Each edit stores a snapshot of the axes it touches (plus the figure-level
state) taken just before it runs. Undo restores that snapshot, so undo and
reset return the figure to the pixels it had before, whatever the edit did.
A property is written back only when it differs from the snapshot, which
keeps untouched artists exactly as the plotting code left them.

Several matplotlib internals are read here (raw colours that may still be
``'auto'``, tick keyword dictionaries, tick padding). They have been stable
since matplotlib 3.5; every access falls back to the public getter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _same(a: Any, b: Any) -> bool:
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return bool(
                np.shape(a) == np.shape(b)
                and np.array_equal(np.asarray(a), np.asarray(b))
            )
        return bool(a == b)
    except (TypeError, ValueError):  # values that cannot be compared differ
        return False


# ----------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------


def text_state(t: Any) -> Dict[str, Any]:
    return {
        "text": t.get_text(),
        "font": t.get_fontproperties().copy(),
        "color": t.get_color(),
        "rotation": t.get_rotation(),
        "visible": t.get_visible(),
        "alpha": t.get_alpha(),
    }


def restore_text(t: Any, s: Dict[str, Any]) -> None:
    if t.get_text() != s["text"]:
        t.set_text(s["text"])
    if t.get_fontproperties() != s["font"]:
        t.set_fontproperties(s["font"].copy())
    if not _same(t.get_color(), s["color"]):
        t.set_color(s["color"])
    if t.get_rotation() != s["rotation"]:
        t.set_rotation(s["rotation"])
    if t.get_visible() != s["visible"]:
        t.set_visible(s["visible"])
    if t.get_alpha() != s["alpha"]:
        t.set_alpha(s["alpha"])


# ----------------------------------------------------------------------
# Lines
# ----------------------------------------------------------------------


def _raw(obj: Any, private: str, public: str) -> Any:
    value = getattr(obj, private, None)
    return value if value is not None else getattr(obj, public)()


def _line_dash(line: Any) -> Any:
    ls = line.get_linestyle()
    pattern = getattr(line, "_unscaled_dash_pattern", None)
    if ls == "--" and pattern is not None and pattern[1] is not None:
        return (pattern[0], tuple(pattern[1]))
    return ls


def line_state(line: Any) -> Dict[str, Any]:
    return {
        "color": _raw(line, "_color", "get_color"),
        "mfc": _raw(line, "_markerfacecolor", "get_markerfacecolor"),
        "mec": _raw(line, "_markeredgecolor", "get_markeredgecolor"),
        "lw": line.get_linewidth(),
        "ls": _line_dash(line),
        "alpha": line.get_alpha(),
        "marker": line.get_marker(),
        "ms": line.get_markersize(),
        "mew": line.get_markeredgewidth(),
        "visible": line.get_visible(),
    }


def restore_line(line: Any, s: Dict[str, Any]) -> None:
    now = line_state(line)
    if not _same(now["color"], s["color"]):
        line.set_color(s["color"])
    if not _same(now["mfc"], s["mfc"]):
        line.set_markerfacecolor(s["mfc"])
    if not _same(now["mec"], s["mec"]):
        line.set_markeredgecolor(s["mec"])
    if now["lw"] != s["lw"]:
        line.set_linewidth(s["lw"])
    if now["ls"] != s["ls"]:
        line.set_linestyle(s["ls"])
    if now["alpha"] != s["alpha"]:
        line.set_alpha(s["alpha"])
    if now["marker"] != s["marker"]:
        line.set_marker(s["marker"])
    if now["ms"] != s["ms"]:
        line.set_markersize(s["ms"])
    if now["mew"] != s["mew"]:
        line.set_markeredgewidth(s["mew"])
    if now["visible"] != s["visible"]:
        line.set_visible(s["visible"])


# ----------------------------------------------------------------------
# Collections and patches
# ----------------------------------------------------------------------


def collection_state(coll: Any) -> Dict[str, Any]:
    sizes = coll.get_sizes() if hasattr(coll, "get_sizes") else None
    return {
        "fc": np.array(coll.get_facecolor(), copy=True),
        "ec": np.array(coll.get_edgecolor(), copy=True),
        "fc_raw": copy.copy(getattr(coll, "_original_facecolor", None)),
        "ec_raw": copy.copy(getattr(coll, "_original_edgecolor", None)),
        "alpha": coll.get_alpha(),
        "lw": np.array(coll.get_linewidth(), copy=True),
        "sizes": None if sizes is None else np.array(sizes, copy=True),
        "visible": coll.get_visible(),
    }


def restore_collection(coll: Any, s: Dict[str, Any]) -> None:
    if coll.get_alpha() != s["alpha"]:
        coll.set_alpha(s["alpha"])
    if not _same(np.asarray(coll.get_facecolor()), s["fc"]):
        coll.set_facecolor(s["fc_raw"] if s["fc_raw"] is not None else s["fc"])
    if not _same(np.asarray(coll.get_edgecolor()), s["ec"]):
        coll.set_edgecolor(s["ec_raw"] if s["ec_raw"] is not None else s["ec"])
    if not _same(np.asarray(coll.get_linewidth()), s["lw"]):
        coll.set_linewidth(s["lw"])
    if s["sizes"] is not None and not _same(np.asarray(coll.get_sizes()), s["sizes"]):
        coll.set_sizes(s["sizes"])
    if coll.get_visible() != s["visible"]:
        coll.set_visible(s["visible"])


def patch_state(p: Any) -> Dict[str, Any]:
    return {
        "fc": p.get_facecolor(),
        "ec": p.get_edgecolor(),
        "fc_raw": getattr(p, "_original_facecolor", None),
        "ec_raw": getattr(p, "_original_edgecolor", None),
        "alpha": p.get_alpha(),
        "lw": p.get_linewidth(),
        "visible": p.get_visible(),
    }


def restore_patch(p: Any, s: Dict[str, Any]) -> None:
    if p.get_alpha() != s["alpha"]:
        p.set_alpha(s["alpha"])
    if not _same(p.get_facecolor(), s["fc"]):
        p.set_facecolor(s["fc_raw"] if s["fc_raw"] is not None else s["fc"])
    if not _same(p.get_edgecolor(), s["ec"]):
        p.set_edgecolor(s["ec_raw"] if s["ec_raw"] is not None else s["ec"])
    if p.get_linewidth() != s["lw"]:
        p.set_linewidth(s["lw"])
    if p.get_visible() != s["visible"]:
        p.set_visible(s["visible"])


# ----------------------------------------------------------------------
# Axis ticks (label size, rotation, direction, grid lines)
# ----------------------------------------------------------------------


def _tick_state(tick: Any) -> Dict[str, Any]:
    return {
        "geom": (
            getattr(tick, "_size", None),
            getattr(tick, "_width", None),
            getattr(tick, "_base_pad", None),
            getattr(tick, "_tickdir", None),
        ),
        "labelrotation": copy.copy(getattr(tick, "_labelrotation", None)),
        "label1": text_state(tick.label1),
        "label2": text_state(tick.label2),
        "grid": line_state(tick.gridline),
        "tick1": line_state(tick.tick1line),
        "tick2": line_state(tick.tick2line),
    }


def _restore_tick(tick: Any, s: Dict[str, Any], *, keep_text: bool) -> None:
    size, width, pad, tickdir = s["geom"]
    now = (
        getattr(tick, "_size", None),
        getattr(tick, "_width", None),
        getattr(tick, "_base_pad", None),
        getattr(tick, "_tickdir", None),
    )
    if now != s["geom"] and None not in s["geom"]:
        tick._apply_params(size=size, width=width, pad=pad, tickdir=tickdir)
    if s["labelrotation"] is not None:
        tick._labelrotation = copy.copy(s["labelrotation"])
    for name in ("label1", "label2"):
        state = dict(s[name])
        if not keep_text:
            # Tick label strings come from the formatter at draw time.
            state["text"] = getattr(tick, name).get_text()
        restore_text(getattr(tick, name), state)
    restore_line(tick.gridline, s["grid"])
    restore_line(tick.tick1line, s["tick1"])
    restore_line(tick.tick2line, s["tick2"])


def axis_state(axis: Any) -> Dict[str, Any]:
    major = list(axis.get_major_ticks())
    minor = list(axis.minorTicks)
    return {
        "kw": (
            copy.deepcopy(getattr(axis, "_major_tick_kw", {})),
            copy.deepcopy(getattr(axis, "_minor_tick_kw", {})),
        ),
        "major": [(t, _tick_state(t)) for t in major],
        "minor": [(t, _tick_state(t)) for t in minor],
    }


def restore_axis(axis: Any, s: Dict[str, Any]) -> None:
    for which, current in (("major", axis.majorTicks), ("minor", axis.minorTicks)):
        captured = s[which]
        known = {id(t): st for t, st in captured}
        template = captured[0][1] if captured else None
        for tick in list(current):
            st = known.get(id(tick))
            if st is not None:
                _restore_tick(tick, st, keep_text=True)
            elif template is not None:
                _restore_tick(tick, template, keep_text=False)
    major_kw, minor_kw = s["kw"]
    if hasattr(axis, "_major_tick_kw"):
        axis._major_tick_kw = copy.deepcopy(major_kw)
        axis._minor_tick_kw = copy.deepcopy(minor_kw)


# ----------------------------------------------------------------------
# Legend
# ----------------------------------------------------------------------


def legend_state(legend: Any) -> Optional[Dict[str, Any]]:
    if legend is None:
        return None
    return {
        "obj": legend,
        "visible": legend.get_visible(),
        "loc": getattr(legend, "_loc", None),
        "frameon": legend.get_frame_on(),
        "texts": [text_state(t) for t in legend.get_texts()],
        "title": text_state(legend.get_title()),
        "frame": patch_state(legend.get_frame()),
        "entries": _legend_entries(legend),
    }


def _legend_entries(legend: Any) -> Dict[str, Any]:
    from ._legend_sync import _handles_attr, handle_boxes

    attr = _handles_attr(legend)
    return {
        "attr": attr,
        "handles": list(getattr(legend, attr)),
        "boxes": [(box, list(box._children)) for box in handle_boxes(legend)],
    }


def set_legend_loc(legend: Any, loc: Any) -> None:
    """``Legend.set_loc`` exists from matplotlib 3.8; older versions keep
    the code in ``_loc``."""
    if hasattr(legend, "set_loc"):
        legend.set_loc(loc)
    else:  # pragma: no cover - matplotlib < 3.8
        from matplotlib.legend import Legend

        codes = getattr(Legend, "codes", {})
        legend._loc = codes.get(loc, loc) if isinstance(loc, str) else loc
    legend.stale = True


def restore_legend(ax: Any, s: Optional[Dict[str, Any]]) -> None:
    current = ax.get_legend()
    if s is None:
        if current is not None:
            current.remove()
        return
    legend = s["obj"]
    if current is not legend:
        if current is not None:
            current.remove()
        ax.legend_ = legend
    if legend.get_visible() != s["visible"]:
        legend.set_visible(s["visible"])
    if getattr(legend, "_loc", None) != s["loc"] and s["loc"] is not None:
        legend._loc = s["loc"]
        legend.stale = True
    if legend.get_frame_on() != s["frameon"]:
        legend.set_frame_on(s["frameon"])
    for t, st in zip(legend.get_texts(), s["texts"]):
        restore_text(t, st)
    restore_text(legend.get_title(), s["title"])
    restore_patch(legend.get_frame(), s["frame"])
    entries = s["entries"]
    if list(getattr(legend, entries["attr"])) != entries["handles"]:
        setattr(legend, entries["attr"], list(entries["handles"]))
        for box, children in entries["boxes"]:
            box._children = list(children)
        legend.stale = True


# ----------------------------------------------------------------------
# Axes and figure
# ----------------------------------------------------------------------


def axes_state(ax: Any) -> Dict[str, Any]:
    return {
        "patch": patch_state(ax.patch),
        "xlim": tuple(ax.get_xlim()),
        "ylim": tuple(ax.get_ylim()),
        "autoscale": (ax.get_autoscalex_on(), ax.get_autoscaley_on()),
        "title": text_state(ax.title),
        "xlabel": text_state(ax.xaxis.label),
        "ylabel": text_state(ax.yaxis.label),
        "texts": [(t, text_state(t)) for t in ax.texts],
        "lines": [(ln, line_state(ln)) for ln in ax.get_lines()],
        "colls": [(c, collection_state(c)) for c in ax.collections],
        "patches": [(p, patch_state(p)) for p in ax.patches],
        "spines": {k: patch_state(v) for k, v in ax.spines.items()},
        "xaxis": axis_state(ax.xaxis),
        "yaxis": axis_state(ax.yaxis),
        "legend": legend_state(ax.get_legend()),
    }


def restore_axes(ax: Any, s: Dict[str, Any]) -> None:
    restore_patch(ax.patch, s["patch"])
    restore_text(ax.title, s["title"])
    restore_text(ax.xaxis.label, s["xlabel"])
    restore_text(ax.yaxis.label, s["ylabel"])

    kept = {id(t) for t, _ in s["texts"]}
    for t in list(ax.texts):
        if id(t) not in kept:
            t.remove()  # added by an edit
    for t, st in s["texts"]:
        restore_text(t, st)
    for ln, st in s["lines"]:
        restore_line(ln, st)
    for c, st in s["colls"]:
        restore_collection(c, st)
    for p, st in s["patches"]:
        restore_patch(p, st)
    for k, st in s["spines"].items():
        restore_patch(ax.spines[k], st)
    restore_axis(ax.xaxis, s["xaxis"])
    restore_axis(ax.yaxis, s["yaxis"])
    restore_legend(ax, s["legend"])

    if tuple(ax.get_xlim()) != s["xlim"]:
        ax.set_xlim(s["xlim"])
    if tuple(ax.get_ylim()) != s["ylim"]:
        ax.set_ylim(s["ylim"])
    ax.set_autoscalex_on(s["autoscale"][0])
    ax.set_autoscaley_on(s["autoscale"][1])


_SUBPLOT_PARS = ("left", "right", "bottom", "top", "wspace", "hspace")


def figure_state(fig: Any) -> Dict[str, Any]:
    suptitle = getattr(fig, "_suptitle", None)
    return {
        "patch": patch_state(fig.patch),
        "size": tuple(float(v) for v in fig.get_size_inches()),
        "dpi": fig.dpi,
        "subplotpars": {k: getattr(fig.subplotpars, k) for k in _SUBPLOT_PARS},
        "positions": [
            (ax, ax.get_position(original=True).frozen()) for ax in fig.get_axes()
        ],
        "suptitle": None if suptitle is None else (suptitle, text_state(suptitle)),
    }


def restore_figure(fig: Any, s: Dict[str, Any]) -> None:
    restore_patch(fig.patch, s["patch"])
    if tuple(float(v) for v in fig.get_size_inches()) != s["size"]:
        fig.set_size_inches(*s["size"])
    if fig.dpi != s["dpi"]:
        fig.set_dpi(s["dpi"])
    pars = {k: getattr(fig.subplotpars, k) for k in _SUBPLOT_PARS}
    if pars != s["subplotpars"]:
        fig.subplots_adjust(**s["subplotpars"])
    for ax, pos in s["positions"]:
        if ax.get_position(original=True).bounds != pos.bounds:
            ax.set_position(pos, which="original")
    current = getattr(fig, "_suptitle", None)
    if s["suptitle"] is None:
        if current is not None:
            current.remove()
            fig._suptitle = None
    else:
        obj, st = s["suptitle"]
        restore_text(obj, st)


@dataclass
class Snapshot:
    """State of the figure plus selected axes, taken before one edit."""

    figure: Dict[str, Any]
    axes: List[Tuple[int, Dict[str, Any]]] = field(default_factory=list)


def capture(fig: Any, axes: Iterable[int]) -> Snapshot:
    all_axes = fig.get_axes()
    picked = sorted({i for i in axes if 0 <= i < len(all_axes)})
    return Snapshot(
        figure=figure_state(fig),
        axes=[(i, axes_state(all_axes[i])) for i in picked],
    )


def restore(fig: Any, snap: Snapshot) -> None:
    all_axes = fig.get_axes()
    for i, st in snap.axes:
        if i < len(all_axes):
            restore_axes(all_axes[i], st)
    restore_figure(fig, snap.figure)
