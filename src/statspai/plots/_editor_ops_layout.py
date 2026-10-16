"""Figure editor operations: spines, grid, ticks, size, legend, annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..exceptions import MethodIncompatibility
from ._editor_ops import (
    AX,
    LEGEND_LOCS,
    LINE_STYLES,
    SPINES,
    AxVar,
    Op,
    Param,
    _axname,
    _color_val,
    _has_set_loc,
    _hex_or_raw,
    _legend_or_raise,
    _locked,
    _pick,
    kwargs_src,
    lit,
    register,
)
from ._editor_roles import ArtistRole

if TYPE_CHECKING:
    from .interactive import FigureEditor


# ----------------------------------------------------------------------
# Layout: spines, grid, ticks, size, dpi
# ----------------------------------------------------------------------


def _apply_spine(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    e._ax(p["ax_index"]).spines[p["spine"]].set_visible(p["visible"])
    return p


register(
    Op(
        "set_spine_visible",
        "Show or hide one axes spine.",
        (
            Param("spine", "choice", required=True, choices=SPINES),
            Param("visible", "bool", required=True),
            AX,
        ),
        _apply_spine,
        lambda p, axv: [
            f"{axv(p['ax_index'])}.spines[{p['spine']!r}].set_visible({p['visible']})"
        ],
        slot=lambda p: ("spine", p["ax_index"], p["spine"]),
        legacy=lambda p: (_axname(p) + f".spines.{p['spine']}", "visible"),
    )
)


def _grid_style_now(ax: Any) -> Dict[str, Any]:
    lines = ax.xaxis.get_gridlines()
    if not lines:
        lines = ax.yaxis.get_gridlines()
    if not lines:
        return {}
    g = lines[0]
    return {
        "color": _hex_or_raw(g.get_color()),
        "alpha": g.get_alpha(),
        "linestyle": g.get_linestyle(),
        "linewidth": float(g.get_linewidth()),
    }


def _apply_grid_common(
    e: "FigureEditor", p: Dict[str, Any], visible: bool, style: Dict[str, Any]
) -> Dict[str, Any]:
    ax = e._ax(p["ax_index"])
    if not visible:
        ax.grid(False)
        return {"visible": False, "style": {}, "ax_index": p["ax_index"]}
    full = _grid_style_now(ax)
    full.update({k: v for k, v in style.items() if v is not None})
    ax.grid(True, **{k: _color_val(v) if k == "color" else v for k, v in full.items()})
    return {"visible": True, "style": full, "ax_index": p["ax_index"]}


def _code_grid(p: Dict[str, Any], axv: AxVar) -> List[str]:
    a = axv(p["ax_index"])
    if not p["visible"]:
        return [f"{a}.grid(False)"]
    style = dict(p["style"])
    if "color" in style:
        style["color"] = _color_val(style["color"])
    extra = (", " + kwargs_src(style)) if style else ""
    return [f"{a}.grid(True{extra})"]


register(
    Op(
        "set_grid",
        "Show or hide the major grid (both axes).",
        (
            Param("visible", "bool", required=True),
            AX,
            Param("kwargs", "dict", default={}),
        ),
        lambda e, p: _apply_grid_common(e, p, p["visible"], p["kwargs"]),
        _code_grid,
        slot=lambda p: ("grid", p["ax_index"]),
        legacy=lambda p: (_axname(p) + ".grid", "visible"),
    )
)

register(
    Op(
        "set_grid_style",
        "Turn the grid on with a colour, alpha and/or line style.",
        (
            Param("color", "color", default=None),
            Param("alpha", "number", default=None),
            Param("linestyle", "choice", default=None, choices=LINE_STYLES[:-1]),
            AX,
        ),
        lambda e, p: _apply_grid_common(
            e,
            p,
            True,
            {"color": p["color"], "alpha": p["alpha"], "linestyle": p["linestyle"]},
        ),
        _code_grid,
        slot=lambda p: ("grid", p["ax_index"]),
        legacy=lambda p: (_axname(p) + ".grid", "style"),
    )
)


def _apply_rotation(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    e._ax(p["ax_index"]).tick_params(
        axis=p["axis"], which="both", labelrotation=p["angle"]
    )
    return p


register(
    Op(
        "set_tick_rotation",
        "Rotate the tick labels of one axis (degrees).",
        (
            Param("axis", "choice", required=True, choices=("x", "y")),
            Param("angle", "number", required=True),
            AX,
        ),
        _apply_rotation,
        lambda p, axv: [
            f"{axv(p['ax_index'])}.tick_params(axis={p['axis']!r}, which='both', "
            f"labelrotation={lit(p['angle'])})"
        ],
        slot=lambda p: ("tick_rotation", p["ax_index"], p["axis"]),
        legacy=lambda p: (_axname(p) + f".{p['axis']}ticks", "rotation"),
    )
)


def _apply_figsize(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    if p["width"] <= 0 or p["height"] <= 0:
        raise MethodIncompatibility("set_figsize: width and height must be positive")
    e.fig.set_size_inches(p["width"], p["height"])
    return p


register(
    Op(
        "set_figsize",
        "Set the figure size in inches (the printed size in a paper).",
        (
            Param("width", "number", required=True),
            Param("height", "number", required=True),
        ),
        _apply_figsize,
        lambda p, axv: [f"fig.set_size_inches({lit(p['width'])}, {lit(p['height'])})"],
        scope="figure",
        slot=lambda p: ("figsize",),
        legacy=lambda p: ("fig", "figsize"),
    )
)


def _apply_dpi(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    if p["dpi"] <= 0:
        raise MethodIncompatibility("set_dpi: dpi must be positive")
    e.fig.set_dpi(p["dpi"])
    return p


register(
    Op(
        "set_dpi",
        "Set the figure DPI.",
        (Param("dpi", "int", required=True),),
        _apply_dpi,
        lambda p, axv: [f"fig.set_dpi({p['dpi']})"],
        scope="figure",
        slot=lambda p: ("dpi",),
        legacy=lambda p: ("fig", "dpi"),
    )
)


def _apply_tight(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    e.fig.tight_layout()
    return p


register(
    Op(
        "tight_layout",
        "Fit subplot margins to the labels (fig.tight_layout()).",
        (),
        _apply_tight,
        lambda p, axv: ["fig.tight_layout()"],
        scope="figure",
        slot=lambda p: ("tight_layout",),
        legacy=lambda p: ("fig", "tight_layout"),
    )
)


# ----------------------------------------------------------------------
# Legend
# ----------------------------------------------------------------------


def _loc_name(legend: Any) -> Any:
    from matplotlib.legend import Legend

    code = getattr(legend, "_loc", None)
    inverse = {v: k for k, v in getattr(Legend, "codes", {}).items()}
    return inverse.get(code, code)


def _apply_legend(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    from ._editor_state import set_legend_loc

    ax = e._ax(p["ax_index"])
    legend = _legend_or_raise(ax, "set_legend")
    texts = legend.get_texts()
    full = {
        "loc": _loc_name(legend),
        "fontsize": float(texts[0].get_fontsize()) if texts else None,
        "frameon": bool(legend.get_frame_on()),
    }
    full.update({k: p[k] for k in ("loc", "fontsize", "frameon") if p[k] is not None})
    if full["loc"] is not None:
        set_legend_loc(legend, full["loc"])
    if full["fontsize"] is not None:
        for t in texts:
            t.set_fontsize(full["fontsize"])
    legend.set_frame_on(full["frameon"])
    return {**full, "ax_index": p["ax_index"]}


def _code_legend(p: Dict[str, Any], axv: AxVar) -> List[str]:
    from matplotlib.legend import Legend

    a = axv(p["ax_index"])
    out = [f"_leg = {a}.get_legend()"]
    if p["loc"] is not None:
        if _has_set_loc():
            out.append(f"_leg.set_loc({lit(p['loc'])})")
        else:  # pragma: no cover - matplotlib < 3.8
            code = (
                Legend.codes.get(p["loc"], p["loc"])
                if isinstance(p["loc"], str)
                else p["loc"]
            )
            out.append(f"_leg._loc = {lit(code)}")
    if p["fontsize"] is not None:
        out += [
            "for _t in _leg.get_texts():",
            f"    _t.set_fontsize({lit(p['fontsize'])})",
        ]
    out.append(f"_leg.set_frame_on({p['frameon']})")
    return out


register(
    Op(
        "set_legend",
        "Set legend position, entry font size and frame.",
        (
            AX,
            Param("loc", "choice", default=None, choices=LEGEND_LOCS),
            Param("fontsize", "number", default=None),
            Param("frameon", "bool", default=None),
        ),
        _apply_legend,
        _code_legend,
        slot=lambda p: ("legend", p["ax_index"]),
        legacy=lambda p: (_axname(p) + ".legend", "properties"),
    )
)


def _apply_legend_visible(
    e: "FigureEditor", p: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    ax = e._ax(p["ax_index"])
    legend = ax.get_legend()
    if legend is None:
        if not p["visible"]:
            return None  # nothing to hide
        handles, _ = ax.get_legend_handles_labels()
        if not handles:
            raise MethodIncompatibility(
                "set_legend_visible: no labelled artists to build a legend from"
            )
        legend = ax.legend()
        e.artist_roles[id(legend)] = ArtistRole.LEGEND
        e._legend_sources[p["ax_index"]] = list(range(len(handles)))
        return {**p, "create": True}
    legend.set_visible(p["visible"])
    return {**p, "create": False}


register(
    Op(
        "set_legend_visible",
        "Show or hide the legend (creates one from the labelled artists if "
        "the axes has none).",
        (Param("visible", "bool", required=True), AX),
        _apply_legend_visible,
        lambda p, axv: (
            [f"{axv(p['ax_index'])}.legend()"]
            if p["create"]
            else [f"{axv(p['ax_index'])}.get_legend().set_visible({p['visible']})"]
        ),
        slot=lambda p: ("legend_visible", p["ax_index"]),
        legacy=lambda p: (_axname(p) + ".legend", "visible"),
    )
)


def _apply_legend_labels(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    ax = e._ax(p["ax_index"])
    texts = _legend_or_raise(ax, "set_legend_labels").get_texts()
    labels = p["labels"]
    if isinstance(labels, dict):
        mapping = {int(k): v for k, v in labels.items()}
    else:
        if len(labels) > len(texts):
            raise MethodIncompatibility(
                f"set_legend_labels: {len(labels)} labels for {len(texts)} entries"
            )
        mapping = dict(enumerate(labels))
    for k, v in mapping.items():
        if not isinstance(v, str):
            raise MethodIncompatibility(
                f"set_legend_labels: label {k} must be a string"
            )
        _pick(texts, k, "legend entry").set_text(v)
    return {**p, "labels": [t.get_text() for t in texts]}


register(
    Op(
        "set_legend_labels",
        "Rename legend entries: a list in entry order, or {index: text}.",
        (Param("labels", "any", required=True), AX),
        _apply_legend_labels,
        lambda p, axv: [
            f"for _t, _s in zip({axv(p['ax_index'])}.get_legend().get_texts(), "
            f"{lit(p['labels'])}):",
            "    _t.set_text(_s)",
        ],
        slot=lambda p: ("legend_labels", p["ax_index"]),
        legacy=lambda p: (_axname(p) + ".legend", "labels"),
    )
)


# ----------------------------------------------------------------------
# Annotations and free text
# ----------------------------------------------------------------------


def _apply_annotation(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    ax = e._ax(p["ax_index"])
    kw = {k: _color_val(v) if k == "color" else v for k, v in p["kwargs"].items()}
    ann = ax.annotate(p["text"], xy=tuple(p["xy"]), **kw)
    if p["draggable"]:
        ann.draggable()
    e.artist_roles[id(ann)] = ArtistRole.ANNOTATION
    return {**p, "serial": e._next_serial()}


def _code_annotation(p: Dict[str, Any], axv: AxVar) -> List[str]:
    extra = (", " + kwargs_src(p["kwargs"])) if p["kwargs"] else ""
    return [
        f"{axv(p['ax_index'])}.annotate({lit(p['text'])}, "
        f"xy={lit(tuple(p['xy']))}{extra})"
    ]


register(
    Op(
        "add_annotation",
        "Add a text annotation at data coordinates xy.",
        (
            Param("text", "text", required=True),
            Param("xy", "xy", required=True),
            AX,
            Param("draggable", "bool", default=False),
            Param("kwargs", "dict", default={}),
        ),
        _apply_annotation,
        _code_annotation,
        slot=lambda p: None,
        legacy=lambda p: (_axname(p) + ".annotation", "add"),
    )
)


def _apply_text(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    ax = e._ax(p["ax_index"])
    t = _pick(ax.texts, p["index"], "text")
    if _locked(e, t):
        raise MethodIncompatibility(
            f"set_text: text {p['index']} ({t.get_text()!r}) reports a result "
            "drawn by the plotting function; its wording is locked. Its style "
            "can still change, and set_text_visible can hide it."
        )
    t.set_text(p["text"])
    return p


register(
    Op(
        "set_text",
        "Change the wording of a text annotation (ax.texts[index]).",
        (
            Param("index", "int", required=True),
            Param("text", "text", required=True),
            AX,
        ),
        _apply_text,
        lambda p, axv: [
            f"{axv(p['ax_index'])}.texts[{p['index']}].set_text({lit(p['text'])})"
        ],
        slot=lambda p: ("text", p["ax_index"], p["index"]),
        legacy=lambda p: (_axname(p) + f".text{p['index']}", "text"),
    )
)


def _apply_text_visible(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    _pick(e._ax(p["ax_index"]).texts, p["index"], "text").set_visible(p["visible"])
    return p


register(
    Op(
        "set_text_visible",
        "Show or hide a text annotation (ax.texts[index]).",
        (
            Param("index", "int", required=True),
            Param("visible", "bool", required=True),
            AX,
        ),
        _apply_text_visible,
        lambda p, axv: [
            f"{axv(p['ax_index'])}.texts[{p['index']}].set_visible({p['visible']})"
        ],
        slot=lambda p: ("text_visible", p["ax_index"], p["index"]),
        legacy=lambda p: (_axname(p) + f".text{p['index']}", "visible"),
    )
)
