"""Figure editor operations: colours, series, line and collection style."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ..exceptions import MethodIncompatibility
from ._editor_ops import (
    AX,
    COLOR_TARGETS,
    LINE_STYLES,
    MIN_ALPHA,
    AxVar,
    Op,
    Param,
    _axname,
    _color_src,
    _color_val,
    _locked,
    _pick,
    _role,
    _target_index,
    _warn_contrast,
    all_palettes,
    lit,
    register,
)

if TYPE_CHECKING:
    from .interactive import FigureEditor


# ----------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------


def _apply_color(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    target, color = p["target"], _color_val(p["color"])
    if target == "suptitle":
        sup = getattr(e.fig, "_suptitle", None)
        if sup is None:
            raise MethodIncompatibility("set_color: the figure has no suptitle")
        sup.set_color(color)
        return p
    ax = e._ax(p["ax_index"])
    if target == "title":
        ax.title.set_color(color)
    elif target == "xlabel":
        ax.xaxis.label.set_color(color)
    elif target == "ylabel":
        ax.yaxis.label.set_color(color)
    else:
        j = _target_index(target, "line")
        if j is None:
            raise MethodIncompatibility(
                f"set_color: unknown target {target!r}; use one of "
                f"{list(COLOR_TARGETS)} or 'line<j>'"
            )
        line = _pick(ax.get_lines(), j, "line")
        if _locked(e, line):
            _warn_contrast(e, ax, color)
        line.set_color(color)
    return p


def _code_color(p: Dict[str, Any], axv: AxVar) -> List[str]:
    t, c = p["target"], _color_src(p["color"])
    if t == "suptitle":
        return [f"fig._suptitle.set_color({c})"]
    a = axv(p["ax_index"])
    if t == "title":
        return [f"{a}.title.set_color({c})"]
    if t in ("xlabel", "ylabel"):
        return [f"{a}.{t[0]}axis.label.set_color({c})"]
    return [f"{a}.get_lines()[{_target_index(t, 'line')}].set_color({c})"]


register(
    Op(
        "set_color",
        "Set the colour of a title, an axis label, the suptitle or one line "
        "('line<j>').",
        (
            Param("target", "text", required=True),
            Param("color", "color", required=True),
            AX,
        ),
        _apply_color,
        _code_color,
        slot=lambda p: ("color", p["ax_index"], p["target"]),
        restyles=lambda p: str(p["target"]).startswith("line"),
        legacy=lambda p: (_axname(p) + "." + p["target"], "color"),
    )
)


def _member(ax: Any, kind: str, j: int) -> Any:
    seq = {"line": ax.get_lines(), "coll": ax.collections, "patch": ax.patches}[kind]
    return _pick(seq, j, kind)


def _recolor_member(artist: Any, kind: str, old_hex: str, color: Any) -> Dict[str, Any]:
    """Recolour one artist of a colour group, keeping its own alpha.

    Returns the statements' arguments so the code generator can repeat them.
    """
    import matplotlib.colors as mcolors
    import numpy as np

    rgb = mcolors.to_rgb(_color_val(color))
    out: Dict[str, Any] = {}
    if kind == "line":
        artist.set_color(rgb)
        out["color"] = list(rgb)
        # Marker colours set explicitly to the series colour follow it;
        # 'auto' ones follow the line colour by themselves.
        for attr, key, setter in (
            ("_markerfacecolor", "mfc", artist.set_markerfacecolor),
            ("_markeredgecolor", "mec", artist.set_markeredgecolor),
        ):
            raw = getattr(artist, attr, None)
            if isinstance(raw, str) and raw in ("auto", "none", "None"):
                continue
            try:
                if mcolors.to_hex(raw, keep_alpha=False) != old_hex:
                    continue
                rgba = [*rgb, mcolors.to_rgba(raw)[3]]
            except (ValueError, TypeError):
                continue
            setter(tuple(rgba))
            out[key] = rgba
        return out
    import matplotlib.collections as mcoll

    def keep_alpha(colors: Any) -> List[List[float]]:
        arr = np.asarray(colors, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return []
        return [[*rgb, float(row[3])] for row in arr]

    if kind == "coll":
        if isinstance(artist, mcoll.LineCollection):
            new = keep_alpha(artist.get_edgecolor())
            artist.set_edgecolor(new)
            out["edgecolor"] = new
            return out
        fc = np.asarray(artist.get_facecolor(), dtype=float)
        if fc.size:
            new = keep_alpha(fc)
            artist.set_facecolor(new)
            out["facecolor"] = new
        ec = np.asarray(artist.get_edgecolor(), dtype=float)
        if ec.size and {mcolors.to_hex(r, keep_alpha=False) for r in ec} == {old_hex}:
            new = keep_alpha(ec)
            artist.set_edgecolor(new)
            out["edgecolor"] = new
        return out
    # patch
    if artist.get_fill():
        rgba = [*rgb, float(artist.get_facecolor()[3])]
        artist.set_facecolor(tuple(rgba))
        out["facecolor"] = rgba
    if mcolors.to_hex(artist.get_edgecolor(), keep_alpha=False) == old_hex:
        rgba = [*rgb, float(artist.get_edgecolor()[3])]
        artist.set_edgecolor(tuple(rgba))
        out["edgecolor"] = rgba
    return out


def _recolor_group(
    e: "FigureEditor", ax_index: int, group: int, color: Any
) -> List[Dict[str, Any]]:
    groups = e.color_groups(ax_index)
    g = _pick(groups, group, "colour group")
    ax = e._ax(ax_index)
    _warn_contrast(e, ax, color)
    changes = []
    from ._editor_roles import single_color

    for kind, j in g["members"]:
        artist = _member(ax, kind, j)
        # Compare against the colour the series has *now*: after an earlier
        # recolour, markers set explicitly to that colour must follow too.
        current = single_color(artist) or g["color"]
        change = _recolor_member(artist, kind, current, color)
        changes.append({"kind": kind, "index": j, **change})
    return changes


def _color_code(value: Any) -> str:
    """Source for one colour: a hex string when it says exactly the same
    thing (opaque, and the hex reads back to the same floats), else the
    RGB(A) tuple."""
    import matplotlib.colors as mcolors

    v = tuple(float(x) for x in value)
    if len(v) == 3 or (len(v) == 4 and v[3] == 1.0):
        hexed = mcolors.to_hex(v[:3])
        if tuple(mcolors.to_rgb(hexed)) == v[:3]:
            return repr(hexed)
    return lit(v)


def _code_changes(a: str, changes: List[Dict[str, Any]]) -> List[str]:
    out = []
    for ch in changes:
        ref = {
            "line": f"{a}.get_lines()[{ch['index']}]",
            "coll": f"{a}.collections[{ch['index']}]",
            "patch": f"{a}.patches[{ch['index']}]",
        }[ch["kind"]]
        if "color" in ch:
            out.append(f"{ref}.set_color({_color_code(ch['color'])})")
        if "mfc" in ch:
            out.append(f"{ref}.set_markerfacecolor({_color_code(ch['mfc'])})")
        if "mec" in ch:
            out.append(f"{ref}.set_markeredgecolor({_color_code(ch['mec'])})")
        for key in ("facecolor", "edgecolor"):
            if key in ch:
                val = ch[key]
                if ch["kind"] == "patch":
                    code = _color_code(val)
                else:
                    code = "[" + ", ".join(_color_code(v) for v in val) + "]"
                out.append(f"{ref}.set_{key}({code})")
    return out


def _apply_series_color(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    changes = _recolor_group(e, p["ax_index"], p["group"], p["color"])
    return {**p, "changes": changes}


register(
    Op(
        "set_series_color",
        "Recolour one series: every locked artist drawn in the series' "
        "original colour (points, line, error bars, confidence band) "
        "changes together. Groups are listed by color_groups().",
        (
            Param("group", "int", required=True),
            Param("color", "color", required=True),
            AX,
        ),
        _apply_series_color,
        lambda p, axv: _code_changes(axv(p["ax_index"]), p["changes"]),
        slot=lambda p: ("series_color", p["ax_index"], p["group"]),
        restyles=lambda p: True,
        legacy=lambda p: (_axname(p) + f".group{p['group']}", "color"),
    )
)


def _apply_palette(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    palettes = all_palettes()
    colors = p["colors"]
    if colors is None:
        if p["palette"] not in palettes:
            raise MethodIncompatibility(
                f"apply_palette: unknown palette {p['palette']!r}; "
                f"available: {sorted(palettes)}"
            )
        colors = palettes[p["palette"]]
    indices = range(len(e.fig.get_axes())) if p["ax_index"] is None else [p["ax_index"]]
    per_axes = []
    for i in indices:
        changes: List[Dict[str, Any]] = []
        for g in range(len(e.color_groups(i))):
            changes += _recolor_group(e, i, g, colors[g % len(colors)])
        per_axes.append({"ax_index": i, "changes": changes})
    return {**p, "per_axes": per_axes}


def _code_palette(p: Dict[str, Any], axv: AxVar) -> List[str]:
    out: List[str] = []
    for block in p["per_axes"]:
        out += _code_changes(axv(block["ax_index"]), block["changes"])
    return out


register(
    Op(
        "apply_palette",
        "Recolour the series of one axes (or all, ax_index=None) with a "
        "palette, one colour per colour group. Palettes: the StatsPAI theme "
        "palettes, 'okabe_ito' (colour-blind safe), 'tab10', 'grayscale'; "
        "or pass colors=[...].",
        (
            Param("palette", "text", default="okabe_ito"),
            Param("colors", "list", default=None),
            Param("ax_index", "int", default=None, description="None = all axes."),
        ),
        _apply_palette,
        _code_palette,
        scope="all",
        slot=lambda p: ("palette", p["ax_index"]),
        restyles=lambda p: True,
        legacy=lambda p: ("palette", "name"),
    )
)


def _apply_scatter_color(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    ax = e._ax(p["ax_index"])
    coll = _pick(ax.collections, p["scatter_index"], "collection")
    if coll.get_array() is not None:
        raise MethodIncompatibility(
            "set_scatter_color: this collection is colour-mapped; its colours "
            "encode data values and are locked."
        )
    if _locked(e, coll):
        _warn_contrast(e, ax, p["color"])
    coll.set_facecolor(_color_val(p["color"]))
    return p


register(
    Op(
        "set_scatter_color",
        "Set the face colour of one scatter collection.",
        (
            Param("scatter_index", "int", required=True),
            Param("color", "color", required=True),
            AX,
        ),
        _apply_scatter_color,
        lambda p, axv: [
            f"{axv(p['ax_index'])}.collections[{p['scatter_index']}]"
            f".set_facecolor({_color_src(p['color'])})"
        ],
        slot=lambda p: ("scatter_color", p["ax_index"], p["scatter_index"]),
        restyles=lambda p: True,
        legacy=lambda p: (_axname(p) + f".scatter{p['scatter_index']}", "color"),
    )
)


def _apply_background(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    color = _color_val(p["color"])
    if p["target"] == "figure":
        e.fig.set_facecolor(color)
    else:
        ax = e._ax(p["ax_index"])
        ax.set_facecolor(color)
        from ._editor_roles import single_color

        for artist in [*ax.get_lines(), *ax.collections, *ax.patches]:
            if _locked(e, artist) and single_color(artist) is not None:
                import matplotlib.colors as mcolors

                bg = mcolors.to_rgb(color)
                fg = mcolors.to_rgb(single_color(artist))
                if sum(abs(a - b) for a, b in zip(bg, fg)) < 0.15:
                    warnings.warn(
                        f"Background {color!r} is almost the colour of a data "
                        "series on this axes; that series will be hard to see.",
                        UserWarning,
                        stacklevel=4,
                    )
                    break
    return p


register(
    Op(
        "set_background_color",
        "Set the figure or axes background colour.",
        (
            Param("color", "color", required=True),
            Param("target", "choice", default="figure", choices=("figure", "axes")),
            AX,
        ),
        _apply_background,
        lambda p, axv: [
            (
                "fig.set_facecolor(" + _color_src(p["color"]) + ")"
                if p["target"] == "figure"
                else f"{axv(p['ax_index'])}.set_facecolor({_color_src(p['color'])})"
            )
        ],
        scope="axis",
        slot=lambda p: ("background", p["target"], p["ax_index"]),
        legacy=lambda p: (
            "fig.facecolor" if p["target"] == "figure" else _axname(p) + ".facecolor",
            "color",
        ),
    )
)


# ----------------------------------------------------------------------
# Line and collection style (data-protected visibility)
# ----------------------------------------------------------------------


def _visible_after(
    line: Any, ls: Any = None, marker: Any = None, lw: Any = None, ms: Any = None
) -> bool:
    ls = line.get_linestyle() if ls is None else ls
    marker = line.get_marker() if marker is None else marker
    lw = line.get_linewidth() if lw is None else lw
    ms = line.get_markersize() if ms is None else ms
    has_line = ls not in ("None", "none", "", " ") and lw > 0
    has_marker = marker not in ("None", "none", "", " ", None) and ms > 0
    return has_line or has_marker


def _refuse_invisible(e: "FigureEditor", line: Any, op: str, **after: Any) -> None:
    if _locked(e, line) and not _visible_after(line, **after):
        raise MethodIncompatibility(
            f"{op}: this would make a locked {_role(e, line).name} line "
            "invisible (no line and no markers). Data stays visible while "
            "protect_data=True."
        )


def _line_op(
    name: str,
    pname: str,
    kind: str,
    setter: str,
    description: str,
    check: Optional[Callable[["FigureEditor", Any, Any], None]] = None,
    choices: Optional[Tuple[Any, ...]] = None,
) -> None:
    def apply(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
        ax = e._ax(p["ax_index"])
        line = _pick(ax.get_lines(), p["line_index"], "line")
        if check is not None:
            check(e, line, p[pname])
        getattr(line, setter)(p[pname])
        return p

    register(
        Op(
            name,
            description,
            (
                Param("line_index", "int", required=True),
                Param(pname, kind, required=True, choices=choices),
                AX,
            ),
            apply,
            lambda p, axv: [
                f"{axv(p['ax_index'])}.get_lines()[{p['line_index']}]"
                f".{setter}({lit(p[pname])})"
            ],
            slot=lambda p: (name, p["ax_index"], p["line_index"]),
            restyles=lambda p: True,
            legacy=lambda p: (
                _axname(p) + f".line{p['line_index']}",
                {
                    "set_linewidth": "linewidth",
                    "set_linestyle": "linestyle",
                    "set_marker": "marker",
                    "set_markersize": "markersize",
                }[name],
            ),
        )
    )


def _check_width(e: "FigureEditor", line: Any, v: float) -> None:
    if v < 0:
        raise MethodIncompatibility("set_linewidth: width must be >= 0")
    _refuse_invisible(e, line, "set_linewidth", lw=v)


def _check_style(e: "FigureEditor", line: Any, v: str) -> None:
    _refuse_invisible(e, line, "set_linestyle", ls=v)


def _check_marker(e: "FigureEditor", line: Any, v: str) -> None:
    _refuse_invisible(e, line, "set_marker", marker=v)


def _check_msize(e: "FigureEditor", line: Any, v: float) -> None:
    if v < 0:
        raise MethodIncompatibility("set_markersize: size must be >= 0")
    _refuse_invisible(e, line, "set_markersize", ms=v)


_line_op(
    "set_linewidth",
    "width",
    "number",
    "set_linewidth",
    "Set a line's width.",
    _check_width,
)
_line_op(
    "set_linestyle",
    "style",
    "choice",
    "set_linestyle",
    "Set a line's style.",
    _check_style,
    choices=LINE_STYLES + ("solid", "dashed", "dashdot", "dotted"),
)
_line_op(
    "set_marker", "marker", "text", "set_marker", "Set a line's marker.", _check_marker
)
_line_op(
    "set_markersize",
    "size",
    "number",
    "set_markersize",
    "Set a line's marker size.",
    _check_msize,
)


def _alpha_artist(ax: Any, target: str) -> Tuple[Any, str]:
    for prefix, seq, ref in (
        ("line", ax.get_lines(), "get_lines()"),
        ("scatter", ax.collections, "collections"),
        ("ci", ax.collections, "collections"),
        ("coll", ax.collections, "collections"),
        ("patch", ax.patches, "patches"),
    ):
        j = _target_index(target, prefix)
        if j is not None:
            return _pick(seq, j, prefix), f"{ref}[{j}]"
    raise MethodIncompatibility(
        f"set_alpha: unknown target {target!r}; use line<j>, scatter<j>, "
        "ci<j>, coll<j> or patch<j>"
    )


def _apply_alpha(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    alpha = p["alpha"]
    if not 0 <= alpha <= 1:
        raise MethodIncompatibility(f"set_alpha: alpha must be in [0, 1], got {alpha}")
    artist, _ = _alpha_artist(e._ax(p["ax_index"]), p["target"])
    if _locked(e, artist):
        floor = MIN_ALPHA[_role(e, artist)]
        if alpha < floor:
            warnings.warn(
                f"alpha={alpha} would make a locked {_role(e, artist).name} "
                f"element nearly invisible; using {floor}.",
                UserWarning,
                stacklevel=4,
            )
            alpha = floor
    artist.set_alpha(alpha)
    return {**p, "alpha": float(alpha)}


register(
    Op(
        "set_alpha",
        "Set the transparency of a line, collection or patch. Locked data "
        "keeps a minimum alpha (0.2 for data and fits, 0.05 for intervals).",
        (
            Param("target", "text", required=True),
            Param("alpha", "number", required=True),
            AX,
        ),
        _apply_alpha,
        lambda p, axv: [
            f"{axv(p['ax_index'])}.{_ref_for(p['target'])}.set_alpha({lit(p['alpha'])})"
        ],
        slot=lambda p: ("alpha", p["ax_index"], _ref_for(p["target"])),
        restyles=lambda p: True,
        legacy=lambda p: (_axname(p) + "." + p["target"], "alpha"),
    )
)


def _ref_for(target: str) -> str:
    for prefix, ref in (
        ("line", "get_lines()"),
        ("scatter", "collections"),
        ("ci", "collections"),
        ("coll", "collections"),
        ("patch", "patches"),
    ):
        j = _target_index(target, prefix)
        if j is not None:
            return f"{ref}[{j}]"
    return target
