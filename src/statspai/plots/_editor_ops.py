"""The edit operations of the figure editor.

Every edit is one named operation with typed parameters. One definition
serves the live edit, the replay of a saved style (JSON), the reproducible
code and the schema shown to agents and web front ends:

* ``apply(editor, params)`` changes the figure and returns the *resolved*
  parameters: the complete state the edit set (a grid edit records colour,
  alpha, line style and width, not only the one that changed). Resolved
  parameters make every operation a full setter of its *slot*, so an earlier
  edit of the same slot can be dropped from the generated code without
  changing the result.
* ``code(resolved, axv)`` returns the matplotlib statements that make the
  same change on a freshly drawn figure. ``axv(i)`` names the variable for
  axes ``i`` (``ax``, ``ax1``, ...); the editor defines those variables.

Code is always generated from the parameters. A ``code`` string arriving
from outside (an old JSON file, a web request) is never trusted or emitted.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ..exceptions import MethodIncompatibility
from ._editor_roles import LOCKED_ROLES, ArtistRole

if TYPE_CHECKING:
    from .interactive import FigureEditor

AxVar = Callable[[int], str]

# Lowest alpha the editor lets locked artists go to: points and fitted lines
# stay clearly visible; confidence bands are often drawn very light already.
MIN_ALPHA = {ArtistRole.DATA: 0.2, ArtistRole.FIT: 0.2, ArtistRole.CI: 0.05}

PALETTES: Dict[str, List[str]] = {
    "okabe_ito": [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ],
    "tab10": [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
    "grayscale": ["#000000", "#555555", "#888888", "#bbbbbb"],
}

LEGEND_LOCS = (
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
)
LINE_STYLES = ("-", "--", "-.", ":", "None")
FONT_FAMILIES = ("serif", "sans-serif", "monospace")
FONT_WEIGHTS = ("normal", "bold", "light", "semibold", "heavy")
SPINES = ("top", "right", "bottom", "left")
FONTSIZE_TARGETS = ("title", "xlabel", "ylabel", "ticks", "legend", "suptitle")
COLOR_TARGETS = ("title", "xlabel", "ylabel", "suptitle")  # plus "line<j>"


def all_palettes() -> Dict[str, List[str]]:
    from .themes import _PALETTES

    out = {k: list(v) for k, v in _PALETTES.items()}
    out.update(PALETTES)
    return out


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    name: str
    kind: str  # text number int bool color choice list dict xy
    required: bool = False
    default: Any = None
    choices: Optional[Tuple[Any, ...]] = None
    description: str = ""

    def schema(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "type": self.kind,
            "required": self.required,
            "description": self.description,
        }
        if not self.required:
            out["default"] = self.default
        if self.choices is not None:
            out["enum"] = list(self.choices)
        return out


def _literal(value: Any, where: str, keywords: bool = False) -> Any:
    """Plain Python value (str, number, bool, None, list, dict) or raise.

    ``keywords=True`` also requires dict keys to be identifiers (the dict
    becomes ``**kwargs`` in generated code).
    """
    import numpy as np

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MethodIncompatibility(f"{where}: non-finite number {value!r}")
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_literal(v, where) for v in list(value)]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if keywords and not (isinstance(k, str) and k.isidentifier()):
                raise MethodIncompatibility(
                    f"{where}: keyword {k!r} is not an identifier"
                )
            if not isinstance(k, (str, int)) or isinstance(k, bool):
                raise MethodIncompatibility(f"{where}: unsupported key {k!r}")
            out[k] = _literal(v, f"{where}.{k}")
        return out
    raise MethodIncompatibility(
        f"{where}: unsupported value {value!r} ({type(value).__name__})"
    )


def _convert(p: Param, value: Any, op: str) -> Any:
    where = f"{op}({p.name}=)"
    if p.kind == "text":
        if not isinstance(value, str):
            raise MethodIncompatibility(f"{where} expects a string, got {value!r}")
        return value
    if p.kind in ("number", "int"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            try:
                import numpy as np

                if isinstance(value, np.generic):
                    value = value.item()
            except ImportError:  # pragma: no cover
                pass
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MethodIncompatibility(f"{where} expects a number, got {value!r}")
        if not math.isfinite(float(value)):
            raise MethodIncompatibility(f"{where} must be finite, got {value!r}")
        return int(value) if p.kind == "int" else float(value)
    if p.kind == "bool":
        if isinstance(value, bool) or value in (0, 1):
            return bool(value)
        raise MethodIncompatibility(f"{where} expects true/false, got {value!r}")
    if p.kind == "color":
        import matplotlib.colors as mcolors

        val = tuple(value) if isinstance(value, list) else value
        try:
            mcolors.to_rgba(val)
        except (ValueError, TypeError):
            raise MethodIncompatibility(
                f"{where}: {value!r} is not a matplotlib colour"
            ) from None
        return _literal(val, where)
    if p.kind == "choice":
        assert p.choices is not None
        if value not in p.choices:
            raise MethodIncompatibility(
                f"{where} must be one of {list(p.choices)}, got {value!r}"
            )
        return value
    if p.kind == "xy":
        vals = _literal(value, where)
        if not (isinstance(vals, list) and len(vals) == 2):
            raise MethodIncompatibility(f"{where} expects (x, y), got {value!r}")
        return [float(v) for v in vals]
    if p.kind == "list":
        vals = _literal(value, where)
        if not isinstance(vals, list):
            raise MethodIncompatibility(f"{where} expects a list, got {value!r}")
        return vals
    if p.kind == "dict":
        vals = _literal(value if value is not None else {}, where, keywords=True)
        if not isinstance(vals, dict):
            raise MethodIncompatibility(f"{where} expects a dict of keyword arguments")
        return vals
    return _literal(value, where)


# ----------------------------------------------------------------------
# Operation registry
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Op:
    name: str
    description: str
    params: Tuple[Param, ...]
    apply: Callable[["FigureEditor", Dict[str, Any]], Optional[Dict[str, Any]]]
    code: Callable[[Dict[str, Any], AxVar], List[str]]
    scope: str = "axis"  # axis | figure | all
    slot: Optional[Callable[[Dict[str, Any]], Optional[tuple]]] = None
    legacy: Callable[[Dict[str, Any]], Tuple[str, str]] = field(
        default=lambda p: ("fig", "edit")
    )
    # True when the edit changes how series look, so legend entries drawn
    # from them must be rebuilt (see ``refresh_legend``).
    restyles: Callable[[Dict[str, Any]], bool] = field(default=lambda p: False)

    def normalize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        known = {p.name: p for p in self.params}
        unknown = sorted(set(raw) - set(known))
        if unknown:
            raise MethodIncompatibility(
                f"{self.name}: unknown parameter(s) {unknown}; "
                f"accepted: {sorted(known)}"
            )
        out: Dict[str, Any] = {}
        for p in self.params:
            if p.name in raw and raw[p.name] is not None:
                out[p.name] = _convert(p, raw[p.name], self.name)
            elif p.required:
                raise MethodIncompatibility(
                    f"{self.name}: missing required parameter {p.name!r}"
                )
            else:
                out[p.name] = p.default
        return out

    def axes(self, params: Dict[str, Any], n_axes: int) -> List[int]:
        if self.scope == "all":
            return list(range(n_axes))
        if self.scope == "figure":
            return []
        return [int(params.get("ax_index") or 0)]

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "params": [p.schema() for p in self.params],
        }


OPS: Dict[str, Op] = {}

AX = Param("ax_index", "int", default=0, description="Axes index in fig.axes.")


def register(op: Op) -> Op:
    OPS[op.name] = op
    return op


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def lit(value: Any) -> str:
    """Source text for a plain value (exact for floats)."""
    import numpy as np

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, tuple):
        return (
            "("
            + ", ".join(lit(v) for v in value)
            + ("," if len(value) == 1 else "")
            + ")"
        )
    if isinstance(value, list):
        return "[" + ", ".join(lit(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {lit(v)}" for k, v in value.items()) + "}"
    return repr(value)


def kwargs_src(kw: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={lit(v)}" for k, v in kw.items())


def _color_src(c: Any) -> str:
    return lit(tuple(c) if isinstance(c, list) else c)


def _color_val(c: Any) -> Any:
    return tuple(c) if isinstance(c, list) else c


def _role(editor: "FigureEditor", artist: Any) -> ArtistRole:
    return editor.artist_roles.get(id(artist), ArtistRole.COSMETIC)


def _locked(editor: "FigureEditor", artist: Any) -> bool:
    return editor.protect_data and _role(editor, artist) in LOCKED_ROLES


def _pick(seq: Any, index: int, what: str) -> Any:
    items = list(seq)
    if not 0 <= index < len(items):
        raise MethodIncompatibility(
            f"{what} index {index} out of range (0..{len(items) - 1})"
        )
    return items[index]


def _target_index(target: str, prefix: str) -> Optional[int]:
    if target.startswith(prefix) and target[len(prefix) :].isdigit():
        return int(target[len(prefix) :])
    return None


def _font_size_pt(value: Any) -> float:
    from matplotlib.font_manager import FontProperties

    return float(FontProperties(size=value).get_size_in_points())


def _has_set_loc() -> bool:
    from matplotlib.legend import Legend

    return hasattr(Legend, "set_loc")


def _has_labelfontfamily() -> bool:
    import matplotlib

    major, minor = (int(x) for x in matplotlib.__version__.split(".")[:2])
    return (major, minor) >= (3, 8)


def _legend_or_raise(ax: Any, op: str) -> Any:
    legend = ax.get_legend()
    if legend is None:
        raise MethodIncompatibility(
            f"{op}: this axes has no legend. Call set_legend_visible(True) "
            "first to create one from the labelled artists."
        )
    return legend


def _warn_contrast(editor: "FigureEditor", ax: Any, color: Any) -> None:
    """Warn when a locked series would be drawn in the background colour."""
    import matplotlib.colors as mcolors

    bg = mcolors.to_rgb(ax.get_facecolor())
    c = mcolors.to_rgb(_color_val(color))
    if sum(abs(a - b) for a, b in zip(bg, c)) < 0.15:
        warnings.warn(
            f"Colour {color!r} is almost the axes background {mcolors.to_hex(bg)}; "
            "the data drawn in it will be hard to see.",
            UserWarning,
            stacklevel=4,
        )


# ----------------------------------------------------------------------
# Text: titles and labels
# ----------------------------------------------------------------------


def _apply_title(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    ax = e._ax(p["ax_index"])
    # Text.set_text keeps the title's size and weight; Axes.set_title would
    # reset them to the rcParams defaults.
    ax.title.set_text(p["text"])
    if p["kwargs"]:
        ax.title.set(**p["kwargs"])
    return p


def _code_title(p: Dict[str, Any], axv: AxVar) -> List[str]:
    a = axv(p["ax_index"])
    out = [f"{a}.title.set_text({lit(p['text'])})"]
    if p["kwargs"]:
        out.append(f"{a}.title.set({kwargs_src(p['kwargs'])})")
    return out


register(
    Op(
        "set_title",
        "Set the axes title text.",
        (Param("text", "text", required=True), AX, Param("kwargs", "dict", default={})),
        _apply_title,
        _code_title,
        slot=lambda p: ("title", p["ax_index"], tuple(sorted(p["kwargs"]))),
        legacy=lambda p: (_axname(p) + ".title", "text"),
    )
)


def _axname(p: Dict[str, Any]) -> str:
    i = int(p.get("ax_index") or 0)
    return "ax" if i == 0 else f"ax{i}"


def _make_label_op(axis: str) -> None:
    attr = f"{axis}axis"

    def apply(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
        ax = e._ax(p["ax_index"])
        getattr(ax, f"set_{axis}label")(p["text"])
        if p["kwargs"]:
            getattr(ax, attr).label.set(**p["kwargs"])
        return p

    def code(p: Dict[str, Any], axv: AxVar) -> List[str]:
        a = axv(p["ax_index"])
        out = [f"{a}.set_{axis}label({lit(p['text'])})"]
        if p["kwargs"]:
            out.append(f"{a}.{attr}.label.set({kwargs_src(p['kwargs'])})")
        return out

    register(
        Op(
            f"set_{axis}label",
            f"Set the {axis}-axis label text.",
            (
                Param("text", "text", required=True),
                AX,
                Param("kwargs", "dict", default={}),
            ),
            apply,
            code,
            slot=lambda p: (f"{axis}label", p["ax_index"], tuple(sorted(p["kwargs"]))),
            legacy=lambda p: (_axname(p) + f".{axis}label", "text"),
        )
    )


_make_label_op("x")
_make_label_op("y")


def _apply_suptitle(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    fig = e.fig
    current = getattr(fig, "_suptitle", None)
    if current is not None:
        x, y = current.get_position()
        res = {
            "text": p["text"],
            "fontsize": float(current.get_fontsize()),
            "fontweight": current.get_fontweight(),
            "color": _hex_or_raw(current.get_color()),
            "x": float(x),
            "y": float(y),
        }
    else:
        res = {"text": p["text"]}
    res.update(p["kwargs"])
    kw = {k: v for k, v in res.items() if k != "text"}
    fig.suptitle(res["text"], **{k: _color_val(v) for k, v in kw.items()})
    return {"text": res["text"], "kwargs": kw}


def _hex_or_raw(c: Any) -> Any:
    import matplotlib.colors as mcolors

    try:
        return mcolors.to_hex(c, keep_alpha=True)
    except (ValueError, TypeError):
        return c


def _code_suptitle(p: Dict[str, Any], axv: AxVar) -> List[str]:
    kw = p["kwargs"]
    extra = (", " + kwargs_src(kw)) if kw else ""
    return [f"fig.suptitle({lit(p['text'])}{extra})"]


register(
    Op(
        "set_suptitle",
        "Set (or create) the figure-level title.",
        (Param("text", "text", required=True), Param("kwargs", "dict", default={})),
        _apply_suptitle,
        _code_suptitle,
        scope="figure",
        slot=lambda p: ("suptitle",),
        legacy=lambda p: ("fig.suptitle", "text"),
    )
)


# ----------------------------------------------------------------------
# Axis limits (data-protected)
# ----------------------------------------------------------------------


def _make_lim_op(axis: str) -> None:
    lo_name, hi_name = ("left", "right") if axis == "x" else ("bottom", "top")

    def apply(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
        from ._editor_roles import expand_to_cover, locked_extent

        ax = e._ax(p["ax_index"])
        low, high = p[lo_name], p[hi_name]
        if low == high:
            raise MethodIncompatibility(
                f"set_{axis}lim: the two limits are equal ({low})"
            )
        if e.protect_data:
            d_lo, d_hi = locked_extent(ax, axis, e.artist_roles)
            if d_lo is not None and d_hi is not None:
                new_lo, new_hi, changed = expand_to_cover(
                    ax, axis, low, high, d_lo, d_hi
                )
                if changed:
                    warnings.warn(
                        f"Axis limits [{low:.4g}, {high:.4g}] would hide data, "
                        f"error bars or confidence bands spanning "
                        f"[{d_lo:.4g}, {d_hi:.4g}]. Expanded to "
                        f"[{new_lo:.4g}, {new_hi:.4g}].",
                        UserWarning,
                        stacklevel=4,
                    )
                low, high = new_lo, new_hi
        getattr(ax, f"set_{axis}lim")(low, high)
        return {lo_name: float(low), hi_name: float(high), "ax_index": p["ax_index"]}

    def code(p: Dict[str, Any], axv: AxVar) -> List[str]:
        return [
            f"{axv(p['ax_index'])}.set_{axis}lim({lit(p[lo_name])}, {lit(p[hi_name])})"
        ]

    register(
        Op(
            f"set_{axis}lim",
            f"Set the {axis}-axis limits. With data protection on, limits that "
            "would hide a data point, error bar or confidence band are widened "
            "(with a warning).",
            (
                Param(lo_name, "number", required=True),
                Param(hi_name, "number", required=True),
                AX,
            ),
            apply,
            code,
            slot=lambda p: (f"{axis}lim", p["ax_index"]),
            legacy=lambda p: (_axname(p) + f".{axis}lim", "range"),
        )
    )


_make_lim_op("x")
_make_lim_op("y")


# ----------------------------------------------------------------------
# Font sizes, fonts, presets
# ----------------------------------------------------------------------


def _apply_fontsize(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    target, size = p["target"], p["size"]
    if size <= 0:
        raise MethodIncompatibility(f"set_fontsize: size must be positive, got {size}")
    if target == "suptitle":
        sup = getattr(e.fig, "_suptitle", None)
        if sup is None:
            raise MethodIncompatibility("set_fontsize: the figure has no suptitle")
        sup.set_fontsize(size)
        return p
    ax = e._ax(p["ax_index"])
    if target == "title":
        ax.title.set_fontsize(size)
    elif target == "xlabel":
        ax.xaxis.label.set_fontsize(size)
    elif target == "ylabel":
        ax.yaxis.label.set_fontsize(size)
    elif target == "ticks":
        ax.tick_params(which="both", labelsize=size)
    elif target == "legend":
        for t in _legend_or_raise(ax, "set_fontsize").get_texts():
            t.set_fontsize(size)
    return p


def _code_fontsize(p: Dict[str, Any], axv: AxVar) -> List[str]:
    t, s = p["target"], lit(p["size"])
    if t == "suptitle":
        return [f"fig._suptitle.set_fontsize({s})"]
    a = axv(p["ax_index"])
    if t == "title":
        return [f"{a}.title.set_fontsize({s})"]
    if t == "xlabel":
        return [f"{a}.xaxis.label.set_fontsize({s})"]
    if t == "ylabel":
        return [f"{a}.yaxis.label.set_fontsize({s})"]
    if t == "ticks":
        return [f"{a}.tick_params(which='both', labelsize={s})"]
    return [f"for _t in {a}.get_legend().get_texts():", f"    _t.set_fontsize({s})"]


register(
    Op(
        "set_fontsize",
        "Set the font size (points) of a title, an axis label, the tick "
        "labels, the legend entries or the suptitle.",
        (
            Param("target", "choice", required=True, choices=FONTSIZE_TARGETS),
            Param("size", "number", required=True),
            AX,
        ),
        _apply_fontsize,
        _code_fontsize,
        slot=lambda p: ("fontsize", p["ax_index"], p["target"]),
        legacy=lambda p: (_axname(p) + "." + p["target"], "fontsize"),
    )
)


def _family_list(family: str, name: Optional[str]) -> List[str]:
    from ._editor_presets import clean_font_name

    if name:
        return [clean_font_name(name), family]
    return [family]


def _check_font_installed(name: str) -> None:
    from matplotlib.font_manager import FontProperties, findfont

    try:
        findfont(FontProperties(family=name), fallback_to_default=False)
    except ValueError:
        warnings.warn(
            f"Font {name!r} is not installed; matplotlib will draw the text in "
            "the fallback font of the chosen family.",
            UserWarning,
            stacklevel=4,
        )


def _apply_family(ax: Any, fam: List[str]) -> None:
    texts = [
        ax.title,
        ax.xaxis.label,
        ax.yaxis.label,
        ax.xaxis.get_offset_text(),
        ax.yaxis.get_offset_text(),
        *ax.texts,
    ]
    legend = ax.get_legend()
    if legend is not None:
        texts += [*legend.get_texts(), legend.get_title()]
    for t in texts:
        t.set_fontfamily(fam)
    if _has_labelfontfamily():
        ax.tick_params(which="both", labelfontfamily=fam)
    else:  # pragma: no cover - matplotlib < 3.8
        for axis in (ax.xaxis, ax.yaxis):
            for tick in list(axis.majorTicks) + list(axis.minorTicks):
                tick.label1.set_fontfamily(fam)
                tick.label2.set_fontfamily(fam)


def _code_family(a: str, fam: List[str]) -> List[str]:
    out = [
        f"_texts = [{a}.title, {a}.xaxis.label, {a}.yaxis.label, "
        f"{a}.xaxis.get_offset_text(), {a}.yaxis.get_offset_text(), *{a}.texts]",
        f"if {a}.get_legend() is not None:",
        f"    _texts += [*{a}.get_legend().get_texts(), {a}.get_legend().get_title()]",
        "for _t in _texts:",
        f"    _t.set_fontfamily({lit(fam)})",
    ]
    if _has_labelfontfamily():
        out.append(f"{a}.tick_params(which='both', labelfontfamily={lit(fam)})")
    else:  # pragma: no cover - matplotlib < 3.8
        out += [
            f"for _tick in {a}.xaxis.majorTicks + {a}.xaxis.minorTicks"
            f" + {a}.yaxis.majorTicks + {a}.yaxis.minorTicks:",
            f"    _tick.label1.set_fontfamily({lit(fam)})",
            f"    _tick.label2.set_fontfamily({lit(fam)})",
        ]
    return out


def _apply_font(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    fam = _family_list(p["font_family"], p["font_name"])
    if p["font_name"]:
        _check_font_installed(fam[0])
    _apply_family(e._ax(p["ax_index"]), fam)
    return {**p, "family_list": fam}


register(
    Op(
        "set_font",
        "Use one font for every text element of the axes (title, labels, "
        "ticks, legend, annotations). Applied to the figure only; global "
        "matplotlib settings are left alone.",
        (
            Param("font_family", "choice", required=True, choices=FONT_FAMILIES),
            Param("font_name", "text", default=None),
            AX,
        ),
        _apply_font,
        lambda p, axv: _code_family(axv(p["ax_index"]), p["family_list"]),
        slot=lambda p: ("font", p["ax_index"]),
        legacy=lambda p: (_axname(p) + ".font", "family"),
    )
)


def _sizes(ax: Any, title: float, label: float, tick: float) -> None:
    ax.title.set_fontsize(title)
    ax.xaxis.label.set_fontsize(label)
    ax.yaxis.label.set_fontsize(label)
    ax.tick_params(which="both", labelsize=tick)


def _code_sizes(a: str, title: float, label: float, tick: float) -> List[str]:
    return [
        f"{a}.title.set_fontsize({lit(title)})",
        f"{a}.xaxis.label.set_fontsize({lit(label)})",
        f"{a}.yaxis.label.set_fontsize({lit(label)})",
        f"{a}.tick_params(which='both', labelsize={lit(tick)})",
    ]


def _apply_font_preset(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    from ._editor_presets import (
        FONT_PRESETS,
        _resolve_preset_fonts,
        resolve_preset_name,
    )

    name = resolve_preset_name(p["preset"])
    preset = FONT_PRESETS[name]
    fam = _resolve_preset_fonts(preset) + [preset["family"]]
    ax = e._ax(p["ax_index"])
    _apply_family(ax, fam)
    sizes = (
        float(preset["title_size"]),
        float(preset["label_size"]),
        float(preset["tick_size"]),
    )
    _sizes(ax, *sizes)
    return {**p, "preset": name, "family_list": fam, "sizes": list(sizes)}


register(
    Op(
        "apply_font_preset",
        "Apply a journal / thesis font preset (font and sizes).",
        (Param("preset", "text", required=True), AX),
        _apply_font_preset,
        lambda p, axv: _code_family(axv(p["ax_index"]), p["family_list"])
        + _code_sizes(axv(p["ax_index"]), *p["sizes"]),
        slot=lambda p: ("font_preset", p["ax_index"]),
        legacy=lambda p: (_axname(p) + ".font_preset", "preset"),
    )
)


def _apply_size_preset(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    from ._editor_presets import SIZE_PRESETS

    if p["preset"] not in SIZE_PRESETS:
        raise MethodIncompatibility(
            f"Unknown size preset {p['preset']!r}. Available: "
            f"{', '.join(SIZE_PRESETS)}"
        )
    sp = SIZE_PRESETS[p["preset"]]
    sizes = (float(sp["title_size"]), float(sp["label_size"]), float(sp["tick_size"]))
    _sizes(e._ax(p["ax_index"]), *sizes)
    return {**p, "sizes": list(sizes)}


register(
    Op(
        "apply_size_preset",
        "Apply a size preset (title, label and tick sizes) without changing "
        "the font.",
        (Param("preset", "text", required=True), AX),
        _apply_size_preset,
        lambda p, axv: _code_sizes(axv(p["ax_index"]), *p["sizes"]),
        slot=lambda p: ("size_preset", p["ax_index"]),
        legacy=lambda p: (_axname(p) + ".size_preset", "preset"),
    )
)


def _apply_title_weight(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    e._ax(p["ax_index"]).title.set_fontweight(p["weight"])
    return p


register(
    Op(
        "set_title_weight",
        "Set the title font weight.",
        (Param("weight", "choice", required=True, choices=FONT_WEIGHTS), AX),
        _apply_title_weight,
        lambda p, axv: [
            f"{axv(p['ax_index'])}.title.set_fontweight({lit(p['weight'])})"
        ],
        slot=lambda p: ("title_weight", p["ax_index"]),
        legacy=lambda p: (_axname(p) + ".title", "fontweight"),
    )
)


# The remaining operations live in their own modules; importing them here
# registers them whenever this module is imported.
from . import _editor_ops_layout, _editor_ops_style, _editor_theme  # noqa: E402,F401
