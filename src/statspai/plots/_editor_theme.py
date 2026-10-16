"""The ``apply_theme`` edit: restyle one existing figure with a theme.

``sp.set_theme`` changes matplotlib's global settings, which only affect
figures drawn *afterwards*. Here the theme's settings are computed inside an
``rc_context`` (so the global settings are untouched) and written onto the
figure's own artists; the generated code repeats those statements, so it
reproduces the result on a freshly drawn figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from ._editor_ops import (
    SPINES,
    AxVar,
    Op,
    Param,
    _apply_family,
    _code_family,
    lit,
    register,
)
from ._editor_ops_style import _code_changes, _recolor_group

if TYPE_CHECKING:
    from .interactive import FigureEditor

_GENERIC = ("serif", "sans-serif", "monospace", "cursive", "fantasy")


def _hex(c: Any) -> str:
    import matplotlib.colors as mcolors

    rgba = mcolors.to_rgba(c)
    return str(mcolors.to_hex(rgba, keep_alpha=rgba[3] < 1))


def theme_spec(name: str) -> Dict[str, Any]:
    """Resolve a theme to concrete values without touching global rcParams."""
    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties

    from .themes import set_theme

    with mpl.rc_context():
        set_theme(name)
        rc = mpl.rcParams

        def pt(key: str) -> float:
            return float(FontProperties(size=rc[key]).get_size_in_points())

        def tick(axis: str) -> Dict[str, Any]:
            color = rc[f"{axis}tick.color"]
            labelcolor = rc[f"{axis}tick.labelcolor"]
            return {
                "labelsize": pt(f"{axis}tick.labelsize"),
                "direction": rc[f"{axis}tick.direction"],
                "color": _hex(color),
                "labelcolor": _hex(color if labelcolor == "inherit" else labelcolor),
            }

        text_color = _hex(rc["text.color"])
        families: List[str] = []
        for fam in list(rc["font.family"]):
            if fam in _GENERIC:
                families += [f for f in rc[f"font.{fam}"] if f not in families]
            elif fam not in families:
                families.append(fam)
        if rc["font.family"] and rc["font.family"][0] in _GENERIC:
            families.append(rc["font.family"][0])
        legend_label = rc["legend.labelcolor"]
        title_color = rc["axes.titlecolor"]
        palette = [_hex(p["color"]) for p in rc["axes.prop_cycle"] if "color" in p]
        spec = {
            "fig_facecolor": _hex(rc["figure.facecolor"]),
            "axes_facecolor": _hex(rc["axes.facecolor"]),
            "spines": {s: bool(rc[f"axes.spines.{s}"]) for s in SPINES},
            "spine_lw": float(rc["axes.linewidth"]),
            "spine_color": _hex(rc["axes.edgecolor"]),
            "axisbelow": rc["axes.axisbelow"],
            "grid": bool(rc["axes.grid"]),
            "grid_axis": rc["axes.grid.axis"],
            "grid_which": rc["axes.grid.which"],
            "grid_style": {
                "color": _hex(rc["grid.color"]),
                "linewidth": float(rc["grid.linewidth"]),
                "alpha": float(rc["grid.alpha"]),
                "linestyle": rc["grid.linestyle"],
            },
            "title": {
                "size": pt("axes.titlesize"),
                "weight": rc["axes.titleweight"],
                "color": text_color if title_color == "auto" else _hex(title_color),
            },
            "label": {
                "size": pt("axes.labelsize"),
                "weight": rc["axes.labelweight"],
                "color": _hex(rc["axes.labelcolor"]),
            },
            "xtick": tick("x"),
            "ytick": tick("y"),
            "legend": {
                "frameon": bool(rc["legend.frameon"]),
                "fontsize": pt("legend.fontsize"),
                "facecolor": _hex(
                    rc["axes.facecolor"]
                    if rc["legend.facecolor"] == "inherit"
                    else rc["legend.facecolor"]
                ),
                "edgecolor": _hex(
                    rc["axes.edgecolor"]
                    if rc["legend.edgecolor"] == "inherit"
                    else rc["legend.edgecolor"]
                ),
                "textcolor": (
                    text_color
                    if legend_label in (None, "None", "none", "linecolor")
                    else _hex(legend_label)
                ),
            },
            "font": families,
            "palette": palette,
        }
    return spec


def _apply_axes(ax: Any, s: Dict[str, Any]) -> None:
    ax.set_facecolor(s["axes_facecolor"])
    for name, visible in s["spines"].items():
        spine = ax.spines[name]
        spine.set_visible(visible)
        spine.set_linewidth(s["spine_lw"])
        spine.set_edgecolor(s["spine_color"])
    ax.set_axisbelow(s["axisbelow"])
    ax.grid(False)
    if s["grid"]:
        ax.grid(True, which=s["grid_which"], axis=s["grid_axis"], **s["grid_style"])
    t, lab = s["title"], s["label"]
    ax.title.set_fontsize(t["size"])
    ax.title.set_fontweight(t["weight"])
    ax.title.set_color(t["color"])
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_fontsize(lab["size"])
        label.set_fontweight(lab["weight"])
        label.set_color(lab["color"])
    for axis in ("x", "y"):
        ax.tick_params(axis=axis, which="both", **s[f"{axis}tick"])
    legend = ax.get_legend()
    if legend is not None:
        lg = s["legend"]
        legend.set_frame_on(lg["frameon"])
        legend.get_frame().set_facecolor(lg["facecolor"])
        legend.get_frame().set_edgecolor(lg["edgecolor"])
        for text in legend.get_texts():
            text.set_fontsize(lg["fontsize"])
            text.set_color(lg["textcolor"])
    _apply_family(ax, s["font"])


def _code_axes(a: str, s: Dict[str, Any]) -> List[str]:
    out = [f"{a}.set_facecolor({lit(s['axes_facecolor'])})"]
    out += [
        f"for _name, _vis in {lit(s['spines'])}.items():",
        f"    {a}.spines[_name].set_visible(_vis)",
        f"    {a}.spines[_name].set_linewidth({lit(s['spine_lw'])})",
        f"    {a}.spines[_name].set_edgecolor({lit(s['spine_color'])})",
        f"{a}.set_axisbelow({lit(s['axisbelow'])})",
        f"{a}.grid(False)",
    ]
    if s["grid"]:
        style = ", ".join(f"{k}={lit(v)}" for k, v in s["grid_style"].items())
        out.append(
            f"{a}.grid(True, which={lit(s['grid_which'])}, "
            f"axis={lit(s['grid_axis'])}, {style})"
        )
    t, lab = s["title"], s["label"]
    out += [
        f"{a}.title.set_fontsize({lit(t['size'])})",
        f"{a}.title.set_fontweight({lit(t['weight'])})",
        f"{a}.title.set_color({lit(t['color'])})",
        f"for _lab in ({a}.xaxis.label, {a}.yaxis.label):",
        f"    _lab.set_fontsize({lit(lab['size'])})",
        f"    _lab.set_fontweight({lit(lab['weight'])})",
        f"    _lab.set_color({lit(lab['color'])})",
    ]
    for axis in ("x", "y"):
        kw = ", ".join(f"{k}={lit(v)}" for k, v in s[f"{axis}tick"].items())
        out.append(f"{a}.tick_params(axis={axis!r}, which='both', {kw})")
    lg = s["legend"]
    out += [
        f"if {a}.get_legend() is not None:",
        f"    _leg = {a}.get_legend()",
        f"    _leg.set_frame_on({lg['frameon']})",
        f"    _leg.get_frame().set_facecolor({lit(lg['facecolor'])})",
        f"    _leg.get_frame().set_edgecolor({lit(lg['edgecolor'])})",
        "    for _t in _leg.get_texts():",
        f"        _t.set_fontsize({lit(lg['fontsize'])})",
        f"        _t.set_color({lit(lg['textcolor'])})",
    ]
    out += _code_family(a, s["font"])
    return out


def _apply_theme(e: "FigureEditor", p: Dict[str, Any]) -> Dict[str, Any]:
    spec = theme_spec(p["theme"])
    e.fig.set_facecolor(spec["fig_facecolor"])
    per_axes = []
    for i, ax in enumerate(e.fig.get_axes()):
        _apply_axes(ax, spec)
        changes: List[Dict[str, Any]] = []
        if p["recolor"] and spec["palette"]:
            colors = spec["palette"]
            for g in range(len(e.color_groups(i))):
                changes += _recolor_group(e, i, g, colors[g % len(colors)])
        per_axes.append({"ax_index": i, "changes": changes})
    return {**p, "spec": spec, "per_axes": per_axes}


def _code_theme(p: Dict[str, Any], axv: AxVar) -> List[str]:
    spec = p["spec"]
    out = [
        f"# theme {p['theme']!r}",
        f"fig.set_facecolor({lit(spec['fig_facecolor'])})",
    ]
    for block in p["per_axes"]:
        a = axv(block["ax_index"])
        out += _code_axes(a, spec)
        out += _code_changes(a, block["changes"])
    return out


register(
    Op(
        "apply_theme",
        "Restyle the figure with a theme (StatsPAI, matplotlib or seaborn "
        "style). Background, spines, grid, fonts, text colours and the legend "
        "follow the theme; with recolor=True each series takes the theme's "
        "palette colour, its error bars and band included. Global matplotlib "
        "settings are not changed; use sp.set_theme for future figures.",
        (Param("theme", "text", required=True), Param("recolor", "bool", default=True)),
        _apply_theme,
        _code_theme,
        scope="all",
        slot=lambda p: ("theme",),
        restyles=lambda p: bool(p["recolor"]),
        legacy=lambda p: ("theme", "name"),
    )
)
