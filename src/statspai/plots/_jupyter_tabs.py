"""Tabs of the Jupyter figure-editor panel (Theme, Text, Style, Layout)."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from ._jupyter_editor import EditorPanel

_LINE_STYLES = [
    ("solid", "-"),
    ("dashed", "--"),
    ("dashdot", "-."),
    ("dotted", ":"),
    ("none", "None"),
]
_MARKERS = [
    ("none", "None"),
    ("circle", "o"),
    ("square", "s"),
    ("triangle", "^"),
    ("diamond", "D"),
    ("tri down", "v"),
    ("star", "*"),
    ("pentagon", "p"),
    ("hexagon", "h"),
    ("point", "."),
]
_FIG_SIZES = {
    "": ("Custom", None),
    "journal_col": ("Journal column (3.5×2.6)", (3.5, 2.6)),
    "journal_full": ("Journal full width (7×4.5)", (7.0, 4.5)),
    "aea": ("AER/AEJ (6.5×4.5)", (6.5, 4.5)),
    "slide_16_9": ("Slide 16:9 (12×6.75)", (12.0, 6.75)),
    "slide_4_3": ("Slide 4:3 (10×7.5)", (10.0, 7.5)),
    "square": ("Square (6×6)", (6.0, 6.0)),
    "poster": ("Poster (14×10)", (14.0, 10.0)),
    "a4_half": ("A4 half (8.27×5.83)", (8.27, 5.83)),
}
_OK = "#2ECC71"
_WARN = "#E67E22"
_ERR = "#E74C3C"


def _to_hex(color: Any) -> str:
    """Convert any matplotlib color to hex string."""
    import matplotlib.colors as mcolors

    try:
        return str(mcolors.to_hex(color, keep_alpha=False))
    except (ValueError, TypeError):
        return "#000000"


def _code_to_html(code: str) -> str:
    """Render code as a selectable <pre> block that supports native copy."""
    return (
        '<pre style="background:#f8f8f8; border:1px solid #ddd; border-radius:4px; '
        "padding:10px; font-family:Menlo,Monaco,Consolas,monospace; font-size:12px; "
        "line-height:1.5; white-space:pre-wrap; word-wrap:break-word; "
        "max-height:300px; overflow:auto; user-select:text; margin:0;"
        f'">{html.escape(code)}</pre>'
    )


def _span(msg: str, color: str = "#999") -> str:
    return f'<span style="font-size:11px; color:{color}">{html.escape(msg)}</span>'


def _if_chosen(value: Any, action: Callable[[], None]) -> None:
    """Run a preset/chooser action unless the placeholder ("") is selected."""
    if value:
        action()


def _short(params: Any) -> str:
    text = repr({k: v for k, v in (params or {}).items() if v not in (None, {}, [])})
    return text if len(text) < 70 else text[:67] + "..."


def _theme_tab(p: EditorPanel) -> Any:
    from ._editor_ops import all_palettes
    from .themes import list_themes

    themes = list_themes()
    options = [("-- choose a theme --", "")]
    for group in ("statspai", "matplotlib", "seaborn"):
        options += [(f"[{group}] {t}", t) for t in themes.get(group, [])]
    recolor = p.w.Checkbox(
        value=True, description="Recolour series with the theme palette"
    )
    p.controls["recolor"] = recolor

    def write_theme(name: str) -> None:
        if name:
            p.ed.apply_theme(name, recolor=recolor.value)

    p.bind("theme", p.dropdown("Theme:", options, ""), lambda: "", write_theme)

    palettes = all_palettes()
    palette = p.dropdown(
        "Palette:", [("-- choose a palette --", "")] + [(k, k) for k in palettes], ""
    )
    palette_info = p.w.HTML(
        _span("One colour per series; points, error bars and bands move together.")
    )

    def write_palette(name: str) -> None:
        if name:
            p.ed.apply_palette(name)

    p.bind("palette", palette, lambda: "", write_palette)
    return p.w.VBox(
        [
            p.header("Theme"),
            p.controls["theme"],
            recolor,
            p.w.HTML(
                _span(
                    "Restyles this figure only; sp.set_theme() changes figures "
                    "drawn later."
                )
            ),
            p.rule(),
            p.header("Palette (all axes)"),
            palette,
            palette_info,
        ]
    )


def _text_tab(p: EditorPanel) -> Any:
    from ._editor_presets import (
        FONT_PRESETS,
        SIZE_PRESETS,
        _resolve_preset_fonts,
        get_font_choices,
    )

    ed = p.ed
    children: List[Any] = [p.header("Text & labels")]
    if len(p.axes) > 1:
        axis = p.dropdown(
            "Axes:",
            [(f"Axes {i}" if i else "Axes 0 (main)", i) for i in range(len(p.axes))],
            0,
        )
        p.controls["axis"] = axis

        def on_axis(_: Any) -> None:
            p.sync()
            _rebuild_style(p)

        axis.observe(on_axis, names="value")
        children.append(axis)

    def ax() -> Any:
        return p.ax

    p.bind(
        "title",
        p.text("Title:"),
        lambda: ax().get_title(),
        lambda v: ed.set_title(v, ax_index=p.ax_index),
    )
    p.bind(
        "title_size",
        p.slider("Title size:", 5, 36, 0.5),
        lambda: float(ax().title.get_fontsize()),
        lambda v: ed.set_fontsize("title", v, ax_index=p.ax_index),
    )
    p.bind(
        "title_weight",
        p.dropdown("Weight:", [("Normal", "normal"), ("Bold", "bold")]),
        lambda: (
            "bold" if str(ax().title.get_fontweight()) in ("bold", "700") else "normal"
        ),
        lambda v: ed.set_title_weight(v, ax_index=p.ax_index),
    )
    p.bind(
        "xlabel",
        p.text("X label:"),
        lambda: ax().get_xlabel(),
        lambda v: ed.set_xlabel(v, ax_index=p.ax_index),
    )
    p.bind(
        "xlabel_size",
        p.slider("X size:", 5, 30, 0.5),
        lambda: float(ax().xaxis.label.get_fontsize()),
        lambda v: ed.set_fontsize("xlabel", v, ax_index=p.ax_index),
    )
    p.bind(
        "ylabel",
        p.text("Y label:"),
        lambda: ax().get_ylabel(),
        lambda v: ed.set_ylabel(v, ax_index=p.ax_index),
    )
    p.bind(
        "ylabel_size",
        p.slider("Y size:", 5, 30, 0.5),
        lambda: float(ax().yaxis.label.get_fontsize()),
        lambda v: ed.set_fontsize("ylabel", v, ax_index=p.ax_index),
    )

    def tick_size() -> float:
        labels = ax().get_xticklabels() or ax().get_yticklabels()
        return float(labels[0].get_fontsize()) if labels else 10.0

    p.bind(
        "tick_size",
        p.slider("Tick size:", 4, 24, 0.5),
        tick_size,
        lambda v: ed.set_fontsize("ticks", v, ax_index=p.ax_index),
    )

    def suptitle() -> str:
        sup = getattr(p.fig, "_suptitle", None)
        return sup.get_text() if sup is not None else ""

    p.bind(
        "suptitle",
        p.text("Suptitle:", placeholder="figure title (optional)"),
        suptitle,
        lambda v: ed.set_suptitle(v),
    )

    font_options = [("-- choose font preset --", "")]
    for name, preset in FONT_PRESETS.items():
        fonts = _resolve_preset_fonts(preset)
        venue = preset.get("venue", "")
        font_options.append((f"{fonts[0]}  ({venue})" if venue else fonts[0], name))
    p.bind(
        "font_preset",
        p.dropdown("Font:", font_options, ""),
        lambda: "",
        lambda v: _if_chosen(v, lambda: ed.apply_font_preset(v, ax_index=p.ax_index)),
    )
    size_options = [("-- choose size preset --", "")] + [
        (
            f"{k}  (title {v['title_size']}, label {v['label_size']}, "
            f"tick {v['tick_size']})",
            k,
        )
        for k, v in SIZE_PRESETS.items()
    ]
    p.bind(
        "size_preset",
        p.dropdown("Sizes:", size_options, ""),
        lambda: "",
        lambda v: _if_chosen(v, lambda: ed.apply_size_preset(v, ax_index=p.ax_index)),
    )

    def font_names(family: str) -> List[Tuple[str, str]]:
        choices = get_font_choices()
        if family == "serif":
            fonts = choices["English Serif"] + choices["CJK"]
        elif family == "sans-serif":
            fonts = choices["English Sans-serif"] + choices["CJK"]
        else:
            fonts = choices["Monospace"]
        return [("(family default)", "")] + [
            (f, f) for f in fonts if not f.startswith("(")
        ]

    family = p.dropdown(
        "Family:",
        [("Serif", "serif"), ("Sans-serif", "sans-serif"), ("Monospace", "monospace")],
        "sans-serif",
    )
    name = p.dropdown("Font name:", font_names("sans-serif"), "")
    p.controls["font_family"], p.controls["font_name"] = family, name

    def apply_font(_: Any = None) -> None:
        if p.syncing:
            return
        p.edit(
            lambda: ed.set_font(family.value, name.value or None, ax_index=p.ax_index)
        )

    def on_family(change: Any) -> None:
        if p.syncing:
            return
        p.syncing = True
        try:
            name.options = font_names(change["new"])
            name.value = ""
        finally:
            p.syncing = False
        apply_font()

    family.observe(on_family, names="value")
    name.observe(apply_font, names="value")

    c = p.controls
    children += [
        c["title"],
        c["title_size"],
        c["title_weight"],
        p.rule(),
        c["xlabel"],
        c["xlabel_size"],
        c["ylabel"],
        c["ylabel_size"],
        c["tick_size"],
        p.rule(),
        c["suptitle"],
        p.rule(),
        p.header("Fonts"),
        c["font_preset"],
        c["size_preset"],
        p.w.HTML(_span("Or pick a family and font:")),
        family,
        name,
    ]
    return p.w.VBox(children)


def _style_box(p: EditorPanel) -> Any:
    box = p.w.VBox([p.w.HTML(_span("Loading..."))])
    p.controls["style_box"] = box
    return box


def _rebuild_style(p: EditorPanel) -> None:
    """Build the per-series / per-line controls for the selected axes."""
    import matplotlib.collections as mcoll

    box = p.controls.get("style_box")
    if box is None:
        return
    # Drop readers of the previous build.
    p.readers = [r for r in p.readers if not getattr(r, "_style", False)]
    ed, ax, i = p.ed, p.ax, p.ax_index
    children: List[Any] = [p.header("Series colours")]

    def style_bind(
        name: str, widget: Any, read: Callable[..., Any], write: Callable[..., Any]
    ) -> Any:
        p.bind(name, widget, read, write)
        p.readers[-1]._style = True  # type: ignore[attr-defined]
        return widget

    groups = ed.color_groups(i)
    if not groups:
        children.append(p.w.HTML(_span("No single-coloured series on this axes.")))
    for g, grp in enumerate(groups):
        members = ", ".join(f"{k}{j}" for k, j in grp["members"])

        def read_group(grp: Dict[str, Any] = grp) -> str:
            from ._editor_roles import single_color

            kind, j = grp["members"][0]
            seq = {"line": ax.get_lines(), "coll": ax.collections, "patch": ax.patches}[
                kind
            ]
            return single_color(seq[j]) or grp["color"]

        widget = p.color(f"Series {g}:", read_group())
        widget.tooltip = members
        style_bind(
            f"group{g}",
            widget,
            read_group,
            lambda v, g=g: ed.set_series_color(g, v, ax_index=i),
        )
        children.append(widget)

    lines = ax.get_lines()[:20]
    for j, line in enumerate(lines):
        label = line.get_label() or ""
        label = (
            f"Line {j}" if not label or label.startswith("_") else f"Line {j}: {label}"
        )
        role = ed.artist_roles.get(id(line))
        children += [
            p.rule(),
            p.w.HTML(f"<i>{html.escape(label)}</i> {_span(role.name if role else '')}"),
        ]
        children.append(
            style_bind(
                f"line{j}_color",
                p.color("Colour:", _to_hex(line.get_color())),
                lambda line=line: _to_hex(line.get_color()),
                lambda v, j=j: ed.set_color(f"line{j}", v, ax_index=i),
            )
        )
        children.append(
            style_bind(
                f"line{j}_width",
                p.slider("Width:", 0, 8, 0.25, line.get_linewidth()),
                lambda line=line: float(line.get_linewidth()),
                lambda v, j=j: ed.set_linewidth(j, v, ax_index=i),
            )
        )
        ls = line.get_linestyle()
        children.append(
            style_bind(
                f"line{j}_style",
                p.dropdown(
                    "Style:",
                    _LINE_STYLES,
                    ls if ls in dict(_LINE_STYLES).values() else None,
                ),
                lambda line=line: (
                    line.get_linestyle()
                    if line.get_linestyle() in dict(_LINE_STYLES).values()
                    else None
                ),
                lambda v, j=j: ed.set_linestyle(j, v, ax_index=i),
            )
        )
        mk = str(line.get_marker())
        children.append(
            style_bind(
                f"line{j}_marker",
                p.dropdown(
                    "Marker:", _MARKERS, mk if mk in dict(_MARKERS).values() else None
                ),
                lambda line=line: (
                    str(line.get_marker())
                    if str(line.get_marker()) in dict(_MARKERS).values()
                    else None
                ),
                lambda v, j=j: ed.set_marker(j, v, ax_index=i),
            )
        )
        children.append(
            style_bind(
                f"line{j}_msize",
                p.slider("Marker size:", 0, 20, 0.5, line.get_markersize()),
                lambda line=line: float(line.get_markersize()),
                lambda v, j=j: ed.set_markersize(j, v, ax_index=i),
            )
        )
        alpha = line.get_alpha()
        children.append(
            style_bind(
                f"line{j}_alpha",
                p.slider("Alpha:", 0, 1, 0.05, 1.0 if alpha is None else alpha),
                lambda line=line: (
                    1.0 if line.get_alpha() is None else float(line.get_alpha())
                ),
                lambda v, j=j: ed.set_alpha(f"line{j}", v, ax_index=i),
            )
        )
    if len(ax.get_lines()) > 20:
        children.append(
            p.w.HTML(
                _span(
                    f"Showing 20 of {len(ax.get_lines())} lines; "
                    "use the API for the rest."
                )
            )
        )

    for j, coll in enumerate(ax.collections[:20]):
        role = ed.artist_roles.get(id(coll))
        kind = "scatter" if isinstance(coll, mcoll.PathCollection) else "coll"
        name = {"scatter": "Scatter", "coll": "Band / bars"}[kind]
        children += [
            p.rule(),
            p.w.HTML(f"<i>{name} {j}</i> {_span(role.name if role else '')}"),
        ]
        if kind == "scatter" and coll.get_array() is None:
            fc = coll.get_facecolor()
            children.append(
                style_bind(
                    f"coll{j}_color",
                    p.color("Colour:", _to_hex(fc[0] if len(fc) else "black")),
                    lambda coll=coll: (
                        _to_hex(coll.get_facecolor()[0])
                        if len(coll.get_facecolor())
                        else None
                    ),
                    lambda v, j=j: ed.set_scatter_color(j, v, ax_index=i),
                )
            )
        alpha = coll.get_alpha()
        children.append(
            style_bind(
                f"coll{j}_alpha",
                p.slider("Alpha:", 0, 1, 0.05, 1.0 if alpha is None else alpha),
                lambda coll=coll: (
                    1.0 if coll.get_alpha() is None else float(coll.get_alpha())
                ),
                lambda v, j=j: ed.set_alpha(f"coll{j}", v, ax_index=i),
            )
        )
    box.children = children
