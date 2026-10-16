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


def _layout_tab(p: EditorPanel) -> Any:
    ed, w = p.ed, p.w

    def ax() -> Any:
        return p.ax

    spines = []
    for s in ("top", "right", "bottom", "left"):
        spines.append(
            p.bind(
                f"spine_{s}",
                w.Checkbox(description=f"{s.title()} spine"),
                lambda s=s: bool(ax().spines[s].get_visible()),
                lambda v, s=s: ed.set_spine_visible(s, v, ax_index=p.ax_index),
            )
        )

    def grid_on() -> bool:
        lines = ax().xaxis.get_gridlines()
        return bool(lines[0].get_visible()) if lines else False

    def grid_attr(attr: str) -> Any:
        lines = ax().xaxis.get_gridlines()
        if not lines:
            return None
        value = getattr(lines[0], f"get_{attr}")()
        if attr == "color":
            return _to_hex(value)
        if attr == "alpha":
            return 1.0 if value is None else float(value)
        return value if value in ("-", "--", ":", "-.") else None

    p.bind(
        "grid",
        w.Checkbox(description="Show grid"),
        grid_on,
        lambda v: ed.set_grid(v, ax_index=p.ax_index),
    )
    p.bind(
        "grid_color",
        p.color("Grid colour:", "#b0b0b0"),
        lambda: grid_attr("color"),
        lambda v: ed.set_grid_style(color=v, ax_index=p.ax_index),
    )
    p.bind(
        "grid_alpha",
        p.slider("Grid alpha:", 0, 1, 0.05, 1.0),
        lambda: grid_attr("alpha"),
        lambda v: ed.set_grid_style(alpha=v, ax_index=p.ax_index),
    )
    p.bind(
        "grid_style",
        p.dropdown("Grid style:", [(n, c) for n, c in _LINE_STYLES[:4]], "-"),
        lambda: grid_attr("linestyle"),
        lambda v: ed.set_grid_style(linestyle=v, ax_index=p.ax_index),
    )

    p.bind(
        "figsize_preset",
        p.dropdown(
            "Size preset:", [(lab, k) for k, (lab, _) in _FIG_SIZES.items()], ""
        ),
        lambda: "",
        lambda k: _if_chosen(k, lambda: ed.set_figsize(*_FIG_SIZES[k][1] or ())),
    )
    p.bind(
        "fig_width",
        p.slider("Width (in):", 1.5, 20, 0.05, p.fig.get_size_inches()[0]),
        lambda: float(p.fig.get_size_inches()[0]),
        lambda v: ed.set_figsize(v, float(p.fig.get_size_inches()[1])),
    )
    p.bind(
        "fig_height",
        p.slider("Height (in):", 1.0, 16, 0.05, p.fig.get_size_inches()[1]),
        lambda: float(p.fig.get_size_inches()[1]),
        lambda v: ed.set_figsize(float(p.fig.get_size_inches()[0]), v),
    )
    p.bind(
        "fig_dpi",
        w.IntSlider(
            value=int(p.fig.dpi),
            min=50,
            max=600,
            step=10,
            description="DPI:",
            continuous_update=False,
            layout=w.Layout(width="95%"),
        ),
        lambda: int(p.fig.dpi),
        lambda v: ed.set_dpi(int(v)),
    )

    p.bind(
        "fig_bg",
        p.color("Figure bg:"),
        lambda: _to_hex(p.fig.get_facecolor()),
        lambda v: ed.set_background_color(v, target="figure"),
    )
    p.bind(
        "ax_bg",
        p.color("Axes bg:"),
        lambda: _to_hex(ax().get_facecolor()),
        lambda v: ed.set_background_color(v, target="axes", ax_index=p.ax_index),
    )

    from ._editor_ops import LEGEND_LOCS
    from ._editor_ops_layout import _loc_name

    def legend_attr(attr: str) -> Any:
        leg = ax().get_legend()
        if leg is None:
            return None
        if attr == "visible":
            return bool(leg.get_visible())
        if attr == "loc":
            loc = _loc_name(leg)
            return loc if loc in LEGEND_LOCS else None
        if attr == "frame":
            return bool(leg.get_frame_on())
        texts = leg.get_texts()
        return float(texts[0].get_fontsize()) if texts else None

    p.bind(
        "legend_visible",
        w.Checkbox(description="Show legend"),
        lambda: legend_attr("visible") if ax().get_legend() is not None else False,
        lambda v: ed.set_legend_visible(v, ax_index=p.ax_index),
    )
    p.bind(
        "legend_loc",
        p.dropdown("Position:", list(LEGEND_LOCS), "best"),
        lambda: legend_attr("loc"),
        lambda v: ed.run("set_legend", loc=v, ax_index=p.ax_index),
    )
    p.bind(
        "legend_size",
        p.slider("Entry size:", 4, 24, 0.5, 10),
        lambda: legend_attr("size"),
        lambda v: ed.run("set_legend", fontsize=v, ax_index=p.ax_index),
    )
    p.bind(
        "legend_frame",
        w.Checkbox(description="Frame"),
        lambda: legend_attr("frame"),
        lambda v: ed.run("set_legend", frameon=v, ax_index=p.ax_index),
    )

    def labels_now() -> str:
        leg = ax().get_legend()
        return (
            "\n".join(t.get_text() for t in leg.get_texts()) if leg is not None else ""
        )

    p.bind(
        "legend_labels",
        w.Textarea(
            description="Entries:",
            continuous_update=False,
            placeholder="one label per line",
            layout=w.Layout(width="95%", height="70px"),
        ),
        labels_now,
        lambda v: ed.set_legend_labels(v.split("\n"), ax_index=p.ax_index),
    )

    def rotation(axis: str) -> float:
        labels = ax().get_xticklabels() if axis == "x" else ax().get_yticklabels()
        return float(labels[0].get_rotation()) if labels else 0.0

    p.bind(
        "xrot",
        p.slider("X tick rot:", 0, 90, 5),
        lambda: rotation("x"),
        lambda v: ed.set_tick_rotation("x", v, ax_index=p.ax_index),
    )
    p.bind(
        "yrot",
        p.slider("Y tick rot:", 0, 90, 5),
        lambda: rotation("y"),
        lambda v: ed.set_tick_rotation("y", v, ax_index=p.ax_index),
    )

    def lim_slider(axis: str) -> Any:
        lo, hi = sorted(getattr(ax(), f"get_{axis}lim")())
        span = max(hi - lo, 1e-9)
        return w.FloatRangeSlider(
            value=[lo, hi],
            min=lo - span,
            max=hi + span,
            step=span / 100,
            description=f"{axis.upper()} range:",
            continuous_update=False,
            readout_format=".3g",
            layout=w.Layout(width="95%"),
        )

    def read_lim(axis: str) -> Any:
        slider = p.controls[f"{axis}lim"]
        lo, hi = sorted(getattr(ax(), f"get_{axis}lim")())
        span = max(hi - lo, 1e-9)
        slider.min, slider.max = min(slider.min, lo - span), max(slider.max, hi + span)
        slider.step = span / 100
        return (lo, hi)

    def write_lim(axis: str, v: Any) -> None:
        current = getattr(ax(), f"get_{axis}lim")()
        lo, hi = (v[1], v[0]) if current[0] > current[1] else (v[0], v[1])
        getattr(ed, f"set_{axis}lim")(lo, hi, ax_index=p.ax_index)

    for axis in ("x", "y"):
        p.bind(
            f"{axis}lim",
            lim_slider(axis),
            lambda axis=axis: read_lim(axis),
            lambda v, axis=axis: write_lim(axis, v),
        )

    annot_text = p.text("Text:", placeholder="annotation text")
    annot_x = w.FloatText(value=0.0, description="X:", layout=w.Layout(width="47%"))
    annot_y = w.FloatText(value=0.0, description="Y:", layout=w.Layout(width="47%"))
    annot_btn = w.Button(
        description="Add annotation", button_style="primary", icon="pencil"
    )
    p.controls.update(
        annot_text=annot_text, annot_x=annot_x, annot_y=annot_y, annot_btn=annot_btn
    )

    def add_annotation(_: Any) -> None:
        if not annot_text.value:
            p.status.value = _span("Type the annotation text first", _ERR)
            return
        p.edit(
            lambda: ed.add_annotation(
                annot_text.value,
                (annot_x.value, annot_y.value),
                ax_index=p.ax_index,
                fontsize=10,
                ha="center",
            )
        )
        annot_text.value = ""

    annot_btn.on_click(add_annotation)

    tight = w.Button(description="Tight layout", button_style="info", icon="compress")
    tight.on_click(lambda _: p.edit(ed.tight_layout))
    p.controls["tight"] = tight

    data_note = (
        w.HTML(
            _span(
                "Data locked: limits widen to keep every point, error bar "
                "and band visible.",
                _ERR,
            )
        )
        if ed.protect_data
        else w.HTML("")
    )
    c = p.controls
    return w.VBox(
        [
            p.header("Spines"),
            w.HBox(spines[:2]),
            w.HBox(spines[2:]),
            p.rule(),
            p.header("Grid"),
            c["grid"],
            c["grid_color"],
            c["grid_alpha"],
            c["grid_style"],
            p.rule(),
            p.header("Figure size"),
            c["figsize_preset"],
            c["fig_width"],
            c["fig_height"],
            c["fig_dpi"],
            p.rule(),
            p.header("Background"),
            c["fig_bg"],
            c["ax_bg"],
            p.rule(),
            p.header("Legend"),
            c["legend_visible"],
            c["legend_loc"],
            c["legend_size"],
            c["legend_frame"],
            c["legend_labels"],
            p.rule(),
            p.header("Tick labels"),
            c["xrot"],
            c["yrot"],
            p.rule(),
            p.header("Axis limits"),
            data_note,
            c["xlim"],
            c["ylim"],
            p.rule(),
            p.header("Annotation"),
            annot_text,
            w.HBox([annot_x, annot_y]),
            w.HBox([annot_btn, tight]),
        ]
    )
