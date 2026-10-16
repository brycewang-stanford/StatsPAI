"""Layout tab of the Jupyter figure-editor panel: spines, grid, figure
size, backgrounds, legend, tick labels, axis limits, annotations and the
text already on the plot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._jupyter_tabs import _ERR, _FIG_SIZES, _LINE_STYLES, _if_chosen, _span, _to_hex

if TYPE_CHECKING:
    from ._jupyter_editor import EditorPanel


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

    # Existing text on the plot: reword annotations, show or hide any text.
    # Text drawn by the plotting function that reports a number keeps its
    # wording (the box is disabled for it).
    text_pick = p.dropdown("Text:", [("(no text on this axes)", -1)], -1)
    text_value = p.text("Wording:")
    text_visible = w.Checkbox(description="Visible")
    p.controls.update(
        text_pick=text_pick, text_value=text_value, text_visible=text_visible
    )

    def picked() -> Any:
        k = text_pick.value
        texts = ax().texts
        return texts[k] if isinstance(k, int) and 0 <= k < len(texts) else None

    def refresh_texts() -> None:
        options = [
            (f"{k}: {t.get_text()[:32] or '(empty)'}", k)
            for k, t in enumerate(ax().texts)
        ] or [("(no text on this axes)", -1)]
        values = [v for _, v in options]
        if list(text_pick.options) != options:
            current = text_pick.value
            text_pick.options = options
            text_pick.value = current if current in values else values[0]
        t = picked()
        text_value.disabled = t is None or not ed.is_editable(t)
        text_visible.disabled = t is None
        if t is not None:
            if text_value.value != t.get_text():
                text_value.value = t.get_text()
            text_visible.value = bool(t.get_visible())
        else:
            text_value.value = ""

    p.readers.append(refresh_texts)

    def on_pick(_: Any) -> None:
        if not p.syncing:
            p.sync()

    text_pick.observe(on_pick, names="value")

    def write_text(v: str) -> None:
        if picked() is not None:
            ed.set_text(int(text_pick.value), v, ax_index=p.ax_index)

    def write_visible(v: bool) -> None:
        if picked() is not None:
            ed.set_text_visible(int(text_pick.value), v, ax_index=p.ax_index)

    for widget, write in ((text_value, write_text), (text_visible, write_visible)):

        def on_change(change: Any, write: Any = write) -> None:
            if not p.syncing:
                p.edit(lambda: write(change["new"]))

        widget.observe(on_change, names="value")

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
            p.rule(),
            p.header("Text on the plot"),
            text_pick,
            text_value,
            text_visible,
        ]
    )
