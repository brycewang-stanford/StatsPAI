"""
Jupyter control panel for the figure editor (ipywidgets).

Layout: live preview on the left, tabbed controls on the right (Theme, Text,
Style, Layout, Export), like Stata's Graph Editor.

Every control is bound with a *reader* (current figure state -> widget value)
and a *writer* (widget value -> editor operation). After any change, undo,
redo, reset or axes switch the panel re-reads all controls, so the widgets
always show the figure as it is. Warnings raised by an edit (an axis limit
widened to keep data visible, an alpha raised to its floor) are shown in the
status line instead of being lost.
"""

from __future__ import annotations

import html
import io
import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ._jupyter_tabs import (  # noqa: F401  (helpers re-exported)
    _ERR,
    _OK,
    _WARN,
    _code_to_html,
    _layout_tab,
    _rebuild_style,
    _short,
    _span,
    _style_box,
    _text_tab,
    _theme_tab,
    _to_hex,
)

if TYPE_CHECKING:
    from .interactive import FigureEditor

logger = logging.getLogger(__name__)


class EditorPanel:
    """The widgets and their bindings. ``create_jupyter_panel`` builds one."""

    def __init__(self, editor: "FigureEditor", preview_dpi: int = 100) -> None:
        import ipywidgets as w

        self.w = w
        self.ed = editor
        self.fig = editor.fig
        self.axes = list(self.fig.get_axes())
        self.preview_dpi = preview_dpi
        self.syncing = False
        self.auto_render = True
        self.pending = 0
        self.readers: List[Callable[[], None]] = []
        self.controls: Dict[str, Any] = {}
        self.root: Any = None

        self.image = w.Image(
            value=self._png(),
            format="png",
            layout=w.Layout(max_width="100%", border="1px solid #eee"),
        )
        self.status = w.HTML(_span("Ready — edit with the controls on the right"))

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _png(self) -> bytes:
        buf = io.BytesIO()
        self.fig.savefig(
            buf,
            format="png",
            dpi=self.preview_dpi,
            bbox_inches="tight",
            facecolor=self.fig.get_facecolor(),
        )
        return buf.getvalue()

    @property
    def ax_index(self) -> int:
        sel = self.controls.get("axis")
        return int(sel.value) if sel is not None else 0

    @property
    def ax(self) -> Any:
        return self.axes[self.ax_index]

    def edit(self, fn: Callable[[], Any]) -> None:
        """Run one editor call, report warnings and errors, then re-sync."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fn()
            except Exception as exc:  # shown to the user, not swallowed
                logger.debug("Edit failed", exc_info=True)
                self.status.value = _span(f"Not applied: {exc}", _ERR)
                self.sync()
                return
        messages = [
            str(c.message) for c in caught if issubclass(c.category, UserWarning)
        ]
        if messages:
            self.status.value = _span(" | ".join(messages), _WARN)
        self.sync()

    def bind(
        self,
        name: str,
        widget: Any,
        read: Callable[..., Any],
        write: Callable[..., Any],
    ) -> Any:
        """Register a control: ``read()`` gives its value from the figure,
        ``write(value)`` applies a change."""
        self.controls[name] = widget

        def on_change(change: Any) -> None:
            if self.syncing:
                return
            self.edit(lambda: write(change["new"]))

        widget.observe(on_change, names="value")

        def reader() -> None:
            value = read()
            if value is not None and widget.value != value:
                widget.value = value

        self.readers.append(reader)
        return widget

    def sync(self) -> None:
        """Re-read every control from the figure."""
        self.syncing = True
        try:
            for reader in list(self.readers):
                try:
                    reader()
                except Exception:
                    logger.debug("Widget sync failed", exc_info=True)
        finally:
            self.syncing = False

    def on_refresh(self, _fig: Any) -> None:
        if self.auto_render:
            self.render()
        else:
            self.pending += 1
            self.controls["apply"].description = f"Apply ({self.pending})"
        self.update_code()

    def render(self) -> None:
        self.image.value = self._png()
        self.pending = 0
        if "apply" in self.controls:
            self.controls["apply"].description = "Apply"
        w_in, h_in = self.fig.get_size_inches()
        dpi = int(self.fig.dpi)
        self.status.value = _span(
            f"{len(self.ed.edits)} edit(s) — {w_in:.2f}×{h_in:.2f} in at "
            f"{dpi} DPI = {int(w_in * dpi)}×{int(h_in * dpi)} px",
            _OK,
        )

    def update_code(self) -> None:
        if "code" in self.controls:
            self.controls["code"].value = _code_to_html(self.ed.generate_code())
        if "history" in self.controls:
            self.controls["history"].value = self._history_html()

    def _history_html(self) -> str:
        if not self.ed.edits:
            return _span("No edits yet")
        rows = "".join(
            '<tr style="font-size:11px">'
            f'<td style="padding:2px 4px;color:#666">{i}</td>'
            f'<td style="padding:2px 4px">{html.escape(e.op or "")}</td>'
            f'<td style="padding:2px 4px;color:#2C3E50">'
            f"{html.escape(_short(e.params))}</td></tr>"
            for i, e in enumerate(self.ed.edits, 1)
        )
        return (
            '<div style="max-height:200px; overflow-y:auto; border:1px solid #eee">'
            '<table style="width:100%; border-collapse:collapse">'
            + rows
            + "</table></div>"
        )

    # ------------------------------------------------------------------
    # Small widget helpers
    # ------------------------------------------------------------------

    def text(self, desc: str, **kw: Any) -> Any:
        return self.w.Text(
            description=desc,
            continuous_update=False,
            layout=self.w.Layout(width="95%"),
            **kw,
        )

    def slider(
        self, desc: str, lo: float, hi: float, step: float, value: float = 0.0
    ) -> Any:
        return self.w.FloatSlider(
            value=min(max(value, lo), hi),
            min=lo,
            max=hi,
            step=step,
            description=desc,
            continuous_update=False,
            layout=self.w.Layout(width="95%"),
        )

    def dropdown(self, desc: str, options: Any, value: Any = None) -> Any:
        dd = self.w.Dropdown(
            options=options, description=desc, layout=self.w.Layout(width="95%")
        )
        if value is not None:
            dd.value = value
        return dd

    def color(self, desc: str, value: str = "#000000") -> Any:
        return self.w.ColorPicker(
            value=value,
            description=desc,
            concise=False,
            layout=self.w.Layout(width="70%"),
        )

    def header(self, title: str) -> Any:
        return self.w.HTML(f"<b>{html.escape(title)}</b>")

    def rule(self) -> Any:
        return self.w.HTML('<hr style="margin:4px 0">')


# ======================================================================
# Export tab and assembly
# ======================================================================


def _export_tab(p: EditorPanel) -> Any:
    ed, w = p.ed, p.w
    fmt = p.dropdown(
        "Format:",
        [("PDF (vector)", "pdf"), ("PNG", "png"), ("SVG (vector)", "svg")],
        "pdf",
    )
    fname = w.Text(
        value="figure.pdf", description="File:", layout=w.Layout(width="70%")
    )
    dpi = w.IntSlider(
        value=300,
        min=50,
        max=600,
        step=10,
        description="DPI:",
        layout=w.Layout(width="95%"),
    )
    embed = w.Checkbox(
        value=True, description="Embed TrueType fonts (journals reject Type 3)"
    )
    save = w.Button(description="Save", button_style="success", icon="download")

    def on_fmt(change: Any) -> None:
        base = fname.value.rsplit(".", 1)[0] if "." in fname.value else fname.value
        fname.value = f"{base}.{change['new']}"

    fmt.observe(on_fmt, names="value")

    def on_save(_: Any) -> None:
        try:
            ed.save(fname.value, dpi=dpi.value, fmt=fmt.value, embed_fonts=embed.value)
            p.status.value = _span(f"Saved {fname.value}", _OK)
        except Exception as exc:
            p.status.value = _span(f"Save failed: {exc}", _ERR)

    save.on_click(on_save)

    check_btn = w.Button(description="Check figure", icon="check")
    check_out = w.HTML("")

    def on_check(_: Any) -> None:
        findings = ed.check()
        if not findings:
            check_out.value = _span("No problems found.", _OK)
        else:
            check_out.value = "<br>".join(
                _span("• " + f["message"], _WARN) for f in findings
            )

    check_btn.on_click(on_check)

    code = w.HTML(_code_to_html(ed.generate_code()), layout=w.Layout(width="95%"))
    history = w.HTML(_span("No edits yet"))
    undo = w.Button(description="Undo", button_style="warning", icon="undo")
    redo = w.Button(description="Redo", button_style="info", icon="repeat")
    reset = w.Button(description="Reset all", button_style="danger", icon="refresh")
    undo.on_click(lambda _: p.edit(ed.undo))
    redo.on_click(lambda _: p.edit(ed.redo))
    reset.on_click(lambda _: p.edit(ed.reset))

    style_json = w.Textarea(
        description="Style:",
        placeholder="paste a saved style (JSON) here",
        layout=w.Layout(width="95%", height="80px"),
    )
    copy_style = w.Button(description="Show style JSON", icon="copy")
    apply_style = w.Button(description="Apply style JSON", icon="paint-brush")

    def on_copy(_: Any) -> None:
        style_json.value = ed.to_edits_json()

    def on_apply(_: Any) -> None:
        text = style_json.value.strip()
        if text:
            p.edit(lambda: ed.apply_edits_json(text))

    copy_style.on_click(on_copy)
    apply_style.on_click(on_apply)
    history_box = w.Accordion(children=[history])
    history_box.set_title(0, "Edit history")
    history_box.selected_index = None
    p.controls.update(
        fmt=fmt,
        fname=fname,
        dpi=dpi,
        embed=embed,
        save=save,
        check=check_btn,
        check_out=check_out,
        code=code,
        history=history,
        undo=undo,
        redo=redo,
        reset=reset,
        style_json=style_json,
        copy_style=copy_style,
        apply_style=apply_style,
    )
    return w.VBox(
        [
            p.header("Export"),
            fmt,
            w.HBox([fname, save]),
            dpi,
            embed,
            p.rule(),
            p.header("Pre-submission check"),
            check_btn,
            check_out,
            p.rule(),
            p.header("Reproducible code"),
            w.HTML(_span("Run after the code that draws `fig`. Select to copy.")),
            code,
            w.HBox([undo, redo, reset]),
            p.rule(),
            p.header("Saved style (apply to other figures)"),
            w.HBox([copy_style, apply_style]),
            style_json,
            history_box,
        ]
    )


# ======================================================================
# Assembly
# ======================================================================


def create_jupyter_panel(
    editor: "FigureEditor", show: bool = True
) -> Optional[EditorPanel]:
    """
    Build the ipywidgets panel for ``editor`` and display it.

    Returns the :class:`EditorPanel` (its ``controls`` dict holds every
    widget), or ``None`` if ipywidgets is missing or the figure has no axes.
    """
    try:
        import ipywidgets as w
        from IPython.display import display
    except ImportError:
        warnings.warn(
            "ipywidgets is not installed (pip install ipywidgets); use the "
            "editor's methods instead, e.g. editor.set_title('New title').",
            UserWarning,
            stacklevel=2,
        )
        return None
    import matplotlib.pyplot as plt

    if not editor.fig.get_axes():
        warnings.warn("The figure has no axes to edit.", UserWarning, stacklevel=2)
        return None

    p = EditorPanel(editor)
    tabs_spec = [
        ("Theme", _theme_tab),
        ("Text", _text_tab),
        ("Style", _style_box),
        ("Layout", _layout_tab),
        ("Export", _export_tab),
    ]
    children = [build(p) for _, build in tabs_spec]
    _rebuild_style(p)
    tabs = w.Tab(children=children)
    for k, (title, _) in enumerate(tabs_spec):
        tabs.set_title(k, title)
    tabs.selected_index = 1
    p.controls["tabs"] = tabs

    mode = w.Button(
        description="Auto",
        button_style="success",
        icon="refresh",
        tooltip="Toggle automatic preview refresh",
        layout=w.Layout(width="76px", height="26px"),
    )
    apply = w.Button(
        description="Apply",
        button_style="warning",
        icon="refresh",
        layout=w.Layout(width="auto", height="26px", display="none"),
    )
    p.controls.update(mode=mode, apply=apply)

    def toggle(_: Any) -> None:
        p.auto_render = not p.auto_render
        mode.description = "Auto" if p.auto_render else "Manual"
        mode.button_style = "success" if p.auto_render else ""
        apply.layout.display = "none" if p.auto_render else ""
        if p.auto_render:
            p.render()

    mode.on_click(toggle)
    apply.on_click(lambda _: p.render())

    editor.on_refresh(p.on_refresh)
    p.sync()

    header = w.HBox(
        [
            w.HTML(
                '<h3 style="margin:0; color:#2C3E50; white-space:nowrap">'
                "StatsPAI Figure Editor</h3>"
            ),
            mode,
        ],
        layout=w.Layout(
            justify_content="space-between", align_items="center", width="100%"
        ),
    )
    note = w.HTML(
        _span("Data locked | left: live preview | right: controls", _ERR)
        if editor.protect_data
        else _span("Data protection OFF", _ERR)
    )
    panel = w.VBox(
        [
            header,
            note,
            w.VBox([tabs], layout=w.Layout(flex="1 1 auto", overflow="auto")),
            w.HBox([apply], layout=w.Layout(justify_content="flex-end", width="100%")),
        ],
        layout=w.Layout(
            display="flex",
            flex_flow="column",
            width="420px",
            min_width="360px",
            padding="8px",
            border="1px solid #ddd",
            height="680px",
        ),
    )
    preview = w.VBox(
        [p.image, p.status], layout=w.Layout(flex="1 1 auto", min_width="400px")
    )
    p.root = w.HBox(
        [preview, panel], layout=w.Layout(align_items="flex-start", width="100%")
    )

    # The preview widget shows the figure; drop the inline copy.
    plt.close(editor.fig)
    if show:
        display(p.root)
    return p
