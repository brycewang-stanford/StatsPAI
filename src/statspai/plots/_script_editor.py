"""
Script-mode editor: a small control window next to a matplotlib GUI figure.

The controls live in their own figure, so the figure being edited keeps its
layout and saves without buttons in it. Click a title, axis label or
annotation in the figure to select it, type the new text in the control
window and press Enter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .interactive import FigureEditor

_THEME_CYCLE = ("academic", "aea", "minimal", "ggplot")


def create_script_editor(editor: "FigureEditor", show: bool = True) -> Any:
    """
    Open a control window for ``editor.fig``.

    Controls: text box (edits the selected text), Undo, Redo, Reset, Grid,
    Spines (toggle top/right), Theme (cycles through a few themes) and Code
    (prints the reproducible code). The code is also printed when the figure
    window closes.

    Parameters
    ----------
    editor : FigureEditor
        The editor wrapping the target figure.
    show : bool, default True
        Call ``plt.show()``. Tests pass False.

    Returns
    -------
    matplotlib.figure.Figure
        The control figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    fig = editor.fig
    plot_axes = list(fig.get_axes())
    state: Dict[str, Any] = {
        "selected": None,
        "loading": False,
        "grid": False,
        "minimal": False,
        "theme": -1,
    }

    def status(msg: str) -> None:
        info.set_text(msg)
        controls.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Click-to-select text in the edited figure (pickers do not change how
    # the figure draws or saves).
    # ------------------------------------------------------------------
    for a in plot_axes:
        for t in (a.title, a.xaxis.label, a.yaxis.label, *a.texts):
            t.set_picker(True)

    def locate(artist: Any) -> Optional[tuple]:
        for i, a in enumerate(plot_axes):
            if artist is a.title:
                return ("title", i)
            if artist is a.xaxis.label:
                return ("xlabel", i)
            if artist is a.yaxis.label:
                return ("ylabel", i)
            for k, t in enumerate(a.texts):
                if artist is t:
                    return ("text", i, k)
        return None

    def on_pick(event: Any) -> None:
        where = locate(event.artist)
        if where is None:
            return
        if where[0] == "text" and not editor.is_editable(event.artist):
            status("That label reports a result; its wording is locked.")
            return
        state["selected"] = where
        # TextBox.set_val fires the submit callback; loading the current
        # text into the box is not an edit.
        state["loading"] = True
        try:
            textbox.set_val(event.artist.get_text())
        finally:
            state["loading"] = False
        status(f"Selected {where[0]} — edit below, press Enter")

    fig.canvas.mpl_connect("pick_event", on_pick)

    # ------------------------------------------------------------------
    # Control window
    # ------------------------------------------------------------------
    controls = plt.figure("StatsPAI figure editor", figsize=(7.5, 1.6))
    textbox = TextBox(controls.add_axes((0.12, 0.62, 0.84, 0.25)), "Text ", initial="")
    info_ax = controls.add_axes((0.02, 0.02, 0.96, 0.14))
    info_ax.set_axis_off()
    info = info_ax.text(
        0,
        0.5,
        "Click a title, label or annotation in the figure to edit it.",
        fontsize=8,
        color="#555",
        va="center",
    )

    def on_submit(text: str) -> None:
        where = state["selected"]
        if where is None or state["loading"]:
            return
        try:
            if where[0] == "text":
                editor.set_text(where[2], text, ax_index=where[1])
            else:
                getattr(editor, f"set_{where[0]}")(text, ax_index=where[1])
        except ValueError as exc:
            status(str(exc))
            return
        status(f"{len(editor.edits)} edit(s)")

    textbox.on_submit(on_submit)

    buttons: List[Any] = []

    def button(x: float, label: str, callback: Any) -> None:
        b = Button(controls.add_axes((x, 0.22, 0.11, 0.28)), label)
        b.on_clicked(callback)
        buttons.append(b)

    def on_undo(_: Any) -> None:
        editor.undo()
        status(f"Undone ({len(editor._redo_stack)} to redo)")

    def on_redo(_: Any) -> None:
        editor.redo()
        status(f"{len(editor.edits)} edit(s)")

    def on_reset(_: Any) -> None:
        editor.reset()
        status("Back to the original figure")

    def on_grid(_: Any) -> None:
        state["grid"] = not state["grid"]
        for i in range(len(plot_axes)):
            editor.set_grid(state["grid"], ax_index=i)
        status(f"{len(editor.edits)} edit(s)")

    def on_spines(_: Any) -> None:
        state["minimal"] = not state["minimal"]
        with editor.batch():
            for i in range(len(plot_axes)):
                for spine in ("top", "right"):
                    editor.set_spine_visible(spine, not state["minimal"], ax_index=i)
        status(f"{len(editor.edits)} edit(s)")

    def on_theme(_: Any) -> None:
        state["theme"] = (state["theme"] + 1) % len(_THEME_CYCLE)
        name = _THEME_CYCLE[state["theme"]]
        editor.apply_theme(name)
        status(f"Theme: {name}")

    def on_code(_: Any) -> None:
        print("\n" + "=" * 50)
        editor.copy_code()
        print("=" * 50)
        status("Code printed to the console")

    for k, (label, cb) in enumerate(
        (
            ("Undo", on_undo),
            ("Redo", on_redo),
            ("Reset", on_reset),
            ("Grid", on_grid),
            ("Spines", on_spines),
            ("Theme", on_theme),
            ("Code", on_code),
        )
    ):
        button(0.02 + k * 0.137, label, cb)

    def on_close(_: Any) -> None:
        if editor.edits:
            print("\n[StatsPAI] Figure closed. Reproducible code:")
            print("=" * 50)
            editor.copy_code()
            print("=" * 50)

    fig.canvas.mpl_connect("close_event", on_close)

    # Keep the widgets alive and reachable.
    controls._statspai_widgets = {
        "textbox": textbox,
        "buttons": buttons,
        "on_pick": on_pick,
        "info": info,
    }
    fig._statspai_controls = controls
    if show:
        plt.show()
    return controls
