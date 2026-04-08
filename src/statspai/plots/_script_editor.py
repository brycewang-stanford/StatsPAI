"""
Script-mode interactive editor using matplotlib.widgets.

Adds a toolbar panel to the matplotlib figure window for
cosmetic editing when running outside Jupyter notebooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interactive import FigureEditor


def create_script_editor(editor: FigureEditor):
    """
    Attach interactive editing controls to a matplotlib figure window.

    Adds:
    - Click-to-edit on text elements (title, labels, annotations)
    - A bottom toolbar with common actions
    - Print Code button that outputs reproducible code on close

    Parameters
    ----------
    editor : FigureEditor
        The editor wrapping the target figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    fig = editor.fig
    # Capture plot axes BEFORE adding toolbar widget axes
    plot_axes = list(fig.get_axes())
    if not plot_axes:
        plt.show()
        return

    ax = plot_axes[0]

    # Store state for click-editing
    _state = {
        'selected_text': None,
        'editing': False,
        'plot_axes': plot_axes,  # only real plot axes, not toolbar
    }

    # ------------------------------------------------------------------
    # Click-to-edit text elements
    # ------------------------------------------------------------------
    # Make text elements pickable (only for plot axes, not toolbar)
    for a in plot_axes:
        a.title.set_picker(True)
        a.xaxis.label.set_picker(True)
        a.yaxis.label.set_picker(True)
        for t in a.texts:
            t.set_picker(True)

    def _on_pick(event):
        """Handle clicking on a text element."""
        artist = event.artist
        if not hasattr(artist, 'get_text'):
            return
        if _state['editing']:
            return

        role = editor.artist_roles.get(id(artist))
        if role is not None and not editor.is_editable(artist):
            print(f"[StatsPAI] This element is data-locked "
                  f"(role: {role.name})")
            return

        _state['selected_text'] = artist
        current = artist.get_text()
        print(f"[StatsPAI] Selected: '{current}' "
              f"(type new text in the Edit box below)")

    fig.canvas.mpl_connect('pick_event', _on_pick)

    # ------------------------------------------------------------------
    # Bottom toolbar
    # ------------------------------------------------------------------
    # Make room at the bottom for controls
    fig.subplots_adjust(bottom=0.22)

    # Text edit box
    ax_textbox = fig.add_axes([0.15, 0.04, 0.45, 0.05])
    textbox = TextBox(ax_textbox, 'Edit Text: ', initial='')

    def _on_submit(text):
        """Apply text edit to selected element."""
        target = _state['selected_text']
        if target is None:
            print("[StatsPAI] Click on a text element first")
            return

        # Determine what we're editing and track it
        for i, a in enumerate(_state['plot_axes']):
            prefix = f'ax{i}' if i > 0 else 'ax'
            if target is a.title:
                editor.set_title(text, ax_index=i)
                break
            elif target is a.xaxis.label:
                editor.set_xlabel(text, ax_index=i)
                break
            elif target is a.yaxis.label:
                editor.set_ylabel(text, ax_index=i)
                break
            elif target in a.texts:
                old = target.get_text()
                target.set_text(text)
                from .interactive import EditRecord
                editor._record_edit(EditRecord(
                    f'{prefix}.text', 'text', old, text,
                    f"# annotation text changed to {text!r}",
                ))
                editor._refresh()
                break

        _state['selected_text'] = None

    textbox.on_submit(_on_submit)

    # Generate Code button
    ax_codebtn = fig.add_axes([0.62, 0.04, 0.12, 0.05])
    btn_code = Button(ax_codebtn, 'Code')

    def _on_code(event):
        print("\n" + "=" * 50)
        editor.copy_code()
        print("=" * 50)

    btn_code.on_clicked(_on_code)

    # Undo button
    ax_undobtn = fig.add_axes([0.75, 0.04, 0.1, 0.05])
    btn_undo = Button(ax_undobtn, 'Undo')

    def _on_undo(event):
        editor.undo()
        print(f"[StatsPAI] Undone last edit "
              f"({len(editor._redo_stack)} redo available)")

    btn_undo.on_clicked(_on_undo)

    # Redo button
    ax_redobtn = fig.add_axes([0.86, 0.04, 0.1, 0.05])
    btn_redo = Button(ax_redobtn, 'Redo')

    def _on_redo(event):
        editor.redo()
        print(f"[StatsPAI] Redone edit "
              f"({len(editor._redo_stack)} redo remaining)")

    btn_redo.on_clicked(_on_redo)

    # Grid toggle button
    ax_gridbtn = fig.add_axes([0.65, 0.10, 0.12, 0.04])
    btn_grid = Button(ax_gridbtn, 'Grid')
    _grid_state = {'on': False}

    def _on_grid(event):
        _grid_state['on'] = not _grid_state['on']
        editor.set_grid(_grid_state['on'])

    btn_grid.on_clicked(_on_grid)

    # Spine toggle
    ax_spinebtn = fig.add_axes([0.78, 0.10, 0.18, 0.04])
    btn_spine = Button(ax_spinebtn, 'Spines')
    _spine_state = {'minimal': False}

    def _on_spine(event):
        _spine_state['minimal'] = not _spine_state['minimal']
        for spine in ('top', 'right'):
            editor.set_spine_visible(
                spine, not _spine_state['minimal'])

    btn_spine.on_clicked(_on_spine)

    # Info label
    ax_info = fig.add_axes([0.15, 0.10, 0.45, 0.04])
    ax_info.set_axis_off()
    ax_info.text(
        0, 0.5,
        'Click text to select, edit below. '
        'Data elements are locked.',
        fontsize=8, color='#666', va='center',
        transform=ax_info.transAxes,
    )

    # Print code on figure close
    def _on_close(event):
        if editor.edits:
            print("\n[StatsPAI] Figure closed. Reproducible code:")
            print("=" * 50)
            editor.copy_code()
            print("=" * 50)

    fig.canvas.mpl_connect('close_event', _on_close)

    # Keep references alive (prevent garbage collection)
    fig._statspai_widgets = {
        'textbox': textbox,
        'btn_code': btn_code,
        'btn_undo': btn_undo,
        'btn_redo': btn_redo,
        'btn_grid': btn_grid,
        'btn_spine': btn_spine,
    }

    plt.show()
