"""
Jupyter notebook interactive editor using ipywidgets.

Creates a tabbed control panel beside the matplotlib figure for
real-time cosmetic editing with live preview and code generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interactive import FigureEditor

logger = logging.getLogger(__name__)


_LEGEND_LOCS = [
    'best', 'upper right', 'upper left', 'lower left',
    'lower right', 'right', 'center left', 'center right',
    'lower center', 'upper center', 'center',
]

_LINE_STYLES = [
    ('-', 'solid'), ('--', 'dashed'), ('-.', 'dashdot'),
    (':', 'dotted'),
]

_MARKERS = [
    ('o', 'circle'), ('s', 'square'), ('^', 'triangle'),
    ('D', 'diamond'), ('v', 'tri down'), ('*', 'star'),
    ('p', 'pentagon'), ('h', 'hexagon'), ('', 'none'),
]


def _code_to_html(code: str) -> str:
    """Render code as a selectable <pre> block that supports native copy."""
    import html as _html
    escaped = _html.escape(code)
    return (
        '<pre style="'
        'background:#f8f8f8; border:1px solid #ddd; border-radius:4px; '
        'padding:10px; font-family:Menlo,Monaco,Consolas,monospace; '
        'font-size:12px; line-height:1.5; white-space:pre-wrap; '
        'word-wrap:break-word; overflow-x:auto; max-height:300px; '
        'overflow-y:auto; user-select:text; cursor:text; '
        'margin:0;'
        f'">{escaped}</pre>'
    )


def create_jupyter_panel(editor: FigureEditor):
    """
    Build and display an ipywidgets control panel for the editor.

    Features:
    - Multi-axis selector for subplot figures
    - Full style controls (color, linewidth, linestyle, alpha, marker)
    - Scatter and CI alpha editing
    - Save/Export and Reset buttons
    - Slider debouncing to prevent edit flooding
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets not installed. Install: pip install ipywidgets")
        print("Falling back to programmatic API. Use editor methods:")
        print("  editor.set_title('New Title')")
        print("  editor.generate_code()")
        return

    import io
    import matplotlib.pyplot as plt
    from .interactive import ArtistRole

    fig = editor.fig
    axes = fig.get_axes()

    if not axes:
        print("No axes found in figure.")
        return

    # ---- Live preview: render figure to Image widget ----
    preview_dpi = 100  # preview resolution (fast)

    def _render_fig_to_png(figure, dpi=preview_dpi):
        """Render a matplotlib figure to PNG bytes."""
        buf = io.BytesIO()
        figure.savefig(buf, format='png', dpi=dpi,
                       bbox_inches='tight', facecolor=figure.get_facecolor())
        buf.seek(0)
        return buf.read()

    fig_image = widgets.Image(
        value=_render_fig_to_png(fig),
        format='png',
        layout=widgets.Layout(
            max_width='100%',
            border='1px solid #eee',
        ),
    )

    # Status bar showing edit count
    status_bar = widgets.HTML(
        '<span style="font-size:11px; color:#999">'
        'Ready — make edits on the right panel</span>'
    )

    # ---- Render mode: Auto vs Manual ----
    _render_state = {'pending': 0, 'auto': True}

    def _safe(fn):
        """Wrap widget callbacks to surface exceptions in status_bar."""
        import functools
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                logger.debug("Widget callback %s failed", fn.__name__,
                             exc_info=True)
                status_bar.value = (
                    f'<span style="font-size:11px; color:#E74C3C">'
                    f'Error in {fn.__name__}: {exc}</span>'
                )
        return wrapper

    render_mode = widgets.Button(
        description='Auto',
        tooltip='Click to toggle auto/manual render',
        button_style='success',
        icon='refresh',
        layout=widgets.Layout(width='72px', height='26px', padding='0'),
    )
    apply_btn = widgets.Button(
        description='Apply',
        button_style='warning',
        icon='refresh',
        layout=widgets.Layout(width='auto', height='26px', display='none'),
    )

    def _do_render(figure):
        """Actually render the figure to the preview widget."""
        fig_image.value = _render_fig_to_png(figure, dpi=preview_dpi)
        n = len(editor.edits)
        w_in, h_in = figure.get_size_inches()
        export_dpi = int(figure.dpi)
        px_w, px_h = int(w_in * export_dpi), int(h_in * export_dpi)
        _render_state['pending'] = 0
        apply_btn.description = 'Apply'
        status_bar.value = (
            f'<span style="font-size:11px; color:#2ECC71">'
            f'{n} edit(s) — DPI: {export_dpi} → '
            f'export size: {px_w}\u00d7{px_h}px</span>'
        )

    def _live_refresh(figure):
        """Callback: re-render or queue depending on render mode."""
        try:
            if _render_state['auto']:
                _do_render(figure)
            else:
                _render_state['pending'] += 1
                p = _render_state['pending']
                n = len(editor.edits)
                apply_btn.description = f'Apply ({p})'
                status_bar.value = (
                    f'<span style="font-size:11px; color:#F39C12">'
                    f'{n} edit(s) — {p} pending re-render '
                    f'(click Apply)</span>'
                )
        except Exception as exc:
            logger.debug("Live refresh failed", exc_info=True)
            status_bar.value = (
                f'<span style="font-size:11px; color:#E74C3C">'
                f'Render error: {exc}</span>'
            )

    @_safe
    def _on_apply(btn):
        _do_render(editor.fig)

    apply_btn.on_click(_on_apply)

    @_safe
    def _on_toggle_mode(btn):
        _render_state['auto'] = not _render_state['auto']
        if _render_state['auto']:
            render_mode.description = 'Auto'
            render_mode.button_style = 'success'
            apply_btn.layout.display = 'none'
            # Always refresh when switching back to Auto
            _do_render(editor.fig)
        else:
            render_mode.description = 'Manual'
            render_mode.button_style = ''
            apply_btn.layout.display = ''

    render_mode.on_click(_on_toggle_mode)

    # Register the live refresh callback
    editor.on_refresh(_live_refresh)

    render_bar = widgets.HBox(
        [render_mode],
        layout=widgets.Layout(
            justify_content='flex-start',
            align_items='center',
        ),
    )
    apply_bar = widgets.HBox(
        [apply_btn],
        layout=widgets.Layout(
            justify_content='flex-end',
            width='100%',
            flex='0 0 auto',
        ),
    )

    # Preview container (figure + status bar)
    fig_container = widgets.VBox([
        fig_image,
        status_bar,
    ], layout=widgets.Layout(
        flex='1 1 auto',
        min_width='400px',
    ))

    # ---- Axis selector (for multi-panel figures) ----
    ax_selector = widgets.Dropdown(
        options=[(f'Axis {i}' if i > 0 else 'Main Axis', i)
                 for i in range(len(axes))],
        value=0,
        description='Axis:',
        layout=widgets.Layout(width='95%'),
    )

    def _get_ax_idx():
        return ax_selector.value

    def _get_ax():
        return axes[_get_ax_idx()]

    # ==================================================================
    # Tab 1: Text (titles, labels)
    # ==================================================================
    title_text = widgets.Text(
        value=axes[0].get_title(), description='Title:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    title_size = widgets.FloatSlider(
        value=axes[0].title.get_fontsize(), min=6, max=30, step=1,
        description='Title size:', layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    xlabel_text = widgets.Text(
        value=axes[0].get_xlabel(), description='X Label:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    xlabel_size = widgets.FloatSlider(
        value=axes[0].xaxis.label.get_fontsize(), min=6, max=24, step=1,
        description='X size:', layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    ylabel_text = widgets.Text(
        value=axes[0].get_ylabel(), description='Y Label:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    ylabel_size = widgets.FloatSlider(
        value=axes[0].yaxis.label.get_fontsize(), min=6, max=24, step=1,
        description='Y size:', layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    tick_size = widgets.FloatSlider(
        value=10, min=6, max=20, step=1,
        description='Tick size:', layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )

    # Guard flag: suppress slider observers during programmatic sync
    _syncing = {'active': False}

    def _on_ax_change(change):
        """Update all widgets when axis selection changes."""
        _syncing['active'] = True
        try:
            ax = axes[change['new']]
            # Text tab
            title_text.value = ax.get_title()
            title_size.value = ax.title.get_fontsize()
            xlabel_text.value = ax.get_xlabel()
            xlabel_size.value = ax.xaxis.label.get_fontsize()
            ylabel_text.value = ax.get_ylabel()
            ylabel_size.value = ax.yaxis.label.get_fontsize()
            # Layout tab: spines
            spine_top.value = ax.spines['top'].get_visible()
            spine_right.value = ax.spines['right'].get_visible()
            spine_bottom.value = ax.spines['bottom'].get_visible()
            spine_left.value = ax.spines['left'].get_visible()
            # Layout tab: grid
            gridlines = ax.xaxis.get_gridlines()
            grid_toggle.value = (
                gridlines[0].get_visible() if gridlines else False
            )
            # Layout tab: legend
            legend_visible.value = ax.get_legend() is not None
            # Layout tab: background
            ax_bg_color.value = _to_hex(ax.get_facecolor())
            # Layout tab: axis limits
            xl = ax.get_xlim()
            yl = ax.get_ylim()
            xr = max(xl[1] - xl[0], 0.01)
            yr = max(yl[1] - yl[0], 0.01)
            xlim_range.min = xl[0] - xr * 0.5
            xlim_range.max = xl[1] + xr * 0.5
            xlim_range.step = xr / 50
            xlim_range.value = [xl[0], xl[1]]
            ylim_range.min = yl[0] - yr * 0.5
            ylim_range.max = yl[1] + yr * 0.5
            ylim_range.step = yr / 50
            ylim_range.value = [yl[0], yl[1]]
        finally:
            _syncing['active'] = False
        # Rebuild style tab content (only if already loaded)
        if _tab_loaded.get(2, False):
            _rebuild_style_widgets()

    ax_selector.observe(_on_ax_change, names='value')

    def _on_title_change(change):
        if _syncing['active']:
            return
        editor.set_title(change['new'], ax_index=_get_ax_idx())

    def _on_title_size(change):
        if _syncing['active']:
            return
        editor.set_fontsize('title', change['new'],
                            ax_index=_get_ax_idx())

    def _on_xlabel_change(change):
        if _syncing['active']:
            return
        editor.set_xlabel(change['new'], ax_index=_get_ax_idx())

    def _on_xlabel_size(change):
        if _syncing['active']:
            return
        editor.set_fontsize('xlabel', change['new'],
                            ax_index=_get_ax_idx())

    def _on_ylabel_change(change):
        if _syncing['active']:
            return
        editor.set_ylabel(change['new'], ax_index=_get_ax_idx())

    def _on_ylabel_size(change):
        if _syncing['active']:
            return
        editor.set_fontsize('ylabel', change['new'],
                            ax_index=_get_ax_idx())

    def _on_tick_size(change):
        if _syncing['active']:
            return
        editor.set_fontsize('ticks', change['new'],
                            ax_index=_get_ax_idx())

    title_text.observe(_on_title_change, names='value')
    title_size.observe(_on_title_size, names='value')
    xlabel_text.observe(_on_xlabel_change, names='value')
    xlabel_size.observe(_on_xlabel_size, names='value')
    ylabel_text.observe(_on_ylabel_change, names='value')
    ylabel_size.observe(_on_ylabel_size, names='value')
    tick_size.observe(_on_tick_size, names='value')

    # ---- Font controls ----
    from .interactive import FONT_PRESETS, SIZE_PRESETS, _resolve_preset_fonts, get_font_choices

    # --- Font preset dropdown (font family/name only) ---
    font_preset_options = [('-- Choose font --', '')]
    for name, preset in FONT_PRESETS.items():
        fonts = _resolve_preset_fonts(preset)
        primary = fonts[0] if fonts else '?'
        venue = preset.get('venue', '')
        label = f"{primary}  ({venue})" if venue else primary
        font_preset_options.append((label, name))
    font_preset = widgets.Dropdown(
        options=font_preset_options, value='',
        description='Font:',
        layout=widgets.Layout(width='95%'),
    )
    font_preset_info = widgets.HTML('')

    def _on_font_preset(change):
        name = change['new']
        if not name:
            return
        try:
            editor.apply_font_preset(name, ax_index=_get_ax_idx())
            preset = FONT_PRESETS[name]
            fonts = _resolve_preset_fonts(preset)
            primary = fonts[0] if fonts else '?'
            font_preset_info.value = (
                f'<span style="color:#2ECC71; font-size:11px">'
                f'Applied: {primary}</span>'
            )
            # Sync size sliders without triggering observer callbacks
            _syncing['active'] = True
            try:
                title_size.value = preset['title_size']
                xlabel_size.value = preset['label_size']
                ylabel_size.value = preset['label_size']
                tick_size.value = preset['tick_size']
            finally:
                _syncing['active'] = False
        except Exception as e:
            font_preset_info.value = (
                f'<span style="color:#E74C3C; font-size:11px">'
                f'Error: {e}</span>'
            )

    font_preset.observe(_on_font_preset, names='value')

    # --- Size preset dropdown (sizes only, independent of font) ---
    size_preset_options = [('-- Choose size --', '')]
    for name, sp_info in SIZE_PRESETS.items():
        desc = sp_info.get('desc', '')
        label = f"{name}  (title {sp_info['title_size']}, label {sp_info['label_size']}, tick {sp_info['tick_size']})"
        size_preset_options.append((label, name))
    size_preset = widgets.Dropdown(
        options=size_preset_options, value='',
        description='Sizes:',
        layout=widgets.Layout(width='95%'),
    )

    def _on_size_preset(change):
        name = change['new']
        if not name:
            return
        try:
            editor.apply_size_preset(name, ax_index=_get_ax_idx())
            sp_info = SIZE_PRESETS[name]
            # Sync sliders without triggering observer callbacks
            _syncing['active'] = True
            try:
                title_size.value = sp_info['title_size']
                xlabel_size.value = sp_info['label_size']
                ylabel_size.value = sp_info['label_size']
                tick_size.value = sp_info['tick_size']
            finally:
                _syncing['active'] = False
        except Exception:
            pass

    size_preset.observe(_on_size_preset, names='value')

    # Font family dropdown
    font_family = widgets.Dropdown(
        options=[('Serif', 'serif'),
                 ('Sans-serif', 'sans-serif'),
                 ('Monospace', 'monospace')],
        value='serif',
        description='Family:',
        layout=widgets.Layout(width='95%'),
    )

    # Specific font dropdown (updates based on family selection)
    def _get_font_options(family):
        choices = get_font_choices()
        if family == 'serif':
            fonts = choices['English Serif'] + choices['CJK']
        elif family == 'sans-serif':
            fonts = choices['English Sans-serif'] + choices['CJK']
        else:
            fonts = choices['Monospace']
        return [('(auto)', '')] + [(f, f) for f in fonts]

    font_name = widgets.Dropdown(
        options=_get_font_options('serif'),
        value='',
        description='Font:',
        layout=widgets.Layout(width='95%'),
    )

    def _on_font_family(change):
        font_name.options = _get_font_options(change['new'])
        font_name.value = ''
        editor.set_font(change['new'], ax_index=_get_ax_idx())

    def _on_font_name(change):
        if change['new']:
            editor.set_font(font_family.value, change['new'],
                            ax_index=_get_ax_idx())

    font_family.observe(_on_font_family, names='value')
    font_name.observe(_on_font_name, names='value')

    text_tab = widgets.VBox([
        widgets.HTML('<b>Text & Labels</b>'),
        ax_selector if len(axes) > 1 else widgets.HTML(''),
        title_text, title_size,
        widgets.HTML('<hr style="margin:4px 0">'),
        xlabel_text, xlabel_size,
        ylabel_text, ylabel_size,
        widgets.HTML('<hr style="margin:4px 0">'),
        tick_size,
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<b>Font</b>'),
        font_preset,
        font_preset_info,
        widgets.HTML('<hr style="margin:2px 0">'),
        widgets.HTML('<b>Size Preset</b>'),
        size_preset,
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML(
            '<span style="font-size:11px; color:#666">'
            'Or choose font manually:</span>'
        ),
        font_family,
        font_name,
    ])

    # ==================================================================
    # Tab 2: Style (colors, linewidth, linestyle, alpha, markers)
    # ==================================================================
    style_container = widgets.VBox([
        widgets.HTML('<b>Colors & Style</b>'),
        widgets.HTML('<i>Loading...</i>'),
    ])

    def _rebuild_style_widgets():
        """Rebuild style widgets for the current axis."""
        ax = _get_ax()
        ax_idx = _get_ax_idx()
        children = [widgets.HTML('<b>Colors & Style</b>')]

        # Title color
        title_color = widgets.ColorPicker(
            value=_to_hex(ax.title.get_color()),
            description='Title color:',
            layout=widgets.Layout(width='55%'),
        )
        title_color_status = widgets.HTML('')

        def _on_tc(change):
            editor.set_color('title', change['new'],
                             ax_index=ax_idx)
            title_color_status.value = (
                '<span style="color:#2ECC71; font-size:11px">'
                '\u2714 Applied</span>')

        title_color.observe(_on_tc, names='value')
        children.append(widgets.HBox([title_color, title_color_status]))

        # Line controls
        lines = ax.get_lines()
        _max_lines = 20
        for i, line in enumerate(lines[:_max_lines]):
            label = line.get_label() or f'Line {i}'
            if label.startswith('_'):
                label = f'Line {i}'
            role = editor.artist_roles.get(id(line))
            role_tag = f' [{role.name}]' if role else ''

            children.append(widgets.HTML(
                f'<hr style="margin:4px 0"><i>{label}{role_tag}</i>'
            ))

            # Color
            cp = widgets.ColorPicker(
                value=_to_hex(line.get_color()),
                description='Color:',
                layout=widgets.Layout(width='55%'),
            )
            cp_status = widgets.HTML('')

            def _make_color_cb(idx, status_widget):
                def _cb(change):
                    editor.set_color(f'line{idx}', change['new'],
                                     ax_index=ax_idx)
                    status_widget.value = (
                        '<span style="color:#2ECC71; font-size:11px">'
                        '\u2714 Applied</span>')
                return _cb

            cp.observe(_make_color_cb(i, cp_status), names='value')
            children.append(widgets.HBox([cp, cp_status]))

            # Linewidth
            lw = widgets.FloatSlider(
                value=line.get_linewidth(), min=0.5, max=6, step=0.5,
                description='Width:',
                layout=widgets.Layout(width='95%'),
                continuous_update=False,
            )

            def _make_lw_cb(idx):
                def _cb(change):
                    editor.set_linewidth(idx, change['new'],
                                         ax_index=ax_idx)
                return _cb

            lw.observe(_make_lw_cb(i), names='value')
            children.append(lw)

            # Linestyle
            current_ls = line.get_linestyle()
            ls_options = [(name, code) for code, name in _LINE_STYLES]
            ls = widgets.Dropdown(
                options=ls_options,
                value=current_ls if current_ls in [c for c, _ in _LINE_STYLES] else '-',
                description='Style:',
                layout=widgets.Layout(width='95%'),
            )

            def _make_ls_cb(idx):
                def _cb(change):
                    editor.set_linestyle(idx, change['new'],
                                         ax_index=ax_idx)
                return _cb

            ls.observe(_make_ls_cb(i), names='value')
            children.append(ls)

            # Alpha
            current_alpha = line.get_alpha()
            al = widgets.FloatSlider(
                value=current_alpha if current_alpha is not None else 1.0,
                min=0, max=1, step=0.05,
                description='Alpha:',
                layout=widgets.Layout(width='95%'),
                continuous_update=False,
            )

            def _make_alpha_cb(idx):
                def _cb(change):
                    editor.set_alpha(f'line{idx}', change['new'],
                                     ax_index=ax_idx)
                return _cb

            al.observe(_make_alpha_cb(i), names='value')
            children.append(al)

            # Marker
            current_marker = line.get_marker()
            mk_options = [(name, code) for code, name in _MARKERS]
            mk = widgets.Dropdown(
                options=mk_options,
                value=(current_marker
                       if current_marker in [c for c, _ in _MARKERS]
                       else ''),
                description='Marker:',
                layout=widgets.Layout(width='95%'),
            )

            def _make_mk_cb(idx):
                def _cb(change):
                    editor.set_marker(idx, change['new'],
                                      ax_index=ax_idx)
                return _cb

            mk.observe(_make_mk_cb(i), names='value')
            children.append(mk)

        if len(lines) > _max_lines:
            children.append(widgets.HTML(
                f'<span style="font-size:11px; color:#999">'
                f'Showing {_max_lines} of {len(lines)} lines. '
                f'Use the API for the rest.</span>'
            ))

        # Scatter / Collection controls
        import matplotlib.collections as mcoll
        _max_collections = 20
        for i, coll in enumerate(ax.collections[:_max_collections]):
            role = editor.artist_roles.get(id(coll))
            if role is None:
                continue

            if isinstance(coll, mcoll.PathCollection):
                coll_label = f'Scatter {i} [{role.name}]'
            elif isinstance(coll, mcoll.PolyCollection):
                coll_label = f'CI/Fill {i} [{role.name}]'
            else:
                coll_label = f'Collection {i} [{role.name}]'

            children.append(widgets.HTML(
                f'<hr style="margin:4px 0"><i>{coll_label}</i>'
            ))

            # Scatter color
            if isinstance(coll, mcoll.PathCollection):
                fc = coll.get_facecolors()
                hex_color = _to_hex(fc[0] if len(fc) > 0 else 'black')
                sc_cp = widgets.ColorPicker(
                    value=hex_color, description='Color:',
                    layout=widgets.Layout(width='55%'),
                )
                sc_cp_status = widgets.HTML('')

                def _make_sc_cb(idx, status_widget):
                    def _cb(change):
                        editor.set_scatter_color(
                            idx, change['new'], ax_index=ax_idx)
                        status_widget.value = (
                            '<span style="color:#2ECC71; font-size:11px">'
                            '\u2714 Applied</span>')
                    return _cb

                sc_cp.observe(_make_sc_cb(i, sc_cp_status), names='value')
                children.append(widgets.HBox([sc_cp, sc_cp_status]))

            # Alpha for CI/fills
            coll_alpha = coll.get_alpha()
            ca = widgets.FloatSlider(
                value=coll_alpha if coll_alpha is not None else 1.0,
                min=0, max=1, step=0.05,
                description='Alpha:',
                layout=widgets.Layout(width='95%'),
                continuous_update=False,
            )
            target_prefix = 'scatter' if isinstance(
                coll, mcoll.PathCollection) else 'ci'

            def _make_ca_cb(prefix, idx):
                def _cb(change):
                    editor.set_alpha(f'{prefix}{idx}', change['new'],
                                     ax_index=ax_idx)
                return _cb

            ca.observe(_make_ca_cb(target_prefix, i), names='value')
            children.append(ca)

        if len(ax.collections) > _max_collections:
            children.append(widgets.HTML(
                f'<span style="font-size:11px; color:#999">'
                f'Showing {_max_collections} of {len(ax.collections)} '
                f'collections. Use the API for the rest.</span>'
            ))

        style_container.children = children

    # NOTE: _rebuild_style_widgets() is deferred — called on first
    # tab switch to Style tab (see _on_tab_switch).

    # ==================================================================
    # Tab 3: Layout (spines, grid, figsize, legend, axes limits, etc.)
    # ==================================================================
    spine_top = widgets.Checkbox(
        value=axes[0].spines['top'].get_visible(),
        description='Top spine',
    )
    spine_right = widgets.Checkbox(
        value=axes[0].spines['right'].get_visible(),
        description='Right spine',
    )
    spine_bottom = widgets.Checkbox(
        value=axes[0].spines['bottom'].get_visible(),
        description='Bottom spine',
    )
    spine_left = widgets.Checkbox(
        value=axes[0].spines['left'].get_visible(),
        description='Left spine',
    )

    # ---- Grid controls (toggle + style) ----
    grid_toggle = widgets.Checkbox(
        value=False, description='Show grid',
    )
    grid_color = widgets.ColorPicker(
        value='#cccccc', description='Grid color:',
        layout=widgets.Layout(width='55%'),
    )
    grid_color_status = widgets.HTML('')
    grid_alpha = widgets.FloatSlider(
        value=0.7, min=0, max=1, step=0.05,
        description='Grid alpha:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    _GRID_STYLES = [('-', 'solid'), ('--', 'dashed'),
                    (':', 'dotted'), ('-.', 'dashdot')]
    grid_linestyle = widgets.Dropdown(
        options=[(name, code) for code, name in _GRID_STYLES],
        value='-',
        description='Grid style:',
        layout=widgets.Layout(width='95%'),
    )
    # Hide grid style controls initially
    grid_style_box = widgets.VBox(
        [widgets.HBox([grid_color, grid_color_status]),
         grid_alpha, grid_linestyle],
        layout=widgets.Layout(display='none'),
    )

    # ---- Figure size with presets ----
    _SIZE_PRESETS = {
        '': ('Custom', None),
        'journal_col': ('Journal column (3.5×2.6)', (3.5, 2.6)),
        'journal_full': ('Journal full-width (7×4.5)', (7, 4.5)),
        'aea': ('AER/AEJ (6.5×4.5)', (6.5, 4.5)),
        'slide_16_9': ('Slide 16:9 (12×6.75)', (12, 6.75)),
        'slide_4_3': ('Slide 4:3 (10×7.5)', (10, 7.5)),
        'square': ('Square (6×6)', (6, 6)),
        'poster': ('Poster (14×10)', (14, 10)),
        'a4_half': ('A4 Half (8.27×5.83)', (8.27, 5.83)),
    }
    size_preset = widgets.Dropdown(
        options=[(label, key) for key, (label, _) in _SIZE_PRESETS.items()],
        value='',
        description='Size preset:',
        layout=widgets.Layout(width='95%'),
    )

    fig_width = widgets.FloatSlider(
        value=fig.get_size_inches()[0], min=2, max=20, step=0.25,
        description='Width:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    fig_height = widgets.FloatSlider(
        value=fig.get_size_inches()[1], min=1.5, max=15, step=0.25,
        description='Height:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )

    # ---- DPI slider ----
    fig_dpi = widgets.IntSlider(
        value=int(fig.dpi), min=72, max=600, step=10,
        description='Figure DPI:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    dpi_hint = widgets.HTML(
        '<span style="font-size:11px; color:#888">'
        'DPI affects export/save resolution. '
        'Preview is scaled to fit — visual size won\'t change.</span>'
    )

    # ---- Legend controls ----
    legend_visible = widgets.Checkbox(
        value=axes[0].get_legend() is not None,
        description='Show legend',
    )
    legend_loc = widgets.Dropdown(
        options=_LEGEND_LOCS, value='best',
        description='Legend loc:',
        layout=widgets.Layout(width='95%'),
    )
    legend_fontsize = widgets.FloatSlider(
        value=9, min=5, max=20, step=0.5,
        description='Legend size:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )

    # ---- Title font weight ----
    title_weight = widgets.Dropdown(
        options=[('Normal', 'normal'), ('Bold', 'bold')],
        value='normal',
        description='Title weight:',
        layout=widgets.Layout(width='95%'),
    )

    # ---- Tick rotation ----
    xtick_rotation = widgets.FloatSlider(
        value=0, min=0, max=90, step=5,
        description='X tick rot:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    ytick_rotation = widgets.FloatSlider(
        value=0, min=0, max=90, step=5,
        description='Y tick rot:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )

    # ---- Background color ----
    fig_bg_color = widgets.ColorPicker(
        value=_to_hex(fig.get_facecolor()),
        description='Fig bg:',
        layout=widgets.Layout(width='55%'),
    )
    fig_bg_status = widgets.HTML('')
    ax_bg_color = widgets.ColorPicker(
        value=_to_hex(axes[0].get_facecolor()),
        description='Axes bg:',
        layout=widgets.Layout(width='55%'),
    )
    ax_bg_status = widgets.HTML('')

    # ---- Tight layout & annotation ----
    tight_btn = widgets.Button(
        description='Tight Layout',
        button_style='info',
        icon='compress',
        layout=widgets.Layout(width='45%'),
    )
    tight_info = widgets.HTML('')

    # ---- Annotation ----
    annot_text = widgets.Text(
        value='', description='Text:',
        placeholder='annotation text',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    annot_x = widgets.FloatText(
        value=0, description='X pos:',
        layout=widgets.Layout(width='47%'),
    )
    annot_y = widgets.FloatText(
        value=0, description='Y pos:',
        layout=widgets.Layout(width='47%'),
    )
    annot_draggable = widgets.Checkbox(
        value=False, description='Draggable',
        layout=widgets.Layout(width='auto'),
    )
    annot_btn = widgets.Button(
        description='Add Annotation',
        button_style='primary',
        icon='pencil',
        layout=widgets.Layout(width='45%'),
    )
    annot_info = widgets.HTML(
        '<span style="font-size:10px; color:#999">'
        'Set X/Y to data coordinates, then click Add. '
        'Draggable works in matplotlib GUI windows.</span>'
    )

    # ---- Axis limits ----
    xlim = axes[0].get_xlim()
    ylim = axes[0].get_ylim()
    # Ensure lo < hi (axes can be inverted)
    x_lo, x_hi = min(xlim), max(xlim)
    y_lo, y_hi = min(ylim), max(ylim)
    x_range = max(x_hi - x_lo, 0.01)
    y_range = max(y_hi - y_lo, 0.01)

    xlim_range = widgets.FloatRangeSlider(
        value=[x_lo, x_hi],
        min=x_lo - x_range * 0.5,
        max=x_hi + x_range * 0.5,
        step=x_range / 50,
        description='X range:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    ylim_range = widgets.FloatRangeSlider(
        value=[y_lo, y_hi],
        min=y_lo - y_range * 0.5,
        max=y_hi + y_range * 0.5,
        step=y_range / 50,
        description='Y range:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )

    # ---- Callbacks ----
    def _on_spine(spine_name):
        def _cb(change):
            if _syncing['active']:
                return
            editor.set_spine_visible(spine_name, change['new'],
                                     ax_index=_get_ax_idx())
        return _cb

    spine_top.observe(_on_spine('top'), names='value')
    spine_right.observe(_on_spine('right'), names='value')
    spine_bottom.observe(_on_spine('bottom'), names='value')
    spine_left.observe(_on_spine('left'), names='value')

    def _on_grid(change):
        if _syncing['active']:
            return
        if change['new']:
            grid_style_box.layout.display = ''
            editor.set_grid(True, ax_index=_get_ax_idx())
        else:
            grid_style_box.layout.display = 'none'
            editor.set_grid(False, ax_index=_get_ax_idx())

    grid_toggle.observe(_on_grid, names='value')

    def _on_grid_color(change):
        if grid_toggle.value:
            editor.set_grid_style(color=change['new'],
                                  ax_index=_get_ax_idx())
            grid_color_status.value = (
                '<span style="color:#2ECC71; font-size:11px">'
                '\u2714 Applied</span>')

    def _on_grid_alpha(change):
        if grid_toggle.value:
            editor.set_grid_style(alpha=change['new'],
                                  ax_index=_get_ax_idx())

    def _on_grid_linestyle(change):
        if grid_toggle.value:
            editor.set_grid_style(linestyle=change['new'],
                                  ax_index=_get_ax_idx())

    grid_color.observe(_on_grid_color, names='value')
    grid_alpha.observe(_on_grid_alpha, names='value')
    grid_linestyle.observe(_on_grid_linestyle, names='value')

    def _on_figsize_preset(change):
        key = change['new']
        if not key:
            return
        _, size = _SIZE_PRESETS[key]
        if size is not None:
            editor.set_figsize(size[0], size[1])
            # Sync sliders without triggering observer callbacks
            _syncing['active'] = True
            try:
                fig_width.value = size[0]
                fig_height.value = size[1]
            finally:
                _syncing['active'] = False

    size_preset.observe(_on_figsize_preset, names='value')

    def _on_figsize_w(change):
        if _syncing['active']:
            return
        editor.set_figsize(change['new'], fig.get_size_inches()[1])

    def _on_figsize_h(change):
        if _syncing['active']:
            return
        editor.set_figsize(fig.get_size_inches()[0], change['new'])

    fig_width.observe(_on_figsize_w, names='value')
    fig_height.observe(_on_figsize_h, names='value')

    def _on_dpi(change):
        if _syncing['active']:
            return
        editor.set_dpi(change['new'])
        # Sync save DPI to match figure DPI
        _syncing['active'] = True
        try:
            save_dpi.value = change['new']
        finally:
            _syncing['active'] = False

    fig_dpi.observe(_on_dpi, names='value')

    def _on_legend_visible(change):
        if _syncing['active']:
            return
        editor.set_legend_visible(change['new'], ax_index=_get_ax_idx())

    def _on_legend_loc(change):
        editor.set_legend(loc=change['new'], ax_index=_get_ax_idx())

    def _on_legend_fontsize(change):
        editor.set_legend(fontsize=change['new'], ax_index=_get_ax_idx())

    legend_visible.observe(_on_legend_visible, names='value')
    legend_loc.observe(_on_legend_loc, names='value')
    legend_fontsize.observe(_on_legend_fontsize, names='value')

    def _on_title_weight(change):
        editor.set_title_weight(change['new'], ax_index=_get_ax_idx())

    title_weight.observe(_on_title_weight, names='value')

    def _on_xtick_rot(change):
        editor.set_tick_rotation('x', change['new'],
                                 ax_index=_get_ax_idx())

    def _on_ytick_rot(change):
        editor.set_tick_rotation('y', change['new'],
                                 ax_index=_get_ax_idx())

    xtick_rotation.observe(_on_xtick_rot, names='value')
    ytick_rotation.observe(_on_ytick_rot, names='value')

    def _on_fig_bg(change):
        if _syncing['active']:
            return
        editor.set_background_color(change['new'], target='figure')
        fig_bg_status.value = (
            '<span style="color:#2ECC71; font-size:11px">'
            '\u2714 Applied</span>')

    def _on_ax_bg(change):
        if _syncing['active']:
            return
        editor.set_background_color(change['new'], target='axes',
                                    ax_index=_get_ax_idx())
        ax_bg_status.value = (
            '<span style="color:#2ECC71; font-size:11px">'
            '\u2714 Applied</span>')

    fig_bg_color.observe(_on_fig_bg, names='value')
    ax_bg_color.observe(_on_ax_bg, names='value')

    def _on_tight(btn):
        editor.tight_layout()
        tight_info.value = (
            '<span style="color:#2ECC71; font-size:11px">'
            'Tight layout applied</span>')

    tight_btn.on_click(_on_tight)

    def _on_annot_add(btn):
        text = annot_text.value
        if not text:
            annot_info.value = (
                '<span style="color:#E74C3C; font-size:11px">'
                'Enter annotation text first</span>')
            return
        editor.add_annotation(
            text, xy=(annot_x.value, annot_y.value),
            ax_index=_get_ax_idx(),
            draggable=annot_draggable.value,
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='wheat', alpha=0.5),
        )
        drag_note = ' (draggable)' if annot_draggable.value else ''
        annot_info.value = (
            f'<span style="color:#2ECC71; font-size:11px">'
            f'Added: "{text}" at ({annot_x.value}, {annot_y.value})'
            f'{drag_note}</span>')
        annot_text.value = ''

    annot_btn.on_click(_on_annot_add)

    def _on_xlim(change):
        if _syncing['active']:
            return
        editor.set_xlim(change['new'][0], change['new'][1],
                        ax_index=_get_ax_idx())

    def _on_ylim(change):
        if _syncing['active']:
            return
        editor.set_ylim(change['new'][0], change['new'][1],
                        ax_index=_get_ax_idx())

    xlim_range.observe(_on_xlim, names='value')
    ylim_range.observe(_on_ylim, names='value')

    protect_label = widgets.HTML(
        '<span style="color:#E74C3C; font-size:11px">'
        'Data protection ON: axis limits auto-expand to include all '
        'data points</span>'
    ) if editor.protect_data else widgets.HTML('')

    layout_tab = widgets.VBox([
        widgets.HTML('<b>Spines</b>'),
        widgets.HBox([spine_top, spine_right]),
        widgets.HBox([spine_bottom, spine_left]),
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<b>Grid</b>'),
        grid_toggle,
        grid_style_box,
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<b>Figure Size</b>'),
        size_preset, fig_width, fig_height,
        fig_dpi, dpi_hint,
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<b>Background</b>'),
        widgets.HBox([fig_bg_color, fig_bg_status]),
        widgets.HBox([ax_bg_color, ax_bg_status]),
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<b>Legend</b>'),
        legend_visible, legend_loc, legend_fontsize,
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<b>Title Style</b>'),
        title_weight,
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<b>Tick Rotation</b>'),
        xtick_rotation, ytick_rotation,
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<b>Axis Limits</b>'),
        protect_label,
        xlim_range, ylim_range,
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<b>Annotation</b>'),
        annot_text,
        widgets.HBox([annot_x, annot_y]),
        widgets.HBox([annot_draggable]),
        widgets.HBox([annot_btn, tight_btn]),
        annot_info, tight_info,
    ])

    # ==================================================================
    # Tab 4: Theme (StatsPAI + matplotlib + seaborn)
    # ==================================================================
    from .themes import list_themes
    all_themes = list_themes()

    # Build grouped options for the dropdown
    theme_options = [('-- Reset to Default --', 'default')]
    # StatsPAI themes first
    for t in all_themes.get('statspai', []):
        theme_options.append((f'[StatsPAI]  {t}', t))
    # matplotlib styles
    for t in all_themes.get('matplotlib', []):
        theme_options.append((f'[matplotlib]  {t}', t))
    # seaborn styles
    for t in all_themes.get('seaborn', []):
        theme_options.append((f'[seaborn]  {t}', t))

    theme_dropdown = widgets.Dropdown(
        options=theme_options,
        value='academic',
        description='Theme:',
        layout=widgets.Layout(width='95%'),
    )

    theme_info = widgets.HTML(
        f'<span style="font-size:11px; color:#666">'
        f'{len(theme_options) - 1} themes available. '
        f'Themes affect global matplotlib settings — '
        f'best applied before generating your plot.</span>'
    )

    theme_preview = widgets.HTML('')

    @_safe
    def _on_theme(change):
        name = change['new']
        try:
            editor.apply_theme(name)
            theme_preview.value = (
                f'<span style="color:#2ECC71; font-size:11px">'
                f'Applied: {name}</span>'
            )
        except Exception as e:
            theme_preview.value = (
                f'<span style="color:#E74C3C; font-size:11px">'
                f'Error: {e}</span>'
            )

    theme_dropdown.observe(_on_theme, names='value')

    theme_tab = widgets.VBox([
        widgets.HTML('<b>Themes</b>'),
        theme_dropdown,
        theme_preview,
        theme_info,
    ])

    # ==================================================================
    # Tab 5: Export & Code
    # ==================================================================
    code_output = widgets.HTML(
        value=_code_to_html('# No edits made'),
        layout=widgets.Layout(width='95%'),
    )
    undo_btn = widgets.Button(
        description='Undo',
        button_style='warning',
        icon='undo',
    )
    redo_btn = widgets.Button(
        description='Redo',
        button_style='info',
        icon='repeat',
    )
    reset_btn = widgets.Button(
        description='Reset All',
        button_style='danger',
        icon='refresh',
    )

    # Save controls
    save_format = widgets.Dropdown(
        options=[
            ('PNG (raster)', 'png'),
            ('SVG (vector)', 'svg'),
            ('PDF (vector)', 'pdf'),
        ],
        value='png',
        description='Format:',
        layout=widgets.Layout(width='95%'),
    )
    save_filename = widgets.Text(
        value='figure.png', description='Filename:',
        layout=widgets.Layout(width='70%'),
    )
    save_dpi = widgets.IntSlider(
        value=300, min=72, max=600, step=50,
        description='Save DPI:', layout=widgets.Layout(width='95%'),
    )
    save_dpi_hint = widgets.HTML(
        '<span style="font-size:11px; color:#888">'
        'Resolution for exported file. '
        'Synced from Figure DPI in Layout tab.</span>'
    )
    save_btn = widgets.Button(
        description='Save Figure',
        button_style='success',
        icon='download',
    )

    def _on_format_change(change):
        ext = change['new']
        fname = save_filename.value
        # Replace extension
        if '.' in fname:
            base = fname.rsplit('.', 1)[0]
        else:
            base = fname
        save_filename.value = f'{base}.{ext}'
        # Hide DPI slider for vector formats
        if ext in ('svg', 'pdf'):
            save_dpi.layout.display = 'none'
            save_dpi_hint.value = (
                '<span style="font-size:11px; color:#888">'
                'Vector format — DPI not applicable for display, '
                'but affects text/marker sizing.</span>'
            )
        else:
            save_dpi.layout.display = ''
            save_dpi_hint.value = (
                '<span style="font-size:11px; color:#888">'
                'Resolution for exported file. '
                'Synced from Figure DPI in Layout tab.</span>'
            )

    save_format.observe(_on_format_change, names='value')

    edit_summary = widgets.HTML('')

    def _on_undo(btn):
        editor.undo()
        _live_refresh(fig)  # force preview update
        edit_summary.value = (
            f'<span style="color:#F39C12">'
            f'Undone. {len(editor.edits)} edit(s) remaining, '
            f'{len(editor._redo_stack)} redo available</span>'
        )

    def _on_redo(btn):
        editor.redo()
        _live_refresh(fig)  # force preview update
        edit_summary.value = (
            f'<span style="color:#3498DB">'
            f'Redone. {len(editor.edits)} edit(s), '
            f'{len(editor._redo_stack)} redo remaining</span>'
        )

    def _on_reset(btn):
        editor.reset()
        _live_refresh(fig)  # force preview update
        edit_summary.value = (
            '<span style="color:#E74C3C">Reset to original</span>'
        )
        code_output.value = _code_to_html('# No edits made')
        # Refresh widget values
        ax = _get_ax()
        title_text.value = ax.get_title()
        xlabel_text.value = ax.get_xlabel()
        ylabel_text.value = ax.get_ylabel()

    def _on_save(btn):
        fname = save_filename.value
        dpi = save_dpi.value
        fmt = save_format.value
        editor.save(fname, dpi=dpi, fmt=fmt)
        fmt_label = fmt.upper()
        edit_summary.value = (
            f'<span style="color:#2ECC71">'
            f'Saved to {fname} ({fmt_label}, {dpi} DPI)</span>'
        )

    undo_btn.on_click(_on_undo)
    redo_btn.on_click(_on_redo)
    reset_btn.on_click(_on_reset)
    save_btn.on_click(_on_save)

    # ---- Edit history panel ----
    history_html = widgets.HTML(
        '<span style="font-size:11px; color:#999">No edits yet</span>'
    )

    def _render_history():
        """Render the edit history as an HTML table."""
        if not editor.edits:
            history_html.value = (
                '<span style="font-size:11px; color:#999">'
                'No edits yet</span>'
            )
            return
        rows = []
        for i, e in enumerate(editor.edits, 1):
            old_str = repr(e.old_value) if e.old_value is not None else '-'
            new_str = repr(e.new_value)
            # Truncate long values
            if len(old_str) > 30:
                old_str = old_str[:27] + '...'
            if len(new_str) > 30:
                new_str = new_str[:27] + '...'
            rows.append(
                f'<tr style="font-size:11px">'
                f'<td style="padding:2px 4px; color:#666">{i}</td>'
                f'<td style="padding:2px 4px">{e.target_desc}</td>'
                f'<td style="padding:2px 4px">{e.property_name}</td>'
                f'<td style="padding:2px 4px; color:#999">{old_str}</td>'
                f'<td style="padding:2px 4px; color:#2C3E50">{new_str}</td>'
                f'</tr>'
            )
        table = (
            '<div style="max-height:200px; overflow-y:auto; '
            'border:1px solid #eee; border-radius:4px">'
            '<table style="width:100%; border-collapse:collapse">'
            '<tr style="background:#f8f8f8; font-size:11px; '
            'font-weight:bold">'
            '<th style="padding:3px 4px">#</th>'
            '<th style="padding:3px 4px">Target</th>'
            '<th style="padding:3px 4px">Property</th>'
            '<th style="padding:3px 4px">Old</th>'
            '<th style="padding:3px 4px">New</th>'
            '</tr>'
            + ''.join(rows)
            + '</table></div>'
        )
        history_html.value = table

    def _live_history_update(figure):
        try:
            _render_history()
        except Exception:
            logger.debug("History update failed", exc_info=True)

    editor.on_refresh(_live_history_update)

    history_section = widgets.Accordion(
        children=[history_html],
    )
    history_section.set_title(0, 'Edit History')
    history_section.selected_index = None  # collapsed by default

    code_tab = widgets.VBox([
        widgets.HTML('<b>Export & Code</b>'),
        save_format,
        widgets.HBox([save_filename, save_btn]),
        save_dpi, save_dpi_hint,
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<b>Reproducible Code</b>'),
        widgets.HTML(
            '<span style="font-size:11px; color:#888">'
            'Code updates automatically as you edit. '
            'Select text below to copy.</span>'
        ),
        edit_summary,
        code_output,
        widgets.HBox([undo_btn, redo_btn, reset_btn]),
        widgets.HTML('<hr style="margin:4px 0">'),
        history_section,
    ])

    # ---- Live code auto-update ----
    def _live_code_update(figure):
        """Auto-update the code display on every edit."""
        try:
            code = editor.generate_code()
            code_output.value = _code_to_html(code)
            edit_summary.value = (
                f'<span style="color:#2ECC71">'
                f'{len(editor.edits)} edit(s) — code updated</span>'
            )
        except Exception:
            logger.debug("Code auto-update failed", exc_info=True)

    editor.on_refresh(_live_code_update)

    # ==================================================================
    # Assemble tabs (with lazy loading for non-default tabs)
    # ==================================================================
    _loading_msg = widgets.HTML(
        '<div style="padding:20px; color:#999; text-align:center">'
        'Loading...</div>'
    )
    _lazy_placeholders = {
        1: widgets.VBox([_loading_msg]),
        2: widgets.VBox([widgets.HTML(
            '<div style="padding:20px; color:#999; text-align:center">'
            'Loading...</div>')]),
        3: widgets.VBox([widgets.HTML(
            '<div style="padding:20px; color:#999; text-align:center">'
            'Loading...</div>')]),
        4: widgets.VBox([widgets.HTML(
            '<div style="padding:20px; color:#999; text-align:center">'
            'Loading...</div>')]),
    }
    _tab_loaded = {0: True, 1: True, 2: False, 3: False, 4: False}
    _real_tabs = {
        2: style_container,
        3: layout_tab,
        4: code_tab,
    }

    tabs = widgets.Tab(children=[
        theme_tab,
        text_tab,
        _lazy_placeholders[2],
        _lazy_placeholders[3],
        _lazy_placeholders[4],
    ])
    tabs.set_title(0, 'Theme')
    tabs.set_title(1, 'Text')
    tabs.set_title(2, 'Style')
    tabs.set_title(3, 'Layout')
    tabs.set_title(4, 'Export')

    def _on_tab_switch(change):
        idx = change['new']
        if idx is not None and not _tab_loaded.get(idx, True):
            _tab_loaded[idx] = True
            children = list(tabs.children)
            children[idx] = _real_tabs[idx]
            tabs.children = children
            # Build style widgets on first visit
            if idx == 2:
                _rebuild_style_widgets()

    tabs.observe(_on_tab_switch, names='selected_index')

    panel_header = widgets.HBox(
        [
            widgets.HTML(
                '<h3 style="margin:0; color:#2C3E50; white-space:nowrap">'
                'StatsPAI Plot Editor</h3>'
            ),
            render_bar,
        ],
        layout=widgets.Layout(
            justify_content='space-between',
            align_items='center',
            width='100%',
        ),
    )

    panel_scrollable = widgets.VBox([
        tabs,
    ], layout=widgets.Layout(
        flex='1 1 auto',
        overflow_y='auto',
    ))

    panel = widgets.VBox([
        panel_header,
        widgets.HTML(
            '<span style="font-size:11px; color:#E74C3C">'
            'Data locked &nbsp;|&nbsp; '
            'Left: live preview &nbsp;|&nbsp; '
            'Right: edit controls'
            '</span>'
        ),
        panel_scrollable,
        apply_bar,
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='column',
        width='400px',
        min_width='360px',
        padding='8px',
        border='1px solid #ddd',
        border_radius='4px',
        height='650px',
    ))

    # ====== Layout: [Live Preview | Controls] ======
    # Like Stata's Graph Editor: figure on left, properties on right
    full_layout = widgets.HBox(
        [fig_container, panel],
        layout=widgets.Layout(
            align_items='flex-start',
            width='100%',
        ),
    )

    # Close the inline figure to avoid duplicate display
    plt.close(fig)

    display(full_layout)


def _to_hex(color) -> str:
    """Convert any matplotlib color to hex string."""
    import matplotlib.colors as mcolors
    try:
        rgba = mcolors.to_rgba(color)
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255),
        )
    except (ValueError, TypeError):
        return '#000000'
