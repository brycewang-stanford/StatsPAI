"""
Jupyter notebook interactive editor using ipywidgets.

Creates a tabbed control panel beside the matplotlib figure for
real-time cosmetic editing with live preview and code generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interactive import FigureEditor


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

    def _live_refresh(figure):
        """Callback: re-render the figure into the Image widget."""
        try:
            fig_image.value = _render_fig_to_png(figure)
            n = len(editor.edits)
            status_bar.value = (
                f'<span style="font-size:11px; color:#2ECC71">'
                f'{n} edit(s) — preview updated</span>'
            )
        except Exception:
            pass

    # Register the live refresh callback
    editor.on_refresh(_live_refresh)

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

    def _on_ax_change(change):
        """Update all widgets when axis selection changes."""
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
        # Rebuild style tab content
        _rebuild_style_widgets()

    ax_selector.observe(_on_ax_change, names='value')

    def _on_title_change(change):
        editor.set_title(change['new'], ax_index=_get_ax_idx())

    def _on_title_size(change):
        editor.set_fontsize('title', change['new'],
                            ax_index=_get_ax_idx())

    def _on_xlabel_change(change):
        editor.set_xlabel(change['new'], ax_index=_get_ax_idx())

    def _on_xlabel_size(change):
        editor.set_fontsize('xlabel', change['new'],
                            ax_index=_get_ax_idx())

    def _on_ylabel_change(change):
        editor.set_ylabel(change['new'], ax_index=_get_ax_idx())

    def _on_ylabel_size(change):
        editor.set_fontsize('ylabel', change['new'],
                            ax_index=_get_ax_idx())

    def _on_tick_size(change):
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
    from .interactive import FONT_PRESETS, get_font_choices

    # Font preset dropdown (journal/thesis standards)
    preset_options = [('-- Custom --', '')] + [
        (name, name) for name in FONT_PRESETS.keys()
    ]
    font_preset = widgets.Dropdown(
        options=preset_options, value='',
        description='Preset:',
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
            font_preset_info.value = (
                f'<span style="color:#2ECC71; font-size:11px">'
                f'{name}: {preset["fonts"][0]}, '
                f'title {preset["title_size"]}pt, '
                f'label {preset["label_size"]}pt</span>'
            )
            # Sync size sliders
            title_size.value = preset['title_size']
            xlabel_size.value = preset['label_size']
            ylabel_size.value = preset['label_size']
            tick_size.value = preset['tick_size']
        except Exception as e:
            font_preset_info.value = (
                f'<span style="color:#E74C3C; font-size:11px">'
                f'Error: {e}</span>'
            )

    font_preset.observe(_on_font_preset, names='value')

    # Font family dropdown
    font_family = widgets.Dropdown(
        options=[('Serif (衬线)', 'serif'),
                 ('Sans-serif (无衬线)', 'sans-serif'),
                 ('Monospace (等宽)', 'monospace')],
        value='serif',
        description='Family:',
        layout=widgets.Layout(width='95%'),
    )

    # Specific font dropdown (updates based on family selection)
    def _get_font_options(family):
        choices = get_font_choices()
        if family == 'serif':
            fonts = choices['English Serif'] + choices['Chinese (中文)']
        elif family == 'sans-serif':
            fonts = choices['English Sans-serif'] + choices['Chinese (中文)']
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
        widgets.HTML(
            '<span style="font-size:11px; color:#666">'
            'Quick preset (journal/thesis standard):</span>'
        ),
        font_preset,
        font_preset_info,
        widgets.HTML(
            '<span style="font-size:11px; color:#666">'
            'Or choose manually:</span>'
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
            layout=widgets.Layout(width='95%'),
        )

        def _on_tc(change):
            editor.set_color('title', change['new'],
                             ax_index=ax_idx)

        title_color.observe(_on_tc, names='value')
        children.append(title_color)

        # Line controls
        lines = ax.get_lines()
        for i, line in enumerate(lines[:8]):
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
                layout=widgets.Layout(width='95%'),
            )

            def _make_color_cb(idx):
                def _cb(change):
                    editor.set_color(f'line{idx}', change['new'],
                                     ax_index=ax_idx)
                return _cb

            cp.observe(_make_color_cb(i), names='value')
            children.append(cp)

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

        # Scatter / Collection controls
        import matplotlib.collections as mcoll
        for i, coll in enumerate(ax.collections[:6]):
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
                    layout=widgets.Layout(width='95%'),
                )

                def _make_sc_cb(idx):
                    def _cb(change):
                        editor.set_scatter_color(
                            idx, change['new'], ax_index=ax_idx)
                    return _cb

                sc_cp.observe(_make_sc_cb(i), names='value')
                children.append(sc_cp)

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

        style_container.children = children

    _rebuild_style_widgets()

    # ==================================================================
    # Tab 3: Layout (spines, grid, figsize, legend, axes limits)
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
    grid_toggle = widgets.Checkbox(
        value=False, description='Show grid',
    )

    fig_width = widgets.FloatSlider(
        value=fig.get_size_inches()[0], min=3, max=20, step=0.5,
        description='Width:', layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    fig_height = widgets.FloatSlider(
        value=fig.get_size_inches()[1], min=2, max=15, step=0.5,
        description='Height:', layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )

    legend_loc = widgets.Dropdown(
        options=_LEGEND_LOCS, value='best',
        description='Legend loc:',
        layout=widgets.Layout(width='95%'),
    )

    xlim = axes[0].get_xlim()
    ylim = axes[0].get_ylim()
    x_range = max(xlim[1] - xlim[0], 0.01)
    y_range = max(ylim[1] - ylim[0], 0.01)

    xlim_range = widgets.FloatRangeSlider(
        value=[xlim[0], xlim[1]],
        min=xlim[0] - x_range * 0.5,
        max=xlim[1] + x_range * 0.5,
        step=x_range / 50,
        description='X range:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )
    ylim_range = widgets.FloatRangeSlider(
        value=[ylim[0], ylim[1]],
        min=ylim[0] - y_range * 0.5,
        max=ylim[1] + y_range * 0.5,
        step=y_range / 50,
        description='Y range:',
        layout=widgets.Layout(width='95%'),
        continuous_update=False,
    )

    def _on_spine(spine_name):
        def _cb(change):
            editor.set_spine_visible(spine_name, change['new'],
                                     ax_index=_get_ax_idx())
        return _cb

    spine_top.observe(_on_spine('top'), names='value')
    spine_right.observe(_on_spine('right'), names='value')
    spine_bottom.observe(_on_spine('bottom'), names='value')
    spine_left.observe(_on_spine('left'), names='value')

    def _on_grid(change):
        editor.set_grid(change['new'], ax_index=_get_ax_idx())

    grid_toggle.observe(_on_grid, names='value')

    def _on_figsize_w(change):
        editor.set_figsize(change['new'], fig.get_size_inches()[1])

    def _on_figsize_h(change):
        editor.set_figsize(fig.get_size_inches()[0], change['new'])

    fig_width.observe(_on_figsize_w, names='value')
    fig_height.observe(_on_figsize_h, names='value')

    def _on_legend_loc(change):
        editor.set_legend(loc=change['new'], ax_index=_get_ax_idx())

    legend_loc.observe(_on_legend_loc, names='value')

    def _on_xlim(change):
        editor.set_xlim(change['new'][0], change['new'][1],
                        ax_index=_get_ax_idx())

    def _on_ylim(change):
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
        widgets.HTML('<b>Layout & Spines</b>'),
        widgets.HBox([spine_top, spine_right]),
        widgets.HBox([spine_bottom, spine_left]),
        grid_toggle,
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<i>Figure Size</i>'),
        fig_width, fig_height,
        widgets.HTML('<hr style="margin:4px 0">'),
        legend_loc,
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<i>Axis Limits</i>'),
        protect_label,
        xlim_range, ylim_range,
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

    def _on_theme(change):
        name = change['new']
        try:
            editor.apply_theme(name)
            theme_preview.value = (
                f'<span style="color:#2ECC71; font-size:11px">'
                f'Applied: {name}</span>'
            )
        except ValueError as e:
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
    code_output = widgets.Textarea(
        value='# Make edits, then click "Generate Code"',
        layout=widgets.Layout(width='95%', height='250px'),
        disabled=True,
    )
    generate_btn = widgets.Button(
        description='Generate Code',
        button_style='primary',
        icon='code',
    )
    undo_btn = widgets.Button(
        description='Undo',
        button_style='warning',
        icon='undo',
    )
    reset_btn = widgets.Button(
        description='Reset All',
        button_style='danger',
        icon='refresh',
    )

    # Save controls
    save_filename = widgets.Text(
        value='figure.png', description='Filename:',
        layout=widgets.Layout(width='70%'),
    )
    save_dpi = widgets.IntSlider(
        value=300, min=72, max=600, step=50,
        description='DPI:', layout=widgets.Layout(width='95%'),
    )
    save_btn = widgets.Button(
        description='Save Figure',
        button_style='success',
        icon='download',
    )

    edit_summary = widgets.HTML('')

    def _on_generate(btn):
        code = editor.generate_code()
        code_output.value = code
        edit_summary.value = (
            f'<span style="color:#2ECC71">'
            f'{len(editor.edits)} edit(s) recorded</span>'
        )

    def _on_undo(btn):
        editor.undo()
        _live_refresh(fig)  # force preview update
        edit_summary.value = (
            f'<span style="color:#F39C12">'
            f'Undone. {len(editor.edits)} edit(s) remaining</span>'
        )

    def _on_reset(btn):
        editor.reset()
        _live_refresh(fig)  # force preview update
        edit_summary.value = (
            '<span style="color:#E74C3C">Reset to original</span>'
        )
        # Refresh widget values
        ax = _get_ax()
        title_text.value = ax.get_title()
        xlabel_text.value = ax.get_xlabel()
        ylabel_text.value = ax.get_ylabel()

    def _on_save(btn):
        fname = save_filename.value
        dpi = save_dpi.value
        editor.save(fname, dpi=dpi)
        edit_summary.value = (
            f'<span style="color:#2ECC71">'
            f'Saved to {fname} ({dpi} DPI)</span>'
        )

    generate_btn.on_click(_on_generate)
    undo_btn.on_click(_on_undo)
    reset_btn.on_click(_on_reset)
    save_btn.on_click(_on_save)

    code_tab = widgets.VBox([
        widgets.HTML('<b>Export & Code</b>'),
        widgets.HBox([save_filename, save_btn]),
        save_dpi,
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HTML('<b>Reproducible Code</b>'),
        widgets.HBox([generate_btn, undo_btn, reset_btn]),
        edit_summary,
        code_output,
        widgets.HTML(
            '<span style="font-size:11px; color:#666">'
            'Paste this code after your plot command to reproduce '
            'all edits.</span>'
        ),
    ])

    # ---- Live code auto-update ----
    def _live_code_update(figure):
        """Auto-update the code textarea on every edit."""
        try:
            code_output.value = editor.generate_code()
            edit_summary.value = (
                f'<span style="color:#2ECC71">'
                f'{len(editor.edits)} edit(s) — code updated</span>'
            )
        except Exception:
            pass

    editor.on_refresh(_live_code_update)

    # ==================================================================
    # Assemble tabs
    # ==================================================================
    tabs = widgets.Tab(children=[
        text_tab, style_container, layout_tab, theme_tab, code_tab,
    ])
    tabs.set_title(0, 'Text')
    tabs.set_title(1, 'Style')
    tabs.set_title(2, 'Layout')
    tabs.set_title(3, 'Theme')
    tabs.set_title(4, 'Export')

    panel = widgets.VBox([
        widgets.HTML(
            '<h3 style="margin:0 0 8px 0; color:#2C3E50">'
            'StatsPAI Plot Editor</h3>'
            '<span style="font-size:11px; color:#E74C3C">'
            'Data locked &nbsp;|&nbsp; '
            'Left: live preview &nbsp;|&nbsp; '
            'Right: edit controls'
            '</span>'
        ),
        tabs,
    ], layout=widgets.Layout(
        width='400px',
        min_width='360px',
        padding='8px',
        border='1px solid #ddd',
        border_radius='4px',
        max_height='650px',
        overflow_y='auto',
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
