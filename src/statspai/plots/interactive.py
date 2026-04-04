"""
Interactive plot editor for StatsPAI.

Provides cosmetic editing of matplotlib figures while protecting
statistical data integrity. All modifications generate reproducible code.

Design Principle
----------------
**Cosmetic editable, data locked.** Users can adjust titles, colors,
fonts, layout, and annotations — but NEVER data points, regression lines,
confidence intervals, or any element that represents statistical results.

Usage
-----
>>> import statspai as sp
>>> result = sp.did(df, y='wage', treat='treated', time='post')
>>> fig, ax = result.event_study_plot()
>>> sp.interactive(fig)          # Opens editor
>>> sp.get_code(fig)             # Get reproducible code string

In Jupyter notebooks, ``sp.interactive(fig)`` shows an ipywidgets
control panel beside the figure. In scripts, it opens a matplotlib
GUI with editing controls.
"""

from __future__ import annotations

import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto


class ArtistRole(Enum):
    """Classification of matplotlib artists by editability."""
    DATA = auto()        # Statistical data — LOCKED
    FIT = auto()         # Regression/fit lines — LOCKED
    CI = auto()          # Confidence intervals — LOCKED
    REFERENCE = auto()   # Reference lines (y=0, cutoff) — style only
    LABEL = auto()       # Titles, axis labels — fully editable
    ANNOTATION = auto()  # User annotations — fully editable
    LEGEND = auto()      # Legend — position/style editable
    SPINE = auto()       # Axis spines — visibility editable
    COSMETIC = auto()    # Purely cosmetic elements — fully editable


@dataclass
class EditRecord:
    """Single tracked modification to a figure element."""
    target_desc: str     # e.g. "ax.title", "ax.xaxis.label"
    property_name: str   # e.g. "text", "fontsize", "color"
    old_value: Any
    new_value: Any
    code_line: str       # Reproducible matplotlib code


@dataclass
class FigureEditor:
    """
    Core editor that wraps a matplotlib figure and tracks modifications.

    Classifies every artist as DATA (locked) or COSMETIC (editable),
    records all edits, and generates reproducible code.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to edit.
    protect_data : bool, default True
        If True (default), all data-representing artists are locked.
        Setting to False is strongly discouraged for statistical plots.

    Attributes
    ----------
    edits : list of EditRecord
        History of all modifications.
    artist_roles : dict
        Mapping from artist id to ArtistRole.
    """
    fig: Any
    protect_data: bool = True
    edits: List[EditRecord] = field(default_factory=list)
    artist_roles: Dict[int, ArtistRole] = field(default_factory=dict)
    _original_state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._classify_artists()
        self._snapshot_state()

    # ------------------------------------------------------------------
    # Artist classification
    # ------------------------------------------------------------------

    def _classify_artists(self):
        """Classify all artists in the figure by their role."""
        import matplotlib.collections as mcoll

        for ax in self.fig.get_axes():
            # Title and labels -> LABEL
            self.artist_roles[id(ax.title)] = ArtistRole.LABEL
            self.artist_roles[id(ax.xaxis.label)] = ArtistRole.LABEL
            self.artist_roles[id(ax.yaxis.label)] = ArtistRole.LABEL

            # Spines -> SPINE
            for spine in ax.spines.values():
                self.artist_roles[id(spine)] = ArtistRole.SPINE

            # Legend -> LEGEND
            legend = ax.get_legend()
            if legend is not None:
                self.artist_roles[id(legend)] = ArtistRole.LEGEND

            # Classify lines
            for line in ax.get_lines():
                role = self._classify_line(line, ax)
                self.artist_roles[id(line)] = role

            # Classify collections (scatter, fill_between, etc.)
            for coll in ax.collections:
                role = self._classify_collection(coll, ax)
                self.artist_roles[id(coll)] = role

            # Classify text artists (annotations, etc.)
            for text in ax.texts:
                self.artist_roles[id(text)] = ArtistRole.ANNOTATION

            # Classify patches
            for patch in ax.patches:
                self.artist_roles[id(patch)] = ArtistRole.COSMETIC

        # Figure-level suptitle
        if self.fig._suptitle is not None:
            self.artist_roles[id(self.fig._suptitle)] = ArtistRole.LABEL

    def _classify_line(self, line, ax) -> ArtistRole:
        """
        Classify a Line2D artist using label hints, then heuristics.

        Priority: explicit label hint > line properties > point count.
        """
        label = (line.get_label() or '').lower()
        xdata = line.get_xdata()
        ydata = line.get_ydata()

        # 1. Label-based hints (most reliable)
        ref_keywords = ('reference', 'zero', 'cutoff', 'onset',
                        'treatment', 'threshold', 'baseline')
        fit_keywords = ('fit', 'trend', 'regression', 'polynomial',
                        'predicted', 'fitted', 'smoothed')
        ci_keywords = ('ci', 'confidence', 'bound', 'interval',
                       'upper', 'lower', 'band')
        if any(kw in label for kw in ref_keywords):
            return ArtistRole.REFERENCE
        if any(kw in label for kw in fit_keywords):
            return ArtistRole.FIT
        if any(kw in label for kw in ci_keywords):
            return ArtistRole.CI

        # 2. Reference lines: constant x or y (horizontal/vertical)
        if len(xdata) >= 2 and len(set(ydata)) == 1:
            return ArtistRole.REFERENCE
        if len(ydata) >= 2 and len(set(xdata)) == 1:
            return ArtistRole.REFERENCE

        # 3. Matplotlib internal artists (label starts with '_')
        if label.startswith('_'):
            # Connector lines from errorbar, etc.
            ls = line.get_linestyle()
            if ls in ('--', ':', '-.', 'None', 'none', ''):
                return ArtistRole.REFERENCE
            return ArtistRole.DATA

        # 4. Dashed/dotted with constant-ish values -> REFERENCE
        ls = line.get_linestyle()
        if ls in ('--', ':', '-.'):
            import numpy as np
            y_arr = np.asarray(ydata, dtype=float)
            if np.ptp(y_arr[np.isfinite(y_arr)]) < 1e-10:
                return ArtistRole.REFERENCE

        # 5. Default: treat as DATA (safe — locks it)
        return ArtistRole.DATA

    def _classify_collection(self, coll, ax) -> ArtistRole:
        """Classify a Collection artist (scatter, fill_between, etc.)."""
        import matplotlib.collections as mcoll

        # PolyCollection (fill_between) -> CI
        if isinstance(coll, mcoll.PolyCollection):
            return ArtistRole.CI

        # PathCollection (scatter) -> DATA
        if isinstance(coll, mcoll.PathCollection):
            return ArtistRole.DATA

        # LineCollection (error bars) -> CI
        if isinstance(coll, mcoll.LineCollection):
            return ArtistRole.CI

        return ArtistRole.COSMETIC

    def is_editable(self, artist) -> bool:
        """Check if an artist's data/position can be modified."""
        role = self.artist_roles.get(id(artist), ArtistRole.COSMETIC)
        if not self.protect_data:
            return True
        return role not in (ArtistRole.DATA, ArtistRole.FIT, ArtistRole.CI)

    def is_style_editable(self, artist) -> bool:
        """Check if an artist's visual style can be modified."""
        # Everything's style is editable EXCEPT data positions
        # Even DATA elements can have color/style changes
        return True

    def get_editable_properties(self, artist) -> Dict[str, List[str]]:
        """Get editable properties for an artist, grouped by category."""
        role = self.artist_roles.get(id(artist), ArtistRole.COSMETIC)
        props = {}

        # Style properties -- always available
        props['style'] = []
        if hasattr(artist, 'set_color'):
            props['style'].append('color')
        if hasattr(artist, 'set_linewidth'):
            props['style'].append('linewidth')
        if hasattr(artist, 'set_linestyle'):
            props['style'].append('linestyle')
        if hasattr(artist, 'set_alpha'):
            props['style'].append('alpha')
        if hasattr(artist, 'set_markersize'):
            props['style'].append('markersize')
        if hasattr(artist, 'set_marker'):
            props['style'].append('marker')

        # Text properties -- for text artists
        if hasattr(artist, 'set_text'):
            props['text'] = ['text', 'fontsize', 'fontweight',
                             'fontstyle', 'fontfamily', 'color',
                             'rotation', 'ha', 'va']

        # Position -- only for non-data elements
        if role not in (ArtistRole.DATA, ArtistRole.FIT, ArtistRole.CI):
            if hasattr(artist, 'set_position'):
                props['position'] = ['position']

        return props

    # ------------------------------------------------------------------
    # State snapshot (for diffing / undo)
    # ------------------------------------------------------------------

    def _snapshot_state(self):
        """Capture the initial state of all editable properties."""
        state = {}
        for i, ax in enumerate(self.fig.get_axes()):
            prefix = f'ax{i}' if i > 0 else 'ax'
            state[f'{prefix}.title.text'] = ax.get_title()
            state[f'{prefix}.title.fontsize'] = ax.title.get_fontsize()
            state[f'{prefix}.xlabel.text'] = ax.get_xlabel()
            state[f'{prefix}.xlabel.fontsize'] = ax.xaxis.label.get_fontsize()
            state[f'{prefix}.ylabel.text'] = ax.get_ylabel()
            state[f'{prefix}.ylabel.fontsize'] = ax.yaxis.label.get_fontsize()
            state[f'{prefix}.xlim'] = ax.get_xlim()
            state[f'{prefix}.ylim'] = ax.get_ylim()

            for spine_name, spine in ax.spines.items():
                state[f'{prefix}.spines.{spine_name}.visible'] = (
                    spine.get_visible()
                )

            gridlines = ax.xaxis.get_gridlines()
            state[f'{prefix}.grid'] = (
                gridlines[0].get_visible() if gridlines else False
            )

        state['fig.figsize'] = tuple(self.fig.get_size_inches())
        state['fig.dpi'] = self.fig.dpi
        if self.fig._suptitle:
            state['fig.suptitle.text'] = self.fig._suptitle.get_text()
            state['fig.suptitle.fontsize'] = (
                self.fig._suptitle.get_fontsize()
            )

        self._original_state = state

    # ------------------------------------------------------------------
    # Edit operations (all record history)
    # ------------------------------------------------------------------

    def set_title(self, text: str, ax_index: int = 0, **kwargs):
        """Set axis title with tracking."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        old = ax.get_title()
        ax.set_title(text, **kwargs)
        code = f"{prefix}.set_title({text!r}"
        if kwargs:
            code += ', ' + ', '.join(
                f'{k}={v!r}' for k, v in kwargs.items()
            )
        code += ')'
        self.edits.append(EditRecord(
            f'{prefix}.title', 'text', old, text, code))
        self._refresh()

    def set_xlabel(self, text: str, ax_index: int = 0, **kwargs):
        """Set x-axis label with tracking."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        old = ax.get_xlabel()
        ax.set_xlabel(text, **kwargs)
        code = f"{prefix}.set_xlabel({text!r}"
        if kwargs:
            code += ', ' + ', '.join(
                f'{k}={v!r}' for k, v in kwargs.items()
            )
        code += ')'
        self.edits.append(EditRecord(
            f'{prefix}.xlabel', 'text', old, text, code))
        self._refresh()

    def set_ylabel(self, text: str, ax_index: int = 0, **kwargs):
        """Set y-axis label with tracking."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        old = ax.get_ylabel()
        ax.set_ylabel(text, **kwargs)
        code = f"{prefix}.set_ylabel({text!r}"
        if kwargs:
            code += ', ' + ', '.join(
                f'{k}={v!r}' for k, v in kwargs.items()
            )
        code += ')'
        self.edits.append(EditRecord(
            f'{prefix}.ylabel', 'text', old, text, code))
        self._refresh()

    def set_xlim(self, left: float, right: float, ax_index: int = 0):
        """Set x-axis limits with tracking and data protection."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'

        if self.protect_data:
            data_min, data_max = self._get_data_range(ax, 'x')
            if data_min is not None:
                if left > data_min or right < data_max:
                    import warnings
                    warnings.warn(
                        f"Axis limits [{left:.2f}, {right:.2f}] would hide "
                        f"data points in [{data_min:.2f}, {data_max:.2f}]. "
                        f"Expanding to include all data.",
                        UserWarning, stacklevel=2,
                    )
                    left = min(left, data_min)
                    right = max(right, data_max)

        old = ax.get_xlim()
        ax.set_xlim(left, right)
        self.edits.append(EditRecord(
            f'{prefix}.xlim', 'range', old, (left, right),
            f"{prefix}.set_xlim({left}, {right})"))
        self._refresh()

    def set_ylim(self, bottom: float, top: float, ax_index: int = 0):
        """Set y-axis limits with tracking and data protection."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'

        if self.protect_data:
            data_min, data_max = self._get_data_range(ax, 'y')
            if data_min is not None:
                if bottom > data_min or top < data_max:
                    import warnings
                    warnings.warn(
                        f"Axis limits [{bottom:.2f}, {top:.2f}] would hide "
                        f"data points in [{data_min:.2f}, {data_max:.2f}]. "
                        f"Expanding to include all data.",
                        UserWarning, stacklevel=2,
                    )
                    bottom = min(bottom, data_min)
                    top = max(top, data_max)

        old = ax.get_ylim()
        ax.set_ylim(bottom, top)
        self.edits.append(EditRecord(
            f'{prefix}.ylim', 'range', old, (bottom, top),
            f"{prefix}.set_ylim({bottom}, {top})"))
        self._refresh()

    def set_fontsize(self, target: str, size: float, ax_index: int = 0):
        """Set font size for a named target."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'

        if target == 'title':
            old = ax.title.get_fontsize()
            ax.title.set_fontsize(size)
            code = f"{prefix}.title.set_fontsize({size})"
        elif target == 'xlabel':
            old = ax.xaxis.label.get_fontsize()
            ax.xaxis.label.set_fontsize(size)
            code = f"{prefix}.xaxis.label.set_fontsize({size})"
        elif target == 'ylabel':
            old = ax.yaxis.label.get_fontsize()
            ax.yaxis.label.set_fontsize(size)
            code = f"{prefix}.yaxis.label.set_fontsize({size})"
        elif target == 'ticks':
            old_labels = ax.xaxis.get_ticklabels()
            old = old_labels[0].get_fontsize() if old_labels else 10
            ax.tick_params(labelsize=size)
            code = f"{prefix}.tick_params(labelsize={size})"
        else:
            raise ValueError(f"Unknown target: {target}")

        self.edits.append(EditRecord(
            f'{prefix}.{target}', 'fontsize', old, size, code))
        self._refresh()

    def set_color(self, target: str, color: str, ax_index: int = 0):
        """Set color for a named target."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'

        if target == 'title':
            old = ax.title.get_color()
            ax.title.set_color(color)
            code = f"{prefix}.title.set_color({color!r})"
        elif target == 'xlabel':
            old = ax.xaxis.label.get_color()
            ax.xaxis.label.set_color(color)
            code = f"{prefix}.xaxis.label.set_color({color!r})"
        elif target == 'ylabel':
            old = ax.yaxis.label.get_color()
            ax.yaxis.label.set_color(color)
            code = f"{prefix}.yaxis.label.set_color({color!r})"
        elif target.startswith('line'):
            idx = int(target.replace('line', ''))
            line = ax.get_lines()[idx]
            old = line.get_color()
            line.set_color(color)
            code = f"{prefix}.get_lines()[{idx}].set_color({color!r})"
        else:
            raise ValueError(f"Unknown target: {target}")

        self.edits.append(EditRecord(
            f'{prefix}.{target}', 'color', old, color, code))
        self._refresh()

    def set_spine_visible(self, spine_name: str, visible: bool,
                          ax_index: int = 0):
        """Toggle spine visibility."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        old = ax.spines[spine_name].get_visible()
        ax.spines[spine_name].set_visible(visible)
        self.edits.append(EditRecord(
            f'{prefix}.spines.{spine_name}', 'visible', old, visible,
            f"{prefix}.spines[{spine_name!r}].set_visible({visible})"))
        self._refresh()

    def set_grid(self, visible: bool, ax_index: int = 0, **kwargs):
        """Toggle grid with tracking."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        ax.grid(visible, **kwargs)
        code = f"{prefix}.grid({visible}"
        if kwargs:
            code += ', ' + ', '.join(
                f'{k}={v!r}' for k, v in kwargs.items()
            )
        code += ')'
        self.edits.append(EditRecord(
            f'{prefix}.grid', 'visible', not visible, visible, code))
        self._refresh()

    def set_figsize(self, width: float, height: float):
        """Set figure size with tracking."""
        old = tuple(self.fig.get_size_inches())
        self.fig.set_size_inches(width, height)
        self.edits.append(EditRecord(
            'fig', 'figsize', old, (width, height),
            f"fig.set_size_inches({width}, {height})"))
        self._refresh()

    def set_legend(self, ax_index: int = 0, **kwargs):
        """Update legend properties."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        legend = ax.get_legend()
        if legend is None:
            return

        if 'loc' in kwargs:
            legend.set_loc(kwargs['loc'])
        if 'fontsize' in kwargs:
            for text in legend.get_texts():
                text.set_fontsize(kwargs['fontsize'])
        if 'frameon' in kwargs:
            legend.set_frame_on(kwargs['frameon'])

        code = f"{prefix}.legend(" + ', '.join(
            f'{k}={v!r}' for k, v in kwargs.items()) + ')'
        self.edits.append(EditRecord(
            f'{prefix}.legend', 'properties', None, kwargs, code))
        self._refresh()

    def set_linewidth(self, line_index: int, width: float,
                      ax_index: int = 0):
        """Set line width with tracking."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        line = ax.get_lines()[line_index]
        old = line.get_linewidth()
        line.set_linewidth(width)
        code = f"{prefix}.get_lines()[{line_index}].set_linewidth({width})"
        self.edits.append(EditRecord(
            f'{prefix}.line{line_index}', 'linewidth', old, width, code))
        self._refresh()

    def set_linestyle(self, line_index: int, style: str,
                      ax_index: int = 0):
        """Set line style with tracking."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        line = ax.get_lines()[line_index]
        old = line.get_linestyle()
        line.set_linestyle(style)
        code = (f"{prefix}.get_lines()[{line_index}]"
                f".set_linestyle({style!r})")
        self.edits.append(EditRecord(
            f'{prefix}.line{line_index}', 'linestyle', old, style, code))
        self._refresh()

    def set_alpha(self, target: str, alpha: float, ax_index: int = 0):
        """Set alpha (transparency) for a target element."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'

        if target.startswith('line'):
            idx = int(target.replace('line', ''))
            artist = ax.get_lines()[idx]
            old = artist.get_alpha()
            artist.set_alpha(alpha)
            code = (f"{prefix}.get_lines()[{idx}]"
                    f".set_alpha({alpha})")
        elif target.startswith('scatter'):
            idx = int(target.replace('scatter', ''))
            artist = ax.collections[idx]
            old = artist.get_alpha()
            artist.set_alpha(alpha)
            code = (f"{prefix}.collections[{idx}]"
                    f".set_alpha({alpha})")
        elif target.startswith('ci'):
            idx = int(target.replace('ci', ''))
            artist = ax.collections[idx]
            old = artist.get_alpha()
            artist.set_alpha(alpha)
            code = (f"{prefix}.collections[{idx}]"
                    f".set_alpha({alpha})")
        else:
            raise ValueError(f"Unknown target: {target}")

        self.edits.append(EditRecord(
            f'{prefix}.{target}', 'alpha', old, alpha, code))
        self._refresh()

    def set_scatter_color(self, scatter_index: int, color: str,
                          ax_index: int = 0):
        """Set scatter (PathCollection) facecolor with tracking."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        coll = ax.collections[scatter_index]
        old = coll.get_facecolors()
        coll.set_facecolors(color)
        code = (f"{prefix}.collections[{scatter_index}]"
                f".set_facecolors({color!r})")
        self.edits.append(EditRecord(
            f'{prefix}.scatter{scatter_index}', 'color',
            str(old), color, code))
        self._refresh()

    def set_marker(self, line_index: int, marker: str,
                   ax_index: int = 0):
        """Set line marker with tracking."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        line = ax.get_lines()[line_index]
        old = line.get_marker()
        line.set_marker(marker)
        code = (f"{prefix}.get_lines()[{line_index}]"
                f".set_marker({marker!r})")
        self.edits.append(EditRecord(
            f'{prefix}.line{line_index}', 'marker', old, marker, code))
        self._refresh()

    def save(self, filename: str, dpi: int = 300, **kwargs):
        """Save the figure to a file with tracking."""
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
        code = (f"fig.savefig({filename!r}, dpi={dpi}, "
                f"bbox_inches='tight')")
        self.edits.append(EditRecord(
            'fig', 'save', None, filename, code))

    def add_annotation(self, text: str, xy: Tuple[float, float],
                       ax_index: int = 0, **kwargs):
        """Add a text annotation (non-data element)."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        ax.annotate(text, xy=xy, **kwargs)
        code = f"{prefix}.annotate({text!r}, xy={xy}"
        if kwargs:
            code += ', ' + ', '.join(
                f'{k}={v!r}' for k, v in kwargs.items()
            )
        code += ')'
        self.edits.append(EditRecord(
            f'{prefix}.annotation', 'add', None, text, code))
        self._refresh()

    def apply_theme(self, theme_name: str):
        """Apply a StatsPAI theme and record it."""
        from .themes import set_theme, _THEMES
        if theme_name not in _THEMES and theme_name != 'default':
            raise ValueError(f"Unknown theme: {theme_name}")
        set_theme(theme_name)
        self.fig.canvas.draw_idle()
        self.edits.append(EditRecord(
            'theme', 'name', None, theme_name,
            f"sp.set_theme({theme_name!r})"))

    def undo(self):
        """Undo the last edit by restoring original state and replaying."""
        if not self.edits:
            return
        self.edits.pop()
        self._restore_original()
        edits_to_replay = self.edits.copy()
        self.edits.clear()
        for e in edits_to_replay:
            self._replay_edit(e)
            self.edits.append(e)
        self._refresh()

    def reset(self):
        """Reset all edits back to original state."""
        self.edits.clear()
        self._restore_original()
        self._refresh()

    def _replay_edit(self, edit: EditRecord):
        """Replay a single edit using safe method dispatch (no exec)."""
        td = edit.target_desc
        prop = edit.property_name
        val = edit.new_value
        axes = self.fig.get_axes()

        # Parse axis index from target_desc like "ax.title" or "ax1.xlabel"
        ax_idx = 0
        if td.startswith('ax') and len(td) > 2 and td[2].isdigit():
            ax_idx = int(td[2])
        ax = axes[ax_idx] if ax_idx < len(axes) else axes[0]

        try:
            if 'title' in td and prop == 'text':
                ax.set_title(val)
            elif 'xlabel' in td and prop == 'text':
                ax.set_xlabel(val)
            elif 'ylabel' in td and prop == 'text':
                ax.set_ylabel(val)
            elif 'title' in td and prop == 'fontsize':
                ax.title.set_fontsize(val)
            elif 'xlabel' in td and prop == 'fontsize':
                ax.xaxis.label.set_fontsize(val)
            elif 'ylabel' in td and prop == 'fontsize':
                ax.yaxis.label.set_fontsize(val)
            elif 'ticks' in td and prop == 'fontsize':
                ax.tick_params(labelsize=val)
            elif 'xlim' in td:
                ax.set_xlim(val)
            elif 'ylim' in td:
                ax.set_ylim(val)
            elif 'spines' in td and prop == 'visible':
                spine_name = td.split('.')[-1]
                ax.spines[spine_name].set_visible(val)
            elif 'grid' in td:
                ax.grid(val)
            elif td == 'fig' and prop == 'figsize':
                self.fig.set_size_inches(val)
            elif 'color' in prop and 'title' in td:
                ax.title.set_color(val)
            elif 'color' in prop and 'line' in td:
                idx = int(td.split('line')[-1]) if 'line' in td else 0
                ax.get_lines()[idx].set_color(val)
            elif td == 'theme':
                from .themes import set_theme
                set_theme(val)
            elif 'legend' in td:
                legend = ax.get_legend()
                if legend and isinstance(val, dict):
                    if 'loc' in val:
                        legend.set_loc(val['loc'])
                    if 'frameon' in val:
                        legend.set_frame_on(val['frameon'])
        except Exception:
            pass  # Skip edits that can't be replayed

    # ------------------------------------------------------------------
    # Data protection helpers
    # ------------------------------------------------------------------

    def _get_data_range(self, ax, axis: str
                        ) -> Tuple[Optional[float], Optional[float]]:
        """Get the min/max of all locked (DATA/FIT/CI) artists on an axis."""
        import numpy as np
        locked_roles = (ArtistRole.DATA, ArtistRole.FIT, ArtistRole.CI)
        vals = []

        for line in ax.get_lines():
            role = self.artist_roles.get(id(line), ArtistRole.COSMETIC)
            if role in locked_roles:
                data = (line.get_xdata() if axis == 'x'
                        else line.get_ydata())
                finite = np.asarray(data, dtype=float)
                vals.extend(finite[np.isfinite(finite)])

        for coll in ax.collections:
            role = self.artist_roles.get(id(coll), ArtistRole.COSMETIC)
            if role in locked_roles:
                offsets = coll.get_offsets()
                if len(offsets) > 0:
                    idx = 0 if axis == 'x' else 1
                    col = np.asarray(offsets[:, idx], dtype=float)
                    vals.extend(col[np.isfinite(col)])

        if not vals:
            return None, None
        return float(np.nanmin(vals)), float(np.nanmax(vals))

    def _restore_original(self):
        """Restore figure to its original state (all snapshotted props)."""
        s = self._original_state
        for i, ax in enumerate(self.fig.get_axes()):
            prefix = f'ax{i}' if i > 0 else 'ax'
            if f'{prefix}.title.text' in s:
                ax.set_title(s[f'{prefix}.title.text'])
                ax.title.set_fontsize(s[f'{prefix}.title.fontsize'])
            if f'{prefix}.xlabel.text' in s:
                ax.set_xlabel(s[f'{prefix}.xlabel.text'])
                ax.xaxis.label.set_fontsize(
                    s[f'{prefix}.xlabel.fontsize'])
            if f'{prefix}.ylabel.text' in s:
                ax.set_ylabel(s[f'{prefix}.ylabel.text'])
                ax.yaxis.label.set_fontsize(
                    s[f'{prefix}.ylabel.fontsize'])
            if f'{prefix}.xlim' in s:
                ax.set_xlim(s[f'{prefix}.xlim'])
            if f'{prefix}.ylim' in s:
                ax.set_ylim(s[f'{prefix}.ylim'])
            if f'{prefix}.grid' in s:
                ax.grid(s[f'{prefix}.grid'])
            for spine_name in ('top', 'right', 'bottom', 'left'):
                key = f'{prefix}.spines.{spine_name}.visible'
                if key in s:
                    ax.spines[spine_name].set_visible(s[key])

        if 'fig.figsize' in s:
            self.fig.set_size_inches(s['fig.figsize'])

    def _refresh(self):
        """Redraw the figure canvas."""
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def generate_code(self, include_comment: bool = True) -> str:
        """
        Generate reproducible matplotlib code for all edits.

        Returns
        -------
        str
            Python code that reproduces all cosmetic modifications.
            Paste after your original plot code to apply the same edits.
        """
        if not self.edits:
            return "# No edits made"

        lines = []
        if include_comment:
            lines.append(
                "# --- StatsPAI interactive edits "
                "(paste after your plot code) ---"
            )

        # Deduplicate: keep last edit per (target, property)
        seen = {}
        for edit in self.edits:
            key = (edit.target_desc, edit.property_name)
            seen[key] = edit

        # Add import if theme was used
        needs_sp = any(e.target_desc == 'theme' for e in seen.values())
        if needs_sp:
            lines.append("import statspai as sp")

        for edit in seen.values():
            if edit.property_name == 'save':
                continue  # Don't include save in reproducible edits
            lines.append(edit.code_line)

        if include_comment:
            lines.append("fig.tight_layout()")
            lines.append("# --- end StatsPAI edits ---")

        return '\n'.join(lines)

    def copy_code(self):
        """Print reproducible code to stdout."""
        code = self.generate_code()
        print(code)

    def summary(self) -> str:
        """Return a summary of all edits."""
        if not self.edits:
            return "No edits made."
        lines = [f"StatsPAI Interactive Editor: {len(self.edits)} edit(s)"]
        for i, edit in enumerate(self.edits, 1):
            lines.append(
                f"  {i}. {edit.target_desc}.{edit.property_name}: "
                f"{edit.old_value!r} -> {edit.new_value!r}"
            )
        return '\n'.join(lines)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

_active_editors: Dict[int, FigureEditor] = {}


def interactive(fig, protect_data: bool = True) -> FigureEditor:
    """
    Open an interactive editor for a matplotlib figure.

    In Jupyter notebooks, displays an ipywidgets control panel beside
    the figure. In scripts, opens an enhanced matplotlib viewer.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to edit. Typically from ``result.plot()`` or
        ``sp.binscatter()``.
    protect_data : bool, default True
        Lock data-representing elements (scatter points, regression
        lines, confidence intervals) from modification.

    Returns
    -------
    FigureEditor
        The editor instance. Call ``.generate_code()`` to get
        reproducible code, or ``.copy_code()`` to print it.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.did(df, y='wage', treat='treated', time='post')
    >>> fig, ax = result.event_study_plot()
    >>> editor = sp.interactive(fig)
    >>> # ... make edits via the GUI ...
    >>> editor.copy_code()   # prints reproducible code
    """
    editor = FigureEditor(fig=fig, protect_data=protect_data)
    _active_editors[id(fig)] = editor

    if _in_jupyter():
        from ._jupyter_editor import create_jupyter_panel
        create_jupyter_panel(editor)
    else:
        from ._script_editor import create_script_editor
        create_script_editor(editor)

    return editor


def get_code(fig) -> str:
    """
    Get reproducible code for all interactive edits made to a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure

    Returns
    -------
    str
        Python code string.
    """
    editor = _active_editors.get(id(fig))
    if editor is None:
        return "# No interactive edits found for this figure"
    return editor.generate_code()


def _in_jupyter() -> bool:
    """Detect if running in a Jupyter-like notebook (Jupyter, Colab, VSCode)."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        name = shell.__class__.__name__
        # ZMQInteractiveShell: standard Jupyter
        # Shell: Google Colab
        # Also check for ipykernel module (VSCode notebooks)
        return name in ('ZMQInteractiveShell', 'Shell') or (
            hasattr(shell, 'kernel') and shell.kernel is not None
        )
    except (ImportError, NameError, AttributeError):
        return False
