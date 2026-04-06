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

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


# ------------------------------------------------------------------
# Font presets for academic publishing
# ------------------------------------------------------------------

def _detect_chinese_fonts() -> Dict[str, List[str]]:
    """
    Detect available Chinese fonts on the current system.

    Returns dict with 'serif' and 'sans-serif' lists, ordered by
    priority (most common/reliable first).
    """
    try:
        from matplotlib.font_manager import fontManager
    except ImportError:
        return {'serif': [], 'sans-serif': []}

    available = {f.name for f in fontManager.ttflist}

    # Candidate fonts in priority order per platform
    serif_candidates = [
        # macOS
        'Songti SC', 'Noto Serif CJK SC', 'Hiragino Mincho ProN',
        # Windows
        'SimSun', 'NSimSun',
        # Linux
        'Noto Serif CJK SC', 'WenQuanYi Micro Hei',
        # Cross-platform
        'STSong', 'AR PL UMing CN',
    ]
    sans_candidates = [
        # macOS
        'PingFang SC', 'PingFang HK', 'Heiti TC', 'Hiragino Sans GB',
        'Hiragino Sans',
        # Windows
        'Microsoft YaHei', 'SimHei',
        # Linux
        'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
        # Cross-platform
        'STHeiti', 'Kaiti SC',
    ]

    found_serif = [f for f in serif_candidates if f in available]
    found_sans = [f for f in sans_candidates if f in available]

    return {'serif': found_serif, 'sans-serif': found_sans}


def _get_chinese_serif() -> List[str]:
    """Get the best available Chinese serif font list for this system."""
    detected = _detect_chinese_fonts()
    result = detected['serif']
    if not result:
        # Fallback: try sans-serif Chinese fonts + generic
        result = detected['sans-serif']
    # Always append English serif as final fallback
    result.extend(['Times New Roman', 'DejaVu Serif'])
    return result


def _get_chinese_sans() -> List[str]:
    """Get the best available Chinese sans-serif font list."""
    detected = _detect_chinese_fonts()
    result = detected['sans-serif']
    if not result:
        result = detected['serif']
    result.extend(['Helvetica', 'Arial', 'DejaVu Sans'])
    return result


FONT_PRESETS = {
    # --- Serif fonts ---
    'Times New Roman': {
        'family': 'serif',
        'fonts': ['Times New Roman', 'DejaVu Serif'],
        'title_size': 11, 'label_size': 10, 'tick_size': 9,
        'venue': 'AER / Econometrica / APA / IEEE',
    },
    'Palatino': {
        'family': 'serif',
        'fonts': ['Palatino', 'Palatino Linotype', 'DejaVu Serif'],
        'title_size': 11, 'label_size': 10, 'tick_size': 9,
        'venue': 'Econometrica / Book typesetting',
    },
    'Charter': {
        'family': 'serif',
        'fonts': ['Charter', 'XCharter', 'DejaVu Serif'],
        'title_size': 11, 'label_size': 10, 'tick_size': 9,
        'venue': 'R default / Academic papers',
    },
    'Computer Modern': {
        'family': 'serif',
        'fonts': ['CMU Serif', 'Computer Modern', 'Latin Modern Roman', 'DejaVu Serif'],
        'title_size': 11, 'label_size': 10, 'tick_size': 9,
        'venue': 'LaTeX / Beamer default',
    },
    # --- Sans-serif fonts ---
    'Helvetica': {
        'family': 'sans-serif',
        'fonts': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'title_size': 10, 'label_size': 9, 'tick_size': 8,
        'venue': 'Nature / Science / Elsevier / Springer',
    },
    'Arial': {
        'family': 'sans-serif',
        'fonts': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'title_size': 10, 'label_size': 9, 'tick_size': 8,
        'venue': 'General / Office documents',
    },
    'Calibri': {
        'family': 'sans-serif',
        'fonts': ['Calibri', 'Arial', 'DejaVu Sans'],
        'title_size': 11, 'label_size': 10, 'tick_size': 9,
        'venue': 'Office / PowerPoint',
    },
    # --- Chinese fonts (auto-detect) ---
    'SimSun / 宋体': {
        'family': 'serif',
        'fonts': None,  # filled at runtime by _get_chinese_serif()
        'title_size': 12, 'label_size': 10.5, 'tick_size': 9,
        'venue': '中文论文 / 学位论文',
    },
    'SimHei / 黑体': {
        'family': 'sans-serif',
        'fonts': None,  # filled at runtime by _get_chinese_sans()
        'title_size': 12, 'label_size': 10.5, 'tick_size': 9,
        'venue': '中文 PPT / 幻灯片',
    },
}

# Size presets — independent of font choice.
# Users pick a font, then pick a size context.
SIZE_PRESETS = {
    'Journal (compact)': {
        'title_size': 9, 'label_size': 8, 'tick_size': 7,
        'desc': 'Nature / Science / small figures',
    },
    'Journal (standard)': {
        'title_size': 11, 'label_size': 10, 'tick_size': 9,
        'desc': 'AER / Econometrica / most journals',
    },
    'Journal (large)': {
        'title_size': 12, 'label_size': 12, 'tick_size': 11,
        'desc': 'APA / full-page figures',
    },
    'Thesis': {
        'title_size': 12, 'label_size': 10.5, 'tick_size': 9,
        'desc': '学位论文 / Dissertation',
    },
    'Slides': {
        'title_size': 16, 'label_size': 14, 'tick_size': 12,
        'desc': 'Beamer / Keynote / PowerPoint',
    },
    'Poster': {
        'title_size': 20, 'label_size': 16, 'tick_size': 14,
        'desc': 'Conference poster',
    },
}

# Backward compatibility aliases (old keys → new keys)
_PRESET_ALIASES = {
    'AER / Econometrica': 'Times New Roman',
    'APA (7th ed.)': 'Times New Roman',
    'Nature / Science': 'Helvetica',
    'IEEE': 'Times New Roman',
    'Elsevier': 'Helvetica',
    'Springer': 'Helvetica',
    'CJK Thesis': 'SimSun / 宋体',
    'CJK Journal': 'SimSun / 宋体',
    'CJK Slide': 'SimHei / 黑体',
    'Beamer / Slides': 'Helvetica (Slides)',
}


def _resolve_preset_fonts(preset: Dict) -> List[str]:
    """Resolve None font lists to auto-detected Chinese fonts."""
    if preset['fonts'] is not None:
        return preset['fonts']
    if preset['family'] == 'serif':
        return _get_chinese_serif()
    return _get_chinese_sans()


# Common fonts — auto-detect Chinese availability
def _build_font_choices() -> Dict[str, List[str]]:
    """Build font choice list with system-detected Chinese fonts."""
    detected = _detect_chinese_fonts()

    # Build Chinese choices with display names
    cn_fonts = []
    _cn_display = {
        'Songti SC': 'Songti SC (宋体)',
        'SimSun': 'SimSun (宋体)',
        'NSimSun': 'NSimSun (新宋体)',
        'STSong': 'STSong (华文宋体)',
        'Noto Serif CJK SC': 'Noto Serif CJK SC (思源宋体)',
        'Hiragino Mincho ProN': 'Hiragino Mincho ProN (ヒラギノ明朝)',
        'PingFang SC': 'PingFang SC (苹方)',
        'PingFang HK': 'PingFang HK (蘋方)',
        'Heiti TC': 'Heiti TC (黑体)',
        'Hiragino Sans GB': 'Hiragino Sans GB (冬青黑体)',
        'Hiragino Sans': 'Hiragino Sans (ヒラギノ角ゴ)',
        'Microsoft YaHei': 'Microsoft YaHei (微软雅黑)',
        'SimHei': 'SimHei (黑体)',
        'STHeiti': 'STHeiti (华文黑体)',
        'Kaiti SC': 'Kaiti SC (楷体)',
        'Noto Sans CJK SC': 'Noto Sans CJK SC (思源黑体)',
        'WenQuanYi Micro Hei': 'WenQuanYi Micro Hei (文泉驿)',
    }
    seen = set()
    for f in detected['serif'] + detected['sans-serif']:
        if f not in seen:
            seen.add(f)
            cn_fonts.append(_cn_display.get(f, f))

    return {
        'English Serif': [
            'Times New Roman', 'Palatino', 'Georgia', 'Garamond',
            'Computer Modern', 'DejaVu Serif',
        ],
        'English Sans-serif': [
            'Helvetica', 'Arial', 'Calibri', 'Verdana',
            'DejaVu Sans', 'Liberation Sans',
        ],
        'CJK': cn_fonts if cn_fonts else [
            '(No CJK fonts detected)',
        ],
        'Monospace': [
            'Courier New', 'Consolas', 'DejaVu Sans Mono',
            'Source Code Pro',
        ],
    }


# Lazy-initialized on first access
_font_choices_cache = None


def get_font_choices() -> Dict[str, List[str]]:
    """Get available font choices (cached, auto-detects Chinese fonts)."""
    global _font_choices_cache
    if _font_choices_cache is None:
        _font_choices_cache = _build_font_choices()
    return _font_choices_cache


# Backward compatibility alias
FONT_CHOICES = property(lambda self: get_font_choices())  # type: ignore


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

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            'target': self.target_desc,
            'property': self.property_name,
            'old': _serialize_value(self.old_value),
            'new': _serialize_value(self.new_value),
            'code': self.code_line,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'EditRecord':
        """Deserialize from a dictionary."""
        return cls(
            target_desc=d['target'],
            property_name=d['property'],
            old_value=d['old'],
            new_value=d['new'],
            code_line=d['code'],
        )


def _serialize_value(val: Any) -> Any:
    """Convert matplotlib/numpy values to JSON-safe Python types."""
    import numpy as np
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, tuple):
        return list(val)
    return val


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
    _on_refresh_callbacks: List = field(default_factory=list)

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

        # 1. Label-based hints (most reliable, word-boundary matching)
        ref_pattern = r'\b(reference|zero|cutoff|onset|treatment|threshold|baseline)\b'
        fit_pattern = r'\b(fit|trend|regression|polynomial|predicted|fitted|smoothed)\b'
        ci_pattern = r'\b(ci|confidence|bound|interval|upper|lower|band)\b'
        if re.search(ref_pattern, label):
            return ArtistRole.REFERENCE
        if re.search(fit_pattern, label):
            return ArtistRole.FIT
        if re.search(ci_pattern, label):
            return ArtistRole.CI

        # 2. Reference lines: constant x or y (horizontal/vertical)
        #    Guard: only classify as REFERENCE if few points (<=10).
        #    Real data series with constant values (e.g., flat baseline)
        #    typically have many points and should remain DATA.
        if len(xdata) >= 2 and len(set(ydata)) == 1 and len(xdata) <= 10:
            return ArtistRole.REFERENCE
        if len(ydata) >= 2 and len(set(xdata)) == 1 and len(ydata) <= 10:
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
            state[f'{prefix}.title.color'] = ax.title.get_color()
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

            # Snapshot line styles (color, width, style, alpha, marker)
            for j, line in enumerate(ax.get_lines()):
                lp = f'{prefix}.line{j}'
                state[f'{lp}.color'] = line.get_color()
                state[f'{lp}.linewidth'] = line.get_linewidth()
                state[f'{lp}.linestyle'] = line.get_linestyle()
                state[f'{lp}.alpha'] = line.get_alpha()
                state[f'{lp}.marker'] = line.get_marker()

            # Snapshot collection styles (facecolor, alpha)
            for j, coll in enumerate(ax.collections):
                cp = f'{prefix}.coll{j}'
                state[f'{cp}.alpha'] = coll.get_alpha()
                try:
                    state[f'{cp}.facecolors'] = coll.get_facecolors().copy()
                except Exception:
                    pass

        state['fig.figsize'] = tuple(self.fig.get_size_inches())
        state['fig.dpi'] = self.fig.dpi
        state['fig.facecolor'] = self.fig.get_facecolor()
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
                # Add 2% margin so edge points aren't clipped by markers
                margin = (data_max - data_min) * 0.02
                safe_min = data_min - margin
                safe_max = data_max + margin
                if left > safe_min or right < safe_max:
                    import warnings
                    warnings.warn(
                        f"Axis limits [{left:.2f}, {right:.2f}] would hide "
                        f"data in [{data_min:.2f}, {data_max:.2f}]. "
                        f"Expanding to include all data.",
                        UserWarning, stacklevel=2,
                    )
                    left = min(left, safe_min)
                    right = max(right, safe_max)

        old = ax.get_xlim()
        ax.set_xlim(left, right)
        # Clean float formatting for generated code
        l_str = f'{float(left):.6g}'
        r_str = f'{float(right):.6g}'
        self.edits.append(EditRecord(
            f'{prefix}.xlim', 'range', old, (left, right),
            f"{prefix}.set_xlim({l_str}, {r_str})"))
        self._refresh()

    def set_ylim(self, bottom: float, top: float, ax_index: int = 0):
        """Set y-axis limits with tracking and data protection."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'

        if self.protect_data:
            data_min, data_max = self._get_data_range(ax, 'y')
            if data_min is not None:
                margin = (data_max - data_min) * 0.02
                safe_min = data_min - margin
                safe_max = data_max + margin
                if bottom > safe_min or top < safe_max:
                    import warnings
                    warnings.warn(
                        f"Axis limits [{bottom:.2f}, {top:.2f}] would hide "
                        f"data in [{data_min:.2f}, {data_max:.2f}]. "
                        f"Expanding to include all data.",
                        UserWarning, stacklevel=2,
                    )
                    bottom = min(bottom, safe_min)
                    top = max(top, safe_max)

        old = ax.get_ylim()
        ax.set_ylim(bottom, top)
        b_str = f'{float(bottom):.6g}'
        t_str = f'{float(top):.6g}'
        self.edits.append(EditRecord(
            f'{prefix}.ylim', 'range', old, (bottom, top),
            f"{prefix}.set_ylim({b_str}, {t_str})"))
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

    def set_font(self, font_family: str, font_name: Optional[str] = None,
                 ax_index: int = 0):
        """
        Set font family for an axis (title, labels, ticks).

        Parameters
        ----------
        font_family : str
            'serif', 'sans-serif', or 'monospace'.
        font_name : str, optional
            Specific font name, e.g. 'Times New Roman', 'SimSun'.
            If provided, sets rcParams for the family to prefer this font.
        ax_index : int
            Which axis to apply to.
        """
        import matplotlib as mpl
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'

        old_family = mpl.rcParams.get('font.family', ['sans-serif'])

        # Set family
        mpl.rcParams['font.family'] = font_family

        # Set specific font at top of preference list
        clean_name = None
        if font_name:
            # Strip CJK label in parens: "Songti SC (宋体)" -> "Songti SC"
            clean_name = font_name.split(' (')[0].strip()
            key = f'font.{font_family}'
            current = list(mpl.rcParams.get(key, []))
            if clean_name not in current:
                current.insert(0, clean_name)
            mpl.rcParams[key] = current

            # Auto-fix minus sign for CJK fonts
            _cn_keywords = ('Song', 'Hei', 'Kai', 'Fang', 'PingFang',
                            'Hiragino', 'Noto', 'WenQuanYi', 'STH', 'STS',
                            'SC', 'TC', 'HK', 'JP', 'KR')
            if any(kw in clean_name for kw in _cn_keywords):
                mpl.rcParams['axes.unicode_minus'] = False

        # Update all text in this axis to use the new font
        for artist in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            artist.set_fontfamily(font_family)
            if clean_name:
                artist.set_fontname(clean_name)

        # Tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            if clean_name:
                label.set_fontname(clean_name)

        code_parts = ["import matplotlib as mpl"]
        code_parts.append(
            f"mpl.rcParams['font.family'] = {font_family!r}")
        if clean_name:
            code_parts.append(
                f"mpl.rcParams['font.{font_family}'] = "
                f"[{clean_name!r}] + mpl.rcParams.get("
                f"'font.{font_family}', [])"
            )
            # Include unicode_minus fix in generated code
            if any(kw in clean_name for kw in ('Song', 'Hei', 'Kai',
                   'Fang', 'PingFang', 'Hiragino', 'Noto', 'SC', 'TC')):
                code_parts.append(
                    "mpl.rcParams['axes.unicode_minus'] = False"
                )

        code = '\n'.join(code_parts)
        display_name = font_name or font_family
        self.edits.append(EditRecord(
            f'{prefix}.font', 'family', str(old_family), display_name,
            code))
        self._refresh()

    def apply_font_preset(self, preset_name: str, ax_index: int = 0):
        """
        Apply a font preset for a specific journal/thesis style.

        Parameters
        ----------
        preset_name : str
            One of the keys from FONT_PRESETS, e.g.
            'AER / Econometrica', 'CJK Thesis',
            'Nature / Science', etc.
        ax_index : int
            Which axis to apply to.
        """
        # Support backward-compatible aliases
        if preset_name not in FONT_PRESETS and preset_name in _PRESET_ALIASES:
            preset_name = _PRESET_ALIASES[preset_name]

        if preset_name not in FONT_PRESETS:
            available = ', '.join(FONT_PRESETS.keys())
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {available}")

        preset = FONT_PRESETS[preset_name]
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'

        import matplotlib as mpl

        # Resolve fonts (Chinese presets auto-detect at runtime)
        family = preset['family']
        fonts = _resolve_preset_fonts(preset)

        mpl.rcParams['font.family'] = family
        mpl.rcParams[f'font.{family}'] = fonts

        # Auto-detect Chinese fonts and fix minus sign
        _cn_keywords = ('Song', 'Hei', 'Kai', 'Fang', 'PingFang',
                        'Hiragino', 'Noto', 'WenQuanYi', 'STH', 'STS')
        if any(kw in str(fonts) for kw in _cn_keywords):
            mpl.rcParams['axes.unicode_minus'] = False

        # Apply to axis text
        primary_font = fonts[0]
        for artist in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            artist.set_fontfamily(family)
            artist.set_fontname(primary_font)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(family)
            label.set_fontname(primary_font)

        # Apply sizes from the preset as well (font + size in one go)
        self.set_fontsize('title', preset['title_size'], ax_index)
        self.set_fontsize('xlabel', preset['label_size'], ax_index)
        self.set_fontsize('ylabel', preset['label_size'], ax_index)
        self.set_fontsize('ticks', preset['tick_size'], ax_index)

        # Record as single edit (sizes already recorded individually)
        code = (
            f"import matplotlib as mpl\n"
            f"mpl.rcParams['font.family'] = {family!r}\n"
            f"mpl.rcParams['font.{family}'] = {fonts!r}"
        )
        self.edits.append(EditRecord(
            f'{prefix}.font_preset', 'preset', None, preset_name, code))
        self._refresh()

    def apply_size_preset(self, preset_name: str, ax_index: int = 0):
        """
        Apply a size-only preset (does NOT change font family/name).

        This lets users pick font and size independently:
        ``editor.set_font('serif', 'Palatino')`` then
        ``editor.apply_size_preset('Slides')``.

        Parameters
        ----------
        preset_name : str
            One of the keys from SIZE_PRESETS.
        ax_index : int
            Which axis to apply to.
        """
        if preset_name not in SIZE_PRESETS:
            available = ', '.join(SIZE_PRESETS.keys())
            raise ValueError(
                f"Unknown size preset '{preset_name}'. "
                f"Available: {available}")

        sp = SIZE_PRESETS[preset_name]
        self.set_fontsize('title', sp['title_size'], ax_index)
        self.set_fontsize('xlabel', sp['label_size'], ax_index)
        self.set_fontsize('ylabel', sp['label_size'], ax_index)
        self.set_fontsize('ticks', sp['tick_size'], ax_index)
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
        w_str = f'{float(width):.6g}'
        h_str = f'{float(height):.6g}'
        self.edits.append(EditRecord(
            'fig', 'figsize', old, (width, height),
            f"fig.set_size_inches({w_str}, {h_str})"))
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

    def set_dpi(self, dpi: int):
        """Set figure DPI with tracking."""
        old = self.fig.dpi
        self.fig.set_dpi(dpi)
        self.edits.append(EditRecord(
            'fig', 'dpi', old, dpi,
            f"fig.set_dpi({dpi})"))
        self._refresh()

    def set_background_color(self, color: str, target: str = 'figure',
                             ax_index: int = 0):
        """Set background color for figure or axes."""
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        if target == 'figure':
            old = self.fig.get_facecolor()
            self.fig.set_facecolor(color)
            code = f"fig.set_facecolor({color!r})"
            desc = 'fig.facecolor'
        else:
            ax = self.fig.get_axes()[ax_index]
            old = ax.get_facecolor()
            ax.set_facecolor(color)
            code = f"{prefix}.set_facecolor({color!r})"
            desc = f'{prefix}.facecolor'
        self.edits.append(EditRecord(desc, 'color', str(old), color, code))
        self._refresh()

    def set_tick_rotation(self, axis: str, angle: float,
                          ax_index: int = 0):
        """Set tick label rotation."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        if axis == 'x':
            old_labels = ax.get_xticklabels()
            old = old_labels[0].get_rotation() if old_labels else 0
            ax.tick_params(axis='x', rotation=angle)
            code = f"{prefix}.tick_params(axis='x', rotation={angle})"
        else:
            old_labels = ax.get_yticklabels()
            old = old_labels[0].get_rotation() if old_labels else 0
            ax.tick_params(axis='y', rotation=angle)
            code = f"{prefix}.tick_params(axis='y', rotation={angle})"
        self.edits.append(EditRecord(
            f'{prefix}.{axis}ticks', 'rotation', old, angle, code))
        self._refresh()

    def set_title_weight(self, weight: str, ax_index: int = 0):
        """Set title font weight (normal/bold)."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        old = ax.title.get_fontweight()
        ax.title.set_fontweight(weight)
        code = f"{prefix}.title.set_fontweight({weight!r})"
        self.edits.append(EditRecord(
            f'{prefix}.title', 'fontweight', old, weight, code))
        self._refresh()

    def set_legend_visible(self, visible: bool, ax_index: int = 0):
        """Show or hide the legend."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        legend = ax.get_legend()
        if legend is None:
            if visible:
                ax.legend()
                code = f"{prefix}.legend()"
            else:
                return
        else:
            legend.set_visible(visible)
            code = (f"{prefix}.get_legend().set_visible({visible})")
        self.edits.append(EditRecord(
            f'{prefix}.legend', 'visible', not visible, visible, code))
        self._refresh()

    def set_grid_style(self, color: Optional[str] = None,
                       alpha: Optional[float] = None,
                       linestyle: Optional[str] = None,
                       ax_index: int = 0):
        """Customize grid appearance."""
        ax = self.fig.get_axes()[ax_index]
        prefix = f'ax{ax_index}' if ax_index > 0 else 'ax'
        kwargs = {}
        if color is not None:
            kwargs['color'] = color
        if alpha is not None:
            kwargs['alpha'] = alpha
        if linestyle is not None:
            kwargs['linestyle'] = linestyle
        ax.grid(True, **kwargs)
        parts = ', '.join(f'{k}={v!r}' for k, v in kwargs.items())
        code = f"{prefix}.grid(True, {parts})"
        self.edits.append(EditRecord(
            f'{prefix}.grid', 'style', None, kwargs, code))
        self._refresh()

    def tight_layout(self):
        """Apply tight_layout to fix overlapping labels."""
        self.fig.tight_layout()
        self.edits.append(EditRecord(
            'fig', 'tight_layout', None, True,
            "fig.tight_layout()"))
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
        """Apply a theme (StatsPAI, matplotlib, or seaborn) and record it."""
        from .themes import set_theme
        set_theme(theme_name)  # handles all three sources
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

        # Parse axis index from target_desc like "ax.title" or "ax12.xlabel"
        ax_idx = 0
        ax_match = re.match(r'^ax(\d+)\.', td)
        if ax_match:
            ax_idx = int(ax_match.group(1))
        ax = axes[ax_idx] if ax_idx < len(axes) else axes[0]

        # Parse line/collection index from target like "ax.line2"
        def _parse_idx(prefix: str) -> int:
            part = td.split('.')[-1]
            if part.startswith(prefix):
                return int(part[len(prefix):])
            return 0

        try:
            # --- Text edits ---
            if 'title' in td and prop == 'text':
                ax.set_title(val)
            elif 'xlabel' in td and prop == 'text':
                ax.set_xlabel(val)
            elif 'ylabel' in td and prop == 'text':
                ax.set_ylabel(val)
            # --- Font sizes ---
            elif 'title' in td and prop == 'fontsize':
                ax.title.set_fontsize(val)
            elif 'xlabel' in td and prop == 'fontsize':
                ax.xaxis.label.set_fontsize(val)
            elif 'ylabel' in td and prop == 'fontsize':
                ax.yaxis.label.set_fontsize(val)
            elif 'ticks' in td and prop == 'fontsize':
                ax.tick_params(labelsize=val)
            # --- Axis limits ---
            elif 'xlim' in td:
                ax.set_xlim(val)
            elif 'ylim' in td:
                ax.set_ylim(val)
            # --- Layout ---
            elif 'spines' in td and prop == 'visible':
                spine_name = td.split('.')[-1]
                ax.spines[spine_name].set_visible(val)
            elif 'grid' in td:
                ax.grid(val)
            elif td == 'fig' and prop == 'figsize':
                self.fig.set_size_inches(val)
            # --- Colors ---
            elif prop == 'color' and 'title' in td:
                ax.title.set_color(val)
            elif prop == 'color' and 'xlabel' in td:
                ax.xaxis.label.set_color(val)
            elif prop == 'color' and 'ylabel' in td:
                ax.yaxis.label.set_color(val)
            elif prop == 'color' and 'line' in td:
                idx = _parse_idx('line')
                ax.get_lines()[idx].set_color(val)
            elif prop == 'color' and 'scatter' in td:
                idx = _parse_idx('scatter')
                ax.collections[idx].set_facecolors(val)
            # --- Line style properties ---
            elif prop == 'linewidth':
                idx = _parse_idx('line')
                ax.get_lines()[idx].set_linewidth(val)
            elif prop == 'linestyle':
                idx = _parse_idx('line')
                ax.get_lines()[idx].set_linestyle(val)
            elif prop == 'marker':
                idx = _parse_idx('line')
                ax.get_lines()[idx].set_marker(val)
            # --- Alpha ---
            elif prop == 'alpha' and 'line' in td:
                idx = _parse_idx('line')
                ax.get_lines()[idx].set_alpha(val)
            elif prop == 'alpha' and ('scatter' in td or 'ci' in td):
                prefix = 'scatter' if 'scatter' in td else 'ci'
                idx = _parse_idx(prefix)
                ax.collections[idx].set_alpha(val)
            # --- Theme ---
            elif td == 'theme':
                from .themes import set_theme
                set_theme(val)
            # --- Font preset ---
            elif 'font_preset' in td and prop == 'preset':
                resolved = _PRESET_ALIASES.get(val, val)
                if resolved in FONT_PRESETS:
                    self.apply_font_preset(resolved, ax_idx)
            elif 'font' in td and prop == 'family':
                import matplotlib as mpl
                mpl.rcParams['font.family'] = val
            # --- Legend ---
            elif 'legend' in td and prop == 'visible':
                legend = ax.get_legend()
                if legend:
                    legend.set_visible(val)
            elif 'legend' in td:
                legend = ax.get_legend()
                if legend and isinstance(val, dict):
                    if 'loc' in val:
                        legend.set_loc(val['loc'])
                    if 'frameon' in val:
                        legend.set_frame_on(val['frameon'])
            # --- DPI ---
            elif td == 'fig' and prop == 'dpi':
                self.fig.set_dpi(val)
            # --- Background color ---
            elif 'facecolor' in td:
                if td == 'fig.facecolor':
                    self.fig.set_facecolor(val)
                else:
                    ax.set_facecolor(val)
            # --- Tick rotation ---
            elif 'ticks' in td and prop == 'rotation':
                axis_char = 'x' if 'xticks' in td else 'y'
                ax.tick_params(axis=axis_char, rotation=val)
            # --- Title weight ---
            elif 'title' in td and prop == 'fontweight':
                ax.title.set_fontweight(val)
            # --- Grid style ---
            elif 'grid' in td and prop == 'style':
                if isinstance(val, dict):
                    ax.grid(True, **val)
            # --- Tight layout ---
            elif td == 'fig' and prop == 'tight_layout':
                self.fig.tight_layout()
        except (IndexError, KeyError, ValueError):
            pass  # Skip edits that reference out-of-range elements

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
            if f'{prefix}.title.color' in s:
                ax.title.set_color(s[f'{prefix}.title.color'])
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

            # Restore line styles
            for j, line in enumerate(ax.get_lines()):
                lp = f'{prefix}.line{j}'
                if f'{lp}.color' in s:
                    line.set_color(s[f'{lp}.color'])
                if f'{lp}.linewidth' in s:
                    line.set_linewidth(s[f'{lp}.linewidth'])
                if f'{lp}.linestyle' in s:
                    line.set_linestyle(s[f'{lp}.linestyle'])
                if f'{lp}.alpha' in s:
                    line.set_alpha(s[f'{lp}.alpha'])
                if f'{lp}.marker' in s:
                    line.set_marker(s[f'{lp}.marker'])

            # Restore collection styles
            for j, coll in enumerate(ax.collections):
                cp = f'{prefix}.coll{j}'
                if f'{cp}.alpha' in s:
                    coll.set_alpha(s[f'{cp}.alpha'])
                if f'{cp}.facecolors' in s:
                    try:
                        coll.set_facecolors(s[f'{cp}.facecolors'])
                    except Exception:
                        pass

        if 'fig.figsize' in s:
            self.fig.set_size_inches(s['fig.figsize'])
        if 'fig.facecolor' in s:
            self.fig.set_facecolor(s['fig.facecolor'])
        if 'fig.dpi' in s:
            self.fig.set_dpi(s['fig.dpi'])

    def on_refresh(self, callback):
        """Register a callback to be called after every edit refresh."""
        self._on_refresh_callbacks.append(callback)

    def _refresh(self):
        """Redraw the figure canvas and notify callbacks."""
        try:
            self.fig.canvas.draw()
        except Exception:
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass
        # Notify registered callbacks (e.g. Jupyter live preview)
        for cb in self._on_refresh_callbacks:
            try:
                cb(self.fig)
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

    # ------------------------------------------------------------------
    # Serialization — for web backends (e.g. CoPaper.ai)
    # ------------------------------------------------------------------

    def to_edits_json(self) -> str:
        """Serialize all edits to a JSON string.

        Returns
        -------
        str
            JSON array of edit records, suitable for sending to a
            web backend that will replay edits on a fresh figure.
        """
        import json
        return json.dumps(
            [e.to_dict() for e in self.edits],
            ensure_ascii=False,
        )

    def apply_edits_json(self, json_str: str):
        """Apply edits from a JSON string onto this figure.

        Parameters
        ----------
        json_str : str
            JSON array of edit dicts (from ``to_edits_json``).
        """
        import json
        records = [EditRecord.from_dict(d) for d in json.loads(json_str)]
        for edit in records:
            self._replay_edit(edit)
            self.edits.append(edit)
        self._refresh()

    def to_edits_list(self) -> list:
        """Return edits as a list of dicts (for direct API use)."""
        return [e.to_dict() for e in self.edits]

    def apply_edits_list(self, edits: list):
        """Apply edits from a list of dicts onto this figure.

        Parameters
        ----------
        edits : list of dict
            Each dict has keys: target, property, old, new, code.
        """
        records = [EditRecord.from_dict(d) for d in edits]
        for edit in records:
            self._replay_edit(edit)
            self.edits.append(edit)
        self._refresh()

    # ------------------------------------------------------------------
    # Headless Web API — for CoPaper.ai and other web backends
    # ------------------------------------------------------------------

    def render_to_base64(self, dpi: int = 300, fmt: str = 'png') -> str:
        """Render the figure to a base64-encoded string.

        Parameters
        ----------
        dpi : int, default 300
            Resolution for the rendered image.
        fmt : str, default 'png'
            Image format ('png', 'svg', 'pdf').

        Returns
        -------
        str
            Base64-encoded image data.
        """
        import base64
        import io
        buf = io.BytesIO()
        self.fig.savefig(buf, format=fmt, dpi=dpi,
                         bbox_inches='tight',
                         facecolor=self.fig.get_facecolor())
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('ascii')

    def get_editable_schema(self) -> dict:
        """Export schema of all editable properties and current values.

        Returns a dict that a web frontend can use to dynamically
        generate an editing panel. Each element includes its current
        value, role, and whether it's editable.

        Returns
        -------
        dict
            Schema with axes, figure-level settings, font presets,
            and theme options.
        """
        from .themes import list_themes

        axes_schema = []
        for i, ax in enumerate(self.fig.get_axes()):
            # Lines
            lines_schema = []
            for j, line in enumerate(ax.get_lines()):
                role = self.artist_roles.get(id(line), ArtistRole.COSMETIC)
                editable = role not in (
                    ArtistRole.DATA, ArtistRole.FIT, ArtistRole.CI
                ) if self.protect_data else True
                lines_schema.append({
                    'index': j,
                    'label': line.get_label() or f'line{j}',
                    'role': role.name,
                    'editable': editable,
                    'color': line.get_color(),
                    'linewidth': line.get_linewidth(),
                    'linestyle': line.get_linestyle(),
                    'alpha': line.get_alpha(),
                })

            # Collections (scatter, fill_between, etc.)
            collections_schema = []
            for j, coll in enumerate(ax.collections):
                role = self.artist_roles.get(id(coll), ArtistRole.COSMETIC)
                editable = role not in (
                    ArtistRole.DATA, ArtistRole.FIT, ArtistRole.CI
                ) if self.protect_data else True
                collections_schema.append({
                    'index': j,
                    'role': role.name,
                    'editable': editable,
                    'alpha': coll.get_alpha(),
                })

            # Spines
            spines = {}
            for name, spine in ax.spines.items():
                spines[name] = spine.get_visible()

            # Grid
            gridlines = ax.xaxis.get_gridlines()
            grid_on = gridlines[0].get_visible() if gridlines else False

            # Legend
            legend = ax.get_legend()
            legend_info = None
            if legend is not None:
                legend_info = {
                    'visible': legend.get_visible(),
                    'loc': legend._loc if hasattr(legend, '_loc') else 'best',
                }

            axes_schema.append({
                'index': i,
                'title': {
                    'text': ax.get_title(),
                    'fontsize': ax.title.get_fontsize(),
                    'color': ax.title.get_color(),
                },
                'xlabel': {
                    'text': ax.get_xlabel(),
                    'fontsize': ax.xaxis.label.get_fontsize(),
                },
                'ylabel': {
                    'text': ax.get_ylabel(),
                    'fontsize': ax.yaxis.label.get_fontsize(),
                },
                'xlim': list(ax.get_xlim()),
                'ylim': list(ax.get_ylim()),
                'lines': lines_schema,
                'collections': collections_schema,
                'spines': spines,
                'grid': grid_on,
                'legend': legend_info,
            })

        # Figure-level
        w, h = self.fig.get_size_inches()

        return {
            'axes': axes_schema,
            'figure': {
                'width': float(w),
                'height': float(h),
                'dpi': int(self.fig.dpi),
                'facecolor': self.fig.get_facecolor(),
            },
            'font_presets': [
                {
                    'name': name,
                    'font': _resolve_preset_fonts(preset)[0],
                    'family': preset['family'],
                    'venue': preset.get('venue', ''),
                    'title_size': preset['title_size'],
                    'label_size': preset['label_size'],
                    'tick_size': preset['tick_size'],
                }
                for name, preset in FONT_PRESETS.items()
            ],
            'size_presets': [
                {
                    'name': name,
                    'desc': sp.get('desc', ''),
                    'title_size': sp['title_size'],
                    'label_size': sp['label_size'],
                    'tick_size': sp['tick_size'],
                }
                for name, sp in SIZE_PRESETS.items()
            ],
            'themes': list_themes(),
            'protect_data': self.protect_data,
        }

    def apply_actions(self, actions: list):
        """Apply a list of action-based edit commands.

        This is the primary API for web frontends. Each action is a
        dict with an ``action`` key specifying the method to call.

        Parameters
        ----------
        actions : list of dict
            Each dict has ``action`` plus method-specific params.
            Supported actions::

                {"action": "set_title", "text": "...", "ax_index": 0}
                {"action": "set_xlabel", "text": "...", "ax_index": 0}
                {"action": "set_ylabel", "text": "...", "ax_index": 0}
                {"action": "set_fontsize", "target": "title", "size": 12, "ax_index": 0}
                {"action": "set_color", "target": "title", "color": "#333", "ax_index": 0}
                {"action": "set_dpi", "dpi": 300}
                {"action": "set_figsize", "width": 8, "height": 6}
                {"action": "set_grid", "visible": true, "ax_index": 0}
                {"action": "set_spine_visible", "spine": "top", "visible": false, "ax_index": 0}
                {"action": "apply_font_preset", "preset": "CJK Journal", "ax_index": 0}
                {"action": "apply_theme", "theme": "clean"}
                {"action": "set_legend_visible", "visible": true, "ax_index": 0}
                {"action": "set_line_color", "line_index": 0, "color": "#E74C3C", "ax_index": 0}
                {"action": "set_line_alpha", "line_index": 0, "alpha": 0.8, "ax_index": 0}

        Raises
        ------
        ValueError
            If an unknown action is encountered.
        """
        _dispatch = {
            'set_title': lambda a: self.set_title(
                a['text'], ax_index=a.get('ax_index', 0)),
            'set_xlabel': lambda a: self.set_xlabel(
                a['text'], ax_index=a.get('ax_index', 0)),
            'set_ylabel': lambda a: self.set_ylabel(
                a['text'], ax_index=a.get('ax_index', 0)),
            'set_fontsize': lambda a: self.set_fontsize(
                a['target'], a['size'], ax_index=a.get('ax_index', 0)),
            'set_color': lambda a: self.set_color(
                a['target'], a['color'], ax_index=a.get('ax_index', 0)),
            'set_dpi': lambda a: self.set_dpi(a['dpi']),
            'set_figsize': lambda a: self.set_figsize(
                a['width'], a['height']),
            'set_grid': lambda a: self.set_grid(
                a['visible'], ax_index=a.get('ax_index', 0)),
            'set_spine_visible': lambda a: self.set_spine_visible(
                a['spine'], a['visible'], ax_index=a.get('ax_index', 0)),
            'apply_font_preset': lambda a: self.apply_font_preset(
                a['preset'], ax_index=a.get('ax_index', 0)),
            'apply_size_preset': lambda a: self.apply_size_preset(
                a['preset'], ax_index=a.get('ax_index', 0)),
            'apply_theme': lambda a: self.apply_theme(a['theme']),
            'set_legend_visible': lambda a: self.set_legend_visible(
                a['visible'], ax_index=a.get('ax_index', 0)),
            'set_line_color': lambda a: self.set_color(
                f"line{a['line_index']}", a['color'],
                ax_index=a.get('ax_index', 0)),
            'set_line_alpha': lambda a: self.set_alpha(
                f"line{a['line_index']}", a['alpha'],
                ax_index=a.get('ax_index', 0)),
            'set_linewidth': lambda a: self.set_linewidth(
                a['line_index'], a['width'],
                ax_index=a.get('ax_index', 0)),
            'set_linestyle': lambda a: self.set_linestyle(
                a['line_index'], a['style'],
                ax_index=a.get('ax_index', 0)),
        }

        for action in actions:
            name = action.get('action')
            handler = _dispatch.get(name)
            if handler is None:
                raise ValueError(
                    f"Unknown action: {name!r}. "
                    f"Supported: {sorted(_dispatch.keys())}")
            handler(action)

    def export_state(self) -> dict:
        """Export the full editor state for persistence.

        Returns a dict containing edits and figure metadata,
        suitable for storing in a database or sending over the wire.
        Pair with ``import_state()`` to restore edits on a fresh figure.

        Returns
        -------
        dict
            Serialized state with edits, figure settings, and metadata.
        """
        w, h = self.fig.get_size_inches()
        return {
            'version': 1,
            'edits': self.to_edits_list(),
            'figure': {
                'width': float(w),
                'height': float(h),
                'dpi': int(self.fig.dpi),
            },
            'code': self.generate_code(include_comment=False),
        }

    def import_state(self, state: dict):
        """Restore editor state from a previously exported dict.

        Parameters
        ----------
        state : dict
            State dict from ``export_state()``.
        """
        # Apply figure-level settings first
        fig_state = state.get('figure', {})
        if 'width' in fig_state and 'height' in fig_state:
            self.set_figsize(fig_state['width'], fig_state['height'])
        if 'dpi' in fig_state:
            self.set_dpi(fig_state['dpi'])

        # Replay edits
        edits = state.get('edits', [])
        if edits:
            self.apply_edits_list(edits)

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
    # Store on figure so it lives and dies with the figure (no leak)
    fig._statspai_editor = editor

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
    editor = getattr(fig, '_statspai_editor', None)
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
