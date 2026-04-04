"""
Academic plot themes for StatsPAI.

Provides one-line theme switching for publication-quality matplotlib
aesthetics, replacing matplotlib's ugly defaults.

Inspired by R's ggplot2::theme_minimal() and Stata's graph schemes.

Usage
-----
>>> import statspai as sp
>>> sp.set_theme('academic')   # clean academic style
>>> sp.set_theme('aea')        # AER/AEA house style
>>> sp.set_theme('minimal')    # ggplot2 minimal equivalent
>>> sp.set_theme('default')    # reset to matplotlib default
"""

from typing import Optional, Dict, Any

_THEMES: Dict[str, Dict[str, Any]] = {
    'academic': {
        'figure.figsize': (8, 5.5),
        'figure.dpi': 150,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.grid': False,
        'axes.prop_cycle': None,  # set below
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.fontsize': 9,
        'legend.frameon': False,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.transparent': False,
    },
    'aea': {
        # American Economic Association style (AER, AEJ, etc.)
        'figure.figsize': (6.5, 4.5),  # AER column width
        'figure.dpi': 150,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.6,
        'axes.grid': False,
        'axes.prop_cycle': None,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    },
    'minimal': {
        # ggplot2 theme_minimal equivalent
        'figure.figsize': (8, 5.5),
        'figure.dpi': 150,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'axes.linewidth': 0,
        'axes.grid': True,
        'axes.grid.which': 'major',
        'axes.prop_cycle': None,
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.6,
        'grid.alpha': 1.0,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'lines.linewidth': 1.5,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    },
    'cn_journal': {
        # Chinese journal style (三线表配套)
        'figure.figsize': (8, 5.5),
        'figure.dpi': 150,
        'font.family': 'serif',
        'font.serif': ['SimSun', 'STSong', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 10.5,  # 五号字
        'axes.titlesize': 12,
        'axes.labelsize': 10.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.grid': False,
        'axes.unicode_minus': False,  # 正确显示负号
        'axes.prop_cycle': None,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    },
}

# Academic color palettes
_PALETTES = {
    'academic': ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71',
                 '#9B59B6', '#F39C12', '#1ABC9C', '#E67E22'],
    'aea': ['#000000', '#4472C4', '#ED7D31', '#A5A5A5',
            '#FFC000', '#5B9BD5', '#70AD47', '#264478'],
    'minimal': ['#4C72B0', '#DD8452', '#55A868', '#C44E52',
                '#8172B3', '#937860', '#DA8BC3', '#8C8C8C'],
    'cn_journal': ['#000000', '#E31A1C', '#1F78B4', '#33A02C',
                   '#FF7F00', '#6A3D9A', '#B15928', '#A6CEE3'],
}

_original_rcparams = None


def set_theme(
    name: str = 'academic',
    palette: Optional[str] = None,
    font_scale: float = 1.0,
) -> None:
    """
    Set global matplotlib theme for publication-quality plots.

    Parameters
    ----------
    name : str, default 'academic'
        Theme name: ``'academic'``, ``'aea'``, ``'minimal'``,
        ``'cn_journal'``, or ``'default'`` (reset).
    palette : str, optional
        Color palette name. Defaults to matching the theme.
    font_scale : float, default 1.0
        Scale factor for all font sizes.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.set_theme('academic')  # clean serif style, no top/right spines
    >>> sp.set_theme('aea')       # AER journal specifications
    >>> sp.set_theme('cn_journal')  # Chinese journal with SimSun/宋体
    >>> sp.set_theme('default')   # reset to matplotlib defaults
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from cycler import cycler
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

    global _original_rcparams

    # Save original params on first call
    if _original_rcparams is None:
        _original_rcparams = mpl.rcParams.copy()

    if name == 'default':
        mpl.rcParams.update(_original_rcparams)
        return

    if name not in _THEMES:
        raise ValueError(
            f"Unknown theme '{name}'. "
            f"Available: {list(_THEMES.keys()) + ['default']}"
        )

    theme = _THEMES[name].copy()

    # Apply font scaling
    if font_scale != 1.0:
        for key in ('font.size', 'axes.titlesize', 'axes.labelsize',
                    'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize'):
            if key in theme:
                theme[key] = theme[key] * font_scale

    # Set color cycle
    pal_name = palette or name
    pal_colors = _PALETTES.get(pal_name, _PALETTES['academic'])
    theme['axes.prop_cycle'] = cycler('color', pal_colors)

    # Apply
    for key, val in theme.items():
        if val is not None:
            try:
                mpl.rcParams[key] = val
            except (KeyError, ValueError):
                pass  # skip unsupported params on older matplotlib
