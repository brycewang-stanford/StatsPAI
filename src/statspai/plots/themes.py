"""
Plot themes for StatsPAI.

Provides one-line theme switching for publication-quality matplotlib
aesthetics. Supports three theme sources:

1. **StatsPAI themes** — custom academic themes (academic, aea, minimal, cn_journal)
2. **matplotlib styles** — built-in styles (ggplot, fivethirtyeight, bmh, dark_background, ...)
3. **seaborn styles** — if seaborn is installed (darkgrid, whitegrid, ticks, ...)

Usage
-----
>>> import statspai as sp
>>> sp.set_theme('academic')          # StatsPAI clean serif style
>>> sp.set_theme('aea')               # AER journal specifications
>>> sp.set_theme('ggplot')            # matplotlib ggplot style
>>> sp.set_theme('seaborn-whitegrid') # seaborn white grid
>>> sp.set_theme('fivethirtyeight')   # FiveThirtyEight style
>>> sp.set_theme('default')           # reset to matplotlib default
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
        # font.serif is set dynamically in set_theme() via _get_cn_serif()
        'figure.figsize': (8, 5.5),
        'figure.dpi': 150,
        'font.family': 'serif',
        'font.serif': None,  # resolved at runtime
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


def _get_cn_serif_fonts() -> list:
    """Auto-detect the best Chinese serif fonts on this system."""
    try:
        from matplotlib.font_manager import fontManager
        available = {f.name for f in fontManager.ttflist}
    except Exception:
        available = set()

    candidates = [
        'Songti SC', 'Noto Serif CJK SC', 'SimSun', 'STSong',
        'Hiragino Mincho ProN', 'AR PL UMing CN',
    ]
    found = [f for f in candidates if f in available]
    found.extend(['Times New Roman', 'DejaVu Serif'])
    return found


def use_chinese(style: str = 'auto') -> str:
    """
    One-line fix for Chinese text rendering in matplotlib.

    Call this **before** creating any plots. Automatically detects
    the best Chinese font on your system (macOS, Windows, Linux).

    Parameters
    ----------
    style : str, default 'auto'
        - ``'auto'``: auto-detect the best available Chinese font
        - ``'serif'``: prefer serif fonts (宋体 Songti SC, SimSun)
        - ``'sans'``: prefer sans-serif fonts (苹方 PingFang, 黑体 SimHei)
        - Any specific font name, e.g. ``'Songti SC'``, ``'SimHei'``

    Returns
    -------
    str
        The font name that was configured.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.use_chinese()           # auto-detect best font
    >>> sp.use_chinese('serif')    # prefer 宋体
    >>> sp.use_chinese('sans')     # prefer 黑体/苹方
    >>> sp.use_chinese('Kaiti SC') # use specific font
    """
    try:
        import matplotlib as mpl
        from matplotlib.font_manager import fontManager, findfont, FontProperties
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

    available = {f.name for f in fontManager.ttflist}

    # Priority lists for each platform
    serif_priority = [
        'Songti SC', 'Noto Serif CJK SC', 'SimSun', 'STSong',
        'Hiragino Mincho ProN', 'AR PL UMing CN',
    ]
    sans_priority = [
        'PingFang SC', 'PingFang HK', 'Hiragino Sans GB',
        'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC',
        'Heiti TC', 'STHeiti', 'WenQuanYi Micro Hei',
        'Hiragino Sans',
    ]
    all_priority = sans_priority + serif_priority + ['Arial Unicode MS']

    chosen = None

    if style == 'auto':
        for font in all_priority:
            if font in available:
                chosen = font
                break
    elif style == 'serif':
        for font in serif_priority:
            if font in available:
                chosen = font
                break
    elif style == 'sans':
        for font in sans_priority:
            if font in available:
                chosen = font
                break
    elif style in available:
        chosen = style
    else:
        # Try as-is, matplotlib will warn if not found
        chosen = style

    if chosen is None:
        import warnings
        warnings.warn(
            "No Chinese font found on this system. "
            "Install one of: Noto CJK, SimSun, PingFang, "
            "Microsoft YaHei, or WenQuanYi.",
            UserWarning, stacklevel=2,
        )
        return ''

    # Determine family
    _serif_names = ('Song', 'Serif', 'Mincho', 'Ming', 'STSong', 'Noto Serif')
    if any(kw in chosen for kw in _serif_names):
        family = 'serif'
        mpl.rcParams['font.family'] = 'serif'
        current = list(mpl.rcParams.get('font.serif', []))
        if chosen not in current:
            current.insert(0, chosen)
        mpl.rcParams['font.serif'] = current
    else:
        family = 'sans-serif'
        mpl.rcParams['font.family'] = 'sans-serif'
        current = list(mpl.rcParams.get('font.sans-serif', []))
        if chosen not in current:
            current.insert(0, chosen)
        mpl.rcParams['font.sans-serif'] = current

    # Fix minus sign
    mpl.rcParams['axes.unicode_minus'] = False

    return chosen


def set_theme(
    name: str = 'academic',
    palette: Optional[str] = None,
    font_scale: float = 1.0,
) -> None:
    """
    Set global matplotlib theme for publication-quality plots.

    Supports three theme sources:

    - **StatsPAI**: ``'academic'``, ``'aea'``, ``'minimal'``, ``'cn_journal'``
    - **matplotlib**: ``'ggplot'``, ``'fivethirtyeight'``, ``'bmh'``,
      ``'dark_background'``, ``'grayscale'``, ``'classic'``, etc.
    - **seaborn**: ``'seaborn-whitegrid'``, ``'seaborn-darkgrid'``,
      ``'seaborn-ticks'``, ``'seaborn-paper'``, ``'seaborn-talk'``, etc.

    Parameters
    ----------
    name : str, default 'academic'
        Theme name. Use ``list_themes()`` to see all available options.
        ``'default'`` resets to matplotlib defaults.
    palette : str, optional
        Color palette name. Defaults to matching the theme.
        Only applies to StatsPAI themes.
    font_scale : float, default 1.0
        Scale factor for all font sizes.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.set_theme('academic')          # clean serif, no top/right spines
    >>> sp.set_theme('aea')               # AER journal specifications
    >>> sp.set_theme('ggplot')            # R ggplot2 style
    >>> sp.set_theme('seaborn-whitegrid') # seaborn white grid
    >>> sp.set_theme('fivethirtyeight')   # FiveThirtyEight journalism style
    >>> sp.set_theme('dark_background')   # dark theme for slides
    >>> sp.set_theme('default')           # reset to matplotlib defaults
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

    # --- 1. StatsPAI custom themes ---
    if name in _THEMES:
        theme = _THEMES[name].copy()

        # Resolve Chinese fonts at runtime for cn_journal theme
        if theme.get('font.serif') is None:
            theme['font.serif'] = _get_cn_serif_fonts()

        # Apply font scaling
        if font_scale != 1.0:
            for key in ('font.size', 'axes.titlesize', 'axes.labelsize',
                        'xtick.labelsize', 'ytick.labelsize',
                        'legend.fontsize'):
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
                    pass
        return

    # --- 2. Normalize seaborn shorthand names ---
    resolved = _resolve_external_style(name)
    if resolved is not None:
        plt.style.use(resolved)
        # Apply font scaling on top of external style
        if font_scale != 1.0:
            for key in ('font.size', 'axes.titlesize', 'axes.labelsize',
                        'xtick.labelsize', 'ytick.labelsize',
                        'legend.fontsize'):
                try:
                    current = mpl.rcParams[key]
                    if isinstance(current, (int, float)):
                        mpl.rcParams[key] = current * font_scale
                except (KeyError, ValueError):
                    pass
        return

    # --- 3. Unknown theme ---
    available = list_themes()
    raise ValueError(
        f"Unknown theme '{name}'. Use list_themes() to see "
        f"{len(available)} available options."
    )


def _resolve_external_style(name: str) -> Optional[str]:
    """
    Resolve a theme name to a matplotlib style string.

    Handles shorthand like 'seaborn-whitegrid' -> 'seaborn-v0_8-whitegrid',
    and validates against available styles.
    """
    import matplotlib.pyplot as plt
    available = plt.style.available

    # Direct match
    if name in available:
        return name

    # seaborn shorthand: 'seaborn-whitegrid' -> 'seaborn-v0_8-whitegrid'
    if name.startswith('seaborn-'):
        suffix = name[len('seaborn-'):]
        versioned = f'seaborn-v0_8-{suffix}'
        if versioned in available:
            return versioned

    # seaborn bare: 'seaborn' -> 'seaborn-v0_8'
    if name == 'seaborn':
        for s in available:
            if s == 'seaborn-v0_8':
                return s

    return None


def list_themes() -> Dict[str, list]:
    """
    List all available themes grouped by source.

    Returns
    -------
    dict
        Keys: 'statspai', 'matplotlib', 'seaborn'.
        Values: list of theme name strings.

    Examples
    --------
    >>> import statspai as sp
    >>> themes = sp.list_themes()
    >>> themes['statspai']
    ['academic', 'aea', 'minimal', 'cn_journal']
    """
    result = {
        'statspai': list(_THEMES.keys()),
        'matplotlib': [],
        'seaborn': [],
    }

    try:
        import matplotlib.pyplot as plt
        for style in sorted(plt.style.available):
            if style.startswith('_'):
                continue  # skip internal styles
            if style.startswith('seaborn'):
                # Show user-friendly names
                friendly = style.replace('seaborn-v0_8-', 'seaborn-')
                friendly = friendly.replace('seaborn-v0_8', 'seaborn')
                result['seaborn'].append(friendly)
            else:
                result['matplotlib'].append(style)
    except ImportError:
        pass

    return result
