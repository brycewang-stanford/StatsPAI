"""
Visualization module for StatsPAI.

Provides publication-quality academic plots:
- binscatter: Binned scatter plots with residualization (Cattaneo et al. 2024)
- coefplot: Coefficient comparison forest plots
- event_study_plot: DID event study (via CausalResult)
- rdplot: RD visualization (via rd module)
- marginsplot: Marginal effects (via postestimation)
- interactive: Interactive plot editor with data protection
"""

from .binscatter import binscatter
from .themes import set_theme, list_themes, use_chinese, _register_cjk_fallback
from .interactive import interactive, get_code, FigureEditor

# Auto-register CJK font fallback at import time so Chinese plots work out of
# the box. Appends detected CJK fonts to font.family list (matplotlib 3.6+
# per-glyph fallback). The user's primary family stays at index 0, so Latin
# glyphs are unchanged; only CJK glyphs missing from the primary fall through.
# Opt-out: STATSPAI_NO_AUTO_CJK=1. User's later rcParams[...] assignment wins.
_register_cjk_fallback()

__all__ = [
    'binscatter',
    'set_theme',
    'list_themes',
    'use_chinese',
    'interactive',
    'get_code',
    'FigureEditor',
]
