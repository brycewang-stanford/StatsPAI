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
from .themes import set_theme
from .interactive import interactive, get_code, FigureEditor

__all__ = [
    'binscatter',
    'set_theme',
    'interactive',
    'get_code',
    'FigureEditor',
]
