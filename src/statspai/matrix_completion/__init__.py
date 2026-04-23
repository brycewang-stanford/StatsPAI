"""
Matrix Completion for Causal Panel Data.

Estimates treatment effects in panel settings by completing the
counterfactual matrix of untreated potential outcomes using
nuclear-norm regularisation (low-rank matrix completion).

References
----------
Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021).
Matrix Completion Methods for Causal Panel Data Models.
JASA, 116(536), 1716-1730. [@athey2021matrix]
"""

from .mc_panel import mc_panel, MCPanel

__all__ = ['mc_panel', 'MCPanel']
