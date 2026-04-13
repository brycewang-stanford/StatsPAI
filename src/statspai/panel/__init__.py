"""
Unified panel regression module for StatsPAI.

Provides a single entry point ``panel()`` covering all panel estimators:

**Static models** — FE, RE, Between, First Difference, Pooled OLS, Two-way FE
**Correlated RE** — Mundlak (1978), Chamberlain (1982)
**Dynamic panel** — Arellano-Bond, Blundell-Bond (System GMM)

All results return ``PanelResults`` with built-in diagnostics:

>>> result = sp.panel(df, "y ~ x1 + x2", entity='id', time='t')
>>> result.hausman_test()        # FE vs RE
>>> result.bp_lm_test()          # Pooled vs RE
>>> result.f_test_effects()      # Joint significance of FE
>>> result.compare('re')         # Side-by-side comparison

References
----------
Wooldridge, J.M. (2010). Econometric Analysis of Cross Section and Panel Data.
Mundlak, Y. (1978). "On the Pooling of Time Series and Cross Section Data."
Chamberlain, G. (1982). "Multivariate Regression Models for Panel Data."
Arellano, M. and Bond, S. (1991). "Some Tests of Specification for Panel Data."
Blundell, R. and Bond, S. (1998). "Initial Conditions and Moment Restrictions."
Hausman, J.A. (1978). "Specification Tests in Econometrics."
Breusch, T.S. and Pagan, A.R. (1980). "The Lagrange Multiplier Test."
Pesaran, M.H. (2004). "General Diagnostic Tests for Cross Section Dependence."
"""

from .panel_reg import (
    panel,
    panel_compare,
    balance_panel,
    PanelResults,
    PanelCompareResults,
    PanelRegression,
)
from .panel_binary import panel_logit, panel_probit
from .panel_plots import plot_within_between

__all__ = [
    'panel',
    'panel_compare',
    'balance_panel',
    'PanelResults',
    'PanelCompareResults',
    'PanelRegression',
    'panel_logit',
    'panel_probit',
    'plot_within_between',
]
