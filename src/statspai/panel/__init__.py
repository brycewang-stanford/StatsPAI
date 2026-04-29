"""
Unified panel regression module for StatsPAI.

Provides a single entry point ``panel()`` covering all panel estimators:

**Static models** — FE, RE, Between, First Difference, Pooled OLS, Two-way FE
**Correlated RE** — Mundlak (1978), Chamberlain (1982)
**Dynamic panel** — Arellano-Bond, Blundell-Bond (System GMM)
**HDFE absorption** — high-dimensional fixed-effects OLS (Stata's reghdfe /
R's fixest)

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
Correia, S. (2017). "Linear Models with High-Dimensional Fixed Effects."
"""

from typing import Any, Dict, Optional

# Underlying estimators
from .panel_reg import (
    panel as _panel_classical,
    panel_compare,
    balance_panel,
    PanelResults,
    PanelCompareResults,
    PanelRegression,
)
from .panel_binary import panel_logit, panel_probit
from .panel_plots import plot_within_between
from .hdfe import Absorber, demean, absorb_ols
from .feols import feols as hdfe_feols, hdfe_ols, FEOLSResult


# ═══════════════════════════════════════════════════════════════════════
#  Unified dispatcher — sp.panel(..., method=...)
# ═══════════════════════════════════════════════════════════════════════
#
# The classical panel() supports method= for fe / re / be / fd / pooled /
# twoway / mundlak / chamberlain / ab / system.  v1.10 adds:
#
#   • Case-insensitive aliases for the 10 classical methods
#     (fixed → fe, random → re, between → be, first_difference → fd,
#      pooled_ols → pooled, two_way → twoway, gmm → ab,
#      arellano_bond → ab, blundell_bond → system, system_gmm → system).
#
#   • New method='hdfe' route to ``feols.hdfe_ols`` for high-dimensional
#     fixed-effects absorption (Stata's reghdfe / R's fixest::feols).
#
# panel_logit / panel_probit / interactive_fe / panel_unitroot have a
# different (data, y, x, id, time)-style signature and are intentionally
# NOT in the dispatcher — they remain accessible as standalone functions.

_PANEL_METHOD_ALIASES: Dict[str, str] = {
    # --- Static models ---
    "fe": "fe", "fixed": "fe", "fixed_effects": "fe", "within": "fe",
    "re": "re", "random": "re", "random_effects": "re",
    "be": "be", "between": "be", "between_effects": "be",
    "fd": "fd", "first_difference": "fd", "first_diff": "fd",
    "pooled": "pooled", "pooled_ols": "pooled", "pols": "pooled",
    "ols": "pooled",
    "twoway": "twoway", "two_way": "twoway", "two_way_fe": "twoway",
    "twoway_fe": "twoway", "2way": "twoway",

    # --- Correlated random effects ---
    "mundlak": "mundlak", "mundlak_cre": "mundlak",
    "chamberlain": "chamberlain", "chamberlain_cre": "chamberlain",

    # --- Dynamic panel (GMM) ---
    "ab": "ab", "arellano_bond": "ab", "gmm": "ab", "diff_gmm": "ab",
    "difference_gmm": "ab",
    "system": "system", "system_gmm": "system",
    "blundell_bond": "system", "bb": "system",

    # --- HDFE absorption (new in v1.10) ---
    "hdfe": "hdfe", "feols": "hdfe", "reghdfe": "hdfe",
    "absorbed_ols": "hdfe",
}

_CLASSICAL_PANEL_METHODS = frozenset({
    "fe", "re", "be", "fd", "pooled", "twoway",
    "mundlak", "chamberlain", "ab", "system",
})


def panel(
    data: Any = None,
    formula: Optional[str] = None,
    entity: Optional[str] = None,
    time: Optional[str] = None,
    *,
    method: str = "fe",
    **kwargs: Any,
):
    """Unified panel-regression dispatcher.

    Parameters
    ----------
    data : DataFrame
    formula : str
        Patsy-style outcome ~ regressors specification.
    entity : str
        Cross-section identifier column.
    time : str
        Time identifier column.
    method : str, default ``'fe'``
        Estimator family:

        - **Static:** ``fe`` / ``fixed``, ``re`` / ``random``,
          ``be`` / ``between``, ``fd`` / ``first_difference``,
          ``pooled`` / ``pooled_ols``, ``twoway`` / ``two_way``.
        - **Correlated random effects:** ``mundlak``,
          ``chamberlain``.
        - **Dynamic GMM:** ``ab`` / ``arellano_bond`` / ``gmm``,
          ``system`` / ``blundell_bond`` / ``system_gmm``.
        - **HDFE absorption:** ``hdfe`` / ``feols`` / ``reghdfe``
          (high-dimensional fixed-effects OLS).
    **kwargs
        Forwarded to the chosen estimator.  Classical methods accept
        ``robust`` / ``cluster`` / ``weights`` / ``alpha`` /
        ``balance`` / ``lags`` / ``gmm_lags``.  HDFE accepts
        ``cluster`` / ``se_type`` / ``wild`` / ``alpha`` etc.

    Returns
    -------
    Result object whose type depends on ``method``.

    Examples
    --------
    >>> # Default: within (FE) estimator
    >>> r = sp.panel(df, "wage ~ exp + edu", entity='id', time='year')

    >>> # Random effects with Hausman test
    >>> r = sp.panel(df, "wage ~ exp + edu", entity='id', time='year',
    ...              method='re')

    >>> # Friendly alias (case insensitive)
    >>> r = sp.panel(df, "wage ~ exp", entity='id', time='year',
    ...              method='Fixed')

    >>> # Arellano-Bond dynamic panel
    >>> r = sp.panel(df, "wage ~ wage_lag + edu", entity='id', time='year',
    ...              method='gmm', lags=1)

    >>> # HDFE absorption (multiple FEs in formula)
    >>> r = sp.panel(df, "wage ~ exp | id + year", method='hdfe',
    ...              cluster='id')
    """
    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got {type(method).__name__}.")
    key = method.lower().strip().replace("-", "_")
    canon = _PANEL_METHOD_ALIASES.get(key)
    if canon is None:
        # Wording note: keep "method must be" — older callers grep for it.
        raise ValueError(
            f"Unknown method '{method}' for sp.panel — method must be "
            f"one of: {sorted(set(_PANEL_METHOD_ALIASES.values()))}"
        )

    # ── Classical: delegate to panel_reg.panel ───────────────────────
    if canon in _CLASSICAL_PANEL_METHODS:
        return _panel_classical(
            data=data, formula=formula, entity=entity, time=time,
            method=canon, **kwargs,
        )

    # ── HDFE absorption: route to feols.hdfe_ols ─────────────────────
    if canon == "hdfe":
        # hdfe_ols's signature is (formula, data, ...) — entity/time are
        # informational only; the absorbed FEs live in the formula via
        # the ``y ~ x | fe1 + fe2`` syntax.  If the caller passed
        # entity/time but the formula has no `|`, bolt them on as a
        # convenience so users don't have to rewrite the formula.
        if formula is None:
            raise ValueError(
                "method='hdfe' requires a formula like 'y ~ x | fe1 + fe2'."
            )
        if "|" not in formula and (entity or time):
            fes = " + ".join(fe for fe in (entity, time) if fe)
            formula = f"{formula} | {fes}"
        return hdfe_ols(formula=formula, data=data, **kwargs)

    raise AssertionError(  # pragma: no cover
        f"Unreachable panel dispatcher branch: canonical='{canon}'."
    )


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
    # HDFE primitives (native Python)
    'Absorber',
    'demean',
    'absorb_ols',
    'hdfe_feols',
    'hdfe_ols',
    'FEOLSResult',
]
