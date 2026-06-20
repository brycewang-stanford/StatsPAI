"""
Matching module for StatsPAI.

Unified interface for matching estimators:

- Nearest-neighbor matching (propensity score, Mahalanobis, Euclidean)
- Exact matching
- Coarsened Exact Matching (CEM)
- Propensity score stratification / subclassification
- Abadie-Imbens (2011) bias correction
- Entropy balancing (Hainmueller 2012)
- Covariate Balancing Propensity Score (Imai-Ratkovic 2014)
- Genetic Matching (Diamond-Sekhon 2013)
- Stable Balancing Weights (Zubizarreta 2015)
- Optimal pair / full / cardinality matching (Rosenbaum 1989, 2012)
- Overlap weights (Li-Morgan-Zaslavsky 2018)

The single entry point is :func:`match` — a method-aware dispatcher
that routes ``method=`` to the correct estimator.  Standalone
functions (``ebalance``, ``cbps``, ``genmatch``, ``sbw``,
``optimal_match``, ``cardinality_match``, ``overlap_weights``) remain
fully accessible for power users who need their estimator-specific
parameters.

References
----------
Rosenbaum, P.R. and Rubin, D.B. (1983). Biometrika, 70(1), 41-55.
Abadie, A. and Imbens, G.W. (2006). Econometrica, 74(1), 235-267.
Abadie, A. and Imbens, G.W. (2011). JBES, 29(1), 1-11.
Iacus, S.M., King, G., and Porro, G. (2012). Political Analysis, 20(1), 1-24.
Hainmueller, J. (2012). Political Analysis, 20(1), 25-46.
Imai, K. and Ratkovic, M. (2014). JRSS-B, 76(1), 243-263.
Diamond, A. and Sekhon, J.S. (2013). REStat, 95(3), 932-945.
Zubizarreta, J.R. (2015). JASA, 110(511), 910-922.
Li, F., Morgan, K.L., and Zaslavsky, A.M. (2018). JASA, 113(521), 390-400.
Rosenbaum, P.R. (2012). JASA, 107(498), 691-700.
Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University
Press. [@rosenbaum1983central]
"""

from typing import Any, Dict, List, Optional

from ..exceptions import MethodIncompatibility

# Underlying estimators — the dispatcher delegates here.
from .match import match as _match_classical
from .match import MatchEstimator, balanceplot, psplot
from .ebalance import ebalance
from .ps_diagnostics import (
    propensity_score,
    overlap_plot,
    trimming,
    love_plot,
    ps_balance,
    PSBalanceResult,
    balance_diagnostics,
    BalanceDiagnosticsResult,
)
from .optimal import (
    optimal_match,
    cardinality_match,
    OptimalMatchResult,
    CardinalityMatchResult,
)
from .overlap_weights import overlap_weights
from .cbps import cbps
from .genmatch import genmatch, GenMatchResult
from .sbw import sbw, SBWResult
from .psmatch2 import psmatch2, PSMatch2Result

# ═══════════════════════════════════════════════════════════════════════
#  Unified dispatcher — sp.match(..., method=...)
# ═══════════════════════════════════════════════════════════════════════
#
# ``sp.match`` was historically a function with built-in
# ``method=`` for `nearest` / `stratify` / `cem` / `psm` /
# `mahalanobis`.  v1.10 expands that to cover every advanced
# matching/weighting estimator in this subpackage:
#
#   classical  → nearest, stratify, cem, psm, mahalanobis
#   weighting  → ebalance, cbps, sbw, overlap
#   genetic    → genmatch
#   optimization-based → optimal, cardinality
#
# The classical methods retain their full kwarg surface
# (caliper / replace / bias_correction / ps_poly / n_strata /
# n_bins).  Advanced methods accept their own kwargs which we
# forward verbatim — see each estimator's docstring for details.

# These are the keyword args meaningful only to the classical
# match() implementation.  We strip them before forwarding to
# advanced estimators so a user explicitly invoking
# ``method='ebalance'`` doesn't get blamed for unknown kwargs.
_CLASSICAL_ONLY_KWARGS = frozenset(
    {
        "distance",
        "n_matches",
        "caliper",
        "replace",
        "bias_correction",
        "ps_poly",
        "n_strata",
        "n_bins",
        "common_support",
        "kernel",
        "bwidth",
        "se_method",
    }
)

_CLASSICAL_METHODS = frozenset(
    {
        "nearest",
        "stratify",
        "cem",
        "psm",
        "mahalanobis",
        "kernel",
        "radius",
    }
)

_MATCH_METHOD_ALIASES: Dict[str, str] = {
    # Classical (delegate to original match.py)
    "nearest": "nearest",
    "stratify": "stratify",
    "stratification": "stratify",
    "subclass": "stratify",
    "subclassification": "stratify",
    "cem": "cem",
    "coarsened_exact": "cem",
    "psm": "psm",  # legacy alias for nearest+propensity
    "mahalanobis": "mahalanobis",  # legacy alias for nearest+mahalanobis
    "kernel": "kernel",  # kernel propensity-score matching (psmatch2)
    "radius": "radius",  # radius matching = uniform kernel, bw=caliper
    # Weighting
    "ebalance": "ebalance",
    "entropy_balancing": "ebalance",
    "entropy": "ebalance",
    "cbps": "cbps",
    "sbw": "sbw",
    "stable_balancing": "sbw",
    "overlap": "overlap",
    "overlap_weights": "overlap",
    "ow": "overlap",
    # Genetic
    "genmatch": "genmatch",
    "genetic": "genmatch",
    # Optimization
    "optimal": "optimal",
    "optimal_match": "optimal",
    "cardinality": "cardinality",
    "cardinality_match": "cardinality",
}


def match(
    data: Any = None,
    y: Optional[str] = None,
    treat: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    *,
    method: str = "nearest",
    **kwargs: Any,
) -> Any:
    """Unified matching/weighting dispatcher.

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome column.
    treat : str
        Binary treatment indicator (0/1).
    covariates : list of str
        Pre-treatment covariates to balance/match on.
    method : str, default ``'nearest'``
        Estimator family:

        - **Classical:** ``'nearest'`` (default), ``'stratify'``,
          ``'cem'``, ``'psm'``, ``'mahalanobis'``.
        - **Weighting:** ``'ebalance'``, ``'cbps'``, ``'sbw'``,
          ``'overlap'``.
        - **Genetic:** ``'genmatch'``.
        - **Optimization:** ``'optimal'``, ``'cardinality'``.
    **kwargs
        Forwarded to the chosen estimator.  Classical methods accept
        ``distance`` / ``estimand`` / ``n_matches`` / ``caliper`` /
        ``replace`` / ``bias_correction`` / ``ps_poly`` /
        ``n_strata`` / ``n_bins`` / ``alpha``.  Advanced methods see
        their own docstring for kwargs.

    Returns
    -------
    Result object whose type depends on ``method``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> # Default: nearest-neighbour propensity-score matching
    >>> r = sp.match(df, y='log_wage', treat='union',
    ...              covariates=['education', 'experience', 'tenure'])

    >>> # Entropy balancing
    >>> r = sp.match(df, y='log_wage', treat='union',
    ...              covariates=['education', 'experience'], method='ebalance')

    >>> # Genetic matching (small donor pool / population for speed)
    >>> r = sp.match(df.head(300), y='log_wage', treat='union',
    ...              covariates=['education', 'experience'], method='genmatch',
    ...              population_size=20)

    >>> # Cardinality matching
    >>> r = sp.match(df.head(300), y='log_wage', treat='union',
    ...              covariates=['education', 'experience'], method='cardinality',
    ...              smd_tolerance=0.1)
    """
    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got {type(method).__name__}.")
    key = method.lower().strip().replace("-", "_")
    canon = _MATCH_METHOD_ALIASES.get(key)
    if canon is None:
        # Wording note: keep "method must be" in the message — older
        # tests grep for it (tests/test_matching.py).
        raise MethodIncompatibility(
            f"Unknown method '{method}' for sp.match — method must be "
            f"one of: {sorted(set(_MATCH_METHOD_ALIASES.values()))}",
            diagnostics={
                "method": method,
                "supported_methods": sorted(set(_MATCH_METHOD_ALIASES.values())),
            },
        )

    if y is None or treat is None or covariates is None:
        missing = [
            name
            for name, value in (
                ("y", y),
                ("treat", treat),
                ("covariates", covariates),
            )
            if value is None
        ]
        raise MethodIncompatibility(
            "sp.match requires y, treat, and covariates.",
            diagnostics={"missing": missing, "method": method},
        )

    # ── Classical: delegate to match.py:match ────────────────────────
    if canon in _CLASSICAL_METHODS:
        # Pass the canonical method name (already lowercased) plus all
        # original kwargs.  match.py knows the legacy aliases too.
        return _match_classical(
            data=data,
            y=y,
            treat=treat,
            covariates=covariates,
            method=canon,
            **kwargs,
        )

    # ── Advanced: strip classical-only kwargs and forward ────────────
    bad = [k for k in kwargs if k in _CLASSICAL_ONLY_KWARGS]
    if bad:
        raise TypeError(
            f"method='{method}' does not accept these classical-matching "
            f"kwargs: {bad}.  See sp.{canon}() docstring for the supported "
            f"parameter list."
        )

    if canon == "ebalance":
        return ebalance(data=data, y=y, treat=treat, covariates=covariates, **kwargs)
    if canon == "cbps":
        return cbps(data=data, y=y, treat=treat, covariates=covariates, **kwargs)
    if canon == "sbw":
        return sbw(data=data, y=y, treat=treat, covariates=covariates, **kwargs)
    if canon == "overlap":
        return overlap_weights(
            data=data,
            y=y,
            treat=treat,
            covariates=covariates,
            **kwargs,
        )
    if canon == "genmatch":
        return genmatch(
            data=data,
            y=y,
            treat=treat,
            covariates=covariates,
            **kwargs,
        )

    # optimal_match / cardinality_match use ``treatment``/``outcome``
    # rather than ``treat``/``y``.  Translate.
    if canon == "optimal":
        return optimal_match(
            data=data,
            treatment=treat,
            outcome=y,
            covariates=covariates,
            **kwargs,
        )
    if canon == "cardinality":
        return cardinality_match(
            data=data,
            treatment=treat,
            outcome=y,
            covariates=covariates,
            **kwargs,
        )

    raise AssertionError(  # pragma: no cover
        f"Unreachable match dispatcher branch: canonical='{canon}'."
    )


__all__ = [
    "match",
    "MatchEstimator",
    "ebalance",
    "balanceplot",
    "psplot",
    "propensity_score",
    "overlap_plot",
    "trimming",
    "love_plot",
    "ps_balance",
    "PSBalanceResult",
    "balance_diagnostics",
    "BalanceDiagnosticsResult",
    "optimal_match",
    "cardinality_match",
    "OptimalMatchResult",
    "CardinalityMatchResult",
    "overlap_weights",
    "cbps",
    "genmatch",
    "GenMatchResult",
    "sbw",
    "SBWResult",
    "psmatch2",
    "PSMatch2Result",
]
