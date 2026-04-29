"""
Regression Discontinuity (RD) module for StatsPAI.

The most comprehensive Python RD toolkit, providing:

**Core estimation:**
- Sharp, Fuzzy, and Kink RD estimation with robust bias-corrected inference (CCT 2014)
- Covariate-adjusted local polynomial estimation (Calonico et al. 2019)
- Donut-hole RD for manipulation near cutoff
- Regression Discontinuity in Time (Hausman-Rapson 2018)
- Multi-cutoff and multi-score RD (Cattaneo et al. 2024)
- Boundary discontinuity / 2D RD designs (Cattaneo-Titiunik-Yu 2025)

**Bandwidth selection:**
- MSE-optimal: mserd, msetwo, msecomb1, msecomb2
- CER-optimal: cerrd, certwo, cercomb1, cercomb2 (Calonico-Cattaneo-Farrell 2020)
- Fuzzy-specific and covariate-adjusted bandwidth selection

**Inference:**
- Honest confidence intervals (Armstrong-Kolesar 2018, 2020)
- Local randomization inference with Fisher exact tests (Cattaneo-Titiunik-VB 2016)
- Window selection and sensitivity analysis
- Rosenbaum sensitivity bounds

**Treatment effect heterogeneity:**
- CATE estimation via fully interacted local linear (Calonico et al. 2025)
- ML + RD: Causal Forest, Gradient Boosting, LASSO-assisted RD

**External validity:**
- Extrapolation away from cutoff (Angrist-Rokkanen 2015)
- Multi-cutoff extrapolation (Cattaneo et al. 2024)
- External validity diagnostics

**Diagnostics & visualization:**
- One-click diagnostic dashboard (rdsummary)
- IMSE-optimal binned scatter with pointwise CI bands
- Density manipulation testing (CJM 2020)
- Bandwidth sensitivity, covariate balance, placebo cutoff tests
- Power analysis and sample size calculations
"""

from .rdrobust import rdrobust, rdplot, rdplotdensity
from .bandwidth import rdbwselect
from .diagnostics import rdbwsensitivity, rdbalance, rdplacebo, rdsummary
from .rkd import rkd
from .honest_ci import rd_honest
from .rdit import rdit
from .rdmulti import rdmc, rdms, RDMultiResult
from .rdpower import rdpower, rdsampsi, RDPowerResult, RDSampSiResult
from .locrand import rdrandinf, rdwinselect, rdsensitivity, rdrbounds
from .hte import rdhte, rdbwhte, rdhte_lincom
from .rd2d import rd2d, rd2d_bw, rd2d_plot
from .rdml import rd_forest, rd_boost, rd_lasso, rd_cate_summary
from .extrapolate import rd_extrapolate, rd_multi_extrapolate, rd_external_validity

# v0.10 RDD frontier
from .interference import rd_interference, RDInterferenceResult
from .multi_score import rd_multi_score, MultiScoreRDResult
from .distribution_valued import rd_distribution, DistRDResult
from .bayes_hte import rd_bayes_hte, BayesRDHTEResult
from .distributional_design import rd_distributional_design, DDDResult

# User-friendly aliases
from ._aliases import (
    multi_cutoff_rd, geographic_rd, boundary_rd, multi_score_rd,
)

# ═══════════════════════════════════════════════════════════════════════
#  Unified dispatcher — sp.rd(..., method=...)
# ═══════════════════════════════════════════════════════════════════════
#
# Mirrors the IV pattern: ``sp.rd`` is the callable subpackage so
# users can write ``sp.rd(data, y, x, c)`` for the default rdrobust
# path, or ``sp.rd(..., method="honest")`` for any of the 18 estimator
# variants below — without giving up ``sp.rd.rdplot`` and friends.
#
# Diagnostics-only functions (``rdbwselect``, ``rdbwsensitivity``,
# ``rdbalance``, ``rdplacebo``, ``rdsummary``, ``rdplotdensity``,
# ``rdpower``, ``rdsampsi``, ``rdwinselect``, ``rdsensitivity``,
# ``rdrbounds``) are intentionally NOT in the ``method=`` table —
# they are not estimators of treatment effects.

import sys as _sys
from types import ModuleType as _ModuleType
from typing import Any as _Any, Dict as _Dict


_RD_METHOD_ALIASES: _Dict[str, str] = {
    # Local-polynomial bias-corrected (default, CCT 2014)
    "rdrobust": "rdrobust", "local_poly": "rdrobust",
    "default": "rdrobust", "rd": "rdrobust", "robust": "rdrobust",

    # Honest CIs (Armstrong-Kolesar 2018, 2020)
    "honest": "honest", "armstrong_kolesar": "honest", "ak": "honest",

    # Local randomization (Cattaneo-Titiunik-VB 2016)
    "randinf": "randinf", "random": "randinf",
    "local_randomization": "randinf", "rdrandinf": "randinf",

    # CATE / heterogeneous effects (Calonico et al. 2025)
    "hte": "hte", "rdhte": "hte", "cate": "hte",

    # ML + RD
    "forest": "forest", "causal_forest": "forest", "rd_forest": "forest",
    "boost": "boost", "gbm": "boost", "rd_boost": "boost",
    "lasso": "lasso", "rd_lasso": "lasso",

    # Bayesian HTE
    "bayes_hte": "bayes_hte", "bayes": "bayes_hte",
    "rd_bayes_hte": "bayes_hte",

    # 2D / boundary RD
    "rd2d": "rd2d", "2d": "rd2d", "boundary": "rd2d",

    # Multi-cutoff
    "rdmc": "rdmc", "multi_cutoff": "rdmc",

    # Multi-score
    "rdms": "rdms", "multi_score": "rdms", "geographic": "rdms",

    # Kink (RKD)
    "rkd": "rkd", "kink": "rkd",

    # RD in time
    "rdit": "rdit", "time": "rdit",

    # Extrapolation
    "extrapolate": "extrapolate", "rd_extrapolate": "extrapolate",
    "multi_extrapolate": "multi_extrapolate",
    "rd_multi_extrapolate": "multi_extrapolate",

    # Spillover / interference
    "interference": "interference", "spillover": "interference",
    "rd_interference": "interference",

    # Distributional treatment effects
    "distribution": "distribution", "distributional": "distribution",
    "rd_distribution": "distribution",
    "distributional_design": "distributional_design",
    "rd_distributional_design": "distributional_design",

    # External validity
    "external_validity": "external_validity",
    "rd_external_validity": "external_validity",
}


def _rd_rename(kwargs: _Dict[str, _Any], mapping: _Dict[str, str]) -> None:
    """Translate alias kwargs to the underlying estimator's expected
    names.  Mutates ``kwargs`` in place.  Raises if both alias and
    target are present (ambiguous).
    """
    for alias, target in mapping.items():
        if alias in kwargs:
            if target in kwargs:
                raise TypeError(
                    f"Got both '{alias}' and '{target}' — pick one. "
                    f"For this method '{target}' is canonical."
                )
            kwargs[target] = kwargs.pop(alias)


def _rd_dispatch(
    data: _Any = None,
    y: _Any = None,
    x: _Any = None,
    c: _Any = 0,
    *,
    method: str = "rdrobust",
    **kwargs: _Any,
):
    """Unified RD dispatcher.

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome column.
    x : str
        Running variable column.  Methods that internally call it
        ``running`` accept the canonical ``x`` here too.
    c : float, default 0
        Cutoff.  Methods that call it ``cutoff`` accept ``c`` here too.
    method : str, default ``'rdrobust'``
        See :data:`_RD_METHOD_ALIASES` for the full alias table.  Aliases
        are case-insensitive and ``-``/``_`` are interchangeable.
    **kwargs
        Method-specific options forwarded to the underlying estimator.

    Returns
    -------
    Result object whose type depends on ``method``.

    Raises
    ------
    ValueError
        Unknown ``method``.
    """
    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got {type(method).__name__}.")
    key = method.lower().strip().replace("-", "_")
    canon = _RD_METHOD_ALIASES.get(key)
    if canon is None:
        raise ValueError(
            f"Unknown method '{method}' for sp.rd. "
            f"Choose from: {sorted(set(_RD_METHOD_ALIASES.values()))}"
        )

    # ── Methods that take (data, y, x, c) directly ───────────────────
    _passthrough_xc = {
        "rdrobust": rdrobust,
        "honest": rd_honest,
        "randinf": rdrandinf,
        "hte": rdhte,
        "forest": rd_forest,
        "boost": rd_boost,
        "lasso": rd_lasso,
        "rkd": rkd,
        "extrapolate": rd_extrapolate,
        "external_validity": rd_external_validity,
    }
    if canon in _passthrough_xc:
        return _passthrough_xc[canon](data=data, y=y, x=x, c=c, **kwargs)

    # ── Methods that use ``running``/``cutoff`` instead of ``x``/``c``
    if canon == "bayes_hte":
        _rd_rename(kwargs, {"x": "running", "c": "cutoff"})
        return rd_bayes_hte(
            data=data, y=y, running=x, cutoff=c, **kwargs,
        )
    if canon == "interference":
        _rd_rename(kwargs, {"x": "running", "c": "cutoff"})
        return rd_interference(
            data=data, y=y, running=x, cutoff=c, **kwargs,
        )
    if canon == "distribution":
        _rd_rename(kwargs, {"x": "running", "c": "cutoff"})
        return rd_distribution(
            data=data, y=y, running=x, cutoff=c, **kwargs,
        )
    if canon == "distributional_design":
        _rd_rename(kwargs, {"x": "running", "c": "cutoff"})
        return rd_distributional_design(
            data=data, y=y, running=x, cutoff=c, **kwargs,
        )

    # ── Multi-cutoff: ``cutoffs`` (plural) replaces ``c`` ────────────
    if canon == "rdmc":
        # Allow user to pass either ``c=[...]`` or ``cutoffs=[...]``.
        cutoffs = kwargs.pop("cutoffs", None)
        if cutoffs is None:
            cutoffs = c
        return rdmc(data=data, y=y, x=x, cutoffs=cutoffs, **kwargs)

    # ── Multi-score: needs (x1, x2, cutoff1, cutoff2) ────────────────
    if canon == "rdms":
        return rdms(data=data, y=y, **kwargs)

    # ── 2D RD: needs (x1, x2, treatment, boundary) ───────────────────
    if canon == "rd2d":
        return rd2d(data=data, y=y, **kwargs)

    # ── Multi-extrapolate: ``cutoffs`` plural ────────────────────────
    if canon == "multi_extrapolate":
        cutoffs = kwargs.pop("cutoffs", None)
        if cutoffs is None:
            cutoffs = c
        return rd_multi_extrapolate(data=data, y=y, x=x, cutoffs=cutoffs, **kwargs)

    # ── RD-in-time: ``time`` and ``cutoff`` ──────────────────────────
    if canon == "rdit":
        # Accept ``x`` as the time axis alias for callers used to RDD.
        if "time" not in kwargs and x is not None:
            kwargs["time"] = x
        if "cutoff" not in kwargs and c is not None:
            kwargs["cutoff"] = c
        return rdit(data=data, y=y, **kwargs)

    raise AssertionError(  # pragma: no cover
        f"Unreachable RD dispatcher branch: canonical='{canon}'."
    )


# Public alias — ``sp.rd.fit`` mirrors ``sp.rd(...)``.
def fit(
    data: _Any = None,
    y: _Any = None,
    x: _Any = None,
    c: _Any = 0,
    *,
    method: str = "rdrobust",
    **kwargs: _Any,
):
    """Alias for :func:`_rd_dispatch`.  See ``sp.rd.__doc__`` for usage."""
    return _rd_dispatch(data=data, y=y, x=x, c=c, method=method, **kwargs)


class _CallableRDModule(_ModuleType):
    """ModuleType subclass that delegates ``sp.rd(...)`` calls to
    :func:`_rd_dispatch` while preserving submodule/attribute access.
    """

    def __call__(self, *args: _Any, **kwargs: _Any):  # noqa: D401
        return _rd_dispatch(*args, **kwargs)


_sys.modules[__name__].__class__ = _CallableRDModule


__all__ = [
    # callable + alias
    'fit',
    # core estimators
    'rdrobust',
    'rdplot',
    'rdplotdensity',
    'rdbwselect',
    'rdbwsensitivity',
    'rdbalance',
    'rdplacebo',
    'rdsummary',
    'rkd',
    'rd_honest',
    'rdit',
    'rdmc',
    'rdms',
    'RDMultiResult',
    'rdpower',
    'rdsampsi',
    'RDPowerResult',
    'RDSampSiResult',
    'rdrandinf',
    'rdwinselect',
    'rdsensitivity',
    'rdrbounds',
    'rdhte',
    'rdbwhte',
    'rdhte_lincom',
    'rd2d',
    'rd2d_bw',
    'rd2d_plot',
    'rd_forest',
    'rd_boost',
    'rd_lasso',
    'rd_cate_summary',
    'rd_extrapolate',
    'rd_multi_extrapolate',
    'rd_external_validity',
    # v0.10 frontier
    'rd_interference', 'RDInterferenceResult',
    'rd_multi_score', 'MultiScoreRDResult',
    'rd_distribution', 'DistRDResult',
    'rd_bayes_hte', 'BayesRDHTEResult',
    'rd_distributional_design', 'DDDResult',
    # User-friendly aliases
    'multi_cutoff_rd',
    'geographic_rd',
    'boundary_rd',
    'multi_score_rd',
]
