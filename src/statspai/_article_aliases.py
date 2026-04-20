"""Top-level aliases matching the public-facing article API.

The StatsPAI README and blog posts advertise a short, Stata-like surface
(`sp.rdd`, `sp.frontdoor`, `sp.xlearner`, ...).  Several of these names
are *thin wrappers* over richer implementations that already live in the
submodules — for example ``sp.rdd`` is shorthand for
``sp.rdrobust`` with the running variable named ``x``.

Keeping the aliases in one place (instead of sprinkling ``xxx = yyy``
across ``__init__.py``) makes it easy to:

* verify the article's documented surface with a single audit pass
* change a wrapper's defaults without editing the package root
* write targeted tests that pin the alias → implementation mapping

Every wrapper here delegates to an *existing* implementation and adds
no numerical code of its own. If you change behaviour, change the
underlying module — not this file.
"""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

from .core.results import CausalResult

__all__ = [
    "rdd",
    "frontdoor",
    "xlearner",
    "conformal_ite",
    "psm",
    "partial_identification",
    "anderson_rubin_ci",
    "conditional_lr_ci",
    "tF_adjustment",
]


# ---------------------------------------------------------------------------
# Regression discontinuity
# ---------------------------------------------------------------------------

def rdd(
    data: pd.DataFrame,
    y: str,
    running: str,
    cutoff: float = 0.0,
    *,
    fuzzy: Optional[str] = None,
    **kwargs: Any,
) -> CausalResult:
    """Sharp / fuzzy RD — article-friendly alias for :func:`rdrobust`.

    Parameters match the blog post signature ``sp.rdd(df, y, running, cutoff)``
    and are forwarded to :func:`statspai.rd.rdrobust` using its
    ``(x=<running>, c=<cutoff>)`` convention.
    """
    from .rd.rdrobust import rdrobust

    return rdrobust(
        data=data,
        y=y,
        x=running,
        c=cutoff,
        fuzzy=fuzzy,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Pearl's front-door criterion
# ---------------------------------------------------------------------------

def frontdoor(
    data: pd.DataFrame,
    y: str,
    d: str,
    m: str,
    X: Optional[List[str]] = None,
    **kwargs: Any,
) -> CausalResult:
    """Front-door adjustment — article-friendly alias for
    :func:`statspai.inference.front_door`.

    ``X`` is mapped to the underlying ``covariates`` argument.
    """
    from .inference.front_door import front_door as _front_door

    return _front_door(
        data=data,
        y=y,
        treat=d,
        mediator=m,
        covariates=X,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Meta-learner shortcuts
# ---------------------------------------------------------------------------

def xlearner(
    data: pd.DataFrame,
    y: str,
    d: str,
    X: List[str],
    **kwargs: Any,
) -> CausalResult:
    """X-Learner CATE — article alias for :func:`metalearner(learner='x')`.

    Kept separate from the generic :func:`metalearner` entry point because
    the blog post advertises ``sp.xlearner(df, y, d, X)`` directly.

    Passing ``learner=...`` is rejected — callers who want a different
    meta-learner should use :func:`sp.metalearner` instead of silently
    getting an X-Learner under a misleading name.
    """
    if "learner" in kwargs:
        raise TypeError(
            "sp.xlearner is fixed to learner='x'. Use sp.metalearner(..., "
            f"learner={kwargs['learner']!r}) for a different meta-learner."
        )

    from .metalearners.metalearners import metalearner

    return metalearner(
        data=data,
        y=y,
        treat=d,
        covariates=X,
        learner="x",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Conformal ITE intervals
# ---------------------------------------------------------------------------

def conformal_ite(
    data: pd.DataFrame,
    y: str,
    d: str,
    X: List[str],
    **kwargs: Any,
) -> CausalResult:
    """Conformal ITE — article alias for :func:`conformal_cate`.

    Covers the ``sp.conformal_ite(df, y, d, X)`` shape advertised in the
    2026-04-20 blog post.  Delegates to
    :func:`statspai.conformal_causal.conformal_cate`.
    """
    from .conformal_causal.conformal_ite import conformal_cate

    return conformal_cate(
        data=data,
        y=y,
        treat=d,
        covariates=X,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Propensity-score matching
# ---------------------------------------------------------------------------

def psm(
    data: pd.DataFrame,
    y: str,
    d: str,
    X: List[str],
    *,
    method: str = "nn",
    **kwargs: Any,
) -> CausalResult:
    """Propensity-score matching — article alias for :func:`match`
    with ``distance='propensity'``.

    ``method='nn'`` (the common Stata/R shorthand) is translated into the
    richer ``method='nearest'`` API of :func:`statspai.matching.match`.
    """
    from .matching.match import match as _match

    # Map the common PSM aliases to the underlying match() API.
    alias_map = {
        "nn": "nearest",
        "nearest": "nearest",
        "psm": "nearest",
        "stratify": "stratify",
        "cem": "cem",
    }
    internal_method = alias_map.get(method, method)

    return _match(
        data=data,
        y=y,
        treat=d,
        covariates=X,
        method=internal_method,
        distance=kwargs.pop("distance", "propensity"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Partial identification / bounds
# ---------------------------------------------------------------------------

def partial_identification(
    data: pd.DataFrame,
    y: str,
    d: str,
    X: Optional[List[str]] = None,
    *,
    method: str = "manski",
    selection: Optional[str] = None,
    instrument: Optional[str] = None,
    assumptions: Optional[List[str]] = None,  # noqa: ARG001 — reserved
    **kwargs: Any,
):
    """Partial identification of ATE — article alias for the ``bounds`` module.

    ``method='manski'``          → :func:`manski_bounds`   (worst-case bounds)
    ``method='lee'``              → :func:`lee_bounds`     (monotone-selection
                                                            bounds; requires
                                                            ``selection=``)
    ``method='horowitz_manski'``  → :func:`horowitz_manski` (requires
                                                            covariates via ``X``)
    ``method='iv'``               → :func:`iv_bounds`      (requires
                                                            ``instrument=``)

    The underlying bounds functions use slightly different parameter names
    (``treat`` vs ``treatment``, ``covariates`` vs ``controls``).  This
    wrapper normalises the public-facing ``(y, d, X)`` surface and routes to
    each backend with its native kwargs.

    The ``assumptions`` keyword is accepted for forward compatibility but
    ignored by all current back-ends; see each underlying function for its
    native assumption interface.
    """
    from . import bounds as _bounds

    method = method.lower()

    if method == "manski":
        # manski_bounds uses `treat`; no covariates supported — warn if given.
        if X:
            raise ValueError(
                "partial_identification(method='manski') does not use "
                "covariates (pure worst-case bounds). Drop X or use "
                "method='horowitz_manski' for a covariate-aware variant."
            )
        return _bounds.manski_bounds(data=data, y=y, treat=d, **kwargs)

    if method == "lee":
        # lee_bounds uses `treat` and REQUIRES `selection`.
        if selection is None:
            raise ValueError(
                "partial_identification(method='lee') requires "
                "`selection=<column name>` — Lee (2009) bounds are for "
                "sample-selection problems where a binary observability "
                "indicator is needed."
            )
        return _bounds.lee_bounds(
            data=data, y=y, treat=d,
            selection=selection, covariates=X, **kwargs,
        )

    if method in {"horowitz_manski", "horowitz-manski", "hm"}:
        # horowitz_manski uses `treatment` (not `treat`) and REQUIRES
        # `covariates` (cannot be None).
        if not X:
            raise ValueError(
                "partial_identification(method='horowitz_manski') requires "
                "a non-empty list of covariates via `X=[...]` — the "
                "Horowitz-Manski bounds condition on X."
            )
        return _bounds.horowitz_manski(
            data=data, y=y, treatment=d, covariates=X, **kwargs,
        )

    if method == "iv":
        # iv_bounds uses `treatment`, `instrument`, and `controls` (not
        # `covariates`).  `X` maps to `controls` here.
        if instrument is None:
            raise ValueError(
                "partial_identification(method='iv') requires "
                "`instrument=<column name>` for the IV bounds (Manski-Pepper)."
            )
        return _bounds.iv_bounds(
            data=data, y=y, treatment=d,
            instrument=instrument, controls=X, **kwargs,
        )

    raise ValueError(
        f"Unknown partial_identification method '{method}'. "
        "Expected one of: 'manski', 'lee', 'horowitz_manski', 'iv'."
    )


# ---------------------------------------------------------------------------
# Weak-IV robust confidence sets (top-level re-exports)
# ---------------------------------------------------------------------------

def anderson_rubin_ci(*args, **kwargs):
    """Anderson-Rubin confidence set — re-export of
    :func:`statspai.iv.weak_iv_ci.anderson_rubin_ci`.

    The AR test remains exact under any level of weak identification, so
    the corresponding confidence set is the canonical weak-IV-robust CI.
    """
    from .iv.weak_iv_ci import anderson_rubin_ci as _impl
    return _impl(*args, **kwargs)


def conditional_lr_ci(*args, **kwargs):
    """Moreira (2003) CLR confidence set — re-export of
    :func:`statspai.iv.weak_iv_ci.conditional_lr_ci`.
    """
    from .iv.weak_iv_ci import conditional_lr_ci as _impl
    return _impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# Lee-McCrary-Moreira-Porter (2022) tF adjustment
# ---------------------------------------------------------------------------

def tF_adjustment(first_stage_F: float, alpha: float = 0.05) -> float:
    """tF adjusted critical value (Lee, McCrary, Moreira & Porter 2022, AER).

    Alias for :func:`statspai.diagnostics.weak_iv.tF_critical_value`.
    Named after the ``tF`` terminology used in the paper and blog post so
    that ``sp.tF_adjustment(F)`` works as advertised.

    Parameters
    ----------
    first_stage_F
        First-stage effective F-statistic.
    alpha
        Two-sided significance level (currently only 0.05 is supported
        exactly; other levels are extrapolated by the underlying table).
    """
    from .diagnostics.weak_iv import tF_critical_value
    return tF_critical_value(first_stage_F, alpha=alpha)
