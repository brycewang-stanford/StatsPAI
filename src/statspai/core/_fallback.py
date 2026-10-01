"""Loud fallbacks for the numerical core.

A numerical routine that catches a failure and substitutes something else
(a marginal mean for a propensity score, NaN for a standard error, one
fewer placebo unit) changes what the caller gets. CLAUDE.md section 3.7
requires that substitution to be visible. These helpers are the uniform
replacement for ``except Exception: <substitute silently>`` inside
estimators; orchestration code uses
:func:`statspai.workflow._degradation.record_degradation` instead.

Nothing here changes a number on a healthy run. The helpers only fire on
paths that previously failed without a trace.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Type

__all__ = ["warn_fallback", "warn_dropped"]


def warn_fallback(
    what: str,
    exc: Optional[BaseException],
    fallback: str,
    *,
    category: Type[Warning] = RuntimeWarning,
    stacklevel: int = 3,
) -> None:
    """Warn that ``what`` failed and say what was used in its place.

    Parameters
    ----------
    what : str
        The step that failed, e.g. ``"focal_cate propensity model"``.
    exc : BaseException or None
        The caught exception. ``None`` when the failure was detected
        without one (a non-finite value, say).
    fallback : str
        What the caller gets instead, written as a clause that completes
        "... failed; <fallback>."
    category : type, default RuntimeWarning
        Warning category.
    stacklevel : int, default 3
        Forwarded to :func:`warnings.warn`; 3 points at the caller of the
        function that called this helper.
    """
    reason = f" ({type(exc).__name__}: {exc})" if exc is not None else ""
    warnings.warn(
        f"{what} failed{reason}; {fallback}.",
        category,
        stacklevel=stacklevel,
    )


def warn_dropped(
    what: str,
    dropped: Sequence[Any],
    n_total: int,
    consequence: str,
    *,
    errors: Optional[Sequence[BaseException]] = None,
    category: Type[Warning] = RuntimeWarning,
    stacklevel: int = 3,
    max_listed: int = 8,
) -> None:
    """Warn that some units of work were dropped from an aggregate.

    Does nothing when ``dropped`` is empty, so it can be called
    unconditionally after a loop.

    Parameters
    ----------
    what : str
        Plural noun for the units, e.g. ``"cohorts"`` or ``"placebo fits"``.
    dropped : sequence
        Labels of the dropped units.
    n_total : int
        Number of units attempted.
    consequence : str
        What the drop does to the result, written as a full clause.
    errors : sequence of BaseException, optional
        Caught exceptions; the first is quoted in the message.
    category, stacklevel
        As for :func:`warn_fallback`.
    max_listed : int, default 8
        How many labels to spell out before truncating.
    """
    n_dropped = len(dropped)
    if n_dropped == 0:
        return
    labels: List[str] = [str(d) for d in list(dropped)[:max_listed]]
    listed = ", ".join(labels)
    if n_dropped > max_listed:
        listed += f", ... ({n_dropped - max_listed} more)"
    first = ""
    if errors:
        e0 = errors[0]
        first = f" First error: {type(e0).__name__}: {e0}."
    warnings.warn(
        f"{n_dropped} of {n_total} {what} dropped [{listed}]; "
        f"{consequence}.{first}",
        category,
        stacklevel=stacklevel,
    )
