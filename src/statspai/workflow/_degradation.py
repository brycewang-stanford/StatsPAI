"""Structured degradation tracking for the workflow / paper layer.

The orchestration code in :mod:`statspai.workflow.paper` and
:mod:`statspai.workflow.causal_workflow` historically wrapped optional
sub-steps (CI rendering, DAG appendix, citation extraction, provenance
attachment, the optional ``compare_estimators / sensitivity_panel /
cate`` pipeline tail) in ``try / except Exception: pass``.  That keeps a
single broken sub-step from crashing the whole pipeline, but it also
silently weakens the artifact: a paper draft can lose its CI, its DAG,
or its references and the user gets no signal.

This module gives those sites a uniform replacement that satisfies
CLAUDE.md §3.7 ("失败要响亮"):

* ``WorkflowDegradedWarning`` is emitted via :func:`warnings.warn`, so
  the failure shows up in pytest, in notebooks, and in CLI output.
* The exception payload is appended to ``target.degradations`` (a list
  of dicts) when ``target`` exposes such an attribute, so downstream
  consumers (``PaperDraft``, ``CausalWorkflow``) can introspect what
  was skipped and why without re-running the pipeline.

This is **not** a substitute for raising in the numerical core — it is
purely for the orchestration glue where best-effort fallback is the
desired UX.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional


__all__ = [
    "WorkflowDegradedWarning",
    "record_degradation",
]


class WorkflowDegradedWarning(UserWarning):
    """Emitted when an orchestration sub-step fails and is being skipped.

    Subclasses :class:`UserWarning` so it shows up by default and can be
    filtered selectively (``warnings.filterwarnings('error',
    category=WorkflowDegradedWarning)``) when callers want strict mode.
    """


def record_degradation(
    target: Any,
    *,
    section: str,
    exc: BaseException,
    detail: Optional[str] = None,
    stacklevel: int = 3,
) -> Dict[str, Any]:
    """Warn loudly and append a structured entry describing the failure.

    Parameters
    ----------
    target
        Where to record the entry.  Three accepted shapes:

        * an object with a ``degradations`` list attribute (typically a
          :class:`PaperDraft` or :class:`CausalWorkflow`) — the entry is
          appended to that list;
        * a bare ``list`` — the entry is appended directly;
        * ``None`` — only the warning fires, nothing is stored.
    section
        Human-readable label for what was being attempted, e.g.
        ``"covariate balance table"`` or ``"causal DAG appendix"``.
        Used both in the warning message and as the ``section`` key of
        the recorded entry.
    exc
        The caught exception.
    detail
        Optional extra context (for instance the column name that
        triggered the failure).  Surfaced in the warning message and
        stored under ``detail`` in the recorded entry.
    stacklevel
        Forwarded to :func:`warnings.warn` so the warning points at the
        caller of ``record_degradation``, not at this module.

    Returns
    -------
    dict
        The entry that was (or would have been) appended.  Always
        returned so callers can include it in unit-test assertions.
    """
    entry: Dict[str, Any] = {
        "section": section,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    if detail:
        entry["detail"] = detail

    if isinstance(target, list):
        target.append(entry)
    elif target is not None:
        bag = getattr(target, "degradations", None)
        if isinstance(bag, list):
            bag.append(entry)

    msg = (
        f"Workflow sub-step '{section}' was skipped: "
        f"{type(exc).__name__}: {exc}"
    )
    if detail:
        msg = f"{msg} ({detail})"
    warnings.warn(msg, WorkflowDegradedWarning, stacklevel=stacklevel)

    return entry


def ensure_degradation_bag(target: Any) -> List[Dict[str, Any]]:
    """Return the ``degradations`` list on ``target``, creating it if missing.

    Convenience for objects whose dataclass field defaults to a list but
    that may have been constructed by hand in older callers / tests.
    """
    bag = getattr(target, "degradations", None)
    if not isinstance(bag, list):
        bag = []
        try:
            setattr(target, "degradations", bag)
        except (AttributeError, TypeError):
            pass
    return bag
