"""``estat simple | group | calendar | event`` after a staggered DiD fit.

Stata's ``csdid`` and ``jwdid`` both report their aggregations through
``estat``. The two results aggregate differently, so the dispatch is on what
the result carries: ATT(g,t) cells with influence functions go to
:func:`statspai.aggte`, an extended-TWFE regression to
:func:`statspai.etwfe_emfx`.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..exceptions import MethodIncompatibility

__all__ = ["DID_AGGREGATIONS", "estat_did_aggregate"]

#: ``estat`` subcommand -> ``sp.aggte(type=)``
DID_AGGREGATIONS = {
    "simple": "simple",
    "group": "group",
    "calendar": "calendar",
    "event": "dynamic",
}


def _window(window: Optional[Sequence[float]]) -> Optional[tuple]:
    if window is None:
        return None
    try:
        lo, hi = (float(v) for v in window)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "estat event: window= takes two event times, (first, last).",
            recovery_hint="For example window=(-4, 5).",
            diagnostics={"window": repr(window)},
        ) from exc
    return (min(lo, hi), max(lo, hi))


def estat_did_aggregate(
    result: Any,
    kind: str,
    *,
    window: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
) -> Any:
    """Aggregate a staggered DiD result the way ``estat <kind>`` does.

    After :func:`statspai.callaway_santanna` the call is
    ``sp.aggte(result, type=...)`` with ``csdid``'s conventions: the cohort
    shares of ``estat group`` are held fixed (``share_variance=False``) and
    ``window=(a, b)`` keeps the event times ``a..b``. After
    :func:`statspai.jwdid` / :func:`statspai.etwfe` it is
    ``sp.etwfe_emfx(result, type=...)``, with the pre-treatment effects
    listed when the model estimated them (``never``).
    """
    info = getattr(result, "model_info", None) or {}
    span = _window(window)
    if span is not None and kind != "event":
        raise MethodIncompatibility(
            f"estat {kind}: window= selects event times and applies to "
            "estat event only.",
            recovery_hint="Drop window=, or ask for the 'event' aggregation.",
        )
    if "treatment_params" in info and "hettype" in info:
        from ..did import etwfe_emfx

        if span is not None:
            raise MethodIncompatibility(
                "estat event after jwdid: window= is not carried to " "sp.etwfe_emfx.",
                recovery_hint="Aggregate with sp.etwfe_emfx(result, "
                "type='event') and select the rows of .detail.",
            )
        return etwfe_emfx(
            result,
            type=kind,
            alpha=alpha,
            include_leads=kind == "event" and info.get("cgroup") == "never",
        )
    if "_unit_cohorts" in info and "cohort_sizes" in info:
        from ..did import aggte

        kwargs: dict = {"type": DID_AGGREGATIONS[kind], "alpha": alpha}
        if kind == "group":
            kwargs["share_variance"] = False
        if span is not None:
            kwargs["min_e"], kwargs["max_e"] = span
        return aggte(result, **kwargs)
    raise MethodIncompatibility(
        f"estat {kind} aggregates a staggered difference-in-differences fit; "
        "this result is neither a Callaway-Sant'Anna nor an extended-TWFE "
        "one.",
        recovery_hint="Fit with sp.callaway_santanna(...) or sp.jwdid(...) "
        "first. Other event-study estimators report their aggregation in "
        "the result itself (.detail, model_info['event_study']).",
        diagnostics={"method": getattr(result, "method", None)},
    )
