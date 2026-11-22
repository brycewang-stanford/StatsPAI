"""Option checking for ``sp.did_workflow``."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np

from ..exceptions import MethodIncompatibility
from ._workflow_spec import STEP_KEYS

_FN = "sp.did_workflow"
_TARGETS = ("overall", "event", "group", "calendar")
_ESTIMATORS = {
    "cs": "cs",
    "callaway_santanna": "cs",
    "sa": "sa",
    "sun_abraham": "sa",
    "bjs": "bjs",
    "did_imputation": "bjs",
}
_CONTROL_GROUPS = {
    "cs": ("nevertreated", "notyettreated"),
    "sa": ("nevertreated", "lastcohort"),
    "bjs": ("notyettreated",),
}
_SKIPPABLE = ("twfe", "pretrends", "sensitivity", "robustness")


def check_options(
    *,
    target: Optional[str],
    estimator: Optional[str],
    control_group: Optional[str],
    anticipation: Optional[int],
    event_window: Optional[Sequence[int]],
    alpha: float,
    through: Optional[str],
    skip: Any,
    jackknife: Optional[bool],
    estimator_options: Optional[Dict[str, Any]],
    seed: int,
) -> Tuple[Dict[str, Any], Set[str]]:
    explicit: Set[str] = set()

    def bad(message: str, hint: str, **diag: Any) -> MethodIncompatibility:
        return MethodIncompatibility(
            f"{_FN}: {message}", recovery_hint=hint, diagnostics=diag
        )

    if target is not None:
        explicit.add("target")
    target = "overall" if target is None else str(target).lower()
    if target not in _TARGETS:
        raise bad(
            f"unknown target {target!r}.",
            "target is 'overall', 'event', 'group' or 'calendar'.",
            target=target,
            valid=list(_TARGETS),
        )

    if estimator is not None:
        explicit.add("estimator")
    key = "cs" if estimator is None else str(estimator).lower()
    if key not in _ESTIMATORS:
        raise bad(
            f"unknown estimator {estimator!r}.",
            "estimator is 'cs' (Callaway-Sant'Anna), 'sa' (Sun-Abraham) or "
            "'bjs' (imputation). Other estimators of the family are "
            "compared in the robustness step; fit them directly with "
            "sp.did(method=...).",
            estimator=str(estimator),
            valid=sorted(set(_ESTIMATORS.values())),
        )
    est = _ESTIMATORS[key]
    if target in ("group", "calendar") and est != "cs":
        raise bad(
            f"target={target!r} is an aggregation of group-time effects and "
            f"needs estimator='cs'; got {est!r}.",
            "Use estimator='cs', or target='overall' / 'event'.",
            target=target,
            estimator=est,
        )

    allowed = _CONTROL_GROUPS[est]
    implied = False
    if control_group is None:
        control = allowed[0]
        implied = est == "bjs"
    else:
        explicit.add("control_group")
        control = str(control_group).lower()
        if control not in allowed:
            raise bad(
                f"control_group={control_group!r} is not a comparison group "
                f"of estimator={est!r}.",
                f"With estimator={est!r} use control_group in {list(allowed)}; "
                "the comparison group is part of the estimator's "
                "identifying assumption.",
                control_group=str(control_group),
                estimator=est,
                valid=list(allowed),
            )

    if anticipation is not None:
        explicit.add("anticipation")
    k = 0 if anticipation is None else anticipation
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
        raise bad(
            f"anticipation must be a non-negative integer, got {anticipation!r}.",
            "anticipation=0 means no response before treatment starts.",
        )
    k = int(k)
    if k and est == "bjs":
        raise bad(
            "anticipation is not an option of estimator='bjs' here: "
            "sp.did_imputation's shift= moves the treatment date, which "
            "changes the estimand.",
            "Use estimator='cs' or 'sa', or pass a cohort column already "
            "shifted by the anticipation window.",
            anticipation=k,
        )

    window: Optional[Tuple[int, int]] = None
    if event_window is not None:
        explicit.add("event_window")
        try:
            lo, hi = (int(v) for v in event_window)
        except (TypeError, ValueError):
            raise bad(
                f"event_window must be a (first, last) pair of event times, "
                f"got {event_window!r}.",
                "e.g. event_window=(-4, 4).",
            ) from None
        if lo > hi or hi < 0:
            raise bad(
                f"event_window={event_window!r} has no post-treatment event " "time.",
                "The window is (first, last) with last >= 0, e.g. (-4, 4).",
            )
        window = (lo, hi)

    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.floating))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise bad(f"alpha must be in (0, 1), got {alpha!r}.", "e.g. alpha=0.05.")

    if through is not None and through not in STEP_KEYS:
        raise bad(
            f"through={through!r} is not a step.",
            f"through is one of {', '.join(STEP_KEYS)}, or None to build the "
            "plan without running it.",
            valid=list(STEP_KEYS),
        )

    if skip is None:
        skip_set: Tuple[str, ...] = ()
    else:
        skip_set = (skip,) if isinstance(skip, str) else tuple(skip)
        unknown = [s for s in skip_set if s not in _SKIPPABLE]
        if unknown:
            raise bad(
                f"skip={unknown!r} cannot be skipped.",
                f"Only {list(_SKIPPABLE)} are optional; the other steps are "
                "what the later ones are computed from.",
                valid=list(_SKIPPABLE),
            )

    if jackknife is not None and not isinstance(jackknife, (bool, np.bool_)):
        raise bad(
            f"jackknife must be True, False or None, got {jackknife!r}.",
            "None runs the cluster jackknife when clusters are few.",
        )
    if estimator_options is not None and not isinstance(estimator_options, dict):
        raise bad(
            "estimator_options must be a dict of keyword arguments for the "
            "estimator.",
            "e.g. estimator_options={'allow_unbalanced_panel': True}.",
        )

    options = {
        "target": target,
        "estimator": est,
        "control_group": control,
        "control_group_implied": implied,
        "anticipation": k,
        "event_window": window,
        "alpha": float(alpha),
        "skip": skip_set,
        "jackknife": None if jackknife is None else bool(jackknife),
        "estimator_options": dict(estimator_options or {}),
        "seed": int(seed),
    }
    return options, explicit
