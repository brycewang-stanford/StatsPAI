"""Shared checks on survival inputs (duration, event, weights).

2026-10-20 probe: ``sp.kaplan_meier`` / ``sp.cox`` / ``sp.rmst`` accepted
negative durations and an event column coded ``0/1/2`` (every non-zero
value read as an event, which is Stata's ``stset failure()`` rule but the
opposite of R's ``Surv(time, status)`` 1/2 coding, where 1 is censored),
and dropped rows with a missing weight without a word.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility


def check_duration_event(T: Any, E: Any, function: str) -> None:
    """Refuse negative durations and event codes other than 0/1."""
    T = np.asarray(T, dtype=float)
    neg = int(np.sum(T < 0))
    if neg:
        raise DataInsufficient(
            f"{function}: {neg} duration(s) are negative.",
            recovery_hint="Durations are times since origin; fix or drop the "
            "rows (Stata stset excludes them, R Surv refuses them).",
            diagnostics={"n_negative": neg},
        )
    try:
        Ev = np.asarray(E, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{function}: the event indicator must be numeric 0/1 (or bool).",
            recovery_hint="Recode it, e.g. event = (status == 'died').astype(int).",
        ) from exc
    vals = np.unique(Ev[np.isfinite(Ev)])
    if not np.isin(vals, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"{function}: the event indicator must be 0/1; found values "
            f"{vals.tolist()[:10]}.",
            recovery_hint="Recode it explicitly (event = (status == k)). R's "
            "Surv() reads 1/2 as censored/event while Stata's stset reads any "
            "non-zero value as an event, so the code is not guessed.",
            diagnostics={"values": vals.tolist()[:10]},
        )


def warn_missing_weights(
    data: pd.DataFrame, weights: Optional[str], function: str
) -> None:
    """Warn before rows with a missing weight are dropped."""
    from ..core._validate import warn_missing_weight_rows

    warn_missing_weight_rows(data, weights, function=function)
