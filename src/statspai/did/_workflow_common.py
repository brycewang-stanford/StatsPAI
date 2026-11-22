"""The nine steps of :func:`statspai.did.workflow.did_workflow`.

Every step is a function ``run_<key>(wf, step)`` that fills in the step's
``status`` / ``finding`` / ``action`` / ``evidence`` and stores what later
steps need on the workflow.  No step estimates anything itself: each calls
the public estimator or diagnostic that already exists, so a number in the
workflow is the number that function returns.

Where a status depends on a benchmark, the benchmark is one the cited
literature states rather than a tuning constant:

* ``Mbar = 1`` in the relative-magnitude sensitivity analysis is Rambachan
  and Roth's (2023) reference case: a post-treatment violation as large as
  the largest one seen before treatment.
* The pre-test is called uninformative when a linear trend it would detect
  only 80 percent of the time is large enough to account for the whole
  estimate, the comparison Roth (2022) recommends reporting.
* Ten treated (or comparison) clusters is the threshold the staggered
  estimators in this package already warn at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import DiDWorkflow

#: Fewer treated or comparison clusters than this is flagged; the same
#: count at which ``_validate.warn_few_treated_clusters`` warns.
FEW_CLUSTERS = 10
#: Rambachan-Roth reference case for the relative-magnitude restriction.
RM_BENCHMARK = 1.0
#: The cluster jackknife refits once per cluster; run it unprompted only
#: where the cluster count makes the analytic interval doubtful.
JACKKNIFE_AUTO_MAX_CLUSTERS = 100
#: Event times reported on each side of treatment when ``event_window`` is
#: not given and the panel offers more. A reporting default, not a
#: finding: the joint pre-test loses power with every lead it adds, and
#: the cost of its power calculation and of the Rambachan-Roth sets grows
#: steeply with the number of event times (minutes at eight leads, no
#: answer at seventeen). The step that applies it says so.
EVENT_WINDOW_DEFAULT_SPAN = 5

AGG_TYPE = {
    "overall": "simple",
    "event": "dynamic",
    "group": "group",
    "calendar": "calendar",
}
ESTIMAND_TEXT = {
    "overall": "ATT averaged over all treated unit-periods",
    "event": "ATT averaged with equal weight over post-treatment event times",
    "group": "ATT averaged over cohorts (each cohort's average effect)",
    "calendar": "ATT averaged over calendar periods",
}
ESTIMATOR_LABEL = {
    "cs": "Callaway-Sant'Anna",
    "sa": "Sun-Abraham",
    "bjs": "Borusyak-Jaravel-Spiess imputation",
}
PT_TEXT = {
    "nevertreated": "parallel trends between each treated cohort and the "
    "never-treated units",
    "notyettreated": "parallel trends between each treated cohort and the "
    "units not yet treated at that date (never-treated units included)",
    "lastcohort": "parallel trends between each treated cohort and the "
    "last-treated cohort before it is treated",
}


def fmt(x: Any, digits: int = 4) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.{digits}g}"


def pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _ran(value: Any, what: str) -> Any:
    if value is None:  # pragma: no cover - steps run in protocol order
        raise RuntimeError(f"did_workflow: {what} is read before its step ran")
    return value


def facts_of(wf: "DiDWorkflow") -> Dict[str, Any]:
    """Design counts, available once the design step has run."""
    out: Dict[str, Any] = _ran(wf._facts, "the design audit")
    return out


def headline_of(wf: "DiDWorkflow") -> Dict[str, Any]:
    """The headline estimate, available once the estimate step has run."""
    out: Dict[str, Any] = _ran(wf._headline, "the estimate")
    return out


def path_of(wf: "DiDWorkflow") -> pd.DataFrame:
    """The event-study table, available once the event-study step has run."""
    out: pd.DataFrame = _ran(wf._path, "the event-study path")
    return out
