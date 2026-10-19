"""Session-level count of the specifications fitted to one outcome.

A p-value from the twentieth regression of the same outcome on the same
data is not the p-value of a pre-specified test: the search is part of the
procedure. Nothing here adjusts anything. Each user-level fit of
``sp.regress`` / ``sp.feols`` / ``sp.fepois`` / ``sp.hdfe_ols`` /
``sp.panel`` and each ``sp.spec_curve`` run records the specifications it
estimated, and the result carries how many distinct ones this Python
process (an MCP server included) has fitted for that outcome, so the
number can be reported next to the p-value.

Keys are cheap on purpose (no hashing of the data):

* the *family* is the outcome name plus the set of column names of the
  frame, so a simulation loop over fresh draws of the same frame layout
  stays one family, and an unrelated dataset is another;
* a *specification* is the entry point, the formula, the absorbed effects,
  the variance estimator, the weights and the number of rows used, so the
  same call repeated (bootstrap replicates, re-runs) counts once while a
  new control set, a new subsample or another standard error counts as a
  new specification.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import pandas as pd

__all__ = [
    "record_specifications",
    "specification_count",
    "reset_specification_counter",
    "multiplicity_violation",
    "MULTIPLICITY_INFO",
    "MULTIPLICITY_WARNING",
]

#: Distinct specifications of one outcome from which the count is reported
#: as a violation (``info``), and from which it is a ``warning``.
MULTIPLICITY_INFO = 5
MULTIPLICITY_WARNING = 20

_MAX_FAMILIES = 256
_MAX_SPECS = 100_000

_LOCK = threading.Lock()
_FAMILIES: "OrderedDict[Tuple[str, Tuple[str, ...]], Set[str]]" = OrderedDict()


def _family(data: Any, outcome: str) -> Tuple[str, Tuple[str, ...]]:
    cols: Tuple[str, ...] = ()
    if isinstance(data, pd.DataFrame):
        cols = tuple(sorted(str(c) for c in data.columns))
    return (str(outcome), cols)


def record_specifications(
    data: Any, outcome: str, specs: Iterable[str]
) -> Dict[str, Any]:
    """Add ``specs`` to the outcome's family and return the running count.

    Returns ``{"specifications_tried": k, "outcome": outcome, "scope": ...}``
    where ``k`` counts distinct specifications, this call's included.
    """
    key = _family(data, outcome)
    with _LOCK:
        seen = _FAMILIES.get(key)
        if seen is None:
            seen = set()
            _FAMILIES[key] = seen
            while len(_FAMILIES) > _MAX_FAMILIES:
                _FAMILIES.popitem(last=False)
        else:
            _FAMILIES.move_to_end(key)
        for s in specs:
            if len(seen) >= _MAX_SPECS:
                break
            seen.add(str(s))
        k = len(seen)
    return {
        "specifications_tried": k,
        "outcome": str(outcome),
        "scope": (
            "distinct specifications of this outcome fitted in this Python "
            "session on a frame with the same columns (sp.regress / feols / "
            "fepois / hdfe_ols / panel / spec_curve)"
        ),
    }


def specification_count(data: Any, outcome: str) -> int:
    """Distinct specifications recorded so far for ``outcome`` on ``data``."""
    with _LOCK:
        return len(_FAMILIES.get(_family(data, outcome), ()))


def reset_specification_counter(outcome: Optional[str] = None) -> None:
    """Forget the recorded specifications (of one outcome, or all)."""
    with _LOCK:
        if outcome is None:
            _FAMILIES.clear()
            return
        for key in [k for k in _FAMILIES if k[0] == str(outcome)]:
            del _FAMILIES[key]


def multiplicity_violation(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The violation dict for a multiplicity record, or ``None`` below the bar."""
    k = int(record.get("specifications_tried", 0) or 0)
    if k < MULTIPLICITY_INFO:
        return None
    y = record.get("outcome", "")
    return {
        "kind": "inference",
        "severity": "warning" if k >= MULTIPLICITY_WARNING else "info",
        "test": "specification_search",
        "value": k,
        "threshold": MULTIPLICITY_INFO,
        "message": (
            f"This is one of {k} distinct specifications of '{y}' fitted in "
            "this session. Its p-value and confidence interval treat it as "
            "the only test; they do not account for the search."
        ),
        "recovery_hint": (
            "Report every specification tried (sp.spec_curve) or adjust the "
            "family of p-values (sp.romano_wolf, sp.adjust_pvalues). The "
            "count is per process; start a new session for a new, "
            "pre-specified analysis."
        ),
        "alternatives": ["sp.spec_curve", "sp.romano_wolf", "sp.adjust_pvalues"],
    }
