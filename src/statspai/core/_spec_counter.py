"""Session-level count of the specifications fitted to one outcome.

A p-value from the twentieth regression of the same outcome on the same
data is not the p-value of a pre-specified test: the search is part of the
procedure. Nothing here adjusts anything. Each user-level fit of
``sp.regress`` / ``sp.feols`` / ``sp.fepois`` / ``sp.hdfe_ols`` /
``sp.panel`` and each ``sp.spec_curve`` run records the specifications it
estimated, and the result carries how many distinct ones this Python
process (an MCP server included) has fitted for that outcome, so the
number can be reported next to the p-value.

Keys are cheap on purpose:

* the *family* is the outcome name plus the set of column names of the
  frame, so an unrelated dataset is another family;
* a *specification* is the entry point, the formula, the absorbed effects,
  the variance estimator, the weights and the number of rows used, so the
  same call repeated (bootstrap replicates, re-runs) counts once while a
  new control set, a new subsample or another standard error counts as a
  new specification;
* within a family the specifications are counted per *dataset*. A search
  is many models of one set of observations, so a fit joins the count of
  the fits whose outcome values it shares, and starts its own count when
  it shares none. Fresh draws in a Monte Carlo loop and the disjoint
  groups of a by-group loop are then separate datasets (the number of
  rows used differs from draw to draw whenever a draw has missing or
  truncated values, which used to make each draw a "new specification"),
  while a subsample of data already fitted still adds to that data's
  count, down to a subsample of about 2 percent of its rows. Sharing is
  read from a small sketch of the outcome column, the smallest hashes of
  its distinct values, in one vectorised pass over the column. Subgroups
  fitted one by one without the pooled sample ever being fitted share
  nothing with one another and are not counted together. An outcome
  with few distinct values (a 0/1 indicator, a
  small count) cannot tell two datasets apart and is counted as one, as
  before; so is an outcome that is not a column of the frame.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases

__all__ = [
    "record_specifications",
    "specification_count",
    "reset_specification_counter",
    "reset_spec_counter",
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
#: Datasets remembered per family (least recently fitted dropped first).
_MAX_DATASETS = 128
#: A fit asks with the ``_SKETCH`` smallest hashes of its distinct outcome
#: values and leaves its ``_STORED`` smallest behind. A subsample keeping a
#: share f of the rows of data fitted earlier then shares about
#: ``min(64, 256 f)`` of the hashes it asks with: it is recognised down to
#: about 2 percent of the rows.
_SKETCH = 64
_STORED = 256
_MAX_DATASET_HASHES = 1024
#: Shared hashes from which two fits are on the same observations. Fits on
#: unrelated continuous data share none, and a common value or two (zeros)
#: does not join them.
_MIN_SHARED = 4

_LOCK = threading.Lock()


class _Family:
    """The datasets of one (outcome, columns) family and their specifications."""

    __slots__ = ("datasets", "owner", "next_id")

    def __init__(self) -> None:
        #: dataset id -> (specifications, hashes of its outcome values)
        self.datasets: "OrderedDict[int, Tuple[Set[str], Set[int]]]" = OrderedDict()
        #: outcome-value hash -> the dataset it was last seen in
        self.owner: Dict[int, int] = {}
        self.next_id = 0

    def dataset(self, sketch: Optional[np.ndarray], create: bool) -> Optional[Set[str]]:
        """Specifications of the dataset ``sketch`` belongs to."""
        if sketch is None:
            key, hashes = -1, []  # no sketch: one dataset for the family
        else:
            hashes = [int(h) for h in sketch]
            hits: Dict[int, int] = {}
            for h in hashes[:_SKETCH]:
                d = self.owner.get(h)
                if d is not None:
                    hits[d] = hits.get(d, 0) + 1
            need = max(1, min(_MIN_SHARED, min(len(hashes), _SKETCH) // 2))
            key = -2
            if hits:
                best = max(hits, key=lambda d: hits[d])
                if hits[best] >= need:
                    key = best
        entry = self.datasets.get(key) if key != -2 else None
        if entry is None:
            if not create:
                return None
            if key == -2:
                key = self.next_id
                self.next_id += 1
            entry = (set(), set())
            self.datasets[key] = entry
            while len(self.datasets) > _MAX_DATASETS:
                old, (_, gone) = self.datasets.popitem(last=False)
                for h in gone:
                    if self.owner.get(h) == old:
                        del self.owner[h]
        elif create:
            self.datasets.move_to_end(key)
        if create:
            seen = entry[1]
            for h in hashes:
                if len(seen) < _MAX_DATASET_HASHES or h in seen:
                    seen.add(h)
                    self.owner[h] = key
        return entry[0]

    def n_specifications(self) -> int:
        return sum(len(specs) for specs, _ in self.datasets.values())


_FAMILIES: "OrderedDict[Tuple[str, Tuple[str, ...]], _Family]" = OrderedDict()


def _family(data: Any, outcome: str) -> Tuple[str, Tuple[str, ...]]:
    cols: Tuple[str, ...] = ()
    if isinstance(data, pd.DataFrame):
        cols = tuple(sorted(str(c) for c in data.columns))
    return (str(outcome), cols)


def _sketch(data: Any, outcome: str) -> Optional[np.ndarray]:
    """The ``_STORED`` smallest hashes of the distinct outcome values, sorted.

    Two frames holding the same observations, or one a subsample of the
    other, share most of these; frames of unrelated values share none.
    None when the outcome is not a column (a transformed outcome such as
    ``log(y)``) or cannot be hashed.
    """
    if not isinstance(data, pd.DataFrame) or outcome not in data.columns:
        return None
    col = data[outcome]
    if not isinstance(col, pd.Series):
        return None  # duplicated column name
    try:
        values = col.to_numpy()
        hashed = pd.util.hash_array(values[~pd.isna(values)])
    except (TypeError, ValueError):
        return None
    if hashed.size == 0:
        return None
    if hashed.size > 8 * _STORED:
        # the smallest couple of thousand are enough to find 256 distinct
        # ones unless the column is mostly ties, when there are few to find
        hashed = np.partition(hashed, 8 * _STORED)[: 8 * _STORED]
    smallest: np.ndarray = np.unique(hashed)[:_STORED]
    return smallest


def record_specifications(
    data: Any, outcome: str, specs: Iterable[str]
) -> Dict[str, Any]:
    """Add ``specs`` to the outcome's family and return the running count.

    Returns ``{"specifications_tried": k, "outcome": outcome, "scope": ...}``
    where ``k`` counts distinct specifications, this call's included.
    """
    key = _family(data, outcome)
    sketch = _sketch(data, str(outcome))
    with _LOCK:
        family = _FAMILIES.get(key)
        if family is None:
            family = _Family()
            _FAMILIES[key] = family
            while len(_FAMILIES) > _MAX_FAMILIES:
                _FAMILIES.popitem(last=False)
        else:
            _FAMILIES.move_to_end(key)
        seen = family.dataset(sketch, create=True)
        assert seen is not None
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
            "session on these observations or a sample sharing them, in a "
            "frame with the same columns (sp.regress / feols / fepois / "
            "hdfe_ols / panel / spec_curve)"
        ),
    }


def specification_count(data: Any, outcome: str) -> int:
    """Distinct specifications recorded so far for ``outcome`` on ``data``."""
    sketch = _sketch(data, str(outcome))
    with _LOCK:
        family = _FAMILIES.get(_family(data, outcome))
        seen = None if family is None else family.dataset(sketch, create=False)
        return 0 if seen is None else len(seen)


def reset_specification_counter(outcome: Optional[str] = None) -> None:
    """Forget the recorded specifications (of one outcome, or all)."""
    with _LOCK:
        if outcome is None:
            _FAMILIES.clear()
            return
        for key in [k for k in _FAMILIES if k[0] == str(outcome)]:
            del _FAMILIES[key]


@accepts_aliases(outcome="y")
def reset_spec_counter(y: Optional[str] = None) -> Dict[str, Any]:
    """Forget the specifications this session has recorded.

    Every fit of ``sp.regress`` / ``sp.feols`` / ``sp.fepois`` /
    ``sp.hdfe_ols`` / ``sp.panel`` and every ``sp.spec_curve`` run adds its
    specifications to a per-process count, and a result's
    ``violations()`` reports how many distinct specifications of the same
    outcome were fitted before it (``test='specification_search'``). Call
    this when a new, pre-specified analysis starts in the same Python
    process or MCP server, so earlier exploration is not counted against
    it. Nothing else is reset.

    Parameters
    ----------
    y : str, optional
        Forget only the specifications of this outcome (every frame layout
        it was fitted on). ``None`` forgets all of them. ``outcome=`` is
        accepted as an alias.

    Returns
    -------
    dict
        ``{"outcome": y, "families_cleared": k,
        "specifications_cleared": m}``: how many (outcome, column set)
        families and distinct specifications were forgotten.

    Examples
    --------
    >>> import statspai as sp
    >>> out = sp.reset_spec_counter()
    >>> sorted(out)
    ['families_cleared', 'outcome', 'specifications_cleared']
    >>> sp.reset_spec_counter()["specifications_cleared"]
    0
    """
    outcome = y
    if outcome is not None and not isinstance(outcome, str):
        from ..exceptions import MethodIncompatibility

        raise MethodIncompatibility(
            f"reset_spec_counter(y={outcome!r}): the outcome must be a "
            "column name (str) or None.",
            recovery_hint="Pass the outcome's column name, or None to reset "
            "every outcome.",
            diagnostics={"outcome_type": type(outcome).__name__},
        )
    with _LOCK:
        keys = [k for k in _FAMILIES if outcome is None or k[0] == str(outcome)]
        n_specs = sum(_FAMILIES[k].n_specifications() for k in keys)
        for k in keys:
            del _FAMILIES[k]
    return {
        "outcome": outcome,
        "families_cleared": len(keys),
        "specifications_cleared": int(n_specs),
    }


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
            "count is per process; call sp.reset_spec_counter() (or start "
            "a new session) before a new, pre-specified analysis."
        ),
        "alternatives": ["sp.spec_curve", "sp.romano_wolf", "sp.adjust_pvalues"],
    }
