"""Catch a DiD column passed in the wrong role.

In the DiD family the keyword ``group`` is the **unit identifier**
(canonical spelling ``id=``): ``sp.did_imputation``, ``sp.gardner_did``,
``sp.stacked_did``, ``sp.etwfe``, ``sp.wooldridge_did`` and
``sp.did_multiplegt``. ``sp.drdid`` was the exception, where ``group`` was
the **binary treatment-group indicator**; its parameter is now ``treat``
and the keyword ``group=`` is deprecated (removed no earlier than 1.42).

Passing the treatment indicator where the unit id belongs used to run.
The checks here make the swap loud. Each rule is chosen so that it cannot
turn down a call that is correct:

* :func:`check_unit_is_not_treatment` raises when the 0/1 "unit" column
  has a cohort that varies within it. ``require_constant_cohort`` refuses
  that right after in the estimators that call it, so there it changes
  the message, not the outcome; in ``sp.wooldridge_did``, which has no
  such check, it refuses a frame in which a unit has two first-treatment
  dates, which no correct call has. Where the swapped call and a
  legitimate one look the same (a 0/1 group column with several rows per
  period, which is also a two-group repeated cross-section with group
  fixed effects), it warns. ``sp.did_multiplegt`` has no cohort; there it
  raises when the treatment varies within a (group, period) cell of a 0/1
  group column. The estimator pairs each group's rows in consecutive
  periods, so a treatment that differs between those rows has no defined
  switchers; with a treatment constant within cells (a genuine two-group
  design) it warns.
* :func:`check_binary_treatment` raises only when the column does not take
  exactly two values, which ``sp.drdid`` already refused before this
  module existed. A two-valued column that is not coded 0/1 still runs,
  with a warning that names the value taken as treated.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility, StatsPAIWarning
from ._validate import normalized_cohort

__all__ = [
    "ArgumentRoleWarning",
    "check_unit_is_not_treatment",
    "check_binary_treatment",
]


class ArgumentRoleWarning(StatsPAIWarning):
    """A column looks like it was passed in the wrong argument."""

    code = "argument_role"


def _is_zero_one(values: List[Any]) -> bool:
    """True when every value is 0 / 1 (numbers or booleans)."""
    if not values:
        return False
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.all((arr == 0.0) | (arr == 1.0)))


def check_unit_is_not_treatment(
    data: pd.DataFrame,
    *,
    unit: str,
    time: str,
    cohort: Optional[str],
    function: str,
    cell_constant: Optional[str] = None,
    example_tail: Optional[str] = None,
    stacklevel: int = 3,
) -> None:
    """Refuse or flag a 0/1 treatment column passed as the unit identifier.

    Parameters
    ----------
    data : pd.DataFrame
        The estimation frame (rows with missing keys already dropped).
    unit, time : str
        The columns the caller passed as unit id and period.
    cohort : str or None
        The cohort (first-treatment period) column, or ``None`` for an
        estimator without one (``sp.did_multiplegt``); then only the
        repeated-rows rule applies.
    function : str
        Public name used in messages.
    cell_constant : str, optional
        A column the estimator needs constant within each (unit, period)
        cell (``sp.did_multiplegt``'s treatment). With a 0/1 unit column
        and repeated rows, a value that varies within a cell raises
        instead of warning: such a call has no defined result.
    example_tail : str, optional
        The end of the example call in the recovery hint (defaults to
        ``cohort=<cohort>``).
    stacklevel : int, default 3
        Passed to :func:`warnings.warn`.
    """
    values = list(pd.unique(data[unit].dropna()))
    if len(values) != 2 or not _is_zero_one(values):
        return
    short = function.split(".")[-1]
    tail = example_tail if example_tail is not None else f"cohort={cohort!r}"
    hint = (
        f"In {function}, group= is the unit identifier (canonical spelling "
        "id=), not the treatment indicator. Pass the column that identifies "
        f"each unit, e.g. {short}(data, y=..., id='unit', time=..., {tail})."
    )
    diagnostics = {
        "function": function,
        "unit": unit,
        "cohort": cohort,
        "unit_values": sorted(float(v) for v in values),
    }
    if cohort is not None:
        g = normalized_cohort(data[cohort])
        varying = g.groupby(data[unit].to_numpy()).nunique(dropna=False)
        if bool((varying > 1).any()):
            raise MethodIncompatibility(
                f"{function}: the unit column {unit!r} takes only the values "
                f"0 and 1, and the cohort column {cohort!r} varies within "
                f"each of them. {unit!r} looks like a treatment indicator, "
                "not a unit identifier.",
                recovery_hint=hint,
                diagnostics=diagnostics,
            )
    if not bool(data.duplicated(subset=[unit, time]).any()):
        return
    if cell_constant is not None:
        spread = data.groupby([unit, time])[cell_constant].nunique(dropna=False)
        if bool((spread > 1).any()):
            raise MethodIncompatibility(
                f"{function}: the unit column {unit!r} takes only the values "
                f"0 and 1, with several rows per ({unit!r}, {time!r}), and "
                f"{cell_constant!r} varies within those cells. {function} "
                "pairs each unit's rows in consecutive periods, so rows of "
                "different units would be compared and the switchers are "
                f"undefined. {unit!r} looks like a treatment indicator, not "
                "a unit identifier.",
                recovery_hint=hint
                + " If the units really are two groups observed through "
                "individuals, collapse the data to one row per group and "
                "period first.",
                diagnostics={**diagnostics, "cell_constant": cell_constant},
            )
    warnings.warn(
        ArgumentRoleWarning(
            f"{function}: the unit column {unit!r} takes only the values 0 "
            f"and 1, with several rows per ({unit!r}, {time!r}). If "
            f"{unit!r} is the treatment indicator, the fit has two "
            "'units' and two clusters, and its standard errors are "
            "meaningless. It is kept as given, because a two-group "
            "repeated cross-section with group fixed effects looks the "
            "same.",
            recovery_hint=hint
            + " If two group fixed effects are intended, pass cluster= "
            "explicitly.",
            diagnostics={"function": function, "unit": unit},
        ),
        stacklevel=stacklevel,
    )


def check_binary_treatment(
    data: pd.DataFrame,
    column: str,
    *,
    function: str,
    id: Optional[str] = None,
    stacklevel: int = 3,
) -> List[Any]:
    """Validate a two-arm treatment-group column; return its sorted values.

    The larger value (in sort order) is the treated arm, as before.

    Raises
    ------
    MethodIncompatibility
        When the column does not take exactly two non-missing values.
    """
    values = sorted(pd.unique(data[column].dropna()))
    # Python scalars, so messages read 1 / 2 rather than np.int64(1).
    shown_all = [v.item() if isinstance(v, np.generic) else v for v in values]
    if len(values) != 2:
        many = len(values) > 2
        if many and id is None:
            extra = (
                f" With {len(values)} distinct values it looks like a unit "
                "identifier; that goes in id=."
            )
        elif many and column == id:
            extra = " It is also the id= column."
        else:
            extra = ""
        shown = shown_all[:6] + (["..."] if len(values) > 6 else [])
        raise MethodIncompatibility(
            f"{function}: {column!r} must be binary (treated / comparison), "
            f"got {len(values)} distinct value(s): {shown}.{extra}",
            recovery_hint=(
                f"In {function}, treat= is the time-invariant 0/1 "
                "treatment-group indicator, 1 for units that are treated in "
                "the post period (group= is a deprecated spelling of it). "
                "The unit identifier goes in id=, the period in time=."
            ),
            diagnostics={
                "function": function,
                "column": column,
                "n_distinct": len(values),
            },
        )
    if not _is_zero_one(values) and not all(
        isinstance(v, (bool, np.bool_)) for v in values
    ):
        warnings.warn(
            ArgumentRoleWarning(
                f"{function}: the treatment-group column {column!r} is coded "
                f"{shown_all[0]!r} / {shown_all[1]!r}, not 0 / 1; "
                f"{shown_all[1]!r} (the "
                "larger value) is taken as the treated group.",
                recovery_hint=(
                    f"Recode {column!r} to 0/1 (1 = treated) to make the arm "
                    "explicit."
                ),
                diagnostics={
                    "function": function,
                    "column": column,
                    "treated_value": repr(shown_all[1]),
                },
            ),
            stacklevel=stacklevel,
        )
    return values
