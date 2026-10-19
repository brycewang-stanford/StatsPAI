"""``reshape wide`` / ``reshape long`` for the data steps of ``sp.stata``.

The two forms a do-file writes most: ``reshape wide stubs, i(id) j(t)`` and
``reshape long stubs, i(id) j(t)`` with a numeric ``j``. The column order and
the rows are Stata's: wide is ``i``, then one block of the stubs per value of
``j``, then the variables that are constant within ``i``; long is ``i``,
``j``, the stubs, then the rest, with a row for every ``i`` and every value
of ``j`` found in the names, missing where the wide data had no such column
value. The reshaping itself is :mod:`statspai.utils.reshape` (``sp.reshape``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ...exceptions import StatsPAIError
from ...utils.reshape import reshape_long_frame, reshape_wide_frame
from ._stata_expr import StataExprError

if TYPE_CHECKING:
    from ._stata_datastep import DataSteps

__all__ = ["run_reshape"]


def run_reshape(steps: "DataSteps", varlist: List[str], options: dict) -> None:
    if not varlist or varlist[0] not in ("wide", "long"):
        raise StataExprError(
            "only `reshape wide stubs, i() j()` and `reshape long stubs, i() "
            "j()` are implemented"
        )
    direction, stubs = varlist[0], list(varlist[1:])
    i_vars = str(options.pop("i", "") or "").split()
    j_parts = str(options.pop("j", "") or "").split()
    string = any(options.pop(k, 0) is None for k in ("string", "s", "str"))
    if options:
        raise StataExprError(
            f"reshape: option(s) {sorted(options)} are not implemented"
        )
    if not stubs or not i_vars or len(j_parts) != 1:
        raise StataExprError("reshape needs the stub names, i(varlist) and j(varname)")
    j_var = j_parts[0]
    data = steps.data
    missing = [v for v in i_vars if v not in data.columns]
    if missing:
        raise StataExprError(f"reshape: variable(s) {missing} are not in the data")
    build = reshape_wide_frame if direction == "wide" else reshape_long_frame
    try:
        out = build(data, stubs, i_vars, j_var, string)
    except StatsPAIError as exc:
        raise StataExprError(exc.message) from None
    steps._own()
    steps.data = out.reset_index(drop=True)
    steps._float &= set(steps.data.columns)
