"""``irf create`` and ``irf table`` in an ``sp.stata`` session.

Stata keeps impulse responses in a file that ``irf create`` writes and
``irf table`` reads. Here nothing is written: ``irf create`` records the
horizon it asked for, and ``irf table`` computes the requested statistics
from the VAR or structural VAR in memory.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

from ._stata_expr import StataExprError
from ._stata_lexer import StataParseError
from ._stata_lexer import parse as _parse

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["irf_line"]

_CREATE = re.compile(r"\s*irf\s+cr(?:e(?:a(?:te?)?)?)?\b", re.I)
_TABLE = re.compile(r"\s*irf\s+t(?:a(?:b(?:le?)?)?)?\b", re.I)
#: statistic -> (needs a structural VAR, method, cumulative)
_STATISTICS = {
    "irf": (False, "irf", False),
    "cirf": (False, "irf", True),
    "oirf": (False, "oirf", False),
    "coirf": (False, "oirf", True),
    "fevd": (False, "fevd", False),
    "sirf": (True, "irf", False),
    "sfevd": (True, "fevd", False),
}


def _step(cmd: Any, default: int) -> int:
    raw = cmd.options.get("step")
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        raise StataExprError(f"irf: step({raw}) is not an integer") from None


def _series(fit: Any, statistic: str, impulse: str, response: str, steps: int) -> Any:
    from ...timeseries.svar import SVARResult
    from ...timeseries.var import irf as var_irf

    structural, kind, cumulative = _STATISTICS[statistic]
    is_svar = isinstance(fit, SVARResult)
    if structural != is_svar:
        raise StataExprError(
            f"irf table {statistic}: the model in memory is "
            + ("a structural VAR" if is_svar else "a reduced-form VAR")
            + (
                "; use sirf / sfevd"
                if is_svar
                else "; fit `svar` first, or use oirf / fevd"
            )
        )
    if is_svar:
        # Stata names a structural shock after the equation it belongs to
        shock = list(fit.shock_names)[list(fit.var_names).index(impulse)]
        long = fit.irf(steps) if kind == "irf" else fit.fevd(steps)
        rows = long[(long["shock"] == shock) & (long["response"] == response)]
        return rows.sort_values("period")[kind].to_numpy()
    if kind == "fevd":
        long = fit.fevd(steps)
        rows = long[(long["shock"] == impulse) & (long["response"] == response)]
        return rows.sort_values("period")["fevd"].to_numpy()
    paths = var_irf(
        fit, periods=steps, orthogonal=kind == "oirf", cumulative=cumulative
    )["irf"]
    return paths[f"{impulse} -> {response}"]


def irf_line(session: "StataSession", line: str) -> Optional[bool]:
    """Run ``irf create`` / ``irf table``. ``None`` for any other line."""
    if _CREATE.match(line):
        try:
            session.stored["irf_step"] = _step(_parse(line), 8)
        except StataParseError as exc:
            raise StataExprError(str(exc)) from None
        return False
    if not _TABLE.match(line):
        return None
    try:
        cmd = _parse(line)
    except StataParseError as exc:
        raise StataExprError(str(exc)) from None
    statistics = [w.lower() for w in cmd.varlist[1:]] or ["irf"]
    unknown = [s for s in statistics if s not in _STATISTICS]
    if unknown:
        raise StataExprError(f"irf table: statistic(s) {unknown} are not implemented")
    fit = session.last
    names: List[str] = list(getattr(fit, "var_names", []) or [])
    if not names:
        raise StataExprError("irf table: there is no VAR in memory")
    impulse = str(cmd.options.get("impulse") or "").split()
    response = str(cmd.options.get("response") or "").split()
    if len(impulse) != 1 or len(response) != 1:
        raise StataExprError(
            "irf table: one impulse() and one response() per call are read"
        )
    for name in impulse + response:
        if name not in names:
            raise StataExprError(f"irf table: {name!r} is not in the VAR ({names})")
    steps = _step(cmd, int(session.stored.get("irf_step", 8)))
    columns: Dict[str, Any] = {
        s: _series(fit, s, impulse[0], response[0], steps) for s in statistics
    }
    session.output = pd.DataFrame(columns, index=pd.RangeIndex(steps + 1, name="step"))
    return True
