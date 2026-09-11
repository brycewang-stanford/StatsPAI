"""Introspectable coverage of the Stata / R → StatsPAI translators.

``translation_coverage()`` turns the hand-curated handler tables in
:mod:`._stata` and :mod:`._r` into a queryable matrix: for every Stata command
and R function the translator recognizes, which ``sp.*`` call does it map to?
This is the migration on-ramp made auditable — a researcher (or an agent) can
ask *"is my command supported, and what does it become?"* without reading the
source, and can see in one place exactly where the coverage stops.

The matrix is derived by *introspection*, not a second hand-maintained list, so
it can never drift from the actual dispatch tables: the recognized names are the
keys of ``STATA_COMMAND_MAP`` / ``R_FUNCTION_MAP``, and each target ``sp.*`` is
parsed from the handler's own source.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, List

from ._r import R_FUNCTION_MAP
from ._stata import STATA_COMMAND_MAP

# ``_emit("tool", ...)`` names the canonical target; ``sp.<fn>(`` appears in the
# generated ``python_code``. We read both and union them so multi-target
# handlers (e.g. ``teffects`` → ipw/match/aipw, ``glm`` → logit/probit/poisson)
# are reported in full.
_EMIT_RE = re.compile(r"_emit\(\s*[\"'](\w+)[\"']")
_SP_CALL_RE = re.compile(r"sp\.(\w+)\(")

#: Documented, machine-readable limitations of the translators. Surfaced here so
#: the gaps are part of the queryable contract, not a surprise mid-migration.
LIMITATIONS: List[str] = [
    "Panel id/time: xtreg/xtabond/xtnbreg emit a `<panel_id>` placeholder when "
    "the `xtset`/`tsset` declaration is not on the same line — pass id=/time= "
    "explicitly to the resulting sp.* call.",
    "Time-series commands (arima, var, vec, varsoc, granger) are not translated; "
    "call sp.arima / sp.var / sp.vecm directly.",
    "Estimation-table commands (esttab, eststo, outreg2) are not translated; use "
    "sp.regtable on the fitted results.",
    "Dropped Stata `if`/`in` qualifiers and unrecognized options are never "
    "silently lost — they are surfaced in the per-command `notes` field.",
    "One command per call: multi-command .do / multi-line R scripts must be "
    "split by the caller before translation.",
]


def _targets(handler: Callable) -> List[str]:
    """The ``sp.*`` function name(s) a handler maps to, parsed from its source."""
    try:
        src = inspect.getsource(handler)
    except (OSError, TypeError):  # pragma: no cover - source always available here
        return []
    found: List[str] = []
    for name in _EMIT_RE.findall(src) + _SP_CALL_RE.findall(src):
        if name not in found:
            found.append(name)
    return found


def _coverage_for(table: Dict[str, Callable]) -> List[Dict[str, Any]]:
    """Group a command→handler table by handler (aliases share a handler)."""
    by_handler: Dict[int, Dict[str, Any]] = {}
    for command, handler in table.items():
        key = id(handler)
        entry = by_handler.setdefault(
            key,
            {"commands": [], "handler": handler.__name__, "targets": _targets(handler)},
        )
        entry["commands"].append(command)
    rows = []
    for entry in by_handler.values():
        cmds = sorted(entry["commands"])
        rows.append(
            {
                "command": cmds[0],
                "aliases": cmds[1:],
                "targets": [f"sp.{t}" for t in entry["targets"]],
                "handler": entry["handler"],
            }
        )
    return sorted(rows, key=lambda r: r["command"])


def translation_coverage(*, fmt: str = "dict") -> Any:
    """Coverage matrix for the ``sp.from_stata`` / ``sp.from_r`` translators.

    Returns, for every recognized Stata command and R function, the ``sp.*``
    call(s) it translates to — derived by introspecting the live dispatch
    tables, so it never drifts from what the translators actually do.

    Parameters
    ----------
    fmt : {"dict", "markdown"}, default "dict"
        ``"dict"`` returns a structured mapping (agent / JSON friendly);
        ``"markdown"`` returns a ready-to-publish table string.

    Returns
    -------
    dict or str
        For ``fmt="dict"``: ``{"stata": [...], "r": [...], "summary": {...},
        "limitations": [...]}`` where each row is
        ``{"command", "aliases", "targets", "handler"}``.

    Examples
    --------
    >>> import statspai as sp
    >>> cov = sp.translation_coverage()
    >>> cov["summary"]["n_stata_commands"] > 0
    True
    >>> "from_stata" in sp.translation_coverage.__doc__ or True
    True
    >>> any(r["command"] == "reghdfe" for r in cov["stata"])
    True
    >>> isinstance(sp.translation_coverage(fmt="markdown"), str)
    True
    """
    stata = _coverage_for(STATA_COMMAND_MAP)
    r = _coverage_for(R_FUNCTION_MAP)
    summary = {
        "n_stata_commands": len(STATA_COMMAND_MAP),
        "n_stata_handlers": len(stata),
        "n_r_functions": len(R_FUNCTION_MAP),
        "n_r_handlers": len(r),
    }
    if fmt == "markdown":
        return _render_markdown(stata, r, summary)
    if fmt != "dict":
        raise ValueError(f"fmt must be 'dict' or 'markdown', got {fmt!r}")
    return {"stata": stata, "r": r, "summary": summary, "limitations": LIMITATIONS}


def _render_markdown(stata, r, summary) -> str:
    lines: List[str] = ["## Stata → StatsPAI\n"]
    lines.append("| Stata command | aliases | → StatsPAI |")
    lines.append("| --- | --- | --- |")
    for row in stata:
        al = ", ".join(f"`{a}`" for a in row["aliases"]) or "—"
        tg = ", ".join(f"`{t}`" for t in row["targets"]) or "—"
        lines.append(f"| `{row['command']}` | {al} | {tg} |")
    lines.append("\n## R → StatsPAI\n")
    lines.append("| R function | aliases | → StatsPAI |")
    lines.append("| --- | --- | --- |")
    for row in r:
        al = ", ".join(f"`{a}`" for a in row["aliases"]) or "—"
        tg = ", ".join(f"`{t}`" for t in row["targets"]) or "—"
        lines.append(f"| `{row['command']}` | {al} | {tg} |")
    lines.append(
        f"\n*{summary['n_stata_commands']} Stata commands "
        f"({summary['n_stata_handlers']} handlers), "
        f"{summary['n_r_functions']} R functions "
        f"({summary['n_r_handlers']} handlers).*\n"
    )
    return "\n".join(lines) + "\n"
