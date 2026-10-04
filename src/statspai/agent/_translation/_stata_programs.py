"""``program ... end`` and ``simulate``: the Monte Carlo idiom of a do-file.

A teaching do-file shows a sampling distribution by wrapping "draw a
sample, estimate, return the estimate" in a program and repeating it:

    program onesample, rclass
        drop _all
        set obs 30
        gen x = runiform()
        summarize x
        return scalar mean_sample = r(mean)
    end
    simulate xbar = r(mean_sample), reps(10000) seed(101): onesample

``sp.stata`` runs exactly that shape: the body is a list of commands it
already runs, ``return scalar name = exp`` fills ``r()``, and ``simulate``
repeats the body and leaves one row per repetition in memory. A program
may take arguments (``args a b``, or `` `1' `` `` `2' ``), loop and set local
macros, which do not outlive the call (``_stata_flow.py``). ``syntax`` and
``mata`` are refused.

The random numbers come from numpy, not from Stata's generator: the design
is reproduced, the individual draws are not.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MethodIncompatibility
from ._stata_expr import StataExprError
from ._stata_script import ScriptError

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["program_line", "run_simulate"]

_PROGRAM = re.compile(
    r"\s*(?:cap(?:ture)?\s+)?pr(?:ogram)?\s+(?:de(?:f(?:i(?:ne?)?)?)?\s+)?"
    r"(?P<name>[A-Za-z_]\w*)\s*(?:,\s*(?P<opts>.*))?$"
)
_PROGRAM_DROP = re.compile(
    r"\s*(?:cap(?:ture)?\s+)?pr(?:ogram)?\s+drop\s+(?P<name>\S+)\s*$"
)
_RETURN = re.compile(
    r"\s*return\s+sca(?:lar)?\s+([A-Za-z_]\w*)\s*=(?!=)\s*(.+)\Z", re.S
)
_SIMULATE = re.compile(
    r"\s*simulate\s+(?P<exps>[^,:]*?)\s*(?:,\s*(?P<opts>[^:]*))?:\s*"
    r"(?P<prog>[A-Za-z_]\w*)\s*$",
    re.S,
)
_UNSUPPORTED_BODY = re.compile(r"\s*(?:syntax|mata)\b")


def _refuse(message: str, line: str) -> MethodIncompatibility:
    return MethodIncompatibility(
        f"sp.stata: cannot run {line!r}: {message}",
        recovery_hint="Write the simulation as a Python loop around sp.stata(...) "
        "or the sp.* call.",
        diagnostics={"command": line},
    )


def program_line(session: "StataSession", line: str) -> Optional[bool]:
    """Collect the lines of a program definition, and run ``return scalar``.

    Returns ``None`` when ``line`` has nothing to do with programs.
    """
    if session._defining is not None:
        name, body = session._defining
        if line.strip() == "end":
            session.programs[name] = body
            session._defining = None
        else:
            if _UNSUPPORTED_BODY.match(line):
                session._defining = None
                raise _refuse(
                    "the program uses `syntax` or `mata`, which are not run",
                    line,
                )
            body.append(line)
        return False
    m = _PROGRAM_DROP.match(line)
    if m:
        if m.group("name") == "_all":
            session.programs.clear()
        else:
            session.programs.pop(m.group("name"), None)
        return False
    m = _PROGRAM.match(line)
    if m and m.group("name") not in ("drop", "dir", "list"):
        session._defining = (m.group("name"), [])
        return False
    m = _RETURN.match(line)
    if m:
        if session._returned is None:
            raise _refuse("`return scalar` outside a program", line)
        try:
            # the value is often a local macro set a few lines above
            expr = session._macros.expand(m.group(2).strip())
            session._returned[m.group(1)] = session.value(expr)
        except (StataExprError, ScriptError) as exc:
            raise _refuse(str(exc), line) from exc
        return False
    return None


def _run_program(session: "StataSession", name: str) -> Dict[str, float]:
    # a program may call another one: its own r() table is put back after
    outer = session._returned
    session._returned = {}
    try:
        for command in session.programs[name]:
            session.run(command)
        returned = dict(session._returned)
    finally:
        session._returned = outer
    session.stored["r"] = returned
    return returned


def run_simulate(session: "StataSession", line: str) -> Optional[bool]:
    """``simulate name = exp ..., reps(#) [seed(#)]: program``."""
    if not re.match(r"\s*simulate\b", line):
        words = line.split()
        if words and words[0] in session.programs:
            from ._stata_flow import call_program

            call_program(session, words[0], words[1:])
            return False
        return None
    m = _SIMULATE.match(line)
    if m is None or m.group("prog") not in session.programs:
        raise _refuse(
            "expected `simulate name = exp ..., reps(#): program` with a "
            "program defined above by `program name ... end`",
            line,
        )
    from ._stata_lexer import _parse_options

    options = _parse_options(m.group("opts") or "")
    try:
        reps = int(str(options.pop("reps", "")).strip())
    except ValueError:
        raise _refuse("simulate needs reps(#)", line) from None
    seed = options.pop("seed", None)
    for display in ("nodots", "dots", "noisily", "nolegend", "verbose", "trace"):
        options.pop(display, None)
    if options or reps < 1:
        raise _refuse(f"simulate option(s) {sorted(options)} are not implemented", line)
    pairs: List[tuple] = []
    for item in re.findall(
        r"(?:([A-Za-z_]\w*)\s*=\s*)?(\([^)]*\)|\S+)", m.group("exps")
    ):
        name, expr = item
        expr = expr.strip()
        if not name:
            raise _refuse("each simulate result needs a name: `name = exp`", line)
        pairs.append((name, expr))
    if not pairs:
        raise _refuse("simulate needs at least one `name = exp`", line)
    if seed is not None:
        session.stored["rng"] = np.random.default_rng(int(str(seed).strip()))
    columns: Dict[str, List[float]] = {name: [] for name, _ in pairs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(reps):
            _run_program(session, m.group("prog"))
            for name, expr in pairs:
                try:
                    columns[name].append(session.value(expr))
                except StataExprError as exc:
                    raise _refuse(str(exc), line) from exc
    session.use(pd.DataFrame(columns))
    session.simulated = True
    session.stored["random_draws"] = True
    return False
