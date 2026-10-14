"""Loops, ``if`` blocks and computed macros for ``sp.stata``.

A do-file repeats itself with ``forvalues`` and ``foreach``, branches with
``if { } else { }`` and keeps counters in local macros. All of that is text
substitution around commands the session already runs, so it is run the way
Stata runs it: the body of a block is collected up to its closing brace,
and for each pass the loop macro is set and the body lines are handed back
to the session one by one.

What is read
------------
``forvalues i = a/b``, ``a(d)b``, ``a b : c`` and ``a b to c``
``foreach x in list``, ``of varlist`` (names, ``a-c`` ranges, ``x*``),
``of numlist``, ``of local name``, ``of global name``, ``of newlist``
``while exp { }``
``if exp { }`` with ``else if exp { }`` and ``else { }``
``continue`` and ``continue, break``
``local name = exp`` / ``global name = exp`` with a numeric expression
(``_b[x]``, ``r(mean)``, scalars ... as in ``scalar name = exp``),
``local ++i`` / ``local --i``, and the inline forms `` `=exp' ``,
`` `r(name)' `` and `` `e(name)' ``
``tempvar`` / ``tempname`` / ``tempfile``: local macros holding fresh names
``args a b`` inside a program called with arguments

A numeric macro holds the value as Stata writes it, with up to 16
significant digits. Extended macro functions (``local n : word count ...``),
``syntax`` and ``mata`` are still refused: their value cannot be read off
the line.

A loop that never ends is stopped after 1,000,000 passes.
"""

from __future__ import annotations

import fnmatch
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MethodIncompatibility
from ._stata_expr import StataExprError
from ._stata_lexer import quote_end, split_words
from ._stata_script import ScriptError

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = [
    "flow_line",
    "macro_line",
    "macro_name_expanded",
    "format_number",
    "call_program",
    "captured_code",
    "drop_temporaries",
]

_PREFIX = (
    r"(?:(?:qui(?:e(?:t(?:ly?)?)?)?|noi(?:s(?:i(?:ly?)?)?)?|"
    r"cap(?:t(?:u(?:re?)?)?)?)\s+)*"
)
_OPENER = re.compile(
    r"\s*" + _PREFIX + r"(forv(?:a(?:l(?:u(?:es?)?)?)?)?|foreach|while|if|else)\b"
    r"(.*)\{\s*$",
    re.I | re.S,
)
_QUIET_WORD = r"(?:qui(?:e(?:t(?:ly?)?)?)?|noi(?:s(?:i(?:ly?)?)?)?)"
#: ``capture {`` (with ``quietly`` / ``noisily`` on either side)
_CAPTURE_OPENER = re.compile(
    rf"\s*(?:{_QUIET_WORD}\s+)*cap(?:t(?:u(?:re?)?)?)?(?:\s+{_QUIET_WORD})*\s*\{{\s*$",
    re.I,
)
#: the name of ``local`` / ``global`` / ``scalar`` when a macro builds it
_MACRO_NAME = re.compile(
    r"(\s*(?:gl(?:o(?:b(?:al?)?)?)?|loc(?:al?)?"
    r"|sca(?:l(?:ar?)?)?(?:\s+def(?:i(?:ne?)?)?)?)\s+)"
    r"((?:[A-Za-z0-9_]|`[^`'\"]*'|\$\{?[A-Za-z_]\w*\}?)+)",
    re.I,
)
_CONTINUE = re.compile(r"\s*continue\s*(,\s*break)?\s*$", re.I)
_FORVALUES = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.S)
_FOREACH = re.compile(
    r"\s*([A-Za-z_]\w*)\s+(in|of\s+(?:var(?:l(?:i(?:st?)?)?)?|num(?:l(?:i(?:st?)?)?)?|"
    r"loc(?:al?)?|glo(?:b(?:al?)?)?|new(?:l(?:i(?:st?)?)?)?))\b\s*(.*?)\s*$",
    re.I | re.S,
)
_ASSIGN = re.compile(
    r"\s*(gl(?:o(?:b(?:al?)?)?)?|loc(?:al?)?)\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s*$",
    re.I | re.S,
)
_STEP = re.compile(r"\s*loc(?:al?)?\s+(\+\+|--)\s*([A-Za-z_]\w*)\s*$", re.I)
_TEMP = re.compile(r"\s*(tempvar|tempname|tempfile)\s+(.+?)\s*$", re.I)
_ARGS = re.compile(r"\s*args\s+(.+?)\s*$", re.I)
_MAX_PASSES = 1_000_000


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


def _refuse(message: str, line: str) -> MethodIncompatibility:
    return MethodIncompatibility(
        f"sp.stata: cannot run {line!r}: {message}",
        recovery_hint="Write this step in Python around sp.stata(...) or the "
        "sp.* call.",
        diagnostics={"command": line},
    )


def format_number(value: float) -> str:
    """A number as Stata writes it into a macro: format ``%18.0g``. Stata
    18: ``local a = 1/3`` is .3333333333333333, ``local b = 1e20`` is
    1.00000000000e+20, and ``forvalues v = 0(0.1)0.35`` gives 0 .1 .2 .3."""
    if not np.isfinite(value):
        return "."
    from ._stata_functions import _general

    return _general(float(value), 18, 0)


def macro_value(session: "StataSession", name: str) -> Optional[str]:
    """What `` `name' `` stands for when it is not a macro that was
    defined: `` `=exp' ``, `` `r(name)' ``, `` `e(name)' ``."""
    if name.isdigit() and session._program_depth:
        return ""  # an argument the caller did not give is empty
    if name.lstrip().startswith(":"):
        from ._stata_macro import extended_function

        try:
            return extended_function(session, name.lstrip()[1:])
        except StataExprError:
            return None
    held = re.fullmatch(r"([rec])\((\w+)\)", name)
    if held and held.group(1) == "c":
        from ._stata_functions import system_value

        data = session.data
        shape = (0, 0) if data is None else data.shape
        value = system_value(held.group(2), *shape)
        if value is None:
            return None
        return value if isinstance(value, str) else format_number(float(value))
    if held:
        # r(varlist), r(levels), e(cmd): results that are text
        texts = session.stored.get(f"{held.group(1)}_macros") or {}
        if held.group(2) in texts:
            return str(texts[held.group(2)])
    try:
        if name.startswith("="):
            return format_number(session.value(name[1:]))
        if held:
            return format_number(session.value(name))
    except StataExprError:
        return None  # nothing stored under that name: an undefined macro
    return None


def _expand(session: "StataSession", text: str, line: str) -> str:
    try:
        return session._macros.expand(text)
    except (ScriptError, StataExprError) as exc:
        raise _refuse(str(exc), line) from exc


# ------------------------------------------------------------------ macros
def macro_name_expanded(session: "StataSession", line: str) -> str:
    """``line`` with the macros in the *name* of a ``local`` / ``global`` /
    ``scalar`` definition written out: after ``local v x``, ``local m`v' =
    5`` defines ``mx`` (Stata 18 then shows mx=5 and leaves ``m`` alone).
    The rest of the line is left for the definition to read."""
    m = _MACRO_NAME.match(line)
    if m is None or not re.search(r"[`$]", m.group(2)):
        return line
    name = _expand(session, m.group(2), line)
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        raise _refuse(f"the macro name {m.group(2)!r} expands to {name!r}", line)
    return m.group(1) + name + line[m.end() :]


def _new_temporary(session: "StataSession", kind: str) -> str:
    """A name for ``tempvar`` / ``tempname`` (Stata's ``__000000``,
    ``__000001`` ... from one counter) or ``tempfile``."""
    count = session._temp_count
    session._temp_count += 1
    if kind == "file":
        return f"__file{count:06d}"
    name = f"__{count:06d}"
    if kind == "var":
        temporaries(session).append(name)
    return name


def temporaries(session: "StataSession") -> List[str]:
    """The ``tempvar`` names handed out so far, oldest first."""
    held: List[str] = session.__dict__.setdefault("_tempvars", [])
    return held


def drop_temporaries(session: "StataSession", since: int = 0) -> None:
    """Drop the variables named by ``tempvar`` after the first ``since``
    names: Stata drops them when the program or do-file that made them
    ends."""
    names = temporaries(session)
    doomed, names[since:] = names[since:], []
    steps = session._steps
    if steps is None or not doomed:
        return
    held = [c for c in doomed if c in steps.data.columns]
    if held:
        steps.data = steps.data.drop(columns=held)


def macro_line(session: "StataSession", line: str) -> bool:
    """Run ``line`` if it sets a macro from a value this session knows."""
    m = _STEP.match(line)
    if m:
        name = m.group(2)
        held = session._macros.locals.get(name)
        try:
            current = float(held) if held else float("nan")
        except ValueError:
            raise _refuse(f"macro `{name}' does not hold a number", line) from None
        step = 1.0 if m.group(1) == "++" else -1.0
        session._macros.locals[name] = format_number(current + step)
        return True
    m = _TEMP.match(line)
    if m:
        for name in m.group(2).split():
            kind = m.group(1).lower()[4:]
            session._macros.locals[name] = _new_temporary(session, kind)
        return True
    m = _ARGS.match(line)
    if m and session._program_depth:
        for position, name in enumerate(m.group(1).split(), 1):
            session._macros.locals[name] = session._macros.locals.get(str(position), "")
        return True
    m = _ASSIGN.match(line)
    if m:
        rhs = _expand(session, m.group(3), line)
        table = (
            session._macros.globals
            if m.group(1).lower().startswith("g")
            else session._macros.locals
        )
        plain = re.fullmatch(r'\s*(`"[^`]*?"\'|"[^"]*")\s*', rhs)
        if plain is not None:
            return False  # one string literal: the macro table reads it
        try:
            number = session.value(rhs)
        except StataExprError:
            # a string expression: upper("abc"), "a" + "b", "x" != ""
            from ._stata_expr import evaluate

            try:
                held = evaluate(rhs, pd.DataFrame({"_": [0.0]}), session.stored)
            except StataExprError:
                return False  # left to the macro table (unknown value)
            if held.dtype != object:
                return False
            table[m.group(2)] = str(held[0])
            return True
        table[m.group(2)] = format_number(number)
        return True
    return False


# ------------------------------------------------------------------- lists
def _numlist(spec: str, line: str) -> List[str]:
    out: List[str] = []
    for token in spec.replace(",", " ").split():
        m = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", token)
        n = re.fullmatch(r"(-?[\d.]+)\((-?[\d.]+)\)(-?[\d.]+)", token)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            step = 1.0 if hi >= lo else -1.0
        elif n:
            lo, step, hi = (float(n.group(i)) for i in (1, 2, 3))
        else:
            try:
                out.append(format_number(float(token)))
            except ValueError:
                raise _refuse(f"{token!r} is not a number list", line) from None
            continue
        if step == 0:
            raise _refuse("a number list with step 0", line)
        count = int(np.floor((hi - lo) / step + 1e-9)) + 1
        out.extend(format_number(lo + k * step) for k in range(max(count, 0)))
    return out


def _forvalues_range(spec: str, line: str) -> List[str]:
    spec = spec.strip()
    m = re.fullmatch(r"(\S+)\s+(\S+)\s*(?::|to)\s*(\S+)", spec)
    if m:  # a b : c  -- the step is b - a
        a, b, c = (float(m.group(i)) for i in (1, 2, 3))
        return _numlist(
            f"{format_number(a)}({format_number(b - a)}){format_number(c)}", line
        )
    values = _numlist(spec.replace(" ", ""), line)
    return values


def _varlist(session: "StataSession", spec: str, line: str) -> List[str]:
    data = session.data
    columns = [] if data is None else [str(c) for c in data.columns]
    names: List[str] = []
    for token in spec.split():
        if "-" in token and token not in columns:
            first, _, last = token.partition("-")
            if first not in columns or last not in columns:
                raise _refuse(f"cannot read the variable range {token!r}", line)
            names.extend(columns[columns.index(first) : columns.index(last) + 1])
        elif any(ch in token for ch in "*?"):
            hits = [c for c in columns if fnmatch.fnmatchcase(c, token)]
            if not hits:
                raise _refuse(f"no variable matches {token!r}", line)
            names.extend(hits)
        elif token in columns:
            names.append(token)
        else:
            raise _refuse(f"variable {token!r} is not in the data", line)
    return names


def _words(text: str) -> List[str]:
    """A macro list split as Stata splits it: blanks, with "quoted items"
    and compound-quoted items (what ``levelsof`` writes for a string
    variable, `` `"x y"' `"z"' ``) as one word each, without the quotes."""
    out: List[str] = []
    for word in split_words(text):
        end = quote_end(word, 0)
        if end == len(word) and word.startswith('`"') and word.endswith("\"'"):
            word = word[2:-2]
        elif end == len(word) and len(word) >= 2 and word.endswith('"'):
            word = word[1:-1]
        out.append(word)
    return out


# ------------------------------------------------------------------ blocks
def _run_body(session: "StataSession", body: List[str]) -> None:
    quiet = session._quiet_blocks
    try:
        for command in body:
            session.run(command)
    finally:
        # a `continue` inside `quietly { }` leaves the block open
        session._quiet_blocks = quiet
        session._flow = None


def _loop(
    session: "StataSession", name: str, values: List[str], body: List[str]
) -> None:
    for value in values:
        session._macros.locals[name] = value
        try:
            _run_body(session, body)
        except _Continue:
            continue
        except _Break:
            break
    session._macros.locals.pop(name, None)


def _truth(session: "StataSession", expr: str, line: str) -> bool:
    try:
        value = session.value(expr)
    except StataExprError as exc:
        raise _refuse(str(exc), line) from exc
    return bool(value != 0)  # missing is not zero, so it is true


def captured_code(exc: BaseException) -> Optional[float]:
    """The ``_rc`` a ``capture`` sets for ``exc``, or ``None`` when ``exc``
    is not an error Stata would have raised (a command this runner cannot
    translate still stops the run)."""
    if isinstance(exc, MethodIncompatibility) and isinstance(
        exc.__cause__, StataExprError
    ):
        from ._stata_run import _return_code

        return _return_code(str(exc))
    if isinstance(exc, KeyError):
        return 111.0  # a variable that is not there
    return None


def _run_captured(session: "StataSession", body: List[str]) -> None:
    """``capture { ... }``: the lines run until one fails; what ran before
    stays done, the rest is skipped and ``_rc`` holds the failure's code
    (Stata 18: 110 for a variable that already exists, 111 for one that is
    not there, 0 when nothing failed). The code is the one the error names
    or, failing that, 111 / 110 / 198 as for a one-line ``capture``."""
    try:
        _run_body(session, body)
    except Exception as exc:
        code = captured_code(exc)
        if code is None:
            raise
        session.stored["_rc"] = code
        return
    session.stored["_rc"] = 0.0


def _run_block(session: "StataSession", head: str, body: List[str]) -> None:
    if _CAPTURE_OPENER.match(head):
        _run_captured(session, body)
        return
    m = _OPENER.match(head)
    assert m is not None
    word = m.group(1).lower()
    rest = m.group(2)
    if word.startswith("forv"):
        spec = _FORVALUES.match(_expand(session, rest, head))
        if spec is None:
            raise _refuse("expected `forvalues name = range {`", head)
        _loop(session, spec.group(1), _forvalues_range(spec.group(2), head), body)
    elif word == "foreach":
        spec = _FOREACH.match(rest)
        if spec is None:
            raise _refuse("expected `foreach name in|of ... {`", head)
        kind = spec.group(2).lower().split()[-1]
        tail = spec.group(3)
        if kind.startswith("loc") or kind.startswith("glo"):
            table = (
                session._macros.locals
                if kind.startswith("loc")
                else session._macros.globals
            )
            held = table.get(tail.strip())
            if held is None and tail.strip() in table:
                raise _refuse(
                    f"macro {tail.strip()!r} has a value only Stata knows", head
                )
            values = _words(held or "")
        else:
            tail = _expand(session, tail, head)
            if kind.startswith("var"):
                values = _varlist(session, tail, head)
            elif kind.startswith("num"):
                values = _numlist(tail, head)
            else:  # in / newlist
                values = _words(tail)
        _loop(session, spec.group(1), values, body)
    elif word == "while":
        for _ in range(_MAX_PASSES):
            if not _truth(session, _expand(session, rest, head), head):
                break
            try:
                _run_body(session, body)
            except _Continue:
                continue
            except _Break:
                break
        else:
            raise _refuse("the loop did not end after 1,000,000 passes", head)
    elif word == "if":
        taken = _truth(session, _expand(session, rest, head), head)
        if taken:
            _run_body(session, body)
        session._if_taken = taken
    else:  # else / else if
        previous = session._if_taken
        if previous is None:
            raise _refuse("`else` without an `if` block before it", head)
        nested = re.match(r"\s*if\b(.*)$", rest, re.S)
        if previous:
            session._if_taken = True  # a later `else` is skipped as well
            return
        if nested is None:
            _run_body(session, body)
            session._if_taken = None
            return
        taken = _truth(session, _expand(session, nested.group(1), head), head)
        if taken:
            _run_body(session, body)
        session._if_taken = taken


def flow_line(session: "StataSession", line: str) -> Optional[bool]:
    """Collect and run blocks. ``None`` when ``line`` is ordinary."""
    stripped = line.strip()
    block = session._flow
    if block is not None:
        if stripped.startswith("}") and stripped != "}":
            # `} else {`: the closing brace, then a new opener
            flow_line(session, "}")
            return flow_line(session, stripped[1:].strip())
        if stripped == "}":
            block["depth"] -= 1
            if block["depth"] == 0:
                session._flow = None
                _run_block(session, block["head"], block["body"])
                return False
        elif stripped.endswith("{"):
            block["depth"] += 1
        block["body"].append(line)
        return False
    if _OPENER.match(line) or _CAPTURE_OPENER.match(line):
        session._flow = {"head": line, "body": [], "depth": 1}
        return False
    m = _CONTINUE.match(line)
    if m:
        raise (_Break() if m.group(1) else _Continue())
    session._if_taken = None  # an `else` must follow its `if` block directly
    return None


# ---------------------------------------------------------------- programs
def call_program(
    session: "StataSession", name: str, arguments: List[str]
) -> Dict[str, Any]:
    """Run a program defined by ``program name ... end`` with its arguments
    in the local macros `` `1' ``, `` `2' `` ... (``args`` names them).
    Local macros set inside do not outlive the call."""
    outer = dict(session._macros.locals)
    first_temporary = len(temporaries(session))
    session._macros.locals = {str(i): a for i, a in enumerate(arguments, 1)}
    session._macros.locals["0"] = " ".join(arguments)
    session._program_depth += 1
    # a program may call another one: its own r() table is put back after
    outer_returned = session._returned
    session._returned = {}
    from ._stata_macro import ProgramExit

    session.stored.pop("returned_macros", None)
    try:
        try:
            for command in session.programs[name]:
                session.run(command)
        except ProgramExit:
            pass  # `exit`: the program ends here
        returned = dict(session._returned)
    finally:
        session._returned = outer_returned
        session._program_depth -= 1
        # the temporary variables of the program go with it
        drop_temporaries(session, first_temporary)
        session._macros.locals = outer
        session._flow = None
    texts = session.stored.pop("returned_macros", None)
    if returned or texts:
        session.stored["r"] = returned
        session.stored["r_macros"] = dict(texts or {})
    return returned
