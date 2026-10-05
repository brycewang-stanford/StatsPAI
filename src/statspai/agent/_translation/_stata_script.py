"""Do-file grammar that spans lines: comments, continuations, macros.

:func:`statspai.from_stata` translates one command. A pasted do-file
snippet is more than a list of commands: a command runs over several
lines with ``///``, comments come in three styles, and variable lists
live in macros defined further up (``global controls "age educ"``). This
module turns such a snippet into the complete commands it contains, with
the macros it defines written out.

Only what can be resolved by reading is resolved. A macro set by an
expression or an extended function, a macro that is never defined, and
loops are refused: their value depends on running Stata.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "split_commands",
    "MacroTable",
    "ScriptError",
    "control_flow",
    "panel_declaration",
    "survival_declaration",
]


class ScriptError(ValueError):
    """A construct of the snippet that cannot be resolved by reading it."""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _strip_block_comments(text: str) -> str:
    """Remove ``/* ... */`` (they nest in Stata), keeping line breaks."""
    out: List[str] = []
    depth, i, quoted = 0, 0, False
    while i < len(text):
        ch, two = text[i], text[i : i + 2]
        if ch == "\n":
            quoted = False
        elif ch == '"' and depth == 0:
            quoted = not quoted
        if not quoted and two == "/*":
            depth += 1
            i += 2
            continue
        if not quoted and depth and two == "*/":
            depth -= 1
            i += 2
            continue
        if depth == 0 or ch == "\n":
            out.append(ch)
        i += 1
    return "".join(out)


def _comment_start(line: str, marker: str) -> int:
    """Index of ``//`` or ``///`` outside quotes, at line start or after a blank."""
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif (
            not quoted
            and line.startswith(marker, i)
            and (i == 0 or line[i - 1].isspace())
        ):
            if marker == "//" and line.startswith("///", i):
                continue
            return i
    return -1


def _split_semicolons(line: str) -> List[str]:
    parts, cur, quoted = [], "", False
    for ch in line:
        if ch == '"':
            quoted = not quoted
        if ch == ";" and not quoted:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


_DELIMIT = re.compile(r"^#d(?:e(?:l(?:i(?:m(?:it?)?)?)?)?)?\s+(;|cr\b)", re.I)


def split_commands(text: str) -> List[str]:
    """The complete commands of a do-file snippet, comments removed.

    ``///`` joins a line to the next one, ``//`` and a leading ``*`` start a
    comment, ``/* */`` is removed, and ``;`` separates commands. After
    ``#delimit ;`` a line break no longer ends a command.
    """
    commands: List[str] = []
    pending = ""
    semicolon_mode = False
    for raw in _strip_block_comments(text).splitlines():
        line = raw.strip()
        m = _DELIMIT.match(line)
        if m:
            semicolon_mode = m.group(1) == ";"
            continue
        if not pending and line.startswith("*"):
            continue
        cont = _comment_start(line, "///")
        plain = _comment_start(line, "//")
        continued = cont >= 0 and (plain < 0 or cont < plain)
        if continued:
            line = line[:cont]
        elif plain >= 0:
            line = line[:plain]
        pending = f"{pending} {line}".strip() if pending else line.strip()
        if continued:
            continue
        pieces = _split_semicolons(pending)
        if semicolon_mode:
            # the last piece is still open until its own semicolon arrives
            pending = pieces.pop().strip()
        else:
            pending = ""
        commands.extend(p.strip() for p in pieces if p.strip())
    if pending.strip():
        commands.append(pending.strip())
    return [c for c in commands if not c.startswith("*")]


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

_CONTROL = re.compile(
    r"^(foreach|forv(?:a(?:l(?:u(?:es?)?)?)?)?|while|program|mata"
    r"|if\b.*\{|else\b|\}|\{)",
    re.I,
)


def control_flow(command: str) -> Optional[str]:
    """The control-flow keyword a command opens or closes, if any."""
    m = _CONTROL.match(command.strip())
    if not m:
        return None
    word = m.group(1).split()[0].lower()
    return "forvalues" if word.startswith("forv") else word


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------

_DEFINE = re.compile(r"^(gl(?:o(?:b(?:al?)?)?)?|loc(?:al?)?)\s+(\S.*)$", re.I)
_NAME = re.compile(r"^[A-Za-z_]\w*")
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
#: ``$name`` / ``${name}`` / `` `name' ``; a compound quote `` `"..."' `` is a string
_USE = re.compile(r"\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)|`(?!\")([^`']*)'")


class MacroTable:
    """Global and local macros defined by the lines read so far."""

    def __init__(self) -> None:
        self.globals: Dict[str, Optional[str]] = {}
        self.locals: Dict[str, Optional[str]] = {}
        #: what `` `name' `` stands for when no macro has that name
        #: (`` `=exp' ``, `` `r(mean)' ``); ``None`` when it cannot say
        self.evaluator: Any = None

    def define(self, command: str) -> bool:
        """Record ``global name ...`` / ``local name ...``; True if it was one.

        A value that only Stata can compute (``= exp``, ``: extended
        function``, ``++i``) is recorded as unknown, so a later use of the
        macro is refused instead of expanded wrongly.
        """
        m = _DEFINE.match(command.strip())
        if not m:
            return False
        table = self.globals if m.group(1).lower().startswith("g") else self.locals
        rest = m.group(2).strip()
        name = _NAME.match(rest.lstrip("+-"))
        if not name:
            raise ScriptError(f"cannot read the macro name in {command!r}")
        if rest[:2] in ("++", "--"):
            table[name.group(0)] = None
            return True
        body = rest[name.end() :]
        table[name.group(0)] = self._value(body)
        return True

    def _value(self, body: str) -> Optional[str]:
        body = body.strip()
        if body.startswith(":"):
            return None
        if body.startswith("="):
            exp = body[1:].strip()
            quoted = _unquote(exp)
            if quoted is not None:
                return self.expand(quoted)
            return exp if _NUMBER.match(exp) else None
        quoted = _unquote(body)
        return self.expand(quoted if quoted is not None else body)

    def expand(self, text: str) -> str:
        """``text`` with every macro written out.

        Raises :class:`ScriptError` for a macro that was never defined or
        whose value is not known from the text.
        """

        def repl(m: "re.Match[str]") -> str:
            if m.group(3) is not None:
                name, table, shown = m.group(3), self.locals, m.group(0)
            else:
                name, table, shown = m.group(1) or m.group(2), self.globals, m.group(0)
            stepped = re.fullmatch(r"(\+\+|--)?([A-Za-z_]\w*)(\+\+|--)?", name or "")
            if (
                m.group(3) is not None
                and stepped
                and (stepped.group(1) or stepped.group(3))
                and not (stepped.group(1) and stepped.group(3))
                and stepped.group(2) in table
            ):
                # `i++' gives the value and then adds one; `++i' adds first
                key = stepped.group(2)
                try:
                    current = float(table[key] or "")
                except ValueError:
                    raise ScriptError(f"macro `{key}' does not hold a number") from None
                step = 1.0 if "+" in (stepped.group(1) or stepped.group(3)) else -1.0
                after = current + step
                table[key] = str(int(after)) if after == int(after) else repr(after)
                shown_value = after if stepped.group(1) else current
                return (str(int(shown_value)) if shown_value == int(shown_value)
                        else repr(shown_value))  # fmt: skip
            if name not in table and self.evaluator is not None and m.group(3):
                computed = self.evaluator(name)
                if computed is not None:
                    return str(computed)
            if name not in table:
                raise ScriptError(
                    f"macro {shown} is not defined in these lines; define it "
                    "above the command or write the variables out"
                )
            value = table[name]
            if value is None:
                raise ScriptError(
                    f"macro {shown} is set by an expression or extended "
                    "function, so its value is only known to Stata; write "
                    "it out"
                )
            return value

        for _ in range(20):  # a macro's value was expanded when it was defined
            new = _USE.sub(repl, text)
            if new == text:
                return new
            text = new
        raise ScriptError("macros nest too deeply to expand")


def _unquote(text: str) -> Optional[str]:
    text = text.strip()
    if len(text) >= 4 and text.startswith('`"') and text.endswith("\"'"):
        return text[2:-2]
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return None


def survival_declaration(command: str) -> Optional[Tuple[str, str]]:
    """``stset time, failure(event)`` -> ``(time, event)``.

    Only the form that needs no data step is read: one time variable and a
    failure indicator, optionally written ``failure(event == 1)``. Anything
    else (``id()``, ``enter()``, ``origin()``, ``scale()``, a failure code
    other than 1, no ``failure()`` at all) raises ``ScriptError`` so that
    the declaration is not half-applied.
    """
    text = command.strip()
    if not re.match(r"^stset\b", text, re.I):
        return None
    m = re.match(
        r"^stset\s+([A-Za-z_]\w*)\s*,\s*f(?:a(?:i(?:l(?:u(?:r(?:e)?)?)?)?)?)?"
        r"\(\s*([A-Za-z_]\w*)\s*(?:==\s*1\s*)?\)\s*$",
        text,
        re.I,
    )
    if not m:
        raise ScriptError(
            "only `stset timevar, failure(eventvar)` is understood; build the "
            "duration and the 0/1 event indicator in the data and declare "
            "them in that form"
        )
    return m.group(1), m.group(2)


def panel_declaration(command: str) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """``xtset id [time]`` -> ``(id, time)``; ``tsset time`` -> ``(None, time)``."""
    m = re.match(
        r"^(xtset|tsset)\s+([A-Za-z_]\w*)(?:\s+([A-Za-z_]\w*))?\s*(?:,.*)?$",
        command.strip(),
        re.I,
    )
    if not m:
        return None
    if m.group(1).lower() == "tsset" and m.group(3) is None:
        return None, m.group(2)
    return m.group(2), m.group(3)
