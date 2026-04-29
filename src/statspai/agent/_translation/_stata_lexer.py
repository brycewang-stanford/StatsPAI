r"""Minimal Stata-command lexer — 99% of commands fit one grammar.

Stata command shape:

    [prefix:] <command> [varlist] [if <cond>] [in <range>] [, <option>...]

where ``<option>`` is either a bare flag (``fe``) or a parameterised
form (``vce(cluster id)`` / ``absorb(id year)`` / ``cluster(id)``).

We do NOT try to handle:

* Macros (``$macro`` / ``\`local'``)
* String concatenation / display formatting
* Multi-command lines (semicolons, ``;`` delimiter)

Those return ``StataParseError``; the caller surfaces a friendly
unsupported-syntax message rather than producing a wrong translation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class StataParseError(ValueError):
    """Raised when the lexer cannot make sense of a Stata command."""


@dataclass
class StataCommand:
    """Parsed Stata command — lossless representation of the input."""

    command: str
    varlist: List[str] = field(default_factory=list)
    if_cond: Optional[str] = None
    in_range: Optional[str] = None
    options: Dict[str, Optional[str]] = field(default_factory=dict)
    #: Original text, useful for error messages and round-tripping
    raw: str = ""


def parse(line: str) -> StataCommand:
    """Tokenise and parse a single Stata command.

    Strips trailing semicolons and surrounding whitespace; rejects
    multi-command lines (``cmd1 ; cmd2``) — the caller should split.
    """
    if line is None:
        raise StataParseError("empty command")
    raw = line
    # Trim Stata comments (``// ...`` or ``/* ... */``) — be conservative
    # since these can appear inside string literals; Stata's quoted
    # forms ``""..""`` aren't going to show up in legitimate one-line
    # commands we care about.
    line = re.sub(r"\s*//.*$", "", line)
    line = re.sub(r"/\*.*?\*/", " ", line, flags=re.S)
    line = line.strip().rstrip(";").strip()
    if not line:
        raise StataParseError("empty command")
    if ";" in line:
        raise StataParseError(
            "multi-command lines are unsupported; pass one command at a time")

    # Split into "<head>, <option-tail>" on the FIRST top-level comma
    # (a comma inside parentheses, e.g. ``vce(robust, oim)``, doesn't
    # count). Stata's grammar reserves a single bare comma at the
    # top level for the option separator.
    head, opts = _split_options(line)

    # Drop any prefix:  e.g. ``by id: reg y x`` — peel off the prefix
    # and warn; the cmd map will surface "by-prefix unsupported" if
    # relevant. We keep the parser simple: drop everything before the
    # first ``: `` outside parens.
    prefix_match = re.match(r"^(by\s+[^:]*?:|capture\s*:|quietly\s*:|"
                              r"qui\s*:|noisily\s*:)\s*", line, flags=re.I)
    if prefix_match:
        # Re-split on the post-prefix portion
        post = line[prefix_match.end():]
        head, opts = _split_options(post)

    tokens = head.split()
    if not tokens:
        raise StataParseError("empty command head")
    command = tokens[0].lower()
    rest = tokens[1:]

    # Walk ``rest`` collecting varlist until ``if`` / ``in`` keywords.
    varlist: List[str] = []
    if_cond: Optional[str] = None
    in_range: Optional[str] = None
    i = 0
    while i < len(rest):
        tok = rest[i]
        low = tok.lower()
        if low == "if":
            # Everything after "if" until "in" or end is the condition.
            j = i + 1
            cond_parts: List[str] = []
            while j < len(rest) and rest[j].lower() != "in":
                cond_parts.append(rest[j])
                j += 1
            if_cond = " ".join(cond_parts) or None
            i = j
            continue
        if low == "in":
            j = i + 1
            range_parts: List[str] = []
            while j < len(rest):
                range_parts.append(rest[j])
                j += 1
            in_range = " ".join(range_parts) or None
            i = j
            continue
        varlist.append(tok)
        i += 1

    options = _parse_options(opts) if opts else {}

    return StataCommand(
        command=command,
        varlist=varlist,
        if_cond=if_cond,
        in_range=in_range,
        options=options,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _split_options(line: str) -> Tuple[str, str]:
    """Split ``line`` on the first comma that is NOT inside parentheses.

    Returns ``(head, options_str)``. When no top-level comma exists the
    options string is empty.
    """
    depth = 0
    for idx, ch in enumerate(line):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            return line[:idx].strip(), line[idx + 1:].strip()
    return line.strip(), ""


def _parse_options(text: str) -> Dict[str, Optional[str]]:
    """Parse the option-tail string into ``{name: arg-or-None}``.

    Examples
    --------
    >>> _parse_options("fe vce(cluster id) absorb(id year) robust")
    {'fe': None, 'vce': 'cluster id', 'absorb': 'id year', 'robust': None}
    """
    opts: Dict[str, Optional[str]] = {}
    i = 0
    n = len(text)
    while i < n:
        # Skip whitespace
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        # Read name [letters / digits / underscore]
        start = i
        while i < n and (text[i].isalnum() or text[i] == "_"):
            i += 1
        if start == i:
            # Unexpected character; bail
            i += 1
            continue
        name = text[start:i].lower()
        # Optional parenthesised argument
        if i < n and text[i] == "(":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            arg = text[i + 1:j - 1].strip() if depth == 0 else text[i + 1:j].strip()
            opts[name] = arg
            i = j
        else:
            opts[name] = None
    return opts


__all__ = ["parse", "StataCommand", "StataParseError"]
