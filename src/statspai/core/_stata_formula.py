"""Stata's factor-variable and time-series syntax inside a formula.

``y ~ i.g##c.x + L.z`` is read the way Stata reads ``regress y i.g##c.x
L.z``:

* factor-variable terms (``i.g``, ``ib2.g``, ``c.x``, ``1.post``, ``a#b``,
  ``a##b``, ``c.x#c.x``, ``i.g#c.(x z)``) become formula terms through the
  translator of ``sp.stata`` / ``sp.from_stata`` -- one implementation,
  checked against Stata 18 there (base level, which cells an interaction
  keeps, ``c.x#c.x`` as the square);
* time-series terms (``L.x``, ``L2.x``, ``F.x``, ``D.x``, ``L(1/3).x``)
  are computed from the panel and time variables declared with
  ``sp.xtset`` and enter the model as columns named ``x_L1``, ``x_L2``,
  ``x_F1``, ``x_D1`` (the names ``sp.stata`` gives them). A lag is read
  from the time variable within panel and is missing where that period
  is absent, as in Stata.

A term that looks like this syntax but cannot be read exactly is refused
with the spelling that works; nothing is guessed.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import ColumnNotFound, MethodIncompatibility

__all__ = [
    "has_stata_ts_terms",
    "resolve_formula_ts_terms",
    "stata_factor_syntax",
]

_PREFIX = r"(?:i|c|ibn|ib\d+|ib\((?:first|last|freq)\)|\d+|o)\."
_COMPONENT = rf"(?:{_PREFIX})?(?:[A-Za-z_]\w*|\([^()]*\))"
_FV_TERM = re.compile(rf"{_COMPONENT}(?:##?{_COMPONENT})*\Z", re.I)
_HAS_PREFIX = re.compile(rf"(?<![\w.]){_PREFIX}[A-Za-z_(]", re.I)
_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")


def _masked(text: str) -> str:
    """``text`` with the inside of quoted names blanked out."""

    def blank(m: "re.Match[str]") -> str:
        q = m.group(0)
        return q[0] + " " * (len(q) - 2) + q[-1]

    return _QUOTED.sub(blank, text)


def _split_top(text: str, seps: str) -> List[Tuple[str, str]]:
    """Split on single characters of ``seps`` outside parentheses and quotes.

    Returns ``[(separator_before, piece), ...]``; the first separator is
    ``''``."""
    masked = _masked(text)
    out: List[Tuple[str, str]] = []
    depth, start, sep = 0, 0, ""
    for i, ch in enumerate(masked):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in seps:
            out.append((sep, text[start:i]))
            sep, start = ch, i + 1
    out.append((sep, text[start:]))
    return out


def _looks_stata(term: str) -> bool:
    t = term.strip()
    if not t or ('"' in t or "'" in t):
        return False
    if not _FV_TERM.match(t):
        return False
    return "#" in t or _HAS_PREFIX.match(t) is not None


def _refuse(message: str, hint: str) -> MethodIncompatibility:
    return MethodIncompatibility(message, recovery_hint=hint)


_HINT = (
    "Write i.g for a factor, c.x for a continuous variable, i.g##c.x for "
    "main effects plus interaction, i.g#c.x for the interaction alone, "
    "ib2.g for base level 2; or the formula spelling C(g) * x."
)


def _fixest_fe_part(parts: List[Tuple[str, str]]) -> Optional[int]:
    """Index of the fixed-effects part of a fixest formula, if any.

    ``y ~ x | fe | d ~ z`` splits into ``y``, ``x``, ``fe``, ``d``, ``z``;
    the part after the first ``|`` is the fixed effects unless it is the
    endogenous side of the IV part (``y ~ x | d ~ z``)."""
    if len(parts) < 3 or parts[1][0] != "~" or parts[2][0] != "|":
        return None
    if len(parts) > 3 and parts[3][0] == "~":
        return None
    return 2


def stata_factor_syntax(
    formula: str,
    columns: Optional[Sequence[str]] = None,
    fixest_fe: bool = False,
) -> str:
    """Translate Stata factor-variable terms of ``formula`` (no-op without).

    ``columns`` expands ranges / wildcards inside grouped terms
    (``c.(x1-x3)``); without it they are refused.

    ``fixest_fe=True`` leaves the fixed-effects part of a fixest formula
    (``y ~ x | fe``) as written: there ``i.f#c.x`` is a varying slope to
    absorb (``reghdfe absorb(i.f#c.x)``), which the fixest layer reads
    itself, not a regressor.
    """
    masked = _masked(formula)
    if "#" not in masked and not _HAS_PREFIX.search(masked):
        return formula
    parts = _split_top(formula, "~|")
    fe_index = _fixest_fe_part(parts) if fixest_fe else None
    plan = []
    any_stata = False
    for k, (sep, part) in enumerate(parts):
        if k == fe_index:
            plan.append((sep, [("", part)], [False]))
            continue
        terms = _split_top(part, "+-")
        flags = [_looks_stata(t) for _, t in terms]
        any_stata = any_stata or any(flags)
        plan.append((sep, terms, flags))
    for k, (_, terms, flags) in enumerate(plan):
        if k == fe_index:
            continue
        for (_, t), f in zip(terms, flags):
            if not f and _HAS_PREFIX.match(t.strip()):
                raise _refuse(
                    f"cannot read the term {t.strip()!r} of {formula!r}: Stata "
                    "factor-variable notation is translated only as a whole "
                    "term (i.g, c.x, i.g##c.x), not inside formula operators",
                    _HINT,
                )
    if not any_stata:
        if fe_index is not None:
            masked = "".join(
                _masked(part) for k, (_, part) in enumerate(parts) if k != fe_index
            )
        if "#" in masked:
            raise _refuse(
                f"cannot read the formula {formula!r}: '#' is Stata's "
                "interaction operator and only whole factor-variable terms "
                "are translated",
                _HINT,
            )
        return formula
    from ..agent._translation._stata import translate_fv_terms

    pieces: List[Tuple[str, List[Tuple[str, str]]]] = []
    for k, (sep, terms, flags) in enumerate(plan):
        if k == fe_index:
            pieces.append((sep, terms))
            continue
        stata = [t.strip() for (_, t), f in zip(terms, flags) if f]
        others = [t.strip() for (_, t), f in zip(terms, flags) if not f and t.strip()]
        if not stata:
            pieces.append((sep, terms))
            continue
        err, translated = translate_fv_terms(stata, others, columns)
        if err is not None:
            raise _refuse(f"cannot read {formula!r}: {err}", _HINT)
        new_terms = []
        it = iter(translated)
        for (op, t), f in zip(terms, flags):
            if not f:
                new_terms.append((op, t))
                continue
            text = next(it)
            lead = " " if t[:1].isspace() else ""
            trail = " " if t[-1:].isspace() else ""
            if op == "-" and " + " in text:
                text = f"({text})"
            new_terms.append((op, f"{lead}{text}{trail}"))
        pieces.append((sep, new_terms))
    out = []
    for sep, terms in pieces:
        out.append(sep + "".join(op + t for op, t in terms))
    return "".join(out)


def has_stata_ts_terms(formula: str) -> bool:
    """Whether ``formula`` holds ``L.x`` / ``F.x`` / ``D.x`` terms (outside
    quoted names)."""
    from ..utils.tsops import TS_TERM

    return TS_TERM.search(_masked(formula)) is not None


def resolve_formula_ts_terms(
    formula: str, data: pd.DataFrame
) -> Tuple[str, pd.DataFrame]:
    """Replace ``L.x`` / ``F.x`` / ``D.x`` terms by computed columns.

    The panel and time variables are the ones declared with ``sp.xtset``
    on ``data``. Returns the rewritten formula and a copy of ``data`` with
    the new columns (``x_L1`` ...).
    """
    from ..utils.tsops import (
        TS_TERM,
        get_xtset,
        parse_ts_ops,
        rewrite_ts_terms,
        ts_column_name,
    )

    if not has_stata_ts_terms(formula):
        return formula, data
    declared = get_xtset(data)
    if declared is None or declared.get("time") is None:
        raise _refuse(
            f"{formula!r} uses a time-series operator (L. / F. / D.), which "
            "reads the time variable, and no time variable is declared on "
            "the data",
            "Call sp.xtset(df, id='unit', time='period') first (time only "
            "for one series), or make the column with sp.lag(df, 'x', "
            "time='period') and name it in the formula.",
        )
    # the computed names must not collide with columns the caller has
    masked = _masked(formula)
    for m in TS_TERM.finditer(masked):
        if m.group("ops") is None:
            continue
        name = ts_column_name(m.group("var"), parse_ts_ops(m.group("ops")))
        if name in data.columns:
            raise _refuse(
                f"{m.group(0)!r} would be computed into the column {name!r}, "
                "which the data already have",
                f"Rename the existing column {name!r}, or use it by name "
                "instead of the operator.",
            )
    panel = (declared.get("id"), declared["time"])
    delta = float(declared.get("delta") or 1.0)
    new_cols = {}
    rebuilt = []
    last = 0
    # rewrite only outside quoted names
    for m in _QUOTED.finditer(formula):
        text, cols = rewrite_ts_terms(
            formula[last : m.start()], data, panel, {}, delta, sep=" + "
        )
        rebuilt.append(text)
        rebuilt.append(m.group(0))
        new_cols.update(cols)
        last = m.end()
    text, cols = rewrite_ts_terms(formula[last:], data, panel, {}, delta, sep=" + ")
    rebuilt.append(text)
    new_cols.update(cols)
    out = "".join(rebuilt)
    left = [m.group(0) for m in TS_TERM.finditer(_masked(out))]
    if left:
        missing = [TS_TERM.match(t).group("var") for t in left]  # type: ignore
        raise ColumnNotFound.from_missing(
            missing, list(data.columns), context="time-series operator"
        )
    frame = data.copy()
    for name, values in new_cols.items():
        frame[name] = np.asarray(values, dtype=float)
    return out, frame
