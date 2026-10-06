"""The data-management commands ``sp.stata`` runs between estimation lines.

A do-file rarely fits a model on the data as loaded: it generates a few
variables, keeps a subsample, sorts. ``sp.stata`` runs the small set of
commands below on a private copy of the DataFrame, so that the estimation
lines that follow see the data Stata would have had:

    generate [type] newvar = exp [if] [in]
    replace var = exp [if] [in]
    keep / drop  if exp | in range | varlist
    sort varlist            gsort [+|-]var ...
    rename old new
    tabulate var, generate(stub)
    collapse (stat) [new=]var ... [if] [, by(varlist)]
    ipolate y x, generate(new) [epolate]
    mvdecode varlist, mv(#)
    encode strvar, generate(newvar)
    decode var, generate(newvar)
    label variable / define / values / data / drop
    preserve / restore
    set obs #               clear / drop _all

Expressions go through :mod:`._stata_expr` (Stata's missing-value rules).
``egen`` is in ``_stata_egen.py``. Anything else -- ``merge``, ``reshape`` -- is
not a data step here, and ``sp.stata`` refuses the snippet.

Storage follows Stata: ``generate`` without a type stores a ``float`` (single
precision), so ``gen x = 0.1`` holds 0.100000001490116 and a regression on it
reproduces Stata's digits. ``generate double`` keeps full precision.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ._stata_expr import (
    StataExprError,
    evaluate,
    in_range_mask,
    names_extended_missing,
    sample_mask,
)
from ._stata_lexer import StataParseError
from ._stata_lexer import parse as _parse_stata

__all__ = ["DataSteps", "is_data_step", "row_mask"]

_INT_TYPES = ("byte", "int", "long")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ASSIGN = re.compile(r"\s*(?:(\w+)\s+)?([A-Za-z_]\w*)\s*=(?!=)\s*(.+)\Z", re.S)


#: ``label <subcommand> ...``, the command abbreviated down to ``la``
_LABEL = re.compile(r"\s*la(?:b(?:e(?:l)?)?)?\s+(\w+)\s*(.*)\Z", re.S | re.I)
#: one `# "text"` pair of `label define`; the text may be bare or in
#: double or compound quotes
_LABEL_PAIR = re.compile(r"\s*(-?\d+|\.[a-z])\s+(`\"[^`]*?\"'|\"[^\"]*\"|[^\s,]+)\s*")


def _abbrev(word: str, full: str, minimum: int) -> bool:
    return len(word) >= minimum and full.startswith(word)


def _unquote(text: str) -> str:
    text = text.strip()
    if text.startswith('`"') and text.endswith("\"'"):
        return text[2:-2]
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def _split_options(rest: str) -> tuple:
    """Split ``body, options`` at the first comma outside quotes."""
    quoted = False
    for i, ch in enumerate(rest):
        if ch == '"':
            quoted = not quoted
        elif ch == "," and not quoted:
            return rest[:i], ",", rest[i + 1 :]
    return rest, "", ""


def _is_generate(word: str) -> bool:
    return bool(word) and "generate".startswith(word)


def _is_tabulate(word: str) -> bool:
    return len(word) >= 2 and "tabulate".startswith(word)


def is_data_step(command: str) -> bool:
    return (
        _is_generate(command)
        or command
        in (
            "replace",
            "keep",
            "drop",
            "sort",
            "gsort",
            "preserve",
            "restore",
            "mvdecode",
            "mvencode",
            "encode",
            "decode",
            "collapse",
            "reshape",
            "ipolate",
            "egen",
            "xi",
            "ebalance",
            "cem",
        )
        or command in ("ren", "rena", "renam", "rename")
    )


def _numlist(spec: str, what: str) -> List[float]:
    """A Stata numlist: ``1 2 3``, ``1/5``, ``-7/-1``, ``0(10)50``."""
    out: List[float] = []
    number = r"-?(?:\d+\.?\d*|\.\d+)"
    for tok in spec.replace(",", " ").split():
        stepped = re.fullmatch(rf"({number})\(({number})\)({number})", tok)
        ranged = re.fullmatch(rf"({number})(?:/|\s*to\s*)({number})", tok)
        if stepped:
            lo, step, hi = (float(g) for g in stepped.groups())
            if step == 0 or (hi - lo) / step < 0:
                raise StataExprError(f"{what}: cannot read the list {tok!r}")
            out.extend(lo + k * step for k in range(int((hi - lo) / step + 1e-9) + 1))
        elif ranged:
            lo, hi = float(ranged.group(1)), float(ranged.group(2))
            step = 1.0 if hi >= lo else -1.0
            out.extend(lo + k * step for k in range(int(abs(hi - lo) + 1e-9) + 1))
        elif re.fullmatch(number, tok):
            out.append(float(tok))
        else:
            raise StataExprError(f"{what}: cannot read the number {tok!r}")
    if not out:
        raise StataExprError(f"{what}: an empty list of numbers")
    return out


def expand_varlist(cols: List[str], varlist: List[str]) -> List[str]:
    """``varlist`` against the columns ``cols``, in Stata's reading.

    ``a-b`` (also written ``a - b``) is every column from ``a`` to ``b`` in
    dataset order; ``x*`` / ``x?`` / ``x~y`` are wildcards; ``_all`` is every
    column; a name that is not a column but begins exactly one is that
    column (``set varabbrev on``, Stata's default). A token that fits
    nothing is returned as it is, for the caller to report.
    """
    import fnmatch

    tokens: List[str] = []
    for tok in varlist:
        if tok == "-" and tokens:
            tokens[-1] += "-"
        elif tokens and tokens[-1].endswith("-") and not tok.startswith("-"):
            tokens[-1] += tok
        elif tok.startswith("-") and len(tok) > 1 and tokens:
            tokens[-1] += tok
        else:
            tokens.append(tok)

    def one(name: str) -> str:
        if name in cols:
            return name
        hits = [c for c in cols if c.startswith(name)]
        if len(hits) > 1:
            raise StataExprError(
                f"{name!r} is an ambiguous abbreviation: it fits {hits[:4]}"
            )
        return hits[0] if hits else name

    out: List[str] = []
    for tok in tokens:
        if tok == "_all":
            out.extend(cols)
            continue
        if tok in cols:
            out.append(tok)
            continue
        first, dash, last = tok.partition("-")
        if dash and first and last:
            first, last = one(first), one(last)
            if first in cols and last in cols:
                i, j = cols.index(first), cols.index(last)
                if i > j:
                    raise StataExprError(
                        f"varlist range {tok!r}: {first!r} comes after "
                        f"{last!r} in the data"
                    )
                out.extend(cols[i : j + 1])
                continue
        if any(ch in tok for ch in "*?~"):
            hits = fnmatch.filter(cols, tok.replace("~", "*"))
            if not hits:
                raise StataExprError(f"varlist {tok!r} matches no variable")
            out.extend(hits)
        else:
            out.append(one(tok))
    return list(dict.fromkeys(out))


def _generate_option(options: dict) -> Optional[str]:
    """The value of ``generate(...)``, however it was abbreviated."""
    for key in list(options):
        if key and "generate".startswith(key):
            return str(options.pop(key) or "").strip() or None
    return None


#: collapse statistics -> the pandas reduction (missing values are skipped,
#: and an all-missing group gives missing, as in Stata)
_COLLAPSE = {
    "mean": "mean",
    "median": "median",
    "p50": "median",
    "sd": "std",
    "sum": "sum",
    "rawsum": "sum",
    "count": "count",
    "max": "max",
    "min": "min",
    # pandas' first / last skip missing values, which is Stata's firstnm /
    # lastnm; Stata's first / last take the row whatever it holds
    "firstnm": "first",
    "lastnm": "last",
    "first": "_row_first",
    "last": "_row_last",
}


def _row_end(how: str) -> Any:
    position = 0 if how == "_row_first" else -1
    return lambda s: s.iloc[position] if len(s) else np.nan


def row_mask(
    data: pd.DataFrame,
    if_cond: Optional[str],
    in_range: Optional[str],
    stored: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Rows selected by an ``if`` condition and / or an ``in`` range."""
    mask = np.ones(len(data), dtype=bool)
    if if_cond:
        mask &= sample_mask(if_cond, data, stored)
    if in_range:
        mask &= in_range_mask(in_range, len(data))
    return mask


class DataSteps:
    """Runs data steps on a private copy of the caller's DataFrame."""

    def __init__(self, data: pd.DataFrame) -> None:
        self._original = data
        self._owned = False
        self.data = data
        #: variables stored as Stata ``float`` (single precision)
        self._float: Set[str] = set()
        self._stack: List[tuple] = []
        #: r() / e() / _b[] / scalars, filled in by the session
        self.stored: Dict[str, Any] = {}
        #: value-label sets defined by `label define`, by name, and the set
        #: each variable was given by `label values`
        self._label_sets: Dict[str, Dict[Any, str]] = {}
        self._set_of: Dict[str, str] = {}
        #: columns made from `L.x` / `F.x` terms: column -> (variable,
        #: operators), and the (panel, time) pair they were read against
        self.ts_derived: Dict[str, tuple] = {}
        self.ts_panel: tuple = (None, None)
        self._adopt_label_sets()

    def _adopt_label_sets(self) -> None:
        """Take over the value-label sets the frame came with.

        A frame read from a .dta file names each variable's set in
        ``attrs['_value_label_names']``; a variable labelled without a
        name has a set named after itself, as `encode` would create.
        """
        attrs = self.data.attrs
        regular = attrs.get("_value_labels") or {}
        gaps = attrs.get("_missing_labels") or {}
        names = attrs.get("_value_label_names") or {}
        for var in list(regular) + [v for v in gaps if v not in regular]:
            if var not in self.data.columns:
                continue
            table = {**(regular.get(var) or {}), **(gaps.get(var) or {})}
            name = names.get(var) or str(var)
            if name in self._label_sets and self._label_sets[name] != table:
                name = str(var)
            self._label_sets.setdefault(name, table)
            self._set_of[var] = name

    def _own(self) -> None:
        if not self._owned:
            self.data = self._original.copy()
            self._owned = True

    def apply(self, line: str) -> bool:
        """Run ``line`` if it is a data step; return whether it was one."""
        try:
            cmd = _parse_stata(line)
        except StataParseError:
            return False
        if not is_data_step(cmd.command):
            return False
        if cmd.command == "egen":
            from ._stata_egen import run_egen

            run_egen(self, line)
            return True
        if cmd.command in ("xi", "ebalance", "cem"):
            from ._stata_balance import run_balance

            if cmd.if_cond or cmd.in_range:
                raise StataExprError(
                    f"`{cmd.command}` with an if / in qualifier is not implemented"
                )
            run_balance(self, cmd.command, list(cmd.varlist), dict(cmd.options))
            return True
        if cmd.command in ("mvdecode", "mvencode"):
            step = self._mvdecode if cmd.command == "mvdecode" else self._mvencode
            qualifier = (
                (cmd.if_cond, cmd.in_range) if cmd.if_cond or cmd.in_range else None
            )
            step(list(cmd.varlist), dict(cmd.options), qualifier)
            return True
        if cmd.command == "encode":
            self._encode(cmd.varlist, dict(cmd.options), cmd.if_cond or cmd.in_range)
            return True
        if cmd.command == "decode":
            self._decode(cmd.varlist, dict(cmd.options), cmd.if_cond or cmd.in_range)
            return True
        if cmd.command == "collapse":
            self._collapse(cmd.varlist, dict(cmd.options), cmd.if_cond, cmd.in_range)
            return True
        if cmd.command == "reshape":
            from ._stata_reshape import run_reshape

            if cmd.if_cond or cmd.in_range:
                raise StataExprError("`reshape` takes no if / in qualifier")
            run_reshape(self, list(cmd.varlist), dict(cmd.options))
            return True
        if cmd.command == "ipolate":
            self._ipolate(cmd.varlist, dict(cmd.options), cmd.if_cond or cmd.in_range)
            return True
        if cmd.options:
            raise StataExprError(
                f"options {sorted(cmd.options)} of `{cmd.command}` are not "
                "implemented"
            )
        body = " ".join(cmd.varlist)
        if _is_generate(cmd.command) or cmd.command == "replace":
            self._assign(cmd.command == "replace", body, cmd.if_cond, cmd.in_range)
        elif cmd.command in ("keep", "drop"):
            self._keep_drop(
                cmd.command == "keep", cmd.varlist, cmd.if_cond, cmd.in_range
            )
        elif cmd.command == "sort":
            self._sort(cmd.varlist, cmd.if_cond or cmd.in_range)
        elif cmd.command == "gsort":
            self._gsort(cmd.varlist, cmd.if_cond or cmd.in_range)
        elif cmd.command.startswith("ren"):
            self._rename(cmd.varlist, cmd.if_cond or cmd.in_range)
        elif cmd.command == "preserve":
            self._stack.append(
                (
                    self.data.copy(),
                    set(self._float),
                    {k: dict(v) for k, v in self._label_sets.items()},
                    dict(self._set_of),
                )
            )
        else:  # restore
            if not self._stack:
                raise StataExprError("`restore` without a `preserve`")
            self.data, self._float, self._label_sets, self._set_of = self._stack.pop()
            self._owned = True
        return True

    # ------------------------------------------------------------ steps
    def _assign(
        self,
        replace: bool,
        body: str,
        if_cond: Optional[str],
        in_range: Optional[str],
        groups: Optional[List[np.ndarray]] = None,
    ) -> None:
        m = _ASSIGN.match(body)
        if m is None:
            raise StataExprError(f"expected `[type] var = exp`, got {body!r}")
        vtype, name, expr = m.group(1), m.group(2), m.group(3)
        if replace and vtype is not None:
            raise StataExprError("`replace` does not take a storage type")
        is_str_type = vtype is not None and re.fullmatch(r"str(\d+|L)", vtype)
        if (
            vtype is not None
            and not is_str_type
            and vtype not in _INT_TYPES + ("float", "double")
        ):
            raise StataExprError(f"storage type {vtype!r} is not implemented")
        exists = name in self.data.columns
        if replace and not exists:
            raise StataExprError(f"`replace`: variable {name!r} does not exist")
        if not replace and exists:
            raise StataExprError(f"`generate`: variable {name!r} already exists")

        if replace and self._replace_in_order(name, expr, if_cond, in_range, groups):
            return
        value, mask = self._evaluate_assignment(expr, if_cond, in_range, groups)
        if groups is None and (value.dtype == object or is_str_type):
            self._assign_text(replace, name, value, mask, bool(is_str_type))
            return

        single = name in self._float if replace else vtype in (None, "float")
        if replace and exists and self.data[name].dtype == np.float32:
            single = True
        if vtype in _INT_TYPES:
            held = value[mask & ~np.isnan(value)]
            if not np.all(held == np.round(held)):
                raise StataExprError(
                    f"`generate {vtype}` of non-integer values truncates in "
                    "Stata; that is not implemented"
                )
        if single:
            with np.errstate(over="ignore"):
                value = value.astype(np.float32).astype(np.float64)
            value = np.where(np.isfinite(value), value, np.nan)

        self._own()
        if replace:
            old = self.data[name]
            if not (
                pd.api.types.is_numeric_dtype(old) or pd.api.types.is_bool_dtype(old)
            ):
                raise StataExprError(
                    f"type mismatch: {name!r} is a string variable and the "
                    "expression is a number"
                )
            current = old.to_numpy(dtype=float, na_value=np.nan)
            self.data[name] = np.where(mask, value, current)
        else:
            self.data[name] = np.where(mask, value, np.nan)
            if single:
                self._float.add(name)
        self._note_missing_codes(name, expr, whole=not replace or bool(mask.all()))

    def _evaluate_assignment(
        self,
        expr: str,
        if_cond: Optional[str],
        in_range: Optional[str],
        groups: Optional[List[np.ndarray]],
    ) -> tuple:
        """The expression on every row, and the rows the qualifiers select."""
        if groups is None:
            value = evaluate(expr, self.data, self.stored)
            mask = row_mask(self.data, if_cond, in_range, self.stored)
            return value, mask
        # `by g:` -- the expression sees one group at a time, so `_n`,
        # `_N` and subscripts count within the group
        if in_range:
            raise StataExprError("`in` may not be combined with `by`")
        value = np.full(len(self.data), np.nan)
        mask = np.zeros(len(self.data), dtype=bool)
        for rows in groups:
            part = self.data.iloc[rows].reset_index(drop=True)
            got = evaluate(expr, part, self.stored)
            if got.dtype == object:
                raise StataExprError("a string variable by group is not generated here")
            value[rows] = got
            mask[rows] = row_mask(part, if_cond, None, self.stored)
        return value, mask

    def _replace_in_order(
        self,
        name: str,
        expr: str,
        if_cond: Optional[str],
        in_range: Optional[str],
        groups: Optional[List[np.ndarray]],
    ) -> bool:
        """``replace`` of a variable that reads its own earlier rows.

        Stata replaces one observation after another, so ``replace x =
        0.5 * x[_n-1] + e in 2/l`` builds a recursion: each row reads the
        value the row before it was just given. A reference from row ``i``
        to row ``j`` of the variable being replaced therefore sees the new
        value when ``j < i`` and the old one otherwise. The same holds for
        ``L.x`` after ``tsset``.

        Returns ``False`` when no row reads an earlier row of ``name``; one
        evaluation of the whole column is exact then.
        """
        old = self.data[name]
        if not (pd.api.types.is_numeric_dtype(old) or pd.api.types.is_bool_dtype(old)):
            return False
        n = len(self.data)
        texts = [expr, if_cond or ""]
        own = re.compile(rf"(?<![\w.]){re.escape(name)}\s*\[([^\[\]]*)\]")
        subscripts: List[str] = []
        for text in texts:
            for sub in own.findall(text):
                if sub.strip() not in subscripts:
                    subscripts.append(sub.strip())
        lagged = [
            col
            for col, (var, _ops) in self.ts_derived.items()
            if var == name
            and col in self.data.columns
            and any(re.search(rf"(?<![\w.]){re.escape(col)}(?!\w)", t) for t in texts)
        ]
        if not subscripts and not lagged:
            return False
        if n == 0:
            return False

        position = np.arange(n)
        sources: Dict[str, np.ndarray] = {}
        rewritten = {}
        for k, sub in enumerate(subscripts):
            source = np.full(n, -1)
            parts = [position] if groups is None else groups
            for rows in parts:
                part = self.data.iloc[rows].reset_index(drop=True)
                at = evaluate(sub, part, self.stored)
                if at.dtype == object:
                    raise StataExprError(f"the subscript {sub!r} is not a number")
                ok = (
                    ~np.isnan(at) & (at >= 1) & (at <= len(rows)) & (at == np.floor(at))
                )
                where = np.asarray(rows)
                source[where[ok]] = where[at[ok].astype(int) - 1]
            column = f"__sp_own{k}"
            sources[column] = source
            rewritten[sub] = column
        if lagged:
            from ._stata_tsops import _Clock

            unit, time = self.ts_panel
            clock = _Clock(self.data, unit, time)
            for col in lagged:
                ops = self.ts_derived[col][1]
                if any(kind == "D" for kind, _ in ops):
                    raise StataExprError(
                        f"`replace {name}` from its own difference is not implemented"
                    )
                offset = sum(k if kind == "F" else -k for kind, k in ops)
                at = clock.at(position.astype(float), offset)
                sources[col] = np.where(np.isnan(at), -1, at).astype(int)
        if not any(((src >= 0) & (src < position)).any() for src in sources.values()):
            return False  # every reference is to the row itself or a later one

        def swap(m: "re.Match[str]") -> str:
            return rewritten[m.group(1).strip()]

        expr_run = own.sub(swap, expr)
        cond_run = own.sub(swap, if_cond) if if_cond else if_cond

        single = name in self._float or old.dtype == np.float32
        original = old.to_numpy(dtype=float, na_value=np.nan)
        current = original.copy()
        rng = self.stored.get("rng")
        draws = None if rng is None else rng.bit_generator.state
        self._own()
        held = {c: self.data[c].to_numpy(copy=True) for c in lagged}
        # Each pass settles the rows whose earlier rows are settled, so the
        # passes needed are the length of the longest chain. A chain through
        # the whole column settles one row per pass, which is quadratic:
        # beyond `limit` rows it is refused, as soon as it shows.
        limit = 20000
        too_long = StataExprError(
            f"`replace {name}` chains more than {limit} observations "
            "through its own earlier rows; that is not run here"
        )
        # by group, an expression that reads nothing but the row it is on
        # (no _n, _N, subscript or running sum left after the rewrite) and a
        # condition that does not read the variable's earlier rows
        counted = re.compile(r"(?<![\w.])(?:_n|_N)(?!\w)|\[|(?<![\w.])sum\s*\(")
        reads_own = re.compile(
            "|".join(rf"(?<![\w.]){re.escape(c)}(?!\w)" for c in sources)
        )
        row_local = (
            groups is not None
            and counted.search(expr_run) is None
            and reads_own.search(cond_run or "") is None
        )
        pending: List[int] = []
        settled = False
        stale: Optional[np.ndarray] = None
        value = mask = np.zeros(0)
        group_of = np.zeros(n, dtype=int)
        for k, rows in enumerate(groups or []):
            group_of[rows] = k
        try:
            for step in range(n + 1):
                if step > limit:
                    raise too_long
                if (
                    len(pending) > 32
                    and pending[-1] > limit
                    and pending[-33] - pending[-1] <= 64
                ):
                    raise too_long
                for column, src in sources.items():
                    safe = np.where(src < 0, 0, src)
                    seen = np.where(src < position, current[safe], original[safe])
                    self.data[column] = np.where(src < 0, np.nan, seen)
                if draws is not None:
                    # the same random draws on every pass
                    rng.bit_generator.state = draws
                if groups is not None and row_local:
                    # the same value whichever group a row is in: one
                    # evaluation of the column; the rows selected do not
                    # change from pass to pass
                    if step == 0:
                        _, mask = self._evaluate_assignment(
                            "0", cond_run, in_range, groups
                        )
                    value = evaluate(expr_run, self.data, self.stored)
                elif groups is None or stale is None:
                    value, mask = self._evaluate_assignment(
                        expr_run, cond_run, in_range, groups
                    )
                else:
                    # by group: only the groups a changed row is read from
                    again = [g for k, g in enumerate(groups) if stale[k]]
                    part_value, part_mask = self._evaluate_assignment(
                        expr_run, cond_run, in_range, again
                    )
                    rows = np.concatenate(again)
                    value, mask = value.copy(), mask.copy()
                    value[rows], mask[rows] = part_value[rows], part_mask[rows]
                if value.dtype == object:
                    raise StataExprError(
                        f"type mismatch: {name!r} is numeric and the expression "
                        "is a string"
                    )
                if single:
                    with np.errstate(over="ignore"):
                        value = value.astype(np.float32).astype(np.float64)
                    value = np.where(np.isfinite(value), value, np.nan)
                new = np.where(mask, value, original)
                moved = ~((new == current) | (np.isnan(new) & np.isnan(current)))
                if not moved.any():
                    break
                pending.append(int(moved.sum()))
                current = new
                if groups is not None:
                    # the rows that read a row which has just changed
                    reads = np.zeros(n, dtype=bool)
                    for src in sources.values():
                        reads |= (src >= 0) & moved[np.where(src < 0, 0, src)]
                    stale = np.zeros(len(groups), dtype=bool)
                    stale[np.unique(group_of[reads])] = True
                    if not stale.any():
                        break
            settled = True
        finally:
            if not settled:
                # the command failed: the lag columns go back as they were
                for col, values in held.items():
                    self.data[col] = values
            self.data = self.data.drop(
                columns=list(rewritten.values()), errors="ignore"
            )
        self.data[name] = current
        for col in lagged:
            src = sources[col]
            self.data[col] = np.where(
                src < 0, np.nan, current[np.where(src < 0, 0, src)]
            )
        self._note_missing_codes(name, expr, whole=bool(mask.all()))
        return True

    def _assign_text(
        self, replace: bool, name: str, value: Any, mask: np.ndarray, typed: bool
    ) -> None:
        """``generate s = "text"`` / ``replace s = strupper(s) if ...``: a
        string variable; where the qualifier is false a new one is empty."""
        if value.dtype != object:
            raise StataExprError(
                f"type mismatch: {name!r} is declared a string and the "
                "expression is a number"
            )
        self._own()
        if replace:
            old = self.data[name]
            if pd.api.types.is_numeric_dtype(old) or pd.api.types.is_bool_dtype(old):
                raise StataExprError(
                    f"type mismatch: {name!r} is numeric and the expression "
                    "is a string"
                )
            current = old.astype(object).where(old.notna(), "").to_numpy(dtype=object)
            self.data[name] = np.where(mask, value, current).astype(object)
        else:
            self.data[name] = np.where(mask, value, "").astype(object)

    # ------------------------------------------------- extended missing values
    @property
    def coded(self) -> Set[str]:
        """Variables that may hold an extended missing value (``.a``-``.z``).

        The data keep every missing value as NaN, so an expression that
        would tell ``.`` from ``.a`` on one of these is refused (see
        ``_stata_expr._compare_missing``).
        """
        held: Set[str] = self.stored.setdefault("ext_missing", set())
        return held

    def adopt_missing_codes(self, data: pd.DataFrame) -> None:
        """Note which variables of a frame that is taken in hold extended
        missing values: the ones its reader listed in
        ``attrs['_ext_missing']``, the ones with a ``<var>__miss`` column
        (``sp.read_data(extended_missing='column')``) and the ones whose
        value labels name such a code."""
        attrs = data.attrs
        names = set(attrs.get("_ext_missing") or ())
        names |= set(attrs.get("_missing_labels") or {})
        names |= {
            str(c)[: -len("__miss")] for c in data.columns if str(c).endswith("__miss")
        }
        self.coded.update(n for n in names if n in data.columns)

    def _note_missing_codes(self, name: str, expr: str, *, whole: bool) -> None:
        if names_extended_missing(expr, self.coded):
            self.coded.add(name)
        elif whole:
            self.coded.discard(name)

    def by_assign(
        self, keys: List[str], order: List[str], sort: bool, line: str
    ) -> bool:
        """``by g: generate`` / ``bysort g (t): replace``; whether it was one.

        ``bysort`` sorts by the group variables and then the ones in
        parentheses (ties keep their order); plain ``by`` needs the data
        already in runs of the group variables, as Stata does.
        """
        try:
            cmd = _parse_stata(line)
        except StataParseError:
            return False
        is_egen = cmd.command == "egen"
        if not (_is_generate(cmd.command) or cmd.command == "replace" or is_egen):
            return False
        if cmd.options and not is_egen:
            raise StataExprError(
                f"options {sorted(cmd.options)} of `{cmd.command}` are not "
                "implemented"
            )
        unknown = [k for k in keys + order if k not in self.data.columns]
        if unknown:
            raise StataExprError(f"by: variable(s) {unknown} are not in the data")
        if sort:
            self._sort(keys + order, None)
        for key in keys + order:
            if key in self.coded and self.data[key].isna().any():
                raise StataExprError(
                    f"by: {key!r} may hold extended missing values (.a-.z); "
                    "Stata groups and sorts each kind apart, and the data "
                    "keep them as one. Drop or recode the missing rows first"
                )
        codes = self.data.groupby(keys, sort=False, dropna=False).ngroup().to_numpy()
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        if len(starts) != len(np.unique(codes)):
            raise StataExprError(
                "not sorted: `by` needs the data sorted by " + " ".join(keys)
            )
        bounds = np.r_[starts, len(codes)]
        groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(len(starts))]
        if is_egen:
            from ._stata_egen import run_egen

            run_egen(self, line, groups)
            return True
        self._assign(
            cmd.command == "replace",
            " ".join(cmd.varlist),
            cmd.if_cond,
            cmd.in_range,
            groups=groups,
        )
        return True

    def _mvdecode(self, varlist: List[str], options: dict, qualifier: Any) -> None:
        """``mvdecode varlist [if] [in], mv(numlist [= mvc] [\\ ...])``: turn
        the listed values into missing values."""
        raw = options.pop("mv", None)
        if options or raw is None:
            raise StataExprError(
                "mvdecode: expected `mvdecode varlist, mv(numlist [=mvc])`"
            )
        rules = []
        for part in str(raw).split("\\"):
            numbers, eq, code = part.partition("=")
            code = code.strip() or "."
            if not re.fullmatch(r"\.[a-z]?", code):
                raise StataExprError(f"mvdecode: {code!r} is not a missing value")
            rules.append((_numlist(numbers.strip(), "mvdecode"), code != "."))
        varlist = self.expand_varlist(varlist)
        mask = self._qualifier_mask(qualifier)
        self._own()
        for name in varlist:
            old = self.data[name]
            if not pd.api.types.is_numeric_dtype(old):
                continue  # Stata skips string variables with a note
            col = old.to_numpy(dtype=float, na_value=np.nan)
            for values, extended in rules:
                hit = mask & np.isin(col, values)
                if hit.any():
                    col = np.where(hit, np.nan, col)
                    if extended:
                        self.coded.add(name)
            self.data[name] = col

    def _mvencode(self, varlist: List[str], options: dict, qualifier: Any) -> None:
        """``mvencode varlist [if] [in], mv(# | mvc=# [\\ ...]) [override]``:
        turn missing values into numbers."""
        raw = options.pop("mv", None)
        override = any(
            options.pop(k, 0) is None for k in ("override", "o", "ov", "over")
        )
        if options or raw is None:
            raise StataExprError("mvencode: expected `mvencode varlist, mv(#)`")
        parts = [p.strip() for p in str(raw).split("\\")]
        every: Optional[float] = None
        sysmiss: Optional[float] = None
        for part in parts:
            code, eq, number = part.partition("=")
            if not eq:
                every = float(_numlist(code, "mvencode")[0])
            elif code.strip() == "else":
                every = float(number)
            elif code.strip() == ".":
                sysmiss = float(number)
            else:
                # `.c = 0` alone: the rows that hold .c are not known
                raise StataExprError(
                    f"mvencode: `{part}` recodes one kind of extended missing "
                    "value, and the data keep all kinds as one"
                )
        varlist = self.expand_varlist(varlist)
        mask = self._qualifier_mask(qualifier)
        self._own()
        for name in varlist:
            old = self.data[name]
            if not pd.api.types.is_numeric_dtype(old):
                continue
            col = old.to_numpy(dtype=float, na_value=np.nan)
            target = every if every is not None else sysmiss
            if every is None and name in self.coded:
                raise StataExprError(
                    f"mvencode: {name!r} may hold extended missing values, "
                    "which `.=#` leaves alone and the data do not tell apart"
                )
            hit = mask & np.isnan(col)
            if not override and hit.any() and np.any(col[mask] == target):
                raise StataExprError(
                    f"mvencode: {name!r} already holds the value "
                    f"{target:g}; Stata stops here unless `override` is given"
                )
            self.data[name] = np.where(hit, float(target or 0.0), col)
            if every is not None and bool(mask.all()):
                self.coded.discard(name)

    def _qualifier_mask(self, qualifier: Any) -> np.ndarray:
        """Rows an ``(if_cond, in_range)`` pair keeps (all rows for None)."""
        if not qualifier:
            return np.ones(len(self.data), dtype=bool)
        if_cond, in_range = qualifier
        return row_mask(self.data, if_cond, in_range, self.stored)

    def _encode(self, varlist: List[str], options: dict, qualifier: Any) -> None:
        """``encode strvar, generate(newvar)``: codes 1..K in sorted order."""
        # encode accepts generate() down to a single letter
        new = (
            options.pop("generate", None)
            or options.pop("gen", None)
            or options.pop("g", None)
        )
        if qualifier or options or new is None or len(varlist) != 1:
            raise StataExprError(
                "only `encode strvar, generate(newvar)` is implemented"
            )
        source, new = varlist[0], str(new).strip()
        if source not in self.data.columns:
            raise StataExprError(f"encode: variable {source!r} is not in the data")
        if new in self.data.columns:
            raise StataExprError(f"encode: variable {new!r} already exists")
        col = self.data[source]
        if pd.api.types.is_numeric_dtype(col):
            raise StataExprError(f"encode: {source!r} is not a string variable")
        present = col.notna() & (col.astype(str) != "")
        levels = sorted(col[present].astype(str).unique())
        codes = col.astype(str).map({lv: i + 1.0 for i, lv in enumerate(levels)})
        self._own()
        self.data[new] = np.where(present, codes, np.nan)
        # Stata defines a value label named after the new variable
        self._label_sets[new] = {i + 1: lv for i, lv in enumerate(levels)}
        self._set_of[new] = new
        self._sync_value_labels([new])
        source_label = (self.data.attrs.get("_labels") or {}).get(source)
        if source_label:
            self._set_attr("_labels", new, source_label)

    def _decode(self, varlist: List[str], options: dict, qualifier: Any) -> None:
        """``decode var, generate(newvar)``: the label texts, as a string.

        A value without a label, and a missing value, gives ``""``.
        """
        new = _generate_option(options)
        if qualifier or options or new is None or len(varlist) != 1:
            raise StataExprError("only `decode var, generate(newvar)` is implemented")
        source = varlist[0]
        if source not in self.data.columns:
            raise StataExprError(f"decode: variable {source!r} is not in the data")
        if new in self.data.columns:
            raise StataExprError(f"decode: variable {new!r} already exists")
        mapping = (self.data.attrs.get("_value_labels") or {}).get(source)
        if not mapping:
            raise StataExprError(f"decode: {source!r} has no value label")
        col = self.data[source]
        texts = col.map(
            lambda v: mapping.get(int(v), "") if pd.notna(v) and v == int(v) else ""
        )
        self._own()
        self.data[new] = texts.astype(object)
        source_label = (self.data.attrs.get("_labels") or {}).get(source)
        if source_label:
            self._set_attr("_labels", new, source_label)

    # ----------------------------------------------------------- labels
    def _set_attr(self, key: str, column: str, value: Any) -> None:
        """Set (or with ``None`` remove) one column's entry of a label attr."""
        self._own()
        store = dict(self.data.attrs.get(key) or {})
        if value is None or value == {} or value == "":
            store.pop(column, None)
        else:
            store[column] = value
        if store:
            self.data.attrs[key] = store
        else:
            self.data.attrs.pop(key, None)

    def _sync_value_labels(self, variables: List[str]) -> None:
        """Write each variable's label set into the frame's attrs."""
        for var in variables:
            name = self._set_of.get(var)
            mapping = self._label_sets.get(name, {}) if name else {}
            regular = {k: v for k, v in mapping.items() if not isinstance(k, str)}
            gaps = {k: v for k, v in mapping.items() if isinstance(k, str)}
            self._set_attr("_value_labels", var, regular)
            self._set_attr("_missing_labels", var, gaps)
            self._set_attr("_value_label_names", var, name if mapping else None)

    def apply_label(self, line: str) -> bool:
        """Run a ``label`` command that changes labels; ``False`` otherwise.

        ``label variable``, ``label define``, ``label values``, ``label
        data`` and ``label drop`` are run on the frame's label metadata
        (``attrs``).  ``label list`` / ``dir`` / ``language`` and the rest
        only print or are not modelled, and are left to the caller.
        """
        m = _LABEL.match(line)
        if m is None:
            return False
        sub, rest = m.group(1).lower(), m.group(2).strip()
        if _abbrev(sub, "variable", 3):
            name, _, text = rest.partition(" ")
            if name not in self.data.columns:
                # Stata takes an unambiguous abbreviation of a variable name
                full = [str(c) for c in self.data.columns if str(c).startswith(name)]
                if len(full) == 1:
                    name = full[0]
            if name not in self.data.columns:
                raise StataExprError(
                    f"label variable: variable {name!r} is not in the data"
                )
            self._set_attr("_labels", name, _unquote(text.strip()))
            return True
        if _abbrev(sub, "data", 2):
            text = _unquote(rest)
            self._own()
            if text:
                self.data.attrs["_data_label"] = text
            else:
                self.data.attrs.pop("_data_label", None)
            return True
        if _abbrev(sub, "define", 2):
            self._label_define(rest)
            return True
        if _abbrev(sub, "values", 3):
            self._label_values(rest)
            return True
        if sub == "drop":
            names = rest.split()
            if names == ["_all"]:
                names = list(self._label_sets)
            for name in names:
                self._label_sets.pop(name, None)
            self._sync_value_labels([v for v, n in self._set_of.items() if n in names])
            return True
        return False

    def _label_define(self, rest: str) -> None:
        body, _, opts = _split_options(rest)
        options = set(opts.replace(",", " ").split())
        unknown = options - {"add", "modify", "replace", "nofix"}
        if unknown:
            raise StataExprError(
                f"label define: option(s) {sorted(unknown)} are not implemented"
            )
        name, _, pairs = body.strip().partition(" ")
        if not _NAME.match(name):
            raise StataExprError(f"label define: {name!r} is not a label name")
        mapping: Dict[Any, str] = {}
        pos = 0
        pairs = pairs.strip()
        while pos < len(pairs):
            m = _LABEL_PAIR.match(pairs, pos)
            if m is None:
                raise StataExprError(
                    f'label define: expected `# "text"` at {pairs[pos:]!r}'
                )
            code = m.group(1)
            key: Any = code if code.startswith(".") else int(code)
            mapping[key] = _unquote(m.group(2))
            pos = m.end()
        exists = name in self._label_sets
        if exists and not options & {"add", "modify", "replace"}:
            raise StataExprError(f"label define: label {name} already defined")
        if "replace" in options or not exists:
            self._label_sets[name] = mapping
        else:
            current = self._label_sets[name]
            if "modify" not in options:
                clash = [k for k in mapping if k in current]
                if clash:
                    raise StataExprError(
                        f"label define, add: {name} already labels {clash}; "
                        "use `modify`"
                    )
            current.update(mapping)
        self._sync_value_labels([v for v, n in self._set_of.items() if n == name])

    def _label_values(self, rest: str) -> None:
        body, _, opts = _split_options(rest)
        if opts.strip() and opts.strip() != "nofix":
            raise StataExprError(
                f"label values: option(s) {opts.strip()!r} are not implemented"
            )
        words = body.split()
        if not words:
            raise StataExprError("label values needs a variable")
        # the last word is the label name, or '.' to detach; a line whose
        # words are all variables detaches too
        if words[-1] == ".":
            variables, name = words[:-1], None
        elif len(words) > 1 and words[-1] not in self.data.columns:
            variables, name = words[:-1], words[-1]
        elif len(words) > 1 and words[-1] in self._label_sets:
            variables, name = words[:-1], words[-1]
        else:
            variables, name = words, None
        unknown = [v for v in variables if v not in self.data.columns]
        if unknown:
            raise StataExprError(
                f"label values: variable(s) {unknown} are not in the data"
            )
        for var in variables:
            if name is None:
                self._set_of.pop(var, None)
            else:
                self._set_of[var] = name
        self._sync_value_labels(variables)

    def add_column(
        self, name: str, values: np.ndarray, *, double: bool, refresh: bool = False
    ) -> None:
        """Store a computed variable (``predict``), as ``generate`` would.

        ``refresh`` lets a column this session computed be computed again
        (a lag of a variable that has changed since).
        """
        if name in self.data.columns and not refresh:
            raise StataExprError(f"variable {name!r} already exists")
        values = np.asarray(values, dtype=float)
        if not double:
            with np.errstate(over="ignore"):
                values = values.astype(np.float32).astype(np.float64)
            values = np.where(np.isfinite(values), values, np.nan)
        self._own()
        self.data[name] = values
        if not double:
            self._float.add(name)

    def _keep_drop(
        self,
        keep: bool,
        varlist: List[str],
        if_cond: Optional[str],
        in_range: Optional[str],
    ) -> None:
        if if_cond or in_range:
            if varlist:
                raise StataExprError(
                    "`keep` / `drop` take a variable list or an if / in "
                    "qualifier, not both"
                )
            mask = row_mask(self.data, if_cond, in_range, self.stored)
            self._own()
            self.data = self.data.loc[mask if keep else ~mask].reset_index(drop=True)
            return
        if not varlist:
            raise StataExprError("`keep` / `drop` need a variable list or a qualifier")
        if varlist == ["_all"]:
            if keep:
                return
            self.reset(0)
            return
        varlist = self._expand_varlist(varlist)
        unknown = [v for v in varlist if v not in self.data.columns]
        if unknown:
            raise StataExprError(f"variable(s) {unknown} are not in the data")
        self._own()
        if keep:
            order = [c for c in self.data.columns if c in set(varlist)]
            self.data = self.data[order].copy()
        else:
            self.data = self.data.drop(columns=varlist)
        self._float &= set(self.data.columns)

    def expand_varlist(self, varlist: List[str]) -> List[str]:
        """A Stata varlist written out, every name checked against the data."""
        out = self._expand_varlist(varlist)
        unknown = [v for v in out if v not in self.data.columns]
        if unknown or not out:
            raise StataExprError(f"variable(s) {unknown} are not in the data")
        return out

    def _expand_varlist(self, varlist: List[str]) -> List[str]:
        """Stata varlist ranges (``a-b``: every column from ``a`` to ``b`` in
        dataset order), wildcards (``x*``, ``x?``), ``_all`` and unambiguous
        abbreviations written out."""
        return expand_varlist([str(c) for c in self.data.columns], varlist)

    def _sort(self, varlist: List[str], qualifier: Optional[str]) -> None:
        if qualifier:
            raise StataExprError("`sort` with if / in is not implemented")
        if not varlist or any(not _NAME.match(v) for v in varlist):
            raise StataExprError("`sort` needs a list of variable names")
        unknown = [v for v in varlist if v not in self.data.columns]
        if unknown:
            raise StataExprError(f"variable(s) {unknown} are not in the data")
        self._own()
        # missing values sort last, as in Stata; ties keep their order
        self.data = self.data.sort_values(
            varlist, kind="stable", na_position="last"
        ).reset_index(drop=True)

    def _gsort(self, varlist: List[str], qualifier: Optional[str]) -> None:
        """``gsort [+|-]var ...``: descending on the variables marked ``-``.

        A missing value is the largest number, so it comes first in a
        descending key and last in an ascending one.
        """
        if qualifier:
            raise StataExprError("`gsort` with if / in is not implemented")
        keys: List[str] = []
        ascending: List[bool] = []
        for tok in varlist:
            keys.append(tok.lstrip("+-"))
            ascending.append(not tok.startswith("-"))
        if not keys or any(not _NAME.match(k) for k in keys):
            raise StataExprError("`gsort` needs a list of [+|-]variable names")
        unknown = [k for k in keys if k not in self.data.columns]
        if unknown:
            raise StataExprError(f"variable(s) {unknown} are not in the data")
        self._own()
        frame = self.data
        # sort by the last key first; a stable sort keeps the earlier keys' order
        for key, asc in reversed(list(zip(keys, ascending))):
            frame = frame.sort_values(
                key,
                ascending=asc,
                kind="stable",
                na_position="last" if asc else "first",
            )
        self.data = frame.reset_index(drop=True)

    def _rename(self, varlist: List[str], qualifier: Optional[str]) -> None:
        """``rename old new`` and ``rename (old1 old2) (new1 new2)``."""
        text = " ".join(varlist)
        grouped = re.fullmatch(r"\(([^)]*)\)\s*\(([^)]*)\)", text.strip())
        if grouped:
            olds = self.expand_varlist(grouped.group(1).split())
            news = grouped.group(2).split()
        else:
            olds, news = varlist[:1], varlist[1:]
        if qualifier or len(olds) != len(news) or not olds:
            raise StataExprError(
                "only `rename old new` and `rename (olds) (news)` are implemented"
            )
        self.rename_columns(dict(zip(olds, news)))

    def rename_columns(self, mapping: Dict[str, str]) -> None:
        """Rename variables; labels, formats and notes go with them."""
        for old, new in mapping.items():
            if old not in self.data.columns:
                raise StataExprError(f"rename: variable {old!r} is not in the data")
            if not _NAME.match(new):
                raise StataExprError(f"rename: {new!r} is not a variable name")
        kept = set(self.data.columns) - set(mapping)
        clash = [n for n in mapping.values() if n in kept]
        if clash or len(set(mapping.values())) != len(mapping):
            raise StataExprError(f"rename: variable {clash[:1]} already exists")
        mapping = {old: new for old, new in mapping.items() if old != new}
        if not mapping:
            return
        self._own()
        attrs = dict(self.data.attrs)
        self.data = self.data.rename(columns=mapping)
        for key in (
            "_labels",
            "_value_labels",
            "_missing_labels",
            "_formats",
            "_value_label_names",
            "_notes",
            "_characteristics",
        ):
            store = attrs.get(key)
            if isinstance(store, dict) and any(old in store for old in mapping):
                self.data.attrs[key] = {mapping.get(k, k): v for k, v in store.items()}
        self._set_of = {mapping.get(k, k): v for k, v in self._set_of.items()}
        self._float = {mapping.get(k, k) for k in self._float}
        coded = self.coded
        for old, new in mapping.items():
            if old in coded:
                coded.discard(old)
                coded.add(new)

    def tabulate_generate(self, var: str, stub: str, mask: np.ndarray) -> List[str]:
        """``tabulate var, generate(stub)``: one indicator per value.

        ``stub1`` marks the smallest value, ``stub2`` the next, and so on;
        an indicator is missing where ``var`` is missing or the row is
        outside the ``if`` sample.
        """
        if var not in self.data.columns:
            raise StataExprError(f"tabulate: variable {var!r} is not in the data")
        if not _NAME.match(stub):
            raise StataExprError(f"tabulate: generate({stub}) is not a name stub")
        col = self.data[var]
        used = mask & col.notna().to_numpy()
        levels = sorted(pd.unique(col[used]))
        names = [f"{stub}{k + 1}" for k in range(len(levels))]
        taken = [n for n in names if n in self.data.columns]
        if taken:
            raise StataExprError(f"tabulate: variable {taken[0]!r} already exists")
        self._own()
        new = {
            name: np.where(used, (col == level).to_numpy().astype(float), np.nan)
            for name, level in zip(names, levels)
        }
        self.data = pd.concat(
            [self.data, pd.DataFrame(new, index=self.data.index)], axis=1
        )
        return names

    def _collapse(
        self,
        varlist: List[str],
        options: dict,
        if_cond: Optional[str],
        in_range: Optional[str],
    ) -> None:
        """``collapse (stat) [new=]var ... , by(varlist)``."""
        by = str(options.pop("by", "") or "").split()
        if options:
            raise StataExprError(
                f"collapse: option(s) {sorted(options)} are not implemented"
            )
        weight = None
        text = " ".join(varlist)
        clause = re.search(r"\[\s*([a-z]+)\s*=\s*([^\]]+)\]", text, re.I)
        if clause:
            values = evaluate(clause.group(2), self.data, self.stored)
            if values.dtype == object:
                raise StataExprError("collapse: a weight must be numeric")
            weight = (clause.group(1).lower()[:2], np.asarray(values, dtype=float))
            varlist = (text[: clause.start()] + " " + text[clause.end() :]).split()
        stat = "mean"
        targets: List[tuple] = []
        # `s = x` and `s=x` name the result the same way
        for tok in re.sub(r"\s*=\s*", "=", " ".join(varlist)).split():
            m = re.fullmatch(r"\((\w+)\)", tok)
            if m:
                stat = m.group(1).lower()
                if stat not in _COLLAPSE:
                    raise StataExprError(
                        f"collapse: statistic ({stat}) is not implemented"
                    )
                continue
            new, _, source = tok.rpartition("=")
            new = new or source
            if not _NAME.match(new) or source not in self.data.columns:
                raise StataExprError(f"collapse: cannot read {tok!r}")
            if weight is not None and stat == "rawsum":
                raise StataExprError(
                    "collapse: (rawsum) with weights is not implemented"
                )
            targets.append((new, source, _COLLAPSE[stat]))
        unknown = [b for b in by if b not in self.data.columns]
        if unknown or not targets:
            raise StataExprError(
                f"collapse: variable(s) {unknown} are not in the data"
                if unknown
                else "collapse needs at least one variable"
            )
        names = [t[0] for t in targets]
        if len(set(names)) != len(names) or set(names) & set(by):
            raise StataExprError("collapse: a result name is used twice")
        keep = row_mask(self.data, if_cond, in_range, self.stored)
        if weight is not None:
            out = self._collapse_weighted(targets, by, keep, weight)
            self._own()
            self.data = out
            self._float = set()
            return
        frame = self.data.loc[keep]
        if by:
            # the rows with a missing by-value are a group too, listed last
            self._missing_by_groups(by, frame)
            grouped = frame.groupby(by, sort=True, dropna=False)
            cols = {}
            for new, source, how in targets:
                if how == "sum":
                    cols[new] = grouped[source].sum(min_count=0)
                elif how.startswith("_row_"):
                    cols[new] = grouped[source].agg(_row_end(how))
                else:
                    cols[new] = getattr(grouped[source], how)()
            out = pd.DataFrame(cols).reset_index()
        else:
            out = pd.DataFrame(
                {
                    new: [
                        (
                            frame[source].sum()
                            if how == "sum"
                            else (
                                _row_end(how)(frame[source])
                                if how.startswith("_row_")
                                else (
                                    _row_end(
                                        "_row_first" if how == "first" else "_row_last"
                                    )(frame[source].dropna())
                                    if how in ("first", "last")
                                    else getattr(frame[source], how)()
                                )
                            )
                        )
                    ]
                    for new, source, how in targets
                }
            )
        # Stata stores a mean, median or sd as a float unless its source is
        # a long or a double (checked against `collapse` in Stata 18), so
        # the collapsed value carries single precision into what follows.
        single = set()
        for new, source, how in targets:
            dtype = self.data[source].dtype
            narrow = source in self._float or (
                isinstance(dtype, np.dtype)
                and (dtype.kind == "b" or (dtype.kind in "iuf" and dtype.itemsize < 4))
                or dtype == np.float32
            )
            if how in ("mean", "median", "std") and narrow:
                with np.errstate(over="ignore"):
                    out[new] = (
                        out[new]
                        .to_numpy(dtype=float, na_value=np.nan)
                        .astype(np.float32)
                        .astype(np.float64)
                    )
                single.add(new)
            elif how in ("min", "max", "first", "last", "_row_first", "_row_last"):
                if source in self._float:
                    single.add(new)
        self._own()
        self.data = out
        self._float = single

    def _missing_by_groups(self, by: List[str], frame: pd.DataFrame) -> None:
        """Stata keeps a group for each kind of missing by-value; with one
        kind (``.``) that is the group pandas forms. A by-variable that may
        hold several kinds cannot be grouped faithfully."""
        for name in by:
            if name in self.coded and frame[name].isna().any():
                raise StataExprError(
                    f"collapse: by({name}) may hold extended missing values "
                    "(.a-.z); Stata makes a group of each kind, and the data "
                    "keep them as one. Drop the missing rows first"
                )

    def _collapse_weighted(
        self, targets: List[tuple], by: List[str], keep: np.ndarray, weight: tuple
    ) -> pd.DataFrame:
        """``collapse ... [weight]``. With weights w on the rows where the
        variable is observed (n of them): the mean is ``sum(w x) /
        sum(w)``; ``sum`` is ``sum(w x)``, for analytic weights after
        scaling them to add up to n; ``count`` is ``sum(w)`` and n for
        analytic weights; ``sd`` uses n - 1, with n = ``sum(w)`` for
        frequency weights; the median is the weighted one."""
        from ...output.sumstats import weighted_percentile

        kind, w_all = weight
        ok = keep & ~np.isnan(w_all) & (w_all != 0)
        if np.any(w_all[ok] < 0):
            raise StataExprError("collapse: negative weights")
        frame = self.data.loc[ok].copy()
        frame["__w"] = w_all[ok]
        if by:
            self._missing_by_groups(by, frame)

        def one(part: pd.DataFrame, source: str, how: str) -> float:
            x = part[source].to_numpy(dtype=float, na_value=np.nan)
            w = part["__w"].to_numpy(dtype=float)
            held = ~np.isnan(x)
            x, w = x[held], w[held]
            n = float(x.size)
            if how == "count":
                return n if kind == "aw" else float(w.sum())
            if n == 0:
                return np.nan
            if how in ("min", "max"):
                return float(x.min() if how == "min" else x.max())
            total = float(w.sum())
            mean = float(np.sum(w * x) / total)
            if how == "mean":
                return mean
            if how == "sum":
                return mean * n if kind == "aw" else float(np.sum(w * x))
            if how == "median":
                return weighted_percentile(x, w, 50.0)
            if how == "std":
                if kind in ("pw", "iw"):
                    raise StataExprError("collapse: (sd) is not allowed with pweights")
                size = total if kind == "fw" else n
                if size <= 1:
                    return np.nan
                return float(
                    np.sqrt(np.sum(w * (x - mean) ** 2) / total * size / (size - 1))
                )
            raise StataExprError(f"collapse: ({how}) with weights is not implemented")

        if by:
            rows = []
            for level, part in frame.groupby(by, sort=True, dropna=False):
                level = level if isinstance(level, tuple) else (level,)
                row = dict(zip(by, level))
                for new, source, how in targets:
                    row[new] = one(part, source, how)
                rows.append(row)
            return pd.DataFrame(rows, columns=by + [t[0] for t in targets])
        return pd.DataFrame(
            {new: [one(frame, source, how)] for new, source, how in targets}
        )

    def _ipolate(self, varlist: List[str], options: dict, qualifier: Any) -> None:
        """``ipolate y x, generate(new) [epolate]``: linear interpolation of
        ``y`` on ``x``; without ``epolate`` nothing is filled outside the
        range of the observed ``x``."""
        new = _generate_option(options)
        epolate = any(k and "epolate".startswith(k) for k in list(options))
        rest = [k for k in options if not (k and "epolate".startswith(k))]
        if qualifier or rest or new is None or len(varlist) != 2:
            raise StataExprError(
                "only `ipolate y x, generate(new) [epolate]` is implemented"
            )
        yname, xname = varlist
        unknown = [v for v in varlist if v not in self.data.columns]
        if unknown:
            raise StataExprError(f"ipolate: variable(s) {unknown} are not in the data")
        if new in self.data.columns:
            raise StataExprError(f"ipolate: variable {new!r} already exists")
        y = self.data[yname].to_numpy(dtype=float, na_value=np.nan)
        x = self.data[xname].to_numpy(dtype=float, na_value=np.nan)
        known = ~np.isnan(y) & ~np.isnan(x)
        out = np.full(len(y), np.nan)
        if known.any():
            # repeated x values enter as their mean y, as in Stata
            pts = pd.Series(y[known]).groupby(x[known]).mean()
            xs, ys = pts.index.to_numpy(dtype=float), pts.to_numpy(dtype=float)
            ok = ~np.isnan(x)
            inside = ok & (x >= xs[0]) & (x <= xs[-1])
            out[inside] = np.interp(x[inside], xs, ys)
            if epolate and len(xs) > 1:
                lo, hi = ok & (x < xs[0]), ok & (x > xs[-1])
                out[lo] = ys[0] + (x[lo] - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
                out[hi] = ys[-1] + (x[hi] - xs[-1]) * (ys[-1] - ys[-2]) / (
                    xs[-1] - xs[-2]
                )
        self.add_column(new, out, double=True)

    def reset(self, n_obs: int) -> None:
        """An empty dataset of ``n_obs`` rows (``clear``, then ``set obs``)."""
        self.data = pd.DataFrame(index=pd.RangeIndex(n_obs))
        self._owned = True
        self._float = set()

    def set_obs(self, n_obs: int) -> None:
        """``set obs #``: lengthen the data to # rows of missing values."""
        if n_obs < len(self.data):
            raise StataExprError(
                f"set obs {n_obs}: the data already hold {len(self.data)} "
                "observations and `set obs` cannot shorten them"
            )
        self._own()
        self.data = self.data.reset_index(drop=True).reindex(pd.RangeIndex(n_obs))

    def replace_data(self, data: pd.DataFrame) -> None:
        """Take ``data`` as the dataset in memory (``use``, ``append``)."""
        self.data = data.reset_index(drop=True)
        self._owned = True
        self._float &= set(self.data.columns)
        self.adopt_missing_codes(self.data)
