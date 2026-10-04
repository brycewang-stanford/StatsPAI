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
Anything else -- ``egen``, ``merge``, ``reshape``, ``by:`` -- is not a data
step here, and ``sp.stata`` refuses the snippet.

Storage follows Stata: ``generate`` without a type stores a ``float`` (single
precision), so ``gen x = 0.1`` holds 0.100000001490116 and a regression on it
reproduces Stata's digits. ``generate double`` keeps full precision.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ._stata_expr import StataExprError, evaluate, in_range_mask, sample_mask
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
            "encode",
            "decode",
            "collapse",
            "ipolate",
        )
        or command in ("ren", "rena", "renam", "rename")
    )


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
    "first": "first",
    "last": "last",
}


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
        if cmd.command == "mvdecode":
            self._mvdecode(cmd.varlist, dict(cmd.options), cmd.if_cond or cmd.in_range)
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
        if vtype is not None and vtype not in _INT_TYPES + ("float", "double"):
            raise StataExprError(f"storage type {vtype!r} is not implemented")
        exists = name in self.data.columns
        if replace and not exists:
            raise StataExprError(f"`replace`: variable {name!r} does not exist")
        if not replace and exists:
            raise StataExprError(f"`generate`: variable {name!r} already exists")

        if groups is None:
            value = evaluate(expr, self.data, self.stored)
            if value.dtype == object:
                raise StataExprError("string variables are not generated here")
            mask = row_mask(self.data, if_cond, in_range, self.stored)
        else:
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
                    raise StataExprError("string variables are not generated here")
                value[rows] = got
                mask[rows] = row_mask(part, if_cond, None, self.stored)

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
                raise StataExprError(f"`replace` of non-numeric {name!r}")
            current = old.to_numpy(dtype=float, na_value=np.nan)
            self.data[name] = np.where(mask, value, current)
        else:
            self.data[name] = np.where(mask, value, np.nan)
            if single:
                self._float.add(name)

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
        if not (_is_generate(cmd.command) or cmd.command == "replace"):
            return False
        if cmd.options:
            raise StataExprError(
                f"options {sorted(cmd.options)} of `{cmd.command}` are not "
                "implemented"
            )
        unknown = [k for k in keys + order if k not in self.data.columns]
        if unknown:
            raise StataExprError(f"by: variable(s) {unknown} are not in the data")
        if sort:
            self._sort(keys + order, None)
        codes = self.data.groupby(keys, sort=False, dropna=False).ngroup().to_numpy()
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        if len(starts) != len(np.unique(codes)):
            raise StataExprError(
                "not sorted: `by` needs the data sorted by " + " ".join(keys)
            )
        bounds = np.r_[starts, len(codes)]
        groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(len(starts))]
        self._assign(
            cmd.command == "replace",
            " ".join(cmd.varlist),
            cmd.if_cond,
            cmd.in_range,
            groups=groups,
        )
        return True

    def _mvdecode(self, varlist: List[str], options: dict, qualifier: Any) -> None:
        """``mvdecode varlist, mv(#)``: turn the value # into missing."""
        raw = options.pop("mv", None)
        if qualifier or options or raw is None:
            raise StataExprError("only `mvdecode varlist, mv(#)` is implemented")
        try:
            code = float(str(raw).strip())
        except ValueError:
            raise StataExprError(
                f"mvdecode: mv({raw}) is not a single number"
            ) from None
        unknown = [v for v in varlist if v not in self.data.columns]
        if unknown or not varlist:
            raise StataExprError(f"mvdecode: variable(s) {unknown} are not in the data")
        self._own()
        for name in varlist:
            col = self.data[name].to_numpy(dtype=float, na_value=np.nan)
            self.data[name] = np.where(col == code, np.nan, col)

    def _encode(self, varlist: List[str], options: dict, qualifier: Any) -> None:
        """``encode strvar, generate(newvar)``: codes 1..K in sorted order."""
        new = options.pop("generate", None) or options.pop("gen", None)
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

    def add_column(self, name: str, values: np.ndarray, *, double: bool) -> None:
        """Store a computed variable (``predict``), as ``generate`` would."""
        if name in self.data.columns:
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
        unknown = [v for v in varlist if v not in self.data.columns]
        if unknown:
            raise StataExprError(
                f"variable(s) {unknown} are not in the data (wildcards and "
                "ranges are not expanded here)"
            )
        self._own()
        if keep:
            order = [c for c in self.data.columns if c in set(varlist)]
            self.data = self.data[order].copy()
        else:
            self.data = self.data.drop(columns=varlist)
        self._float &= set(self.data.columns)

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
        if qualifier or len(varlist) != 2:
            raise StataExprError("only `rename old new` is implemented")
        old, new = varlist
        if old not in self.data.columns:
            raise StataExprError(f"rename: variable {old!r} is not in the data")
        if not _NAME.match(new):
            raise StataExprError(f"rename: {new!r} is not a variable name")
        if new in self.data.columns:
            raise StataExprError(f"rename: variable {new!r} already exists")
        self._own()
        attrs = dict(self.data.attrs)
        self.data = self.data.rename(columns={old: new})
        # the labels go with the variable
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
            if isinstance(store, dict) and old in store:
                store = dict(store)
                store[new] = store.pop(old)
                self.data.attrs[key] = store
        if old in self._set_of:
            self._set_of[new] = self._set_of.pop(old)
        if old in self._float:
            self._float.discard(old)
            self._float.add(new)

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
        if "[" in " ".join(varlist):
            raise StataExprError("collapse with weights is not implemented")
        stat = "mean"
        targets: List[tuple] = []
        for tok in varlist:
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
        frame = self.data.loc[row_mask(self.data, if_cond, in_range, self.stored)]
        if by:
            frame = frame.dropna(subset=by)
            grouped = frame.groupby(by, sort=True, dropna=False)
            cols = {}
            for new, source, how in targets:
                if how == "sum":
                    cols[new] = grouped[source].sum(min_count=0)
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
                            else getattr(frame[source], how)()
                        )
                    ]
                    for new, source, how in targets
                }
            )
        self._own()
        self.data = out
        self._float = set()

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
