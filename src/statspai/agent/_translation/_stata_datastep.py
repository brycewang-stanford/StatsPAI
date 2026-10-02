"""The data-management commands ``sp.stata`` runs between estimation lines.

A do-file rarely fits a model on the data as loaded: it generates a few
variables, keeps a subsample, sorts. ``sp.stata`` runs the small set of
commands below on a private copy of the DataFrame, so that the estimation
lines that follow see the data Stata would have had:

    generate [type] newvar = exp [if] [in]
    replace var = exp [if] [in]
    keep / drop  if exp | in range | varlist
    sort varlist
    mvdecode varlist, mv(#)
    encode strvar, generate(newvar)
    preserve / restore

Expressions go through :mod:`._stata_expr` (Stata's missing-value rules).
Anything else -- ``egen``, ``merge``, ``reshape``, ``collapse``, ``by:`` --
is not a data step here, and ``sp.stata`` refuses the snippet.

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


def _is_generate(word: str) -> bool:
    return bool(word) and "generate".startswith(word)


def is_data_step(command: str) -> bool:
    return _is_generate(command) or command in (
        "replace",
        "keep",
        "drop",
        "sort",
        "preserve",
        "restore",
        "mvdecode",
        "encode",
    )


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
        elif cmd.command == "preserve":
            self._stack.append((self.data.copy(), set(self._float)))
        else:  # restore
            if not self._stack:
                raise StataExprError("`restore` without a `preserve`")
            self.data, self._float = self._stack.pop()
            self._owned = True
        return True

    # ------------------------------------------------------------ steps
    def _assign(
        self,
        replace: bool,
        body: str,
        if_cond: Optional[str],
        in_range: Optional[str],
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

        value = evaluate(expr, self.data, self.stored)
        if value.dtype == object:
            raise StataExprError("string variables are not generated here")
        mask = row_mask(self.data, if_cond, in_range, self.stored)

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
