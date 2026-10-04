"""Data steps that build variables for matching and weighting.

``xi i.g, prefix(stub)`` (indicator variables), ``ebalance`` (entropy
balancing weights) and ``cem`` (coarsened exact matching strata and
weights) do not estimate anything by themselves: they add columns that a
later ``regress ... [pw = wt]`` uses. They are run by ``sp.stata`` only,
which holds the data; ``sp.from_stata`` translates a single line into a
call and has nowhere to put new columns.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._stata_expr import StataExprError

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_datastep import DataSteps

__all__ = ["BALANCE_COMMANDS", "run_balance"]

BALANCE_COMMANDS = ("xi", "ebalance", "cem")

_NAME = re.compile(r"[A-Za-z_]\w*\Z")


def _pop_abbreviated(options: Dict[str, Any], full: str, shortest: int) -> Any:
    for key in list(options):
        if key and len(key) >= shortest and full.startswith(key):
            return options.pop(key)
    return None


def _numeric(steps: "DataSteps", name: str, command: str) -> np.ndarray:
    if name not in steps.data.columns:
        raise StataExprError(f"{command}: variable {name!r} is not in the data")
    col = steps.data[name]
    if not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)):
        raise StataExprError(f"{command}: {name!r} is not numeric")
    return col.to_numpy(dtype=float, na_value=np.nan)


def _store(steps: "DataSteps", name: str, values: np.ndarray, replace: bool) -> None:
    if name in steps.data.columns:
        if not replace:
            raise StataExprError(f"variable {name!r} already exists")
        steps._own()
        steps.data = steps.data.drop(columns=[name])
    steps.add_column(name, values, double=True)


# ------------------------------------------------------------------- xi
def _xi(steps: "DataSteps", varlist: List[str], options: Dict[str, Any]) -> None:
    """``xi i.g [i.h ...], prefix(stub)``: one indicator per level of ``g``
    but the smallest, named ``<stub><g>_<level>``.

    Stata keeps ``stub`` plus the first ``11 - len(stub)`` characters of the
    variable name (``_I`` and nine characters by default), then ``_`` and
    the level; a string variable is numbered 1, 2, ... in sorted order.
    """
    stub = _pop_abbreviated(options, "prefix", 3)
    stub = "_I" if stub is None else str(stub).strip()
    if options:
        raise StataExprError(f"xi: option(s) {sorted(options)} are not implemented")
    if not _NAME.match(stub) or len(stub) > 4:
        raise StataExprError("xi: prefix() must be a name of at most four characters")
    terms = [re.fullmatch(r"i\.([A-Za-z_]\w*)", tok) for tok in varlist]
    if not terms or None in terms:
        raise StataExprError(
            "only `xi i.var [i.var ...] [, prefix(stub)]` is implemented "
            "(no interactions)"
        )
    for m in terms:
        source = m.group(1)  # type: ignore[union-attr]
        if source not in steps.data.columns:
            raise StataExprError(f"xi: variable {source!r} is not in the data")
        col = steps.data[source]
        numeric = pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)
        if numeric:
            values = col.to_numpy(dtype=float, na_value=np.nan)
            present = ~np.isnan(values)
            levels = np.unique(values[present])
            if not np.all(levels == np.round(levels)) or np.any(levels < 0):
                raise StataExprError(
                    f"xi: {source!r} holds values that are not nonnegative " "integers"
                )
            labels = [f"{int(v)}" for v in levels]
        else:
            text = col.astype(object)
            present = (col.notna() & (col.astype(str) != "")).to_numpy()
            names = sorted(set(text[present].astype(str)))
            codes = {name: i + 1.0 for i, name in enumerate(names)}
            values = np.array(
                [codes[str(v)] if ok else np.nan for v, ok in zip(text, present)]
            )
            levels = np.arange(1.0, len(names) + 1.0)
            labels = [f"{int(v)}" for v in levels]
        head = stub + source[: max(11 - len(stub), 1)]
        for level, label in list(zip(levels, labels))[1:]:
            name = f"{head}_{label}"
            _store(steps, name, np.where(present, values == level, np.nan), True)
            steps._float.discard(name)


# ------------------------------------------------------------- ebalance
def _ebalance(steps: "DataSteps", varlist: List[str], options: Dict[str, Any]) -> None:
    """``ebalance treat x1 x2 ..., targets(numlist) generate(wt)``.

    The weight is 1 for a treated unit and, for the controls, the entropy
    balancing weight scaled to sum to the number of treated units, as
    Stata stores it. The moments are matched in Stata's scaling (sample
    variance and skewness, ``dof_adjust=True`` of ``sp.ebalance``).

    Stata stops at ``tolerance(.015)`` by default, before the moments are
    balanced; the weights here are solved to machine precision, which is
    what ``ebalance ..., tolerance(1e-10)`` converges to. Estimates that use
    the default-tolerance weights agree to about three digits.
    """
    from ...matching.ebalance import ebalance_weights
    from ._stata_datastep import _generate_option

    new = _generate_option(options)
    targets = _pop_abbreviated(options, "targets", 3)
    for accepted in ("tolerance", "maxiter"):
        _pop_abbreviated(options, accepted, 3)
    replace = any(k == "replace" for k in list(options))
    options.pop("replace", None)
    if options:
        raise StataExprError(
            f"ebalance: option(s) {sorted(options)} are not implemented"
        )
    if new is None or not _NAME.match(new) or len(varlist) < 2:
        raise StataExprError(
            "only `ebalance treat covariates, targets(numlist) "
            "generate(newvar)` is implemented"
        )
    treat, covariates = varlist[0], list(varlist[1:])
    try:
        orders = [int(t) for t in str(targets or "1").split()]
    except ValueError:
        raise StataExprError(f"ebalance: targets({targets}) is not a list of 1/2/3")
    if len(orders) == 1:
        orders = orders * len(covariates)
    columns = {name: _numeric(steps, name, "ebalance") for name in [treat] + covariates}
    frame = pd.DataFrame(columns)
    used = frame.notna().all(axis=1).to_numpy()
    try:
        weights = ebalance_weights(
            frame.loc[used], treat, covariates, moments=orders, dof_adjust=True
        )
    except Exception as exc:
        raise StataExprError(f"ebalance: {exc}") from exc
    out = np.full(len(frame), np.nan)
    out[used] = weights
    _store(steps, new, out, replace)


# ------------------------------------------------------------------ cem
def _cem_cut(x: np.ndarray, spec: Optional[str], n: int, name: str) -> np.ndarray:
    """Coarsen one variable the way ``cem`` for Stata does.

    ``(#k)`` asks for ``k`` equally spaced cut *points* from the minimum to
    the maximum, hence ``k - 1`` intervals, each closed on the right (the
    first also on the left); the default is Sturges' ``ceil(log2(n) + 1)``
    points. A list of numbers gives the points themselves; a single number
    is one cut between the minimum and itself. ``(#0)`` matches exactly.
    """
    if spec is None:
        points = int(np.ceil(np.log2(n) + 1))
    elif re.fullmatch(r"#\d+", spec):
        points = int(spec[1:])
    else:
        try:
            edges = np.array([float(t) for t in spec.split()])
        except ValueError:
            raise StataExprError(
                f"cem: the cutpoints ({spec}) of {name!r} are not read; use "
                "(#k) or a list of numbers"
            )
        if edges.size == 1:
            edges = np.array([np.min(x), edges[0]])
        return _cem_cells(x, edges)
    if points == 0:
        return x.copy()
    return _cem_cells(x, np.linspace(np.min(x), np.max(x), points))


def _cem_cells(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    cell = (x >= edges[0]).astype(float)
    for edge in edges[1:]:
        cell += x > edge
    return cell


def _cem(steps: "DataSteps", varlist: List[str], options: Dict[str, Any]) -> None:
    """``cem x1 x2(#k) x3(c1 c2 ...), treatment(d)``: the variables
    ``cem_strata``, ``cem_matched`` and ``cem_weights``.

    A stratum is matched when it holds both treated and control units. The
    weight is 1 for a matched treated unit, ``(treated in stratum /
    controls in stratum) * (matched controls / matched treated)`` for a
    matched control and 0 for an unmatched unit. Strata are numbered in
    the sorted order of their cells, which need not be Stata's numbering.
    """
    treat = _pop_abbreviated(options, "treatment", 2)
    if options:
        raise StataExprError(f"cem: option(s) {sorted(options)} are not implemented")
    if treat is None or not varlist:
        raise StataExprError("only `cem varlist, treatment(var)` is implemented")
    specs: List[Tuple[str, Optional[str]]] = []
    # the cut points of `x(0 10 20)` hold blanks, so read the list as a whole
    text = " ".join(varlist).strip()
    pos = 0
    for m in re.finditer(r"\s*([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?", text):
        if m.start() != pos or m.end() == m.start():
            break
        specs.append((m.group(1), None if m.group(2) is None else m.group(2).strip()))
        pos = m.end()
    if pos != len(text) or not specs:
        raise StataExprError(f"cem: cannot read {text[pos:pos + 20]!r}")
    d = _numeric(steps, str(treat).strip(), "cem")
    columns = [_numeric(steps, name, "cem") for name, _ in specs]
    used = ~np.isnan(d) & ~np.isnan(np.column_stack(columns)).any(axis=1)
    groups = np.unique(d[used])
    if groups.size != 2:
        raise StataExprError("cem: treatment() must take two values")
    n = int(used.sum())
    cells = np.column_stack(
        [
            _cem_cut(col[used], spec, n, name)
            for col, (name, spec) in zip(columns, specs)
        ]
    )
    _, strata = np.unique(cells, axis=0, return_inverse=True)
    strata = np.asarray(strata).ravel()
    treated = d[used] == groups[1]
    n_strata = int(strata.max()) + 1
    n_t = np.bincount(strata, weights=treated, minlength=n_strata)
    n_c = np.bincount(strata, weights=~treated, minlength=n_strata)
    matched = ((n_t > 0) & (n_c > 0))[strata]
    total_t = float(n_t[(n_t > 0) & (n_c > 0)].sum())
    total_c = float(n_c[(n_t > 0) & (n_c > 0)].sum())
    weight = np.zeros(n)
    if total_t > 0:
        ratio = np.divide(n_t, n_c, out=np.zeros(n_strata), where=n_c > 0)
        weight = np.where(treated, 1.0, ratio[strata] * total_c / total_t) * matched
    for name, values in (
        ("cem_strata", strata + 1.0),
        ("cem_matched", matched.astype(float)),
        ("cem_weights", weight),
    ):
        out = np.full(len(d), np.nan)
        out[used] = values
        _store(steps, name, out, True)


def run_balance(
    steps: "DataSteps", command: str, varlist: List[str], options: Dict[str, Any]
) -> None:
    if command == "xi":
        _xi(steps, varlist, options)
    elif command == "ebalance":
        _ebalance(steps, varlist, options)
    else:
        _cem(steps, varlist, options)
