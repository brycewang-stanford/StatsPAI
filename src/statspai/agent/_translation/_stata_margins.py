"""``margins`` beyond ``margins, dydx(...)`` with one ``at()`` point.

``sp.from_stata`` translates the average marginal effect. A session knows
the data, so it can also run the forms that need the levels of a factor or
a grid of values:

* ``margins`` -- the average prediction;
* ``margins f`` / ``margins f#g`` -- predictive margins at each level;
* ``margins, at(x = (10(10)50) z = (0 1))`` and several ``at()`` options --
  predictive margins over a grid;
* ``margins f, at(...)``, ``margins, dydx(x) at(x = (...))``,
  ``margins f, dydx(x)`` -- the same for marginal effects;
* ``atmeans`` and ``over(g)``.

Each is a call of ``sp.margins_at`` (predictive margins) or ``sp.margins``
(marginal effects) per scenario; the rows are stacked in Stata's order.
"""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ._stata_datastep import _numlist
from ._stata_expr import StataExprError

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["run_margins"]

_MARGINS = re.compile(r"\s*margins\b(.*)\Z", re.S | re.I)
_OPTION = re.compile(r"\s*([A-Za-z_]\w*)\s*")


def _options(text: str) -> List[Tuple[str, Optional[str]]]:
    """``name(value) flag name(value)`` in order, parentheses balanced; an
    option may be repeated (``at() at()``), which a dict would lose."""
    out: List[Tuple[str, Optional[str]]] = []
    pos = 0
    while pos < len(text):
        m = _OPTION.match(text, pos)
        if m is None:
            if text[pos:].strip():
                raise StataExprError(f"margins: cannot read the options {text[pos:]!r}")
            break
        name, pos = m.group(1).lower(), m.end()
        if pos < len(text) and text[pos] == "(":
            depth, start = 0, pos
            while pos < len(text):
                depth += text[pos] == "("
                depth -= text[pos] == ")"
                pos += 1
                if depth == 0:
                    break
            if depth:
                raise StataExprError("margins: unbalanced parentheses")
            out.append((name, text[start + 1 : pos - 1].strip()))
        else:
            out.append((name, None))
    return out


def _at_grid(spec: str) -> Dict[str, List[float]]:
    """``x = (10(10)30) z = 1`` -> {x: [10, 20, 30], z: [1]}."""
    grid: Dict[str, List[float]] = {}
    pattern = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*(\(([^()]|\([^()]*\))*\)|[^\s()]+)")
    pos = 0
    while pos < len(spec):
        m = pattern.match(spec, pos)
        if m is None:
            raise StataExprError(
                f"margins: at({spec}) is not `var = # | (numlist)`; statistics "
                "such as (mean) are not implemented"
            )
        value = m.group(2).strip()
        if value.startswith("("):
            value = value[1:-1]
        grid[m.group(1)] = _numlist(value, "margins, at()")
        pos = m.end()
        while pos < len(spec) and spec[pos].isspace():
            pos += 1
    return grid


def _levels(data: pd.DataFrame, name: str) -> List[float]:
    if name not in data.columns:
        raise StataExprError(f"margins: variable {name!r} is not in the data")
    values = pd.to_numeric(data[name], errors="coerce").dropna().unique()
    return sorted(float(v) for v in values)


def _model_names(
    session: "StataSession", data: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """The covariates of the last fit: (continuous, factor)."""
    call = session._last_call or {}
    formula = str((call.get("arguments") or {}).get("formula") or "")
    if "~" not in formula:
        return [], []
    rhs = formula.split("~", 1)[1]
    factors = re.findall(r"C\(\s*([A-Za-z_]\w*)", rhs)
    names = [n for n in dict.fromkeys(re.findall(r"[A-Za-z_]\w*", rhs))
             if n in data.columns]  # fmt: skip
    return [n for n in names if n not in factors], list(dict.fromkeys(factors))


def _residual_df(result: Any) -> Optional[float]:
    """Residual degrees of freedom of a linear fit (None after a model
    whose inference is normal)."""
    model = getattr(result, "model_info", None) or {}
    if model.get("family") not in (None, "gaussian"):
        return None
    info = getattr(result, "data_info", None) or {}
    x = info.get("X")
    if x is None:
        return None
    n, k = np.asarray(x).shape
    return float(n - k) if n > k else None


def run_margins(session: "StataSession", line: str) -> Optional[bool]:
    """Run a ``margins`` line that needs the session; ``None`` leaves the
    line to the single-call translation."""
    m = _MARGINS.match(line)
    if m is None:
        return None
    import statspai as sp

    body, _, option_text = m.group(1).partition(",")
    options = _options(option_text)
    names = [name for name, _ in options]
    terms = body.split()
    at_specs = [value or "" for name, value in options if name == "at"]
    dydx = next((value for name, value in options if name == "dydx"), None)
    over = next((value for name, value in options if name == "over"), None)
    atmeans = "atmeans" in names
    simple_at = len(at_specs) <= 1 and all("(" not in s for s in at_specs)
    if dydx is not None and not terms and simple_at and over is None:
        return None  # the plain marginal effect: one sp.margins call
    known = {"at", "dydx", "over", "atmeans", "level", "post", "noestimcheck",
             "nose", "vsquish", "noatlegend", "cformat", "predict",
             "asobserved"}  # fmt: skip
    unknown = sorted(set(names) - known)
    if unknown:
        raise StataExprError(f"margins: option(s) {unknown} are not implemented")
    predicted = next((value for name, value in options if name == "predict"), None)
    if predicted is not None and predicted.strip() not in ("pr", "xb", ""):
        raise StataExprError(
            f"margins, predict({predicted}) is not implemented; the margin is "
            "on the model's default prediction"
        )
    result, data = session.last, session.last_data
    if result is None or data is None:
        raise StataExprError("margins follows an estimation command")
    continuous, factors = _model_names(session, data)
    # the estimation sample: the rows complete on the model's variables
    call = session._last_call or {}
    formula = str((call.get("arguments") or {}).get("formula") or "")
    used = [n for n in dict.fromkeys(re.findall(r"[A-Za-z_]\w*", formula))
            if n in data.columns]  # fmt: skip
    if used:
        data = data.loc[data[used].notna().all(axis=1)]
    level = next((value for name, value in options if name == "level"), None)
    alpha = 1 - float(level) / 100 if level else 0.05
    # the factor terms: `f`, `i.f`, `f#g`; each term is its own table
    tables: List[List[str]] = []
    for term in terms:
        parts = [re.sub(r"^i[bn]?\d*\.", "", p) for p in term.split("#")]
        tables.append(parts)
    if not tables:
        tables = [[]]
    scenarios = [_at_grid(spec) for spec in at_specs] or [{}]
    means: Dict[str, List[float]] = {}
    if atmeans:
        if factors:
            raise StataExprError(
                "margins, atmeans with factor covariates sets each indicator "
                "to its mean, which is not implemented"
            )
        for name in continuous:
            means[name] = [float(pd.to_numeric(data[name], errors="coerce").mean())]
    groups: List[Tuple[Any, pd.DataFrame]] = [(None, data)]
    if over:
        keys = over.split()
        groups = [
            (key, sub) for key, sub in data.groupby(keys if len(keys) > 1 else keys[0])
        ]
    pieces = []
    for parts in tables:
        factor_grid = {name: _levels(data, name) for name in parts}
        for scenario in scenarios:
            grid = {**means, **scenario, **factor_grid}
            if dydx is None:
                for key, sub in groups:
                    table = sp.margins_at(result, data=sub, at=grid, alpha=alpha)
                    # the test statistic and its p-value, as the table prints
                    table["z"] = table["margin"] / table["se"]
                    dof = _residual_df(result)
                    table["pvalue"] = (
                        2 * stats.norm.sf(np.abs(table["z"])) if dof is None
                        else 2 * stats.t.sf(np.abs(table["z"]), dof)
                    )  # fmt: skip
                    if over:
                        table.insert(0, "over", [key] * len(table))
                    pieces.append(table)
                continue
            variables = None if set(dydx.split()) & {"*", "_all"} else dydx.split()
            order = list(scenario) + list(parts)
            points = list(itertools.product(*[grid[k] for k in order])) or [()]
            fixed = {k: v[0] for k, v in means.items() if k not in order}
            for point in points:
                at = {**fixed, **dict(zip(order, point))}
                for key, sub in groups:
                    table = sp.margins(
                        result, data=sub, variables=variables, at=at or None,
                        method="mem" if atmeans and not at else "ame", alpha=alpha,
                    )  # fmt: skip
                    table = table.copy()
                    for name, value in zip(order, point):
                        table[name] = value
                    if over:
                        table["over"] = [key] * len(table)
                    pieces.append(table)
    out = pd.concat(pieces, ignore_index=True)
    out.attrs["session_margins"] = True
    session.output = out
    session.stored["r"] = {"N": float(len(data))}
    return True
