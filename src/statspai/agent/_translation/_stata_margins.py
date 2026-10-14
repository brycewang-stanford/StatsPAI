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

Here too run ``predict(xb)`` (the linear index instead of the model's
default prediction), any ``margins`` after a weighted fit (averaged with
the fit's weights, which the estimation sample alone does not carry), and
``post``: the margins replace the model in ``e()`` under the names Stata
gives them (``x1 1b.g 2.g``, ``1.g 2.g``, ``1._at 2._at``, ``_cons``), with
their covariance, so that ``_b[]``, ``_se[]``, ``e(b)``, ``e(V)``, ``test``
and ``lincom`` read the margins. ``post`` after a table whose names are
not implemented (``over()``, several terms, ``at()`` with a term) is
refused.
"""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ...postestimation._covariance import coefficient_covariance, inference_df
from ...postestimation._design import model_setting
from ._stata_datastep import _numlist
from ._stata_expr import StataExprError

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["run_margins", "PostedMargins"]

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
    if isinstance(session.last, PostedMargins):
        raise StataExprError(
            "margins after `margins, post`: the estimates in memory are "
            "margins, not a model (restore the model with `estimates restore`)"
        )
    body, _, option_text = m.group(1).partition(",")
    options = _options(option_text)
    names = [name for name, _ in options]
    terms = body.split()
    at_specs = [value or "" for name, value in options if name == "at"]
    dydx = next((value for name, value in options if name == "dydx"), None)
    over = next((value for name, value in options if name == "over"), None)
    atmeans = "atmeans" in names
    simple_at = len(at_specs) <= 1 and all("(" not in s for s in at_specs)
    post = "post" in names
    predicted = next((value for name, value in options if name == "predict"), None)
    plain = dydx is not None and not terms and simple_at and over is None
    if plain and not post and predicted is None and not _weighted(session.last):
        return None  # the plain marginal effect: one sp.margins call
    known = {"at", "dydx", "over", "atmeans", "level", "post", "noestimcheck",
             "nose", "vsquish", "noatlegend", "cformat", "predict",
             "asobserved"}  # fmt: skip
    unknown = sorted(set(names) - known)
    if unknown:
        raise StataExprError(f"margins: option(s) {unknown} are not implemented")
    result, data = session.last, session.last_data
    if result is None or data is None:
        raise StataExprError("margins follows an estimation command")
    result = _on_scale(result, predicted)
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
    if atmeans and not plain:
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
    shape = _posted_shape(tables, at_specs, dydx, over) if post else ""

    def compute(fit: Any) -> pd.DataFrame:
        if plain:
            # one sp.margins call, as `sp.from_stata` translates the line
            import statspai as sp

            assert dydx is not None
            wanted = None if set(dydx.split()) & {"*", "_all"} else dydx.split()
            at = {k: v[0] for k, v in scenarios[0].items()}
            return sp.margins(
                fit, data=data, variables=wanted, at=at or None,
                method="mem" if atmeans else "ame", alpha=alpha,
            )  # fmt: skip
        return _compute(fit, data, tables, scenarios, means, groups, dydx,
                        over, atmeans, alpha)  # fmt: skip

    out = compute(result)
    out.attrs["session_margins"] = True
    session.output = out
    session.stored["r"] = {"N": float(len(data))}
    if post:
        _post(session, result, out, shape, compute, data)
    return True


def _compute(
    result: Any,
    data: pd.DataFrame,
    tables: List[List[str]],
    scenarios: List[Dict[str, List[float]]],
    means: Dict[str, List[float]],
    groups: List[Tuple[Any, pd.DataFrame]],
    dydx: Optional[str],
    over: Optional[str],
    atmeans: bool,
    alpha: float,
) -> pd.DataFrame:
    """The margins table of ``result`` (one fit; see :func:`run_margins`)."""
    import statspai as sp

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
    out.attrs.update(pieces[0].attrs)
    return out


# ------------------------------------------------------------ predict(), post
class _View:
    """``base`` with some attributes replaced (the parameters at a nudged
    point, or the linear index as the prediction)."""

    def __init__(self, base: Any, **replaced: Any) -> None:
        self._base = base
        self.__dict__.update(replaced)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _weighted(result: Any) -> bool:
    try:
        return result is not None and model_setting(result, "weights") is not None
    except Exception:  # noqa: BLE001 - a result without the stores
        return False


def _on_scale(result: Any, predicted: Optional[str]) -> Any:
    """The fit as ``margins, predict()`` asks for it. ``predict(xb)`` is the
    linear index (with any offset), whatever the model's default; the
    default prediction (``pr`` after a binary model) is the fit itself."""
    if predicted is None:
        return result
    what = predicted.strip().lower()
    model = dict(getattr(result, "model_info", None) or {})
    data = dict(getattr(result, "data_info", None) or {})
    link = str(model.get("link") or getattr(data.get("link_obj"), "name", "") or "")
    binary = link.lower() in ("logit", "probit", "cloglog")
    if what == "" or (what == "pr" and binary):
        return result
    if what == "xb":
        if not link or link.lower() == "identity":
            return result
        model["link"] = "identity"
        data.pop("link_obj", None)
        return _View(result, model_info=model, data_info=data)
    raise StataExprError(
        f"margins, predict({predicted}) is not implemented here; only "
        "predict(xb) and the model's default prediction are"
    )


class PostedMargins:
    """What ``margins, post`` leaves in ``e()``: the margins as the
    coefficients (named as Stata names them) with their covariance, for
    ``_b[]``, ``_se[]``, ``e(b)``, ``e(V)``, ``test`` and ``lincom``."""

    def __init__(
        self, names: List[str], b: np.ndarray, V: np.ndarray, n: int, df: Any
    ) -> None:
        self.params = pd.Series(b, index=names, dtype=float)
        self.std_errors = pd.Series(np.sqrt(np.clip(np.diag(V), 0.0, None)),
                                    index=names)  # fmt: skip
        self.vcov = pd.DataFrame(V, index=names, columns=names)
        self.nobs = n
        self.data_info: Dict[str, Any] = {"var_cov": V, "nobs": n}
        if df is None or not np.isfinite(df):
            self.data_info["inference"] = "z"
        else:
            self.data_info["df_resid"] = float(df)
        self.model_info: Dict[str, Any] = {"stata_cmd": "margins"}

    def __repr__(self) -> str:
        return f"<PostedMargins: {', '.join(self.params.index)}>"


def _posted_shape(
    tables: List[List[str]], at_specs: List[str], dydx: Optional[str], over: Any
) -> str:
    """Which of the shapes ``post`` is implemented for, by its names."""
    if over is None:
        if dydx is not None and tables == [[]] and not at_specs:
            return "dydx"  # x1, 1b.g, 2.g ...
        if dydx is None and tables == [[]]:
            return "at" if at_specs else "cons"  # 1._at ... / _cons
        if dydx is None and len(tables) == 1 and len(tables[0]) == 1:
            if not at_specs:
                return "factor"  # 1.g 2.g ...
    raise StataExprError(
        "margins, post is implemented for `margins, dydx() post`, "
        "`margins f, post`, `margins, at() post` and `margins, post`; Stata's "
        "names for the other tables (over(), several terms, at() with a "
        "term or with dydx()) are not"
    )


def _level(value: Any) -> str:
    v = float(value)
    if v != round(v) or v < 0:
        raise StataExprError(f"margins, post: {value!r} is not a factor level")
    return str(int(v))


def _post(
    session: "StataSession",
    result: Any,
    out: pd.DataFrame,
    shape: str,
    compute: Any,
    data: pd.DataFrame,
) -> None:
    """``margins, post``: the margins replace the model in ``e()``."""
    column = "dy/dx" if "dy/dx" in out.columns else "margin"
    b = out[column].to_numpy(dtype=float)
    se = out["se"].to_numpy(dtype=float)
    if shape == "cons":
        names = ["_cons"]
    elif shape == "at":
        names = [f"{k + 1}._at" for k in range(len(b))]
    elif shape == "factor":
        level_col = [c for c in out.columns if c not in _TABLE_COLUMNS][0]
        names = [f"{_level(v)}.{level_col}" for v in out[level_col]]
    else:
        names = [str(v) for v in out["variable"]]
    # the delta method across the margins: numeric derivatives of the
    # margins in the coefficients (Stata's margins differentiates
    # numerically too), scaled to the standard errors of the table
    V0, _ = coefficient_covariance(result)
    if V0 is None:
        raise StataExprError(
            "margins, post needs the full covariance matrix of the fit, "
            "which this result does not carry"
        )
    params = pd.Series(result.params).astype(float)
    jac = np.empty((len(b), len(params)))
    for j in range(len(params)):
        h = 1e-5 * max(1.0, abs(float(params.iloc[j])))
        up, down = params.copy(), params.copy()
        up.iloc[j] += h
        down.iloc[j] -= h
        hi = compute(_View(result, params=up))[column].to_numpy(dtype=float)
        lo = compute(_View(result, params=down))[column].to_numpy(dtype=float)
        jac[:, j] = (hi - lo) / (2 * h)
    V = jac @ V0 @ jac.T
    scale = np.sqrt(np.clip(np.diag(V), 0.0, None))
    ratio = np.divide(se, scale, out=np.zeros_like(se), where=scale > 0)
    V = V * np.outer(ratio, ratio)
    if shape == "dydx":
        # in the order the variables enter the model, whatever the order
        # of dydx() (Stata 18: `dydx(g x1)` posts x1 1b.g 2.g 3.g)
        formula = str(((session._last_call or {}).get("arguments") or {})
                      .get("formula") or "")  # fmt: skip
        rhs = re.findall(r"[A-Za-z_]\w*", formula.partition("~")[2])

        def place(name: str) -> int:
            var = re.sub(r"^-?\d+b?\.", "", name)
            return rhs.index(var) if var in rhs else len(rhs)

        order = sorted(range(len(names)), key=lambda k: place(names[k]))
        names = [names[k] for k in order]
        b, V = b[order], V[np.ix_(order, order)]
        # Stata keeps the base level of a factor: 0, with variance 0
        for var, base in (out.attrs.get("base_levels") or {}).items():
            first = next(k for k, n in enumerate(names) if n.endswith(f".{var}"))
            names.insert(first, f"{_level(base)}b.{var}")
            b = np.insert(b, first, 0.0)
            V = np.insert(np.insert(V, first, 0.0, axis=0), first, 0.0, axis=1)
    df = inference_df(result)
    posted = PostedMargins(names, b, V, int(len(data)), df)
    session.last = posted
    session.last_data = data
    session._last_call = {"tool": "margins", "arguments": {}}
    session._store_estimates(posted)
    session.stored["e_macros"] = {"cmd": "margins", "depvar": ""}
    session.stored.pop("e_sample", None)


#: columns of a margins table that are not the levels it is over
_TABLE_COLUMNS = frozenset(
    {"margin", "se", "ci_lower", "ci_upper", "z", "pvalue", "t", "over"}
)
