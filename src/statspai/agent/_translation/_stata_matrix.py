"""Matrices in a ``sp.stata`` session.

A do-file uses a matrix as a table of results: it sizes one with ``J()``,
fills it cell by cell inside a loop, and turns it into variables with
``svmat``. That, and reading ``e(b)`` / ``e(V)``, is what is run here.

``matrix [define] A = J(r, c, v)`` / ``I(n)`` / ``e(b)`` / ``e(V)`` /
    ``(1, 2 \\ 3, 4)`` / ``B'`` / ``d * V * d'`` / ``inv(X' * X)``
``matrix [define] A[i, j] = exp``
``matrix rownames A = names`` / ``matrix colnames A = names``
``matrix list A``, ``matrix drop A | _all``
``svmat [type] A [, names(stub | col)]``
``mkmat varlist [if] [, matrix(A)]``

In an expression ``A[i, j]``, ``el(A, i, j)``, ``rowsof(A)`` and
``colsof(A)`` read a matrix. ``e(b)`` is a row vector with the constant
last, as Stata stores it. The right-hand side may be a matrix expression:
sums, products, a transpose, ``inv()``, ``diag()``, ``vecdiag()``,
``trace()``, and the old spellings ``get(_b)`` / ``get(VCE)``. Other matrix
functions (``cholesky``, ``det``, ``hadamard`` ...) and the Kronecker
product are refused.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._stata_expr import StataExprError

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["matrix_line"]

_MATRIX = re.compile(r"\s*mat(?:r(?:ix?)?)?\s+(.+?)\s*$", re.I | re.S)
_DEFINE = re.compile(
    r"(?:def(?:ine)?\s+|input\s+)?([A-Za-z_]\w*)\s*(?:\[(.+?),(.+?)\])?\s*"
    r"=(?!=)\s*(.+)\Z",
    re.I | re.S,
)
_SVMAT = re.compile(r"\s*svmat\s+(.+?)\s*$", re.I | re.S)
_MKMAT = re.compile(r"\s*mkmat\s+(.+?)\s*$", re.I | re.S)
_CONSTANT = ("Intercept", "_cons", "const")


def _table(session: "StataSession") -> Dict[str, Dict[str, Any]]:
    held: Dict[str, Dict[str, Any]] = session.stored.setdefault("matrices", {})
    return held


def _store(session: "StataSession", name: str, values: Any, rows: Any = None,
           cols: Any = None) -> None:  # fmt: skip
    values = np.atleast_2d(np.asarray(values, dtype=float))
    _table(session)[name] = {
        "values": values,
        "rows": (
            list(rows)
            if rows is not None
            else [f"r{i + 1}" for i in range(values.shape[0])]
        ),
        "cols": (
            list(cols)
            if cols is not None
            else [f"c{j + 1}" for j in range(values.shape[1])]
        ),
    }


def _constant_last(names: List[str]) -> List[int]:
    order = [i for i, n in enumerate(names) if n not in _CONSTANT]
    return order + [i for i, n in enumerate(names) if n in _CONSTANT]


def _estimates(session: "StataSession", which: str) -> Dict[str, Any]:
    result = session.last
    params = getattr(result, "params", None)
    if params is None:
        raise StataExprError(f"e({which}) needs an estimation command before it")
    names = [str(n) for n in params.index]
    order = _constant_last(names)
    shown = ["_cons" if names[i] in _CONSTANT else names[i] for i in order]
    if which == "b":
        values = np.asarray(params, dtype=float)[order][None, :]
        return {"values": values, "rows": ["y1"], "cols": shown}
    cov = getattr(result, "vcov", None)
    cov = cov() if callable(cov) else cov
    if cov is None:
        cov = getattr(result, "cov_params", None)
        cov = cov() if callable(cov) else cov
    if cov is None:
        raise StataExprError("e(V) is not available for this result")
    values = np.asarray(cov, dtype=float)[np.ix_(order, order)]
    return {"values": values, "rows": shown, "cols": shown}


def _residual_covariance(session: "StataSession") -> Dict[str, Any]:
    """``e(Sigma)`` after ``var``: the covariance of the residuals as Stata
    stores it, divided by the number of observations (no ``dfk``)."""
    result = session.last
    sigma = getattr(result, "sigma_u", None)
    resid = getattr(result, "resid", None)
    if sigma is None:
        raise StataExprError("e(Sigma) needs a var before it")
    names = [str(c) for c in getattr(sigma, "columns", range(len(sigma)))]
    if resid is not None:
        e = np.asarray(resid, dtype=float)
        values = e.T @ e / e.shape[0]
    else:
        values = np.asarray(sigma, dtype=float)
    return {"values": values, "rows": names, "cols": names}


def _mat(values: Any, rows: Any = None, cols: Any = None) -> Dict[str, Any]:
    return {"values": np.atleast_2d(np.asarray(values, dtype=float)), "rows": rows,
            "cols": cols}  # fmt: skip


def _top_level(text: str, seps: str) -> List[str]:
    """``text`` split on the separators that are outside parentheses."""
    parts, depth, start = [], 0, 0
    for k, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch in seps and depth == 0:
            parts.append(text[start:k])
            start = k + 1
    parts.append(text[start:])
    return parts


class _Algebra:
    r"""Matrix expressions: ``+ - *``, ``/`` by a scalar, the transpose
    ``'``, ``inv()`` / ``invsym()`` / ``syminv()``, ``diag()``,
    ``vecdiag()``, ``trace()``, ``J()``, ``I()``, ``e(b)``, ``e(V)``,
    ``get(_b)``, ``get(VCE)``, a literal ``(1, 2 \ 3, 4)`` and names of
    matrices and scalars. A 1 x 1 matrix acts as a scalar in a product."""

    def __init__(self, session: "StataSession", text: str) -> None:
        self.session = session
        self.text = text
        self.pos = 0

    def _skip(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str:
        self._skip()
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def parse(self) -> Dict[str, Any]:
        out = self._sum()
        if self._peek():
            raise StataExprError(
                f"the matrix expression {self.text!r} is not understood at "
                f"{self.text[self.pos:]!r}"
            )
        return out

    def _sum(self) -> Dict[str, Any]:
        left = self._product()
        while self._peek() in ("+", "-"):
            op = self.text[self.pos]
            self.pos += 1
            right = self._product()
            a, b = left["values"], right["values"]
            if a.shape != b.shape:
                raise StataExprError("conformability error")
            left = _mat(a + b if op == "+" else a - b, left["rows"], left["cols"])
        return left

    def _product(self) -> Dict[str, Any]:
        left = self._unary()
        while self._peek() in ("*", "/"):
            op = self.text[self.pos]
            self.pos += 1
            right = self._unary()
            a, b = left["values"], right["values"]
            if op == "/":
                if b.shape != (1, 1):
                    raise StataExprError("a matrix is divided by a scalar only")
                left = _mat(a / b[0, 0], left["rows"], left["cols"])
            elif a.shape == (1, 1) and b.shape != (1, 1):
                left = _mat(a[0, 0] * b, right["rows"], right["cols"])
            elif b.shape == (1, 1) and a.shape != (1, 1):
                left = _mat(a * b[0, 0], left["rows"], left["cols"])
            else:
                if a.shape[1] != b.shape[0]:
                    raise StataExprError("conformability error")
                left = _mat(a @ b, left["rows"], right["cols"])
        return left

    def _unary(self) -> Dict[str, Any]:
        if self._peek() == "-":
            self.pos += 1
            inner = self._unary()
            return _mat(-inner["values"], inner["rows"], inner["cols"])
        out = self._primary()
        while self._peek() == "'":
            self.pos += 1
            out = _mat(out["values"].T.copy(), out["cols"], out["rows"])
        return out

    def _group(self) -> str:
        """The text inside the parentheses that open at ``self.pos``."""
        depth, start = 0, self.pos
        for k in range(self.pos, len(self.text)):
            if self.text[k] == "(":
                depth += 1
            elif self.text[k] == ")":
                depth -= 1
                if depth == 0:
                    self.pos = k + 1
                    return self.text[start + 1 : k]
        raise StataExprError("unbalanced parentheses in the matrix expression")

    def _primary(self) -> Dict[str, Any]:
        ch = self._peek()
        value = self.session.value
        held = _table(self.session)
        if ch == "(":
            inner = self._group()
            rows = _top_level(inner, "\\")
            if len(rows) > 1 or len(_top_level(inner, ",")) > 1:
                cells = [[value(c) for c in _top_level(r, ",")] for r in rows]
                if len({len(r) for r in cells}) != 1:
                    raise StataExprError("the rows of the matrix differ in length")
                return _mat(cells)
            return _Algebra(self.session, inner).parse()
        m = re.compile(r"[A-Za-z_]\w*").match(self.text, self.pos)
        if m is None:
            n = re.compile(r"\d*\.?\d+(?:[eE][-+]?\d+)?").match(self.text, self.pos)
            if n is None:
                raise StataExprError(
                    f"the matrix expression {self.text!r} is not understood"
                )
            self.pos = n.end()
            return _mat(float(n.group(0)))
        name = m.group(0)
        self.pos = m.end()
        if self._peek() == "(":
            inner = self._group()
            args = _top_level(inner, ",")
            low = name.lower()
            if name == "J" and len(args) == 3:
                r, c, v = (value(a) for a in args)
                return _mat(np.full((int(r), int(c)), v, dtype=float))
            if name == "I" and len(args) == 1:
                return _mat(np.eye(int(value(args[0]))))
            if low in ("e", "get") and len(args) == 1:
                which = {"b": "b", "_b": "b", "V": "V", "VCE": "V"}.get(args[0].strip())
                if args[0].strip() == "Sigma":
                    return _residual_covariance(self.session)
                if which is None:
                    raise StataExprError(f"{name}({inner}) is not implemented")
                return _estimates(self.session, which)
            if low in ("inv", "invsym", "syminv") and len(args) == 1:
                a = _Algebra(self.session, args[0]).parse()
                if a["values"].shape[0] != a["values"].shape[1]:
                    raise StataExprError("conformability error")
                return _mat(np.linalg.pinv(a["values"]), a["cols"], a["rows"])
            if low in ("diag", "vecdiag", "trace") and len(args) == 1:
                a = _Algebra(self.session, args[0]).parse()["values"]
                if low == "diag":
                    return _mat(np.diag(np.ravel(a)))
                if a.shape[0] != a.shape[1]:
                    raise StataExprError("conformability error")
                return _mat(np.diag(a)[None, :] if low == "vecdiag" else np.trace(a))
            # a scalar function of scalars: sqrt(2), rowsof(A) ...
            return _mat(value(f"{name}({inner})"))
        if name in held:
            source = held[name]
            return _mat(source["values"].copy(), source["rows"], source["cols"])
        return _mat(value(name))  # a scalar


def _evaluate(session: "StataSession", expr: str) -> Dict[str, Any]:
    return _Algebra(session, expr.strip()).parse()


def _names(spec: str) -> List[str]:
    return [m.group(1) or m.group(0) for m in re.finditer(r'"([^"]*)"|\S+', spec)]


def matrix_line(session: "StataSession", line: str) -> Optional[bool]:
    """Run ``line`` if it is a matrix command, ``svmat`` or ``mkmat``."""
    m = _SVMAT.match(line)
    if m:
        return _svmat(session, m.group(1))
    m = _MKMAT.match(line)
    if m:
        return _mkmat(session, m.group(1))
    m = _MATRIX.match(line)
    if m is None:
        return None
    body = m.group(1)
    held = _table(session)
    word, _, rest = body.partition(" ")
    low = word.lower()
    if low in ("list", "l", "li", "dir"):
        name = rest.split(",")[0].strip()
        if low == "dir" or not name:
            return False
        if re.fullmatch(r"e\((b|V|Sigma)\)", name):
            mat = _evaluate(session, name)  # `matrix list e(V)`
        elif name not in held:
            raise StataExprError(f"matrix {name} not found")
        else:
            mat = held[name]
        session.output = pd.DataFrame(
            mat["values"], index=mat["rows"], columns=mat["cols"]
        )
        return True
    if low == "drop":
        for name in rest.split():
            if name == "_all":
                held.clear()
            elif name in held:
                del held[name]
            else:
                raise StataExprError(f"matrix {name} not found")
        return False
    if low in ("colnames", "rownames", "coln", "rown"):
        name, _, spec = rest.partition("=")
        name = name.strip()
        if name not in held:
            raise StataExprError(f"matrix {name} not found")
        names = _names(spec)
        axis = "cols" if low.startswith("col") else "rows"
        size = held[name]["values"].shape[1 if axis == "cols" else 0]
        if len(names) != size:
            raise StataExprError(f"matrix {low}: {len(names)} names for {size}")
        held[name][axis] = names
        return False
    d = _DEFINE.fullmatch(body)
    if d is None:
        raise StataExprError(f"the matrix command {body!r} is not implemented")
    name, i_expr, j_expr, value = d.group(1), d.group(2), d.group(3), d.group(4)
    if i_expr is None:
        made = _evaluate(session, value)
        _store(session, name, made["values"], made["rows"], made["cols"])
        return False
    if name not in held:
        raise StataExprError(f"matrix {name} not found")
    values = held[name]["values"]
    i, j = int(session.value(i_expr)), int(session.value(j_expr))
    if not (1 <= i <= values.shape[0] and 1 <= j <= values.shape[1]):
        raise StataExprError(f"matrix {name}[{i},{j}] is out of range")
    values[i - 1, j - 1] = session.value(value)
    if session.simulated or session.stored.get("random_draws"):
        held[name]["random"] = True  # filled from simulated or resampled data
    return False


def _svmat(session: "StataSession", rest: str) -> bool:
    from ._stata_datastep import DataSteps
    from ._stata_lexer import parse as _parse

    cmd = _parse("svmat " + rest)
    options = {str(k).lower(): v for k, v in dict(cmd.options).items()}
    words = [w for w in cmd.varlist if w.lower() not in ("float", "double")]
    double = any(w.lower() == "double" for w in cmd.varlist)
    if len(words) != 1 or set(options) - {"names", "n"}:
        raise StataExprError("`svmat` is run as `svmat [type] A [, names(stub)]`")
    name = words[0]
    held = _table(session)
    if name not in held:
        raise StataExprError(f"matrix {name} not found")
    mat = held[name]
    values = mat["values"]
    spec = str(options.get("names") or options.get("n") or name).strip().strip('"')
    if spec.lower() == "col":
        columns = list(mat["cols"])
    elif spec.lower() in ("eqcol", "matcol"):
        columns = [f"{name}{c}" for c in mat["cols"]]
    else:
        columns = [f"{spec}{k + 1}" for k in range(values.shape[1])]
    if session._steps is None:
        session._steps = DataSteps(pd.DataFrame())
        session._steps.stored = session.stored
    steps = session._steps
    clash = [c for c in columns if c in steps.data.columns]
    if clash:
        raise StataExprError(f"variable(s) {clash} already exist")
    if len(steps.data) < values.shape[0]:
        steps.set_obs(values.shape[0])
    for k, column in enumerate(columns):
        full = np.full(len(steps.data), np.nan)
        full[: values.shape[0]] = values[:, k]
        steps.add_column(column, full, double=double)
    if mat.get("random"):
        session.stored["random_draws"] = True
    return False


def _mkmat(session: "StataSession", rest: str) -> bool:
    from ._stata_datastep import row_mask
    from ._stata_lexer import parse as _parse

    if session._steps is None:
        raise StataExprError("`mkmat` needs data in memory")
    cmd = _parse("mkmat " + rest)
    options = {str(k).lower(): v for k, v in dict(cmd.options).items()}
    if set(options) - {"matrix", "mat"}:
        raise StataExprError("`mkmat` is run as `mkmat varlist [, matrix(A)]`")
    data = session._steps.data
    missing = [v for v in cmd.varlist if v not in data.columns]
    if missing or not cmd.varlist:
        raise StataExprError(f"mkmat: variable(s) {missing} are not in the data")
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    block = data.loc[mask, list(cmd.varlist)].to_numpy(dtype=float)
    target = options.get("matrix") or options.get("mat")
    if target:
        _store(session, str(target).strip(), block, None, list(cmd.varlist))
    else:
        for k, name in enumerate(cmd.varlist):
            _store(session, name, block[:, [k]], None, [name])
    return False
