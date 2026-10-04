"""Nonlinear combinations of coefficients by the delta method.

``sp.nlcom(result, "x1 / (1 - x2)")`` evaluates a smooth function of the
estimated coefficients and gives it a standard error from the first-order
expansion ``g(b) ~ g(beta) + G (b - beta)``: ``Var = G V G'`` with ``G`` the
gradient at the estimates. Stata's ``nlcom`` and R's ``car::deltaMethod`` are
the references.

The expression is parsed into a syntax tree and evaluated node by node with
its gradient (forward-mode differentiation); nothing is handed to ``eval``,
and only arithmetic and a short list of functions are accepted.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..exceptions import MethodIncompatibility
from ._covariance import require_covariance
from .hypothesis import _params, _resolve_name

__all__ = ["nlcom"]

_B = re.compile(r"_b\[([^\]]+)\]")
_Dual = Tuple[float, np.ndarray]


def _functions() -> Dict[str, Any]:
    """name -> (value, derivative) of the unary functions accepted."""
    return {
        "exp": (np.exp, np.exp),
        "ln": (np.log, lambda v: 1.0 / v),
        "log": (np.log, lambda v: 1.0 / v),
        "sqrt": (np.sqrt, lambda v: 0.5 / np.sqrt(v)),
        "abs": (np.abs, np.sign),
        "normal": (sp_stats.norm.cdf, sp_stats.norm.pdf),
        "normalden": (sp_stats.norm.pdf, lambda v: -v * sp_stats.norm.pdf(v)),
        "invlogit": (
            lambda v: 1.0 / (1.0 + np.exp(-v)),
            lambda v: np.exp(-v) / (1.0 + np.exp(-v)) ** 2,
        ),
    }


def _evaluate(
    node: ast.AST, beta: np.ndarray, lookup: Dict[str, int], expression: str
) -> _Dual:
    k = beta.size
    zero = np.zeros(k)

    def bad(what: str) -> MethodIncompatibility:
        return MethodIncompatibility(
            f"nlcom({expression!r}): {what}.",
            recovery_hint="Use + - * / ** (or ^), parentheses, numbers, "
            "coefficient names (or _b[name]) and "
            + ", ".join(sorted(_functions()))
            + ".",
        )

    def walk(n: ast.AST) -> _Dual:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value), zero
        if isinstance(n, ast.Name):
            if n.id not in lookup:
                raise bad(f"unknown coefficient {n.id!r}")
            grad = zero.copy()
            grad[lookup[n.id]] = 1.0
            return float(beta[lookup[n.id]]), grad
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v, g = walk(n.operand)
            return (-v, -g) if isinstance(n.op, ast.USub) else (v, g)
        if isinstance(n, ast.BinOp):
            a, ga = walk(n.left)
            b, gb = walk(n.right)
            if isinstance(n.op, ast.Add):
                return a + b, ga + gb
            if isinstance(n.op, ast.Sub):
                return a - b, ga - gb
            if isinstance(n.op, ast.Mult):
                return a * b, ga * b + gb * a
            if isinstance(n.op, ast.Div):
                if b == 0:
                    raise bad("division by zero at the estimates")
                return a / b, ga / b - gb * a / b**2
            if isinstance(n.op, ast.Pow):
                if np.any(gb):
                    if a <= 0:
                        raise bad("a non-positive base with an estimated exponent")
                    value = a**b
                    return value, value * (gb * np.log(a) + ga * b / a)
                return a**b, ga * b * a ** (b - 1)
            raise bad("an operator other than + - * / **")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            table = _functions()
            if n.func.id not in table or len(n.args) != 1 or n.keywords:
                raise bad(f"function {n.func.id!r} is not available")
            v, g = walk(n.args[0])
            f, df = table[n.func.id]
            with np.errstate(all="ignore"):
                value, slope = float(f(v)), float(df(v))
            if not np.isfinite(value) or not np.isfinite(slope):
                raise bad(f"{n.func.id}() is not defined at the estimates")
            return value, g * slope
        raise bad("it is not an arithmetic expression")

    return walk(node)


def nlcom(result: Any, expression: str, alpha: float = 0.05) -> Dict[str, Any]:
    """Nonlinear combination of coefficients with a delta-method standard error.

    Equivalent to Stata's ``nlcom`` and R's ``car::deltaMethod``. For a
    linear combination use :func:`lincom`, whose inference is exact given
    the covariance matrix.

    Parameters
    ----------
    result : EconometricResults or CausalResult
        Fitted model with ``.params`` and a covariance matrix.
    expression : str
        A function of the coefficients, written with their names or with
        ``_b[name]`` (required when a name is not a plain identifier, such
        as ``_b[C(race)[T.2]]`` or Stata's ``_b[2.race]``). ``_cons`` names
        the intercept. Operators ``+ - * / **`` (``^`` is read as a power),
        parentheses, numbers and the functions ``exp``, ``ln`` / ``log``,
        ``sqrt``, ``abs``, ``normal``, ``normalden``, ``invlogit``.

        - ``"x1 / x2"`` — a ratio of two coefficients
        - ``"x / (1 - L_y)"`` — the long-run effect in a partial-adjustment
          model ``y = a + b x + c L.y``
        - ``"-x1 / (2 * x1sq)"`` — the turning point of a quadratic
        - ``"exp(x1) - 1"`` — a semi-elasticity as a percentage change
    alpha : float, default 0.05
        ``1 - alpha`` is the confidence level.

    Returns
    -------
    dict
        ``estimate``, ``se``, ``statistic`` (and ``z``), ``pvalue``, ``ci``,
        ``distribution`` (always ``"z"``: the approximation is asymptotic,
        as in Stata), ``gradient`` (a Series over the coefficients) and
        ``expression``.

    Raises
    ------
    MethodIncompatibility
        An unknown coefficient or function, a syntax error, an expression
        with no coefficients, or one not defined at the estimates.

    Notes
    -----
    The delta method is a first-order approximation. For a ratio whose
    denominator is not well separated from zero it can be poor, and the
    interval is symmetric where the sampling distribution is not; a
    Fieller interval or the bootstrap is then the better tool. The Wald
    test of a nonlinear hypothesis is also not invariant to how the
    hypothesis is written.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> y = 1.0 + 2.0 * x1 + 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    >>> fit = sp.regress("y ~ x1 + x2", data=df)
    >>> out = sp.nlcom(fit, "x1 / x2")
    >>> bool(out["ci"][0] < 4.0 < out["ci"][1])
    True
    >>> sorted(out)  # doctest: +NORMALIZE_WHITESPACE
    ['ci', 'distribution', 'estimate', 'expression', 'gradient',
     'pvalue', 'se', 'statistic', 'z']
    """
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"nlcom: alpha must be in (0, 1), got {alpha!r}.",
            recovery_hint="alpha is 1 minus the confidence level, e.g. 0.05.",
        )
    params = _params(result)
    beta = params.to_numpy(dtype=float)
    names: List[str] = [str(n) for n in params.index]

    # _b[...] may hold any coefficient name; swap each for an identifier
    lookup: Dict[str, int] = {}

    def placeholder(match: "re.Match[str]") -> str:
        position = _resolve_name(match.group(1).strip(), params, expression)
        key = f"_coef_{position}_"
        lookup[key] = position
        return key

    text = _B.sub(placeholder, expression).replace("^", "**")
    for position, name in enumerate(names):
        if name.isidentifier():
            lookup.setdefault(name, position)
    for alias in ("_cons", "Intercept", "const"):
        if alias.isidentifier() and alias not in lookup:
            try:
                lookup[alias] = _resolve_name(alias, params, expression)
            except MethodIncompatibility:
                pass
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise MethodIncompatibility(
            f"nlcom({expression!r}) is not a valid expression ({exc.msg}).",
            recovery_hint="Write coefficient names that are not plain "
            "identifiers as _b[name].",
        ) from exc
    estimate, gradient = _evaluate(tree.body, beta, lookup, expression)
    if not np.any(gradient):
        raise MethodIncompatibility(
            f"nlcom({expression!r}) contains no coefficients.",
            recovery_hint=f"Available terms: {names}.",
        )
    V = require_covariance(result, gradient.reshape(1, -1), f"nlcom({expression!r})")
    se = float(np.sqrt(max(float(gradient @ V @ gradient), 0.0)))
    statistic = estimate / se if se > 0 else float("nan")
    crit = float(sp_stats.norm.ppf(1 - alpha / 2))
    return {
        "estimate": float(estimate),
        "se": se,
        "statistic": statistic,
        "z": statistic,
        "distribution": "z",
        "pvalue": float(2 * sp_stats.norm.sf(abs(statistic))),
        "ci": (estimate - crit * se, estimate + crit * se),
        "gradient": pd.Series(gradient, index=params.index),
        "expression": expression,
    }
