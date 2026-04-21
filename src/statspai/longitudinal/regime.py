"""
Dynamic treatment regime DSL.

A *regime* is a function from patient history to a treatment value.  The
most common forms:

  - **Static**: ``[1, 1, 1]`` (always treat), ``[0, 0, 0]`` (never treat),
    or any fixed sequence of length K.
  - **Dynamic**: condition on time-varying covariates, e.g.,
    "treat when CD4 < 200, otherwise don't."

This module provides:

  - :func:`regime` — parse a string / list / callable into a unified
    :class:`Regime` object
  - :class:`Regime` — holds the rule and evaluates it against a row of history

The string DSL is evaluated via a tiny AST-walking interpreter that
supports comparisons, boolean logic, arithmetic, and variable lookup
only — it never calls :func:`eval` on user input.
"""

from __future__ import annotations

import ast
import operator as op
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Union

import numpy as np
import pandas as pd


__all__ = ["Regime", "regime", "always_treat", "never_treat"]


# --------------------------------------------------------------------------- #
#  Tiny AST-walking interpreter (no eval)
# --------------------------------------------------------------------------- #

_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Mod: op.mod, ast.Pow: op.pow, ast.FloorDiv: op.floordiv,
}
_UNARY_OPS = {
    ast.UAdd: op.pos, ast.USub: op.neg, ast.Not: op.not_,
}
_CMP_OPS = {
    ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge,
    ast.Eq: op.eq, ast.NotEq: op.ne,
}


def _walk(node: ast.AST, env: Dict[str, Any]):
    """Recursively evaluate a whitelisted AST node against ``env``."""
    if isinstance(node, ast.Expression):
        return _walk(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            val = env[node.id]
            # Reject non-numeric types to prevent callable / string / None
            # leakage via a user-controlled history dict.
            if not isinstance(val, (int, float, bool, np.integer, np.floating)):
                raise TypeError(
                    f"Regime variable {node.id!r} must be numeric or "
                    f"boolean; got {type(val).__name__}"
                )
            return val
        raise NameError(f"Unknown variable in regime: {node.id!r}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_walk(node.operand, env))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_walk(node.left, env), _walk(node.right, env))
    if isinstance(node, ast.BoolOp):
        vals = [_walk(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            out = True
            for v in vals:
                out = out and v
            return out
        if isinstance(node.op, ast.Or):
            out = False
            for v in vals:
                out = out or v
            return out
    if isinstance(node, ast.Compare):
        left = _walk(node.left, env)
        for op_node, right_node in zip(node.ops, node.comparators):
            right = _walk(right_node, env)
            cmp_fn = _CMP_OPS.get(type(op_node))
            if cmp_fn is None:
                raise ValueError(f"Unsupported comparison: {type(op_node).__name__}")
            if not cmp_fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _walk(node.body, env) if _walk(node.test, env) else _walk(node.orelse, env)
    raise ValueError(f"Unsupported syntax in regime expression: {type(node).__name__}")


_ALLOWED_NODE_TYPES = (
    ast.Expression, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp,
    ast.Name, ast.Load, ast.Constant,
    ast.And, ast.Or, ast.Not,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.IfExp, ast.USub, ast.UAdd,
)


def _compile(expr: str):
    """Parse ``expr`` into an AST root and return a callable ``env -> value``.

    Also validates at compile time that only whitelisted AST nodes are
    present — this catches ``__import__(...)``, attribute access, etc.
    *before* a regime is ever evaluated.
    """
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(
                f"Disallowed syntax in regime expression: "
                f"{type(node).__name__}. Only comparisons, and/or/not, "
                f"arithmetic, constants, and variable names are allowed."
            )
    root = tree.body

    def _eval(env: Dict[str, Any]):
        return _walk(root, env)

    return _eval


# --------------------------------------------------------------------------- #
#  Regime
# --------------------------------------------------------------------------- #


@dataclass
class Regime:
    """A dynamic treatment regime.

    Attributes
    ----------
    kind : {"static", "dynamic"}
    name : str
    rule : Union[list, Callable]
        Static regimes store a list/sequence; dynamic regimes store a
        callable ``(history_dict, t) -> treatment_value``.
    """

    kind: str
    name: str
    rule: Union[list, Callable]

    def treatment(self, history: dict, t: int = 0, K: int = 1):
        """Evaluate the regime at time ``t`` given history."""
        if self.kind == "static":
            seq = list(self.rule)
            if t >= len(seq):
                return seq[-1]
            return seq[t]
        return self.rule(history, t)

    def apply(self, history_df: pd.DataFrame) -> pd.Series:
        """Vectorized: apply the regime to each row of a history dataframe."""
        if self.kind == "static":
            return pd.Series(
                [self.rule[0]] * len(history_df),
                index=history_df.index,
            )
        values = [self.rule(row.to_dict(), 0) for _, row in history_df.iterrows()]
        return pd.Series(values, index=history_df.index)


def regime(
    rule: Union[str, Sequence, Callable, int, float],
    *,
    name: str = "",
    K: int = 1,
) -> Regime:
    """Build a :class:`Regime` from a rule.

    Parameters
    ----------
    rule : str | sequence | callable | scalar
        - ``"if cd4 < 200 then 1 else 0"`` — string DSL (safely evaluated).
        - ``[1, 1, 0]`` — static sequence of treatment values per period.
        - ``lambda h, t: int(h['cd4'] < 200)`` — explicit callable.
        - ``1`` — scalar, interpreted as always-1 for ``K`` periods.
    name : str, optional
    K : int, default 1
        Number of time periods (used when ``rule`` is scalar).

    Returns
    -------
    Regime
    """
    if callable(rule):
        return Regime(kind="dynamic", name=name or "dynamic", rule=rule)
    if isinstance(rule, bool):
        return Regime(
            kind="static",
            name=name or f"always={int(rule)}",
            rule=[float(rule)] * K,
        )
    if isinstance(rule, (int, float, np.integer, np.floating)):
        return Regime(
            kind="static",
            name=name or f"always={rule}",
            rule=[float(rule)] * K,
        )
    if isinstance(rule, (list, tuple, np.ndarray)) and not isinstance(rule, str):
        return Regime(
            kind="static",
            name=name or "static",
            rule=[float(v) for v in rule],
        )
    if isinstance(rule, str):
        return _compile_string_regime(rule, name=name)
    raise TypeError(f"Cannot build regime from {type(rule).__name__}: {rule!r}")


# --------------------------------------------------------------------------- #
#  String DSL
# --------------------------------------------------------------------------- #


def _compile_string_regime(text: str, *, name: str) -> Regime:
    """Compile a minimal if/then/else DSL into a :class:`Regime`.

    Supported forms
    ---------------
    * ``"if <cond> then <a> else <b>"``
    * ``"<bool_expr>"``   — returns 1 if True, 0 if False
    * ``"always_treat"`` / ``"never_treat"``

    The condition is parsed into an AST and interpreted via
    :func:`_walk`; it never calls :func:`eval`.
    """
    s = text.strip()
    low = s.lower()
    if low in ("always_treat", "always 1", "always=1", "always"):
        return Regime(kind="static", name=name or "always_treat", rule=[1.0])
    if low in ("never_treat", "never 0", "never=0", "never"):
        return Regime(kind="static", name=name or "never_treat", rule=[0.0])

    import re
    m = re.match(r"^\s*if\s+(.+?)\s+then\s+(.+?)\s+else\s+(.+)$", s, re.IGNORECASE)
    if m:
        cond_fn = _compile(m.group(1))
        then_fn = _compile(m.group(2))
        else_fn = _compile(m.group(3))

        def _rule(history, t=0):
            env = dict(history)
            env["t"] = t
            return then_fn(env) if cond_fn(env) else else_fn(env)

        return Regime(kind="dynamic", name=name or s, rule=_rule)

    cond_fn = _compile(s)

    def _rule_bool(history, t=0):
        env = dict(history)
        env["t"] = t
        return int(bool(cond_fn(env)))

    return Regime(kind="dynamic", name=name or s, rule=_rule_bool)


def always_treat(K: int = 1) -> Regime:
    """Convenience: the always-treat regime over K periods."""
    return Regime(kind="static", name="always_treat", rule=[1.0] * K)


def never_treat(K: int = 1) -> Regime:
    """Convenience: the never-treat regime over K periods."""
    return Regime(kind="static", name="never_treat", rule=[0.0] * K)
