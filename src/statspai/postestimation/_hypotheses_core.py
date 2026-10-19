"""Restriction parsing and joint Wald tests behind :func:`sp.hypotheses`."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility

_B_INDEX = re.compile(r"(?<![A-Za-z0-9_])b(\d+)(?![A-Za-z0-9_])")


def _table_jacobian(table: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """The Jacobian of the rows of ``table`` and the coefficient covariance.

    The Jacobian is stored for the rows the table was built with; a subset
    (``tab.iloc[[0, 2]]``, ``tab[tab.term == "x"]``) keeps the attrs of the
    whole table, so its rows are picked by index label.
    """
    jac = table.attrs.get("jacobian")
    V = table.attrs.get("vcov_coefficients")
    if jac is None or V is None:
        raise MethodIncompatibility(
            "hypotheses: this table does not carry its Jacobian; pass a "
            "table from sp.predictions / comparisons / slopes or a fit.",
            recovery_hint="Pass the table returned by sp.avg_slopes / "
            "avg_comparisons / avg_predictions (or a subset of its rows).",
        )
    J = np.asarray(jac, dtype=float)
    labels = table.attrs.get("jacobian_index")
    if labels is not None and tuple(table.index) != tuple(labels):
        pos = {lab: i for i, lab in enumerate(labels)}
        try:
            rows = [pos[lab] for lab in table.index]
        except (KeyError, TypeError):
            rows = None
        if rows is None or len(set(table.index)) != len(table.index):
            raise MethodIncompatibility(
                "hypotheses: the rows of this table are not rows of the table "
                "its Jacobian was computed for (re-indexed or combined "
                "tables lose the link).",
                recovery_hint="Test on the table returned by the "
                "marginaleffects function, or on a subset of its rows that "
                "keeps the original index.",
            )
        J = J[rows]
    elif J.shape[0] != len(table):
        raise MethodIncompatibility(
            f"hypotheses: the table has {len(table)} rows but its Jacobian "
            f"has {J.shape[0]}.",
            recovery_hint="Test on the table returned by the marginaleffects "
            "function.",
        )
    return J, np.asarray(V, dtype=float)


def _hypothesis_list(hypothesis: Any) -> List[str]:
    if isinstance(hypothesis, str):
        return [hypothesis]
    try:
        texts = list(hypothesis)
    except TypeError:
        texts = None
    if not texts or not all(isinstance(h, str) for h in texts):
        raise MethodIncompatibility(
            "hypotheses: `hypothesis` must be a string or a list of strings; "
            f"got {hypothesis!r}.",
            recovery_hint='E.g. "x1 = 0" or ["x1 = 0", "x2 = 0"].',
        )
    return texts


def _parse_restriction(
    hypothesis: str, beta: np.ndarray, names: List[str]
) -> Tuple[float, np.ndarray]:
    """Value of ``lhs - rhs`` at ``beta`` and its gradient."""
    from .nlcom import _evaluate

    text = hypothesis.strip()
    if text.count("=") > 1 or "==" in text:
        raise MethodIncompatibility(
            "hypotheses: use a single '='.",
            recovery_hint="Write one restriction per string.",
        )
    if "=" in text:
        lhs, rhs = text.split("=")
        text = f"({lhs}) - ({rhs})"
    lookup: Dict[str, int] = {}
    for i in range(beta.size):
        lookup[f"b{i + 1}"] = i
    for i, n in enumerate(names):
        if n.isidentifier():
            lookup.setdefault(n, i)
    for m in _B_INDEX.finditer(text):
        if int(m.group(1)) > beta.size or int(m.group(1)) < 1:
            raise MethodIncompatibility(
                f"hypotheses: b{m.group(1)} is out of range (1..{beta.size})."
            )
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise MethodIncompatibility(
            f"hypotheses: cannot parse {hypothesis!r}."
        ) from exc
    value, grad = _evaluate(tree, beta, lookup, hypothesis)
    return float(value), np.asarray(grad, dtype=float).reshape(-1)


def _joint_wald(
    result: Any,
    hypothesis: Any,
    joint: Any,
    joint_test: str,
    df: Any,
    beta: np.ndarray,
    cov: np.ndarray,
    names: List[str],
) -> pd.DataFrame:
    """One Wald test of several (possibly non-linear) restrictions."""
    test = str(joint_test).lower()
    if test not in ("f", "chisq"):
        raise MethodIncompatibility(
            f"hypotheses: joint_test must be 'f' or 'chisq'; got {joint_test!r}.",
            recovery_hint="Use joint_test='f' or joint_test='chisq'.",
        )
    if joint is True:
        if hypothesis is None:
            labels = names or [f"b{i + 1}" for i in range(beta.size)]
            texts = [f"{lab} = 0" for lab in labels]
            G = np.eye(beta.size)
            g = beta.copy()
        else:
            texts = _hypothesis_list(hypothesis)
            parsed = [_parse_restriction(h, beta, names) for h in texts]
            g = np.array([v for v, _ in parsed])
            G = np.vstack([gr for _, gr in parsed])
    else:
        if hypothesis is not None:
            raise MethodIncompatibility(
                "hypotheses: pass either joint=[names] or hypothesis with "
                "joint=True, not both.",
                recovery_hint='Use joint=["x1", "x2"] alone, or '
                'hypothesis=["x1 = 0", "x2 = 0"], joint=True.',
            )
        wanted = [joint] if isinstance(joint, str) else list(joint)
        idx = []
        for w in wanted:
            w = str(w)
            m = re.fullmatch(r"b(\d+)", w)
            if w in names:
                idx.append(names.index(w))
            elif m and 1 <= int(m.group(1)) <= beta.size:
                idx.append(int(m.group(1)) - 1)
            else:
                raise MethodIncompatibility(
                    f"hypotheses: {w!r} is not an estimate of this result.",
                    recovery_hint=f"Name one of {names or 'b1, b2, ...'}.",
                    diagnostics={"available": names},
                )
        texts = [f"{wanted[j]} = 0" for j in range(len(idx))]
        G = np.eye(beta.size)[idx]
        g = beta[idx]
    q = G.shape[0]
    M = G @ cov @ G.T
    if np.linalg.matrix_rank(M) < q:
        raise MethodIncompatibility(
            "hypotheses: the restrictions are linearly dependent (or one has "
            "zero variance); their Wald statistic is not defined.",
            recovery_hint="Drop the redundant restriction(s).",
            diagnostics={"n_restrictions": q, "rank": int(np.linalg.matrix_rank(M))},
        )
    W = float(g @ np.linalg.solve(M, g))
    row: Dict[str, Any] = {"hypothesis": "; ".join(texts)}
    if test == "chisq":
        row.update(statistic=W, p_value=float(stats.chi2.sf(W, q)), df1=q)
    else:
        if df is not None and df != "model":
            df2 = float(df)
        else:
            df2 = _residual_df(result, beta.size)
        F = W / q
        p = stats.f.sf(F, q, df2) if np.isfinite(df2) else stats.chi2.sf(W, q)
        row.update(statistic=F, p_value=float(p), df1=q, df2=df2)
    row["test"] = test
    return pd.DataFrame([row])


def _residual_df(result: Any, k: int) -> float:
    """Residual degrees of freedom ``n - k`` (marginaleffects' F ``df2``)."""
    if isinstance(result, pd.DataFrame):
        return float("inf")
    di = getattr(result, "data_info", None) or {}
    dr = di.get("df_resid")
    if dr is not None:
        try:
            v = float(dr)
            if np.isfinite(v) and v > 0:
                return v
        except (TypeError, ValueError):
            pass
    n = di.get("nobs", getattr(result, "nobs", None))
    try:
        return float(int(n) - k)
    except (TypeError, ValueError):
        return float("inf")
