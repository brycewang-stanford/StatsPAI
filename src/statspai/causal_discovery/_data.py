"""Input checks shared by the structure learners.

A partial-correlation test on a constant column, or on columns that are an
exact linear combination of one another, has no defined answer. The
learners used to turn the resulting NaN or singular matrix into a decision
("dependent" in ``pc_algorithm``, "independent" after a singular inverse),
so the graph changed without a word. These helpers refuse such data
instead, and pick the numeric columns the same way everywhere.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = [
    "numeric_columns",
    "check_gaussian_columns",
    "check_alpha",
    "check_no_missing",
    "is_acyclic_matrix",
    "warn_if_cyclic",
]


def numeric_columns(data: pd.DataFrame) -> List[str]:
    """Columns pandas counts as numeric, booleans included."""
    return [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]


def check_alpha(alpha: float, function: str, name: str = "alpha") -> None:
    """``alpha`` must be a level strictly between 0 and 1."""
    try:
        ok = 0.0 < float(alpha) < 1.0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise MethodIncompatibility(
            f"{function}: {name}={alpha!r} is not a significance level in (0, 1).",
            recovery_hint=f"Pass {name}=0.05 or another value between 0 and 1.",
            diagnostics={name: alpha},
        )


def check_gaussian_columns(X: np.ndarray, names: Sequence[str], function: str) -> None:
    """Refuse constant columns and exactly collinear sets of columns.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        The data, complete rows only.
    names : sequence of str
        Column names, aligned with ``X``.
    function : str
        Caller name for the message.
    """
    names = [str(n) for n in names]
    spread = np.ptp(X, axis=0) if len(X) else np.zeros(X.shape[1])
    scale = np.max(np.abs(X), axis=0) if len(X) else np.zeros(X.shape[1])
    constant = [n for n, s, m in zip(names, spread, scale) if s <= 1e-12 * m or s == 0]
    if constant:
        raise DataInsufficient(
            f"{function}: column(s) {constant} are constant, so their partial "
            "correlation with any other column is not defined.",
            recovery_hint="Drop the constant column(s).",
            diagnostics={"constant_columns": constant},
        )
    if X.shape[0] <= X.shape[1]:
        raise DataInsufficient(
            f"{function}: {X.shape[0]} complete rows for {X.shape[1]} "
            "columns; the partial correlations need more rows than columns.",
            recovery_hint="Use fewer columns (variables=[...]) or more rows.",
            diagnostics={"n_rows": int(X.shape[0]), "n_columns": int(X.shape[1])},
        )
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    rank = 0
    for k in range(Z.shape[1]):
        new_rank = int(np.linalg.matrix_rank(Z[:, : k + 1]))
        if new_rank == rank:
            raise DataInsufficient(
                f"{function}: column {names[k]!r} is an exact linear "
                f"combination of {names[:k]} (collinear), so the conditional "
                "independence tests that involve it have no answer.",
                recovery_hint=(
                    "Drop the redundant column; a deterministic relation "
                    "breaks the faithfulness assumption the learner rests on."
                ),
                diagnostics={"collinear_column": names[k]},
            )
        rank = new_rank


def check_no_missing(frame: pd.DataFrame, function: str) -> None:
    """Time-series learners: rows are time points, so none may be dropped."""
    holes = frame.columns[frame.isna().any()].tolist()
    if holes:
        raise DataInsufficient(
            f"{function}: column(s) {holes} have missing values. The rows are "
            "consecutive time points, so a row cannot be deleted without "
            "shifting every lag after it.",
            recovery_hint=(
                "Fill the gaps first (interpolate, carry forward) or cut the "
                "series to a stretch without missing values."
            ),
            diagnostics={"columns_with_missing": holes},
        )


def is_acyclic_matrix(W: np.ndarray) -> bool:
    """True when the nonzero pattern of ``W`` (``W[i, j]``: ``i -> j``) has
    no directed cycle, self-loops included."""
    A = np.asarray(W) != 0
    d = A.shape[0]
    indeg = A.sum(axis=0).astype(int)
    stack = [j for j in range(d) if indeg[j] == 0]
    seen = 0
    while stack:
        i = stack.pop()
        seen += 1
        for j in np.flatnonzero(A[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                stack.append(int(j))
    return seen == d


def warn_if_cyclic(W: np.ndarray, function: str, h_value: float) -> bool:
    """Warn when a continuous-optimisation learner returns a cyclic graph.

    Returns whether the graph is acyclic.
    """
    ok = is_acyclic_matrix(W)
    if not ok:
        import warnings

        warnings.warn(
            f"{function}: the thresholded graph has a directed cycle "
            f"(acyclicity constraint h = {h_value:.3g}); the optimiser "
            "stopped before reaching a DAG. Raise max_iter, or raise the "
            "threshold, before reading the edges as a causal graph.",
            RuntimeWarning,
            stacklevel=3,
        )
    return ok
