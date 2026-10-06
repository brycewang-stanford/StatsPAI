"""Omitting perfectly collinear regressors before a likelihood is maximised.

A rank-deficient design has no unique maximum-likelihood estimate: the
likelihood is flat along the dependent direction. Newton iterations then
stop wherever the pseudo-inverse or the step tolerance leaves them, and
the "estimates" and their standard errors (zero, 1e6 or ``nan``) mean
nothing. The estimators that call :func:`drop_collinear` instead omit the
later member of each dependent set, as Stata does, and say so.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: a column is dependent when less than this share of its length is left
#: after projecting on the columns kept before it
_TOL = 1e-9

_INTERCEPTS = ("Intercept", "_cons", "const", "(Intercept)")


def independent_columns(
    X: np.ndarray,
    names: Sequence[str],
    order: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[Dict[str, str]]]:
    """Columns that add information, scanned in ``order``.

    Returns the positions to keep (in the original column order) and one
    ``{'variable', 'reason'}`` record per omitted column. The intercept is
    scanned first whatever the order, so it is never the one omitted; the
    later member of a dependent set goes.
    """
    X = np.asarray(X, dtype=float)
    k = X.shape[1]
    labels = [str(n) for n in names]
    scan = list(range(k)) if order is None else [int(j) for j in order]
    first = [j for j in scan if labels[j] in _INTERCEPTS]
    scan = first + [j for j in scan if j not in first]
    length = np.sqrt(np.einsum("ij,ij->j", X, X))
    n = X.shape[0]
    # Fast path. With unit-length columns in scan order, the diagonal of R
    # in an unpivoted QR is what is left of each column after projecting on
    # the ones before it, as long as none of those is dependent. So a
    # diagonal that stays above the tolerance certifies full rank in one
    # LAPACK call; only a deficient design pays for the column-by-column
    # scan that says which column goes and why.
    if n >= k and np.all(length[scan] > 0):
        r = np.linalg.qr(X[:, scan] / length[scan], mode="r")
        if np.abs(np.diag(r)).min() >= _TOL:
            return sorted(scan), []
    basis = np.empty((n, min(n, k)))
    m = 0
    kept: List[int] = []
    omitted: List[Dict[str, str]] = []
    for j in scan:
        if not length[j] > 0:
            omitted.append({"variable": labels[j], "reason": "zero column"})
            continue
        v = X[:, j] / length[j]
        for _ in range(2):  # twice: one Gram-Schmidt pass loses digits
            v = v - basis[:, :m] @ (basis[:, :m].T @ v)
        norm = float(np.sqrt(v @ v))
        if norm < _TOL or m == basis.shape[1]:
            omitted.append(
                {"variable": labels[j], "reason": _dependence(X, labels, kept, j)}
            )
            continue
        basis[:, m] = v / norm
        m += 1
        kept.append(j)
    return sorted(kept), omitted


def _dependence(X: np.ndarray, labels: List[str], kept: List[int], j: int) -> str:
    """Which kept columns reproduce column ``j``."""
    if not kept:
        return "collinear"
    coef, *_ = np.linalg.lstsq(X[:, kept], X[:, j], rcond=None)
    scale = np.sqrt(np.einsum("ij,ij->j", X[:, kept], X[:, kept])) * np.abs(coef)
    involved = [labels[c] for c, s in zip(kept, scale) if s > 1e-7 * scale.max()]
    if len(involved) == 1:
        return f"collinear with '{involved[0]}'"
    shown = involved[:6] + (["..."] if len(involved) > 6 else [])
    return "linear combination of '" + "', '".join(shown) + "'"


def drop_collinear(
    X: np.ndarray,
    names: Sequence[str],
    who: str,
    formula: Optional[str] = None,
    design_info: Any = None,
    stacklevel: int = 3,
) -> Tuple[np.ndarray, List[str], List[Dict[str, str]], List[int]]:
    """Remove dependent columns from a design and announce it.

    Returns the reduced design, its column names, the omission records
    (empty when the design has full rank) and the positions kept. With a
    formula and its patsy design, columns are scanned in the order the
    formula writes them, so ``y ~ a + b + c`` with ``a + b + c = 1`` omits
    ``c``, as Stata would.
    """
    X = np.asarray(X, dtype=float)
    labels = [str(n) for n in names]
    if X.ndim != 2 or X.shape[1] < 2:
        return X, labels, [], list(range(X.shape[1] if X.ndim == 2 else 0))
    order: Optional[List[int]] = None
    if formula and design_info is not None:
        from ..regression.ols import _written_column_order

        order = _written_column_order(formula, design_info, X.shape[1])
    keep, omitted = independent_columns(X, labels, order)
    if not omitted:
        return X, labels, [], keep
    warnings.warn(
        f"{who}: "
        + "; ".join(
            f"note: {o['variable']} omitted because of collinearity ({o['reason']})"
            for o in omitted
        )
        + ". The remaining coefficients are identified; the omitted one is not.",
        UserWarning,
        stacklevel=stacklevel,
    )
    return X[:, keep], [labels[j] for j in keep], omitted, keep


def drop_collinear_names(
    frame: Any, x: Sequence[str], who: str, stacklevel: int = 3
) -> Tuple[List[str], List[Dict[str, str]]]:
    """The regressors of ``x`` that are independent given a constant.

    For the estimators that take a list of columns and add the constant
    themselves. ``frame`` holds the estimation rows.
    """
    cols = [str(v) for v in x]
    if not cols:
        return cols, []
    design = np.column_stack(
        [np.ones(len(frame))] + [np.asarray(frame[v], dtype=float) for v in cols]
    )
    _, kept, omitted, _ = drop_collinear(
        design, ["_cons"] + cols, who, stacklevel=stacklevel + 1
    )
    return [v for v in kept if v != "_cons"], omitted


__all__ = ["independent_columns", "drop_collinear", "drop_collinear_names"]
