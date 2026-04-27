"""``sp.fast.dsl`` — minimal expression helpers for HDFE specs.

This module mirrors the most common pieces of fixest's formula DSL:

- :func:`i`            — one-hot indicators with optional reference category
                          (the ``i(year, ref=2010)`` event-study idiom).
- :func:`fe_interact`  — combine two or more FE columns into a single
                          interacted code (the ``i^j`` operator).
- :func:`sw` / :func:`csw` — stepwise / cumulative-stepwise spec
                              expansion (one regression per element).

These are functional helpers, not a parser — the parser is intentionally
deferred to a later phase. You feed the helpers into the existing
``sp.feols`` / ``sp.fast.fepois`` callers manually:

    >>> dummies = sp.fast.i(df['year'], ref=2010)
    >>> X = pd.concat([df[['x']], dummies], axis=1)
    >>> sp.fast.fepois('y ~ ' + ' + '.join(X.columns) + ' | firm', data=df.join(dummies))
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# i() — event-study / interacted dummies
# ---------------------------------------------------------------------------

def i(
    var: Union[pd.Series, np.ndarray],
    ref: object = None,
    *,
    drop_first: bool = True,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """One-hot indicators for ``var`` with an optional reference category dropped.

    Mirrors fixest's ``i(var, ref=...)``.

    Parameters
    ----------
    var : Series or ndarray
        Categorical variable (numeric or string). Cast via ``pd.Categorical``.
    ref : scalar, optional
        Level to drop (the reference). If None and ``drop_first`` is True,
        drops the first level alphabetically/sortedly.
    drop_first : bool, default True
        If ``ref`` is None, whether to drop the first level (matches
        statsmodels / patsy convention).
    prefix : str, optional
        Column name prefix. Defaults to the Series name (or "i" for ndarrays).

    Returns
    -------
    DataFrame
        Columns named ``"<prefix>::<level>"`` for each non-reference level.
        Float64 values (so they can stack with continuous regressors).
    """
    if isinstance(var, pd.Series):
        ser = var
    else:
        ser = pd.Series(np.asarray(var))
    if prefix is None:
        prefix = str(ser.name) if ser.name is not None else "i"

    cats = pd.Categorical(ser)
    levels = cats.categories
    out = pd.DataFrame(
        {f"{prefix}::{lv}": (cats == lv).astype(np.float64) for lv in levels},
        index=ser.index,
    )

    if ref is not None:
        col = f"{prefix}::{ref}"
        if col not in out.columns:
            raise ValueError(f"ref={ref!r} not a level of {prefix!r}")
        out = out.drop(columns=[col])
    elif drop_first and len(out.columns) > 0:
        out = out.iloc[:, 1:]

    return out


# ---------------------------------------------------------------------------
# ^ operator: FE interactions
# ---------------------------------------------------------------------------

def fe_interact(*cols: Union[pd.Series, np.ndarray, Sequence]) -> np.ndarray:
    """Combine K FE columns into a single int64 code per (c_1, ..., c_K) tuple.

    Mirrors fixest's ``i^j`` operator: ``fe_interact(df['firm'], df['year'])``
    produces one integer code per unique (firm, year) combination, suitable
    to drop into the ``fe`` slot of any HDFE estimator.

    Returns
    -------
    ndarray of int64
    """
    if not cols:
        raise ValueError("fe_interact: at least one column required")

    arrays = [np.asarray(c) for c in cols]
    n = arrays[0].shape[0]
    for a in arrays[1:]:
        if a.shape[0] != n:
            raise ValueError("fe_interact: all columns must have the same length")
    if len(arrays) == 1:
        codes, _ = pd.factorize(arrays[0], sort=False)
        return codes.astype(np.int64)

    # Build one-shot tuples; pd.factorize on the tuple-Series gives codes.
    # Avoid pandas object overhead for the simple two-column case via a
    # quick path: pack into int128-equivalent by concatenating factorised
    # halves with a multiplier. Falls through to the safe path otherwise.
    if len(arrays) == 2:
        c0, _ = pd.factorize(arrays[0], sort=False)
        c1, _ = pd.factorize(arrays[1], sort=False)
        n1 = int(c1.max()) + 1 if c1.size else 0
        if n1 == 0:
            return c0.astype(np.int64)
        packed = c0.astype(np.int64) * n1 + c1.astype(np.int64)
        codes, _ = pd.factorize(packed, sort=False)
        return codes.astype(np.int64)

    # General K-way fallback.
    df = pd.DataFrame({f"_c{i}": a for i, a in enumerate(arrays)})
    codes, _ = pd.factorize(df.apply(tuple, axis=1), sort=False)
    return codes.astype(np.int64)


# ---------------------------------------------------------------------------
# sw() / csw() — stepwise spec expansion
# ---------------------------------------------------------------------------

def sw(*specs: Iterable[str]) -> List[List[str]]:
    """Stepwise: each spec emitted independently.

    ``sw(['x1'], ['x2'], ['x1', 'x2'])`` → three separate variable lists.
    """
    return [list(s) for s in specs]


def csw(*specs: Iterable[str]) -> List[List[str]]:
    """Cumulative-stepwise: each spec is the union of itself and all prior.

    ``csw(['x1'], ['x2'], ['x3'])`` →
    ``[['x1'], ['x1', 'x2'], ['x1', 'x2', 'x3']]``.
    """
    cur: List[str] = []
    out: List[List[str]] = []
    for s in specs:
        cur = cur + list(s)
        out.append(list(cur))
    return out


__all__ = ["i", "fe_interact", "sw", "csw"]
