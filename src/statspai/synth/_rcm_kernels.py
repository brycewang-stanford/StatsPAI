"""Compiled kernels of the exact best-subset search (regression control).

Kept apart from :mod:`statspai.synth.rcm` so that importing the package does
not import numba: this module is loaded when a best-subset search first runs.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

__all__ = ["best_subsets"]

try:  # numba is a core dependency; fall back to plain Python without it
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without numba

    def njit(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        def wrap(fn: Any) -> Any:
            return fn

        return args[0] if args and callable(args[0]) else wrap


@njit(cache=True)
def _sweep_out(M: np.ndarray, k: int) -> None:  # pragma: no cover - numba
    """Remove variable ``k`` from the regression held in ``M`` (in place).

    ``M`` is the symmetric cross-product matrix of ``[X, y]`` with the
    variables of the current model swept in: its last diagonal entry is the
    residual sum of squares, the last column holds the coefficients and the
    diagonal of a swept variable is minus its entry of ``(X'X)^{-1}``.
    """
    n = M.shape[0]
    d = M[k, k]
    for i in range(n):
        if i != k:
            b = M[i, k] / d
            for j in range(n):
                if j != k:
                    M[i, j] = M[i, j] - b * M[k, j]
    for i in range(n):
        if i != k:
            M[i, k] = -M[i, k] / d
            M[k, i] = M[i, k]
    M[k, k] = -1.0 / d


@njit(cache=True)
def best_subsets(A: np.ndarray, max_nodes: int) -> Tuple:  # pragma: no cover
    """Exact best subset of every size by branch and bound.

    ``A`` is the centred cross-product matrix of ``[X, y]`` with every
    predictor swept in. Subsets are reached by deleting predictors in
    increasing index order; a branch is abandoned when its residual sum of
    squares already exceeds the best one known for every size it can still
    reach (deleting a predictor never lowers the residual sum of squares).
    Returns the best residual sum of squares and membership mask per size,
    and the number of nodes visited (``-1`` if ``max_nodes`` was reached).
    """
    p = A.shape[0] - 1
    best = np.full(p + 1, np.inf)
    masks = np.zeros((p + 1, p), dtype=np.bool_)
    # greedy backward pass: a good incumbent for every size
    M = A.copy()
    inside = np.ones(p, dtype=np.bool_)
    best[p] = M[p, p]
    masks[p, :] = True
    for size in range(p - 1, -1, -1):
        choice = -1
        smallest = np.inf
        for j in range(p):
            if inside[j]:
                rise = M[j, p] * M[j, p] / (-M[j, j])
                if rise < smallest:
                    smallest = rise
                    choice = j
        _sweep_out(M, choice)
        inside[choice] = False
        best[size] = M[p, p]
        masks[size, :] = inside

    stack = np.zeros((p + 1, p + 1, p + 1))
    start = np.zeros(p + 1, dtype=np.int64)
    current = np.ones((p + 1, p), dtype=np.bool_)
    stack[0] = A
    depth = 0
    start[0] = 0
    nodes = 0
    while depth >= 0:
        j = start[depth]
        if j >= p:
            depth -= 1
            continue
        start[depth] = j + 1
        size = p - depth  # predictors in the model at this depth
        rss_here = stack[depth, p, p]
        # sizes still reachable by deleting predictors with index >= j
        low = size - (p - j)
        prune = True
        for k in range(max(low, 0), size):
            if rss_here < best[k]:
                prune = False
                break
        if prune:
            depth -= 1
            continue
        nodes += 1
        if nodes > max_nodes:
            return best, masks, -1
        stack[depth + 1] = stack[depth]
        _sweep_out(stack[depth + 1], j)
        current[depth + 1] = current[depth]
        current[depth + 1, j] = False
        rss = stack[depth + 1, p, p]
        if rss < best[size - 1]:
            best[size - 1] = rss
            masks[size - 1] = current[depth + 1]
        depth += 1
        start[depth] = j + 1
    return best, masks, nodes
