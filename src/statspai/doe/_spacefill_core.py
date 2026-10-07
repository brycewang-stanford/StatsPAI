"""Compiled simulated annealing over Latin hypercubes.

Imported on the first call of ``sp.space_filling`` with a searched method,
so that ``import statspai`` does not import numba.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit

MAXPRO, MAXIMIN, UNIFORM = 0, 1, 2


@njit(cache=True)
def _piece(kind: int, diff: float, delta: float) -> float:
    if kind == MAXPRO:
        return float(-np.log(diff * diff + delta))
    if kind == MAXIMIN:
        return diff * diff
    a = abs(diff)
    return float(np.log(1.5 - a * (1.0 - a)))


@njit(cache=True)
def _term(kind: int, state: float, r: float, shift: float) -> float:
    if kind == MAXIMIN:
        return float(np.exp(-0.5 * r * np.log(state) - shift))
    return float(np.exp(state - shift))


@njit(cache=True)
def anneal(
    X0: np.ndarray,
    kind: int,
    r: float,
    delta: float,
    shift: float,
    iterations: int,
    seed: int,
    offset: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Swap two levels within a column; Metropolis rule on the log criterion.

    The state holds, for every pair of runs, a quantity additive over the
    factors (log pair terms, or the squared distance), so a swap updates
    two rows at a cost linear in the number of runs. ``offset`` is added
    to the state of every pair: the fixed contribution of factors that the
    search does not move (qualitative factors).
    """
    np.random.seed(seed)
    n, p = X0.shape
    X = X0.copy()
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    cur = 0.0
    for i in range(n):
        for m in range(i + 1, n):
            s = offset[i, m]
            for k in range(p):
                s += _piece(kind, X[i, k] - X[m, k], delta)
            S[i, m] = s
            S[m, i] = s
            t = _term(kind, s, r, shift)
            T[i, m] = t
            T[m, i] = t
            cur += t
    best = cur
    best_X = X.copy()
    if n < 3 or iterations <= 0:
        return best_X, best
    si = np.zeros(n)
    sj = np.zeros(n)
    ti = np.zeros(n)
    tj = np.zeros(n)
    temp = 0.0
    cool = 1e-3 ** (1.0 / max(iterations - 1, 1))
    warm = 64
    trial = np.zeros(warm)
    n_trial = 0
    for q in range(-warm, iterations):
        i = np.random.randint(n)
        j = (i + 1 + np.random.randint(n - 1)) % n
        k = np.random.randint(p)
        xi = X[i, k]
        xj = X[j, k]
        d = 0.0
        for m in range(n):
            if m == i or m == j:
                continue
            change = _piece(kind, xj - X[m, k], delta) - _piece(
                kind, xi - X[m, k], delta
            )
            a = S[i, m] + change
            b = S[j, m] - change
            si[m] = a
            sj[m] = b
            ta = _term(kind, a, r, shift)
            tb = _term(kind, b, r, shift)
            ti[m] = ta
            tj[m] = tb
            d += ta + tb - T[i, m] - T[j, m]
        new = cur + d
        if not (new > 0.0 and np.isfinite(new)):
            continue
        dlog = np.log(new / cur)
        if q < 0:
            # warm-up: the size of typical moves sets the temperature
            trial[n_trial] = abs(dlog)
            n_trial += 1
            if q == -1:
                temp = float(max(0.5 * np.median(trial[:n_trial]), 1e-10))
            continue
        if dlog <= 0.0 or np.random.random() < np.exp(-dlog / temp):
            X[i, k] = xj
            X[j, k] = xi
            for m in range(n):
                if m == i or m == j:
                    continue
                S[i, m] = si[m]
                S[m, i] = si[m]
                S[j, m] = sj[m]
                S[m, j] = sj[m]
                T[i, m] = ti[m]
                T[m, i] = ti[m]
                T[j, m] = tj[m]
                T[m, j] = tj[m]
            cur = new
            if cur < best:
                best = cur
                best_X = X.copy()
        temp *= cool
    return best_X, best
