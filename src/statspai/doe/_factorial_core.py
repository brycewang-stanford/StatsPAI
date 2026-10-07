"""Compiled search for minimum-aberration two-level fractions.

Imported on demand by ``sp.factorial_design`` when the number of generator
choices is large, so that ``import statspai`` does not import numba.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit


@njit(cache=True)
def _popcount(v: int) -> int:
    c = 0
    while v:
        v &= v - 1
        c += 1
    return c


@njit(cache=True)
def search(
    cand: np.ndarray, m: int, base: int, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Best ``m`` of the candidate interaction columns, by word length pattern.

    Every combination is visited; for each, the ``2^m - 1`` defining words
    are generated in Gray-code order and counted by length.
    """
    n = cand.shape[0]
    idx = np.arange(m)
    best = np.full(k + 1, 1 << 60, dtype=np.int64)
    best_idx = idx.copy()
    words = np.zeros(m, dtype=np.int64)
    counts = np.zeros(k + 1, dtype=np.int64)
    total = 1 << m
    while True:
        for j in range(m):
            words[j] = cand[idx[j]] | (1 << (base + j))
        counts[:] = 0
        w = 0
        for g in range(1, total):
            # Gray code: flip the generator at the lowest set bit of g
            low = 0
            t = g
            while t & 1 == 0:
                t >>= 1
                low += 1
            w ^= words[low]
            counts[_popcount(w)] += 1
        better = False
        for j in range(1, k + 1):
            if counts[j] < best[j]:
                better = True
                break
            if counts[j] > best[j]:
                break
        if better:
            best[:] = counts
            best_idx[:] = idx
        # next combination
        i = m - 1
        while i >= 0 and idx[i] == n - m + i:
            i -= 1
        if i < 0:
            break
        idx[i] += 1
        for j in range(i + 1, m):
            idx[j] = idx[j - 1] + 1
    return best_idx, best[1:]
