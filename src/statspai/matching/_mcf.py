"""Minimum-cost flow for matching one sample according to two criteria.

The network of Zhang et al. [@zhang2023matching] has four layers::

    treated --pair--> control --use--> control copy --balance--> treated copy

Each treated unit sends ``ratio`` units of flow, each copy ``treated'``
absorbs ``ratio``, and every arc carries at most one unit. A feasible
integer flow picks ``ratio`` distinct controls for each treated unit (the
arcs of the first layer, which define the matched sets) and at the same
time assigns the chosen controls a second time to the treated units (the
arcs of the third layer). The second assignment is free to differ from the
first, so its cost measures how well the control group *as a whole* can be
lined up with the treated group: a fine-balance penalty on the third layer
is zero exactly when the marginal distributions agree. An optional arc
``treated -> treated'`` lets a treated unit leave the match at a price.

The solver is successive shortest augmenting paths with node potentials
(Dijkstra on reduced costs). All costs are non-negative, so zero
potentials are a valid start, and the flow is optimal for the costs as
given: they are not rounded to integers.
"""

from __future__ import annotations

import heapq
from typing import Any, Tuple

import numpy as np

try:  # numba is a core dependency; the fallback keeps imports safe.
    from numba import njit  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - numba missing

    def njit(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        def wrap(fn: Any) -> Any:
            return fn

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return wrap


@njit(cache=True)
def _solve(
    pair: np.ndarray,
    balance: np.ndarray,
    use: np.ndarray,
    skip: np.ndarray,
    ratio: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return ``(owner, owner2, skipped, status)``.

    ``owner[c]`` is the treated unit matched to control ``c`` (``-1`` if
    unused), ``owner2[c]`` the treated copy it is assigned to on the
    balance side, ``skipped[t]`` whether treated unit ``t`` left the match.
    ``status`` is ``0`` on success and ``1`` when the supply cannot be
    routed. ``skip[t]`` is the price of leaving ``t`` out, ``inf`` to
    forbid it.
    """
    n_t, n_c = pair.shape
    source = 0
    first_t, first_c = 1, 1 + n_t
    first_d, first_e = 1 + n_t + n_c, 1 + n_t + 2 * n_c
    sink = 1 + 2 * n_t + 2 * n_c
    n_nodes = sink + 1

    owner = np.full(n_c, -1, np.int64)
    owner2 = np.full(n_c, -1, np.int64)
    used = np.zeros(n_c, np.bool_)
    skipped = np.zeros(n_t, np.bool_)
    supply = np.full(n_t, ratio, np.int64)
    demand = np.full(n_t, ratio, np.int64)
    pot = np.zeros(n_nodes)
    dist = np.empty(n_nodes)
    prev = np.empty(n_nodes, np.int64)
    done = np.empty(n_nodes, np.bool_)

    for _ in range(n_t * ratio):
        dist[:] = np.inf
        prev[:] = -1
        done[:] = False
        dist[source] = 0.0
        heap = [(0.0, source)]
        reached = False
        while len(heap) > 0:
            d, v = heapq.heappop(heap)
            if done[v]:
                continue
            done[v] = True
            if v == sink:
                reached = True
                break
            base = d + pot[v]
            if v == source:
                for t in range(n_t):
                    if supply[t] > 0:
                        nd = base - pot[first_t + t]
                        if nd < d:
                            nd = d
                        if nd < dist[first_t + t]:
                            dist[first_t + t] = nd
                            prev[first_t + t] = v
                            heapq.heappush(heap, (nd, first_t + t))
            elif v < first_c:  # treated
                t = v - first_t
                for c in range(n_c):
                    if owner[c] != t:
                        u = first_c + c
                        if not done[u]:
                            nd = base + pair[t, c] - pot[u]
                            if nd < d:
                                nd = d
                            if nd < dist[u]:
                                dist[u] = nd
                                prev[u] = v
                                heapq.heappush(heap, (nd, u))
                if not skipped[t] and skip[t] < np.inf:
                    u = first_e + t
                    nd = base + skip[t] - pot[u]
                    if nd < d:
                        nd = d
                    if nd < dist[u]:
                        dist[u] = nd
                        prev[u] = v
                        heapq.heappush(heap, (nd, u))
            elif v < first_d:  # control
                c = v - first_c
                if not used[c]:
                    u = first_d + c
                    nd = base + use[c] - pot[u]
                    if nd < d:
                        nd = d
                    if nd < dist[u]:
                        dist[u] = nd
                        prev[u] = v
                        heapq.heappush(heap, (nd, u))
                if owner[c] >= 0:
                    u = first_t + owner[c]
                    nd = base - pair[owner[c], c] - pot[u]
                    if nd < d:
                        nd = d
                    if nd < dist[u]:
                        dist[u] = nd
                        prev[u] = v
                        heapq.heappush(heap, (nd, u))
            elif v < first_e:  # control copy
                c = v - first_d
                for t in range(n_t):
                    if owner2[c] != t:
                        u = first_e + t
                        if not done[u]:
                            nd = base + balance[t, c] - pot[u]
                            if nd < d:
                                nd = d
                            if nd < dist[u]:
                                dist[u] = nd
                                prev[u] = v
                                heapq.heappush(heap, (nd, u))
                if used[c]:
                    u = first_c + c
                    nd = base - use[c] - pot[u]
                    if nd < d:
                        nd = d
                    if nd < dist[u]:
                        dist[u] = nd
                        prev[u] = v
                        heapq.heappush(heap, (nd, u))
            else:  # treated copy
                t = v - first_e
                if demand[t] > 0:
                    nd = base - pot[sink]
                    if nd < d:
                        nd = d
                    if nd < dist[sink]:
                        dist[sink] = nd
                        prev[sink] = v
                        heapq.heappush(heap, (nd, sink))
                for c in range(n_c):
                    if owner2[c] == t:
                        u = first_d + c
                        if not done[u]:
                            nd = base - balance[t, c] - pot[u]
                            if nd < d:
                                nd = d
                            if nd < dist[u]:
                                dist[u] = nd
                                prev[u] = v
                                heapq.heappush(heap, (nd, u))
                if skipped[t]:
                    u = first_t + t
                    nd = base - skip[t] - pot[u]
                    if nd < d:
                        nd = d
                    if nd < dist[u]:
                        dist[u] = nd
                        prev[u] = v
                        heapq.heappush(heap, (nd, u))
        if not reached:
            return owner, owner2, skipped, 1

        far = dist[sink]
        for v in range(n_nodes):
            pot[v] += dist[v] if dist[v] < far else far

        # Walk the path back from the sink and push one unit along it.
        v = sink
        while v != source:
            p = prev[v]
            if v == sink:
                demand[p - first_e] -= 1
            elif p == source:
                supply[v - first_t] -= 1
            elif p < first_c:  # leaving a treated node
                t = p - first_t
                if v >= first_e:
                    skipped[t] = True
                else:
                    owner[v - first_c] = t
            elif p < first_d:  # leaving a control
                c = p - first_c
                if v >= first_d:
                    used[c] = True
                elif owner[c] == v - first_t:
                    owner[c] = -1
            elif p < first_e:  # leaving a control copy
                c = p - first_d
                if v >= first_e:
                    owner2[c] = v - first_e
                else:
                    used[c] = False
            else:  # leaving a treated copy
                t = p - first_e
                if v >= first_d:
                    c = v - first_d
                    if owner2[c] == t:
                        owner2[c] = -1
                else:
                    skipped[t] = False
            v = p
    return owner, owner2, skipped, 0


def solve_two_criteria(
    pair: np.ndarray,
    balance: np.ndarray,
    use: np.ndarray,
    skip: np.ndarray,
    ratio: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Solve the network; the last value says whether it was feasible."""
    owner, owner2, skipped, status = _solve(
        np.ascontiguousarray(pair, dtype=np.float64),
        np.ascontiguousarray(balance, dtype=np.float64),
        np.ascontiguousarray(use, dtype=np.float64),
        np.ascontiguousarray(skip, dtype=np.float64),
        int(ratio),
    )
    return owner, owner2, skipped, status == 0


__all__ = ["solve_two_criteria"]
