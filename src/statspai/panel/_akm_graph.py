"""Connected sets of a worker-firm panel (``sp.akm``).

The observations are the edges of a bipartite multigraph whose nodes are
workers and firms. Worker and firm effects are identified, up to one
constant per connected component, exactly on the rows of a component.

Leave-one-out connectivity (Kline, Saggio and Solvsten 2020) asks more:
every observation can be dropped without losing identification, i.e. its
leverage ``P_ii`` is below one. Dropping observation ``i`` deletes one
edge of the multigraph, so ``P_ii < 1`` exactly when that edge is not a
*bridge*. A worker seen once is a pendant edge (a bridge); an edge that a
worker-firm pair carries more than once never is. Deleting all bridges
leaves the 2-edge-connected components, which have no bridges of their
own, so one pass suffices: the leave-one-out set is the largest of them.

Stayers (one firm) are leaves hanging off their firm and cannot change
whether any other edge is a bridge, so the bridge search runs on the
movers' edges only; a stayer's rows are bridges only when the stayer has
a single row.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def components(wc: np.ndarray, fc: np.ndarray, n_w: int, n_f: int) -> np.ndarray:
    """Connected component of every row (labels 0..C-1)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = n_w + n_f
    adj = coo_matrix((np.ones(len(wc)), (wc, fc + n_w)), shape=(n, n)).tocsr()
    _, lab = connected_components(adj, directed=False)
    return lab[wc]


def largest(comp: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Rows of the component with the most firms (ties: most rows, then the
    lowest label)."""
    if comp.size == 0:
        return np.zeros(0, dtype=bool)
    labels, inv = np.unique(comp, return_inverse=True)
    n_rows = np.bincount(inv)
    pairs = np.unique(np.column_stack([inv, fc]), axis=0)
    n_firms = np.bincount(pairs[:, 0], minlength=len(labels))
    best = np.lexsort((-np.arange(len(labels)), n_rows, n_firms))[-1]
    return inv == best


def _bridges(u: np.ndarray, v: np.ndarray, n_nodes: int) -> np.ndarray:
    """Bridge flag of each edge ``(u[e], v[e])`` of a simple undirected
    graph (iterative Tarjan; parallel edges must be merged beforehand)."""
    m = len(u)
    is_bridge = np.zeros(m, dtype=bool)
    if m == 0:
        return is_bridge
    # adjacency in CSR form, each edge listed from both ends
    ends = np.concatenate([u, v])
    nbrs = np.concatenate([v, u])
    eids = np.concatenate([np.arange(m), np.arange(m)])
    order = np.argsort(ends, kind="stable")
    nbr = nbrs[order]
    eid = eids[order]
    start = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(ends, minlength=n_nodes), out=start[1:])
    start_l = start.tolist()
    nbr_l = nbr.tolist()
    eid_l = eid.tolist()
    disc = [-1] * n_nodes
    low = [0] * n_nodes
    ptr = start_l[:-1]
    timer = 0
    for root in range(n_nodes):
        if disc[root] != -1 or start_l[root] == start_l[root + 1]:
            continue
        disc[root] = low[root] = timer
        timer += 1
        stack = [(root, -1)]  # (node, edge id used to enter it)
        while stack:
            node, via = stack[-1]
            p = ptr[node]
            if p < start_l[node + 1]:
                ptr[node] = p + 1
                e = eid_l[p]
                if e == via:
                    continue
                nxt = nbr_l[p]
                if disc[nxt] == -1:
                    disc[nxt] = low[nxt] = timer
                    timer += 1
                    stack.append((nxt, e))
                elif disc[nxt] < low[node]:
                    low[node] = disc[nxt]
            else:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    if low[node] < low[parent]:
                        low[parent] = low[node]
                    if low[node] > disc[parent]:
                        is_bridge[via] = True
    return is_bridge


def bridge_rows(wc: np.ndarray, fc: np.ndarray, n_w: int, n_f: int) -> np.ndarray:
    """Rows whose removal disconnects their component (``P_ii = 1``)."""
    n = len(wc)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    pair = wc.astype(np.int64) * n_f + fc
    upair, pinv, mult = np.unique(pair, return_inverse=True, return_counts=True)
    pw, pf = upair // n_f, upair % n_f
    n_firms_of_worker = np.bincount(pw, minlength=n_w)
    mover_pair = n_firms_of_worker[pw] > 1
    # stayers: their one pair is a bridge only when it carries one row
    flag = np.zeros(len(upair), dtype=bool)
    flag[~mover_pair] = mult[~mover_pair] == 1
    if mover_pair.any():
        mw, mf = pw[mover_pair], pf[mover_pair]
        wid, wloc = np.unique(mw, return_inverse=True)
        b = _bridges(wloc, mf + len(wid), len(wid) + n_f)
        flag[np.flatnonzero(mover_pair)] = b & (mult[mover_pair] == 1)
    out = flag[pinv]
    return out


def leave_one_out(
    wc: np.ndarray, fc: np.ndarray, n_w: int, n_f: int
) -> Tuple[np.ndarray, int]:
    """Rows of the largest leave-one-observation-out connected set and the
    number of bridge rows removed on the way."""
    bridge = bridge_rows(wc, fc, n_w, n_f)
    keep = ~bridge
    comp = np.full(len(wc), -1, dtype=np.int64)
    if keep.any():
        comp[keep] = components(wc[keep], fc[keep], n_w, n_f)
        sel = np.zeros(len(wc), dtype=bool)
        sel[keep] = largest(comp[keep], fc[keep])
    else:
        sel = keep
    return sel, int(bridge.sum())


__all__ = ["bridge_rows", "components", "largest", "leave_one_out"]
