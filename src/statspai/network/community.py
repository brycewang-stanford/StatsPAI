"""Community detection for ``sp.network``.

Modularity, the Louvain method, greedy modularity (Clauset-Newman-Moore),
and label propagation — exposed through the :func:`community_detection`
dispatcher (``sp.community_detection(g, method=...)``), mirroring the
"family methods use a dispatcher" convention of ``sp.synth`` / ``sp.dml``.

References
----------
Newman, M. E. J. (2006). "Modularity and community structure in networks."
*PNAS*, 103(23), 8577-8582. [@newman2006modularity]

Blondel, V. D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E. (2008).
"Fast unfolding of communities in large networks." *Journal of Statistical
Mechanics*, P10008. [@blondel2008fast]

Clauset, A., Newman, M. E. J. & Moore, C. (2004). "Finding community
structure in very large networks." *Physical Review E*, 70, 066111.
[@clauset2004finding]

Raghavan, U. N., Albert, R. & Kumara, S. (2007). "Near linear time
algorithm to detect community structures in large-scale networks."
*Physical Review E*, 76, 036106. [@raghavan2007near]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ._core import as_graph

__all__ = [
    "community_detection",
    "CommunityResult",
    "modularity",
]


# ====================================================================== #
#  Modularity
# ====================================================================== #


def _as_membership_array(membership: Any, labels: List[Any]) -> np.ndarray:
    """Coerce a membership spec to a contiguous integer label array."""
    if isinstance(membership, pd.Series):
        membership = membership.reindex(labels).to_numpy()
    arr = np.asarray(list(membership))
    if arr.shape[0] != len(labels):
        raise ValueError(f"membership length {arr.shape[0]} != n_nodes {len(labels)}")
    # map arbitrary labels to 0..k-1
    _, inv = np.unique(arr, return_inverse=True)
    return np.asarray(inv, dtype=int)


def modularity(graph: Any, membership: Any, resolution: float = 1.0) -> float:
    """Newman-Girvan modularity ``Q`` of a partition.

    ``Q = (1/2m) sum_ij (A_ij - gamma k_i k_j / 2m) delta(c_i, c_j)`` with
    weighted degrees ``k`` and total edge weight ``m``.  ``Q`` rises as ties
    concentrate within communities relative to a degree-preserving null
    model; ``resolution`` (``gamma``) tunes community size.

    Parameters
    ----------
    graph : Graph or adjacency-like
    membership : Series, sequence, or mapping
        Community label per node (any hashable labels).
    resolution : float, default 1.0

    Returns
    -------
    float

    References
    ----------
    newman2006modularity

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.karate_club()
    >>> # Zachary's observed factional split:
    >>> faction = [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,
    ...            1,1,1,1,1,1,1,1,1,1]
    >>> round(sp.network_modularity(g, faction), 4)  # Newman's Q for the split
    0.3715
    """
    g = as_graph(graph)
    A = g.adjacency_matrix() if g.is_weighted else g.binary()
    c = _as_membership_array(membership, g.labels)
    m2 = A.sum()  # 2m for undirected
    if m2 <= 0:
        return 0.0
    k = A.sum(axis=1)
    Q = 0.0
    for com in np.unique(c):
        idx = np.flatnonzero(c == com)
        a_in = A[np.ix_(idx, idx)].sum()
        k_tot = k[idx].sum()
        Q += a_in / m2 - resolution * (k_tot / m2) ** 2
    return float(Q)


# ====================================================================== #
#  Louvain (Blondel et al. 2008)
# ====================================================================== #


def _louvain_one_level(
    A: np.ndarray, resolution: float, order: np.ndarray
) -> np.ndarray:
    """One Louvain level: local moving until no node improves Q."""
    n = A.shape[0]
    m2 = A.sum()
    if m2 <= 0:
        return np.arange(n)
    k = A.sum(axis=1)
    comm = np.arange(n)
    # Sigma_tot[c] = total degree of community c
    sigma_tot = k.copy()

    improved = True
    while improved:
        improved = False
        for i in order:
            ci = comm[i]
            ki = k[i]
            # weight from i to each community (excluding self-loop)
            row = A[i].copy()
            row[i] = 0.0
            # remove i from its community
            sigma_tot[ci] -= ki
            # neighbour-community weights
            neigh = np.flatnonzero(row > 0)
            wic: Dict[int, float] = {}
            for j in neigh:
                wic[comm[j]] = wic.get(comm[j], 0.0) + row[j]
            # candidate communities include current (weight 0 if none)
            best_c = ci
            best_gain = wic.get(ci, 0.0) - resolution * sigma_tot[ci] * ki / m2
            for c, w_in in wic.items():
                gain = w_in - resolution * sigma_tot[c] * ki / m2
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_c = c
            comm[i] = best_c
            sigma_tot[best_c] += ki
            if best_c != ci:
                improved = True
    # relabel 0..k-1
    _, comm = np.unique(comm, return_inverse=True)
    return comm


def _aggregate(A: np.ndarray, comm: np.ndarray) -> np.ndarray:
    """Aggregate nodes within each community into a super-node graph."""
    ncomm = comm.max() + 1
    # B = membership indicator (n x ncomm)
    B = np.zeros((A.shape[0], ncomm))
    B[np.arange(A.shape[0]), comm] = 1.0
    return np.asarray(B.T @ A @ B, dtype=float)


def _louvain(
    A: np.ndarray, resolution: float, rng: Optional[np.random.Generator]
) -> np.ndarray:
    """Full Louvain: iterate level passes until modularity stops improving."""
    n = A.shape[0]
    node_comm = np.arange(n)  # final membership in original node space
    cur = A.copy()
    while True:
        order = np.arange(cur.shape[0])
        if rng is not None:
            rng.shuffle(order)
        comm = _louvain_one_level(cur, resolution, order)
        if comm.max() + 1 == cur.shape[0]:
            break  # no node moved -> converged
        # propagate super-node membership down to original nodes
        node_comm = comm[node_comm]
        nxt = _aggregate(cur, comm)
        if nxt.shape[0] == cur.shape[0]:
            break
        cur = nxt
    _, node_comm = np.unique(node_comm, return_inverse=True)
    return node_comm


# ====================================================================== #
#  Greedy modularity (Clauset-Newman-Moore 2004)
# ====================================================================== #


def _greedy_modularity(A: np.ndarray, resolution: float) -> np.ndarray:
    """Agglomerative modularity maximisation (CNM)."""
    n = A.shape[0]
    m2 = A.sum()
    if m2 <= 0:
        return np.arange(n)
    members: Dict[int, set] = {i: {i} for i in range(n)}
    k = A.sum(axis=1)
    # community-degree and inter-community weight bookkeeping on dense A
    active = set(range(n))

    def merged_membership() -> np.ndarray:
        out = np.empty(n, dtype=int)
        for c, mem in members.items():
            for node in mem:
                out[node] = c
        _, inv = np.unique(out, return_inverse=True)
        return inv

    best_member_arr = merged_membership()
    best_Q = modularity_from_array(A, best_member_arr, resolution, m2, k)

    while len(active) > 1:
        # find merge with max delta-Q
        best_pair = None
        best_dq = 0.0
        act = sorted(active)
        for a_i in range(len(act)):
            ca = act[a_i]
            ia = np.array(sorted(members[ca]))
            for b_i in range(a_i + 1, len(act)):
                cb = act[b_i]
                ib = np.array(sorted(members[cb]))
                e_ab = A[np.ix_(ia, ib)].sum()  # undirected: this is one side
                if e_ab == 0:
                    continue
                ka = k[ia].sum()
                kb = k[ib].sum()
                dq = 2.0 * (e_ab / m2 - resolution * ka * kb / (m2 * m2))
                if dq > best_dq + 1e-15:
                    best_dq = dq
                    best_pair = (ca, cb)
        if best_pair is None:
            break
        ca, cb = best_pair
        members[ca] |= members[cb]
        del members[cb]
        active.discard(cb)
        cur_arr = merged_membership()
        cur_Q = best_Q + best_dq
        if cur_Q > best_Q:
            best_Q = cur_Q
            best_member_arr = cur_arr
    return best_member_arr


def modularity_from_array(
    A: np.ndarray,
    c: np.ndarray,
    resolution: float,
    m2: float,
    k: np.ndarray,
) -> float:
    Q = 0.0
    for com in np.unique(c):
        idx = np.flatnonzero(c == com)
        Q += A[np.ix_(idx, idx)].sum() / m2 - resolution * (k[idx].sum() / m2) ** 2
    return float(Q)


# ====================================================================== #
#  Label propagation (Raghavan et al. 2007)
# ====================================================================== #


def _label_propagation(
    A: np.ndarray, rng: np.random.Generator, max_iter: int = 1000
) -> np.ndarray:
    n = A.shape[0]
    labels = np.arange(n)
    for _ in range(max_iter):
        order = np.arange(n)
        rng.shuffle(order)
        changed = False
        for i in order:
            neigh = np.flatnonzero(A[i] > 0)
            if neigh.size == 0:
                continue
            # weighted vote
            votes: Dict[int, float] = {}
            for j in neigh:
                votes[labels[j]] = votes.get(labels[j], 0.0) + A[i, j]
            best = max(votes.values())
            winners = sorted(c for c, v in votes.items() if v >= best - 1e-12)
            new = (
                winners[rng.integers(len(winners))] if len(winners) > 1 else winners[0]
            )
            if new != labels[i]:
                labels[i] = new
                changed = True
        if not changed:
            break
    _, labels = np.unique(labels, return_inverse=True)
    return labels


# ====================================================================== #
#  Result + dispatcher
# ====================================================================== #


@dataclass
class CommunityResult(ResultProtocolMixin):
    """Community-detection partition (the ``sp.community_detection`` output).

    Attributes
    ----------
    membership : pandas.Series
        Community id per node, indexed by node label.
    n_communities : int
    modularity : float
        Newman modularity ``Q`` of the partition.
    method : str
    sizes : list of int
        Community sizes, descending.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.community_detection(sp.karate_club(), method="louvain")
    >>> res.n_communities >= 2
    True
    >>> res.modularity > 0.38
    True
    """

    _citation_keys = ("newman2006modularity", "blondel2008fast")

    membership: pd.Series
    n_communities: int
    modularity: float
    method: str
    sizes: List[int] = field(default_factory=list)

    def summary(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Community detection ({self.method})\n"
            f"  communities : {self.n_communities}\n"
            f"  modularity  : {self.modularity:.4f}\n"
            f"  sizes       : {self.sizes[:12]}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CommunityResult(method={self.method!r}, "
            f"n_communities={self.n_communities}, Q={self.modularity:.4f})"
        )


def community_detection(
    graph: Any,
    method: str = "louvain",
    resolution: float = 1.0,
    seed: Optional[int] = None,
) -> CommunityResult:
    """Partition a network into communities (family dispatcher).

    Parameters
    ----------
    graph : Graph or adjacency-like
    method : {"louvain", "greedy", "label_prop"}, default "louvain"
        ``"louvain"`` — multi-level modularity optimisation (Blondel 2008);
        ``"greedy"`` — agglomerative modularity (Clauset-Newman-Moore 2004);
        ``"label_prop"`` — label propagation (Raghavan 2007).
    resolution : float, default 1.0
        Modularity resolution ``gamma`` (Louvain / greedy).  Higher values
        yield more, smaller communities.
    seed : int, optional
        Random seed.  Louvain visits nodes in fixed order when ``seed`` is
        ``None`` (deterministic); label propagation always needs a seed for
        its random tie-breaking (defaults to 0).

    Returns
    -------
    CommunityResult

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.karate_club()
    >>> sp.community_detection(g, method="greedy").n_communities
    3
    """
    g = as_graph(graph)
    A = g.adjacency_matrix() if g.is_weighted else g.binary()
    method = method.lower()
    if method == "louvain":
        rng = np.random.default_rng(seed) if seed is not None else None
        comm = _louvain(A, resolution, rng)
    elif method == "greedy":
        comm = _greedy_modularity(A, resolution)
    elif method in ("label_prop", "label_propagation", "lpa"):
        rng = np.random.default_rng(0 if seed is None else seed)
        comm = _label_propagation(A, rng)
    else:
        raise ValueError(
            f"unknown method {method!r}; choose 'louvain', 'greedy', or "
            f"'label_prop'"
        )
    membership = pd.Series(comm, index=g.labels, name="community")
    _, counts = np.unique(comm, return_counts=True)
    sizes = sorted((int(c) for c in counts), reverse=True)
    return CommunityResult(
        membership=membership,
        n_communities=int(comm.max() + 1) if len(comm) else 0,
        modularity=modularity(g, membership, resolution=resolution),
        method=method,
        sizes=sizes,
    )
