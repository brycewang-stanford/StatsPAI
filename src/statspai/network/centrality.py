"""Centrality measures for ``sp.network``.

Degree, closeness, betweenness (Brandes), eigenvector, Katz, PageRank,
Bonacich power, and HITS hub/authority scores — the centrality battery of
R's ``igraph`` / ``sna`` and Stata's ``nwcommands``, exposed through a
single :func:`centrality` dispatcher (``sp.centrality(g, kind=...)``) plus
named convenience functions.

All measures operate on a :class:`~statspai.network.Graph` (or any
adjacency-like object) and return a :class:`pandas.Series` indexed by node
label, so per-node scores carry their identity for agents and humans alike.

References
----------
Freeman, L. C. (1978). "Centrality in social networks: conceptual
clarification." *Social Networks*, 1(3), 215-239. [@freeman1978centrality]

Brandes, U. (2001). "A faster algorithm for betweenness centrality."
*Journal of Mathematical Sociology*, 25(2), 163-177. [@brandes2001faster]

Bonacich, P. (1987). "Power and centrality: a family of measures."
*American Journal of Sociology*, 92(5), 1170-1182. [@bonacich1987power]

Katz, L. (1953). "A new status index derived from sociometric analysis."
*Psychometrika*, 18(1), 39-43. [@katz1953new]

Brin, S. & Page, L. (1998). "The anatomy of a large-scale hypertextual web
search engine." *Computer Networks and ISDN Systems*, 30(1-7), 107-117.
[@brin1998anatomy]

Kleinberg, J. M. (1999). "Authoritative sources in a hyperlinked
environment." *Journal of the ACM*, 46(5), 604-632.
[@kleinberg1999authoritative]
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ._core import as_graph

__all__ = [
    "centrality",
    "CentralityResult",
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
    "katz_centrality",
    "pagerank",
    "bonacich_power",
    "hits",
]


# ====================================================================== #
#  Degree
# ====================================================================== #


def degree_centrality(
    graph: Any, mode: str = "all", normalized: bool = True
) -> pd.Series:
    """Degree centrality.

    Parameters
    ----------
    graph : Graph or adjacency-like
    mode : {"all", "in", "out"}, default "all"
    normalized : bool, default True
        Divide by ``n - 1`` (the maximum possible degree).

    Returns
    -------
    pandas.Series

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (1, 2), (1, 3)])
    >>> sp.degree_centrality(g, normalized=False).loc[1]
    3.0
    """
    g = as_graph(graph)
    d: np.ndarray = g.degree(mode=mode).astype(float)
    if normalized and g.n_nodes > 1:
        d = d / (g.n_nodes - 1)
    return pd.Series(d, index=g.labels, name="degree")


# ====================================================================== #
#  Closeness (Wasserman-Faust improved, matches igraph/networkx default)
# ====================================================================== #


def closeness_centrality(graph: Any, weighted: Optional[bool] = None) -> pd.Series:
    """Closeness centrality with the Wasserman-Faust disconnected correction.

    For node ``i`` reaching ``r`` others with total distance ``T``:
    ``C(i) = (r / T) * (r / (n - 1))``.  On a connected graph this reduces to
    ``(n - 1) / T``.

    Parameters
    ----------
    graph : Graph or adjacency-like
        Network to score.
    weighted : bool, optional
        Use tie weights as distances. Defaults to the graph's weighted flag.

    Returns
    -------
    pandas.Series

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (1, 2), (2, 3)])
    >>> sp.closeness_centrality(g).idxmax()
    1
    """
    from ._core import shortest_path_lengths

    g = as_graph(graph)
    n = g.n_nodes
    D = shortest_path_lengths(g, weighted=weighted)
    out = np.zeros(n)
    for i in range(n):
        di = D[i]
        mask = np.isfinite(di) & (np.arange(n) != i)
        tot = di[mask].sum()
        r = int(mask.sum())
        if tot > 0 and n > 1:
            out[i] = (r / tot) * (r / (n - 1))
    return pd.Series(out, index=g.labels, name="closeness")


# ====================================================================== #
#  Betweenness (Brandes 2001)
# ====================================================================== #


def _brandes(A: np.ndarray, directed: bool, weighted: bool) -> np.ndarray:
    """Brandes betweenness accumulation on a dense adjacency."""
    n = A.shape[0]
    CB = np.zeros(n)
    # Out-neighbour lists (and weights for the weighted variant).
    nbrs: List[np.ndarray] = [np.flatnonzero(A[v] != 0) for v in range(n)]
    for s in range(n):
        S: List[int] = []
        P: List[List[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1.0
        if not weighted:
            d = np.full(n, -1.0)
            d[s] = 0.0
            Q: deque = deque([s])
            while Q:
                v = Q.popleft()
                S.append(v)
                dv = d[v]
                for w in nbrs[v]:
                    if d[w] < 0:
                        Q.append(w)
                        d[w] = dv + 1
                    if d[w] == dv + 1:
                        sigma[w] += sigma[v]
                        P[w].append(v)
        else:
            dist = np.full(n, np.inf)
            seen = {s: 0.0}
            pq: List = [(0.0, s, s)]
            while pq:
                dv, pred, v = heapq.heappop(pq)
                if np.isfinite(dist[v]):
                    continue
                dist[v] = dv
                S.append(v)
                if pred != v:
                    pass
                for w in nbrs[v]:
                    cost = dv + A[v, w]
                    if w not in seen or cost < seen[w] - 1e-15:
                        seen[w] = cost
                        heapq.heappush(pq, (cost, v, w))
                        sigma[w] = sigma[v]
                        P[w] = [v]
                    elif abs(cost - seen[w]) <= 1e-15:
                        sigma[w] += sigma[v]
                        P[w].append(v)
        delta = np.zeros(n)
        for w in reversed(S):
            coeff = (1.0 + delta[w]) / sigma[w] if sigma[w] > 0 else 0.0
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                CB[w] += delta[w]
    if not directed:
        CB = CB / 2.0
    return CB


def betweenness_centrality(
    graph: Any, normalized: bool = True, weighted: Optional[bool] = None
) -> pd.Series:
    """Shortest-path betweenness centrality (Brandes 2001).

    Parameters
    ----------
    graph : Graph or adjacency-like
    normalized : bool, default True
        Scale by ``1/((n-1)(n-2))`` (directed) or ``2/((n-1)(n-2))``
        (undirected), so scores lie in ``[0, 1]``.
    weighted : bool, optional
        Use tie weights as path lengths (Dijkstra).  Defaults to the graph's
        own weighted flag.

    Returns
    -------
    pandas.Series

    References
    ----------
    brandes2001faster

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (1, 2), (2, 3)])
    >>> sp.betweenness_centrality(g, normalized=False).loc[1]
    2.0
    """
    g = as_graph(graph)
    n = g.n_nodes
    use_w = g.is_weighted if weighted is None else weighted
    A = g.adjacency_matrix() if use_w else g.binary()
    CB = _brandes(A, directed=g.is_directed, weighted=use_w)
    if normalized and n > 2:
        scale = 1.0 / ((n - 1) * (n - 2))
        if not g.is_directed:
            scale *= 2.0
        CB = CB * scale
    return pd.Series(CB, index=g.labels, name="betweenness")


# ====================================================================== #
#  Eigenvector
# ====================================================================== #


def eigenvector_centrality(
    graph: Any,
    weighted: Optional[bool] = None,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> pd.Series:
    """Eigenvector centrality (leading eigenvector of the adjacency matrix).

    Computed by power iteration and L2-normalised, matching the
    ``igraph`` / ``networkx`` convention.  For directed graphs the right
    eigenvector is used (centrality flows along out-ties' reverse).

    Parameters
    ----------
    graph : Graph or adjacency-like
        Network to score.
    weighted : bool, optional
        Use tie weights; defaults to the graph's weighted flag.
    max_iter : int, default 1000
        Maximum power-iteration steps.
    tol : float, default 1e-9
        Convergence tolerance on the score vector.

    Returns
    -------
    pandas.Series

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (1, 2), (1, 3)])
    >>> sp.eigenvector_centrality(g).idxmax()
    1
    """
    g = as_graph(graph)
    use_w = g.is_weighted if weighted is None else weighted
    A = g.adjacency_matrix() if use_w else g.binary()
    n = g.n_nodes
    if n == 0:
        return pd.Series([], dtype=float, name="eigenvector")
    x = np.full(n, 1.0 / np.sqrt(n))
    for _ in range(max_iter):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm == 0:
            break
        x_new = x_new / norm
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    # Perron-Frobenius: choose the non-negative orientation.
    if x.sum() < 0:
        x = -x
    return pd.Series(x, index=g.labels, name="eigenvector")


# ====================================================================== #
#  Katz
# ====================================================================== #


def katz_centrality(
    graph: Any,
    alpha: float = 0.1,
    beta: float = 1.0,
    normalized: bool = True,
    weighted: Optional[bool] = None,
) -> pd.Series:
    """Katz centrality.

    ``x = beta (I - alpha A^T)^{-1} 1`` — every node gets a base score
    ``beta`` plus ``alpha`` times the centrality of nodes pointing to it.
    Requires ``alpha < 1 / lambda_max(A)`` for convergence.

    Parameters
    ----------
    graph : Graph or adjacency-like
        Network to score.
    alpha : float, default 0.1
        Attenuation parameter. Must be below ``1 / lambda_max(A)``.
    beta : float, default 1.0
        Baseline score for each node.
    normalized : bool, default True
        L2-normalise the returned vector.
    weighted : bool, optional
        Use tie weights; defaults to the graph's weighted flag.

    Returns
    -------
    pandas.Series

    References
    ----------
    katz1953new

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (1, 2), (1, 3)])
    >>> sp.katz_centrality(g, alpha=0.05).idxmax()
    1
    """
    g = as_graph(graph)
    use_w = g.is_weighted if weighted is None else weighted
    A = g.adjacency_matrix() if use_w else g.binary()
    n = g.n_nodes
    if n == 0:
        return pd.Series([], dtype=float, name="katz")
    lam = float(np.max(np.abs(np.linalg.eigvals(A)).real))
    if lam > 0 and alpha >= 1.0 / lam:
        raise ValueError(
            f"alpha={alpha} is too large for convergence; need "
            f"alpha < 1/lambda_max = {1.0 / lam:.4f}"
        )
    M = np.eye(n) - alpha * A.T
    x = np.linalg.solve(M, beta * np.ones(n))
    if normalized:
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm
    return pd.Series(x, index=g.labels, name="katz")


# ====================================================================== #
#  PageRank
# ====================================================================== #


def pagerank(
    graph: Any,
    alpha: float = 0.85,
    max_iter: int = 1000,
    tol: float = 1e-12,
    weighted: Optional[bool] = None,
) -> pd.Series:
    """Google PageRank (Brin & Page 1998).

    The stationary distribution of a random surfer who follows out-ties with
    probability ``alpha`` and teleports uniformly with probability
    ``1 - alpha``.  Dangling nodes redistribute their mass uniformly.

    Parameters
    ----------
    graph : Graph or adjacency-like
        Directed or undirected network to score.
    alpha : float, default 0.85
        Probability of following a tie rather than teleporting.
    max_iter : int, default 1000
        Maximum power-iteration steps.
    tol : float, default 1e-12
        L1 convergence tolerance.
    weighted : bool, optional
        Use tie weights; defaults to the graph's weighted flag.

    Returns
    -------
    pandas.Series
        Sums to 1.

    References
    ----------
    brin1998anatomy

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (1, 2), (1, 3)])
    >>> round(float(sp.pagerank(g).sum()), 6)
    1.0
    """
    g = as_graph(graph)
    use_w = g.is_weighted if weighted is None else weighted
    A = g.adjacency_matrix() if use_w else g.binary()
    n = g.n_nodes
    if n == 0:
        return pd.Series([], dtype=float, name="pagerank")
    out = A.sum(axis=1)
    dangling = out == 0
    # Column-stochastic transition (row i -> distributes its mass over out-ties)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(out[:, None] > 0, A / out[:, None], 0.0)
    x = np.full(n, 1.0 / n)
    teleport = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        dangle_mass = x[dangling].sum()
        x_new = alpha * (T.T @ x + dangle_mass * teleport) + (1 - alpha) * teleport
        if np.abs(x_new - x).sum() < tol:
            x = x_new
            break
        x = x_new
    x = x / x.sum()
    return pd.Series(x, index=g.labels, name="pagerank")


# ====================================================================== #
#  Bonacich power
# ====================================================================== #


def bonacich_power(
    graph: Any,
    beta: float = 0.1,
    weighted: Optional[bool] = None,
) -> pd.Series:
    """Bonacich (1987) power centrality.

    ``c = (I - beta A)^{-1} A 1``, rescaled so ``sum(c^2) = n`` (the ``sna``
    ``bonpow`` normalisation).  ``beta > 0`` rewards being connected to
    powerful others; ``beta < 0`` (bargaining contexts) rewards being
    connected to *weak* others; ``beta = 0`` reduces to degree.

    Parameters
    ----------
    graph : Graph or adjacency-like
        Network to score.
    beta : float, default 0.1
        Bonacich attenuation parameter. Positive values reward ties to
        powerful nodes; negative values reward ties to weak nodes.
    weighted : bool, optional
        Use tie weights; defaults to the graph's weighted flag.

    Returns
    -------
    pandas.Series

    References
    ----------
    bonacich1987power

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (1, 2), (1, 3)])
    >>> sp.bonacich_power(g, beta=0).idxmax()
    1
    """
    g = as_graph(graph)
    use_w = g.is_weighted if weighted is None else weighted
    A = g.adjacency_matrix() if use_w else g.binary()
    n = g.n_nodes
    if n == 0:
        return pd.Series([], dtype=float, name="bonacich")
    lam = float(np.max(np.abs(np.linalg.eigvals(A)).real))
    if lam > 0 and abs(beta) >= 1.0 / lam:
        raise ValueError(
            f"|beta|={abs(beta)} is too large; need |beta| < 1/lambda_max = "
            f"{1.0 / lam:.4f} for the resolvent to converge"
        )
    c = np.linalg.solve(np.eye(n) - beta * A, A @ np.ones(n))
    ss: float = float(np.sum(c**2))
    if ss > 0:
        c = c * np.sqrt(n / ss)
    return pd.Series(c, index=g.labels, name="bonacich")


# ====================================================================== #
#  HITS
# ====================================================================== #


def hits(graph: Any, max_iter: int = 1000, tol: float = 1e-12) -> pd.DataFrame:
    """Kleinberg HITS hub and authority scores.

    Authorities are nodes pointed to by good hubs; hubs point to good
    authorities.  Scores are L1-normalised (sum to 1).  On an undirected
    graph hub and authority coincide with the eigenvector centrality.

    Parameters
    ----------
    graph : Graph or adjacency-like
        Directed or undirected network to score.
    max_iter : int, default 1000
        Maximum power-iteration steps.
    tol : float, default 1e-12
        L1 convergence tolerance for hub and authority vectors.

    Returns
    -------
    pandas.DataFrame
        Columns ``hub`` and ``authority``, indexed by node label.

    References
    ----------
    kleinberg1999authoritative

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (0, 2)], directed=True)
    >>> set(sp.hits(g).columns)
    {'authority', 'hub'}
    """
    g = as_graph(graph)
    A = g.binary()
    n = g.n_nodes
    if n == 0:
        return pd.DataFrame({"hub": [], "authority": []})
    h = np.ones(n)
    AtA = A.T @ A
    AAt = A @ A.T
    a = np.ones(n)
    for _ in range(max_iter):
        a_new = AtA @ a
        a_new = a_new / a_new.sum() if a_new.sum() else a_new
        h_new = AAt @ h
        h_new = h_new / h_new.sum() if h_new.sum() else h_new
        if np.abs(a_new - a).sum() < tol and np.abs(h_new - h).sum() < tol:
            a, h = a_new, h_new
            break
        a, h = a_new, h_new
    return pd.DataFrame({"hub": h, "authority": a}, index=g.labels)


# ====================================================================== #
#  Dispatcher
# ====================================================================== #

_MEASURES = {
    "degree": lambda g, **k: degree_centrality(g, **_filt(k, {"normalized", "mode"})),
    "closeness": lambda g, **k: closeness_centrality(g, **_filt(k, {"weighted"})),
    "betweenness": lambda g, **k: betweenness_centrality(
        g, **_filt(k, {"normalized", "weighted"})
    ),
    "eigenvector": lambda g, **k: eigenvector_centrality(g, **_filt(k, {"weighted"})),
    "katz": lambda g, **k: katz_centrality(
        g, **_filt(k, {"alpha", "beta", "weighted"})
    ),
    "pagerank": lambda g, **k: pagerank(g, **_filt(k, {"alpha", "weighted"})),
    "bonacich": lambda g, **k: bonacich_power(g, **_filt(k, {"beta", "weighted"})),
}


def _filt(kwargs: Dict[str, Any], allowed: set) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in allowed}


@dataclass
class CentralityResult(ResultProtocolMixin):
    """Per-node centrality table (the ``sp.centrality`` output).

    Attributes
    ----------
    scores : pandas.DataFrame
        One column per requested measure, indexed by node label.
    measures : list of str
    most_central : dict
        ``measure -> node label`` of the top-scoring node.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.centrality(sp.karate_club(), kind=["degree", "betweenness"])
    >>> list(res.scores.columns)
    ['degree', 'betweenness']
    """

    _citation_keys = ("freeman1978centrality", "brandes2001faster")

    scores: pd.DataFrame
    measures: List[str]
    most_central: Dict[str, Any] = field(default_factory=dict)

    def top(self, measure: str, k: int = 5) -> pd.Series:
        """The ``k`` highest-scoring nodes on ``measure``."""
        return self.scores[measure].sort_values(ascending=False).head(k)

    def summary(self) -> str:  # pragma: no cover - cosmetic
        lines = ["Centrality scores", "-----------------"]
        for m in self.measures:
            lines.append(f"  most central by {m}: {self.most_central.get(m)}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CentralityResult(measures={self.measures}, n={len(self.scores)})"


def centrality(
    graph: Any,
    kind: Union[str, Sequence[str]] = "all",
    normalized: bool = True,
    **kwargs: Any,
) -> CentralityResult:
    """Compute one or more centrality measures (family dispatcher).

    Parameters
    ----------
    graph : Graph or adjacency-like
    kind : str or sequence of str, default "all"
        Any of ``"degree"``, ``"closeness"``, ``"betweenness"``,
        ``"eigenvector"``, ``"katz"``, ``"pagerank"``, ``"bonacich"``, or
        ``"all"`` for the standard battery
        (degree/closeness/betweenness/eigenvector/pagerank).
    normalized : bool, default True
        Passed to degree and betweenness.
    **kwargs
        Forwarded to the underlying measure (``alpha``, ``beta``,
        ``weighted``, ``mode``).

    Returns
    -------
    CentralityResult

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.karate_club()
    >>> res = sp.centrality(g, kind="all")
    >>> int(res.top("betweenness", 1).index[0])
    0
    """
    g = as_graph(graph)
    if kind == "all":
        names: List[str] = [
            "degree",
            "closeness",
            "betweenness",
            "eigenvector",
            "pagerank",
        ]
    elif isinstance(kind, str):
        names = [kind]
    else:
        names = list(kind)

    kwargs = dict(kwargs)
    kwargs.setdefault("normalized", normalized)
    cols: Dict[str, pd.Series] = {}
    for name in names:
        if name not in _MEASURES:
            raise ValueError(
                f"unknown centrality {name!r}; choose from "
                f"{sorted(_MEASURES)} or 'all'"
            )
        cols[name] = _MEASURES[name](g, **kwargs)
    scores = pd.DataFrame(cols, index=g.labels)
    most = {m: scores[m].idxmax() for m in names}
    return CentralityResult(scores=scores, measures=names, most_central=most)
