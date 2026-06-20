"""Network-level descriptive statistics for ``sp.network``.

Density, components, diameter, average path length, reciprocity, the global
clustering coefficient (transitivity), per-node local clustering, and Newman
degree assortativity — the structural summary that R's ``igraph`` / ``sna``
and Stata's ``nwcommands`` report for a network.

References
----------
Watts, D. J. & Strogatz, S. H. (1998). "Collective dynamics of 'small-world'
networks." *Nature*, 393, 440-442. [@watts1998collective]

Newman, M. E. J. (2002). "Assortative mixing in networks." *Physical Review
Letters*, 89, 208701. [@newman2002assortative]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ._core import as_graph, shortest_path_lengths

__all__ = [
    "network_summary",
    "NetworkSummaryResult",
    "transitivity",
    "clustering",
    "reciprocity",
    "assortativity",
    "network_components",
    "ComponentsResult",
]


# ====================================================================== #
#  Local + global clustering
# ====================================================================== #


def _triangles_and_triples(B: np.ndarray) -> tuple:
    """Return ``(triangles_through_node, k*(k-1) per node)`` for binary B."""
    k = B.sum(axis=1)
    # (B^3)_ii / 2 = number of triangles through node i (undirected).
    B2 = B @ B
    tri_node = np.einsum("ij,ji->i", B2, B) / 2.0  # diag(B^3) / 2
    denom = k * (k - 1.0)
    return tri_node, denom


def transitivity(graph: Any) -> float:
    """Global clustering coefficient (transitivity).

    The fraction of connected triples of nodes that are closed into a
    triangle: ``3 * (#triangles) / (#connected triples)``.

    Parameters
    ----------
    graph : Graph or adjacency-like

    Returns
    -------
    float

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.karate_club()
    >>> round(sp.transitivity(g), 4)
    0.2557
    """
    g = as_graph(graph)
    B = g.binary()
    tri_node, denom = _triangles_and_triples(B)
    triples = denom.sum()  # sum k_i(k_i-1) = 2 * (#connected triples)
    if triples <= 0:
        return 0.0
    return float(2.0 * tri_node.sum() / triples)


def clustering(graph: Any) -> pd.Series:
    """Per-node local clustering coefficient (Watts-Strogatz).

    ``c_i = 2 e_i / (k_i (k_i - 1))`` where ``e_i`` is the number of ties
    among the ``k_i`` neighbours of ``i``.  Nodes with degree < 2 receive 0.

    Parameters
    ----------
    graph : Graph or adjacency-like
        Network whose local clustering coefficients should be computed.

    Returns
    -------
    pandas.Series
        Indexed by node label.

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (1, 2), (2, 0)])
    >>> sp.clustering(g).tolist()
    [1.0, 1.0, 1.0]
    """
    g = as_graph(graph)
    B = g.binary()
    tri_node, denom = _triangles_and_triples(B)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(denom > 0, 2.0 * tri_node / denom, 0.0)
    return pd.Series(c, index=g.labels, name="clustering")


# ====================================================================== #
#  Reciprocity (directed)
# ====================================================================== #


def reciprocity(graph: Any) -> float:
    """Directed reciprocity: share of arcs that are reciprocated.

    ``sum_{i!=j} A_ij A_ji / sum_{i!=j} A_ij`` on the binarised graph.  An
    undirected graph returns ``1.0`` (every tie is mutual by construction).

    Parameters
    ----------
    graph : Graph or adjacency-like
        Directed network. Undirected inputs return ``1.0`` by construction.

    Returns
    -------
    float

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (1, 0), (1, 2)], directed=True)
    >>> round(sp.reciprocity(g), 3)
    0.667
    """
    g = (
        as_graph(graph, directed=True)
        if not as_graph(graph).is_directed
        else as_graph(graph)
    )
    B = (g.adjacency_matrix() != 0).astype(float)
    np.fill_diagonal(B, 0.0)
    total = B.sum()
    if total <= 0:
        return 0.0
    mutual = float(np.sum(B * B.T))
    return mutual / float(total)


# ====================================================================== #
#  Degree assortativity (Newman 2002)
# ====================================================================== #


def assortativity(graph: Any) -> float:
    """Newman degree-assortativity coefficient.

    The Pearson correlation of the degrees at the two ends of an edge.
    Positive values mean high-degree nodes tend to attach to high-degree
    nodes (assortative mixing); negative values indicate disassortativity
    (hub-and-spoke), as in many social and technological networks.

    Parameters
    ----------
    graph : Graph or adjacency-like
        Network whose endpoint-degree correlation should be computed.

    Returns
    -------
    float

    References
    ----------
    newman2002assortative

    Examples
    --------
    >>> import statspai as sp
    >>> val = sp.assortativity(sp.karate_club())
    >>> bool(val < 0)
    True
    """
    g = as_graph(graph)
    B = g.binary()
    deg = B.sum(axis=1)
    iu = np.argwhere(B != 0)
    if iu.size == 0:
        return float("nan")
    # Each undirected edge contributes both orientations (i,j) and (j,i);
    # for a symmetric B the argwhere already yields both, matching Newman's
    # edge-stub correlation.
    x = deg[iu[:, 0]].astype(float)
    y = deg[iu[:, 1]].astype(float)
    if np.allclose(x, x[0]) and np.allclose(y, y[0]):
        return float("nan")
    sx, sy = x.sum(), y.sum()
    m = len(x)
    num = (x * y).mean() - (sx / m) * (sy / m)
    den = np.sqrt((x * x).mean() - (sx / m) ** 2) * np.sqrt(
        (y * y).mean() - (sy / m) ** 2
    )
    if den == 0:
        return float("nan")
    return float(num / den)


# ====================================================================== #
#  Connected components
# ====================================================================== #


@dataclass
class ComponentsResult(ResultProtocolMixin):
    """Connected-component decomposition of a graph.

    Attributes
    ----------
    n_components : int
    membership : pandas.Series
        Component id per node (indexed by node label).
    sizes : list of int
        Component sizes, descending.
    largest_size : int
    connection : str
        ``"weak"`` or ``"strong"`` (directed graphs only).

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (2, 3)], node_labels=[0, 1, 2, 3])
    >>> sp.network_components(g).sizes
    [2, 2]
    """

    n_components: int
    membership: pd.Series
    sizes: List[int]
    largest_size: int
    connection: str = "weak"

    def summary(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Connected components ({self.connection})\n"
            f"  n_components : {self.n_components}\n"
            f"  largest size : {self.largest_size}\n"
            f"  sizes        : {self.sizes[:10]}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ComponentsResult(n_components={self.n_components})"


def network_components(graph: Any, connection: str = "weak") -> ComponentsResult:
    """Connected-component decomposition.

    Parameters
    ----------
    graph : Graph or adjacency-like
    connection : {"weak", "strong"}, default "weak"
        For directed graphs, whether to use weak or strong connectivity.
        Ignored for undirected graphs.

    Returns
    -------
    ComponentsResult

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (2, 3)], node_labels=[0, 1, 2, 3])
    >>> sp.network_components(g).n_components
    2
    """
    from scipy.sparse import csgraph

    g = as_graph(graph)
    B = g.binary()
    n_comp, labels = csgraph.connected_components(
        csgraph.csgraph_from_dense(B, null_value=0),
        directed=g.is_directed,
        connection=connection,
    )
    membership = pd.Series(labels, index=g.labels, name="component")
    _, counts = np.unique(labels, return_counts=True)
    sizes = sorted((int(c) for c in counts), reverse=True)
    return ComponentsResult(
        n_components=int(n_comp),
        membership=membership,
        sizes=sizes,
        largest_size=sizes[0] if sizes else 0,
        connection=connection if g.is_directed else "weak",
    )


# ====================================================================== #
#  Omnibus network summary
# ====================================================================== #


@dataclass
class NetworkSummaryResult(ResultProtocolMixin):
    """Structural summary of a network (the ``sp.network_summary`` output).

    Attributes
    ----------
    n_nodes, n_edges : int
    directed, weighted : bool
    density : float
    n_components : int
    largest_component_frac : float
    is_connected : bool
    diameter : float
        Longest shortest path *within the largest component* (``inf`` only
        for an empty graph).
    average_path_length : float
        Mean shortest-path length over reachable ordered pairs.
    mean_degree : float
    transitivity : float
        Global clustering coefficient.
    average_clustering : float
        Mean local clustering coefficient.
    reciprocity : float
        Directed graphs only; ``nan`` for undirected.
    assortativity : float
        Newman degree-assortativity coefficient.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.network_summary(sp.karate_club())
    >>> res.n_nodes, res.n_edges
    (34, 78)
    >>> round(res.density, 4)
    0.139
    """

    _citation_keys = ("watts1998collective", "newman2002assortative")

    n_nodes: int
    n_edges: int
    directed: bool
    weighted: bool
    density: float
    n_components: int
    largest_component_frac: float
    is_connected: bool
    diameter: float
    average_path_length: float
    mean_degree: float
    transitivity: float
    average_clustering: float
    reciprocity: float
    assortativity: float
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover - cosmetic
        rec = "n/a" if np.isnan(self.reciprocity) else f"{self.reciprocity:.4f}"
        return (
            "Network summary\n"
            "---------------\n"
            f"  nodes / edges       : {self.n_nodes} / {self.n_edges}\n"
            f"  directed / weighted : {self.directed} / {self.weighted}\n"
            f"  density             : {self.density:.4f}\n"
            f"  components          : {self.n_components} "
            f"(largest {self.largest_component_frac:.1%})\n"
            f"  diameter            : {self.diameter:.0f}\n"
            f"  avg path length     : {self.average_path_length:.4f}\n"
            f"  mean degree         : {self.mean_degree:.4f}\n"
            f"  transitivity        : {self.transitivity:.4f}\n"
            f"  avg clustering      : {self.average_clustering:.4f}\n"
            f"  reciprocity         : {rec}\n"
            f"  assortativity       : {self.assortativity:+.4f}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"NetworkSummaryResult(n_nodes={self.n_nodes}, "
            f"n_edges={self.n_edges}, density={self.density:.4f})"
        )


def network_summary(graph: Any) -> NetworkSummaryResult:
    """Compute the structural summary of a network.

    Parameters
    ----------
    graph : Graph or adjacency-like
        A :class:`~statspai.network.Graph`, a dense/sparse adjacency matrix,
        or a libpysal ``W`` object.

    Returns
    -------
    NetworkSummaryResult

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.network_summary(sp.karate_club())
    >>> res.diameter
    5.0
    >>> res.n_components
    1
    """
    g = as_graph(graph)
    n = g.n_nodes
    comp = network_components(g)
    is_connected = comp.n_components == 1 and n > 0

    D = shortest_path_lengths(g, weighted=False)
    finite = D[np.isfinite(D)]
    offdiag = finite[finite > 0]
    diameter = float(offdiag.max()) if offdiag.size else float("inf")
    avg_path = float(offdiag.mean()) if offdiag.size else float("inf")

    loc = clustering(g)
    summary = NetworkSummaryResult(
        n_nodes=n,
        n_edges=g.n_edges,
        directed=g.is_directed,
        weighted=g.is_weighted,
        density=g.density,
        n_components=comp.n_components,
        largest_component_frac=(comp.largest_size / n) if n else 0.0,
        is_connected=is_connected,
        diameter=diameter,
        average_path_length=avg_path,
        mean_degree=float(g.degree().mean()) if n else 0.0,
        transitivity=transitivity(g),
        average_clustering=float(loc.mean()) if n else 0.0,
        reciprocity=reciprocity(g) if g.is_directed else float("nan"),
        assortativity=assortativity(g),
        detail={"component_sizes": comp.sizes},
    )
    return summary
