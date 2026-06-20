"""Social network analysis for StatsPAI (``sp.network``).

A numpy/scipy-native SNA toolkit aligned with R's ``igraph`` / ``sna`` /
``statnet`` and Stata's ``nwcommands``, covering the layers an applied
network analyst needs:

Construction
    :func:`network_graph` (factory) and the :class:`Graph` object — build
    from a dense/sparse adjacency, an edge list, or a tidy ``DataFrame``.

Descriptives
    :func:`network_summary`, :func:`transitivity`, :func:`clustering`,
    :func:`reciprocity`, :func:`assortativity`, :func:`network_components`.

Centrality
    :func:`centrality` dispatcher plus :func:`degree_centrality`,
    :func:`closeness_centrality`, :func:`betweenness_centrality`,
    :func:`eigenvector_centrality`, :func:`katz_centrality`, :func:`pagerank`,
    :func:`bonacich_power`, :func:`hits`.

Community detection
    :func:`community_detection` (Louvain / greedy / label propagation) and
    :func:`network_modularity`.

Network regression
    :func:`netlm` / :func:`netlogit` (QAP / MRQAP) and
    :func:`dyadic_regression` (dyadic-cluster-robust SEs).

Network formation
    :func:`ergm` (exponential random graph models via MPLE).

Data & plots
    :func:`karate_club`, :func:`florentine_families`, :func:`network_plot`.

Roadmap
-------
Sparse-CSR storage for very large graphs; full **MCMC-MLE** ERGM estimation
(the current :func:`ergm` uses maximum pseudo-likelihood); **SAOM / RSiena**
stochastic actor-oriented models for network *dynamics*; temporal/multiplex
networks.  These are tracked rather than silently stubbed.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ._core import Graph, as_graph, shortest_path_lengths
from .descriptives import (
    network_summary,
    NetworkSummaryResult,
    transitivity,
    clustering,
    reciprocity,
    assortativity,
    network_components,
    ComponentsResult,
)
from .centrality import (
    centrality,
    CentralityResult,
    degree_centrality,
    closeness_centrality,
    betweenness_centrality,
    eigenvector_centrality,
    katz_centrality,
    pagerank,
    bonacich_power,
    hits,
)
from .community import (
    community_detection,
    CommunityResult,
    modularity as network_modularity,
)
from .regression import (
    netlm,
    netlogit,
    QAPResult,
    dyadic_regression,
    DyadicRegressionResult,
)
from .ergm import ergm, ERGMResult
from .datasets import karate_club, florentine_families, KARATE_FACTION
from .plots import network_plot, spring_layout, circular_layout


def network_graph(
    adjacency: Any = None,
    edges: Optional[Sequence] = None,
    directed: bool = False,
    node_labels: Optional[Sequence] = None,
    weights: Optional[Sequence] = None,
    weighted: Optional[bool] = None,
) -> Graph:
    """Construct a :class:`Graph` from an adjacency matrix or an edge list.

    This is the single agent-friendly entry point for building a network.
    Provide *either* ``adjacency`` (a dense/sparse matrix or libpysal ``W``)
    *or* ``edges`` (an iterable of ``(u, v)`` pairs).

    Parameters
    ----------
    adjacency : ndarray or scipy.sparse or W-like, optional
        Square adjacency matrix.
    edges : sequence of (hashable, hashable), optional
        Edge list (mutually exclusive with ``adjacency``).
    directed : bool, default False
    node_labels : sequence, optional
        Node names (adjacency input) or explicit node ordering (edge input).
    weights : sequence of float, optional
        Per-edge weights (edge-list input only).
    weighted : bool, optional
        Force weighted/unweighted interpretation (adjacency input).

    Returns
    -------
    Graph

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.network_graph(edges=[(0, 1), (1, 2), (2, 0)])
    >>> g.n_nodes, g.n_edges
    (3, 3)
    >>> import numpy as np
    >>> A = np.array([[0, 1], [1, 0]], float)
    >>> sp.network_graph(A).density
    1.0
    """
    if adjacency is not None and edges is not None:
        raise ValueError("provide either 'adjacency' or 'edges', not both")
    if edges is not None:
        return Graph.from_edgelist(
            edges, directed=directed, weights=weights, nodes=node_labels
        )
    if adjacency is None:
        raise ValueError("provide 'adjacency' or 'edges'")
    return Graph(
        adjacency,
        directed=directed,
        node_labels=node_labels,
        weighted=weighted,
    )


__all__ = [
    # construction
    "network_graph",
    "Graph",
    "as_graph",
    "shortest_path_lengths",
    # descriptives
    "network_summary",
    "NetworkSummaryResult",
    "transitivity",
    "clustering",
    "reciprocity",
    "assortativity",
    "network_components",
    "ComponentsResult",
    # centrality
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
    # community
    "community_detection",
    "CommunityResult",
    "network_modularity",
    # regression
    "netlm",
    "netlogit",
    "QAPResult",
    "dyadic_regression",
    "DyadicRegressionResult",
    # ergm
    "ergm",
    "ERGMResult",
    # datasets
    "karate_club",
    "florentine_families",
    "KARATE_FACTION",
    # plots
    "network_plot",
    "spring_layout",
    "circular_layout",
]
