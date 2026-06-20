"""Core graph object and shared primitives for ``sp.network``.

A lightweight, numpy/scipy-only graph representation backed by a dense
adjacency matrix.  It targets social-network scale (tens to a few
thousand nodes) and aligns conceptually with R's ``igraph`` / ``sna``
and Stata's ``nwcommands``: a single :class:`Graph` object flows through
the descriptive, centrality, community-detection, network-regression and
ERGM layers.

Design notes
------------
* Storage is a dense ``float`` adjacency matrix.  Dense storage keeps the
  algorithms simple and correct at SNA scale; sparse CSR storage for very
  large graphs is on the roadmap (see ``network/__init__.py``).
* Undirected graphs are stored symmetrically; an asymmetric matrix passed
  with ``directed=False`` is symmetrised with :func:`numpy.maximum` and a
  warning is emitted (never silently).
* Self-loops are dropped by default (the SNA convention); pass
  ``allow_self_loops=True`` to keep the diagonal.

This module deliberately holds the shared primitives (the ``Graph`` class,
the :func:`as_graph` input adapter, shortest-path helpers) so that the
estimator files never re-implement adjacency handling — mirroring the
``rd/_core.py`` and ``decomposition/_common.py`` conventions.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:  # SciPy is a core dependency; the import guard is defensive only.
    from scipy.sparse import csgraph, issparse

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is required at runtime
    _HAVE_SCIPY = False

GraphLike = Any  # Graph | np.ndarray | scipy.sparse | W-like


__all__ = ["Graph", "as_graph", "shortest_path_lengths"]


def _to_dense(adjacency: GraphLike) -> np.ndarray:
    """Coerce an adjacency-like object to a dense ``float`` ndarray."""
    if _HAVE_SCIPY and issparse(adjacency):
        return np.asarray(adjacency.toarray(), dtype=float)
    if hasattr(adjacency, "full"):  # libpysal-style W object
        return np.asarray(adjacency.full()[0], dtype=float)
    if hasattr(adjacency, "toarray"):
        return np.asarray(adjacency.toarray(), dtype=float)
    return np.asarray(adjacency, dtype=float)


class Graph:
    """A social network as a dense adjacency matrix.

    Parameters
    ----------
    adjacency : ndarray or scipy.sparse or W-like
        Square ``(n, n)`` adjacency.  Entry ``A[i, j]`` is the weight of the
        tie from ``i`` to ``j`` (``1`` for an unweighted tie).
    directed : bool, default False
        If ``False`` the graph is treated as undirected and stored
        symmetrically.
    node_labels : sequence of str, optional
        Human-readable node names (defaults to ``"0".."n-1"``).
    weighted : bool, optional
        Whether to treat off-diagonal entries as weights.  If ``None``
        (default) it is inferred: the graph is *weighted* when any entry is
        not in ``{0, 1}``.
    allow_self_loops : bool, default False
        Keep the diagonal of ``adjacency``.  Off by default (SNA convention).

    Attributes
    ----------
    n_nodes : int
    n_edges : int
        Number of (directed) arcs; for an undirected graph this is the number
        of edges (each unordered pair counted once).
    is_directed : bool
    is_weighted : bool

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)  # path 0-1-2
    >>> g = sp.network_graph(A)
    >>> g.n_nodes, g.n_edges
    (3, 2)
    >>> g.degree().tolist()
    [1.0, 2.0, 1.0]
    """

    __slots__ = ("_A", "_directed", "_labels", "_weighted")

    def __init__(
        self,
        adjacency: GraphLike,
        directed: bool = False,
        node_labels: Optional[Sequence[Any]] = None,
        weighted: Optional[bool] = None,
        allow_self_loops: bool = False,
    ) -> None:
        A = _to_dense(adjacency)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(
                f"adjacency must be a square 2-D matrix; got shape {A.shape}"
            )
        if not np.isfinite(A).all():
            raise ValueError("adjacency contains non-finite values (NaN/inf)")
        if (A < 0).any():
            raise ValueError(
                "adjacency has negative entries; network measures assume "
                "non-negative tie weights"
            )
        n = A.shape[0]
        if not allow_self_loops:
            A = A.copy()
            np.fill_diagonal(A, 0.0)
        if not directed and not _is_symmetric(A):
            warnings.warn(
                "adjacency is not symmetric but directed=False; symmetrising "
                "with element-wise max (A := max(A, A.T)). Pass directed=True "
                "to keep arc directions.",
                stacklevel=2,
            )
            A = np.maximum(A, A.T)
        if weighted is None:
            offdiag = A[~np.eye(n, dtype=bool)] if n else A.ravel()
            weighted = bool(np.any((offdiag != 0) & (offdiag != 1)))

        self._A = np.ascontiguousarray(A, dtype=float)
        self._directed = bool(directed)
        self._weighted = bool(weighted)
        if node_labels is None:
            self._labels: List[Any] = list(range(n))
        else:
            labels = list(node_labels)
            if len(labels) != n:
                raise ValueError(
                    f"node_labels has length {len(labels)} but adjacency is " f"{n}x{n}"
                )
            self._labels = labels

    # -- construction helpers ------------------------------------------ #

    @classmethod
    def from_edgelist(
        cls,
        edges: Sequence[Tuple[Any, Any]],
        directed: bool = False,
        weights: Optional[Sequence[float]] = None,
        nodes: Optional[Sequence[Any]] = None,
    ) -> "Graph":
        """Build a :class:`Graph` from an iterable of ``(u, v)`` pairs.

        Parameters
        ----------
        edges : sequence of (hashable, hashable)
        directed : bool, default False
        weights : sequence of float, optional
            Per-edge weights (default ``1`` for every edge).
        nodes : sequence, optional
            Explicit node ordering / inclusion of isolates.  If omitted, the
            node set is the sorted union of endpoints.
        """
        edges = list(edges)
        if nodes is None:
            seen: List[Any] = []
            seen_set = set()
            for u, v in edges:
                for x in (u, v):
                    if x not in seen_set:
                        seen_set.add(x)
                        seen.append(x)
            try:
                node_list = sorted(seen)
            except TypeError:
                node_list = seen
        else:
            node_list = list(nodes)
        index = {lab: i for i, lab in enumerate(node_list)}
        n = len(node_list)
        A: np.ndarray = np.zeros((n, n), dtype=float)
        w = [1.0] * len(edges) if weights is None else list(weights)
        if len(w) != len(edges):
            raise ValueError("weights must match the number of edges")
        for (u, v), wi in zip(edges, w):
            if u not in index or v not in index:
                raise ValueError(f"edge ({u!r}, {v!r}) references unknown node")
            i, j = index[u], index[v]
            A[i, j] += wi
            if not directed:
                A[j, i] = A[i, j]
        return cls(A, directed=directed, node_labels=node_list)

    @classmethod
    def from_pandas_edgelist(
        cls,
        df: Any,
        source: str,
        target: str,
        weight: Optional[str] = None,
        directed: bool = False,
        nodes: Optional[Sequence[Any]] = None,
    ) -> "Graph":
        """Build a :class:`Graph` from a tidy edge-list ``DataFrame``."""
        edges = list(zip(df[source].tolist(), df[target].tolist()))
        weights = df[weight].tolist() if weight is not None else None
        return cls.from_edgelist(edges, directed=directed, weights=weights, nodes=nodes)

    # -- basic accessors ----------------------------------------------- #

    @property
    def n_nodes(self) -> int:
        return self._A.shape[0]

    @property
    def is_directed(self) -> bool:
        return self._directed

    @property
    def is_weighted(self) -> bool:
        return self._weighted

    @property
    def labels(self) -> List[Any]:
        return list(self._labels)

    @property
    def n_edges(self) -> int:
        nz = int(np.count_nonzero(self._A))
        return nz if self._directed else nz // 2

    def adjacency_matrix(self, binary: bool = False) -> np.ndarray:
        """Return a copy of the adjacency matrix (optionally binarised)."""
        if binary:
            return np.asarray(self._A != 0, dtype=float)
        return np.asarray(self._A.copy(), dtype=float)

    def binary(self) -> np.ndarray:
        """0/1 adjacency (ties present), self-loops already removed."""
        return np.asarray(self._A != 0, dtype=float)

    def degree(self, mode: str = "all") -> np.ndarray:
        """Unweighted degree.

        Parameters
        ----------
        mode : {"all", "in", "out"}, default "all"
            For undirected graphs all three coincide.  For directed graphs,
            ``"out"`` counts ``A[i, :]`` and ``"in"`` counts ``A[:, i]``;
            ``"all"`` is their sum.
        """
        B = self.binary()
        if mode == "out":
            return np.asarray(B.sum(axis=1), dtype=float)
        if mode == "in":
            return np.asarray(B.sum(axis=0), dtype=float)
        if mode == "all":
            out = B.sum(axis=1) if not self._directed else B.sum(0) + B.sum(1)
            return np.asarray(out, dtype=float)
        raise ValueError("mode must be 'all', 'in', or 'out'")

    def strength(self, mode: str = "all") -> np.ndarray:
        """Weighted degree (sum of incident tie weights)."""
        A = self._A
        if mode == "out":
            return np.asarray(A.sum(axis=1), dtype=float)
        if mode == "in":
            return np.asarray(A.sum(axis=0), dtype=float)
        if mode == "all":
            out = A.sum(axis=1) if not self._directed else A.sum(0) + A.sum(1)
            return np.asarray(out, dtype=float)
        raise ValueError("mode must be 'all', 'in', or 'out'")

    @property
    def density(self) -> float:
        """Fraction of possible ties present."""
        n = self.n_nodes
        if n < 2:
            return 0.0
        possible = n * (n - 1)
        present = int(np.count_nonzero(self.binary()))
        return present / possible  # directed; undirected: both counts present

    def node_index(self, label: Any) -> int:
        return self._labels.index(label)

    def subgraph(self, nodes: Sequence[Any]) -> "Graph":
        """Induced subgraph on ``nodes`` (given as labels)."""
        idx = [self.node_index(x) if x in self._labels else int(x) for x in nodes]
        A = self._A[np.ix_(idx, idx)]
        return Graph(
            A,
            directed=self._directed,
            node_labels=[self._labels[i] for i in idx],
            weighted=self._weighted,
            allow_self_loops=True,
        )

    def copy(self) -> "Graph":
        return Graph(
            self._A.copy(),
            directed=self._directed,
            node_labels=list(self._labels),
            weighted=self._weighted,
            allow_self_loops=True,
        )

    # -- dunders -------------------------------------------------------- #

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "directed" if self._directed else "undirected"
        w = "weighted" if self._weighted else "unweighted"
        return f"Graph({kind}, {w}, n_nodes={self.n_nodes}, " f"n_edges={self.n_edges})"

    def summary(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Graph: {'directed' if self._directed else 'undirected'}, "
            f"{'weighted' if self._weighted else 'unweighted'}\n"
            f"  nodes : {self.n_nodes}\n"
            f"  edges : {self.n_edges}\n"
            f"  density: {self.density:.4f}"
        )


def _is_symmetric(A: np.ndarray, tol: float = 1e-10) -> bool:
    if A.shape[0] != A.shape[1]:
        return False
    return bool(np.allclose(A, A.T, atol=tol, rtol=0.0))


def as_graph(obj: GraphLike, directed: Optional[bool] = None) -> Graph:
    """Coerce any adjacency-like input into a :class:`Graph`.

    The universal adapter used by every public ``sp.network`` function so
    that callers may pass a :class:`Graph`, a dense/sparse adjacency matrix,
    or a libpysal ``W`` object interchangeably.

    Parameters
    ----------
    obj : Graph or ndarray or scipy.sparse or W-like
    directed : bool, optional
        Override the directedness.  Ignored when ``obj`` is already a
        :class:`Graph` unless it differs, in which case the graph is
        re-interpreted.

    Returns
    -------
    Graph
    """
    if isinstance(obj, Graph):
        if directed is None or directed == obj.is_directed:
            return obj
        return Graph(
            obj.adjacency_matrix(),
            directed=directed,
            node_labels=obj.labels,
            weighted=obj.is_weighted,
            allow_self_loops=True,
        )
    return Graph(obj, directed=bool(directed) if directed is not None else False)


def shortest_path_lengths(g: Graph, weighted: Optional[bool] = None) -> np.ndarray:
    """All-pairs shortest-path length matrix.

    Uses :func:`scipy.sparse.csgraph.shortest_path`.  Unreachable pairs are
    ``inf``.  Weighted distances use the tie weights as edge lengths when the
    graph is weighted (or ``weighted=True``); otherwise every edge has length
    one.

    Parameters
    ----------
    g : Graph
    weighted : bool, optional
        Force weighted/unweighted distances.  Defaults to ``g.is_weighted``.

    Returns
    -------
    ndarray of shape (n, n)
    """
    use_w = g.is_weighted if weighted is None else weighted
    A = g.adjacency_matrix() if use_w else g.binary()
    if not _HAVE_SCIPY:  # pragma: no cover - scipy always present at runtime
        raise RuntimeError("scipy is required for shortest-path computations")
    D = csgraph.shortest_path(
        csgraph.csgraph_from_dense(A, null_value=0),
        method="D",
        directed=g.is_directed,
        unweighted=not use_w,
    )
    return np.asarray(D, dtype=float)
