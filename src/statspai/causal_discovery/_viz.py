"""Shared graph visualization / export helpers for causal discovery.

Every result class in :mod:`statspai.causal_discovery` (NOTEARS, PC, FCI,
GES, LiNGAM, ICP, PCMCI, LPCMCI, DYNOTEARS) exposes the same trio of
helpers via this module:

* :func:`to_networkx` — build a :class:`networkx.DiGraph` (or
  :class:`networkx.MultiDiGraph` for time-series methods) from an
  adjacency matrix.
* :func:`to_dot` — render a Graphviz DOT string for plug-in viz tools.
* :func:`plot_dag` — produce a Matplotlib figure with circular / spring /
  hierarchical layout, weighted edge widths, and (optionally) edge labels.

The helpers are kept layout-agnostic and do not require Graphviz; pure
NetworkX/Matplotlib is enough for the standard report. Graphviz is
auto-detected and used if available for cleaner DAG layout.

Adjacency convention
--------------------
``A[i, j] != 0`` means ``names[i] → names[j]`` (parent → child). This
matches the convention used by NOTEARS, PC's CPDAG, and DYNOTEARS. The
LiNGAM convention (``B[i, j]`` = effect *of j on i*) is handled by the
caller transposing before passing in.

References
----------
- NetworkX layout choices follow Hagberg, Schult, and Swart (2008).
- Graphviz DOT format: https://graphviz.org/doc/info/lang.html
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, List

import numpy as np

__all__ = [
    "to_networkx",
    "to_dot",
    "plot_dag",
    "edge_list",
    "shd",
    "DAGDict",
]


class DAGDict(dict):
    """Dict-backed result wrapper that exposes :meth:`to_networkx` /
    :meth:`to_dot` / :meth:`plot` while preserving legacy ``result['key']``
    access. Returned by :func:`statspai.causal_discovery.notears` and
    :func:`statspai.causal_discovery.pc_algorithm`.

    Looks for an adjacency matrix under the keys ``cpdag`` / ``adjacency``
    / ``dag`` (in that order, first match wins), and the variable names
    under ``variables`` / ``names``.
    """

    _ADJ_KEYS = ("cpdag", "adjacency", "dag")
    _NAME_KEYS = ("variables", "names")

    def _get_adj(self) -> tuple[Any, Any]:
        import pandas as _pd

        for k in self._ADJ_KEYS:
            if k in self:
                v = self[k]
                if isinstance(v, _pd.DataFrame):
                    return v.values, list(v.index)
                return v, self.get("variables") or self.get("names")
        raise KeyError(f"DAGDict has no adjacency under {self._ADJ_KEYS!r}")

    def _get_names(self, fallback: Any) -> list[str]:
        for k in self._NAME_KEYS:
            if k in self:
                return list(self[k])
        return list(fallback)

    def to_networkx(self, **kwargs: Any) -> Any:
        adj, names = self._get_adj()
        names = self._get_names(names)
        return to_networkx(adj, names, **kwargs)

    def to_dot(self, **kwargs: Any) -> str:
        adj, names = self._get_adj()
        names = self._get_names(names)
        return to_dot(adj, names, **kwargs)

    def plot(self, **kwargs: Any) -> tuple[Any, Any]:
        adj, names = self._get_adj()
        names = self._get_names(names)
        return plot_dag(adj, names, **kwargs)

    def edge_list(self, **kwargs: Any) -> List[Tuple[str, str, float]]:
        adj, names = self._get_adj()
        names = self._get_names(names)
        return edge_list(adj, names, **kwargs)


def edge_list(
    adjacency: np.ndarray,
    names: Sequence[str],
    threshold: float = 0.0,
    directed: bool = True,
) -> List[Tuple[str, str, float]]:
    """Extract a sorted ``[(parent, child, weight), ...]`` list.

    Edges with ``|w| ≤ threshold`` are dropped. Output is sorted by
    descending ``|weight|`` for stable display.
    """
    A = np.asarray(adjacency, dtype=float)
    edges: List[Tuple[str, str, float]] = []
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not directed and j < i:
                continue
            w = float(A[i, j])
            if abs(w) > threshold:
                edges.append((names[i], names[j], w))
    edges.sort(key=lambda e: abs(e[2]), reverse=True)
    return edges


def to_networkx(
    adjacency: np.ndarray,
    names: Sequence[str],
    directed: bool = True,
    threshold: float = 0.0,
) -> Any:
    """Build a :class:`networkx.DiGraph` (or :class:`networkx.Graph`) from
    an adjacency matrix. Edge weight equals the matrix entry.

    Requires the optional ``networkx`` dependency.
    """
    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "networkx required for to_networkx(). Install: " "pip install networkx"
        ) from e
    A = np.asarray(adjacency, dtype=float)
    G = (nx.DiGraph if directed else nx.Graph)()
    for nm in names:
        G.add_node(nm)
    for parent, child, w in edge_list(A, names, threshold=threshold, directed=directed):
        G.add_edge(parent, child, weight=w)
    return G


def to_dot(
    adjacency: np.ndarray,
    names: Sequence[str],
    directed: bool = True,
    threshold: float = 0.0,
    title: Optional[str] = None,
    digits: int = 2,
) -> str:
    """Render a Graphviz DOT-format string for the DAG.

    Edge labels are weights rounded to ``digits``; positive edges are
    drawn solid, negative edges dashed (matches the bnlearn convention).
    """
    A = np.asarray(adjacency, dtype=float)
    head = "digraph" if directed else "graph"
    sep = "->" if directed else "--"
    lines = [f"{head} G {{"]
    if title:
        lines.append(f'  label="{title}"; labelloc="t"; fontsize=14;')
    lines.append('  node [shape=circle, style=filled, fillcolor="#e8f0fe"];')
    for nm in names:
        lines.append(f'  "{nm}";')
    for parent, child, w in edge_list(A, names, threshold=threshold, directed=directed):
        style = "solid" if w >= 0 else "dashed"
        color = "#1f77b4" if w >= 0 else "#d62728"
        label = f"{w:+.{digits}f}"
        lines.append(
            f'  "{parent}" {sep} "{child}" '
            f'[label="{label}", style={style}, color="{color}", '
            f"penwidth={1 + min(abs(w) * 2, 3):.2f}];"
        )
    lines.append("}")
    return "\n".join(lines)


def plot_dag(
    adjacency: np.ndarray,
    names: Sequence[str],
    *,
    directed: bool = True,
    threshold: float = 0.0,
    layout: str = "circular",
    ax: Optional[Any] = None,
    edge_labels: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6.0, 6.0),
    node_color: str = "#e8f0fe",
    pos_edge_color: str = "#1f77b4",
    neg_edge_color: str = "#d62728",
    digits: int = 2,
) -> tuple[Any, Any]:
    """Draw the DAG with Matplotlib + NetworkX.

    Parameters
    ----------
    adjacency : (k, k) ndarray
    names : sequence of str
    directed : bool, default True
    threshold : float, default 0.0
        Drop edges with ``|w| <= threshold``.
    layout : {"circular", "spring", "kamada_kawai", "shell", "graphviz"}
        ``"graphviz"`` requires pygraphviz; falls back to "spring"
        when missing.
    ax : matplotlib Axes, optional
    edge_labels : bool, default False
        Annotate edges with the weight (rounded to ``digits``).
    title : str, optional
    figsize : (w, h)
    node_color, pos_edge_color, neg_edge_color : str
    digits : int

    Returns
    -------
    fig, ax
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matplotlib + networkx required for plot_dag(). "
            "Install: pip install matplotlib networkx"
        ) from e

    G = to_networkx(adjacency, names, directed=directed, threshold=threshold)

    if layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, seed=0)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "graphviz":
        try:
            from networkx.drawing.nx_agraph import graphviz_layout

            pos = graphviz_layout(G, prog="dot")
        except ImportError:
            pos = nx.spring_layout(G, seed=0)
    else:
        raise ValueError(f"Unknown layout: {layout!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_color,
        node_size=1200,
        edgecolors="#333",
        linewidths=1.2,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 0]

    def _widths(edges: Sequence[tuple[Any, Any]]) -> list[float]:
        return [1.0 + min(abs(G[u][v]["weight"]) * 2, 3.0) for (u, v) in edges]

    if pos_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=pos_edges,
            edge_color=pos_edge_color,
            width=_widths(pos_edges),
            arrowsize=18 if directed else 0,
            arrows=directed,
            ax=ax,
        )
    if neg_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=neg_edges,
            edge_color=neg_edge_color,
            width=_widths(neg_edges),
            style="dashed",
            arrowsize=18 if directed else 0,
            arrows=directed,
            ax=ax,
        )

    if edge_labels:
        labels = {(u, v): f"{G[u][v]['weight']:+.{digits}f}" for u, v in G.edges}
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=labels,
            font_size=8,
            ax=ax,
        )

    ax.set_title(title or "Estimated DAG", fontsize=12)
    ax.set_axis_off()
    return fig, ax


def shd(
    estimated: np.ndarray,
    truth: np.ndarray,
    threshold: float = 0.0,
) -> int:
    r"""Structural Hamming Distance between two adjacency matrices.

    Counts the number of edge insertions / deletions / reversals required
    to transform :math:`\hat A` into the true DAG. Both inputs are
    binarised at ``|·| > threshold`` first.

    Reference: Tsamardinos, Brown, Aliferis (2006). "The max-min
    hill-climbing Bayesian network structure learning algorithm."
    *Machine Learning* 65(1): 31-78. DOI: 10.1007/s10994-006-6889-7.
    """
    A = (np.abs(np.asarray(estimated)) > threshold).astype(int)
    B = (np.abs(np.asarray(truth)) > threshold).astype(int)
    diff = (A != B).astype(int)
    # Count reversals as one (not two) by halving symmetric disagreements.
    sym = diff & diff.T
    asym = diff - sym
    return int(asym.sum() + sym.sum() // 2)
