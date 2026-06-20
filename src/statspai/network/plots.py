"""Network visualisation for ``sp.network`` (lazy matplotlib).

A dependency-light node-link drawing routine.  ``matplotlib`` is imported
lazily, so importing :mod:`statspai.network` never requires the optional
``plotting`` extra; :func:`network_plot` raises a clear, actionable error
only if it is actually called without matplotlib installed.

The force-directed (Fruchterman-Reingold) and circular layouts are computed
in numpy — no networkx / graphviz dependency.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ._core import as_graph

__all__ = ["network_plot", "spring_layout", "circular_layout"]


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt  # noqa: F401

        return plt
    except Exception as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            "network_plot requires matplotlib. Install the plotting extra:\n"
            "    pip install 'statspai[plotting]'"
        ) from exc


def circular_layout(n: int) -> np.ndarray:
    """Evenly spaced positions on the unit circle."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def spring_layout(
    A: np.ndarray,
    iterations: int = 100,
    seed: Optional[int] = 0,
    k: Optional[float] = None,
) -> np.ndarray:
    """Fruchterman-Reingold force-directed layout (numpy).

    Parameters
    ----------
    A : (n, n) ndarray
        Adjacency (symmetrised internally).
    iterations : int, default 100
    seed : int, optional
    k : float, optional
        Ideal edge length; defaults to ``1/sqrt(n)``.

    Returns
    -------
    (n, 2) ndarray of positions in roughly ``[-1, 1]^2``.
    """
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=0.5, size=(n, 2))
    B = ((A + A.T) > 0).astype(float)
    if k is None:
        k = 1.0 / np.sqrt(n)
    t = 0.1
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]  # (n,n,2)
        dist = np.sqrt((delta**2).sum(-1)) + 1e-9
        # repulsive force k^2/d, attractive force d^2/k along edges
        rep = (k * k) / dist
        att = (dist * dist) / k * B
        force_mag = (rep - att) / dist
        np.fill_diagonal(force_mag, 0.0)
        disp = (delta * force_mag[:, :, None]).sum(axis=1)
        length = np.sqrt((disp**2).sum(-1)) + 1e-9
        pos = pos + (disp / length[:, None]) * np.minimum(length, t)[:, None]
        t = max(t * 0.95, 0.005)
    # normalise to [-1, 1]
    span = pos.max(0) - pos.min(0)
    span[span == 0] = 1.0
    pos = 2 * (pos - pos.min(0)) / span - 1
    return np.asarray(pos, dtype=float)


def network_plot(
    graph: Any,
    layout: str = "spring",
    node_color: Optional[Sequence] = None,
    node_size: Optional[Sequence] = None,
    labels: bool = False,
    ax: Any = None,
    seed: Optional[int] = 0,
    cmap: str = "tab10",
    edge_alpha: float = 0.35,
    title: Optional[str] = None,
) -> Any:
    """Draw a network as a node-link diagram.

    Parameters
    ----------
    graph : Graph or adjacency-like
    layout : {"spring", "circular"}, default "spring"
    node_color : sequence, optional
        Per-node values (e.g. a community membership :class:`pandas.Series`)
        mapped through ``cmap``.
    node_size : sequence, optional
        Per-node sizes (e.g. a centrality score); rescaled to a sensible
        point range.
    labels : bool, default False
        Annotate nodes with their labels.
    ax : matplotlib Axes, optional
    seed : int, optional
        Layout seed (spring layout).
    cmap : str, default "tab10"
    edge_alpha : float, default 0.35
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import statspai as sp  # doctest: +SKIP
    >>> g = sp.karate_club()  # doctest: +SKIP
    >>> com = sp.community_detection(g)  # doctest: +SKIP
    >>> sp.network_plot(g, node_color=com.membership)  # doctest: +SKIP
    """
    plt = _require_matplotlib()
    g = as_graph(graph)
    A = g.adjacency_matrix()
    n = g.n_nodes
    if layout == "circular":
        pos = circular_layout(n)
    elif layout == "spring":
        pos = spring_layout(A, seed=seed)
    else:
        raise ValueError("layout must be 'spring' or 'circular'")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    # edges
    src, dst = np.where(np.triu(A, 1) != 0) if not g.is_directed else np.where(A != 0)
    for s, d in zip(src, dst):
        ax.plot(
            [pos[s, 0], pos[d, 0]],
            [pos[s, 1], pos[d, 1]],
            color="0.5",
            alpha=edge_alpha,
            linewidth=0.8,
            zorder=1,
        )

    # node sizes
    if node_size is not None:
        s = np.asarray(list(node_size), dtype=float)
        s = s - s.min()
        s = 100 + 600 * (s / s.max()) if s.max() > 0 else np.full(n, 200.0)
    else:
        s = np.full(n, 200.0)

    # node colors
    if node_color is not None:
        c = np.asarray(list(node_color))
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=c,
            s=s,
            cmap=cmap,
            zorder=2,
            edgecolors="white",
            linewidths=0.7,
        )
    else:
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            s=s,
            zorder=2,
            color="#3b6ea5",
            edgecolors="white",
            linewidths=0.7,
        )

    if labels:
        for i, lab in enumerate(g.labels):
            ax.annotate(
                str(lab),
                (pos[i, 0], pos[i, 1]),
                fontsize=8,
                ha="center",
                va="center",
                zorder=3,
            )

    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax
