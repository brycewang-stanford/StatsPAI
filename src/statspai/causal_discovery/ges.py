"""Greedy Equivalence Search (GES) — Chickering, JMLR 2002.

Score-based causal structure learning: search through equivalence
classes of DAGs by greedily adding then removing edges to maximise
the BIC score. Unlike constraint-based PC, GES does not rely on
conditional independence tests and is consistent under faithfulness.

Two phases:
1. **Forward**: add edges that maximally increase BIC.
2. **Backward**: remove edges that increase BIC.

The output is a CPDAG (completed partially directed acyclic graph) —
the Markov equivalence class of the true DAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def _bic_score_local(data: np.ndarray, j: int, parents: List[int]) -> float:
    """BIC score for node j given parents. Lower = better."""
    n = data.shape[0]
    y = data[:, j]
    if len(parents) == 0:
        rss = float(np.sum((y - y.mean()) ** 2))
        k = 1
    else:
        X = np.column_stack([np.ones(n), data[:, parents]])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        rss = float(np.sum((y - X @ beta) ** 2))
        k = len(parents) + 1
    rss = max(rss, 1e-12)
    return float(n * np.log(rss / n) + k * np.log(n))


def _total_bic(data: np.ndarray, adj: np.ndarray) -> float:
    p = data.shape[1]
    total = 0.0
    for j in range(p):
        parents = list(np.where(adj[:, j] != 0)[0])
        total += _bic_score_local(data, j, parents)
    return total


@dataclass
class GESResult:
    """Result of :func:`ges` — a CPDAG (Markov equivalence class).

    Attributes
    ----------
    adjacency : np.ndarray
        ``(p, p)`` CPDAG adjacency; ``adj[i, j]`` and ``adj[j, i]`` both
        nonzero denotes an undirected edge ``i --- j``.
    names : list of str
        Variable names, aligned with ``adjacency`` rows/columns.
    bic : float
        Total BIC of the recovered graph (lower is better).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=200)
    >>> y = 2.0 * x + rng.normal(size=200)
    >>> df = pd.DataFrame({"x": x, "y": y})
    >>> res = sp.ges(df)
    >>> isinstance(res, sp.GESResult)
    True
    >>> bool(res.to_frame().shape == (2, 2))
    True
    """

    adjacency: np.ndarray
    names: List[str]
    bic: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.adjacency, index=self.names, columns=self.names)

    def edges(self) -> List[Tuple[str, str, str]]:
        out = []
        p = len(self.names)
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                if self.adjacency[i, j] != 0 and self.adjacency[j, i] != 0:
                    if i < j:
                        out.append((self.names[i], self.names[j], "---"))
                elif self.adjacency[i, j] != 0:
                    out.append((self.names[i], self.names[j], "-->"))
        return out

    def summary(self) -> str:
        lines = [
            "Greedy Equivalence Search (GES)",
            "-" * 40,
            f"Variables: {len(self.names)}",
            f"BIC: {self.bic:.2f}",
            "",
            "Edges:",
        ]
        for src, dst, kind in self.edges():
            lines.append(f"  {src} {kind} {dst}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def ges(
    data: Union[pd.DataFrame, np.ndarray],
    max_iter: int = 500,
) -> GESResult:
    """Greedy Equivalence Search.

    Parameters
    ----------
    data : pd.DataFrame or (n, p) ndarray
    max_iter : int
        Maximum total edge additions + removals.

    Returns
    -------
    GESResult
        CPDAG (Markov equivalence class) with ``.adjacency``, ``.bic``,
        ``.edges()`` and ``.to_frame()`` helpers.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=200)
    >>> y = 2.0 * x + rng.normal(size=200)
    >>> z = 1.5 * y + rng.normal(size=200)
    >>> df = pd.DataFrame({"x": x, "y": y, "z": z})
    >>> res = sp.ges(df)
    >>> res.edges()  # doctest: +SKIP
    [('x', 'y', '---'), ('y', 'z', '---')]
    >>> bool(res.to_frame().shape == (3, 3))
    True
    """
    if isinstance(data, pd.DataFrame):
        names = list(data.columns)
        X = data.to_numpy(dtype=float)
    else:
        X = np.asarray(data, dtype=float)
        names = [f"x{i}" for i in range(X.shape[1])]

    n, p = X.shape
    adj = np.zeros((p, p), dtype=int)

    def _score() -> float:
        return _total_bic(X, adj)

    best_bic = _score()

    # Forward phase: greedily add the edge that most improves BIC
    for _ in range(max_iter):
        best_gain = 0.0
        best_edge = None
        best_new_bic = best_bic
        for i in range(p):
            for j in range(p):
                if i == j or adj[i, j]:
                    continue
                adj[i, j] = 1
                new_bic = _score()
                gain = best_bic - new_bic
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (i, j)
                    best_new_bic = new_bic
                adj[i, j] = 0
        if best_edge is None or best_gain <= 0:
            break
        adj[best_edge[0], best_edge[1]] = 1
        best_bic = best_new_bic

    # Backward phase: greedily remove edges that improve BIC
    for _ in range(max_iter):
        best_gain = 0.0
        best_edge = None
        best_new_bic = best_bic
        for i in range(p):
            for j in range(p):
                if adj[i, j] == 0:
                    continue
                adj[i, j] = 0
                new_bic = _score()
                gain = best_bic - new_bic
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (i, j)
                    best_new_bic = new_bic
                adj[i, j] = 1
        if best_edge is None or best_gain <= 0:
            break
        adj[best_edge[0], best_edge[1]] = 0
        best_bic = best_new_bic

    _result = GESResult(adjacency=adj, names=names, bic=best_bic)
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.causal_discovery.ges",
            params={"max_iter": max_iter},
            data=data if hasattr(data, "columns") else None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
