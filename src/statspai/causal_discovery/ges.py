"""Greedy Equivalence Search (GES), Chickering (2002).

Score-based causal structure learning over Markov equivalence classes.
The state is a CPDAG, and every step applies the best of Chickering's
operators under the Gaussian BIC (a score-equivalent, decomposable score):

1. **Forward**: ``Insert(X, Y, T)`` adds ``X -> Y`` and orients ``T - Y``
   into ``Y``; valid when ``NA_{Y,X} | T`` is a clique and blocks every
   semi-directed path from ``Y`` to ``X``.
2. **Backward**: ``Delete(X, Y, H)`` removes the ``X``-``Y`` edge and
   orients ``Y - h`` (and ``X - h``) towards ``h``; valid when
   ``NA_{Y,X} \\ H`` is a clique.

After each operator the PDAG is extended to a DAG (Dor and Tarsi) and
re-completed (v-structures plus Meek's rules R1-R3).

Before 2026-10-19 this module searched DAG space with single-edge
additions and removals and then marked as undirected every edge not in a
v-structure, so compelled edges such as ``Z -> W`` below a collider
``X -> Z <- Y`` were reported as ``Z - W``, and the search could stop in a
DAG that is not the GES optimum. It matched ``pcalg::ges(phase =
c("forward", "backward"))`` on 11 of 30 random linear-Gaussian graphs; this
implementation (written from the paper; pcalg is GPL and compared only as
a black box) matches all 30 (``tests/reference_parity/test_ges_pcalg_parity.py``).

References
----------
[@chickering2002optimal]
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, Iterator, List, Set, Tuple, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin


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


# ---------------------------------------------------------------------------
# CPDAG bookkeeping. ``g[i, j] == 1 and g[j, i] == 0`` is ``i -> j``;
# ``g[i, j] == g[j, i] == 1`` is the undirected edge ``i - j``.
# ---------------------------------------------------------------------------


def _undirected_nbrs(g: np.ndarray, y: int) -> set:
    return {int(v) for v in np.flatnonzero((g[y] == 1) & (g[:, y] == 1))}


def _parents(g: np.ndarray, y: int) -> set:
    return {int(v) for v in np.flatnonzero((g[:, y] == 1) & (g[y] == 0))}


def _adjacent(g: np.ndarray, a: int, b: int) -> bool:
    return bool(g[a, b] or g[b, a])


def _is_clique(g: np.ndarray, nodes: set) -> bool:
    ordered = sorted(nodes)
    return all(
        _adjacent(g, a, b) for i, a in enumerate(ordered) for b in ordered[i + 1 :]
    )


def _semi_directed_path_blocked(
    g: np.ndarray, start: int, target: int, blocked: set
) -> bool:
    """True iff every semi-directed path ``start ~> target`` meets ``blocked``.

    A semi-directed path follows ``a -> b`` forwards and ``a - b`` either way.
    """
    stack, seen = [start], {start}
    while stack:
        v = stack.pop()
        for w in np.flatnonzero(g[v]):  # v -> w or v - w
            w = int(w)
            if w == target:
                return False
            if w in seen or w in blocked:
                continue
            seen.add(w)
            stack.append(w)
    return True


def _pdag_to_dag(g: np.ndarray) -> np.ndarray:
    """A consistent DAG extension of a PDAG (Dor and Tarsi's procedure).

    Repeatedly removes a sink ``x`` (no outgoing directed edge) whose
    undirected neighbours are adjacent to every other neighbour of ``x``,
    orienting its undirected edges into ``x``. GES's operators always leave
    an extendable PDAG; failing to find such a node is a bug, not data.
    """
    p = g.shape[0]
    work = g.copy()
    dag = np.zeros_like(g)
    alive = set(range(p))
    while alive:
        for x in sorted(alive):
            out = [
                int(v) for v in np.flatnonzero(work[x]) if v in alive and not work[v, x]
            ]
            if out:
                continue
            und = [int(v) for v in np.flatnonzero(work[x] & work[:, x]) if v in alive]
            adj = {int(v) for v in np.flatnonzero(work[x] | work[:, x]) if v in alive}
            nbrs = {
                u: {int(w) for w in np.flatnonzero(work[u] | work[:, u])} for u in und
            }
            if all(adj - {u} <= nbrs[u] for u in und):
                for v in adj:
                    dag[v, x] = 1
                for v in adj:
                    work[x, v] = work[v, x] = 0
                alive.discard(x)
                break
        else:  # pragma: no cover - guarded by the operator validity tests
            raise RuntimeError("ges: PDAG has no consistent extension")
    return dag


def _dag_to_cpdag(dag: np.ndarray) -> np.ndarray:
    """The CPDAG of a DAG: v-structures, then Meek's rules R1-R3 to closure."""
    p = dag.shape[0]
    skel = ((dag + dag.T) > 0).astype(int)
    g = skel.copy()
    for b in range(p):
        pa = [int(v) for v in np.flatnonzero(dag[:, b])]
        for i, a in enumerate(pa):
            for c in pa[i + 1 :]:
                if skel[a, c] == 0:
                    g[b, a] = 0
                    g[b, c] = 0
    changed = True
    while changed:
        changed = False
        for a in range(p):
            for b in range(p):
                if not (g[a, b] == 1 and g[b, a] == 1):
                    continue
                # R1: c -> a - b, c and b non-adjacent  =>  a -> b
                r1 = any(
                    g[c, a] == 1 and g[a, c] == 0 and skel[c, b] == 0
                    for c in range(p)
                    if c not in (a, b)
                )
                # R2: a -> c -> b  =>  a -> b
                r2 = any(
                    g[a, c] == 1 and g[c, a] == 0 and g[c, b] == 1 and g[b, c] == 0
                    for c in range(p)
                )
                # R3: a - c1 -> b, a - c2 -> b, c1 and c2 non-adjacent
                mids = [
                    c
                    for c in range(p)
                    if g[a, c] == 1 and g[c, a] == 1 and g[c, b] == 1 and g[b, c] == 0
                ]
                r3 = any(
                    skel[c1, c2] == 0
                    for i, c1 in enumerate(mids)
                    for c2 in mids[i + 1 :]
                )
                if r1 or r2 or r3:
                    g[b, a] = 0
                    changed = True
    out: np.ndarray = g
    return out


def _subsets(items: List[int]) -> Iterator[Set[int]]:
    for k in range(len(items) + 1):
        yield from (set(c) for c in combinations(items, k))


@dataclass
class GESResult(ResultProtocolMixin):
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
        Maximum number of operators applied (forward insertions plus
        backward deletions).

    Returns
    -------
    GESResult
        CPDAG (Markov equivalence class) with ``.adjacency``, ``.bic``,
        ``.edges()`` and ``.to_frame()`` helpers. ``bic`` is the Gaussian
        BIC ``sum_j n log(RSS_j / n) + (|pa_j| + 1) log n`` of any DAG in the
        class (lower is better). The search is the forward-backward GES of
        Chickering (2002); ``pcalg::ges``'s optional turning phase is not run.

    References
    ----------
    [@chickering2002optimal]

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
        labelled = [c for c in names if not pd.api.types.is_numeric_dtype(data[c])]
        if labelled:
            from ..exceptions import MethodIncompatibility

            raise MethodIncompatibility(
                "ges: the score is the Gaussian BIC and needs numeric "
                f"columns; {labelled} are not.",
                recovery_hint=(
                    "For categorical data use "
                    "sp.pc_algorithm(df, ci_test='chi-square')."
                ),
                diagnostics={"non_numeric_columns": labelled},
                alternative_functions=["sp.pc_algorithm"],
            )
        X = data.to_numpy(dtype=float)
    else:
        X = np.asarray(data, dtype=float)
        names = [f"x{i}" for i in range(X.shape[1])]

    n, p = X.shape
    cache: Dict[Tuple[int, FrozenSet[int]], float] = {}

    def local(j: int, pa: set) -> float:
        key = (j, frozenset(pa))
        if key not in cache:
            cache[key] = _bic_score_local(X, j, sorted(pa))
        return cache[key]

    g = np.zeros((p, p), dtype=int)

    def _tie_tol(scale: float) -> float:
        """Gain below which two candidate operators count as tied.

        Score-equivalent moves (``i -> j`` against ``j -> i`` for an isolated
        pair) score the same in exact arithmetic; comparing raw gains with
        ``>`` let last-bit LAPACK rounding, which differs between BLAS builds,
        pick between them, and on a collider the wrong pick cascades (the
        Windows CI leg once returned a complete graph). A candidate has to
        beat the incumbent by more than this margin, so the deterministic
        scan order breaks ties on every platform.
        """
        return 1e-9 * max(1.0, abs(scale))

    n_steps = 0
    scale = sum(local(j, set()) for j in range(p))

    # Forward phase (Chickering 2002, Insert(X, Y, T)): add the best edge to
    # the equivalence class until no insertion lowers the BIC.
    while n_steps < max_iter:
        tol = _tie_tol(scale)
        best = None
        best_gain = 0.0
        for y in range(p):
            pa_y = _parents(g, y)
            ne_y = _undirected_nbrs(g, y)
            for x in range(p):
                if x == y or _adjacent(g, x, y):
                    continue
                na_yx = {v for v in ne_y if _adjacent(g, v, x)}
                t0 = sorted(v for v in ne_y if not _adjacent(g, v, x))
                for t in _subsets(t0):
                    cond = na_yx | t
                    if not _is_clique(g, cond):
                        continue
                    if not _semi_directed_path_blocked(g, y, x, cond):
                        continue
                    base = cond | pa_y
                    gain = local(y, base) - local(y, base | {x})
                    if gain > best_gain + tol:
                        best_gain, best = gain, (x, y, t)
        if best is None:
            break
        x, y, t = best
        g[x, y], g[y, x] = 1, 0
        for v in t:
            g[v, y], g[y, v] = 1, 0
        g = _dag_to_cpdag(_pdag_to_dag(g))
        n_steps += 1

    # Backward phase (Delete(X, Y, H)): remove edges while that lowers it.
    while n_steps < max_iter:
        tol = _tie_tol(scale)
        best = None
        best_gain = 0.0
        for y in range(p):
            pa_y = _parents(g, y)
            ne_y = _undirected_nbrs(g, y)
            for x in range(p):
                # x -> y, or x - y (visited once per orientation)
                if not g[x, y]:
                    continue
                na_yx = {v for v in ne_y if _adjacent(g, v, x)}
                for h in _subsets(sorted(na_yx)):
                    rest = na_yx - h
                    if not _is_clique(g, rest):
                        continue
                    base = (rest | pa_y) - {x}
                    gain = local(y, base | {x}) - local(y, base)
                    if gain > best_gain + tol:
                        best_gain, best = gain, (x, y, h)
        if best is None:
            break
        x, y, h = best
        g[x, y] = g[y, x] = 0
        for v in h:
            g[y, v], g[v, y] = 1, 0
            if g[x, v] and g[v, x]:
                g[x, v], g[v, x] = 1, 0
        g = _dag_to_cpdag(_pdag_to_dag(g))
        n_steps += 1

    adj = g
    dag = _pdag_to_dag(g)
    best_bic = float(
        sum(local(j, {int(v) for v in np.flatnonzero(dag[:, j])}) for j in range(p))
    )

    _result = GESResult(adjacency=adj, names=names, bic=best_bic)
    # attach_provenance never raises (it records nothing on failure).
    from ..output._lineage import attach_provenance as _attach_prov

    _attach_prov(
        _result,
        function="sp.causal_discovery.ges",
        params={"max_iter": max_iter},
        data=data if hasattr(data, "columns") else None,
        overwrite=False,
    )
    return _result
