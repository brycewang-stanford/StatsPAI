"""Score-based structure learning by hill climbing, and edge stability.

``hill_climb``
    Greedy search over directed acyclic graphs: from the empty graph, apply
    the single arc addition, deletion or reversal that raises the BIC most,
    and stop when none does. Works on categorical data (multinomial BIC) and
    on continuous data (Gaussian BIC). This is the ``hc`` of R's ``bnlearn``.
``bootstrap_edges``
    Refit a structure-learning algorithm on bootstrap resamples and report
    how often each edge, and each direction, comes back. A learned graph is
    a point estimate with no standard error; this is the usual way to say
    which of its edges the data support.

A hill-climbing result is one member of its Markov equivalence class: arcs
whose reversal leaves the score unchanged point in an arbitrary direction.
The ``cpdag`` in the result shows which directions are determined.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._viz import DAGDict

__all__ = ["hill_climb", "bootstrap_edges"]


# --------------------------------------------------------------- local scores
class _DiscreteScore:
    """Multinomial log-likelihood of a node given its parents, less the BIC
    penalty ``0.5 log(n)`` per free parameter."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.n = len(frame)
        self.codes = np.column_stack(
            [pd.Categorical(frame[c]).codes for c in frame.columns]
        ).astype(np.int64)
        if (self.codes < 0).any():
            raise DataInsufficient("hill_climb: missing values in the data.")
        self.levels = np.array(
            [len(pd.Categorical(frame[c]).categories) for c in frame.columns]
        )
        self.penalty = 0.5 * math.log(self.n)

    def __call__(self, node: int, parents: FrozenSet[int]) -> float:
        r = int(self.levels[node])
        if parents:
            pa = sorted(parents)
            config = np.zeros(self.n, dtype=np.int64)
            q = 1
            for j in pa:
                config = config * int(self.levels[j]) + self.codes[:, j]
                q *= int(self.levels[j])
            _, config = np.unique(config, return_inverse=True)
        else:
            config = np.zeros(self.n, dtype=np.int64)
            q = 1
        counts = np.zeros((int(config.max()) + 1, r))
        np.add.at(counts, (config, self.codes[:, node]), 1.0)
        row = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = np.where(counts > 0, counts * np.log(counts / row), 0.0).sum()
        return float(ll - self.penalty * (r - 1) * q)


class _GaussianScore:
    """Gaussian log-likelihood of a node regressed on its parents, less the
    BIC penalty for the coefficients, the intercept and the variance."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.X = frame.to_numpy(dtype=float)
        self.n = self.X.shape[0]
        self.penalty = 0.5 * math.log(self.n)

    def __call__(self, node: int, parents: FrozenSet[int]) -> float:
        y = self.X[:, node]
        Z = np.column_stack([np.ones(self.n)] + [self.X[:, j] for j in sorted(parents)])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        rss = float(np.sum((y - Z @ beta) ** 2))
        k = Z.shape[1]
        # the residual variance with its degrees of freedom, as bnlearn's
        # Gaussian likelihood uses
        s2 = rss / max(self.n - k, 1)
        if s2 <= 0:
            return -math.inf
        ll = -0.5 * self.n * math.log(2 * math.pi * s2) - 0.5 * rss / s2
        return float(ll - self.penalty * (k + 1))


def _is_discrete(col: pd.Series) -> bool:
    return bool(
        isinstance(col.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(col)
        or pd.api.types.is_string_dtype(col)
        or pd.api.types.is_bool_dtype(col)
    )


def _creates_cycle(parents: List[set], frm: int, to: int) -> bool:
    """Whether adding ``frm -> to`` closes a directed cycle."""
    stack, seen = [frm], set()
    while stack:
        v = stack.pop()
        if v == to:
            return True
        if v in seen:
            continue
        seen.add(v)
        stack.extend(parents[v])
    return False


def _dag_to_cpdag(dag: np.ndarray) -> np.ndarray:
    """The equivalence class of a DAG: colliders kept, Meek's rules R1-R3."""
    d = dag.shape[0]
    skel = ((dag + dag.T) > 0).astype(int)
    g = skel.copy()  # g[i, j] = 1: i -> j, or i - j when g[j, i] is 1 too
    for b in range(d):
        pa = [int(v) for v in np.flatnonzero(dag[:, b])]
        for i, a in enumerate(pa):
            for c in pa[i + 1 :]:
                if skel[a, c] == 0:
                    g[b, a] = 0
                    g[b, c] = 0
    changed = True
    while changed:
        changed = False
        for a in range(d):
            for b in range(d):
                if not (g[a, b] == 1 and g[b, a] == 1):
                    continue
                # R1: c -> a - b, c and b non-adjacent  =>  a -> b
                r1 = any(
                    g[c, a] == 1 and g[a, c] == 0 and skel[c, b] == 0
                    for c in range(d)
                    if c not in (a, b)
                )
                # R2: a -> c -> b  =>  a -> b
                r2 = any(
                    g[a, c] == 1 and g[c, a] == 0 and g[c, b] == 1 and g[b, c] == 0
                    for c in range(d)
                )
                # R3: a - c1 -> b, a - c2 -> b, c1 and c2 non-adjacent
                mids = [
                    c
                    for c in range(d)
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
    return g


def hill_climb(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    *,
    data_type: str = "auto",
    max_parents: Optional[int] = None,
    forbidden: Optional[Sequence[Tuple[str, str]]] = None,
    required: Optional[Sequence[Tuple[str, str]]] = None,
    restarts: int = 0,
    perturb: int = 3,
    seed: Optional[int] = None,
    max_iter: int = 10_000,
) -> Dict[str, Any]:
    """Learn a DAG by greedy hill climbing on the BIC.

    Parameters
    ----------
    data : pandas.DataFrame
    variables : sequence of str, optional
        Columns to use; all of them by default.
    data_type : {'auto', 'discrete', 'gaussian'}, default 'auto'
        ``'discrete'`` treats every column as categorical and scores with
        the multinomial BIC (bnlearn's ``bic``); ``'gaussian'`` treats every
        column as continuous and scores linear-Gaussian regressions
        (``bic-g``). ``'auto'`` picks ``'discrete'`` when every column is
        categorical, string or boolean, ``'gaussian'`` when every column is
        numeric, and refuses a mixture: numeric codes such as 0/1 or a
        1-5 scale can be read either way, and the two readings give
        different graphs.
    max_parents : int, optional
        Largest number of parents a node may have. The number of
        parameters of a discrete node grows with the product of its
        parents' levels, so a cap keeps the tables estimable.
    forbidden, required : sequence of (str, str), optional
        Arcs ``(from, to)`` that may not, or must, appear.
    restarts : int, default 0
        Number of random restarts. Each perturbs the best graph so far by
        ``perturb`` random arc changes and climbs again; the best-scoring
        graph is kept. The result can only improve.
    perturb : int, default 3
    seed : int, optional
        Seed of the restarts.
    max_iter : int, default 10000

    Returns
    -------
    dict
        ``dag`` (adjacency DataFrame, ``dag.loc[a, b] = 1`` for ``a -> b``),
        ``edges`` (list of ``(from, to)``), ``cpdag`` (the equivalence
        class: an edge that is 1 in both directions is one whose direction
        the score cannot tell), ``score`` (the BIC of the graph, higher is
        better, on bnlearn's scale), ``node_scores``, ``n_iter``, ``n_obs``,
        ``data_type``, ``variables``. Has ``.plot()``, ``.to_networkx()``
        and ``.to_dot()``.

    Notes
    -----
    The search stops at a local optimum of the score. Different data
    orders, or a restart from another graph, can end elsewhere;
    :func:`bootstrap_edges` shows how much of the result is stable. The
    score is bnlearn's (``score(graph, data, type = "bic")`` or
    ``"bic-g"``) to rounding. The graph need not be the one ``bnlearn::hc``
    returns: both are local optima, reached by breaking ties between
    equally good moves in different orders. On the three data sets of the
    comparison the skeleton was the same each time and the score equal
    twice and 0.5 lower once (on a BIC of -26,135).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> a = rng.normal(size=n)
    >>> b = rng.normal(size=n)
    >>> c = a + b + 0.5 * rng.normal(size=n)       # a -> c <- b
    >>> out = sp.hill_climb(pd.DataFrame({"a": a, "b": b, "c": c}))
    >>> sorted(out["edges"])
    [('a', 'c'), ('b', 'c')]
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("hill_climb: data must be a DataFrame.")
    cols = list(variables) if variables is not None else list(data.columns)
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"hill_climb: columns not in data: {missing}.",
        )
    if len(cols) < 2:
        raise MethodIncompatibility("hill_climb: at least two variables are needed.")
    frame = data[cols].dropna()
    n_dropped = len(data) - len(frame)
    if len(frame) < 10:
        raise DataInsufficient("hill_climb: fewer than 10 complete rows.")

    kind = str(data_type).lower()
    if kind not in ("auto", "discrete", "gaussian"):
        raise MethodIncompatibility(
            f"hill_climb: data_type must be 'auto', 'discrete' or 'gaussian', "
            f"got {data_type!r}."
        )
    if kind == "auto":
        flags = [_is_discrete(frame[c]) for c in cols]
        if all(flags):
            kind = "discrete"
        elif not any(flags):
            kind = "gaussian"
        else:
            raise MethodIncompatibility(
                "hill_climb: the columns mix categorical and numeric types "
                f"(categorical: {[c for c, f in zip(cols, flags) if f]}).",
                recovery_hint="Pass data_type='discrete' to treat every "
                "column as categorical, or encode / drop the categorical "
                "ones and pass data_type='gaussian'.",
            )
    if kind == "gaussian":
        bad = [c for c in cols if not pd.api.types.is_numeric_dtype(frame[c])]
        if bad:
            raise MethodIncompatibility(
                f"hill_climb: data_type='gaussian' needs numeric columns; {bad} "
                "are not."
            )
        score: Callable[[int, FrozenSet[int]], float] = _GaussianScore(frame)
    else:
        score = _DiscreteScore(frame)

    d = len(cols)
    idx = {c: i for i, c in enumerate(cols)}

    def pairs(items: Optional[Sequence[Tuple[str, str]]], what: str) -> set:
        out = set()
        for a, b in items or []:
            if a not in idx or b not in idx:
                raise MethodIncompatibility(
                    f"hill_climb: {what} arc ({a!r}, {b!r}) names a column "
                    "that is not among the variables."
                )
            out.add((idx[a], idx[b]))
        return out

    banned = pairs(forbidden, "forbidden")
    forced = pairs(required, "required")
    if banned & forced:
        raise MethodIncompatibility(
            "hill_climb: an arc is both required and forbidden."
        )

    parents: List[set] = [set() for _ in range(d)]
    for a, b in sorted(forced):
        if _creates_cycle(parents, a, b):
            raise MethodIncompatibility(
                "hill_climb: the required arcs contain a cycle."
            )
        parents[b].add(a)
    cache: Dict[Tuple[int, FrozenSet[int]], float] = {}

    def local(node: int, pa: set) -> float:
        key = (node, frozenset(pa))
        if key not in cache:
            cache[key] = score(node, key[1])
        return cache[key]

    cap = d if max_parents is None else int(max_parents)
    tol = 1e-8
    n_iter = 0

    def total(pa: List[set]) -> float:
        return float(sum(local(i, pa[i]) for i in range(d)))

    def better(gain: float, best: float) -> bool:
        # moves whose gains agree to rounding are ties; the first one
        # found is kept, so float noise does not pick the direction of an
        # arc that the score cannot orient
        return gain > best + 1e-9 * max(1.0, abs(best))

    def reverse_gain(a: int, b: int) -> Optional[float]:
        """Gain of turning a -> b into b -> a, or None if not allowed."""
        if (a, b) in forced or (b, a) in banned or len(parents[a]) >= cap:
            return None
        parents[b].discard(a)
        cyc = _creates_cycle(parents, b, a)
        parents[b].add(a)
        if cyc:
            return None
        return (
            local(b, parents[b] - {a})
            - local(b, parents[b])
            + local(a, parents[a] | {b})
            - local(a, parents[a])
        )

    def best_move() -> Optional[Tuple[str, int, int]]:
        best_gain: float = tol
        best_op: Optional[Tuple[str, int, int]] = None
        for a in range(d):
            for b in range(d):
                if a == b:
                    continue
                if a in parents[b]:
                    if (a, b) in forced:
                        continue
                    gain = local(b, parents[b] - {a}) - local(b, parents[b])
                    if better(gain, best_gain):
                        best_gain, best_op = gain, ("delete", a, b)
                    rev = reverse_gain(a, b)
                    if rev is not None and better(rev, best_gain):
                        best_gain, best_op = rev, ("reverse", a, b)
                elif b not in parents[a]:
                    if (a, b) in banned or len(parents[b]) >= cap:
                        continue
                    if _creates_cycle(parents, a, b):
                        continue
                    gain = local(b, parents[b] | {a}) - local(b, parents[b])
                    if better(gain, best_gain):
                        best_gain, best_op = gain, ("add", a, b)
        return best_op

    def apply(op: Tuple[str, int, int]) -> None:
        kind_, a, b = op
        if kind_ == "add":
            parents[b].add(a)
        elif kind_ == "delete":
            parents[b].discard(a)
        else:
            parents[b].discard(a)
            parents[a].add(b)

    def signature() -> Tuple[FrozenSet[int], ...]:
        return tuple(frozenset(x) for x in parents)

    def escape(depth: int, seen: set) -> Optional[Tuple[str, int, int]]:
        """Walk score-neutral reversals, up to ``depth`` of them, until an
        improving move appears; leave the graph there and return the move.
        If none is found the graph is put back as it was."""
        for a, b in [(a, b) for b in range(d) for a in sorted(parents[b])]:
            rev = reverse_gain(a, b)
            scale = max(1.0, abs(local(b, parents[b])))
            if rev is None or abs(rev) > 1e-9 * scale:
                continue
            apply(("reverse", a, b))
            sig = signature()
            if sig not in seen:
                seen.add(sig)
                op = best_move()
                if op is None and depth > 1:
                    op = escape(depth - 1, seen)
                if op is not None:
                    return op
            apply(("reverse", b, a))
        return None

    def climb() -> None:
        nonlocal n_iter
        for _ in range(int(max_iter)):
            op = best_move()
            if op is None:
                # A plateau: no single change raises the score. Reversing
                # arcs inside the equivalence class leaves the score as it
                # is and can open a move that does raise it (a collider
                # whose arcs were first drawn the wrong way round). Such
                # reversals are kept only if an improving move follows.
                op = escape(3, {signature()})
                if op is None:
                    break
            apply(op)
            n_iter += 1

    climb()
    if restarts:
        # Random restarts: perturb the best graph found by a few random
        # arc changes and climb again. A greedy search ends at a local
        # optimum that depends on how ties were broken on the way.
        rng = np.random.default_rng(seed)
        best_pa = [set(x) for x in parents]
        best_total = total(parents)
        for _ in range(int(restarts)):
            parents[:] = [set(x) for x in best_pa]
            for _k in range(int(perturb)):
                a, b = (int(v) for v in rng.choice(d, 2, replace=False))
                if a in parents[b]:
                    if (a, b) not in forced:
                        parents[b].discard(a)
                elif (
                    b not in parents[a]
                    and (a, b) not in banned
                    and len(parents[b]) < cap
                    and not _creates_cycle(parents, a, b)
                ):
                    parents[b].add(a)
            climb()
            if total(parents) > best_total + tol:
                best_total = total(parents)
                best_pa = [set(x) for x in parents]
        parents[:] = best_pa

    dag = np.zeros((d, d), dtype=int)
    for b in range(d):
        for a in parents[b]:
            dag[a, b] = 1
    node_scores = {cols[i]: local(i, parents[i]) for i in range(d)}
    return DAGDict(
        {
            "dag": pd.DataFrame(dag, index=cols, columns=cols),
            "cpdag": pd.DataFrame(_dag_to_cpdag(dag), index=cols, columns=cols),
            "edges": [
                (cols[a], cols[b]) for a in range(d) for b in range(d) if dag[a, b]
            ],
            "score": float(sum(node_scores.values())),
            "node_scores": node_scores,
            "score_type": "bic" if kind == "discrete" else "bic-g",
            "data_type": kind,
            "variables": cols,
            "n_obs": int(len(frame)),
            "n_dropped_missing": int(n_dropped),
            "n_iter": int(n_iter),
        }
    )


# ------------------------------------------------------------- edge stability
def _directed_and_undirected(result: Any, names: List[str]) -> Tuple[set, set]:
    """Directed arcs and undirected edges of any discovery result."""
    directed: set = set()
    undirected: set = set()
    if isinstance(result, dict) and "cpdag" in result and "undirected_edges" in result:
        directed = {tuple(e) for e in result["edges"]}
        undirected = {frozenset(e) for e in result["undirected_edges"]}
        return directed, undirected
    if isinstance(result, dict) and "dag" in result:
        # read the equivalence class: an arc the score cannot orient is
        # counted half each way, not as whichever way the search left it
        g = result["cpdag"]
        for a in g.index:
            for b in g.columns:
                if g.loc[a, b] and g.loc[b, a]:
                    undirected.add(frozenset((a, b)))
                elif g.loc[a, b]:
                    directed.add((a, b))
        return directed, undirected
    if isinstance(result, dict) and "edges" in result:
        return {tuple(e[:2]) for e in result["edges"]}, undirected
    edges = getattr(result, "edges", None)
    if callable(edges):
        edges = edges()
    if edges is None:
        raise MethodIncompatibility(
            "bootstrap_edges: cannot read edges from the algorithm's result.",
            recovery_hint="Return a dict with 'edges' (and optionally "
            "'undirected_edges'), as sp.pc_algorithm does.",
        )
    for e in edges:
        if len(e) == 3 and isinstance(e[1], str) and not _is_number(e[2]):
            a, mark, b = e  # a PAG edge such as ("X", "o->", "Y")
            if mark == "-->":
                directed.add((a, b))
            elif mark == "<--":
                directed.add((b, a))
            else:
                undirected.add(frozenset((a, b)))
        else:
            directed.add((e[0], e[1]))
    return directed, undirected


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating))


def bootstrap_edges(
    data: pd.DataFrame,
    method: Any = "pc",
    *,
    n_boot: int = 200,
    seed: Optional[int] = None,
    threshold: float = 0.5,
    **kwargs: Any,
) -> pd.DataFrame:
    """How often each edge survives a nonparametric bootstrap.

    The algorithm is refitted on ``n_boot`` resamples of the rows (drawn
    with replacement, same size). For each pair of variables the result
    gives the share of resamples in which they are adjacent and, among
    those, the share in which the arc points each way.

    Parameters
    ----------
    data : pandas.DataFrame
    method : {'pc', 'hill_climb', 'mmhc', 'ges', 'fci', 'lingam', 'notears'} or callable
        The structure-learning algorithm. A callable receives a DataFrame
        (and ``**kwargs``) and returns a discovery result.
    n_boot : int, default 200
    seed : int, optional
    threshold : float, default 0.5
        Edges with ``strength`` at or above it are flagged ``stable``.
    **kwargs
        Passed to the algorithm, e.g. ``alpha=0.01`` or
        ``data_type='discrete'``.

    Returns
    -------
    pandas.DataFrame
        One row per ordered pair that was adjacent at least once: ``from``,
        ``to``, ``strength`` (share of resamples with the two adjacent, in
        either direction), ``direction`` (share of those in which the arc
        runs ``from -> to``; an undirected or ambiguous edge counts one
        half each way), ``stable``. Sorted by ``strength``. The frame's
        ``attrs`` hold ``n_boot``, ``n_failed`` and ``method``.

    Notes
    -----
    This is ``bnlearn::boot.strength``: ``strength`` and ``direction`` are
    its two columns. Strength near one says the adjacency is robust to
    sampling variation; it says nothing about whether the assumptions of
    the algorithm (no hidden common causes, for ``pc`` and ``hill_climb``)
    hold. A direction near one half means the data do not orient the edge.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 600
    >>> a = rng.normal(size=n)
    >>> b = rng.normal(size=n)
    >>> c = a + b + 0.5 * rng.normal(size=n)
    >>> df = pd.DataFrame({"a": a, "b": b, "c": c})
    >>> out = sp.bootstrap_edges(df, "pc", n_boot=30, seed=1)
    >>> row = out[(out["from"] == "a") & (out["to"] == "c")].iloc[0]
    >>> bool(row["strength"] > 0.9 and row["direction"] > 0.5)
    True
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("bootstrap_edges: data must be a DataFrame.")
    if n_boot < 2:
        raise MethodIncompatibility("bootstrap_edges: n_boot must be at least 2.")
    if callable(method):
        fit: Callable[..., Any] = method
        label = getattr(method, "__name__", "callable")
    else:
        from . import fci, ges, lingam, notears, pc_algorithm
        from .mmhc import mmhc

        table: Dict[str, Callable[..., Any]] = {
            "pc": pc_algorithm,
            "hill_climb": hill_climb,
            "hc": hill_climb,
            "mmhc": mmhc,
            "ges": ges,
            "fci": fci,
            "lingam": lingam,
            "notears": notears,
        }
        label = str(method).lower()
        if label not in table:
            raise MethodIncompatibility(
                f"bootstrap_edges: unknown method {method!r}.",
                recovery_hint=f"Use one of {sorted(table)} or pass a callable.",
            )
        fit = table[label]

    rng = np.random.default_rng(seed)
    n = len(data)
    names = [str(c) for c in data.columns]
    adjacent: Dict[FrozenSet[str], float] = {}
    forward: Dict[Tuple[str, str], float] = {}
    n_ok = n_failed = 0
    import warnings

    for _ in range(int(n_boot)):
        sample = data.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fit(sample, **kwargs)
        except (ValueError, np.linalg.LinAlgError):
            # a resample can be degenerate (a constant column, a singular
            # correlation matrix); it is counted, not hidden
            n_failed += 1
            continue
        directed, undirected = _directed_and_undirected(res, names)
        n_ok += 1
        both = {frozenset(e) for e in directed if (e[1], e[0]) in directed}
        for a, b in directed:
            pair = frozenset((a, b))
            if pair in both:
                continue
            adjacent[pair] = adjacent.get(pair, 0.0) + 1.0
            forward[(a, b)] = forward.get((a, b), 0.0) + 1.0
        for pair in undirected | both:
            a, b = sorted(pair)
            adjacent[pair] = adjacent.get(pair, 0.0) + 1.0
            forward[(a, b)] = forward.get((a, b), 0.0) + 0.5
            forward[(b, a)] = forward.get((b, a), 0.0) + 0.5
    if n_ok == 0:
        raise DataInsufficient(
            "bootstrap_edges: the algorithm failed on every resample.",
        )
    rows = []
    for pair, count in adjacent.items():
        a, b = sorted(pair)
        for frm, to in ((a, b), (b, a)):
            rows.append(
                {
                    "from": frm,
                    "to": to,
                    "strength": count / n_ok,
                    "direction": forward.get((frm, to), 0.0) / count,
                }
            )
    out = pd.DataFrame(rows, columns=["from", "to", "strength", "direction"])
    out["stable"] = out["strength"] >= threshold
    out = out.sort_values(
        ["strength", "direction", "from", "to"], ascending=[False, False, True, True]
    ).reset_index(drop=True)
    out.attrs.update({"n_boot": int(n_ok), "n_failed": int(n_failed), "method": label})
    return out
