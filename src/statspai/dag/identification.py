"""
Shpitser-Pearl ID algorithm for non-parametric identification of
interventional distributions P(Y | do(X)) in semi-Markovian models.

Reference
---------
Shpitser, I. & Pearl, J. (2006). "Identification of Joint Interventional
Distributions in Recursive Semi-Markovian Causal Models." AAAI.

Tian, J. & Pearl, J. (2002). "A General Identification Condition for
Causal Effects."
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Set, List
from itertools import combinations


@dataclass
class IdentificationResult:
    """Outcome of an identification query.

    Attributes
    ----------
    identifiable : bool
        True iff P(Y | do(X)) is identifiable from the observed distribution.
    estimand : str
        Do-free formula when identifiable; a structured hedge otherwise.
    c_components : list[set[str]]
        The c-components of the ancestral semi-Markovian graph G[An(Y)].
    hedge : tuple[frozenset, frozenset] | None
        Witness C-forest pair (F, F') that proves non-identifiability.
    explanation : str
        Human-readable proof / refutation.
    """

    identifiable: bool
    estimand: str
    c_components: list
    hedge: tuple | None
    explanation: str

    def __repr__(self) -> str:
        status = "identifiable" if self.identifiable else "NOT identifiable"
        return f"IdentificationResult({status}: {self.estimand})"


def identify(dag, treatment, outcome) -> IdentificationResult:
    """Run Shpitser-Pearl ID algorithm on ``dag``.

    Parameters
    ----------
    dag : DAG
        ``statspai.dag.DAG`` instance, possibly with latent nodes
        ``_L_*`` (representing bidirected edges).
    treatment : str | Iterable[str]
        Set of variables X being intervened on.
    outcome : str | Iterable[str]
        Set of outcome variables Y.

    Returns
    -------
    IdentificationResult
    """
    X = frozenset({treatment} if isinstance(treatment, str) else set(treatment))
    Y = frozenset({outcome} if isinstance(outcome, str) else set(outcome))

    V = frozenset(dag._nodes)
    if not X.issubset(V) or not Y.issubset(V):
        missing = (X | Y) - V
        raise KeyError(f"Variables not in DAG: {missing}")

    observable = frozenset(v for v in V if not _is_latent(v))
    X &= observable
    Y &= observable

    ccs = _c_components(dag, observable)

    try:
        estimand = _ID(Y, X, dag, observable)
        return IdentificationResult(
            identifiable=True,
            estimand=estimand,
            c_components=[set(c) for c in ccs],
            hedge=None,
            explanation=(
                f"Query P({_fmt(Y)} | do({_fmt(X)})) is identified via "
                f"repeated c-component factorization; the resulting "
                f"expression involves only observed joint distributions."
            ),
        )
    except _NotIdentifiable as exc:
        return IdentificationResult(
            identifiable=False,
            estimand=f"hedge({_fmt(exc.F)}, {_fmt(exc.F_prime)})",
            c_components=[set(c) for c in ccs],
            hedge=(exc.F, exc.F_prime),
            explanation=(
                f"P({_fmt(Y)} | do({_fmt(X)})) is NOT identifiable. "
                f"Witness hedge: F={sorted(exc.F)}, F'={sorted(exc.F_prime)}."
            ),
        )


# --------------------------------------------------------------------------- #
#  Core recursion (Shpitser-Pearl Alg. 1)
# --------------------------------------------------------------------------- #

class _NotIdentifiable(Exception):
    def __init__(self, F, F_prime):
        self.F = frozenset(F)
        self.F_prime = frozenset(F_prime)


def _ID(Y: frozenset, X: frozenset, dag, V: frozenset) -> str:
    """Non-parametric identification over observed set V. Returns a
    string-form estimand (sum / product of conditional densities)."""
    # Line 1: if X empty -> marginalize
    if not X:
        extra = V - Y
        if extra:
            return f"sum_{{{_fmt(extra)}}} P({_fmt(V)})"
        return f"P({_fmt(Y)})"

    # Line 2: restrict to ancestors of Y
    An_Y = _ancestors_in(dag, Y, V)
    if An_Y != V:
        return _ID(Y, X & An_Y, dag, An_Y)

    # Line 3: insert do-null term
    W = (V - X) - _ancestors_in(dag, Y, V - X)
    if W:
        return _ID(Y, X | W, dag, V)

    # Line 4: c-component decomposition
    G_minus_X = _subgraph_without_nodes(dag, X)
    ccs = _c_components(G_minus_X, V - X)
    if len(ccs) > 1:
        parts = []
        for S in ccs:
            parts.append(_ID(frozenset(S), V - frozenset(S), dag, V))
        extra = (V - Y - X)
        body = " * ".join(parts)
        if extra:
            return f"sum_{{{_fmt(extra)}}} [{body}]"
        return body

    S = ccs[0]
    # Line 5: entire graph is one c-component
    CV = _c_components(dag, V)
    if len(CV) == 1 and frozenset(CV[0]) == V:
        raise _NotIdentifiable(V, S)

    # Line 6: S is itself a c-component of the whole graph
    if frozenset(S) in [frozenset(c) for c in CV]:
        order = _topo_order(dag, V)
        pieces = []
        for v in order:
            if v in S:
                pre = _prior(order, v)
                pieces.append(f"P({v} | {_fmt(pre)})" if pre else f"P({v})")
        extra = frozenset(S) - Y
        body = " * ".join(pieces)
        if extra:
            return f"sum_{{{_fmt(extra)}}} [{body}]"
        return body

    # Line 7: S strictly contained in some c-component S' of full graph
    for Sp in CV:
        if set(S).issubset(Sp) and set(S) != set(Sp):
            V_new = frozenset(Sp)
            X_new = X & V_new
            return _ID(Y, X_new, _subgraph(dag, V_new), V_new)

    raise _NotIdentifiable(V, S)


# --------------------------------------------------------------------------- #
#  Graph utilities (latent-aware)
# --------------------------------------------------------------------------- #

def _is_latent(node: str) -> bool:
    return node.startswith("_L_") or node.startswith("U_")


def _observed_parents(dag, node: str) -> Set[str]:
    out = set()
    for p, children in dag._edges.items():
        if node in children and not _is_latent(p):
            out.add(p)
    return out


def _bidirected_neighbors(dag, node: str) -> Set[str]:
    """Return observed nodes sharing a latent parent with ``node``."""
    latent_parents = [
        p for p, ch in dag._edges.items() if node in ch and _is_latent(p)
    ]
    out = set()
    for L in latent_parents:
        out |= {v for v in dag._edges.get(L, set()) if not _is_latent(v)}
    out.discard(node)
    return out


def _c_components(dag, observed: Iterable[str]) -> List[Set[str]]:
    """Bidirected-connected components restricted to ``observed``."""
    observed = set(observed)
    unvisited = set(observed)
    components: List[Set[str]] = []
    while unvisited:
        seed = next(iter(unvisited))
        stack = [seed]
        comp: Set[str] = set()
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)
            neigh = _bidirected_neighbors(dag, v) & observed
            stack.extend(neigh - comp)
        components.append(comp)
        unvisited -= comp
    return components


def _ancestors_in(dag, nodes: Iterable[str], universe: Iterable[str]) -> frozenset:
    universe = set(universe)
    result: Set[str] = set()
    stack = list(nodes)
    while stack:
        v = stack.pop()
        if v in result or v not in universe:
            continue
        result.add(v)
        stack.extend(_observed_parents(dag, v) & universe)
    return frozenset(result)


def _subgraph(dag, V: Iterable[str]):
    """Return a fresh DAG restricted to nodes in V (plus relevant latents)."""
    from .graph import DAG as _DAG
    sub = _DAG()
    keep_obs = set(V)
    for v in keep_obs:
        sub.add_node(v)
    for p, ch in dag._edges.items():
        if _is_latent(p):
            preserved = ch & keep_obs
            if len(preserved) >= 2:
                sub._nodes.add(p)
                sub._edges.setdefault(p, set()).update(preserved)
        elif p in keep_obs:
            for c in ch:
                if c in keep_obs and not _is_latent(c):
                    sub.add_edge(p, c)
    return sub


def _subgraph_without_nodes(dag, remove: Iterable[str]):
    V = set(dag._nodes) - set(remove)
    V = {v for v in V if not _is_latent(v)}
    return _subgraph(dag, V)


def _topo_order(dag, V: Iterable[str]) -> list:
    V = set(V)
    indeg = {v: 0 for v in V}
    for p, ch in dag._edges.items():
        if _is_latent(p):
            continue
        for c in ch:
            if p in V and c in V:
                indeg[c] += 1
    stack = [v for v, d in indeg.items() if d == 0]
    order: list = []
    while stack:
        v = stack.pop(0)
        order.append(v)
        for c in dag._edges.get(v, set()):
            if c in V and not _is_latent(c):
                indeg[c] -= 1
                if indeg[c] == 0:
                    stack.append(c)
    if len(order) != len(V):
        order.extend(sorted(V - set(order)))
    return order


def _prior(order: list, v: str) -> list:
    return order[: order.index(v)] if v in order else []


def _fmt(s: Iterable[str]) -> str:
    return ", ".join(sorted(s))
