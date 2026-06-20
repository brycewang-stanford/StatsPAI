"""
Pearl's do-calculus — mechanical verification of the three rules on
a user-supplied DAG. Each function returns a boolean plus a structured
reason, letting callers inspect *why* a rule does or doesn't apply.

Rule 1  (insertion/deletion of observations):
    P(y | do(x), z, w) = P(y | do(x), w)
    if (Y ⊥ Z | X, W) in G_{bar X}  (mutilated by deleting edges INTO X)

Rule 2  (action/observation exchange):
    P(y | do(x), do(z), w) = P(y | do(x), z, w)
    if (Y ⊥ Z | X, W) in G_{bar X, underline Z}
    (delete edges into X and out of Z)

Rule 3  (insertion/deletion of actions):
    P(y | do(x), do(z), w) = P(y | do(x), w)
    if (Y ⊥ Z | X, W) in G_{bar X, bar Z(W)}
    where Z(W) = Z \\ An(W)_{G_{bar X}}.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

NodeInput = str | Iterable[str] | None
NodeSet = set[str]


@dataclass
class RuleCheck:
    """Result of checking one do-calculus rule on a DAG.

    Attributes
    ----------
    applicable : bool
        Whether the rule's d-separation condition holds.
    rule : int
        Which rule was checked (1, 2, or 3).
    reason : str
        Human-readable statement of the rule's d-separation condition.
    transformed : str
        The resulting interventional expression; unchanged if not applicable.

    Examples
    --------
    >>> import statspai as sp
    >>> rc = sp.RuleCheck(applicable=True, rule=1, reason="(Y ⊥ Z | X,W)",
    ...                   transformed="P(Y | do(X))")
    >>> bool(rc.applicable), rc.rule
    (True, 1)
    >>> rc.transformed
    'P(Y | do(X))'
    """

    applicable: bool
    rule: int
    reason: str
    transformed: str


def rule1(
    dag: Any,
    Y: NodeInput,
    X: NodeInput,
    Z: NodeInput,
    W: NodeInput = None,
) -> RuleCheck:
    """Check Rule 1: can we *insert or delete observation* of Z?

    Rule 1 licenses ``P(y | do(x), z, w) = P(y | do(x), w)`` when
    ``(Y ⊥ Z | X, W)`` holds in the graph with edges into ``X`` deleted.

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.dag('Z -> X -> Y')
    >>> chk = sp.do_rule1(g, Y='Y', X='X', Z='Z')
    >>> bool(chk.applicable)  # observing Z is irrelevant once we do(X)
    True
    >>> chk.rule
    1
    """
    Y, X, Z, W = _standardize(Y, X, Z, W)
    mutilated = _bar(dag, into=X)
    ok = _d_separated(mutilated, Y, Z, X | W)
    reason = (
        "(Y ⊥ Z | X,W) in G_{bar X}"
        if ok
        else "Y and Z are NOT d-separated given X,W in G_{bar X}"
    )
    transformed = (
        f"P({_s(Y)} | do({_s(X)}), {_s(W)})"
        if ok
        else f"P({_s(Y)} | do({_s(X)}), {_s(Z)}, {_s(W)})  [unchanged]"
    )
    return RuleCheck(
        applicable=ok,
        rule=1,
        reason=reason,
        transformed=transformed,
    )


def rule2(
    dag: Any,
    Y: NodeInput,
    X: NodeInput,
    Z: NodeInput,
    W: NodeInput = None,
) -> RuleCheck:
    """Check Rule 2: can do(Z) be swapped for observing Z?

    Rule 2 licenses ``P(y | do(x), do(z), w) = P(y | do(x), z, w)`` when
    ``(Y ⊥ Z | X, W)`` holds in the graph with edges into ``X`` and edges
    out of ``Z`` deleted (action/observation exchange).

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.dag('X -> Y; Z -> Y')
    >>> chk = sp.do_rule2(g, Y='Y', X='X', Z='Z')
    >>> bool(chk.applicable)  # no back-door from Z to Y, so do(Z) == observe Z
    True
    >>> chk.rule
    2
    """
    Y, X, Z, W = _standardize(Y, X, Z, W)
    mutilated = _bar(_underline(dag, out_of=Z), into=X)
    ok = _d_separated(mutilated, Y, Z, X | W)
    reason = (
        "(Y ⊥ Z | X,W) in G_{bar X, underline Z}"
        if ok
        else "Y and Z not d-separated in G_{bar X, underline Z}"
    )
    transformed = (
        f"P({_s(Y)} | do({_s(X)}), {_s(Z)}, {_s(W)})"
        if ok
        else f"P({_s(Y)} | do({_s(X)}), do({_s(Z)}), {_s(W)})  [unchanged]"
    )
    return RuleCheck(
        applicable=ok,
        rule=2,
        reason=reason,
        transformed=transformed,
    )


def rule3(
    dag: Any,
    Y: NodeInput,
    X: NodeInput,
    Z: NodeInput,
    W: NodeInput = None,
) -> RuleCheck:
    """Check Rule 3: can we delete do(Z)?

    Rule 3 licenses ``P(y | do(x), do(z), w) = P(y | do(x), w)`` when
    ``(Y ⊥ Z | X, W)`` holds in the graph with edges into ``X`` and into
    ``Z(W)`` deleted (insertion/deletion of actions).

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.dag('X -> Y; Z -> W')
    >>> chk = sp.do_rule3(g, Y='Y', X='X', Z='Z')
    >>> bool(chk.applicable)  # Z has no effect on Y, so do(Z) drops out
    True
    >>> chk.rule
    3
    """
    Y, X, Z, W = _standardize(Y, X, Z, W)
    # Z(W) = Z \ ancestors of W in G_{bar X}
    bar_x = _bar(dag, into=X)
    An_W = _ancestors_in(bar_x, W) if W else set()
    Z_W = Z - An_W
    mutilated = _bar(bar_x, into=Z_W)
    ok = _d_separated(mutilated, Y, Z, X | W)
    reason = (
        f"(Y ⊥ Z | X,W) in G_{{bar X, bar Z(W)}}, Z(W) = {sorted(Z_W)}"
        if ok
        else "Y,Z not d-separated in G_{bar X, bar Z(W)}"
    )
    transformed = (
        f"P({_s(Y)} | do({_s(X)}), {_s(W)})"
        if ok
        else f"P({_s(Y)} | do({_s(X)}), do({_s(Z)}), {_s(W)})  [unchanged]"
    )
    return RuleCheck(
        applicable=ok,
        rule=3,
        reason=reason,
        transformed=transformed,
    )


def apply_rules(
    dag: Any,
    Y: NodeInput,
    X: NodeInput,
    Z: NodeInput,
    W: NodeInput = None,
) -> list[RuleCheck]:
    """Try all three rules and return every applicable simplification.

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.dag('Z -> X -> Y')
    >>> checks = sp.do_calculus_apply(g, Y='Y', X='X', Z='Z')
    >>> [c.rule for c in checks]
    [1, 2, 3]
    >>> bool(checks[0].applicable)  # Rule 1 fires on this chain
    True
    """
    return [
        rule1(dag, Y, X, Z, W),
        rule2(dag, Y, X, Z, W),
        rule3(dag, Y, X, Z, W),
    ]


# --------------------------------------------------------------------------- #
#  Graph mutilation helpers
# --------------------------------------------------------------------------- #


def _bar(dag: Any, into: Iterable[str]) -> Any:
    """G_{bar S}: remove all edges INTO nodes in S."""
    from .graph import DAG as _DAG

    S = set(into)
    new = _DAG()
    new._nodes = set(dag._nodes)
    new._edges = {}
    for p, ch in dag._edges.items():
        kept = {c for c in ch if c not in S}
        if kept:
            new._edges[p] = kept
    return new


def _underline(dag: Any, out_of: Iterable[str]) -> Any:
    """G_{underline S}: remove all edges OUT of nodes in S."""
    from .graph import DAG as _DAG

    S = set(out_of)
    new = _DAG()
    new._nodes = set(dag._nodes)
    new._edges = {}
    for p, ch in dag._edges.items():
        if p in S:
            continue
        if ch:
            new._edges[p] = set(ch)
    return new


def _ancestors_in(dag: Any, nodes: Iterable[str]) -> NodeSet:
    result: NodeSet = set()
    stack = list(nodes)
    while stack:
        v = stack.pop()
        if v in result:
            continue
        result.add(v)
        for p, ch in dag._edges.items():
            if v in ch and p not in result:
                stack.append(p)
    return result


def _d_separated(
    dag: Any,
    A: Iterable[str],
    B: Iterable[str],
    C: Iterable[str],
) -> bool:
    """Standard d-separation via moralisation of the ancestral subgraph.

    A classic textbook algorithm — fine for reasonably sized DAGs.
    """
    A = set(A)
    B = set(B)
    C = set(C)
    nodes = A | B | C
    anc = set()
    stack = list(nodes)
    while stack:
        v = stack.pop()
        if v in anc:
            continue
        anc.add(v)
        for p, ch in dag._edges.items():
            if v in ch and p not in anc:
                stack.append(p)
    # Build undirected moralized ancestral graph: add parent--child edges, then
    # "marry" the co-parents of each node (parents that share a common child).
    adj: dict[str, NodeSet] = {v: set() for v in anc}
    parents_in: dict[str, list[str]] = {v: [] for v in anc}
    for p, ch in dag._edges.items():
        if p not in anc:
            continue
        for c in ch:
            if c in anc:
                adj[p].add(c)
                adj[c].add(p)
                parents_in[c].append(p)
    # Moralisation: connect every pair of parents of the same node.
    for ps in parents_in.values():
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = ps[i], ps[j]
                adj[a].add(b)
                adj[b].add(a)
    # Delete nodes in C
    for c in C:
        adj.pop(c, None)
    for v in list(adj):
        adj[v] -= C
    # Any path from A to B?
    for source in A:
        if source not in adj:
            continue
        seen = {source}
        stack = [source]
        while stack:
            v = stack.pop()
            if v in B:
                return False
            for u in adj[v]:
                if u not in seen:
                    seen.add(u)
                    stack.append(u)
    return True


def _standardize(
    Y: NodeInput,
    X: NodeInput,
    Z: NodeInput,
    W: NodeInput,
) -> tuple[NodeSet, NodeSet, NodeSet, NodeSet]:
    def _mk(s: NodeInput) -> NodeSet:
        if s is None:
            return set()
        if isinstance(s, str):
            return {s}
        return set(s)

    return _mk(Y), _mk(X), _mk(Z), _mk(W)


def _s(s: Iterable[str]) -> str:
    return ", ".join(sorted(s)) if s else "∅"
