"""Integrity guard for the ``FunctionSpec.inherits_from`` graph.

Agent cards merge metadata up the ``inherits_from`` chain (a variant like
``ivreg`` inherits ``iv``'s assumptions / failure-modes). If that graph ever
grows a dangling parent, a self-reference, or a cycle, the merge either drops
metadata silently or recurses forever. These checks lock the graph healthy:

* every ``inherits_from`` target is itself a registered function,
* no spec inherits from itself,
* the graph is acyclic,
* chain depth stays shallow (a deep chain usually signals a modelling mistake;
  the bound is generous and only trips on something clearly wrong).
"""

import pytest

from statspai import registry as R


def _edges():
    R._ensure_full_registry()
    reg = R._REGISTRY
    return reg, {
        name: spec.inherits_from
        for name, spec in reg.items()
        if getattr(spec, "inherits_from", None)
    }


def test_no_dangling_parents():
    reg, edges = _edges()
    dangling = [(c, p) for c, p in edges.items() if p not in reg]
    assert not dangling, (
        f"inherits_from targets that are not registered functions: {dangling}. "
        f"Agent-card metadata merge would silently drop the parent."
    )


def test_no_self_reference():
    _reg, edges = _edges()
    selfref = [c for c, p in edges.items() if c == p]
    assert not selfref, f"specs that inherit from themselves: {selfref}"


def test_graph_is_acyclic():
    _reg, edges = _edges()
    cycles = []
    for start in edges:
        seen = {start}
        node = edges.get(start)
        while node is not None:
            if node in seen:
                cycles.append(start)
                break
            seen.add(node)
            node = edges.get(node)
    assert not cycles, (
        f"inherits_from cycles reachable from: {sorted(set(cycles))}. "
        f"Agent-card merge would recurse without terminating."
    )


def test_chain_depth_is_shallow():
    _reg, edges = _edges()

    def depth(n, seen):
        p = edges.get(n)
        if p is None or p in seen:
            return 0
        return 1 + depth(p, seen | {n})

    deepest = max((depth(n, set()) for n in edges), default=0)
    assert deepest <= 4, (
        f"inherits_from chain depth {deepest} exceeds the sane bound (4). "
        f"A long inheritance chain usually means a variant should point at a "
        f"closer parent."
    )
