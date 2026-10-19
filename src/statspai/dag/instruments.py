"""Instrumental variables read off a causal DAG.

A variable ``Z`` is a *conditional instrument* for the total effect of
``X`` on ``Y`` given a set ``W`` of observed variables when (the graphical
definition of [@pearl2009causality])

(a) ``Z`` is d-connected to ``X`` given ``W``;
(b) ``W`` d-separates ``Z`` from ``Y`` in ``G_c``, the graph without the
    first edge of every proper causal path from ``X`` to ``Y``;
(c) ``W`` holds no descendant of a node on those paths (in particular of
    ``Y``), and neither ``X`` nor ``Y``.

It is *ancestral* when in addition ``W`` consists of ancestors of ``Y`` or
``Z``. A conditional instrument exists exactly when an ancestral one does
(the result behind R ``dagitty::instrumentalVariables``, whose
documentation gives the reference), and the ancestral ones avoid
conditioning on a collider to manufacture the instrument's correlation with
``X``; they are what R ``dagitty::instrumentalVariables`` lists and the
default here. ``ancestral=False`` lists every conditional instrument.

For each instrument the smallest such ``W`` is reported (ties broken by
the sorted variable names); ``dagitty`` reports its nearest separator,
which can be a different valid set.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from ..exceptions import MethodIncompatibility

if TYPE_CHECKING:  # pragma: no cover
    from .graph import DAG

#: Largest candidate conditioning pool searched exhaustively.
MAX_POOL = 16


def _without_first_causal_edges(dag: "DAG", x: str, on_path: Set[str]) -> "DAG":
    """``G_c``: the DAG without the edges from ``x`` into causal paths."""
    from .graph import DAG

    g = DAG()
    for n in dag.nodes:
        g.add_node(n)
    for parent, child in dag.edges:
        if not (parent == x and child in on_path):
            g.add_edge(parent, child)
    g._latent = set(dag._latent)
    return g


def instrumental_variables(
    dag: "DAG",
    exposure: str,
    outcome: str,
    ancestral: bool = True,
) -> List[Dict[str, object]]:
    """Instruments for ``exposure -> outcome`` with their conditioning sets.

    Returns a list of ``{'instrument': name, 'conditioning': [names]}``
    sorted by instrument; an empty list when there is none.
    """
    for v in (exposure, outcome):
        if v not in dag.nodes:
            raise MethodIncompatibility(
                f"instrumental_variables: {v!r} is not a node of the DAG.",
                recovery_hint=f"Nodes: {sorted(dag.observed_nodes)}",
            )
    if exposure == outcome:
        raise MethodIncompatibility(
            "instrumental_variables: exposure and outcome are the same node.",
            recovery_hint="Name two different nodes.",
        )
    x, y = exposure, outcome
    # Nodes on proper causal paths (excluding x). Without such a path the
    # total effect is zero and G_c is G itself.
    on_path = (dag.descendants(x)) & (dag.ancestors(y) | {y})
    gc = _without_first_causal_edges(dag, x, on_path)
    forbidden: Set[str] = {x, y} | dag.descendants(y)
    for v in on_path:
        forbidden |= dag.descendants(v) | {v}
    observed = dag.observed_nodes
    pool_all = sorted(observed - forbidden)
    out: List[Dict[str, object]] = []
    for z in pool_all:
        if ancestral:
            anc = dag.ancestors(y) | dag.ancestors(z)
            pool = [v for v in pool_all if v != z and v in anc]
        else:
            pool = [v for v in pool_all if v != z]
        if len(pool) > MAX_POOL:
            raise MethodIncompatibility(
                f"instrumental_variables: {len(pool)} candidate conditioning "
                f"variables for {z!r}; the exhaustive search stops at "
                f"{MAX_POOL}.",
                recovery_hint="Mark irrelevant variables latent or prune the DAG.",
            )
        found: Optional[List[str]] = None
        for size in range(len(pool) + 1):
            for w in combinations(pool, size):
                ws = set(w)
                if gc.d_separated(z, y, ws) and not dag.d_separated(z, x, ws):
                    found = list(w)
                    break
            if found is not None:
                break
        if found is not None:
            out.append({"instrument": z, "conditioning": found})
    return out
