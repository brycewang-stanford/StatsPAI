"""
Single World Intervention Graphs (Richardson & Robins 2013).

SWIGs are the bridge between Pearl's SCM/do-operator and the
counterfactual language used by Hernan-Robins. Given a DAG G and an
intervention do(X=x), the SWIG is built by *node-splitting*:

- Each intervened node X is split into two: an ``observation half`` X
  that inherits incoming edges, and an ``action half`` X(x) that
  inherits outgoing edges.
- All other nodes carry an explicit potential-outcome label Y(x),
  representing the counterfactual world under the intervention.

This makes counterfactual independencies (``modularity``) and
exchangeability assumptions explicit — e.g. Y(x) ⊥ X | L is just
d-separation on the SWIG.

References
----------
Richardson & Robins (2013). "Single World Intervention Graphs (SWIGs):
A Unification of the Counterfactual and Graphical Approaches to Causality."
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class SWIGGraph:
    """Single World Intervention Graph for a DAG under ``do(X=x)``.

    Attributes
    ----------
    parent : DAG
        Source DAG.
    intervention : dict[str, str]
        Variable → value label.
    nodes : set[str]
        All SWIG nodes (split halves + potential outcomes).
    edges : dict[str, set[str]]
        Adjacency map on SWIG nodes.

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.dag("L -> X; L -> Y; X -> Y")
    >>> sw = sp.swig(g, {"X": "x"})
    >>> isinstance(sw, sp.SWIGGraph)
    True
    >>> len(sw.nodes)
    4
    >>> print(sw.ascii())  # doctest: +NORMALIZE_WHITESPACE
    SWIG under do({'X': 'x'}):
      L(X=x) -> X
      L(X=x) -> Y(X=x)
      X(x) -> Y(X=x)
    """

    def __init__(self, parent: Any, intervention: Mapping[str, str]) -> None:
        self.parent = parent
        self.intervention = dict(intervention)
        self.nodes: set[str] = set()
        self.edges: dict[str, set[str]] = {}
        self._build()

    def _build(self) -> None:
        V = set(self.parent._nodes)
        X = set(self.intervention)

        def _label(v: str) -> str:
            if v in X:
                return v  # observation half
            # potential outcome labeled by intervention arguments:
            args = ",".join(f"{k}={val}" for k, val in self.intervention.items())
            return f"{v}({args})"

        for v in V:
            self.nodes.add(_label(v))
            if v in X:
                action_half = f"{v}({self.intervention[v]})"
                self.nodes.add(action_half)

        for p, children in self.parent._edges.items():
            p_lab = _label(p)
            for c in children:
                c_lab = _label(c)
                # Outgoing from X comes from its action half:
                src = f"{p}({self.intervention[p]})" if p in X else p_lab
                self.edges.setdefault(src, set()).add(c_lab)

    def counterfactual_nodes(self) -> set[str]:
        """Return only the potential-outcome / action-half labels."""
        return {v for v in self.nodes if "(" in v or v in self.intervention}

    def ascii(self) -> str:
        """Compact edge-list representation."""
        lines = [f"SWIG under do({self.intervention}):"]
        for p, ch in sorted(self.edges.items()):
            for c in sorted(ch):
                lines.append(f"  {p} -> {c}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SWIG(n={len(self.nodes)}, intervention={self.intervention})"


def swig(
    dag: Any,
    intervention: Mapping[str, str] | Iterable[str] | str,
) -> SWIGGraph:
    """Construct a SWIG for ``dag`` under ``intervention``.

    Parameters
    ----------
    dag : DAG
    intervention : dict[str, str] | Iterable[str]
        Either a mapping var → intervention label (``{"X": "x"}``),
        or an iterable of variable names (labels default to lowercase).

    Returns
    -------
    SWIGGraph

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.dag("L -> X; L -> Y; X -> Y")  # confounded triangle
    >>> sw = sp.swig(g, {"X": "x"})
    >>> repr(sw)
    "SWIG(n=4, intervention={'X': 'x'})"
    >>> "Y(X=x)" in sw.nodes  # potential outcome under do(X=x)
    True
    >>> sorted(sw.counterfactual_nodes())
    ['L(X=x)', 'X', 'X(x)', 'Y(X=x)']
    """
    if isinstance(intervention, Mapping):
        mapping = dict(intervention)
    elif isinstance(intervention, str):
        mapping = {intervention: intervention.lower()}
    else:
        mapping = {v: v.lower() for v in intervention}
    return SWIGGraph(dag, mapping)
