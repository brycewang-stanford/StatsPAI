"""
Pearl-Bareinboim transportability identification.

Given a selection diagram (a DAG annotated with square nodes that
indicate *differences* across populations), determine whether
P(Y | do(X)) in the target can be written as a do-free functional
of source distributions.

This implementation supports the most common case covered by
**s-admissibility**: if there exists a set Z separating every
square-node S from Y in the mutilated graph G_{bar X}, then the
effect is transportable via the transport formula

    P(Y | do(X))_T = sum_z  P(Y | do(X), Z=z)_S * P(Z=z)_T
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin


@dataclass
class TransportIdentificationResult(ResultProtocolMixin):
    """Result of a Pearl-Bareinboim transportability identification check.

    Attributes
    ----------
    transportable : bool
        Whether ``P(Y | do(X))`` in the target is identifiable from
        source distributions and target covariate margins.
    formula : str
        The transport formula (or ``"NOT IDENTIFIABLE"``).
    admissible_set : frozenset
        The s-admissible conditioning set ``Z`` found (empty if the
        effect transports directly).
    reason : str
        Human-readable justification.

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.dag("X -> Y; W -> Y; W -> X; S -> W")
    >>> res = sp.identify_transport(
    ...     g, treatment="X", outcome="Y", selection_nodes="S")
    >>> type(res).__name__
    'TransportIdentificationResult'
    >>> res.transportable
    True
    >>> sorted(res.admissible_set)
    ['W']
    """

    transportable: bool
    formula: str
    admissible_set: frozenset
    reason: str

    def __repr__(self) -> str:
        status = "transportable" if self.transportable else "NOT transportable"
        return f"TransportIdentificationResult({status}: {self.formula})"


@accepts_aliases(treat="treatment", y="outcome")
def identify_transport(
    dag: Any,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    selection_nodes: str | Iterable[str],
) -> TransportIdentificationResult:
    """Test s-admissibility of some pre-treatment conditioning set.

    Searches for a set ``Z`` of non-descendants of the treatment that
    d-separates every selection node from the outcome given ``X ∪ Z`` in
    the graph with arrows into ``X`` removed. That is a *sufficient*
    condition suited to the covariates-only target design of
    :func:`sp.transport_weights_fn`; it is not the complete
    Bareinboim-Pearl algorithm (R ``causaleffect::transport``), which also
    uses observational outcome data from the target and identifies more
    effects (e.g. ``X -> Y, S -> Y`` via ``P*(Y | X)``).

    Parameters
    ----------
    dag : DAG
        Causal graph.
    treatment : str | Iterable[str]
        Intervention set X.
    outcome : str | Iterable[str]
        Outcome set Y.
    selection_nodes : Iterable[str]
        Nodes (``S1``, ``S2``, ...) pointing to variables whose
        distributions differ across source and target populations.

    Returns
    -------
    TransportIdentificationResult

    Examples
    --------
    Conditioning on ``W`` blocks the ``S -> W -> Y`` selection path, so the
    effect transports:

    >>> import statspai as sp
    >>> g = sp.dag("X -> Y; W -> Y; W -> X; S -> W")
    >>> res = sp.identify_transport(
    ...     g, treatment="X", outcome="Y", selection_nodes="S")
    >>> res.transportable
    True
    >>> sorted(res.admissible_set)
    ['W']
    """

    def _ensure_set(s: str | Iterable[str]) -> set[str]:
        if isinstance(s, str):
            return {s}
        return set(s)

    X = _ensure_set(treatment)
    Y = _ensure_set(outcome)
    S = _ensure_set(selection_nodes)

    # Z must be a non-descendant of X: the transport formula weights by the
    # target *margin* P(Z)_T, which equals P(Z | do(X))_T only then. Before
    # 1.30 mediators were admitted, e.g. X -> Z -> Y, S -> Z returned
    # sum_z P(Y | do(X), z)_S P(z)_T, which is wrong (the correct functional
    # needs P(z | do(X))_T, i.e. outcome-stage data in the target).
    desc_x: set = set()
    for x in X:
        desc_x |= set(dag.descendants(x))
    candidates = set(dag._nodes) - X - Y - S - desc_x
    from ..dag.do_calculus import _bar, _d_separated

    bar_x = _bar(dag, into=X)
    # Try the empty set first, then singletons, then all pairs...
    from itertools import combinations

    for k in range(len(candidates) + 1):
        for Z in combinations(sorted(candidates), k):
            Zset = set(Z)
            ok = all(_d_separated(bar_x, {s}, Y, X | Zset) for s in S)
            if ok:
                if Zset:
                    formula = (
                        f"sum_{{{', '.join(sorted(Zset))}}} "
                        f"P({', '.join(sorted(Y))} | do({', '.join(sorted(X))}), "
                        f"{', '.join(sorted(Zset))})_S "
                        f"* P({', '.join(sorted(Zset))})_T"
                    )
                else:
                    formula = (
                        f"P({', '.join(sorted(Y))} | do({', '.join(sorted(X))}))_S"
                    )
                return TransportIdentificationResult(
                    transportable=True,
                    formula=formula,
                    admissible_set=frozenset(Zset),
                    reason=(
                        f"All selection nodes {sorted(S)} are d-separated "
                        f"from {sorted(Y)} given X∪Z in G_{{bar X}} "
                        f"with Z={sorted(Zset)}."
                    ),
                )

    return TransportIdentificationResult(
        transportable=False,
        formula="NOT IDENTIFIABLE",
        admissible_set=frozenset(),
        reason=(
            f"No s-admissible set found -- some selection node in "
            f"{sorted(S)} remains d-connected to {sorted(Y)} in "
            f"G_{{bar X}} under every candidate conditioning set."
        ),
    )
