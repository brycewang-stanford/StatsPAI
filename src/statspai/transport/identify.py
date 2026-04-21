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
from typing import Iterable


@dataclass
class TransportIdentificationResult:
    transportable: bool
    formula: str
    admissible_set: frozenset
    reason: str

    def __repr__(self) -> str:
        status = "transportable" if self.transportable else "NOT transportable"
        return f"TransportIdentificationResult({status}: {self.formula})"


def identify_transport(
    dag,
    treatment,
    outcome,
    selection_nodes,
) -> TransportIdentificationResult:
    """Test s-admissibility of some conditioning set.

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
    """
    def _ensure_set(s):
        if isinstance(s, str):
            return {s}
        return set(s)

    X = _ensure_set(treatment)
    Y = _ensure_set(outcome)
    S = _ensure_set(selection_nodes)

    candidates = set(dag._nodes) - X - Y - S
    from ..dag.do_calculus import _bar, _d_separated

    bar_x = _bar(dag, into=X)
    # Try the empty set first, then singletons, then all pairs...
    from itertools import combinations
    for k in range(len(candidates) + 1):
        for Z in combinations(sorted(candidates), k):
            Zset = set(Z)
            ok = all(
                _d_separated(bar_x, {s}, Y, X | Zset)
                for s in S
            )
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
