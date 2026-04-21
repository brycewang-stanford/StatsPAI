"""
LLM-assisted DAG proposal (Kiciman-Sharma 2025, arXiv 2402.11068).

Given a list of variable names + a domain description, propose
candidate causal edges. Default backend is a deterministic heuristic
based on common-sense patterns (time-ordering keywords, prevalence
of confounders in econometrics literature). With an actual LLM
client passed via ``client=`` the function delegates to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np


# Heuristic priors: variable-name patterns → role hints
_TREATMENT_KEYWORDS = {
    'treat', 'treatment', 'policy', 'intervention', 'program',
    'subsidy', 'tax', 'reform', 'shock',
}
_OUTCOME_KEYWORDS = {
    'outcome', 'wage', 'income', 'gdp', 'employment', 'mortality',
    'survival', 'sales', 'revenue', 'response', 'effect',
}
_CONFOUNDER_KEYWORDS = {
    'age', 'sex', 'gender', 'race', 'ethnicity', 'education', 'income',
    'history', 'baseline', 'pretreatment', 'health',
}
_INSTRUMENT_KEYWORDS = {
    'distance', 'lottery', 'birth_quarter', 'random', 'assignment',
    'eligible', 'iv', 'instrument',
}


def _classify_variable(name: str) -> str:
    """Heuristic role classification by name."""
    n = name.lower()
    for kw in _TREATMENT_KEYWORDS:
        if kw in n:
            return "treatment"
    for kw in _OUTCOME_KEYWORDS:
        if kw in n:
            return "outcome"
    for kw in _INSTRUMENT_KEYWORDS:
        if kw in n:
            return "instrument"
    for kw in _CONFOUNDER_KEYWORDS:
        if kw in n:
            return "confounder"
    return "unknown"


@dataclass
class LLMDAGProposal:
    """Result of an LLM (or heuristic) DAG proposal."""
    edges: List[tuple]                # (parent, child)
    roles: dict                       # var_name → role
    rationale: List[str]              # one sentence per edge
    backend: str = "heuristic"
    confidence: float = 0.5

    def to_dag_string(self) -> str:
        """Format as 'A -> B; C -> D' for sp.dag(...)."""
        return "; ".join(f"{p} -> {c}" for p, c in self.edges)

    def summary(self) -> str:
        rows = [
            "LLM DAG Proposal",
            "=" * 42,
            f"  Backend     : {self.backend}",
            f"  Confidence  : {self.confidence:.2f}",
            f"  Edges       : {len(self.edges)}",
            "",
            "  Variable roles:",
        ]
        for v, r in self.roles.items():
            rows.append(f"    {v:>20s} → {r}")
        rows.extend(["", "  Edges + rationale:"])
        for (p, c), why in zip(self.edges, self.rationale):
            rows.append(f"    {p} -> {c}: {why}")
        return "\n".join(rows)


def llm_dag_propose(
    variables: List[str],
    domain: str = "",
    client: Optional[Any] = None,
    seed: int = 0,
) -> LLMDAGProposal:
    """
    Propose a candidate DAG from variable names + domain description.

    Parameters
    ----------
    variables : list of str
        Names of variables in the dataset.
    domain : str
        Free-text domain description (e.g. "labor economics, education
        and earnings"). Helps the LLM but ignored by the heuristic
        backend.
    client : object, optional
        An LLM client implementing ``.complete(prompt: str) -> str``.
        If ``None``, use the deterministic heuristic backend.
    seed : int

    Returns
    -------
    LLMDAGProposal
    """
    if client is not None and hasattr(client, "complete"):
        try:
            prompt = (
                f"Domain: {domain}\n"
                f"Variables: {', '.join(variables)}\n"
                "Propose a DAG (parent -> child edges) consistent with "
                "well-known causal relationships in this domain. Return "
                "edges as a JSON list of [parent, child] pairs."
            )
            raw = client.complete(prompt)
            import json
            edges = [tuple(e) for e in json.loads(raw)]
            roles = {v: _classify_variable(v) for v in variables}
            return LLMDAGProposal(
                edges=edges,
                roles=roles,
                rationale=["LLM proposed"] * len(edges),
                backend=type(client).__name__,
                confidence=0.7,
            )
        except Exception:
            # Fall through to heuristic
            pass

    # Heuristic backend
    roles = {v: _classify_variable(v) for v in variables}
    treatments = [v for v, r in roles.items() if r == "treatment"]
    outcomes = [v for v, r in roles.items() if r == "outcome"]
    confounders = [v for v, r in roles.items() if r == "confounder"]
    instruments = [v for v, r in roles.items() if r == "instrument"]
    unknowns = [v for v, r in roles.items() if r == "unknown"]

    edges = []
    rationale = []
    # Treatment → Outcome
    for t in treatments:
        for o in outcomes:
            edges.append((t, o))
            rationale.append(f"{t} is treatment-named; {o} is outcome-named")
    # Confounders → Treatment, Confounders → Outcome
    for c in confounders:
        for t in treatments:
            edges.append((c, t))
            rationale.append(f"{c} is a baseline/demographic confounder")
        for o in outcomes:
            edges.append((c, o))
            rationale.append(f"{c} likely directly affects outcome")
    # Instruments → Treatment (no direct → Outcome by exclusion restriction)
    for iv in instruments:
        for t in treatments:
            edges.append((iv, t))
            rationale.append(f"{iv} is instrument-named (assumed exclusion)")
    # Unknowns: treat as potential confounders, draw to Outcome only
    for u in unknowns:
        for o in outcomes:
            edges.append((u, o))
            rationale.append(f"{u} role unknown; assumed to affect outcome")

    return LLMDAGProposal(
        edges=edges,
        roles=roles,
        rationale=rationale,
        backend="heuristic",
        confidence=0.5,
    )
