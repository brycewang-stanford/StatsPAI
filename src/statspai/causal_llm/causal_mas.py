"""
Multi-Agent LLM framework for causal discovery + effect estimation.

Implements the "Causal MAS" (Multi-Agent System) paradigm of

    arXiv:2509.00987 (September 2025).
    "Causal MAS: A Survey of Large Language Model Architectures for
    Discovery and Effect Estimation."

Instead of relying on a single LLM prompt for DAG proposal, multiple
agents with distinct *roles* collaborate over several debate rounds:

  * **Proposer** — proposes candidate edges from prior pattern heuristics
    or LLM knowledge retrieval.
  * **Critic** — adversarial; challenges every proposed edge by
    suggesting alternative explanations / common causes.
  * **Domain expert** — filters edges against substantive knowledge
    encoded in the caller-supplied ``domain`` description.
  * **Synthesiser** — aggregates the debate into a final edge list with
    per-edge confidence scores.

The implementation is *offline-by-default* so it runs without an API key
— each agent has a deterministic heuristic backend.  Supplying an LLM
``client`` (OpenAI / Anthropic / local) with a ``chat()``-compatible
interface reroutes each role to the model; the debate transcript is
preserved for auditability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .llm_dag import (
    _classify_variable,
    _TREATMENT_KEYWORDS,
    _OUTCOME_KEYWORDS,
    _CONFOUNDER_KEYWORDS,
    _INSTRUMENT_KEYWORDS,
)


__all__ = ["causal_mas", "CausalMASResult"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CausalMASResult:
    """Structured output of :func:`causal_mas`.

    Attributes
    ----------
    edges : list of (parent, child)
        Consensus edge list surviving the debate at ``final_threshold``.
    confidence : dict {(p, c): float}
        Fraction of agents / rounds that endorsed the edge (in ``[0, 1]``).
    roles : dict {var: role}
        Proposer-assigned roles (treatment / outcome / confounder / instrument
        / unknown).
    transcript : list of dict
        Round-by-round debate log.  Each entry has keys
        ``{round, agent, action, payload}`` so reviewers can audit how
        the consensus formed.
    rounds : int
    backend : str
        ``'heuristic'`` or the LLM client's repr.
    final_threshold : float
        Confidence cutoff that produced ``edges``.
    """

    edges: List[Tuple[str, str]]
    confidence: Dict[Tuple[str, str], float]
    roles: Dict[str, str]
    transcript: List[Dict[str, Any]]
    rounds: int
    backend: str = "heuristic"
    final_threshold: float = 0.5

    def to_dag_string(self) -> str:
        return "; ".join(f"{p} -> {c}" for p, c in self.edges)

    def summary(self) -> str:
        lines = [
            "Causal MAS — Multi-Agent Causal Discovery",
            "=" * 52,
            f"  Backend          : {self.backend}",
            f"  Debate rounds    : {self.rounds}",
            f"  Confidence thr.  : {self.final_threshold:.2f}",
            f"  Edges accepted   : {len(self.edges)}",
            "",
            "  Variable roles:",
        ]
        for v, r in self.roles.items():
            lines.append(f"    {v:>24s} → {r}")
        lines.append("")
        lines.append("  Top-confidence edges:")
        ranked = sorted(
            self.confidence.items(), key=lambda kv: -kv[1]
        )
        for (p, c), conf in ranked[:12]:
            mark = "*" if (p, c) in self.edges else " "
            lines.append(f"    [{conf:.2f}] {mark} {p} -> {c}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent implementations (heuristic backend)
# ---------------------------------------------------------------------------


def _proposer_heuristic(
    variables: Sequence[str], domain: str, roles: Dict[str, str]
) -> List[Tuple[str, str]]:
    """Propose edges by role-based heuristics."""
    edges: List[Tuple[str, str]] = []
    treat = [v for v, r in roles.items() if r == "treatment"]
    out = [v for v, r in roles.items() if r == "outcome"]
    conf = [v for v, r in roles.items() if r == "confounder"]
    inst = [v for v, r in roles.items() if r == "instrument"]
    # Confounders -> Treatment, Outcome
    for c in conf:
        for t in treat:
            edges.append((c, t))
        for y in out:
            edges.append((c, y))
    # Treatment -> Outcome
    for t in treat:
        for y in out:
            edges.append((t, y))
    # Instrument -> Treatment
    for z in inst:
        for t in treat:
            edges.append((z, t))
    return edges


def _critic_heuristic(
    proposed: List[Tuple[str, str]],
    variables: Sequence[str],
    roles: Dict[str, str],
) -> List[Tuple[str, str]]:
    """Return the subset of edges the critic rejects.

    Rules:
    * Critic rejects edges from outcome to anything (causes don't flow
      back in time for completed outcomes).
    * Critic rejects instrument → outcome edges (exclusion restriction).
    * Critic flags self-loops.
    """
    rej: List[Tuple[str, str]] = []
    for (p, c) in proposed:
        if p == c:
            rej.append((p, c))
            continue
        if roles.get(p) == "outcome":
            rej.append((p, c))
            continue
        if roles.get(p) == "instrument" and roles.get(c) == "outcome":
            rej.append((p, c))
    return rej


def _domain_expert_heuristic(
    proposed: List[Tuple[str, str]],
    domain: str,
) -> List[Tuple[str, str]]:
    """Endorse edges whose parent keyword appears in the domain text."""
    endorsed: List[Tuple[str, str]] = []
    lower = domain.lower() if domain else ""
    for (p, c) in proposed:
        if not lower or p.lower() in lower or c.lower() in lower:
            endorsed.append((p, c))
    return endorsed


def _run_llm_agent(
    client: Any, role: str, prompt: str
) -> str:
    """Adapter to a minimal chat-completion interface."""
    if hasattr(client, "chat"):
        return str(client.chat(role=role, prompt=prompt))
    if hasattr(client, "complete"):
        return str(client.complete(prompt=prompt))
    if callable(client):
        return str(client(prompt))
    raise TypeError(
        "LLM client must expose chat(role, prompt), complete(prompt), or "
        "be callable(prompt)."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def causal_mas(
    variables: Sequence[str],
    *,
    domain: str = "",
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    instruments: Optional[Sequence[str]] = None,
    confounders: Optional[Sequence[str]] = None,
    rounds: int = 3,
    final_threshold: float = 0.5,
    client: Optional[Any] = None,
) -> CausalMASResult:
    """Multi-agent causal discovery over ``variables``.

    Parameters
    ----------
    variables : sequence of str
        Variable names available in the analysis.
    domain : str, default ""
        Free-text domain description (e.g. "cardiovascular epidemiology:
        statins reduce LDL which reduces MI risk").  Used by the
        domain-expert agent.
    treatment, outcome, instruments, confounders : str / sequence, optional
        Role overrides.  If unspecified, roles are inferred from variable
        names via the heuristic in :mod:`statspai.causal_llm.llm_dag`.
    rounds : int, default 3
        Number of propose–critic–synthesise rounds.  Each round lowers
        the threshold by ``1/rounds`` for edges with rising support.
    final_threshold : float, default 0.5
        Minimum fraction of rounds that must endorse an edge for it to
        enter the consensus DAG.
    client : object, optional
        LLM client.  If provided, each agent's heuristic is replaced by
        a prompt to the client.  Client must expose ``chat(role, prompt)``,
        ``complete(prompt)``, or be callable.

    Returns
    -------
    CausalMASResult

    Notes
    -----
    This function is **offline-safe**: without a ``client`` it uses the
    same deterministic heuristics that power
    :func:`sp.causal_llm.llm_dag_propose`.  The value of the multi-agent
    wrapper is procedural: it forces explicit critique and per-edge
    confidence, which improves auditability even without an LLM.

    For effect-estimation dispatch (Level-2 Causal MAS of the survey),
    pipe the resulting DAG into :func:`sp.dag` and then
    :func:`sp.smart.recommend`.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.causal_llm.causal_mas(
    ...     variables=['age','sex','income','treatment','mortality'],
    ...     domain='observational mortality cohort',
    ...     rounds=3, final_threshold=0.5,
    ... )
    >>> ('treatment', 'mortality') in res.edges
    True
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if not (0 <= final_threshold <= 1):
        raise ValueError("final_threshold must be in [0, 1]")
    variables = list(variables)

    # --- Resolve roles ----------------------------------------------------
    roles: Dict[str, str] = {}
    for v in variables:
        roles[v] = _classify_variable(v)
    if treatment is not None:
        roles[treatment] = "treatment"
    if outcome is not None:
        roles[outcome] = "outcome"
    for v in instruments or []:
        roles[v] = "instrument"
    for v in confounders or []:
        roles[v] = "confounder"

    transcript: List[Dict[str, Any]] = []
    endorsement_counts: Dict[Tuple[str, str], int] = {}

    # --- Debate loop ------------------------------------------------------
    for r in range(rounds):
        # 1) Proposer
        if client is None:
            proposed = _proposer_heuristic(variables, domain, roles)
            transcript.append({
                "round": r,
                "agent": "proposer",
                "action": "propose",
                "payload": list(proposed),
            })
        else:
            prompt = (
                f"You are a causal-discovery proposer.  Variables: "
                f"{variables}.  Domain: {domain}.  Current roles: {roles}. "
                "Return one edge per line as 'A -> B'."
            )
            raw = _run_llm_agent(client, "proposer", prompt)
            proposed = _parse_edges(raw)
            transcript.append({
                "round": r,
                "agent": "proposer",
                "action": "propose",
                "payload": list(proposed),
                "raw": raw,
            })

        # 2) Critic
        if client is None:
            rejected = _critic_heuristic(proposed, variables, roles)
        else:
            prompt = (
                f"You are a critic.  Proposed edges: {proposed}.  Roles: "
                f"{roles}.  Reject any edge that violates time-ordering, "
                "exclusion restrictions, or common-sense.  Return "
                "rejections one per line as 'A -> B'."
            )
            rejected = _parse_edges(_run_llm_agent(client, "critic", prompt))
        transcript.append({
            "round": r,
            "agent": "critic",
            "action": "reject",
            "payload": list(rejected),
        })

        # 3) Domain expert
        if client is None:
            endorsed = _domain_expert_heuristic(proposed, domain)
        else:
            prompt = (
                f"You are a domain expert in {domain or 'the relevant field'}. "
                f"Of these proposed edges, endorse those consistent with "
                f"domain knowledge: {proposed}.  Return endorsements one "
                "per line as 'A -> B'."
            )
            endorsed = _parse_edges(
                _run_llm_agent(client, "domain_expert", prompt)
            )
        transcript.append({
            "round": r,
            "agent": "domain_expert",
            "action": "endorse",
            "payload": list(endorsed),
        })

        # 4) Synthesiser: accept = proposed - rejected, boost endorsed
        surviving = [e for e in proposed if e not in rejected]
        for e in surviving:
            endorsement_counts[e] = endorsement_counts.get(e, 0) + 1
        for e in endorsed:
            # Domain endorsement adds an extra half-vote, capped at rounds.
            endorsement_counts[e] = min(
                endorsement_counts.get(e, 0) + 0.5, rounds
            )
        transcript.append({
            "round": r,
            "agent": "synthesiser",
            "action": "score",
            "payload": dict(endorsement_counts),
        })

    # --- Aggregate confidences --------------------------------------------
    confidence: Dict[Tuple[str, str], float] = {
        e: count / rounds for e, count in endorsement_counts.items()
    }
    final_edges = [e for e, c in confidence.items() if c >= final_threshold]

    backend = "heuristic" if client is None else repr(client)
    return CausalMASResult(
        edges=sorted(set(final_edges)),
        confidence=confidence,
        roles=roles,
        transcript=transcript,
        rounds=rounds,
        backend=backend,
        final_threshold=float(final_threshold),
    )


def _parse_edges(text: str) -> List[Tuple[str, str]]:
    """Parse ``A -> B`` edges from free-text agent output."""
    edges: List[Tuple[str, str]] = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line or "->" not in line:
            continue
        p, _, c = line.partition("->")
        p = p.strip().strip("-*• ").split()[-1] if p.strip() else ""
        c = c.strip().split()[0] if c.strip() else ""
        if p and c:
            edges.append((p, c))
    return edges
