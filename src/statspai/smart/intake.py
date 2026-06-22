"""Design-intake contract for agentic method routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DESIGN_INTAKE_OUTCOMES = (
    "matched",
    "ambiguous",
    "not_identifiable_yet",
    "not_implemented",
)


@dataclass
class IntakeResult:
    """Structured routing outcome before estimator selection.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.IntakeResult(outcome="matched", recommended_design="rd")
    >>> result.to_dict()["recommended_design"]
    'rd'
    """

    outcome: str
    recommended_design: Optional[str] = None
    candidate_designs: List[str] = field(default_factory=list)
    why: str = ""
    required_data: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    deciding_question: Optional[str] = None
    next_step: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in DESIGN_INTAKE_OUTCOMES:
            raise ValueError(
                f"outcome must be one of {DESIGN_INTAKE_OUTCOMES}; "
                f"got {self.outcome!r}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "recommended_design": self.recommended_design,
            "candidate_designs": list(self.candidate_designs),
            "why": self.why,
            "required_data": list(self.required_data),
            "assumptions": list(self.assumptions),
            "risks": list(self.risks),
            "deciding_question": self.deciding_question,
            "next_step": self.next_step,
        }

    def summary(self) -> str:
        lines = [f"Outcome: {self.outcome}"]
        if self.recommended_design:
            lines.append(f"Recommended design: {self.recommended_design}")
        if self.candidate_designs:
            lines.append(f"Candidates: {', '.join(self.candidate_designs)}")
        if self.why:
            lines.append(f"Why: {self.why}")
        if self.deciding_question:
            lines.append(f"Deciding question: {self.deciding_question}")
        if self.next_step:
            lines.append(f"Next step: {self.next_step}")
        return "\n".join(lines)


def design_intake(
    *,
    estimand: Optional[str] = None,
    assignment: Optional[str] = None,
    data_topology: Optional[str] = None,
    controls: Optional[str] = None,
    identification_support: Optional[str] = None,
    needs: Optional[str] = None,
) -> IntakeResult:
    """Route design facts to a method-selection status.

    This is a pre-``recommend`` intake helper.  It does not fit models and it
    does not pretend missing identification facts are harmless.

    Examples
    --------
    >>> import statspai as sp
    >>> intake = sp.design_intake(
    ...     assignment="threshold cutoff",
    ...     data_topology="cross-section",
    ...     identification_support="continuity",
    ... )
    >>> intake.outcome
    'matched'
    >>> intake.recommended_design
    'rd'
    """
    missing = [
        name
        for name, value in {
            "assignment": assignment,
            "data_topology": data_topology,
            "identification_support": identification_support,
        }.items()
        if not value
    ]
    if missing:
        return IntakeResult(
            outcome="not_identifiable_yet",
            why=f"Missing design fact(s): {', '.join(missing)}.",
            deciding_question=f"What is the {missing[0].replace('_', ' ')}?",
            next_step="Supply the missing design fact before choosing an estimator.",
        )

    a = str(assignment).lower()
    topo = str(data_topology).lower()
    ctrl = (controls or "").lower()
    support = str(identification_support).lower()
    need = (needs or "").lower()

    if "cutoff" in a or "threshold" in a:
        return IntakeResult(
            outcome="matched",
            recommended_design="rd",
            why="Treatment is assigned by a cutoff or threshold.",
            required_data=["outcome", "running variable", "cutoff", "treatment rule"],
            assumptions=["continuity at cutoff", "no manipulation near cutoff"],
            risks=["heaping/manipulation", "bandwidth sensitivity"],
            next_step="Call sp.recommend(..., design='rd') or an RD estimator.",
        )
    if "instrument" in a:
        return IntakeResult(
            outcome="matched",
            recommended_design="iv",
            why="The assignment story relies on an instrument.",
            required_data=["outcome", "treatment", "instrument", "covariates"],
            assumptions=["relevance", "exclusion", "monotonicity"],
            risks=["weak instruments", "invalid exclusion restriction"],
            next_step="Call sp.recommend(..., design='iv') and inspect weak-IV checks.",
        )
    if "stagger" in a or "adoption" in a:
        return IntakeResult(
            outcome="matched",
            recommended_design="did",
            why="Units adopt treatment at different times in panel data.",
            required_data=["unit id", "time", "outcome", "first treated period"],
            assumptions=["parallel trends", "no anticipation", "no interference"],
            risks=["forbidden comparisons", "pre-trend placebo failures"],
            next_step="Call sp.recommend(..., design='did') or Callaway-Sant'Anna.",
        )
    if "known intervention time" in a or "policy change" in a:
        if "donor" in ctrl or "wide" in topo:
            return IntakeResult(
                outcome="ambiguous",
                candidate_designs=["synth", "did", "interrupted_time_series"],
                why=(
                    "A known intervention time with donor/control information "
                    "can map to several quasi-experimental designs."
                ),
                deciding_question=(
                    "Do untreated donor units need to form the counterfactual "
                    "trajectory?"
                ),
                next_step=(
                    "Answer the deciding question, then call sp.recommend with "
                    "the resolved design."
                ),
            )
        return IntakeResult(
            outcome="matched",
            recommended_design="interrupted_time_series",
            why="A single treated time series has a known intervention time.",
            required_data=["time", "outcome", "intervention time"],
            assumptions=["stable pre-period trend", "no concurrent shocks"],
            risks=["seasonality", "structural breaks unrelated to treatment"],
            next_step="Use sp.causal_impact or time-series intervention tools.",
        )
    if "observed treatment" in a or "selection" in support or "overlap" in support:
        return IntakeResult(
            outcome="matched",
            recommended_design="observational",
            why=(
                "Treatment is observed with measured confounders and "
                "overlap support."
            ),
            required_data=["outcome", "treatment", "confounders"],
            assumptions=["unconfoundedness", "positivity", "SUTVA"],
            risks=["unmeasured confounding", "poor overlap"],
            next_step="Call sp.recommend(..., design='observational').",
        )
    if "causal discovery" in need:
        return IntakeResult(
            outcome="not_implemented",
            why=(
                "The request asks for causal discovery rather than an "
                "identified effect estimator."
            ),
            candidate_designs=["causal_discovery"],
            risks=["weak identification from observational structure alone"],
            next_step=(
                "Use StatsPAI causal-discovery tools only as exploratory "
                "evidence."
            ),
        )
    return IntakeResult(
        outcome="ambiguous",
        candidate_designs=["did", "rd", "iv", "observational", "synth"],
        why="The supplied facts do not uniquely determine one design.",
        deciding_question=(
            "What assignment mechanism makes the counterfactual credible?"
        ),
        next_step="Resolve assignment mechanism before fitting.",
    )
