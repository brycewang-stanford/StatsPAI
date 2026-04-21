"""
LLM-assisted sensitivity-parameter prior elicitation.

For Cinelli-Hazlett (2020) sensitivity analysis you need to choose
``rho_max`` (max correlation between unobserved confounder and
treatment) and ``r2`` (max R² of the unobserved confounder with
outcome). This module proposes domain-aware default values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


_DOMAIN_PRIORS = {
    "health": {"rho_max": 0.3, "r2": 0.04, "comment":
               "Health behaviours and unmeasured biomarkers typically "
               "leave residual confounding around 30%."},
    "education": {"rho_max": 0.4, "r2": 0.06, "comment":
                  "Innate ability/peer effects often correlate strongly "
                  "with both treatment and outcome."},
    "labor": {"rho_max": 0.35, "r2": 0.05, "comment":
              "Ability and motivation are classic labour-econ confounders."},
    "policy": {"rho_max": 0.25, "r2": 0.03, "comment":
               "Concurrent reforms and selection are smaller in well-"
               "designed policy evaluations."},
    "marketing": {"rho_max": 0.5, "r2": 0.10, "comment":
                  "Selection bias and platform algorithms can introduce "
                  "large residual confounding."},
}


@dataclass
class SensitivityPriorProposal:
    """Suggested sensitivity parameter priors for sensemakr-style analysis."""
    rho_max: float
    r2: float
    rationale: str
    domain: str
    backend: str = "heuristic"

    def summary(self) -> str:
        return (
            "Sensitivity Prior Proposal\n"
            "=" * 42 + "\n"
            f"  Domain   : {self.domain}\n"
            f"  Backend  : {self.backend}\n"
            f"  rho_max  : {self.rho_max:.3f}\n"
            f"  R² (Y|U) : {self.r2:.3f}\n"
            f"  Reason   : {self.rationale}\n"
        )


def llm_sensitivity_priors(
    treatment: str,
    outcome: str,
    domain: str = "health",
    client: Optional[Any] = None,
) -> SensitivityPriorProposal:
    """
    Propose sensitivity-analysis priors for the substantive setting.

    Parameters
    ----------
    treatment, outcome : str
    domain : {'health', 'education', 'labor', 'policy', 'marketing'}
    client : object, optional
        LLM client with ``.complete(prompt: str) -> str``.

    Returns
    -------
    SensitivityPriorProposal
    """
    if client is not None and hasattr(client, "complete"):
        try:
            prompt = (
                f"Treatment: {treatment}\nOutcome: {outcome}\nDomain: {domain}\n"
                "Suggest reasonable values for rho_max (max correlation "
                "of unobserved confounder with treatment) and R² (max "
                "outcome variance explained by confounder) for a "
                "Cinelli-Hazlett sensitivity analysis. Return JSON: "
                "{\"rho_max\": float, \"r2\": float, \"rationale\": str}."
            )
            import json
            raw = client.complete(prompt)
            obj = json.loads(raw)
            return SensitivityPriorProposal(
                rho_max=float(obj['rho_max']),
                r2=float(obj['r2']),
                rationale=obj.get('rationale', 'LLM proposed'),
                domain=domain,
                backend=type(client).__name__,
            )
        except Exception:
            pass

    prior = _DOMAIN_PRIORS.get(
        domain.lower(),
        {"rho_max": 0.3, "r2": 0.05,
         "comment": "Generic default for unspecified domain."}
    )
    return SensitivityPriorProposal(
        rho_max=prior['rho_max'],
        r2=prior['r2'],
        rationale=prior['comment'],
        domain=domain,
        backend="heuristic",
    )
