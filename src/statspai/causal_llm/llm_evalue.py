"""
LLM-assisted unobserved-confounder enumeration for E-value
sensitivity analysis (arXiv 2603.14273, 2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


# Catalogue of common unobserved confounders by domain
_DOMAIN_CONFOUNDERS = {
    "health": [
        "patient adherence", "comorbidities", "lifestyle factors",
        "socioeconomic status", "access to care", "genetic predisposition",
    ],
    "education": [
        "parental involvement", "innate ability", "peer effects",
        "school resources", "neighborhood quality",
    ],
    "labor": [
        "ability", "motivation", "network effects", "family background",
        "job-search effort",
    ],
    "policy": [
        "concurrent reforms", "anticipation effects", "selection into program",
        "differential reporting",
    ],
    "marketing": [
        "selection bias", "seasonal trends", "competing campaigns",
        "platform algorithm changes",
    ],
}


@dataclass
class UnobservedConfounderProposal:
    """List of plausible unobserved confounders + suggested E-values."""
    candidates: List[str]
    suggested_evalue_thresholds: List[float]
    domain: str
    backend: str = "heuristic"

    def summary(self) -> str:
        rows = [
            "Unobserved Confounder Proposal",
            "=" * 42,
            f"  Domain  : {self.domain}",
            f"  Backend : {self.backend}",
            "  Candidates (with E-value threshold to nullify):",
        ]
        for c, e in zip(self.candidates, self.suggested_evalue_thresholds):
            rows.append(f"    - {c} (E-val ≈ {e:.2f})")
        return "\n".join(rows)


def llm_unobserved_confounders(
    treatment: str,
    outcome: str,
    domain: str = "health",
    client: Optional[Any] = None,
    point_estimate_rr: float = 1.5,
) -> UnobservedConfounderProposal:
    """
    Enumerate plausible unobserved confounders for a study.

    Parameters
    ----------
    treatment, outcome : str
        Free-text descriptions (used by LLM, ignored by heuristic).
    domain : {'health', 'education', 'labor', 'policy', 'marketing'}
    client : object, optional
        LLM client with ``.complete(prompt: str) -> str``.
    point_estimate_rr : float, default 1.5
        Observed risk ratio; suggested E-values are scaled relative
        to this so the user can read "to nullify a RR of X you'd need
        an unobserved RR of Y".

    Returns
    -------
    UnobservedConfounderProposal
    """
    if client is not None and hasattr(client, "complete"):
        try:
            prompt = (
                f"Treatment: {treatment}\nOutcome: {outcome}\n"
                f"Domain: {domain}\n"
                "List 5 plausible unobserved confounders for this "
                "exposure-outcome pair. Return as a JSON array of strings."
            )
            import json
            raw = client.complete(prompt)
            cands = json.loads(raw)
            evals = _suggested_evalues(point_estimate_rr, len(cands))
            return UnobservedConfounderProposal(
                candidates=cands,
                suggested_evalue_thresholds=evals,
                domain=domain,
                backend=type(client).__name__,
            )
        except Exception:
            pass

    # Heuristic backend
    candidates = _DOMAIN_CONFOUNDERS.get(domain.lower(), [
        "unmeasured baseline characteristics",
        "selection into observation",
        "time-varying confounders",
    ])
    evals = _suggested_evalues(point_estimate_rr, len(candidates))
    return UnobservedConfounderProposal(
        candidates=candidates,
        suggested_evalue_thresholds=evals,
        domain=domain,
        backend="heuristic",
    )


def _suggested_evalues(rr: float, k: int) -> List[float]:
    """E-value to nullify: E = RR + sqrt(RR*(RR-1)).

    Returns the canonical E-value, then scaled variants (×0.8, ×1.2, ...).
    """
    base = rr + (rr * max(rr - 1, 0)) ** 0.5
    factors = [1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3]
    return [base * f for f in factors[:k]]
