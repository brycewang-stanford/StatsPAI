"""
Bradford-Hill causal inference viewpoints.

Bradford-Hill (1965) proposed 9 "viewpoints" to weigh when moving from
a statistical association to a causal claim.  This module provides a
structured scoring rubric and a narrative report.

These are not a statistical test — they are a structured checklist
meant to discipline the leap from association to causation, and to
surface exactly which assumptions the investigator is willing to
defend.

Reference
---------
Hill, A.B. (1965). "The environment and disease: association or
causation?" *Proceedings of the Royal Society of Medicine*, 58(5), 295-300.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


__all__ = [
    "BradfordHillResult",
    "bradford_hill",
]


VIEWPOINTS = (
    "strength",
    "consistency",
    "specificity",
    "temporality",
    "biological_gradient",
    "plausibility",
    "coherence",
    "experiment",
    "analogy",
)

# Temporality is effectively a prerequisite, not a continuous score.
_PREREQUISITES = ("temporality",)


@dataclass
class BradfordHillResult:
    scores: Dict[str, float]
    evidence: Dict[str, str]
    total: float
    max_total: float
    verdict: str
    missing_prerequisites: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bar = "=" * 60
        lines = ["Bradford-Hill Causal Assessment", bar]
        for vp in VIEWPOINTS:
            score = self.scores.get(vp, float("nan"))
            ev = self.evidence.get(vp, "")
            score_disp = "N/A" if np.isnan(score) else f"{score:.1f}/1.0"
            lines.append(f"  {vp:<22s} {score_disp}")
            if ev:
                lines.append(f"      > {ev}")
        lines.append("-" * 60)
        lines.append(f"  Total: {self.total:.2f} / {self.max_total:.2f}")
        if self.missing_prerequisites:
            lines.append(
                "  Missing prerequisites: "
                + ", ".join(self.missing_prerequisites)
            )
        lines.append(f"  Verdict: {self.verdict}")
        lines.append(bar)
        return "\n".join(lines)


def bradford_hill(
    evidence: Optional[Dict[str, float]] = None,
    notes: Optional[Dict[str, str]] = None,
    *,
    strength: Optional[float] = None,
    consistency: Optional[float] = None,
    specificity: Optional[float] = None,
    temporality: Optional[float] = None,
    biological_gradient: Optional[float] = None,
    plausibility: Optional[float] = None,
    coherence: Optional[float] = None,
    experiment: Optional[float] = None,
    analogy: Optional[float] = None,
) -> BradfordHillResult:
    """Score an association against the Bradford-Hill viewpoints.

    Each viewpoint takes a score in [0, 1]:
      - 0: no evidence / counter-evidence
      - 0.5: partial / mixed evidence
      - 1: strong evidence

    ``None`` marks a viewpoint as "not assessed" (counts toward neither
    numerator nor denominator).

    Parameters
    ----------
    evidence : dict, optional
        Mapping viewpoint name -> score.  Alternative to keyword args.
    notes : dict, optional
        Mapping viewpoint name -> narrative justification.

    Returns
    -------
    BradfordHillResult

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.epi.bradford_hill(
    ...     strength=1.0,
    ...     consistency=1.0,
    ...     temporality=1.0,
    ...     biological_gradient=0.5,
    ...     plausibility=1.0,
    ...     specificity=0.0,
    ...     coherence=1.0,
    ...     experiment=0.5,
    ...     analogy=0.5,
    ...     notes={"strength": "HR > 5 in meta-analysis"},
    ... )
    >>> print(res.summary())
    """
    kw_scores = dict(
        strength=strength,
        consistency=consistency,
        specificity=specificity,
        temporality=temporality,
        biological_gradient=biological_gradient,
        plausibility=plausibility,
        coherence=coherence,
        experiment=experiment,
        analogy=analogy,
    )
    # Merge evidence dict (if given) with explicit kwargs; kwargs win.
    scores: Dict[str, float] = {}
    if evidence:
        for k, v in evidence.items():
            if k not in VIEWPOINTS:
                raise ValueError(
                    f"Unknown viewpoint {k!r}; valid names: {VIEWPOINTS}"
                )
            scores[k] = float(v)
    for k, v in kw_scores.items():
        if v is not None:
            scores[k] = float(v)

    for k, v in scores.items():
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"Viewpoint {k!r} score must be in [0, 1]; got {v}"
            )

    assessed = {k: v for k, v in scores.items() if not (isinstance(v, float) and np.isnan(v))}
    total = float(sum(assessed.values()))
    max_total = float(len(assessed))

    # Full score map (NaN for unassessed)
    full_scores = {vp: float(assessed.get(vp, float("nan"))) for vp in VIEWPOINTS}

    # Prerequisite check
    missing_prereqs = []
    for p in _PREREQUISITES:
        if p not in assessed or assessed[p] <= 0.0:
            missing_prereqs.append(p)

    # Verdict heuristic
    if missing_prereqs:
        verdict = "INSUFFICIENT — temporality not established."
    elif max_total == 0:
        verdict = "No viewpoints assessed."
    else:
        frac = total / max_total
        if frac >= 0.75:
            verdict = "STRONG support for causality."
        elif frac >= 0.5:
            verdict = "MODERATE support; several viewpoints weak."
        elif frac >= 0.25:
            verdict = "WEAK support; treat association as tentative."
        else:
            verdict = "LITTLE support; association likely non-causal or confounded."

    ev_notes = dict(notes or {})
    for k in list(ev_notes):
        if k not in VIEWPOINTS:
            raise ValueError(
                f"Unknown viewpoint in notes: {k!r}"
            )

    _result = BradfordHillResult(
        scores=full_scores,
        evidence={vp: ev_notes.get(vp, "") for vp in VIEWPOINTS},
        total=total,
        max_total=max_total,
        verdict=verdict,
        missing_prerequisites=missing_prereqs,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.epi.bradford_hill",
            params={"viewpoints_with_evidence": list(full_scores.keys())},
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
