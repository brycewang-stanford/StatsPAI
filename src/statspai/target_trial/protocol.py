"""
Target Trial Protocol (Hernan & Robins 2016; JAMA 2022).

The 7 components of a target trial that must be specified *before* any
analysis:

1. Eligibility criteria
2. Treatment strategies
3. Assignment procedure
4. Follow-up period (time zero definition, end)
5. Outcome
6. Causal contrast of interest (ITT vs per-protocol)
7. Analysis plan

Writing a protocol explicitly is the single most effective tool for
preventing immortal time bias, prevalent-user bias, and selection bias
in observational studies.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Sequence


_VALID_CONTRASTS = ("ITT", "per-protocol", "as-treated", "observational-analogue")


@dataclass
class TargetTrialProtocol:
    """Formal 7-component target trial protocol.

    Parameters
    ----------
    eligibility : str | list[str] | Callable
        Entry criteria at time zero. May be a SQL-like string filter,
        a list of human-readable conditions, or a predicate that takes a
        DataFrame row and returns bool.
    treatment_strategies : list[str]
        Named arms being contrasted, e.g. ``["initiate statin at t0", "no statin"]``.
        At least two strategies required.
    assignment : str
        "randomization" (for RCT) or "observational emulation"
        (conditional exchangeability assumed given baseline covariates).
    time_zero : str
        Explicit rule defining time zero — usually the moment of
        eligibility + treatment assignment alignment. This is the
        single most important field for preventing immortal time bias.
    followup_end : str
        Administrative censoring rule (e.g. ``"min(event, loss, 2026-01-01)"``).
    outcome : str
        Primary outcome definition.
    causal_contrast : str
        One of ``"ITT"``, ``"per-protocol"``, ``"as-treated"``,
        ``"observational-analogue"``.
    analysis_plan : str
        Which estimator will recover the target estimand
        (e.g. ``"IPW + Cox"``, ``"g-formula + pooled logistic"``).
    baseline_covariates : list[str]
        Covariates measured at or before time zero that render
        treatment assignment conditionally exchangeable.
    time_varying_covariates : list[str], optional
        Post-baseline covariates affected by prior treatment; trigger
        g-methods (MSM / parametric g-formula / LTMLE).
    notes : str, optional
        Free-form notes (e.g. known sources of confounding).
    """

    eligibility: Any
    treatment_strategies: Sequence[str]
    assignment: str
    time_zero: str
    followup_end: str
    outcome: str
    causal_contrast: str
    analysis_plan: str
    baseline_covariates: list[str] = field(default_factory=list)
    time_varying_covariates: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if len(list(self.treatment_strategies)) < 2:
            raise ValueError(
                "treatment_strategies must contain at least 2 arms."
            )
        if self.causal_contrast not in _VALID_CONTRASTS:
            raise ValueError(
                f"causal_contrast must be one of {_VALID_CONTRASTS}, "
                f"got {self.causal_contrast!r}"
            )
        if self.assignment not in ("randomization", "observational emulation"):
            raise ValueError(
                "assignment must be 'randomization' or 'observational emulation'"
            )

    def summary(self) -> str:
        """Return a human-readable protocol summary."""
        lines = ["Target Trial Protocol", "=" * 40]
        d = asdict(self)
        pretty = {
            "eligibility": "1. Eligibility",
            "treatment_strategies": "2. Treatment strategies",
            "assignment": "3. Assignment",
            "time_zero": "4a. Time zero",
            "followup_end": "4b. Follow-up end",
            "outcome": "5. Outcome",
            "causal_contrast": "6. Causal contrast",
            "analysis_plan": "7. Analysis plan",
            "baseline_covariates": "Baseline covariates",
            "time_varying_covariates": "Time-varying covariates",
            "notes": "Notes",
        }
        for k, label in pretty.items():
            v = d[k]
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v) if v else "(none)"
            if v in ("", "(none)") and k in ("notes", "time_varying_covariates"):
                continue
            lines.append(f"{label}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


def protocol(
    eligibility: Any,
    treatment_strategies: Sequence[str],
    assignment: str,
    time_zero: str,
    followup_end: str,
    outcome: str,
    causal_contrast: str = "ITT",
    analysis_plan: str = "IPW + pooled logistic regression",
    baseline_covariates: list[str] | None = None,
    time_varying_covariates: list[str] | None = None,
    notes: str = "",
) -> TargetTrialProtocol:
    """Create a target trial protocol. See :class:`TargetTrialProtocol`."""
    return TargetTrialProtocol(
        eligibility=eligibility,
        treatment_strategies=list(treatment_strategies),
        assignment=assignment,
        time_zero=time_zero,
        followup_end=followup_end,
        outcome=outcome,
        causal_contrast=causal_contrast,
        analysis_plan=analysis_plan,
        baseline_covariates=list(baseline_covariates or []),
        time_varying_covariates=list(time_varying_covariates or []),
        notes=notes,
    )
