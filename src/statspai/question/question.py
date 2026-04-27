"""
Causal question DSL implementation.

See :mod:`statspai.question` for user-facing docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "CausalQuestion", "causal_question",
    "IdentificationPlan", "EstimationResult",
]


_VALID_ESTIMANDS = ("ATE", "ATT", "ATU", "LATE", "CATE", "ITT")
_VALID_TIME = ("cross_section", "panel", "repeated_cross_section",
               "longitudinal", "time_series", "pre_post")
_VALID_DESIGNS = (
    "auto", "rct", "selection_on_observables", "iv", "natural_experiment",
    "policy_shock", "regression_discontinuity", "synthetic_control",
    "did", "event_study", "longitudinal_observational",
)


# --------------------------------------------------------------------------- #
#  Identification plan
# --------------------------------------------------------------------------- #


@dataclass
class IdentificationPlan:
    """Output of :meth:`CausalQuestion.identify`.

    Describes which estimator is planned, why it is identifying, and
    which assumptions the user must defend.
    """

    estimator: str
    estimand: str
    identification_story: str
    assumptions: List[str]
    fallback_estimators: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Identification Plan",
            "=" * 58,
            f"  Estimand              : {self.estimand}",
            f"  Primary estimator     : sp.{self.estimator}",
            "  Identification story  :",
            f"    {self.identification_story}",
            "  Assumptions (must defend):",
        ]
        for a in self.assumptions:
            lines.append(f"    - {a}")
        if self.fallback_estimators:
            lines.append("  Fallbacks:")
            for f in self.fallback_estimators:
                lines.append(f"    - {f}")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ! {w}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Estimation result
# --------------------------------------------------------------------------- #


@dataclass
class EstimationResult:
    """Unified view of a causal-question estimate.

    Thin wrapper that preserves the underlying estimator's full result
    object while exposing a canonical ``estimate / se / ci`` interface.
    """

    estimand: str
    estimator: str
    estimate: float
    se: float
    ci: tuple[float, float]
    n: int
    underlying: Any
    plan: IdentificationPlan

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Causal Question Estimate ({self.estimand} via sp.{self.estimator})\n"
            f"  Estimate = {self.estimate:+.4f}   "
            f"SE = {self.se:.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   n = {self.n}"
        )


# --------------------------------------------------------------------------- #
#  CausalQuestion
# --------------------------------------------------------------------------- #


@dataclass
class CausalQuestion:
    """Pre-registered causal question declaration.

    Fields map directly onto the Target Trial Protocol (Hernán 2016)
    and the "PICOTS + identification" rubric the article describes.
    """

    treatment: str
    outcome: str
    data: Optional[pd.DataFrame] = None
    population: str = ""
    estimand: str = "ATE"
    design: str = "auto"
    time_structure: str = "cross_section"
    time: Optional[str] = None
    id: Optional[str] = None
    covariates: List[str] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)
    running_variable: Optional[str] = None
    cutoff: Optional[float] = None
    cohort: Optional[str] = None
    notes: str = ""

    # Filled by identify() / estimate()
    _plan: Optional[IdentificationPlan] = None
    _result: Optional[EstimationResult] = None

    # --- Serialization / reproducibility -------------------------------- #

    def save(self, filename, *, fmt: str = "auto", note: str = "") -> "Path":
        """Save the question to a pre-registration file.

        See :func:`statspai.question.preregister.preregister` for details.
        """
        from .preregister import preregister as _pre
        return _pre(self, filename, fmt=fmt, note=note)

    @classmethod
    def load(cls, filename) -> "CausalQuestion":
        """Load a CausalQuestion from a preregistration file."""
        from .preregister import load_preregister
        return load_preregister(filename)

    def to_yaml(self) -> str:
        """Render the question as a YAML string (no file I/O)."""
        from .preregister import _yaml_dumps
        return _yaml_dumps({"question": self.to_dict()})

    # --- Introspection --------------------------------------------------- #

    def to_dict(self) -> dict:
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "population": self.population,
            "estimand": self.estimand,
            "design": self.design,
            "time_structure": self.time_structure,
            "time": self.time,
            "id": self.id,
            "covariates": list(self.covariates),
            "instruments": list(self.instruments),
            "running_variable": self.running_variable,
            "cutoff": self.cutoff,
            "cohort": self.cohort,
            "notes": self.notes,
        }

    # --- Identification -------------------------------------------------- #

    def identify(self) -> IdentificationPlan:
        """Choose an estimator based on the declared design / estimand."""
        self._plan = _pick_plan(self)
        return self._plan

    # --- Estimation ------------------------------------------------------ #

    def estimate(self, **kwargs) -> EstimationResult:
        """Execute the identification plan against ``self.data``."""
        if self.data is None:
            raise ValueError("CausalQuestion.data must be set before estimate().")
        if self._plan is None:
            self.identify()
        result = _dispatch_estimator(self, self._plan, **kwargs)
        self._result = result
        return result

    # --- Report ---------------------------------------------------------- #

    def report(self, fmt: str = "markdown") -> str:
        """Render a publication-ready Methods + Results narrative."""
        if self._plan is None:
            self.identify()
        if self._result is None:
            raise ValueError("Run .estimate() before .report().")
        return _render_report(self, self._plan, self._result, fmt=fmt)

    # --- Paper builder --------------------------------------------------- #

    def paper(self, *, fmt: str = "markdown",
              output_path: Optional[str] = None,
              dag: Any = None,
              include_robustness: bool = True,
              cite: bool = True):
        """Build a full :class:`PaperDraft` from this declared question.

        Convenience wrapper around :func:`statspai.paper_from_question`.
        Calls ``identify()`` and ``estimate()`` on demand, then assembles
        a Question / Data / Identification / Estimator / Results /
        Robustness / References draft. Renders to markdown by default;
        pass ``fmt='qmd'`` for a Quarto document with statspai
        provenance and an auto-appended Reproducibility appendix.

        Examples
        --------
        >>> q = sp.causal_question("trained", "wage", data=df, design="did",
        ...                        time="year", id="worker_id")
        >>> draft = q.paper(fmt='qmd')
        >>> draft.write("paper.qmd")
        """
        from ..workflow.paper import paper_from_question
        return paper_from_question(
            self,
            fmt=fmt,
            output_path=output_path,
            include_robustness=include_robustness,
            cite=cite,
            dag=dag,
        )


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #


def causal_question(
    treatment: str,
    outcome: str,
    *,
    data: Optional[pd.DataFrame] = None,
    population: str = "",
    estimand: str = "ATE",
    design: str = "auto",
    time_structure: str = "cross_section",
    time: Optional[str] = None,
    id: Optional[str] = None,
    covariates: Optional[Sequence[str]] = None,
    instruments: Optional[Sequence[str]] = None,
    running_variable: Optional[str] = None,
    cutoff: Optional[float] = None,
    cohort: Optional[str] = None,
    notes: str = "",
) -> CausalQuestion:
    """Declare a causal question (see :class:`CausalQuestion`)."""
    if estimand not in _VALID_ESTIMANDS:
        raise ValueError(
            f"estimand must be one of {_VALID_ESTIMANDS}; got {estimand!r}"
        )
    if design not in _VALID_DESIGNS:
        raise ValueError(
            f"design must be one of {_VALID_DESIGNS}; got {design!r}"
        )
    if time_structure not in _VALID_TIME:
        raise ValueError(
            f"time_structure must be one of {_VALID_TIME}; got {time_structure!r}"
        )
    return CausalQuestion(
        treatment=treatment,
        outcome=outcome,
        data=data,
        population=population,
        estimand=estimand,
        design=design,
        time_structure=time_structure,
        time=time,
        id=id,
        covariates=list(covariates or []),
        instruments=list(instruments or []),
        running_variable=running_variable,
        cutoff=cutoff,
        cohort=cohort,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
#  Internal: identification logic
# --------------------------------------------------------------------------- #


def _pick_plan(q: CausalQuestion) -> IdentificationPlan:
    design = q.design

    # Explicit design always wins
    if design == "auto":
        design = _auto_design(q)

    if design in ("rct",):
        return IdentificationPlan(
            estimator="regress",
            estimand=q.estimand,
            identification_story=(
                "Randomization guarantees mean independence between "
                "treatment and potential outcomes; OLS on treatment is "
                "unbiased for the ATE."
            ),
            assumptions=["random assignment", "SUTVA / no interference"],
            fallback_estimators=["aipw", "ipw"],
        )
    if design in ("iv", "natural_experiment"):
        if not q.instruments:
            return IdentificationPlan(
                estimator="iv",
                estimand="LATE",
                identification_story="IV / natural experiment.",
                assumptions=[],
                warnings=[
                    "design='iv' but no `instruments=[...]` were supplied. "
                    "Set instruments before estimate().",
                ],
            )
        return IdentificationPlan(
            estimator="iv",
            estimand="LATE",
            identification_story=(
                "Instrumental variables "
                f"{q.instruments} are relevant (first stage) and satisfy "
                "the exclusion restriction; 2SLS recovers the LATE."
            ),
            assumptions=[
                "relevance (cov(Z, D) != 0; first-stage F >> 10)",
                "exclusion restriction (Z affects Y only through D)",
                "monotonicity (no defiers)",
            ],
            fallback_estimators=["liml", "anderson_rubin_ci"],
        )
    if design == "regression_discontinuity":
        return IdentificationPlan(
            estimator="rdrobust",
            estimand=q.estimand if q.estimand in ("ATT", "LATE") else "LATE",
            identification_story=(
                f"Sharp / fuzzy RD at cutoff={q.cutoff} in running "
                f"variable '{q.running_variable}'; continuity of potential "
                "outcomes at the cutoff identifies the local ATE."
            ),
            assumptions=[
                "continuity of E[Y(d) | X=x] at the cutoff",
                "no precise manipulation of the running variable",
            ],
            fallback_estimators=["rd_honest", "rddensity"],
        )
    if design == "did":
        return IdentificationPlan(
            estimator="did",
            estimand="ATT",
            identification_story=(
                "DiD — parallel trends in untreated potential outcomes "
                "plus no anticipation identifies the ATT."
            ),
            assumptions=[
                "parallel trends",
                "no anticipation",
                "stable unit composition",
            ],
            fallback_estimators=["callaway_santanna", "sun_abraham",
                                 "did_imputation", "honest_did"],
        )
    if design == "event_study":
        return IdentificationPlan(
            estimator="event_study",
            estimand="ATT",
            identification_story=(
                "Event study — dynamic ATT relative to treatment onset "
                "under parallel trends."
            ),
            assumptions=["parallel trends", "no anticipation"],
            fallback_estimators=["callaway_santanna", "sun_abraham"],
        )
    if design == "synthetic_control":
        return IdentificationPlan(
            estimator="synth",
            estimand="ATT",
            identification_story=(
                "Synthetic control — construct a weighted combination of "
                "untreated units that matches pre-treatment outcomes; "
                "post-treatment divergence identifies the ATT."
            ),
            assumptions=[
                "convex hull: treated unit's pre-trend lies inside donors'",
                "no interference from donor pool",
            ],
            fallback_estimators=["sdid", "augsynth", "scpi"],
        )
    if design == "longitudinal_observational":
        return IdentificationPlan(
            estimator="longitudinal.analyze",
            estimand=q.estimand if q.estimand != "LATE" else "ATE",
            identification_story=(
                "Sequential exchangeability + positivity given time-"
                "varying covariates; estimated via MSM / g-formula."
            ),
            assumptions=[
                "sequential exchangeability",
                "positivity at every time point",
                "consistency",
            ],
            fallback_estimators=["msm", "ltmle", "gformula"],
        )
    # selection_on_observables
    return IdentificationPlan(
        estimator="aipw",
        estimand=q.estimand,
        identification_story=(
            f"Conditional ignorability given {q.covariates}; AIPW "
            "(doubly robust) is consistent if either the propensity score "
            "or outcome model is correctly specified."
        ),
        assumptions=[
            "conditional exchangeability given covariates",
            "positivity (overlap in propensity score)",
            "no interference",
        ],
        fallback_estimators=["ipw", "regress", "match", "ebalance",
                             "causal_forest"],
    )


def _auto_design(q: CausalQuestion) -> str:
    """Pick a design from the declared fields."""
    if q.instruments:
        return "iv"
    if q.running_variable is not None and q.cutoff is not None:
        return "regression_discontinuity"
    if q.time_structure in ("panel", "pre_post") and q.time is not None:
        return "did"
    if q.time_structure == "longitudinal":
        return "longitudinal_observational"
    return "selection_on_observables"


# --------------------------------------------------------------------------- #
#  Dispatcher: plan -> estimate
# --------------------------------------------------------------------------- #


def _dispatch_estimator(q: CausalQuestion,
                        plan: IdentificationPlan,
                        **kwargs) -> EstimationResult:
    import statspai as sp

    est_name = plan.estimator
    data = q.data
    n = int(len(data))

    if est_name == "regress":
        formula_parts = [q.outcome, "~", q.treatment]
        if q.covariates:
            formula_parts += ["+"] + [" + ".join(q.covariates)]
        formula = " ".join(formula_parts)
        res = sp.regress(formula, data=data, **kwargs)
        est = float(res.params.get(q.treatment, float("nan")))
        se = float(res.std_errors.get(q.treatment, float("nan")))
        ci_lo = float(res.conf_int_lower.get(q.treatment, float("nan")))
        ci_hi = float(res.conf_int_upper.get(q.treatment, float("nan")))
        return EstimationResult(
            estimand=plan.estimand, estimator=est_name,
            estimate=est, se=se, ci=(ci_lo, ci_hi),
            n=n, underlying=res, plan=plan,
        )

    if est_name == "aipw":
        res = sp.aipw(
            data=data,
            y=q.outcome,
            treat=q.treatment,
            covariates=list(q.covariates),
            **kwargs,
        )
        est, se, ci = _extract_generic(res)
        return EstimationResult(
            estimand=plan.estimand, estimator=est_name,
            estimate=est, se=se, ci=ci, n=n,
            underlying=res, plan=plan,
        )

    if est_name == "iv":
        # 2SLS via formula interface
        instrs = " + ".join(q.instruments)
        covs = " + " + " + ".join(q.covariates) if q.covariates else ""
        formula = f"{q.outcome} ~ [{q.treatment} ~ {instrs}]{covs}"
        res = sp.iv(formula, data=data, **kwargs)
        est, se, ci = _extract_generic(res)
        return EstimationResult(
            estimand=plan.estimand, estimator=est_name,
            estimate=est, se=se, ci=ci, n=n,
            underlying=res, plan=plan,
        )

    if est_name == "did":
        res = sp.did(
            data=data,
            y=q.outcome,
            treat=q.treatment,
            time=q.time,
            id=q.id,
            **kwargs,
        )
        est, se, ci = _extract_generic(res)
        return EstimationResult(
            estimand=plan.estimand, estimator=est_name,
            estimate=est, se=se, ci=ci, n=n,
            underlying=res, plan=plan,
        )

    if est_name == "rdrobust":
        res = sp.rdrobust(
            y=data[q.outcome],
            x=data[q.running_variable],
            c=q.cutoff,
            **kwargs,
        )
        est, se, ci = _extract_generic(res)
        return EstimationResult(
            estimand=plan.estimand, estimator=est_name,
            estimate=est, se=se, ci=ci, n=n,
            underlying=res, plan=plan,
        )

    if est_name == "synth":
        res = sp.synth(
            data=data,
            outcome=q.outcome,
            treat=q.treatment,
            time=q.time,
            id=q.id,
            **kwargs,
        )
        est, se, ci = _extract_generic(res)
        return EstimationResult(
            estimand=plan.estimand, estimator=est_name,
            estimate=est, se=se, ci=ci, n=n,
            underlying=res, plan=plan,
        )

    if est_name == "longitudinal.analyze":
        from ..longitudinal import analyze as _long_analyze
        res = _long_analyze(
            data=data,
            id=q.id, time=q.time,
            treatment=q.treatment, outcome=q.outcome,
            time_varying=q.covariates or [],
            **kwargs,
        )
        return EstimationResult(
            estimand=plan.estimand, estimator=est_name,
            estimate=float(res.estimate),
            se=float(res.se),
            ci=(float(res.ci[0]), float(res.ci[1])),
            n=n, underlying=res, plan=plan,
        )

    if est_name == "event_study":
        res = sp.event_study(
            data=data, y=q.outcome, treat=q.treatment,
            time=q.time, id=q.id, **kwargs,
        )
        est, se, ci = _extract_generic(res)
        return EstimationResult(
            estimand=plan.estimand, estimator=est_name,
            estimate=est, se=se, ci=ci, n=n,
            underlying=res, plan=plan,
        )

    raise NotImplementedError(
        f"Dispatch for estimator {est_name!r} not implemented."
    )


def _extract_generic(res) -> tuple[float, float, tuple[float, float]]:
    """Pull (estimate, se, ci) from a heterogeneous result."""
    est = float(getattr(res, "estimate", float("nan")))
    se = float(getattr(res, "se", float("nan")))
    ci = getattr(res, "ci", None)
    if ci is None or (isinstance(ci, tuple) and any(c is None for c in ci)):
        ci = (est - 1.96 * se, est + 1.96 * se)
    return est, se, (float(ci[0]), float(ci[1]))


# --------------------------------------------------------------------------- #
#  Report rendering
# --------------------------------------------------------------------------- #


def _render_report(
    q: CausalQuestion,
    plan: IdentificationPlan,
    result: EstimationResult,
    *,
    fmt: str = "markdown",
) -> str:
    lo, hi = result.ci
    if fmt == "markdown":
        return (
            "## Causal Question\n\n"
            f"**Treatment:** {q.treatment}  \n"
            f"**Outcome:** {q.outcome}  \n"
            f"**Population:** {q.population or 'not specified'}  \n"
            f"**Estimand:** {plan.estimand}  \n"
            f"**Design:** {q.design}  \n"
            f"**Time structure:** {q.time_structure}\n\n"
            "## Identification\n\n"
            f"{plan.identification_story}\n\n"
            "Required assumptions:\n"
            + "\n".join(f"- {a}" for a in plan.assumptions)
            + "\n\n"
            "## Estimation\n\n"
            f"Estimator: `sp.{plan.estimator}`  \n"
            f"Estimate = **{result.estimate:+.4f}**, "
            f"95% CI [{lo:+.4f}, {hi:+.4f}], "
            f"SE = {result.se:.4f}, n = {result.n}.\n"
        )
    if fmt == "text":
        return (
            f"Causal Question: {q.treatment} -> {q.outcome}\n"
            f"Estimand: {plan.estimand} via sp.{plan.estimator}\n"
            f"Estimate = {result.estimate:+.4f} "
            f"(95% CI [{lo:+.4f}, {hi:+.4f}], SE = {result.se:.4f})\n"
            f"Identification: {plan.identification_story}\n"
            "Assumptions: " + "; ".join(plan.assumptions)
        )
    raise ValueError("fmt must be 'markdown' or 'text'")
