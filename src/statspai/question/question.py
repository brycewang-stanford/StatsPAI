"""
Causal question DSL implementation.

See :mod:`statspai.question` for user-facing docstring.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np
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
    # ML-based selection-on-observables (v1.13+):
    "dml", "tmle", "metalearner", "causal_forest",
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
    """Declare a causal question (see :class:`CausalQuestion`).

    Supported ``design`` values
    ---------------------------
    Classical / quasi-experimental:
      - ``'rct'`` — randomised assignment; OLS ATE.
      - ``'iv'`` / ``'natural_experiment'`` — 2SLS / LATE.
      - ``'regression_discontinuity'`` — local polynomial RD.
      - ``'did'`` / ``'event_study'`` — difference-in-differences.
      - ``'synthetic_control'`` — convex-hull weighting.
      - ``'longitudinal_observational'`` — MSM / g-formula.
      - ``'selection_on_observables'`` — AIPW (default).

    ML-based selection-on-observables (v1.13+):
      - ``'dml'`` — Double/debiased ML for ATE / LATE
        [chernozhukov2018double].
      - ``'tmle'`` — Targeted Maximum Likelihood with Super Learner
        [vanderlaan2006targeted].
      - ``'metalearner'`` — S/T/X/R/DR-Learner for tau(x);
        population ATE summary via AIPW influence function
        [kunzel2019metalearners; nie2021quasi].
      - ``'causal_forest'`` — honest random forest for tau(x)
        [athey2019generalized; wager2018estimation]; population
        ATE inference uses cross-fit AIPW
        [vanderlaan2003unified; chernozhukov2018double].

    All bib keys above resolve in ``paper.bib``.
    """
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
        long_warns = []
        if q.estimand == "CATE":
            long_warns.append(
                "Longitudinal MSM / g-formula returns a population-level "
                "marginal effect; declared estimand 'CATE' was coerced to "
                "'ATE'. For heterogeneous longitudinal effects, use "
                "design='metalearner' on a single time slice."
            )
        elif q.estimand == "LATE":
            long_warns.append(
                "Longitudinal design declared but estimand='LATE'; the "
                "MSM / g-formula path identifies ATE, not LATE."
            )
        return IdentificationPlan(
            estimator="longitudinal.analyze",
            estimand=q.estimand if q.estimand in ("ATE", "ATT") else "ATE",
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
            warnings=long_warns,
        )
    if design == "dml":
        # IV-flavoured DML when instruments supplied; otherwise PLR/IRM.
        if q.instruments:
            iv_warns = []
            if q.estimand not in ("LATE", "ATE"):
                iv_warns.append(
                    f"DML+IV identifies the LATE; declared estimand "
                    f"{q.estimand!r} was coerced to 'LATE'."
                )
            iv_flavour = (
                "Double/debiased ML for IV (PLIV / IIVM): orthogonalises "
                "the outcome and treatment with respect to high-dimensional "
                "controls before applying the IV moment; recovers LATE "
                "without requiring linearity in X."
            )
            iv_plan_warns = list(iv_warns)
            if not q.covariates:
                iv_plan_warns.append(
                    "design='dml' requires covariates=[...] (DML "
                    "orthogonalises Y and D against X). Set covariates "
                    "before estimate()."
                )
            return IdentificationPlan(
                estimator="dml",
                estimand="LATE",
                identification_story=iv_flavour,
                assumptions=[
                    "conditional exchangeability of Z given X",
                    "first-stage relevance after controlling for X",
                    "exclusion restriction",
                    "monotonicity (no defiers)",
                    "Donsker / cross-fitting regularity for nuisance learners",
                ],
                fallback_estimators=["iv", "liml", "anderson_rubin_ci"],
                warnings=iv_plan_warns,
            )
        # Non-IV DML: scalar estimate is ATE (PLR/IRM both report ATE).
        coerce_warns = []
        if q.estimand not in ("ATE", "ATT"):
            coerce_warns.append(
                f"DML returns a scalar ATE (PLR/IRM); declared estimand "
                f"{q.estimand!r} was coerced to 'ATE'. For heterogeneous "
                "effects, use design='metalearner' or design='causal_forest'."
            )
        if not q.covariates:
            coerce_warns.append(
                "design='dml' requires covariates=[...] (DML orthogonalises "
                "Y and D against X). Set covariates before estimate()."
            )
        return IdentificationPlan(
            estimator="dml",
            estimand="ATE",
            identification_story=(
                "Double/debiased ML (Chernozhukov et al. 2018): "
                "Neyman-orthogonal moment with cross-fitted ML nuisance "
                "estimators. Recovers ATE under conditional ignorability "
                "without requiring a correctly specified parametric "
                "outcome or propensity model."
            ),
            assumptions=[
                "conditional exchangeability given covariates",
                "positivity (overlap)",
                "cross-fitted nuisance learners converge at o_p(n^{-1/4})",
                "no interference",
            ],
            fallback_estimators=["aipw", "tmle", "metalearner"],
            warnings=coerce_warns,
        )
    if design == "tmle":
        tmle_warns = []
        if q.estimand not in ("ATE", "ATT"):
            tmle_warns.append(
                f"TMLE supports ATE / ATT only; declared estimand "
                f"{q.estimand!r} was coerced to 'ATE'."
            )
        if not q.covariates:
            tmle_warns.append(
                "design='tmle' requires covariates=[...] for the outcome "
                "and propensity nuisance models."
            )
        return IdentificationPlan(
            estimator="tmle",
            estimand=q.estimand if q.estimand in ("ATE", "ATT") else "ATE",
            identification_story=(
                "Targeted Maximum Likelihood (van der Laan & Rubin 2006) "
                "with Super Learner nuisance: doubly robust + "
                "semiparametrically efficient under conditional "
                "ignorability; the targeting step solves the efficient "
                "influence-function score equation exactly."
            ),
            assumptions=[
                "conditional exchangeability given covariates",
                "positivity (propensity bounded away from 0/1)",
                "no interference",
                "Super Learner library rich enough for Q and g",
            ],
            fallback_estimators=["aipw", "dml"],
            warnings=tmle_warns,
        )
    if design == "metalearner":
        # Method targets CATE = tau(x); the SCALAR returned by
        # sp.metalearner is the population ATE (mean of estimated
        # CATEs via the AIPW influence function, v1.11.4+). The plan's
        # estimand reflects what `EstimationResult.estimate` represents
        # — i.e. ATE — to keep result labels honest. Per-unit CATEs
        # are accessible via `result.underlying.model_info['cate']`.
        meta_warns = []
        if not q.covariates:
            meta_warns.append(
                "design='metalearner' requires covariates=[...] — these "
                "are the effect modifiers used to estimate tau(x)."
            )
        return IdentificationPlan(
            estimator="metalearner",
            estimand="ATE",
            identification_story=(
                "Meta-learner for heterogeneous treatment effects "
                "(Künzel et al. 2019, Nie & Wager 2021): estimates "
                "tau(x) = E[Y(1) - Y(0) | X=x] via S/T/X/R/DR-Learner "
                "stages. The reported scalar is the population ATE "
                "(mean over units of estimated tau(x)) with the AIPW "
                "(doubly robust) influence-function SE; per-unit CATEs "
                "are available via result.underlying.model_info['cate']."
            ),
            assumptions=[
                "conditional exchangeability given X",
                "positivity / overlap",
                "no interference",
                "ML nuisance learners converge fast enough for the chosen "
                "learner family (R/DR require o_p(n^{-1/4}))",
            ],
            fallback_estimators=["causal_forest", "dml", "aipw", "tmle"],
            warnings=meta_warns,
        )
    if design == "causal_forest":
        # Two-layer estimation: the honest causal forest provides
        # heterogeneous tau(x) (per-unit CATEs accessible via
        # cf.effect(X)); the population-ATE point and SE come from the
        # cross-fit AIPW influence function — semiparametrically
        # efficient and B-independent. Same pattern as
        # grf::average_treatment_effect in R. Binary treatment only;
        # for continuous use design='dml'.
        cf_warns = []
        if not q.covariates:
            cf_warns.append(
                "design='causal_forest' requires covariates=[...] as "
                "effect modifiers."
            )
        return IdentificationPlan(
            estimator="causal_forest",
            estimand="ATE",
            identification_story=(
                "Generalised Random Forest / Causal Forest "
                "(Athey, Tibshirani & Wager 2019): honest random-forest "
                "estimator of tau(x). Per-unit CATEs and pointwise GRF "
                "intervals via result.underlying.effect(X) / "
                ".effect_interval(X). The reported scalar ATE point "
                "and SE come from the cross-fit AIPW influence "
                "function (van der Laan & Robins 2003; Chernozhukov et "
                "al. 2018) — doubly robust, semiparametrically "
                "efficient, B-independent."
            ),
            assumptions=[
                "conditional exchangeability given X",
                "positivity / overlap",
                "no interference",
                "honest splitting (separate sub-samples for partition vs leaf estimates)",
                "binary treatment (for AIPW-IF inference)",
            ],
            fallback_estimators=["metalearner", "dml", "aipw"],
            warnings=cf_warns,
        )
    # selection_on_observables
    # When the user asks for heterogeneous effects (CATE) AND has
    # covariates declared, promote metalearner; otherwise AIPW remains
    # the doubly robust default. The CATE→metalearner promotion gates
    # on covariates so identify() and estimate() agree (Bug 1).
    if q.estimand == "CATE" and q.covariates:
        return IdentificationPlan(
            estimator="metalearner",
            estimand="ATE",
            identification_story=(
                "Conditional ignorability given covariates; meta-learner "
                "(DR-Learner default) targets tau(x) under doubly robust "
                "moments. The reported scalar is the population ATE "
                "(mean of estimated CATEs) with AIPW influence-function "
                "SE; per-unit CATEs are in "
                "result.underlying.model_info['cate']."
            ),
            assumptions=[
                "conditional exchangeability given covariates",
                "positivity (overlap in propensity score)",
                "no interference",
            ],
            fallback_estimators=["causal_forest", "dml", "aipw", "tmle"],
        )
    if q.estimand == "CATE" and not q.covariates:
        # Promised CATE but no effect modifiers — AIPW is the right
        # primary, but flag that CATE was dropped.
        return IdentificationPlan(
            estimator="aipw",
            estimand="ATE",
            identification_story=(
                f"Conditional ignorability given {q.covariates}; AIPW "
                "(doubly robust). Declared estimand 'CATE' requires "
                "covariates as effect modifiers; with none supplied the "
                "scalar ATE is the only quantity identified."
            ),
            assumptions=[
                "conditional exchangeability given covariates",
                "positivity (overlap in propensity score)",
                "no interference",
            ],
            fallback_estimators=["dml", "tmle", "ipw", "regress", "match",
                                 "ebalance", "causal_forest"],
            warnings=[
                "estimand='CATE' was requested but no covariates were "
                "declared as effect modifiers — falling back to ATE via "
                "AIPW. Pass covariates=[...] to enable a CATE estimator."
            ],
        )
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
        fallback_estimators=["dml", "tmle", "ipw", "regress", "match",
                             "ebalance", "causal_forest"],
    )


def _auto_design(q: CausalQuestion) -> str:
    """Pick a design from the declared fields.

    Order of precedence:
      1. Explicit instruments → IV.
      2. Running variable + cutoff → RD.
      3. Panel / pre-post + time variable → DiD.
      4. Longitudinal → longitudinal_observational.
      5. Estimand=CATE with covariates → metalearner (heterogeneous effects).
      6. Otherwise → selection_on_observables (AIPW).
    """
    if q.instruments:
        return "iv"
    if q.running_variable is not None and q.cutoff is not None:
        return "regression_discontinuity"
    if q.time_structure in ("panel", "pre_post") and q.time is not None:
        return "did"
    if q.time_structure == "longitudinal":
        return "longitudinal_observational"
    if q.estimand == "CATE" and q.covariates:
        return "metalearner"
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

    if est_name == "dml":
        if not q.covariates:
            raise ValueError(
                "design='dml' requires covariates=[...] (DML orthogonalises "
                "Y and D against X; an empty X is not a DML problem)."
            )
        _reject_reserved_kwargs(
            kwargs, "dml",
            reserved=("data", "y", "treat", "covariates"),
        )
        # Pick PLR/IRM/PLIV/IIVM from declared fields. Caller can still
        # override via kwargs['model']; if so, we honour their choice and
        # only forward `instrument=` when the chosen model supports it.
        if "model" in kwargs:
            model = kwargs.pop("model")
        elif q.instruments:
            # IIVM requires BOTH a single binary instrument AND a
            # binary treatment (LATE for binary D, binary Z). When
            # either is non-binary, fall through to PLIV. Float-typed
            # binary columns are handled via dropna().nunique().
            if len(q.instruments) == 1:
                z_col = q.instruments[0]
                if z_col not in data.columns:
                    raise KeyError(
                        f"instrument {z_col!r} not found in data columns"
                    )
                z_binary = data[z_col].dropna().nunique() == 2
                d_binary = data[q.treatment].dropna().nunique() == 2
                model = "iivm" if (z_binary and d_binary) else "pliv"
            else:
                model = "pliv"
        else:
            d = data[q.treatment].dropna()
            model = "irm" if d.nunique() == 2 else "plr"
        is_iv_model = model in ("pliv", "iivm")
        # IV-flavoured models REQUIRE an instrument; non-IV models must
        # NOT receive one (sp.dml will reject the kwarg). This is the
        # collision Bug 2/6 — gate strictly.
        if is_iv_model and not q.instruments:
            raise ValueError(
                f"DML model={model!r} requires instruments=[...] on the "
                "CausalQuestion; none were declared."
            )
        # sp.dml's PLIV / IIVM expect a single scalar instrument. Reject
        # multi-instrument up-front with a pointer to the scalar
        # projection helper, instead of letting sp.dml's error surface
        # deep inside the dispatch.
        if is_iv_model and q.instruments and len(q.instruments) > 1:
            raise ValueError(
                f"DML model={model!r} accepts a single scalar instrument; "
                f"got {len(q.instruments)}: {list(q.instruments)!r}. For "
                "multiple instruments, build a scalar first-stage index "
                "first via sp.scalar_iv_projection(data, treat=..., "
                "instruments=[...], covariates=[...]), then declare it "
                "as the single instrument on the CausalQuestion."
            )
        if is_iv_model:
            instrument = kwargs.pop(
                "instrument",
                q.instruments[0] if len(q.instruments) == 1
                else list(q.instruments),
            )
        else:
            # Non-IV model: drop any incoming instrument (from the
            # question's declaration or kwargs). Warn loudly when the
            # user's explicit model override clashes with declared
            # instruments — that's almost always a mistake.
            instrument = None
            if q.instruments:
                warnings.warn(
                    f"DML model={model!r} does not accept instruments; "
                    f"the declared instruments {list(q.instruments)!r} "
                    "are being ignored. Use model='pliv' or 'iivm' to "
                    "include them.",
                    UserWarning,
                    stacklevel=2,
                )
            kwargs.pop("instrument", None)
        res = sp.dml(
            data=data, y=q.outcome, treat=q.treatment,
            covariates=list(q.covariates),
            model=model, instrument=instrument, **kwargs,
        )
        est, se, ci = _extract_generic(res)
        return EstimationResult(
            estimand=plan.estimand, estimator=est_name,
            estimate=est, se=se, ci=ci, n=n,
            underlying=res, plan=plan,
        )

    if est_name == "tmle":
        if not q.covariates:
            raise ValueError(
                "design='tmle' requires covariates=[...] for the outcome "
                "and propensity nuisance models."
            )
        _reject_reserved_kwargs(
            kwargs, "tmle",
            reserved=("data", "y", "treat", "covariates"),
        )
        # plan.estimand has already been normalised to ATE/ATT by
        # _pick_plan; respect any explicit kwargs['estimand'] override.
        tmle_estimand = kwargs.pop("estimand", plan.estimand)
        if tmle_estimand not in ("ATE", "ATT"):
            tmle_estimand = "ATE"
        res = sp.tmle(
            data=data, y=q.outcome, treat=q.treatment,
            covariates=list(q.covariates),
            estimand=tmle_estimand, **kwargs,
        )
        est, se, ci = _extract_generic(res)
        return EstimationResult(
            estimand=plan.estimand, estimator=est_name,
            estimate=est, se=se, ci=ci, n=n,
            underlying=res, plan=plan,
        )

    if est_name == "metalearner":
        if not q.covariates:
            raise ValueError(
                "design='metalearner' requires covariates=[...] — these "
                "are the effect modifiers used to estimate tau(x)."
            )
        _reject_reserved_kwargs(
            kwargs, "metalearner",
            reserved=("data", "y", "treat", "covariates"),
        )
        # sp.metalearner does not accept `random_state` directly. When
        # the user passes one via q.estimate(), translate it into
        # seeded outcome / propensity nuisance models so the AIPW SE
        # path becomes reproducible (the KFold inside
        # `_cross_fit_aipw_phi` uses a fixed seed already; the GBM
        # nuisance fits are the remaining source of randomness).
        meta_random_state = kwargs.pop("random_state", None)
        if meta_random_state is not None:
            from sklearn.ensemble import (
                GradientBoostingRegressor, GradientBoostingClassifier,
            )
            kwargs.setdefault(
                "outcome_model",
                GradientBoostingRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=meta_random_state,
                ),
            )
            kwargs.setdefault(
                "propensity_model",
                GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=meta_random_state,
                ),
            )
        # learner='dr' is sp.metalearner's default; SE comes from the
        # AIPW influence function regardless of learner family
        # (v1.11.4+ behaviour).
        res = sp.metalearner(
            data=data, y=q.outcome, treat=q.treatment,
            covariates=list(q.covariates),
            **kwargs,
        )
        est, se, ci = _extract_generic(res)
        return EstimationResult(
            estimand=plan.estimand, estimator=est_name,
            estimate=est, se=se, ci=ci, n=n,
            underlying=res, plan=plan,
        )

    if est_name == "causal_forest":
        if not q.covariates:
            raise ValueError(
                "design='causal_forest' requires covariates=[...] as "
                "effect modifiers."
            )
        _reject_reserved_kwargs(
            kwargs, "causal_forest",
            reserved=("data", "Y", "T", "X", "formula"),
        )
        # AIPW kwargs we recognise (mirrors sp.metalearner). All other
        # kwargs flow through to sp.causal_forest. `random_state` is
        # *peeked* — left in kwargs so the forest receives it AND
        # forwarded to AIPW so a single seed reproduces the whole
        # branch (forest CATE + ATE/SE/CI). Passing nothing → both
        # random; passing an int → both reproducible (no asymmetry).
        aipw_n_folds = kwargs.pop("aipw_n_folds", 5)
        alpha = kwargs.pop("alpha", 0.05)
        aipw_random_state = kwargs.get("random_state", None)
        Y = data[q.outcome].to_numpy()
        T_raw = data[q.treatment].to_numpy()
        X = data[list(q.covariates)].to_numpy()
        # AIPW-IF for ATE assumes binary D. Validate early so the user
        # gets a pointed error instead of an opaque sklearn classifier
        # failure deep inside sp.causal_forest's nuisance fit. Cast
        # defensively (non-numeric strings raise here with the
        # binary-treatment hint, not a cryptic NumPy error). After
        # validation forward the cleaned numeric array — otherwise a
        # numeric-string column (e.g. '0'/'1') would crash the forest.
        try:
            T_numeric = np.asarray(T_raw).astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "design='causal_forest' requires a numeric binary "
                "treatment (0/1) for AIPW-IF inference; got a "
                f"non-numeric column ({exc!s}). For continuous "
                "treatments use design='dml'."
            ) from exc
        T_unique = np.unique(T_numeric[~np.isnan(T_numeric)])
        if not set(T_unique.tolist()) <= {0.0, 1.0}:
            raise ValueError(
                "design='causal_forest' requires a binary treatment "
                "(0/1) for AIPW-IF inference; for continuous treatments "
                "use design='dml'."
            )
        # Use the cleaned numeric array everywhere downstream.
        T = T_numeric
        cf = sp.causal_forest(Y=Y, T=T, X=X, **kwargs)
        # The forest is preserved as `underlying` for CATE access via
        # cf.effect(X). Population ATE inference uses the cross-fit
        # AIPW influence function (van der Laan & Robins 2003;
        # Chernozhukov et al. 2018) — semiparametrically efficient and
        # B-independent, exactly the approach
        # grf::average_treatment_effect uses in R for ATE inference on
        # top of a causal forest.
        ate, se, ci = _ate_inference_aipw(
            X=X, Y=Y, D=T,
            n_folds=aipw_n_folds, alpha=alpha,
            random_state=aipw_random_state,
        )
        return EstimationResult(
            estimand=plan.estimand, estimator=est_name,
            estimate=ate, se=se, ci=ci, n=n,
            underlying=cf, plan=plan,
        )

    raise NotImplementedError(
        f"Dispatch for estimator {est_name!r} not implemented."
    )


def _reject_reserved_kwargs(kwargs: dict, design: str, *,
                            reserved: tuple[str, ...]) -> None:
    """Raise a clear TypeError if user kwargs collide with positional
    arguments the dispatcher fills from the CausalQuestion fields.

    Plain ``**kwargs`` forwarding would otherwise produce
    ``TypeError: <fn>() got multiple values for keyword argument 'y'``
    far from the user's call site; this guard catches it early.
    """
    bad = [k for k in reserved if k in kwargs]
    if bad:
        raise TypeError(
            f"design={design!r}: the following kwargs collide with the "
            f"CausalQuestion fields and cannot be passed to estimate(): "
            f"{bad!r}. They are populated from "
            "the CausalQuestion's treatment/outcome/covariates/data."
        )


def _ate_inference_aipw(
    *, X, Y, D,
    n_folds: int = 5,
    alpha: float = 0.05,
    random_state: Optional[int] = 42,
) -> tuple[float, float, tuple[float, float]]:
    """Population ATE point + SE + Wald CI via cross-fit AIPW (DR)
    influence function — semiparametrically efficient under
    conditional ignorability (van der Laan & Robins 2003;
    Chernozhukov et al. 2018).

    Used by the ``causal_forest`` dispatch branch to attach valid
    inference to the forest's CATE estimator: forest provides
    heterogeneous tau(x), AIPW provides the population ATE summary.
    This mirrors what grf::average_treatment_effect does in R.

    Nuisance hyper-parameters (n_estimators=200, max_depth=4,
    learning_rate=0.05, subsample=0.8) match
    :func:`sp.metalearner`'s ``_default_outcome_model`` /
    ``_default_propensity_model`` so the two dispatch paths share the
    same effective regularisation. ``random_state`` is threaded to
    BOTH nuisance models AND the cross-fit fold split, so a single
    seed makes the whole estimator branch reproducible.

    Treatment must be binary (0/1); for continuous treatments use
    ``design='dml'`` instead.
    """
    from ..metalearners.metalearners import _cross_fit_aipw_phi
    from sklearn.ensemble import (
        GradientBoostingRegressor, GradientBoostingClassifier,
    )

    Y = np.asarray(Y).astype(float).ravel()
    # Cast to float so isnan works on originally-int treatments too.
    D_raw = np.asarray(D).astype(float).ravel()
    D_clean = D_raw[~np.isnan(D_raw)]
    if set(np.unique(D_clean).tolist()) - {0.0, 1.0}:
        raise ValueError(
            "AIPW-IF for ATE requires binary treatment (0/1); for "
            "continuous treatments use design='dml'."
        )
    D = D_raw.astype(int)
    X = np.asarray(X)
    n = len(Y)

    outcome_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=random_state,
    )
    propensity_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=random_state,
    )

    # KFold inherits the same seed: passing None → sklearn uses np
    # global state (genuinely random); passing int → deterministic.
    # _cross_fit_aipw_phi accepts None via KFold's standard semantics.
    phi, _diag = _cross_fit_aipw_phi(
        X=X, Y=Y, D=D,
        outcome_model=outcome_model,
        propensity_model=propensity_model,
        n_folds=n_folds,
        seed=random_state,
    )
    ate = float(np.mean(phi))
    se = float(np.std(phi, ddof=1) / np.sqrt(n))
    from scipy.stats import norm
    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci = (ate - z * se, ate + z * se)
    return ate, se, ci


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
