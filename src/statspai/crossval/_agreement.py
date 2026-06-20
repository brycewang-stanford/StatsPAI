"""Engine-estimate normalisation, tolerance policy and verdict logic.

This module is the numerical heart of :func:`statspai.cross_validate`. It has
no heavy dependencies (numpy only) so the agreement machinery can be unit
tested in isolation from any estimator backend.

Three concepts live here:

``EngineEstimate``
    A backend-agnostic record of *one* engine's answer for *one* focal
    coefficient: point estimate, standard error, CI, n, plus a ``status`` so
    unavailable / errored engines flow through the same pipeline as successful
    ones (failures stay loud, never silently dropped — CLAUDE.md §3 #7).

``TolerancePolicy``
    How close two engines *should* be, and **why**. The "why" matters: a plain
    OLS coefficient computed by two libraries should agree to floating-point
    noise (``exact`` mode), whereas a double-ML point estimate depends on the
    random cross-fitting split and can only be expected to agree *statistically*
    (``statistical`` mode — within a few standard errors). Encoding the
    rationale on the policy is the in-place ``atol``/``rtol`` justification
    CLAUDE.md §5 asks for.

``reconcile``
    Turn a list of ``EngineEstimate`` into a verdict
    (AGREE / PARTIAL / DISAGREE / INSUFFICIENT) plus spread diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Status + verdict vocabularies
# --------------------------------------------------------------------------- #

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"  # engine / binary / package not installed
STATUS_ERROR = "error"  # engine ran but raised
STATUS_SKIPPED = "skipped"  # engine cannot express this estimand

VERDICT_AGREE = "AGREE"
VERDICT_PARTIAL = "PARTIAL"  # point estimates agree, inference (SE) does not
VERDICT_DISAGREE = "DISAGREE"
VERDICT_INSUFFICIENT = "INSUFFICIENT"  # < 2 engines produced an estimate


@dataclass
class EngineEstimate:
    """One engine's answer for one focal coefficient.

    Every backend adapter normalises its native result into this shape so the
    reconciliation logic never has to special-case a library. ``status`` keeps
    unavailable / failed engines in the data flow rather than dropping them.

    Parameters
    ----------
    engine : str
        Backend label, e.g. ``"statspai"``, ``"pyfixest"``,
        ``"linearmodels"``, ``"R::fixest"``, ``"Stata::reghdfe"``.
    estimand : str
        The estimand key that was requested (``"ols"``, ``"iv"``, ``"did"`` …).
    term : str
        Name of the focal coefficient this estimate refers to.
    coef, se : float, optional
        Point estimate and standard error of ``term``. ``None`` when the
        engine did not produce one (status != ok).
    tstat, pvalue, ci_lower, ci_upper : float, optional
    nobs : int, optional
    vcov : str, optional
        Variance estimator flavour actually used (``"iid"``, ``"HC1"``,
        ``"cluster"`` …) so a SE mismatch can be attributed to a vcov
        difference rather than a genuine disagreement.
    status : str
        One of ``ok`` / ``unavailable`` / ``error`` / ``skipped``.
    message : str, optional
        Human-readable note (why it was unavailable, the exception text …).
    elapsed_s : float, optional
    extra : dict
        Free-form backend extras (first-stage F, n_folds, full coef table …).
    """

    engine: str
    estimand: str
    term: Optional[str] = None
    coef: Optional[float] = None
    se: Optional[float] = None
    tstat: Optional[float] = None
    pvalue: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    nobs: Optional[int] = None
    vcov: Optional[str] = None
    status: str = STATUS_OK
    message: Optional[str] = None
    elapsed_s: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and self.coef is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "estimand": self.estimand,
            "term": self.term,
            "coef": _nullable_float(self.coef),
            "se": _nullable_float(self.se),
            "tstat": _nullable_float(self.tstat),
            "pvalue": _nullable_float(self.pvalue),
            "ci_lower": _nullable_float(self.ci_lower),
            "ci_upper": _nullable_float(self.ci_upper),
            "nobs": self.nobs,
            "vcov": self.vcov,
            "status": self.status,
            "message": self.message,
            "elapsed_s": _nullable_float(self.elapsed_s),
        }


# --------------------------------------------------------------------------- #
# Tolerance policy
# --------------------------------------------------------------------------- #

# Estimand families and the agreement regime we can *honestly* expect.
#
#   exact       — closed-form / deterministic algebra. Two correct
#                 implementations differ only by floating-point and
#                 dof-convention noise. Coefficients must match tightly.
#   statistical — the estimator embeds randomness (sample splitting, RNG
#                 forests, MCMC). Two correct runs need NOT match to many
#                 digits; we judge agreement on a standard-error scale.
#
# These keys are matched case-insensitively against the requested estimand.
_EXACT_ESTIMANDS = {
    "ols",
    "regress",
    "feols",
    "iv",
    "ivreg",
    "2sls",
    "tsls",
    "liml",
    "did",
    "did_2x2",
    "twfe",
    "panel",
    "panel_fe",
    "fixest",
    "hdfe",
    "wls",
}
_ITERATIVE_ESTIMANDS = {
    "poisson",
    "fepois",
    "logit",
    "probit",
    "glm",
    "ppml",
    "ppmlhdfe",
    "negbin",
    "nbreg",
}
_STATISTICAL_ESTIMANDS = {
    "dml",
    "double_ml",
    "causal_forest",
    "grf",
    "metalearner",
    "xlearner",
    "aipw",
    "tmle",
    "bcf",
    "bart",
    "bayes",
}


@dataclass
class TolerancePolicy:
    """How close two engines should be on a focal coefficient, and why.

    Attributes
    ----------
    mode : str
        ``"exact"`` (judge coefficients on a relative-difference scale) or
        ``"statistical"`` (judge on a standard-error scale).
    coef_rtol, coef_atol : float
        Relative / absolute tolerance on the point estimate (``exact`` mode).
    se_rtol : float
        Relative tolerance on the standard error (``exact`` mode). SEs are
        looser than coefficients because dof corrections and default vcov
        flavours legitimately differ across libraries.
    se_band : float
        ``statistical`` mode: two estimates agree if
        ``|Δcoef| <= se_band * max(se)``.
    rationale : str
        Plain-language justification, surfaced in the report so a reader can
        see *why* a given tolerance was applied.
    """

    mode: str
    coef_rtol: float = 1e-6
    coef_atol: float = 1e-8
    se_rtol: float = 5e-3
    se_band: float = 0.25
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "coef_rtol": self.coef_rtol,
            "coef_atol": self.coef_atol,
            "se_rtol": self.se_rtol,
            "se_band": self.se_band,
            "rationale": self.rationale,
        }


def default_policy(estimand: str) -> TolerancePolicy:
    """Pick a tolerance regime from the estimand name.

    The defaults are deliberately conservative and *annotated*: each carries a
    ``rationale`` string so the report explains, rather than hides, why the
    pass/fail bar sits where it does.
    """
    key = (estimand or "").strip().lower()

    if key == "did":
        return TolerancePolicy(
            mode="exact",
            coef_rtol=1e-3,
            coef_atol=1e-6,
            se_rtol=0.2,
            rationale=(
                "Callaway–Sant'Anna overall ATT. The point estimate is "
                "deterministic given (estimator, control group, base period, "
                "aggregation); StatsPAI reproduces R's `did` to ~1e-6 on the "
                "canonical mpdta. Compared at rtol=1e-3 to absorb small-sample "
                "doubly-robust propensity/outcome differences; standard errors "
                "loosely (R defaults to a multiplier bootstrap, StatsPAI to "
                "the analytic influence-function SE)."
            ),
        )
    if key in _STATISTICAL_ESTIMANDS:
        return TolerancePolicy(
            mode="statistical",
            se_band=0.25,
            rationale=(
                "Estimator embeds randomness (cross-fitting / RNG / MCMC); two "
                "correct runs need not match to many digits. Agreement is "
                "judged on a standard-error scale (|Δcoef| <= 0.25·max SE)."
            ),
        )
    if key in _ITERATIVE_ESTIMANDS:
        return TolerancePolicy(
            mode="exact",
            coef_rtol=1e-4,
            coef_atol=1e-6,
            se_rtol=1e-2,
            rationale=(
                "Iterative MLE/IRLS fit; convergence tolerances differ across "
                "libraries, so coefficients are compared at rtol=1e-4 rather "
                "than floating-point noise."
            ),
        )
    if key in _EXACT_ESTIMANDS:
        return TolerancePolicy(
            mode="exact",
            coef_rtol=1e-6,
            coef_atol=1e-8,
            se_rtol=5e-3,
            rationale=(
                "Closed-form least-squares / IV algebra; two correct "
                "implementations differ only by floating-point and dof-"
                "convention noise. Coefficients compared at rtol=1e-6; SEs at "
                "rtol=5e-3 to absorb default vcov / dof differences."
            ),
        )
    # Unknown estimand: be permissive on coefficients but still flag gross
    # disagreement. Statistical-scale judgement is the safer default because it
    # does not assume the method is deterministic.
    return TolerancePolicy(
        mode="statistical",
        se_band=0.5,
        rationale=(
            "Unrecognised estimand; defaulting to a permissive standard-error-"
            "scale comparison (|Δcoef| <= 0.5·max SE). Pass an explicit `tol=` "
            "to tighten."
        ),
    )


def resolve_policy(estimand: str, tol: Optional[Any] = None) -> TolerancePolicy:
    """Resolve the effective policy: explicit ``tol`` overrides the default.

    ``tol`` may be a :class:`TolerancePolicy`, a ``dict`` of overrides merged
    onto the default, or ``None`` (use the default for ``estimand``).
    """
    if isinstance(tol, TolerancePolicy):
        return tol
    base = default_policy(estimand)
    if tol is None:
        return base
    if isinstance(tol, dict):
        merged = {**base.to_dict(), **tol}
        merged.pop("rationale", None)  # keep base rationale unless overridden
        rationale = tol.get(
            "rationale", base.rationale + " [user-overridden tolerances]"
        )
        return TolerancePolicy(rationale=rationale, **merged)
    raise TypeError(
        f"tol must be a dict, TolerancePolicy or None, got {type(tol).__name__}"
    )


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #


@dataclass
class AgreementReport:
    """Outcome of reconciling several engines on one focal coefficient."""

    verdict: str
    policy: TolerancePolicy
    reference: Optional[str]  # engine used as the comparison anchor
    n_ok: int
    n_requested: int
    max_abs_coef_diff: Optional[float]
    max_rel_coef_diff: Optional[float]
    max_abs_se_diff: Optional[float]
    sign_agree: Optional[bool]
    significance_agree: Optional[bool]
    ci_overlap: Optional[bool]
    pairwise: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "policy": self.policy.to_dict(),
            "reference": self.reference,
            "n_ok": self.n_ok,
            "n_requested": self.n_requested,
            "max_abs_coef_diff": _nullable_float(self.max_abs_coef_diff),
            "max_rel_coef_diff": _nullable_float(self.max_rel_coef_diff),
            "max_abs_se_diff": _nullable_float(self.max_abs_se_diff),
            "sign_agree": self.sign_agree,
            "significance_agree": self.significance_agree,
            "ci_overlap": self.ci_overlap,
            "pairwise": self.pairwise,
            "notes": self.notes,
        }


def reconcile(
    estimates: List[EngineEstimate],
    *,
    policy: TolerancePolicy,
    reference: Optional[str] = None,
    alpha: float = 0.05,
) -> AgreementReport:
    """Reduce a list of engine estimates to a verdict + spread diagnostics.

    Parameters
    ----------
    estimates : list of EngineEstimate
        All engines that were *requested*, including unavailable / errored
        ones (they count toward ``n_requested`` but not the comparison).
    policy : TolerancePolicy
    reference : str, optional
        Engine name to anchor pairwise comparison on. Defaults to the first
        ``ok`` estimate (callers usually pass ``"statspai"``).
    alpha : float
        Significance level for the significance-agreement check.
    """
    ok = [e for e in estimates if e.ok]
    n_ok = len(ok)
    n_req = len(estimates)
    notes: List[str] = []

    if n_ok < 2:
        if n_ok == 0:
            notes.append("No engine produced an estimate.")
        else:
            notes.append(
                f"Only one engine ({ok[0].engine}) produced an estimate; "
                "cross-validation needs at least two independent engines."
            )
        # Surface why the others did not run.
        for e in estimates:
            if not e.ok:
                notes.append(f"{e.engine}: {e.status} — {e.message or ''}".strip())
        return AgreementReport(
            verdict=VERDICT_INSUFFICIENT,
            policy=policy,
            reference=ok[0].engine if ok else None,
            n_ok=n_ok,
            n_requested=n_req,
            max_abs_coef_diff=None,
            max_rel_coef_diff=None,
            max_abs_se_diff=None,
            sign_agree=None,
            significance_agree=None,
            ci_overlap=None,
            notes=notes,
        )

    # Choose the anchor.
    ref = _pick_reference(ok, reference)
    ref_coef = float(ref.coef)  # type: ignore[arg-type]

    coefs = np.array([float(e.coef) for e in ok])  # type: ignore[arg-type]
    ses = np.array([float(e.se) if e.se is not None else np.nan for e in ok])

    max_abs = float(np.max(np.abs(coefs - ref_coef)))
    denom = max(abs(ref_coef), policy.coef_atol)
    max_rel = float(np.max(np.abs(coefs - ref_coef)) / denom)
    valid_se = ses[~np.isnan(ses)]
    max_se_diff = (
        float(np.max(np.abs(valid_se - _ref_se(ref))))
        if ref.se is not None and valid_se.size
        else None
    )

    sign_agree = bool(len(set(np.sign(np.where(coefs == 0, 0.0, coefs)))) <= 1)
    significance_agree = _significance_agreement(ok, alpha)
    ci_overlap = _ci_overlap(ok)

    # Pairwise records (vs reference) for the report table.
    pairwise: List[Dict[str, Any]] = []
    for e in ok:
        if e.engine == ref.engine:
            continue
        d_coef = float(e.coef) - ref_coef  # type: ignore[arg-type]
        rel = abs(d_coef) / denom
        pairwise.append(
            {
                "engine": e.engine,
                "reference": ref.engine,
                "delta_coef": d_coef,
                "rel_coef": rel,
                "delta_se": (
                    float(e.se) - _ref_se(ref)
                    if (e.se is not None and ref.se is not None)
                    else None
                ),
                "vcov_self": e.vcov,
                "vcov_ref": ref.vcov,
            }
        )

    verdict = _verdict(
        ok,
        ref=ref,
        policy=policy,
        max_rel=max_rel,
        max_abs=max_abs,
        notes=notes,
    )

    return AgreementReport(
        verdict=verdict,
        policy=policy,
        reference=ref.engine,
        n_ok=n_ok,
        n_requested=n_req,
        max_abs_coef_diff=max_abs,
        max_rel_coef_diff=max_rel,
        max_abs_se_diff=max_se_diff,
        sign_agree=sign_agree,
        significance_agree=significance_agree,
        ci_overlap=ci_overlap,
        pairwise=pairwise,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _verdict(
    ok: List[EngineEstimate],
    *,
    ref: EngineEstimate,
    policy: TolerancePolicy,
    max_rel: float,
    max_abs: float,
    notes: List[str],
) -> str:
    coefs = np.array([float(e.coef) for e in ok])  # type: ignore[arg-type]
    ref_coef = float(ref.coef)  # type: ignore[arg-type]

    if policy.mode == "statistical":
        ses = np.array([float(e.se) for e in ok if e.se is not None])
        if ses.size == 0:
            notes.append(
                "Statistical-mode comparison needs standard errors but none "
                "were reported; falling back to sign agreement only."
            )
            signs = set(np.sign(np.where(coefs == 0, 0.0, coefs)))
            return VERDICT_AGREE if len(signs) <= 1 else VERDICT_DISAGREE
        band = policy.se_band * float(np.max(ses))
        within = float(np.max(np.abs(coefs - ref_coef))) <= band
        signs = set(np.sign(np.where(coefs == 0, 0.0, coefs)))
        if within and len(signs) <= 1:
            return VERDICT_AGREE
        if len(signs) <= 1:
            notes.append(
                f"Estimates share sign but differ by more than "
                f"{policy.se_band:g}·max SE ({band:.4g})."
            )
            return VERDICT_PARTIAL
        return VERDICT_DISAGREE

    # exact mode
    coef_ok = (max_abs <= policy.coef_atol) or (max_rel <= policy.coef_rtol)
    if not coef_ok:
        return VERDICT_DISAGREE

    # Coefficients agree — now check inference (SE) when available.
    ref_se = _ref_se(ref)
    se_vals = [float(e.se) for e in ok if e.se is not None]
    if ref.se is None or len(se_vals) < 2:
        notes.append(
            "Point estimates agree; standard errors not compared (missing on "
            "one or more engines)."
        )
        return VERDICT_AGREE
    se_arr = np.array(se_vals)
    se_denom = max(abs(ref_se), 1e-12)
    se_rel = float(np.max(np.abs(se_arr - ref_se)) / se_denom)
    if se_rel <= policy.se_rtol:
        return VERDICT_AGREE
    notes.append(
        f"Point estimates agree (rel {max_rel:.2e}) but standard errors differ "
        f"(rel {se_rel:.2e} > {policy.se_rtol:g}); check vcov flavour "
        "alignment across engines."
    )
    return VERDICT_PARTIAL


def _pick_reference(
    ok: List[EngineEstimate], reference: Optional[str]
) -> EngineEstimate:
    if reference is not None:
        for e in ok:
            if e.engine == reference:
                return e
    # Prefer statspai as the natural anchor, else the first ok engine.
    for e in ok:
        if e.engine == "statspai":
            return e
    return ok[0]


def _ref_se(ref: EngineEstimate) -> float:
    return float(ref.se) if ref.se is not None else float("nan")


def _significance_agreement(ok: List[EngineEstimate], alpha: float) -> Optional[bool]:
    flags = []
    for e in ok:
        if e.pvalue is not None:
            flags.append(e.pvalue < alpha)
        elif e.ci_lower is not None and e.ci_upper is not None:
            flags.append(not (e.ci_lower <= 0.0 <= e.ci_upper))
    if len(flags) < 2:
        return None
    return len(set(flags)) <= 1


def _ci_overlap(ok: List[EngineEstimate]) -> Optional[bool]:
    intervals = [
        (e.ci_lower, e.ci_upper)
        for e in ok
        if e.ci_lower is not None and e.ci_upper is not None
    ]
    if len(intervals) < 2:
        return None
    lo = max(i[0] for i in intervals)  # type: ignore[type-var]
    hi = min(i[1] for i in intervals)  # type: ignore[type-var]
    return bool(lo <= hi)


def _nullable_float(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(xf) or np.isinf(xf):
        return None
    return xf
