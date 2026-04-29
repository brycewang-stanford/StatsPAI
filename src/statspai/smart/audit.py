"""Reviewer-checklist audit of a fitted StatsPAI result.

``sp.audit(result)`` returns the *missing-evidence* view of a result:
which robustness / sensitivity / diagnostic checks a careful reviewer
would expect for this estimator family — and which of those have
already been run vs. are still missing on the result.

Distinct from neighbouring methods:

* :meth:`CausalResult.violations` — items already on ``model_info`` whose
  values *fail* their threshold ("checked but failed").
* :meth:`CausalResult.next_steps` — recommendations of what to *do next*,
  oriented around action (export, robustness, alternative method).
* :func:`statspai.smart.assumption_audit` — heavyweight: takes
  ``(result, data)`` and *re-runs* statistical tests. ``audit`` is
  pure introspection — it never re-runs anything, never touches data,
  and runs in microseconds.

The agent's mental model: ``audit`` answers "what evidence is missing
for a reviewer to trust this estimate?"; ``assumption_audit`` answers
"given the data, do the assumptions actually hold?".

Returns a JSON-safe ``dict`` so MCP-mediated agents can branch on the
``status`` field of each check without parsing prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core._agent_summary import (
    _as_float,
    _safe_get,
    # Threshold constants imported (not re-stated) so audit's verdict
    # cannot drift from violations() when a future correctness fix
    # updates a cutoff. Single source of truth for numerical thresholds.
    _ESS_MIN,
    _OVERLAP_MIN,
    _PRETREND_ALPHA,
    _RHAT_MAX,
    _SMD_MAX,
    _WEAK_IV_F,
)


# ====================================================================== #
#  Check specification
# ====================================================================== #


# A single evidence path is a tuple of dict keys; ``EvidencePaths`` is
# either one path or a tuple-of-tuples for estimators that store the
# same diagnostic under different aliases (e.g. IV first-stage F lives
# under ``first_stage_f`` *or* ``first_stage.f_stat`` *or*
# ``weak_iv_f`` depending on which estimator wrote the result).
EvidencePaths = Tuple[Tuple[str, ...], ...]


@dataclass(frozen=True)
class _Check:
    """One reviewer-relevant check the result either has or lacks.

    Attributes
    ----------
    name : str
        Short identifier — agents branch on this.
    question : str
        Reviewer-style human-readable prompt.
    applies_to : tuple[str, ...]
        Method families (per :func:`statspai.core.next_steps._detect_family`)
        for which this check is relevant.
    evidence_paths : tuple[tuple[str, ...], ...]
        Candidate paths of keys into ``result.model_info`` where the
        evidence would live. A *tuple of paths* — the first one that
        resolves wins — so estimators that store the same diagnostic
        under different aliases all surface as ``passed``.
    threshold : float | None
        Numeric threshold used to decide passed vs. failed when the
        evidence value is a number. ``None`` for existence-only checks.
    compare : str
        How to evaluate the value vs. the threshold:

        - ``"greater_passes"`` — value > threshold → passed
        - ``"less_passes"``    — value < threshold → passed
        - ``"exists"``          — non-empty dict at the path → passed.
                                   Strict-dict-only intentionally — a
                                   bare ``False`` / ``0`` / ``""``
                                   sentinel must NOT mark the check
                                   passed. The estimator contract is
                                   to write a sub-dict (e.g.
                                   ``{"pvalue": 0.5}`` or
                                   ``{"has_run": True}``) when a
                                   diagnostic has actually been
                                   computed.
        - ``"not_nonrobust"``   — robust SE flag is set to anything but
                                   ``"nonrobust"`` → passed
    suggest_function : str | None
        ``sp.xxx`` to run when this check is missing.
    importance : str
        ``"high"`` / ``"medium"`` / ``"low"`` — how badly missing this
        check undermines the conclusion. Constant per check; the
        output ``importance`` field always carries this value
        regardless of status, so agents can do
        ``if c["importance"] == "high"`` without worrying about the
        status branch.
    rationale : str
        One-sentence reviewer justification: *why* this check matters.
    """

    name: str
    question: str
    applies_to: Tuple[str, ...]
    evidence_paths: EvidencePaths
    threshold: Optional[float]
    compare: str
    suggest_function: Optional[str]
    importance: str
    rationale: str


def _p(*keys: str) -> EvidencePaths:
    """Single-path helper: ``_p("foo", "bar")`` → ``(("foo", "bar"),)``."""
    return (tuple(keys),)


def _pp(*paths: Tuple[str, ...]) -> EvidencePaths:
    """Multi-path helper: ``_pp(("a",), ("b", "c"))`` for aliases."""
    return tuple(paths)


# ====================================================================== #
#  Reviewer checklist by method family
# ====================================================================== #
#
# The thresholds mirror ``core/_agent_summary.py`` so a "passed" verdict
# from ``audit`` matches the absence of a violation in
# ``result.violations()`` — single source of truth for numerical
# cutoffs.

_CAUSAL_CHECKS: Tuple[_Check, ...] = (
    # --- DID --------------------------------------------------------- #
    _Check(
        name="parallel_trends",
        question="Are pre-treatment trends statistically parallel?",
        applies_to=("did",),
        evidence_paths=_p("pretrend_test", "pvalue"),
        threshold=_PRETREND_ALPHA,
        compare="greater_passes",
        suggest_function="sp.pretrends_test",
        importance="high",
        rationale="DID identification rests on parallel trends; without "
                  "a pre-trend test the design is unfalsifiable.",
    ),
    _Check(
        name="honest_did",
        question="Is the estimate robust to small parallel-trends violations?",
        applies_to=("did",),
        evidence_paths=_p("honest_did"),
        threshold=None,
        compare="exists",
        suggest_function="sp.sensitivity_rr",
        importance="medium",
        rationale="Pre-trend tests are low-power; Rambachan-Roth (2023) "
                  "honest CIs quantify how much violation the estimate "
                  "tolerates.",
    ),
    _Check(
        name="bacon_decomposition",
        question="Are TWFE weights non-negative across cohort comparisons?",
        applies_to=("did",),
        evidence_paths=_p("bacon_decomposition"),
        threshold=None,
        compare="exists",
        suggest_function="sp.bacon_decomposition",
        importance="medium",
        rationale="Goodman-Bacon (2021): staggered-DID TWFE can hide "
                  "negative weights that flip the sign of the average.",
    ),
    # --- RD ---------------------------------------------------------- #
    _Check(
        name="mccrary_density",
        question="Is the running variable density continuous at the cutoff?",
        applies_to=("rd",),
        evidence_paths=_p("mccrary", "pvalue"),
        threshold=0.05,
        compare="greater_passes",
        suggest_function="sp.mccrary",
        importance="high",
        rationale="Manipulation of the running variable invalidates the "
                  "RD identifying assumption.",
    ),
    _Check(
        name="bandwidth_sensitivity",
        question="Is the estimate stable across alternative bandwidths?",
        applies_to=("rd",),
        evidence_paths=_p("bandwidth_sensitivity"),
        threshold=None,
        compare="exists",
        suggest_function="sp.rd_bandwidth_sensitivity",
        importance="medium",
        rationale="Optimal-bandwidth selection is a researcher degree "
                  "of freedom; reviewers expect a sensitivity sweep.",
    ),
    _Check(
        name="placebo_cutoff",
        question="Does the effect disappear at placebo cutoffs?",
        applies_to=("rd",),
        evidence_paths=_p("placebo_cutoff"),
        threshold=None,
        compare="exists",
        suggest_function="sp.rd_placebo",
        importance="medium",
        rationale="A real RD effect should not appear at off-cutoff "
                  "values; placebos guard against spurious discontinuity.",
    ),
    # --- IV ---------------------------------------------------------- #
    _Check(
        name="weak_instrument",
        question="Is the first-stage F above the weak-instrument threshold?",
        applies_to=("iv",),
        # Three aliases mirror ``causal_violations`` (see
        # core/_agent_summary.py) so audit and violations agree on the
        # same result regardless of which IV estimator wrote it.
        evidence_paths=_pp(
            ("first_stage_f",),
            ("first_stage", "f_stat"),
            ("weak_iv_f",),
        ),
        threshold=_WEAK_IV_F,
        compare="greater_passes",
        suggest_function=None,
        importance="high",
        rationale="Stock-Yogo (2005): below F≈10 the 2SLS sampling "
                  "distribution is far from normal and inference is "
                  "untrustworthy.",
    ),
    _Check(
        name="overid_test",
        question="Is the over-identification (Hansen J / Sargan) test passed?",
        applies_to=("iv",),
        evidence_paths=_pp(
            ("hansen_j", "pvalue"),
            ("sargan", "pvalue"),
        ),
        threshold=0.05,
        compare="greater_passes",
        suggest_function="sp.hansen_j",
        importance="medium",
        rationale="With ≥2 instruments, a J test rejection is direct "
                  "evidence one of them is invalid.",
    ),
    _Check(
        name="anderson_rubin_ci",
        question="Has weak-IV-robust inference been computed?",
        applies_to=("iv",),
        evidence_paths=_p("anderson_rubin"),
        threshold=None,
        compare="exists",
        suggest_function="sp.anderson_rubin_ci",
        importance="medium",
        rationale="Anderson-Rubin CIs are valid even under weak "
                  "instruments — the safe inference to pair with 2SLS.",
    ),
    # --- Matching / IPW --------------------------------------------- #
    _Check(
        name="overlap",
        question="Is propensity-score overlap adequate?",
        applies_to=("matching", "dml"),
        evidence_paths=_p("overlap", "min_share"),
        threshold=_OVERLAP_MIN,
        compare="greater_passes",
        suggest_function="sp.overlap_check",
        importance="high",
        rationale="Without common support across treatment arms, IPW / "
                  "matching extrapolate beyond the data.",
    ),
    _Check(
        name="balance_after",
        question="Is the post-match standardised mean difference acceptable?",
        applies_to=("matching",),
        evidence_paths=_p("balance", "max_smd_after"),
        threshold=_SMD_MAX,
        compare="less_passes",
        suggest_function="sp.balance",
        importance="high",
        rationale="Imbalance after matching reintroduces confounding "
                  "that the design was meant to remove.",
    ),
    # --- Synth ------------------------------------------------------ #
    _Check(
        name="pretreatment_fit",
        question="Is the synthetic control's pre-treatment RMSE small "
                  "relative to the outcome SD?",
        applies_to=("synth",),
        evidence_paths=_p("pretreatment_rmse_ratio"),
        threshold=0.5,
        compare="less_passes",
        suggest_function=None,
        importance="high",
        rationale="A synth that doesn't match the treated unit pre-period "
                  "cannot credibly extrapolate the counterfactual.",
    ),
    _Check(
        name="placebo_inference",
        question="Has placebo-permutation inference been run on donor units?",
        applies_to=("synth",),
        evidence_paths=_p("placebo_inference"),
        threshold=None,
        compare="exists",
        suggest_function="sp.synth_placebo",
        importance="high",
        rationale="Synth p-values come from ranking treated effect among "
                  "in-place placebos; without this, the estimate has no "
                  "inferential statement.",
    ),
    # --- Bayesian convergence (any family with MCMC) --------------- #
    _Check(
        name="convergence_rhat",
        question="Has the MCMC sampler converged (max R-hat ≤ 1.01)?",
        applies_to=("did", "rd", "iv", "matching", "synth", "dml", "hte",
                    "mediation", "generic"),
        evidence_paths=_pp(
            ("rhat_max",),
            ("diagnostics", "rhat_max"),
        ),
        threshold=_RHAT_MAX,
        compare="less_passes",
        suggest_function=None,
        importance="low",  # only "low" because non-Bayes runs legitimately
                            # have no rhat — high if Bayesian, low if not.
        rationale="Non-converged chains produce posterior summaries "
                  "that are not from the target distribution.",
    ),
    _Check(
        name="ess_bulk",
        question="Is the bulk effective sample size adequate (≥ 400)?",
        applies_to=("did", "rd", "iv", "matching", "synth", "dml", "hte",
                    "mediation", "generic"),
        evidence_paths=_pp(
            ("ess_bulk_min",),
            ("diagnostics", "ess_bulk_min"),
        ),
        threshold=_ESS_MIN,
        compare="greater_passes",
        suggest_function=None,
        importance="low",
        rationale="Small ESS makes MCSE on posterior summaries large "
                  "enough to swamp the substantive effect.",
    ),
    # --- Universal: omitted-variable bias --------------------------- #
    _Check(
        name="ovb_sensitivity",
        question="Has omitted-variable bias sensitivity been quantified?",
        applies_to=("matching", "dml", "generic"),
        evidence_paths=_p("oster"),
        threshold=None,
        compare="exists",
        suggest_function="sp.oster",
        importance="medium",
        rationale="Observational designs cannot rule out unobserved "
                  "confounders; sensitivity bounds tell readers how "
                  "strong such confounding would have to be.",
    ),
)


_REGRESSION_CHECKS: Tuple[_Check, ...] = (
    _Check(
        name="robust_se",
        question="Are robust / clustered standard errors used?",
        applies_to=("regression",),
        evidence_paths=_p("robust"),
        threshold=None,
        compare="not_nonrobust",
        suggest_function=None,
        importance="medium",
        rationale="Default homoskedastic SEs are almost never correct; "
                  "robust or clustered SEs are the modern baseline.",
    ),
    _Check(
        name="ovb_oster",
        question="Has Oster (2019) omitted-variable bias sensitivity been run?",
        applies_to=("regression",),
        evidence_paths=_p("oster"),
        threshold=None,
        compare="exists",
        suggest_function="sp.oster",
        importance="medium",
        rationale="Oster bounds quantify how strong a missing covariate "
                  "would need to be to nullify the result.",
    ),
    _Check(
        name="ovb_cinelli_hazlett",
        question="Have Cinelli-Hazlett (2020) bias bounds been computed?",
        applies_to=("regression",),
        evidence_paths=_p("cinelli_hazlett"),
        threshold=None,
        compare="exists",
        suggest_function="sp.cinelli_hazlett",
        importance="low",
        rationale="The robustness-value diagnostic complements Oster "
                  "and is easier to interpret graphically.",
    ),
)


# ====================================================================== #
#  Status evaluation
# ====================================================================== #


def _resolve_evidence(model_info: Dict[str, Any],
                       paths: EvidencePaths) -> Any:
    """Walk each candidate path in order; return the first resolved value.

    Returns ``None`` if no path resolves. Mirrors the multi-alias
    fallback ``causal_violations`` uses (e.g. IV first-stage F lives
    under ``first_stage_f`` *or* ``first_stage.f_stat`` *or*
    ``weak_iv_f``).
    """
    for path in paths:
        val = _safe_get(model_info, *path)
        if val is not None:
            return val
    return None


def _evaluate(check: _Check, model_info: Dict[str, Any]
              ) -> Tuple[str, Any]:
    """Return ``(status, raw_value)`` for one check on this result.

    ``status`` is one of ``"passed"`` / ``"failed"`` / ``"missing"``.
    """
    raw = _resolve_evidence(model_info, check.evidence_paths)

    # Existence-only checks. Strict-dict-only on purpose: a bare
    # ``False`` / ``0`` / ``""`` sentinel must NOT mark the check
    # passed, otherwise an estimator that wrote
    # ``model_info["honest_did"] = False`` to mean "tried but failed"
    # would silently report passed. Estimators must store a non-empty
    # sub-dict (e.g. ``{"pvalue": 0.5}`` or ``{"has_run": True}``)
    # when a diagnostic has actually been computed.
    if check.compare == "exists":
        if raw is None:
            return "missing", None
        if isinstance(raw, dict) and raw:
            return "passed", raw
        # Truthy but non-dict (e.g. a scalar) — leniently accept.
        if raw and not isinstance(raw, dict):
            return "passed", raw
        # Falsy or empty-dict sentinel → treat as not-actually-run.
        return "missing", raw

    if check.compare == "not_nonrobust":
        if raw is None:
            return "missing", None
        if isinstance(raw, str) and raw.lower() == "nonrobust":
            return "failed", raw
        return "passed", raw

    # Numeric comparisons
    if check.compare in ("greater_passes", "less_passes"):
        val = _as_float(raw)
        if val is None:
            return "missing", None
        thr = check.threshold
        if thr is None:  # malformed check spec
            return "missing", val
        if check.compare == "greater_passes":
            return ("passed" if val > thr else "failed"), val
        return ("passed" if val < thr else "failed"), val

    # Unknown compare verb — surface as missing rather than raise.
    return "missing", raw


# ====================================================================== #
#  Public API
# ====================================================================== #


def audit(result: Any) -> Dict[str, Any]:
    """Reviewer-checklist audit of a fitted StatsPAI result.

    Returns the *missing-evidence* view: which robustness / sensitivity /
    diagnostic checks the estimator family expects, and which of those
    are present, failed, or absent on the result.

    This is **read-only** — never re-runs a statistical test, never
    touches the original DataFrame, runs in microseconds. Pair it
    with :func:`statspai.smart.assumption_audit` (which *does* re-run
    tests against the data) when you need both perspectives.

    Parameters
    ----------
    result : CausalResult or EconometricResults
        Any fitted StatsPAI result with ``model_info`` attached.

    Returns
    -------
    dict
        JSON-safe payload with keys:

        - ``method`` (str) — the estimator's name
        - ``method_family`` (str) — one of ``"did"`` / ``"rd"`` / ``"iv"``
          / ``"synth"`` / ``"matching"`` / ``"dml"`` / ``"hte"`` /
          ``"regression"`` / ``"generic"``
        - ``checks`` (list[dict]) — every applicable check, each with
          ``name`` / ``question`` / ``status`` / ``severity`` / ``value``
          / ``threshold`` / ``suggest_function`` / ``rationale``
        - ``summary`` (dict) — count of ``passed`` / ``failed`` /
          ``missing`` / ``n_total``
        - ``coverage`` (float in [0, 1]) — ``passed / n_total``; agents
          can sort multiple results by reviewer-readiness

    Examples
    --------
    >>> r = sp.did(df, y='wage', treat='treated', time='post')
    >>> audit_card = sp.audit(r)
    >>> for c in audit_card['checks']:
    ...     if c['status'] == 'missing' and c['severity'] == 'high':
    ...         print(c['suggest_function'])
    sp.pretrends_test

    See Also
    --------
    statspai.smart.assumption_audit :
        Heavyweight counterpart: re-runs statistical tests against the
        original data and returns pass/fail per assumption.
    CausalResult.violations :
        Items already on ``model_info`` whose values fail thresholds
        ("checked-but-failed" view).
    CausalResult.next_steps :
        Action-oriented recommendations for what to do next.
    """
    from ..core.next_steps import _detect_family

    method = getattr(result, "method", None)
    if method is None:
        # EconometricResults: family determined from model_type.
        mi = getattr(result, "model_info", None) or {}
        model_type = (mi.get("model_type", "") or "").lower()
        if any(k in model_type for k in ("iv", "2sls", "liml", "gmm")):
            family = "iv"
        elif any(k in model_type for k in ("panel", "fixed", "random")):
            family = "regression"
        else:
            family = "regression"
        method_label = mi.get("model_type", "") or ""
    else:
        method_label = str(method)
        family = _detect_family(method_label.lower())

    model_info = getattr(result, "model_info", None) or {}
    # EconometricResults stores diagnostics under .diagnostics, not
    # model_info. Merge both so cross-class checks find their evidence.
    diag = getattr(result, "diagnostics", None) or {}
    if diag:
        merged = dict(model_info)
        for k, v in diag.items():
            merged.setdefault(k, v)
        model_info = merged

    pool: Sequence[_Check] = (
        _CAUSAL_CHECKS if family != "regression" else _REGRESSION_CHECKS
    )

    checks: List[Dict[str, Any]] = []
    n_passed = n_failed = n_missing = 0

    for chk in pool:
        if family not in chk.applies_to:
            continue
        status, value = _evaluate(chk, model_info)
        # ``severity`` and ``importance`` carry orthogonal vocabularies
        # so agents can branch on either without ambiguity:
        #   severity   ∈ {"info", "warning", "error"}
        #     — matches what ``violations()`` uses; reports observed
        #     status only.
        #   importance ∈ {"high", "medium", "low"}
        #     — constant per check; how badly missing/failing this
        #     check undermines the conclusion.
        if status == "passed":
            n_passed += 1
            severity_out = "info"
        elif status == "failed":
            n_failed += 1
            severity_out = "error"
        else:
            n_missing += 1
            # A high-importance missing check is a stronger signal than
            # a passed informational check, but agents that branch on
            # ``severity == "error"`` should not pick up missing items
            # — those go through ``importance``. Use ``"warning"``
            # here to mark missing-but-important consistently.
            severity_out = (
                "warning" if chk.importance == "high" else "info"
            )

        checks.append({
            "name": chk.name,
            "question": chk.question,
            "status": status,
            "severity": severity_out,
            "importance": chk.importance,
            "value": value if (isinstance(value, (int, float, str, bool))
                                or value is None) else str(value),
            "threshold": chk.threshold,
            "suggest_function": chk.suggest_function,
            "rationale": chk.rationale,
        })

    n_total = len(checks)
    coverage = (n_passed / n_total) if n_total else 0.0

    return {
        "method": method_label,
        "method_family": family,
        "checks": checks,
        "summary": {
            "passed": n_passed,
            "failed": n_failed,
            "missing": n_missing,
            "n_total": n_total,
        },
        "coverage": round(coverage, 3),
    }


__all__ = ["audit"]
