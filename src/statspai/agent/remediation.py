"""Error -> actionable-fix registry for agent self-repair loops.

Given a Python exception raised from a StatsPAI tool call, ``remediate()``
returns a structured suggestion the agent can use to repair its next
attempt.  The registry covers the most common failure modes seen in
practice:

- Missing columns / typos
- Non-binary treatment / wrong dtype
- Too few observations per cohort
- Identification blockers from ``strict=True`` mode
- Formula-parse errors
- Missing dependencies (sklearn, matplotlib, ...)
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Each pattern:
#   'match'       : regex matched against str(exception)
#   'exception'   : optional type constraint (class name string)
#   'category'    : short label
#   'diagnosis'   : what went wrong (for the agent to understand)
#   'fix'         : concrete next action (what the agent should try)

REMEDIATIONS = [
    # --- Column errors ---
    {
        # KeyError itself; any message inside (quoted column name);
        # also catches "not in index" / "column not found" for other types.
        'match': r".*",                  # matches when 'exception' filter hits
        'exception': ['KeyError'],
        'category': 'missing_column',
        'diagnosis': (
            "A requested column is not in the DataFrame. Most common "
            "cause: typo, or column belongs to a different design."
        ),
        'fix': (
            "Print df.columns and pick the correct name.  If the "
            "estimator needs a cohort column (g=) but you only have "
            "a binary treat, derive it first:\n"
            "  treated = df[df[treat] == 1]; cohort_per_unit = "
            "treated.groupby('id')['time'].min()"
        ),
    },
    {
        # Non-KeyError variants that still indicate missing columns
        'match': r"(column not found|not in index|column.*does not exist)",
        'category': 'missing_column',
        'diagnosis': "A requested column is not in the DataFrame.",
        'fix': "Print df.columns and pick the correct name.",
    },

    # --- Treatment variable errors ---
    {
        'match': r"(treatment must be binary|could not convert|invalid "
                 r"treat)",
        'category': 'non_binary_treatment',
        'diagnosis': (
            "Treatment column is not binary/integer-coded (0/1).  DID "
            "and most matching estimators require 0/1 coding."
        ),
        'fix': (
            "Recode: df['treat'] = (df['treat_original'] == 'yes').astype(int)."
        ),
    },

    # --- Identification blockers ---
    {
        'match': r"Identification has .* blocker",
        'exception': ['IdentificationError'],
        'category': 'identification_blocker',
        'diagnosis': (
            "strict=True + check_identification found a design blocker "
            "(no overlap / near-separation / no controls / etc.).  "
            "Do NOT retry the same spec; fix the design first."
        ),
        'fix': (
            "Re-run check_identification(strict=False) to see the full "
            "report.  Common fixes: (a) drop bad controls, (b) trim "
            "extreme propensity scores, (c) expand the sample, (d) "
            "pick a different design (e.g. IV instead of observational)."
        ),
    },

    # --- Singular matrix / collinearity ---
    {
        'match': r"(Singular matrix|LinAlgError|not.+invert|perfectly "
                 r"collinear)",
        'category': 'collinearity',
        'diagnosis': (
            "Design matrix is rank-deficient — two or more covariates "
            "are linearly dependent, or you're including a variable "
            "that's perfectly predicted by fixed effects."
        ),
        'fix': (
            "Drop one of the collinear variables, or use sp.vif(df) "
            "to identify them.  If you're using FE, make sure no "
            "time-invariant covariate enters alongside entity FE."
        ),
    },

    # --- Weak instrument ---
    {
        'match': r"(weak instrument|first-stage F.*<|first stage is "
                 r"undefined)",
        'category': 'weak_instrument',
        'diagnosis': (
            "First-stage F below Staiger-Stock threshold; 2SLS point "
            "estimate unreliable.  Your CI is not a valid 95% "
            "confidence region for the LATE."
        ),
        'fix': (
            "Use sp.liml() instead of ivreg (less bias under weak IVs), "
            "or construct Anderson-Rubin confidence sets with "
            "sp.weak_iv_ci(result, method='ar').  Alternatively, find "
            "a stronger instrument or collect a larger sample."
        ),
    },

    # --- Sample size / cohort size ---
    {
        'match': r"(too few|n_units.*<|small sample|insufficient "
                 r"observations|no observations)",
        'category': 'sample_size',
        'diagnosis': (
            "Sample / subsample / cohort is too small for the "
            "requested estimator."
        ),
        'fix': (
            "For DID with thin cohorts, aggregate cohorts (e.g. by year "
            "range) or report the simple ATT only.  For RD near the "
            "cutoff, widen the bandwidth.  For matching, relax the "
            "caliper or use coarsened exact matching."
        ),
    },

    # --- Formula parsing ---
    {
        'match': r"(PatsyError|Error evaluating factor|not in formula|"
                 r"tilde|'~')",
        'category': 'formula',
        'diagnosis': "Wilkinson-formula syntax error.",
        'fix': (
            "Ensure the formula has the form 'y ~ x1 + x2'.  For IV: "
            "'y ~ exog_x + (endog_d ~ instrument)'.  For interactions: "
            "'y ~ x1 * x2' (expands to x1 + x2 + x1:x2).  Column names "
            "with dots / spaces need backticks: `y` ~ `x 1`."
        ),
    },

    # --- Missing dependency ---
    {
        'match': r"(No module named|ModuleNotFoundError|ImportError)",
        'exception': ['ImportError', 'ModuleNotFoundError'],
        'category': 'missing_dependency',
        'diagnosis': (
            "An optional dependency is missing (sklearn / matplotlib "
            "/ torch / pyarrow).  StatsPAI core works without these; "
            "specific estimators or plots may need them."
        ),
        'fix': (
            "Install the extra: e.g. `pip install scikit-learn` for "
            "CBPS/ML-based estimators; `pip install matplotlib` for "
            "plots; `pip install torch` for deep-learning estimators "
            "(dragonnet, deepiv)."
        ),
    },

    # --- Bad kwargs ---
    {
        'match': r"(unexpected keyword argument|got an unexpected "
                 r"keyword)",
        'exception': ['TypeError'],
        'category': 'bad_argument',
        'diagnosis': (
            "Passed a keyword that the target function doesn't accept."
        ),
        'fix': (
            "Check the function signature: `help(sp.<fn>)` or "
            "`sp.<fn>?` in IPython.  Tool-manifest JSON schemas list "
            "every accepted argument."
        ),
    },

    # --- Data type ---
    {
        'match': r"(cannot convert|could not cast|not a valid type)",
        'category': 'data_type',
        'diagnosis': (
            "A column's dtype is not what the estimator expects "
            "(e.g. object/string where numeric is needed)."
        ),
        'fix': (
            "Coerce the column: df[col] = pd.to_numeric(df[col], "
            "errors='coerce').  Drop or impute resulting NaNs before "
            "re-running."
        ),
    },

    # ==================================================================
    # Causal-specific failure modes (P0 expansion, 2026-04-21)
    # ==================================================================

    # --- DID: parallel trends ---
    {
        'match': r"(parallel trend|pre-trend|pretrend).{0,40}(reject|violat|fail)",
        'category': 'parallel_trends_fail',
        'diagnosis': (
            "The parallel-trends assumption is in doubt — a pre-trend "
            "joint test rejects.  Two-way fixed-effects ATT estimates "
            "are likely biased."
        ),
        'fix': (
            "Run sp.sensitivity_rr(result) for Rambachan-Roth (2023) "
            "honest CIs.  Or switch to sp.callaway_santanna / "
            "sp.did_imputation which are robust to heterogeneous "
            "effects.  Consider dropping the longest-pre-period cohorts."
        ),
    },

    # --- DID: negative weights under TWFE ---
    {
        'match': r"(negative weight|forbidden comparison|bacon.*decomp)",
        'category': 'negative_weights_twfe',
        'diagnosis': (
            "Two-way FE gives negative weights to some 2x2 comparisons "
            "(Goodman-Bacon / de Chaisemartin).  Under heterogeneous "
            "effects the aggregate ATT estimate can flip sign."
        ),
        'fix': (
            "Switch to sp.callaway_santanna (group-time ATT) or "
            "sp.did_multiplegt (de Chaisemartin).  Run sp.bacon_decomposition "
            "to quantify how bad the weighting is."
        ),
    },

    # --- RD: McCrary / manipulation ---
    {
        'match': r"(mccrary|manipulation|density discontinuity).{0,40}(reject|significant|detect)",
        'category': 'mccrary_reject',
        'diagnosis': (
            "Running-variable density has a discontinuity at the "
            "cutoff (McCrary test rejects).  Sorting/manipulation "
            "undermines the continuity assumption — estimates may be "
            "confounded with selection."
        ),
        'fix': (
            "Try sp.rd_donut to exclude units just around the cutoff, "
            "or sp.bounds for partial identification.  Investigate the "
            "mechanism — administrative rounding often causes benign "
            "discontinuities."
        ),
    },

    # --- DML: orthogonality / score mean ≠ 0 ---
    {
        'match': r"(orthogonal.{0,20}(fail|violat)|score mean.{0,10}≠|score mean.{0,10}!=|dml.*bias)",
        'category': 'dml_ortho_fail',
        'diagnosis': (
            "Double-ML nuisance estimates did not yield a zero-mean "
            "orthogonal score — the debiasing step is compromised."
        ),
        'fix': (
            "Increase cross-fitting folds (n_folds=10+), use richer "
            "nuisance learners (GradientBoosting / XGBoost), and check "
            "propensity-score overlap.  For small n consider sp.tmle."
        ),
    },

    # --- Bayesian convergence ---
    {
        'match': r"(rhat.{0,6}>|max.?rhat|ess.{0,8}<|divergen|target_accept)",
        'category': 'bayes_convergence',
        'diagnosis': (
            "MCMC convergence diagnostics failed: rhat > 1.01, low ESS, "
            "or post-warmup divergences.  Posterior summaries are "
            "unreliable."
        ),
        'fix': (
            "Raise tune and draws (tune=4000, draws=4000).  Increase "
            "target_accept to 0.95+.  For funnel geometries use non-"
            "centered parametrisation.  Check priors for improper "
            "support."
        ),
    },

    # --- Matching / IPW: overlap violation ---
    {
        'match': r"(overlap.{0,40}(violat|fail|thin|poor)|common support|extreme propensity)",
        'category': 'overlap_violation',
        'diagnosis': (
            "Propensity-score overlap is thin — treated and control "
            "populations don't cover the same covariate region.  ATT / "
            "ATE on the full sample isn't point-identified."
        ),
        'fix': (
            "Apply Crump (2009) trimming via sp.trimming, or narrow "
            "the estimand to ATT on the overlap region.  Consider "
            "sp.ebalance (entropy balancing) or sp.sbw (stable-balancing "
            "weights) which side-step the propensity-score step."
        ),
    },

    # --- SBW infeasibility ---
    {
        'match': r"(sbw.*infeasib|balance constraint.*tight|sbw.*fail)",
        'category': 'sbw_infeasible',
        'diagnosis': (
            "Stable-balancing-weights optimisation is infeasible — "
            "covariate balance tolerance cannot be met given the "
            "sample."
        ),
        'fix': (
            "Relax the balance tolerance (moments=1 only, or loosen "
            "SMD target to 0.10).  Drop units with extreme propensity "
            "scores first.  Alternatively use sp.ebalance which is "
            "always feasible."
        ),
    },

    # --- Matching: post-match imbalance ---
    {
        'match': r"(smd.{0,6}>|covariates unbalanc|imbalance.{0,20}match)",
        'category': 'matching_unbalanced',
        'diagnosis': (
            "Matching left covariates unbalanced (max SMD > 0.10) — "
            "propensity-score model may be mis-specified."
        ),
        'fix': (
            "Tighten the caliper, add interactions / higher-order terms "
            "to the PS model, or switch to sp.cem (coarsened exact "
            "matching) or sp.ebalance (entropy balancing)."
        ),
    },

    # --- Synthetic control: poor pre-fit ---
    {
        'match': r"(pre.?treatment rmse.{0,20}(large|high)|synth.{0,20}no.?fit|pre.fit poor)",
        'category': 'synth_no_pretrend_fit',
        'diagnosis': (
            "Pre-treatment RMSE is large relative to outcome scale — "
            "donor pool does not fit the treated unit well.  ATT "
            "estimates are noisy and placebo inference is weak."
        ),
        'fix': (
            "Expand the donor pool, add more pre-treatment predictors, "
            "or switch to sp.synthdid (synthetic DiD) or sp.mscm "
            "(multiple synthetic controls) which share information "
            "across treated units."
        ),
    },

    # --- Over-identification test reject (Hansen J) ---
    {
        'match': r"(hansen.?j.{0,20}reject|over.?id.{0,15}reject|sargan.{0,15}reject)",
        'category': 'iv_exclusion_fail',
        'diagnosis': (
            "The over-identification test rejects — at least one "
            "instrument fails the exclusion restriction.  2SLS point "
            "estimate is biased even with strong first stage."
        ),
        'fix': (
            "Drop the weakest-theoretical instrument and re-estimate.  "
            "Or accept point-identification loss and use sp.bounds for "
            "partial identification.  For heterogeneous effects, "
            "sp.liml has smaller bias than 2SLS."
        ),
    },

    # --- Hausman test reject (FE vs RE) ---
    {
        'match': r"hausman.{0,15}reject",
        'category': 'hausman_reject',
        'diagnosis': (
            "Hausman test rejects random effects in favour of fixed "
            "effects.  RE estimates are inconsistent for this panel."
        ),
        'fix': (
            "Use fixed-effects estimation: sp.fixest(..., fe=['id','time']) "
            "or sp.panel(method='within').  If FE is infeasible "
            "(time-invariant treatment), consider Mundlak's device."
        ),
    },

    # --- Placebo test fails ---
    {
        'match': r"(placebo.{0,20}(significant|reject)|anticipation.{0,20}detect)",
        'category': 'placebo_fail',
        'diagnosis': (
            "A placebo test (pre-treatment 'effect') is significant — "
            "suggests anticipation, mis-timed treatment, or bad "
            "controls."
        ),
        'fix': (
            "Inspect event-study pre-period coefficients.  Consider "
            "sp.did_imputation (Borusyak-Jaravel-Spiess) which "
            "explicitly models anticipation, or exclude "
            "anticipating cohorts."
        ),
    },

    # --- Conformal / coverage failure ---
    {
        'match': r"(conformal.{0,20}(coverage|fail)|empirical coverage.{0,15}<)",
        'category': 'ci_coverage_fail',
        'diagnosis': (
            "Conformal prediction intervals under-cover the nominal "
            "1-alpha level on validation data."
        ),
        'fix': (
            "Increase the calibration-set share, use sp.jackknife_plus "
            "(Barber et al. 2021) instead of split-conformal, or "
            "switch to a more flexible nuisance learner."
        ),
    },

    # --- Sample / cohort too thin (expanded) ---
    {
        'match': r"(thin cohort|single treated|only one treatment|cohort size.{0,10}[=<]\s*1)",
        'category': 'small_cohort',
        'diagnosis': (
            "One or more cohorts has too few units to estimate "
            "group-time ATTs reliably.  Most staggered DID estimators "
            "need ≥ 10 treated units per cohort."
        ),
        'fix': (
            "Aggregate cohorts by treatment year range (e.g. 2010-2012 "
            "→ one cohort).  Or report only the simple pooled ATT with "
            "sp.did(method='2x2')."
        ),
    },

    # --- Design unknown ---
    {
        'match': r"(design.{0,20}(not inferr|unknown|ambiguous)|could not.{0,10}identif.{0,10}design)",
        'category': 'identification_unknown',
        'diagnosis': (
            "sp.recommend / sp.causal could not unambiguously infer "
            "the research design from the provided columns."
        ),
        'fix': (
            "Call sp.check_identification(design='...') explicitly — "
            "one of 'did', 'rd', 'iv', 'observational', 'panel', 'rct'. "
            "Or call sp.recommend(design='...') to constrain the search."
        ),
    },

    # ==================================================================
    # StatsPAIError taxonomy bridge — explicit exception classes raised
    # by estimators already carry recovery_hint; this section makes sure
    # agents see them even if they match via regex first.
    # ==================================================================
    {
        'match': r".*",
        'exception': ['IdentificationFailure'],
        'category': 'identification_failure',
        'diagnosis': (
            "The estimand is not identified under the chosen design — "
            "no re-tuning within this method will fix it."
        ),
        'fix': (
            "Inspect the exception's recovery_hint and "
            "alternative_functions.  Typically: change design (IV over "
            "observational), add structure (DAG / bounds), or narrow "
            "the estimand (ATT over ATE on overlap)."
        ),
    },
    {
        'match': r".*",
        'exception': ['AssumptionViolation'],
        'category': 'assumption_violation',
        'diagnosis': (
            "A statistical / identifying assumption is violated by "
            "the data (parallel trends, exclusion, overlap, SUTVA, …)."
        ),
        'fix': (
            "Read err.recovery_hint for the exact violated assumption "
            "and try err.alternative_functions[0].  Common switches: "
            "sp.callaway_santanna (parallel trends), sp.liml / "
            "sp.anderson_rubin_ci (weak/exogenous IV), sp.ebalance "
            "(overlap)."
        ),
    },
    {
        'match': r".*",
        'exception': ['ConvergenceFailure'],
        'category': 'convergence_failure',
        'diagnosis': (
            "An iterative algorithm (optimizer / EM / MCMC / cross-fit) "
            "did not reach tolerance."
        ),
        'fix': (
            "Inspect err.diagnostics for the failing metric.  Typical "
            "fixes: raise max_iter, loosen tol, reparameterise, or "
            "switch solver (L-BFGS-B ↔ trust-constr; NUTS ↔ ADVI)."
        ),
    },
    {
        'match': r".*",
        'exception': ['NumericalInstability'],
        'category': 'numerical_instability',
        'diagnosis': (
            "Computation hit a numerical corner case — singular design "
            "matrix, near-zero weights, NaN in variance."
        ),
        'fix': (
            "Run sp.vif / sp.estat to find collinearity.  Drop near-"
            "zero-variance covariates.  If weights explode, apply "
            "trimming via sp.trimming."
        ),
    },
    {
        'match': r".*",
        'exception': ['MethodIncompatibility'],
        'category': 'method_incompatibility',
        'diagnosis': (
            "The method is incompatible with the data shape / options "
            "you passed (e.g. HAC without time index)."
        ),
        'fix': (
            "Read err.recovery_hint for the offending option; common "
            "fixes are setting the correct fe= / time= argument, or "
            "switching method (panel → cross-section with sp.regress)."
        ),
    },
    {
        'match': r".*",
        'exception': ['DataInsufficient'],
        'category': 'data_insufficient',
        'diagnosis': (
            "Sample / cohorts / clusters are too small for the "
            "requested estimator (check err.diagnostics)."
        ),
        'fix': (
            "Aggregate cohorts / use simpler estimator / collect more "
            "data.  For inference, sp.wild_bootstrap or sp.cr2 handle "
            "few-cluster scenarios better than sandwich."
        ),
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _enrich_with_statspai_error(result: Dict[str, Any],
                                 error: BaseException) -> Dict[str, Any]:
    """If ``error`` is a :class:`StatsPAIError`, fold its ``recovery_hint``,
    ``diagnostics`` and ``alternative_functions`` into the remediation
    payload.  Hand-written hints on the exception win over registry
    boilerplate because they know the specific failing metric.
    """
    try:
        from ..exceptions import StatsPAIError
    except Exception:
        return result
    if not isinstance(error, StatsPAIError):
        return result

    hint = getattr(error, 'recovery_hint', '') or ''
    diag = getattr(error, 'diagnostics', {}) or {}
    alts = getattr(error, 'alternative_functions', []) or []

    if hint:
        # Prepend to the registry 'fix' so the specific hint shows first.
        result['fix'] = hint if not result.get('fix') else (
            f"{hint}  (registry hint: {result['fix']})"
        )
    if diag:
        result['diagnostics'] = {
            str(k): v for k, v in diag.items()
            if isinstance(v, (int, float, str, bool)) or v is None
        }
    if alts:
        result['alternative_functions'] = [str(a) for a in alts]
    return result


def remediate(error: BaseException,
              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a structured remediation suggestion for an exception.

    Parameters
    ----------
    error : BaseException
        The exception raised from a failed tool call.
    context : dict, optional
        Extra info to include in the response (e.g. which tool failed,
        what arguments were passed).  Forwarded verbatim.

    Returns
    -------
    dict with keys:
        'category'             : short label
        'diagnosis'            : human-readable explanation
        'fix'                  : concrete next action
        'matched'              : True if a registry entry matched
        'exception_type'       : class name of the raised error
        'exception_message'    : str(error) (truncated on the agent side)
        'recovery_hint'        : (StatsPAIError only) hint from the raiser
        'diagnostics'          : (StatsPAIError only) scalar-key map
        'alternative_functions': (StatsPAIError only) ranked sp.* names
    """
    err_type = type(error).__name__
    err_msg = str(error)
    # Match exception class AND all of its base classes so registry
    # entries keyed on 'AssumptionViolation' also fire for
    # 'IdentificationFailure' etc.
    err_mro = [c.__name__ for c in type(error).__mro__]

    for entry in REMEDIATIONS:
        exc_filter = entry.get('exception')
        if exc_filter:
            if not any(t in exc_filter for t in err_mro):
                continue
        pat = entry['match']
        if re.search(pat, err_msg, flags=re.IGNORECASE):
            result = {
                'category': entry['category'],
                'diagnosis': entry['diagnosis'],
                'fix': entry['fix'],
                'matched': True,
                'exception_type': err_type,
                'exception_message': err_msg,
            }
            if context:
                result['context'] = context
            return _enrich_with_statspai_error(result, error)

    # Generic fallback
    result = {
        'category': 'unknown',
        'diagnosis': (
            f"{err_type} raised; no registry match.  "
            f"This is likely a novel error pattern."
        ),
        'fix': (
            f"Read the full message: {err_msg[:300]}... "
            f"Check the tool's input_schema in sp.agent.tool_manifest() "
            f"and the target function's help string."
        ),
        'matched': False,
        'exception_type': err_type,
        'exception_message': err_msg,
        **({'context': context} if context else {}),
    }
    return _enrich_with_statspai_error(result, error)
