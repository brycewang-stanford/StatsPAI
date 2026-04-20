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
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
        'category'  : short label ('missing_column', 'weak_instrument', ...)
        'diagnosis' : human-readable explanation
        'fix'       : concrete next action
        'matched'   : True if a registry entry matched, False for generic
    """
    err_type = type(error).__name__
    err_msg = str(error)

    for entry in REMEDIATIONS:
        # Type filter
        if 'exception' in entry and err_type not in entry['exception']:
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
            return result

    # Generic fallback
    return {
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
