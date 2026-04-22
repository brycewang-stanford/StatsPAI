"""
Agent-native structured views of StatsPAI result objects.

This module hosts the logic behind ``result.to_agent_summary()`` and
``result.violations()`` for both :class:`EconometricResults` and
:class:`CausalResult`.  Kept separate from ``results.py`` to avoid
bloating that file and to let the per-method rules evolve
independently of the core data model.

Design principles
-----------------

* **Non-invasive.** Neither method alters the underlying result
  object.  Callers can invoke them any number of times.
* **Structured, not prose.**  ``to_agent_summary()`` returns a
  plain ``dict`` suitable for ``json.dumps`` or direct consumption
  by an LLM tool loop.  ``violations()`` returns a list of dicts.
* **Pattern-matching, not fitting.**  Violation detection only
  *inspects* diagnostics that the estimator already stored
  (``pretrend_test``, ``rhat``, ``first_stage_f``, …). It never
  re-runs a test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ====================================================================== #
#  Thresholds (literature-based rules of thumb)
# ====================================================================== #

#: Pre-trend test p-value below which we flag a DID parallel-trends
#: concern.  Deliberately *not* 0.05: agents should treat 0.10 as
#: "warrants follow-up" per Roth (2022) on low-power pre-trend tests.
_PRETREND_ALPHA = 0.10

#: Stock-Yogo 5% bias threshold (single endogenous regressor, 2SLS).
_WEAK_IV_F = 10.0

#: Gelman-Rubin / MCMC convergence thresholds (mirrors PyMC / arviz).
_RHAT_MAX = 1.01
_ESS_MIN = 400
_DIVERGENCES_MAX = 0

#: Maximum tolerable standardized mean difference after matching (SMD).
_SMD_MAX = 0.10

#: Propensity score overlap: treated weight share below this → bad
#: common support.
_OVERLAP_MIN = 0.05


# ====================================================================== #
#  CausalResult helpers
# ====================================================================== #


def _safe_get(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get ``obj[keys[0]][keys[1]]...`` or ``default`` if any step
    is missing / not a dict."""
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_float(x: Any) -> Optional[float]:
    """Coerce to float or return ``None`` on failure / NaN."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def causal_violations(result) -> List[Dict[str, Any]]:
    """Detect assumption / diagnostic violations on a ``CausalResult``.

    Each violation is a dict with keys ``kind`` / ``severity`` /
    ``test`` / ``value`` / ``threshold`` / ``message`` /
    ``recovery_hint`` / ``alternatives``.

    ``severity`` is one of ``"error"`` (identifying assumption
    clearly rejected), ``"warning"`` (borderline / low-power signal),
    or ``"info"`` (worth mentioning but unlikely to change the
    conclusion).
    """
    from .next_steps import _detect_family  # lazy to avoid cycle

    mi: Dict[str, Any] = result.model_info or {}
    method_family = _detect_family((result.method or "").lower())
    out: List[Dict[str, Any]] = []

    # --- DID: parallel trends ------------------------------------------
    pretrend_p = _as_float(_safe_get(mi, "pretrend_test", "pvalue"))
    if pretrend_p is not None and pretrend_p < _PRETREND_ALPHA:
        out.append({
            "kind": "assumption",
            "severity": "error" if pretrend_p < 0.05 else "warning",
            "test": "pretrend",
            "value": pretrend_p,
            "threshold": _PRETREND_ALPHA,
            "message": (
                f"Pre-trend joint test p = {pretrend_p:.3g} "
                f"< {_PRETREND_ALPHA} — parallel trends is likely violated."
            ),
            "recovery_hint": (
                "Run sp.sensitivity_rr(result) for Rambachan & Roth (2023) "
                "honest CIs, and consider sp.callaway_santanna or "
                "sp.did_imputation (robust to heterogeneous effects)."
            ),
            "alternatives": [
                "sp.sensitivity_rr",
                "sp.callaway_santanna",
                "sp.did_imputation",
            ],
        })

    # --- IV: weak instruments -------------------------------------------
    first_f = (
        _as_float(_safe_get(mi, "first_stage_f"))
        or _as_float(_safe_get(mi, "first_stage", "f_stat"))
        or _as_float(mi.get("weak_iv_f"))
    )
    if first_f is not None and first_f < _WEAK_IV_F and method_family == "iv":
        out.append({
            "kind": "assumption",
            "severity": "warning",
            "test": "weak_instrument",
            "value": first_f,
            "threshold": _WEAK_IV_F,
            "message": (
                f"First-stage F = {first_f:.2f} < {_WEAK_IV_F} (Stock-Yogo "
                "5% bias) — weak instrument bias is likely."
            ),
            "recovery_hint": (
                "Use sp.anderson_rubin_ci (weak-IV-robust) or "
                "sp.iv(..., method='liml') which has smaller weak-IV bias."
            ),
            "alternatives": ["sp.anderson_rubin_ci", "sp.iv"],
        })

    # --- Matching: covariate balance ------------------------------------
    smd_max = _as_float(_safe_get(mi, "balance", "max_smd_after"))
    if smd_max is not None and smd_max > _SMD_MAX and method_family == "matching":
        out.append({
            "kind": "assumption",
            "severity": "warning",
            "test": "balance",
            "value": smd_max,
            "threshold": _SMD_MAX,
            "message": (
                f"Max standardized mean difference after matching = "
                f"{smd_max:.3f} > {_SMD_MAX} — imbalance remains."
            ),
            "recovery_hint": (
                "Tighten caliper, add interactions, or try sp.entropy_balance."
            ),
            "alternatives": ["sp.entropy_balance", "sp.psmatch"],
        })

    # --- Matching / IPW: overlap ----------------------------------------
    overlap = _as_float(_safe_get(mi, "overlap", "min_share"))
    if overlap is not None and overlap < _OVERLAP_MIN:
        out.append({
            "kind": "assumption",
            "severity": "error",
            "test": "overlap",
            "value": overlap,
            "threshold": _OVERLAP_MIN,
            "message": (
                f"Propensity score overlap min share = {overlap:.3f} "
                f"< {_OVERLAP_MIN} — thin common support."
            ),
            "recovery_hint": (
                "Apply Crump (2009) trimming via sp.trimming or narrow "
                "the estimand to ATT on the overlap region."
            ),
            "alternatives": ["sp.trimming"],
        })

    # --- Bayesian: convergence ------------------------------------------
    rhat = _as_float(mi.get("rhat_max") or _safe_get(mi, "diagnostics", "rhat_max"))
    if rhat is not None and rhat > _RHAT_MAX:
        out.append({
            "kind": "convergence",
            "severity": "error",
            "test": "rhat",
            "value": rhat,
            "threshold": _RHAT_MAX,
            "message": (
                f"Max R-hat = {rhat:.3f} > {_RHAT_MAX} — MCMC has not mixed."
            ),
            "recovery_hint": (
                "Increase ``tune`` (≥ 4000), check for divergences, "
                "reparameterize (non-centered), or verify priors."
            ),
            "alternatives": [],
        })

    ess = _as_float(mi.get("ess_bulk_min") or _safe_get(mi, "diagnostics", "ess_bulk_min"))
    if ess is not None and ess < _ESS_MIN:
        out.append({
            "kind": "convergence",
            "severity": "warning",
            "test": "ess_bulk",
            "value": ess,
            "threshold": _ESS_MIN,
            "message": (
                f"Min bulk effective sample size = {ess:.0f} < {_ESS_MIN}."
            ),
            "recovery_hint": "Increase draws, or rerun with more chains.",
            "alternatives": [],
        })

    divs = mi.get("divergences") or _safe_get(mi, "diagnostics", "divergences")
    divs_val = _as_float(divs)
    if divs_val is not None and divs_val > _DIVERGENCES_MAX:
        out.append({
            "kind": "convergence",
            "severity": "error",
            "test": "divergences",
            "value": divs_val,
            "threshold": _DIVERGENCES_MAX,
            "message": (
                f"{int(divs_val)} post-warmup divergent transitions — "
                "posterior geometry is problematic."
            ),
            "recovery_hint": (
                "Raise ``target_accept`` to 0.95+ and/or reparameterize."
            ),
            "alternatives": [],
        })

    # --- RD: manipulation (McCrary) -------------------------------------
    mccrary_p = _as_float(_safe_get(mi, "mccrary", "pvalue"))
    if mccrary_p is not None and mccrary_p < 0.05 and method_family == "rd":
        out.append({
            "kind": "assumption",
            "severity": "error",
            "test": "mccrary_density",
            "value": mccrary_p,
            "threshold": 0.05,
            "message": (
                f"McCrary density test p = {mccrary_p:.3g} < 0.05 — "
                "running variable may be manipulated at the cutoff."
            ),
            "recovery_hint": (
                "Inspect manipulation mechanism; consider donut-RD "
                "(sp.rd_donut) or partial-identification bounds."
            ),
            "alternatives": ["sp.rd_donut", "sp.bounds"],
        })

    # --- NaN / degenerate estimate --------------------------------------
    est = _as_float(result.estimate)
    se = _as_float(result.se)
    if est is None:
        out.append({
            "kind": "numerical",
            "severity": "error",
            "test": "estimate_finite",
            "value": result.estimate,
            "threshold": None,
            "message": "Point estimate is NaN or ±inf.",
            "recovery_hint": "Check data for perfect collinearity / zero variance.",
            "alternatives": [],
        })
    if se is None or (se is not None and se <= 0):
        out.append({
            "kind": "numerical",
            "severity": "error",
            "test": "se_positive",
            "value": result.se,
            "threshold": 0,
            "message": "Standard error is non-positive / NaN.",
            "recovery_hint": "Check sandwich / cluster setup; inspect influence functions.",
            "alternatives": [],
        })

    return out


def causal_agent_summary(result) -> Dict[str, Any]:
    """Return a JSON-ready structured summary of a ``CausalResult``.

    Payload (all keys always present; empty containers when N/A):

    * ``method`` / ``method_family`` — estimator identity
    * ``estimand`` — ``"ATT"`` / ``"ATE"`` / ``"LATE"`` / etc.
    * ``point`` — dict with ``estimate`` / ``se`` / ``ci`` / ``pvalue``
      / ``alpha``
    * ``n_obs`` — sample size
    * ``diagnostics`` — the scalar-valued entries from
      ``model_info`` (DataFrames/arrays are replaced with a
      ``"<type>(shape)"`` placeholder so the output stays JSON-ready)
    * ``violations`` — output of :func:`causal_violations`
    * ``next_steps`` — output of ``result.next_steps(print_result=False)``
    * ``citation_key`` — key into :attr:`CausalResult._CITATIONS`
    """
    from .next_steps import _detect_family

    method = result.method or ""
    family = _detect_family(method.lower())

    est = _as_float(result.estimate)
    se = _as_float(result.se)
    pval = _as_float(result.pvalue)
    ci_lo = _as_float(result.ci[0]) if result.ci else None
    ci_hi = _as_float(result.ci[1]) if result.ci else None

    # Flatten scalar diagnostics so the payload stays JSON-safe.
    mi = result.model_info or {}
    scalar_diagnostics: Dict[str, Any] = {}
    for key, val in mi.items():
        if isinstance(val, (str, int, float, bool)) or val is None:
            scalar_diagnostics[key] = val
        elif isinstance(val, (pd.DataFrame, pd.Series, np.ndarray)):
            scalar_diagnostics[key] = f"<{type(val).__name__} shape={getattr(val, 'shape', '?')}>"
        elif isinstance(val, dict):
            # One level deep is enough for most diagnostic subtrees.
            nested = {}
            for k2, v2 in val.items():
                if isinstance(v2, (str, int, float, bool)) or v2 is None:
                    nested[k2] = v2
            if nested:
                scalar_diagnostics[key] = nested

    try:
        next_steps = result.next_steps(print_result=False)
    except Exception:  # pragma: no cover - defensive
        next_steps = []

    return {
        "kind": "causal_result",
        "method": method,
        "method_family": family,
        "estimand": result.estimand,
        "point": {
            "estimate": est,
            "se": se,
            "pvalue": pval,
            "ci": [ci_lo, ci_hi] if (ci_lo is not None and ci_hi is not None) else None,
            "alpha": _as_float(result.alpha),
        },
        "n_obs": int(result.n_obs) if result.n_obs is not None else None,
        "diagnostics": scalar_diagnostics,
        "violations": causal_violations(result),
        "next_steps": next_steps,
        "citation_key": getattr(result, "_citation_key", None),
    }


# ====================================================================== #
#  EconometricResults helpers
# ====================================================================== #


def econometric_violations(result) -> List[Dict[str, Any]]:
    """Detect common violations on an :class:`EconometricResults`."""
    out: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = getattr(result, "diagnostics", None) or {}
    mi: Dict[str, Any] = getattr(result, "model_info", None) or {}
    model_type = (mi.get("model_type", "") or "").lower()

    # IV weak-instrument check
    first_f = (
        _as_float(mi.get("first_stage_f"))
        or _as_float(_safe_get(mi, "first_stage", "f_stat"))
        or _as_float(diag.get("first_stage_f"))
    )
    is_iv = any(k in model_type for k in ("iv", "2sls", "liml", "gmm"))
    if is_iv and first_f is not None and first_f < _WEAK_IV_F:
        out.append({
            "kind": "assumption",
            "severity": "warning",
            "test": "weak_instrument",
            "value": first_f,
            "threshold": _WEAK_IV_F,
            "message": (
                f"First-stage F = {first_f:.2f} < {_WEAK_IV_F} — weak "
                "instrument bias likely."
            ),
            "recovery_hint": (
                "Use sp.anderson_rubin_ci or sp.iv(..., method='liml')."
            ),
            "alternatives": ["sp.anderson_rubin_ci", "sp.iv"],
        })

    # Non-positive SE
    ses = getattr(result, "std_errors", None)
    try:
        if ses is not None and (ses <= 0).any():
            bad = [str(k) for k in ses.index[ses <= 0]]
            out.append({
                "kind": "numerical",
                "severity": "error",
                "test": "se_positive",
                "value": bad,
                "threshold": 0,
                "message": f"Non-positive SE on: {bad}",
                "recovery_hint": (
                    "Inspect collinearity (sp.estat(result, 'vif')) and "
                    "sandwich / cluster setup."
                ),
                "alternatives": [],
            })
    except Exception:  # pragma: no cover - defensive
        pass

    return out


def _positional(arr: Any, i: int) -> Optional[float]:
    """Read element ``i`` from a Series or ndarray, best-effort."""
    if arr is None:
        return None
    # Series: prefer iloc so we don't depend on label alignment.
    if hasattr(arr, "iloc"):
        try:
            return _as_float(arr.iloc[i])
        except Exception:
            return None
    try:
        return _as_float(arr[i])
    except Exception:
        return None


def econometric_agent_summary(result) -> Dict[str, Any]:
    """JSON-ready structured summary of an :class:`EconometricResults`."""
    params = getattr(result, "params", None)
    ses = getattr(result, "std_errors", None)
    pvals = getattr(result, "pvalues", None)
    tvals = getattr(result, "tvalues", None)

    coefs: List[Dict[str, Any]] = []
    if params is not None and hasattr(params, "index"):
        for i, name in enumerate(params.index):
            coefs.append({
                "term": str(name),
                "estimate": _positional(params, i),
                "std_error": _positional(ses, i),
                "statistic": _positional(tvals, i),
                "p_value": _positional(pvals, i),
            })

    mi = getattr(result, "model_info", None) or {}
    data_info = getattr(result, "data_info", None) or {}
    diag = getattr(result, "diagnostics", None) or {}

    scalar_diagnostics = {
        k: v for k, v in diag.items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }

    try:
        next_steps = result.next_steps(print_result=False)
    except Exception:  # pragma: no cover - defensive
        next_steps = []

    return {
        "kind": "econometric_result",
        "model_type": mi.get("model_type", ""),
        "robust": mi.get("robust", "nonrobust"),
        "n_obs": int(data_info.get("nobs", 0)) if data_info.get("nobs") is not None else None,
        "df_resid": data_info.get("df_resid"),
        "dependent_var": data_info.get("dependent_var", ""),
        "coefficients": coefs,
        "diagnostics": scalar_diagnostics,
        "violations": econometric_violations(result),
        "next_steps": next_steps,
    }
