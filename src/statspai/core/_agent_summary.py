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

#: Maximum tolerable standardized mean difference after matching (SMD). 0.10 is
#: the conventional "well balanced" target (Rosenbaum-Rubin); a match leaving
#: more than this on a covariate is not ideal but often acceptable.
_SMD_MAX = 0.10

#: Standardized mean difference above which residual imbalance is a *concern*
#: worth flagging loudly (Stuart 2010; What Works Clearinghouse). Set above
#: _SMD_MAX so the warning fires on genuinely poor matches, not borderline ones.
_SMD_IMBALANCE_MAX = 0.25

#: Propensity score overlap: treated weight share below this → bad
#: common support.
_OVERLAP_MIN = 0.05

#: DML / AIPW propensity overlap: share of units whose cross-fitted propensity
#: sits at or beyond the trimming bound (near 0 or 1). Above this the IRM /
#: AIPW estimate leans on a few near-degenerate-weight units and is unstable.
#: 0.05 fires on genuinely poor overlap (strong confounding) while clearing
#: moderate confounding, which trims essentially nothing.
_DML_OVERLAP_EXTREME_SHARE = 0.05

#: Absolute logit/probit slope coefficient above which (quasi-)complete
#: separation is the likely cause: an odds ratio of e^15 ≈ 3.3M is not a real
#: effect, it is the MLE diverging when a predictor perfectly splits the outcome
#: (Albert-Anderson 1984; Heinze-Schemper 2002). Real coefficients are O(1-5).
_LOGIT_SEPARATION_COEF = 15.0

#: Pearson dispersion (χ²/df) above which a Poisson fit is over-dispersed —
#: its variance exceeds its mean, so model-based SEs are too small. 1.5 is a
#: conservative bar (equidispersion is 1.0) that clears clean Poisson data.
_POISSON_DISPERSION_MAX = 1.5

#: Minimum number of clusters for cluster-robust SE to be reliable. Below this,
#: the CRVE is downward-biased and t-tests over-reject; the wild cluster
#: bootstrap (Cameron-Gelbach-Miller 2008; MacKinnon-Webb 2017) is the standard
#: remedy. 30 is the conservative end of the common 30-50 rule of thumb.
_FEW_CLUSTERS_MIN = 30

#: Synthetic-control pre-fit quality: pre-treatment RMSPE divided by the
#: pre-period SD of the treated outcome. Deliberately conservative (0.6 ⇒ the
#: synthetic unit explains < ~64% of pre-period variance) so it clears the
#: canonical *good* example — California Prop-99 sits at ~0.42 — and only fires
#: on genuinely poor fits (an unmatchable treated trend lands ~2.7). A poor
#: pre-fit means the synthetic control does not track the treated unit before
#: treatment, so the post-period gap cannot be read as a treatment effect.
_SYNTH_PREFIT_RATIO_MAX = 0.6


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


def _max_covariate_smd(mi: Dict[str, Any]) -> Optional[float]:
    """Largest absolute post-matching standardized mean difference across
    covariates. Handles both the scalar convention
    (``mi["balance"]["max_smd_after"]``) and the per-variable balance table
    that ``sp.match`` stores (a DataFrame with ``variable`` / ``smd`` columns),
    excluding the propensity score / distance rows which are not covariates."""
    scalar = _as_float(_safe_get(mi, "balance", "max_smd_after"))
    if scalar is not None:
        return scalar
    bal = mi.get("balance")
    if isinstance(bal, pd.DataFrame) and {"variable", "smd"}.issubset(bal.columns):
        cov = bal[~bal["variable"].isin(["propensity_score", "distance"])]
        if not cov.empty:
            m = cov["smd"].abs().max()
            return _as_float(m)
    return None


def causal_violations(result: Any) -> List[Dict[str, Any]]:
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
    # Detect the family from the result's ``method`` attribute, falling back to
    # ``model_info`` keys (``model_type`` / ``method``). Several estimators
    # (e.g. IV) leave ``result.method`` unset but record an identifying
    # ``model_type`` such as ``"IV-2SLS"`` — without this fallback their
    # family-gated checks (weak IV, RD manipulation, …) silently never fire.
    method_family = _detect_family((getattr(result, "method", None) or "").lower())
    if method_family == "generic":
        for _key in ("model_type", "method"):
            _fam = _detect_family(str(mi.get(_key, "")).lower())
            if _fam != "generic":
                method_family = _fam
                break
    out: List[Dict[str, Any]] = []

    # --- DID: parallel trends ------------------------------------------
    pretrend_p = _as_float(_safe_get(mi, "pretrend_test", "pvalue"))
    if pretrend_p is not None and pretrend_p < _PRETREND_ALPHA:
        out.append(
            {
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
            }
        )

    # --- IV: weak instruments -------------------------------------------
    first_f = (
        _as_float(_safe_get(mi, "first_stage_f"))
        or _as_float(_safe_get(mi, "first_stage", "f_stat"))
        or _as_float(mi.get("weak_iv_f"))
    )
    if first_f is not None and first_f < _WEAK_IV_F and method_family == "iv":
        out.append(
            {
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
            }
        )

    # --- Cluster-robust inference: too few clusters ---------------------
    # Gated on the estimator having recorded ``n_clusters`` (i.e. cluster-robust
    # SEs were actually requested) rather than on family, since the small-G
    # problem applies wherever a CRVE is used.
    n_clusters = _as_float(mi.get("n_clusters"))
    if n_clusters is not None and n_clusters < _FEW_CLUSTERS_MIN:
        out.append(
            {
                "kind": "inference",
                "severity": "warning",
                "test": "few_clusters",
                "value": int(n_clusters),
                "threshold": _FEW_CLUSTERS_MIN,
                "message": (
                    f"Only {int(n_clusters)} clusters (< {_FEW_CLUSTERS_MIN}) — "
                    "cluster-robust SEs are downward-biased and t-tests "
                    "over-reject (Cameron-Gelbach-Miller 2008)."
                ),
                "recovery_hint": (
                    "Report sp.wild_cluster_bootstrap (or sp.wild_cluster_ci_inv "
                    "for confidence intervals), which has correct size with few "
                    "clusters."
                ),
                "alternatives": [
                    "sp.wild_cluster_bootstrap",
                    "sp.wild_cluster_ci_inv",
                ],
            }
        )

    # --- Synthetic control: poor pre-treatment fit ----------------------
    # Gated on synth-specific keys rather than family detection. The ratio is
    # scale-free (pre-RMSPE / pre-period SD of the treated outcome); see
    # _SYNTH_PREFIT_RATIO_MAX for the calibration.
    pre_rmspe = _as_float(mi.get("pre_treatment_rmse"))
    if (
        pre_rmspe is not None
        and "n_donors" in mi
        and mi.get("Y_treated") is not None
        and mi.get("times") is not None
        and mi.get("treatment_time") is not None
    ):
        try:
            y_treated = np.asarray(mi["Y_treated"], dtype=float)
            times = np.asarray(mi["times"])
            pre = y_treated[times < mi["treatment_time"]]
            pre_sd = float(np.std(pre, ddof=1)) if pre.size >= 2 else 0.0
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pre_sd = 0.0
        ratio = pre_rmspe / pre_sd if pre_sd > 0 else None
        if ratio is not None and ratio > _SYNTH_PREFIT_RATIO_MAX:
            out.append(
                {
                    "kind": "assumption",
                    "severity": "warning",
                    "test": "synth_prefit",
                    "value": ratio,
                    "threshold": _SYNTH_PREFIT_RATIO_MAX,
                    "message": (
                        f"Pre-treatment fit is poor: RMSPE / pre-period SD = "
                        f"{ratio:.2f} > {_SYNTH_PREFIT_RATIO_MAX}. The synthetic "
                        "control does not track the treated unit before "
                        "treatment, so the post-period gap is unreliable."
                    ),
                    "recovery_hint": (
                        "Improve the donor pool / predictors, compare estimators "
                        "with sp.synth_compare, try sp.augsynth (ridge-augmented), "
                        "and gauge robustness with sp.synth_sensitivity."
                    ),
                    "alternatives": [
                        "sp.synth_compare",
                        "sp.augsynth",
                        "sp.synth_sensitivity",
                    ],
                }
            )

    # --- Matching: covariate balance ------------------------------------
    smd_max = _max_covariate_smd(mi)
    if (
        smd_max is not None
        and smd_max > _SMD_IMBALANCE_MAX
        and method_family == "matching"
    ):
        out.append(
            {
                "kind": "assumption",
                "severity": "warning",
                "test": "balance",
                "value": smd_max,
                "threshold": _SMD_IMBALANCE_MAX,
                "message": (
                    f"Max standardized mean difference after matching = "
                    f"{smd_max:.3f} > {_SMD_IMBALANCE_MAX} — notable residual "
                    "imbalance, so the matched comparison is still confounded."
                ),
                "recovery_hint": (
                    "Tighten the caliper, add interactions/polynomials to the "
                    "propensity model, or reweight with sp.ebalance "
                    "(entropy balancing) / sp.cbps, and re-check sp.love_plot."
                ),
                "alternatives": ["sp.ebalance", "sp.cbps", "sp.love_plot"],
            }
        )

    # --- Matching / IPW: overlap ----------------------------------------
    overlap = _as_float(_safe_get(mi, "overlap", "min_share"))
    if overlap is not None and overlap < _OVERLAP_MIN:
        out.append(
            {
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
            }
        )

    # --- DML / AIPW: propensity overlap ---------------------------------
    # Gated on the IRM/AIPW-specific keys: the cross-fitted propensity scores
    # and the trimming bound the estimator already stored. A large share of
    # units at/beyond that bound means near-degenerate inverse-propensity
    # weights, so the estimate rests on a handful of influential observations.
    _pscore = mi.get("_pscore")
    if _pscore is not None and mi.get("trimming_threshold") is not None:
        try:
            psc = np.asarray(_pscore, dtype=float)
            psc = psc[np.isfinite(psc)]
        except (TypeError, ValueError):  # pragma: no cover - defensive
            psc = np.empty(0)
        trim = _as_float(mi.get("trimming_threshold")) or 0.01
        if psc.size:
            extreme = float(np.mean((psc < trim) | (psc > 1.0 - trim)))
            if extreme > _DML_OVERLAP_EXTREME_SHARE:
                out.append(
                    {
                        "kind": "assumption",
                        "severity": "warning",
                        "test": "dml_overlap",
                        "value": extreme,
                        "threshold": _DML_OVERLAP_EXTREME_SHARE,
                        "message": (
                            f"{extreme:.1%} of units have a propensity score at "
                            f"or beyond the trimming bound ({trim:g}) — weak "
                            "overlap, so the IRM/AIPW estimate leans on a few "
                            "near-degenerate-weight units."
                        ),
                        "recovery_hint": (
                            "Narrow the estimand to the overlap region with "
                            "sp.trimming (Crump 2009), reweight with "
                            "sp.overlap_weights (Li et al. 2018), or improve the "
                            "propensity model (sp.cbps)."
                        ),
                        "alternatives": [
                            "sp.trimming",
                            "sp.overlap_weights",
                            "sp.cbps",
                        ],
                    }
                )

    # --- Bayesian: convergence ------------------------------------------
    rhat = _as_float(mi.get("rhat_max") or _safe_get(mi, "diagnostics", "rhat_max"))
    if rhat is not None and rhat > _RHAT_MAX:
        out.append(
            {
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
            }
        )

    ess = _as_float(
        mi.get("ess_bulk_min") or _safe_get(mi, "diagnostics", "ess_bulk_min")
    )
    if ess is not None and ess < _ESS_MIN:
        out.append(
            {
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
            }
        )

    divs = mi.get("divergences") or _safe_get(mi, "diagnostics", "divergences")
    divs_val = _as_float(divs)
    if divs_val is not None and divs_val > _DIVERGENCES_MAX:
        out.append(
            {
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
            }
        )

    # --- RD: manipulation (McCrary) -------------------------------------
    mccrary_p = _as_float(_safe_get(mi, "mccrary", "pvalue"))
    if mccrary_p is not None and mccrary_p < 0.05 and method_family == "rd":
        out.append(
            {
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
            }
        )

    # --- NaN / degenerate estimate --------------------------------------
    est = _as_float(result.estimate)
    se = _as_float(result.se)
    if est is None:
        out.append(
            {
                "kind": "numerical",
                "severity": "error",
                "test": "estimate_finite",
                "value": result.estimate,
                "threshold": None,
                "message": "Point estimate is NaN or ±inf.",
                "recovery_hint": "Check data for perfect collinearity / zero variance.",
                "alternatives": [],
            }
        )
    if se is None or (se is not None and se <= 0):
        out.append(
            {
                "kind": "numerical",
                "severity": "error",
                "test": "se_positive",
                "value": result.se,
                "threshold": 0,
                "message": "Standard error is non-positive / NaN.",
                "recovery_hint": "Check sandwich / cluster setup; inspect influence functions.",
                "alternatives": [],
            }
        )

    return out


def causal_agent_summary(result: Any) -> Dict[str, Any]:
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
            scalar_diagnostics[
                key
            ] = f"<{type(val).__name__} shape={getattr(val, 'shape', '?')}>"
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


def econometric_violations(result: Any) -> List[Dict[str, Any]]:
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
        out.append(
            {
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
            }
        )

    # Too few clusters for reliable cluster-robust inference. Gated on the
    # estimator having recorded ``n_clusters`` (cluster-robust SEs were
    # requested), mirroring the fit-time warning so the two never disagree.
    n_clusters = _as_float(mi.get("n_clusters"))
    if n_clusters is not None and n_clusters < _FEW_CLUSTERS_MIN:
        out.append(
            {
                "kind": "inference",
                "severity": "warning",
                "test": "few_clusters",
                "value": int(n_clusters),
                "threshold": _FEW_CLUSTERS_MIN,
                "message": (
                    f"Only {int(n_clusters)} clusters (< {_FEW_CLUSTERS_MIN}) — "
                    "cluster-robust SEs are downward-biased and t-tests "
                    "over-reject (Cameron-Gelbach-Miller 2008)."
                ),
                "recovery_hint": (
                    "Report sp.wild_cluster_bootstrap (or sp.wild_cluster_ci_inv "
                    "for confidence intervals), correct with few clusters."
                ),
                "alternatives": [
                    "sp.wild_cluster_bootstrap",
                    "sp.wild_cluster_ci_inv",
                ],
            }
        )

    # Logit / probit (quasi-)complete separation. A slope coefficient this
    # large is the MLE diverging because a predictor perfectly splits the
    # outcome, not a real effect — the point estimate and its SE are unusable.
    family = (mi.get("family", "") or "").lower()
    is_binary_glm = (
        "logit" in model_type or "probit" in model_type or family == "binomial"
    )
    if is_binary_glm:
        params = getattr(result, "params", None)
        if params is not None:
            try:
                slopes = params.drop(
                    [
                        i
                        for i in params.index
                        if str(i).lower() in ("intercept", "const")
                    ],
                    errors="ignore",
                )
                max_abs = float(np.max(np.abs(slopes.values))) if len(slopes) else 0.0
            except (AttributeError, TypeError, ValueError, KeyError):
                max_abs = 0.0
            if max_abs > _LOGIT_SEPARATION_COEF:
                out.append(
                    {
                        "kind": "numerical",
                        "severity": "error",
                        "test": "separation",
                        "value": max_abs,
                        "threshold": _LOGIT_SEPARATION_COEF,
                        "message": (
                            f"A coefficient of {max_abs:.1f} indicates "
                            "(quasi-)complete separation — a predictor perfectly "
                            "splits the outcome and the MLE has diverged, so the "
                            "estimate and its SE are meaningless."
                        ),
                        "recovery_hint": (
                            "Use penalised logistic regression (Firth), drop or "
                            "combine the separating predictor, or report an exact "
                            "/ profile-likelihood interval."
                        ),
                        "alternatives": ["sp.logit", "sp.rlasso"],
                    }
                )

    # Poisson over-dispersion. Compute the Pearson dispersion from the stored
    # fitted values (not a re-fit) — variance far above the mean means Poisson
    # SEs are too small and a negative-binomial / robust fit is warranted.
    if "poisson" in model_type or family == "poisson":
        di = getattr(result, "data_info", None) or {}
        mu = di.get("fitted_values")
        yv = di.get("y")
        df_resid = _as_float(di.get("df_resid"))
        if mu is not None and yv is not None and df_resid and df_resid > 0:
            try:
                mu_a = np.clip(np.asarray(mu, dtype=float), 1e-10, None)
                y_a = np.asarray(yv, dtype=float)
                dispersion = float(np.sum((y_a - mu_a) ** 2 / mu_a) / df_resid)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                dispersion = None
            if dispersion is not None and dispersion > _POISSON_DISPERSION_MAX:
                out.append(
                    {
                        "kind": "assumption",
                        "severity": "warning",
                        "test": "overdispersion",
                        "value": dispersion,
                        "threshold": _POISSON_DISPERSION_MAX,
                        "message": (
                            f"Pearson dispersion = {dispersion:.2f} > "
                            f"{_POISSON_DISPERSION_MAX} — the variance exceeds the "
                            "mean, so Poisson standard errors are too small and "
                            "t-tests over-reject."
                        ),
                        "recovery_hint": (
                            "Refit with sp.nbreg (negative binomial) for "
                            "over-dispersion, or keep Poisson point estimates "
                            "with robust='hc1' SEs (quasi-Poisson)."
                        ),
                        "alternatives": ["sp.nbreg", "sp.zinb", "sp.poisson"],
                    }
                )

    # Non-positive SE
    ses = getattr(result, "std_errors", None)
    try:
        if ses is not None and (ses <= 0).any():
            bad = [str(k) for k in ses.index[ses <= 0]]
            out.append(
                {
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
                }
            )
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


def econometric_agent_summary(result: Any) -> Dict[str, Any]:
    """JSON-ready structured summary of an :class:`EconometricResults`."""
    params = getattr(result, "params", None)
    ses = getattr(result, "std_errors", None)
    pvals = getattr(result, "pvalues", None)
    tvals = getattr(result, "tvalues", None)

    coefs: List[Dict[str, Any]] = []
    if params is not None and hasattr(params, "index"):
        for i, name in enumerate(params.index):
            coefs.append(
                {
                    "term": str(name),
                    "estimate": _positional(params, i),
                    "std_error": _positional(ses, i),
                    "statistic": _positional(tvals, i),
                    "p_value": _positional(pvals, i),
                }
            )

    mi = getattr(result, "model_info", None) or {}
    data_info = getattr(result, "data_info", None) or {}
    diag = getattr(result, "diagnostics", None) or {}

    scalar_diagnostics = {
        k: v
        for k, v in diag.items()
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
        "n_obs": (
            int(data_info.get("nobs", 0)) if data_info.get("nobs") is not None else None
        ),
        "df_resid": data_info.get("df_resid"),
        "dependent_var": data_info.get("dependent_var", ""),
        "coefficients": coefs,
        "diagnostics": scalar_diagnostics,
        "violations": econometric_violations(result),
        "next_steps": next_steps,
    }
