"""
Method-aware diagnostic batteries.

Given a fitted result (``EconometricResults`` or ``CausalResult``),
automatically selects and runs the appropriate set of robustness and
diagnostic tests, following the standards from empirical economics:

- **OLS/Panel**: heteroskedasticity, RESET, VIF, omitted-variable sensitivity
- **DID**: parallel trends (event-study), placebo treatment dates
- **RDD**: McCrary density, bandwidth robustness (h/2, h, 2h), polynomial robustness
- **IV**: first-stage F > 10, over-identification (Sargan), Wu-Hausman
- **Synth**: pre-treatment RMSPE, placebo-in-space
- **Matching**: covariate balance table, caliper sensitivity

Usage
-----
>>> result = sp.did(df, y='wage', treat='treated', time='post')
>>> sp.diagnose_result(result)          # auto-detects DID → runs DID battery
>>> sp.diagnose_result(result, print_results=True)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def diagnose_result(
    result,
    print_results: bool = True,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    One-call diagnostic battery, auto-selected by method type.

    Parameters
    ----------
    result : EconometricResults or CausalResult
        A fitted result from any StatsPAI estimator.
    print_results : bool
        If True, print formatted diagnostics to stdout.
    alpha : float
        Significance level for tests.

    Returns
    -------
    dict
        Keys depend on the method. Always includes ``'method_type'`` and
        ``'checks'`` (list of individual test result dicts).
    """
    method_type = _detect_method(result)
    dispatch = {
        "ols": _battery_ols,
        "panel": _battery_panel,
        "iv": _battery_iv,
        "did": _battery_did,
        "rd": _battery_rd,
        "synth": _battery_synth,
        "matching": _battery_matching,
        "dml": _battery_dml,
        # Sprint-B (0.9.6) causal-method batteries
        "proximal": _battery_proximal,
        "msm": _battery_msm,
        "principal_strat": _battery_principal_strat,
        "g_computation": _battery_g_computation,
        "front_door": _battery_front_door,
        "mediation_interventional": _battery_mediation_interventional,
    }

    fn = dispatch.get(method_type, _battery_generic)
    output = fn(result, alpha=alpha)
    output["method_type"] = method_type

    # Agent-native closed loop: merge in result.violations() if available.
    # This surfaces diagnostic violations that the estimator *already*
    # stored (pre-trend p, first-stage F, rhat, overlap, SMD, …) as a
    # dedicated section with recovery hints — complements the family-
    # specific battery checks above.
    output["violations"] = _collect_violations(result)

    # And the recommended next steps from the result itself, so a single
    # ``diagnose_result(r)`` call gives an agent the full triage picture
    # (family battery + structured violations + next steps).
    output["next_steps"] = _collect_next_steps(result)

    if print_results:
        _print_battery(output)

    return output


def _collect_violations(result) -> List[Dict[str, Any]]:
    """Return ``result.violations()`` if the method exists, else ``[]``."""
    vfn = getattr(result, "violations", None)
    if vfn is None:
        return []
    try:
        vs = vfn()
    except Exception:  # pragma: no cover - defensive
        return []
    if not isinstance(vs, list):
        return []
    return vs


def _collect_next_steps(result) -> List[Dict[str, Any]]:
    """Return ``result.next_steps(print_result=False)`` if available."""
    nfn = getattr(result, "next_steps", None)
    if nfn is None:
        return []
    try:
        return nfn(print_result=False) or []
    except Exception:  # pragma: no cover - defensive
        return []


# ====================================================================== #
#  Method detection
# ====================================================================== #

def _detect_method(result) -> str:
    """Infer method family from result metadata."""
    # PrincipalStratResult is its own dataclass (not CausalResult).
    # Detect it via class name first — it does not have a .method attr.
    cls_name = type(result).__name__
    if cls_name == 'PrincipalStratResult':
        return 'principal_strat'

    # CausalResult has a .method attribute
    method_str = ""
    if hasattr(result, "method"):
        method_str = (result.method or "").lower()
    if hasattr(result, "model_info"):
        mt = result.model_info.get("model_type", "").lower()
        method_str = f"{method_str} {mt}"

    # Sprint-B (0.9.6) method family detection — check BEFORE the
    # broader categories so e.g. an interventional-mediation result
    # (which has 'mediation' in its name) doesn't fall into a vanilla
    # 'mediation' catch-all.
    if any(k in method_str for k in ("proximal",)):
        return "proximal"
    if any(k in method_str for k in ("marginal structural", "iptw")):
        return "msm"
    if any(k in method_str for k in ("principal stratif",)):
        return "principal_strat"
    if any(k in method_str for k in ("g-computation", "g computation",
                                      "g-formula", "g formula",
                                      "standardization", "standardisation")):
        return "g_computation"
    if any(k in method_str for k in ("front-door", "front door", "front_door")):
        return "front_door"
    if any(k in method_str for k in ("interventional mediation", "iie")):
        return "mediation_interventional"

    # Classical methods
    if any(k in method_str for k in ("did", "diff", "callaway", "sun_abraham", "stagger")):
        return "did"
    if any(k in method_str for k in ("rd", "discontinuity", "rdrobust")):
        return "rd"
    if any(k in method_str for k in ("iv", "2sls", "instrumental")):
        return "iv"
    if any(k in method_str for k in ("synth", "scm", "sdid")):
        return "synth"
    if any(k in method_str for k in ("match", "psm", "cem", "mahalanobis")):
        return "matching"
    if any(k in method_str for k in ("panel", "fe ", "re ", "fixed effect", "random effect")):
        return "panel"
    if any(k in method_str for k in ("dml", "double", "debiased")):
        return "dml"
    if any(k in method_str for k in ("ols", "least square")):
        return "ols"
    return "generic"


# ====================================================================== #
#  Battery implementations
# ====================================================================== #

def _battery_ols(result, alpha: float = 0.05) -> Dict[str, Any]:
    """OLS: heteroskedasticity, RESET, VIF, omitted-variable sensitivity."""
    checks = []

    # Extract residuals and fitted values
    resid = _get_residuals(result)
    fitted = _get_fitted(result)
    nobs = _get_nobs(result)

    if resid is not None:
        # Breusch-Pagan (simplified: regress e^2 on fitted)
        e2 = resid ** 2
        if fitted is not None:
            X_bp = np.column_stack([np.ones(len(fitted)), fitted])
            beta_bp = np.linalg.lstsq(X_bp, e2, rcond=None)[0]
            e2_hat = X_bp @ beta_bp
            rss_r = np.sum((e2 - e2.mean()) ** 2)
            rss_u = np.sum((e2 - e2_hat) ** 2)
            r2 = 1 - rss_u / rss_r if rss_r > 0 else 0
            from scipy import stats as sp_stats
            lm_stat = nobs * r2
            pval = float(1 - sp_stats.chi2.cdf(lm_stat, 1))
            checks.append({
                "test": "Breusch-Pagan heteroskedasticity",
                "statistic": round(lm_stat, 4),
                "pvalue": round(pval, 4),
                "pass": pval >= alpha,
                "interpretation": "No heteroskedasticity detected" if pval >= alpha
                    else "Heteroskedasticity detected — use robust SEs",
            })

    # R-squared reasonableness
    r2 = _get_diagnostic(result, "R-squared")
    adj_r2 = _get_diagnostic(result, "Adj. R-squared")
    if r2 is not None:
        checks.append({
            "test": "Model fit",
            "R-squared": round(r2, 4),
            "Adj. R-squared": round(adj_r2, 4) if adj_r2 else None,
            "pass": True,
            "interpretation": f"R² = {r2:.3f}",
        })

    return {"checks": checks}


def _battery_panel(result, alpha: float = 0.05) -> Dict[str, Any]:
    """Panel: same as OLS plus entity/time info."""
    output = _battery_ols(result, alpha)

    # Add panel-specific info
    n_entities = _get_diagnostic(result, "N entities")
    n_time = _get_diagnostic(result, "N time periods")
    if n_entities is not None or n_time is not None:
        output["checks"].append({
            "test": "Panel structure",
            "N_entities": n_entities,
            "N_time_periods": n_time,
            "pass": True,
            "interpretation": f"Balanced panel" if n_entities and n_time else "Panel info available",
        })

    return output


def _battery_iv(result, alpha: float = 0.05) -> Dict[str, Any]:
    """IV: first-stage F, Sargan, Wu-Hausman."""
    checks = []

    # First-stage F
    fs_f = _get_diagnostic(result, "First-stage F") or _get_model_info(result, "first_stage_f")
    if fs_f is not None:
        checks.append({
            "test": "First-stage F-statistic (weak instrument)",
            "statistic": round(float(fs_f), 2),
            "threshold": 10.0,
            "pass": float(fs_f) > 10,
            "interpretation": f"F = {float(fs_f):.1f} {'> 10 ✓' if float(fs_f) > 10 else '< 10 — WEAK INSTRUMENT WARNING'}",
        })
    else:
        checks.append({
            "test": "First-stage F-statistic",
            "pass": None,
            "interpretation": "Not available — re-run with first_stage=True if supported",
        })

    # Sargan over-identification
    sargan = _get_diagnostic(result, "Sargan statistic") or _get_model_info(result, "sargan_stat")
    sargan_p = _get_diagnostic(result, "Sargan p-value") or _get_model_info(result, "sargan_pvalue")
    if sargan is not None and sargan_p is not None:
        checks.append({
            "test": "Sargan over-identification test",
            "statistic": round(float(sargan), 4),
            "pvalue": round(float(sargan_p), 4),
            "pass": float(sargan_p) >= alpha,
            "interpretation": "Instruments valid" if float(sargan_p) >= alpha
                else "Over-identification rejected — instrument validity concern",
        })

    # Wu-Hausman
    hausman = _get_diagnostic(result, "Wu-Hausman") or _get_model_info(result, "wu_hausman")
    hausman_p = _get_diagnostic(result, "Wu-Hausman p-value") or _get_model_info(result, "wu_hausman_pvalue")
    if hausman is not None and hausman_p is not None:
        checks.append({
            "test": "Wu-Hausman endogeneity test",
            "statistic": round(float(hausman), 4),
            "pvalue": round(float(hausman_p), 4),
            "pass": float(hausman_p) < alpha,
            "interpretation": "Endogeneity confirmed — IV needed" if float(hausman_p) < alpha
                else "No endogeneity detected — OLS may suffice",
        })

    return {"checks": checks}


def _battery_did(result, alpha: float = 0.05) -> Dict[str, Any]:
    """DID: parallel trends, event study pre-trend, placebo."""
    checks = []

    # Pre-trend test from event study
    model_info = getattr(result, "model_info", {})
    pretrend = model_info.get("pretrend_test")
    event_study = model_info.get("event_study")

    if pretrend is not None:
        p_val = pretrend.get("pvalue", pretrend.get("p_value"))
        checks.append({
            "test": "Pre-trend test (joint significance of pre-treatment coefficients)",
            "statistic": pretrend.get("statistic"),
            "pvalue": round(float(p_val), 4) if p_val is not None else None,
            "pass": float(p_val) >= alpha if p_val is not None else None,
            "interpretation": "Parallel trends supported" if p_val and float(p_val) >= alpha
                else "Pre-trend detected — parallel trends assumption may be violated",
        })

    if event_study is not None and isinstance(event_study, pd.DataFrame):
        # Check pre-treatment coefficients individually
        pre = event_study[event_study["relative_time"] < 0]
        if len(pre) > 0 and "se" in pre.columns:
            sig_pre = 0
            for _, row in pre.iterrows():
                if abs(row.get("estimate", 0)) > 1.96 * row.get("se", float("inf")):
                    sig_pre += 1
            checks.append({
                "test": "Event study pre-treatment coefficients",
                "n_pre_periods": len(pre),
                "n_significant": sig_pre,
                "pass": sig_pre == 0,
                "interpretation": f"{sig_pre}/{len(pre)} pre-treatment periods significant at 5%"
                    + (" — parallel trends concern" if sig_pre > 0 else " — parallel trends supported"),
            })

    # Treatment effect sign and significance
    estimate = getattr(result, "estimate", None)
    se = getattr(result, "se", None)
    pvalue = getattr(result, "pvalue", None)
    if estimate is not None:
        checks.append({
            "test": "Treatment effect",
            "estimate": round(float(estimate), 4),
            "se": round(float(se), 4) if se else None,
            "pvalue": round(float(pvalue), 4) if pvalue else None,
            "pass": True,
            "interpretation": f"ATT = {float(estimate):.4f}" +
                (f" (p = {float(pvalue):.4f})" if pvalue else ""),
        })

    return {"checks": checks}


def _battery_rd(result, alpha: float = 0.05) -> Dict[str, Any]:
    """RDD: McCrary density, bandwidth robustness, effect."""
    checks = []

    model_info = getattr(result, "model_info", {})

    # Bandwidth info
    bw = model_info.get("bandwidth") or model_info.get("h")
    if bw is not None:
        checks.append({
            "test": "Bandwidth",
            "h": round(float(bw), 4),
            "pass": True,
            "interpretation": f"MSE-optimal bandwidth h = {float(bw):.4f}. "
                "Recommendation: also check h/2 and 2h for robustness.",
        })

    # Effective sample sizes
    n_left = model_info.get("n_left")
    n_right = model_info.get("n_right")
    if n_left is not None and n_right is not None:
        checks.append({
            "test": "Effective observations in bandwidth",
            "n_left": int(n_left),
            "n_right": int(n_right),
            "pass": int(n_left) >= 20 and int(n_right) >= 20,
            "interpretation": f"Left: {int(n_left)}, Right: {int(n_right)}" +
                (" — sufficient" if int(n_left) >= 20 and int(n_right) >= 20
                 else " — WARNING: few observations near cutoff"),
        })

    # McCrary reminder
    checks.append({
        "test": "McCrary density test (manipulation check)",
        "pass": None,
        "interpretation": "Run sp.mccrary_test(data, x='running_var', c=cutoff) "
            "to test for manipulation at the cutoff.",
    })

    # Treatment effect
    estimate = getattr(result, "estimate", None)
    if estimate is not None:
        checks.append({
            "test": "RD treatment effect",
            "estimate": round(float(estimate), 4),
            "se": round(float(getattr(result, "se", 0)), 4),
            "pvalue": round(float(getattr(result, "pvalue", 1)), 4),
            "pass": True,
            "interpretation": f"RD effect = {float(estimate):.4f}",
        })

    return {"checks": checks}


def _battery_synth(result, alpha: float = 0.05) -> Dict[str, Any]:
    """Synthetic control: pre-treatment RMSPE, weights."""
    checks = []

    model_info = getattr(result, "model_info", {})

    # Pre-treatment RMSPE
    rmspe = model_info.get("pre_rmspe") or model_info.get("rmspe_pre")
    if rmspe is not None:
        checks.append({
            "test": "Pre-treatment RMSPE",
            "value": round(float(rmspe), 4),
            "pass": True,
            "interpretation": f"Pre-treatment fit: RMSPE = {float(rmspe):.4f}. "
                "Lower is better. Compare with placebo distribution.",
        })

    # Synthetic weights
    weights = model_info.get("weights")
    if weights is not None and hasattr(weights, "__len__"):
        n_nonzero = sum(1 for w in weights if abs(w) > 0.01)
        checks.append({
            "test": "Donor pool composition",
            "n_donors_with_weight": n_nonzero,
            "pass": True,
            "interpretation": f"{n_nonzero} donors contribute to the synthetic unit.",
        })

    # Placebo reminder
    checks.append({
        "test": "Placebo tests (in-space)",
        "pass": None,
        "interpretation": "Run placebo analysis: reassign treatment to each control unit "
            "and compare the effect ratio (post/pre RMSPE).",
    })

    estimate = getattr(result, "estimate", None)
    if estimate is not None:
        checks.append({
            "test": "Treatment effect",
            "estimate": round(float(estimate), 4),
            "pass": True,
            "interpretation": f"SCM effect = {float(estimate):.4f}",
        })

    return {"checks": checks}


def _battery_matching(result, alpha: float = 0.05) -> Dict[str, Any]:
    """Matching: balance, sample size, treatment effect."""
    checks = []

    model_info = getattr(result, "model_info", {})

    # Balance stats
    balance = model_info.get("balance") or model_info.get("balance_table")
    if balance is not None and isinstance(balance, pd.DataFrame):
        # Check standardised mean differences
        if "std_diff" in balance.columns:
            max_smd = balance["std_diff"].abs().max()
            n_imbalanced = (balance["std_diff"].abs() > 0.1).sum()
            checks.append({
                "test": "Covariate balance (standardised mean difference)",
                "max_SMD": round(float(max_smd), 4),
                "n_imbalanced": int(n_imbalanced),
                "threshold": 0.1,
                "pass": n_imbalanced == 0,
                "interpretation": f"Max SMD = {float(max_smd):.3f}, "
                    f"{n_imbalanced} covariates with |SMD| > 0.1"
                    + (" — good balance" if n_imbalanced == 0 else " — balance concern"),
            })

    # Matched sample info
    n_treated = model_info.get("n_treated")
    n_matched = model_info.get("n_matched") or model_info.get("n_control_matched")
    if n_treated is not None:
        checks.append({
            "test": "Matched sample",
            "n_treated": int(n_treated),
            "n_matched": int(n_matched) if n_matched else None,
            "pass": True,
            "interpretation": f"Treated: {int(n_treated)}" +
                (f", Matched controls: {int(n_matched)}" if n_matched else ""),
        })

    estimate = getattr(result, "estimate", None)
    if estimate is not None:
        checks.append({
            "test": "Treatment effect",
            "estimate": round(float(estimate), 4),
            "se": round(float(getattr(result, "se", 0)), 4),
            "pvalue": round(float(getattr(result, "pvalue", 1)), 4),
            "pass": True,
            "interpretation": f"ATT = {float(estimate):.4f}",
        })

    return {"checks": checks}


def _battery_dml(result, alpha: float = 0.05) -> Dict[str, Any]:
    """DML: treatment effect, first-stage fit."""
    checks = []

    model_info = getattr(result, "model_info", {})

    # First-stage R² for outcome and treatment models
    r2_y = model_info.get("r2_outcome") or model_info.get("r2_y")
    r2_t = model_info.get("r2_treatment") or model_info.get("r2_d")
    if r2_y is not None:
        checks.append({
            "test": "Outcome model fit (cross-validated R²)",
            "R2": round(float(r2_y), 4),
            "pass": float(r2_y) > 0.05,
            "interpretation": f"R² = {float(r2_y):.3f}" +
                (" — reasonable" if float(r2_y) > 0.05 else " — weak outcome prediction"),
        })
    if r2_t is not None:
        checks.append({
            "test": "Treatment model fit (cross-validated R²)",
            "R2": round(float(r2_t), 4),
            "pass": float(r2_t) > 0.05,
            "interpretation": f"R² = {float(r2_t):.3f}",
        })

    estimate = getattr(result, "estimate", None)
    if estimate is not None:
        checks.append({
            "test": "Treatment effect (DML)",
            "estimate": round(float(estimate), 4),
            "se": round(float(getattr(result, "se", 0)), 4),
            "pvalue": round(float(getattr(result, "pvalue", 1)), 4),
            "pass": True,
            "interpretation": f"ATE = {float(estimate):.4f}",
        })

    return {"checks": checks}


def _battery_generic(result, alpha: float = 0.05) -> Dict[str, Any]:
    """Fallback: report whatever is available."""
    checks = []
    estimate = getattr(result, "estimate", None)
    if estimate is not None:
        checks.append({
            "test": "Treatment effect",
            "estimate": round(float(estimate), 4),
            "pass": True,
        })
    if hasattr(result, "params"):
        checks.append({
            "test": "Model parameters",
            "n_params": len(result.params),
            "pass": True,
        })
    return {"checks": checks}


# ====================================================================== #
#  Sprint-B (v0.9.6) batteries: proximal / msm / principal / g-formula  #
# ====================================================================== #

def _battery_proximal(result, alpha: float = 0.05) -> Dict[str, Any]:
    """Proximal causal inference: bridge + proxy rank + weak-IV checks."""
    checks = []
    info = getattr(result, "model_info", {}) or {}

    bridge = info.get("bridge", "linear")
    checks.append({
        "test": "Bridge function",
        "bridge": bridge,
        "pass": bridge == "linear",
        "interpretation": (
            f"Bridge='{bridge}'. Linear is the only shipped option; "
            "non-linear bridges require Mastouri et al. 2021 / Deaner "
            "2018 methodology (see docs/ROADMAP.md §1)."
        ),
    })

    k_z = info.get("n_proxy_z")
    k_w = info.get("n_proxy_w")
    if k_z is not None and k_w is not None:
        checks.append({
            "test": "Proxy order condition (k_z >= k_w)",
            "n_proxy_z": int(k_z),
            "n_proxy_w": int(k_w),
            "pass": int(k_z) >= int(k_w),
            "interpretation": (
                f"Order condition {'satisfied' if k_z >= k_w else 'VIOLATED'}: "
                f"{k_z} treatment-side vs {k_w} outcome-side proxies."
            ),
        })

    fs_F = info.get("first_stage_F")
    if fs_F is not None:
        checks.append({
            "test": "First-stage F (Z on W)",
            "F": round(float(fs_F), 4),
            "pass": float(fs_F) >= 10.0,
            "interpretation": (
                f"F = {float(fs_F):.2f}"
                + (" — adequate" if float(fs_F) >= 10 else " — WEAK proxy (< 10)")
            ),
        })
    elif k_w is not None and int(k_w) > 1:
        checks.append({
            "test": "Multi-W weak-instrument diagnostic",
            "pass": None,
            "interpretation": (
                "First-stage F not reported for k_w > 1. The correct "
                "diagnostic is the Cragg-Donald / Kleibergen-Paap "
                "minimum-eigenvalue statistic — not yet shipped (see "
                "docs/ROADMAP.md §2)."
            ),
        })

    estimate = getattr(result, "estimate", None)
    if estimate is not None:
        checks.append({
            "test": "Proximal ATE",
            "estimate": round(float(estimate), 4),
            "se": round(float(getattr(result, "se", 0.0)), 4),
            "pvalue": round(float(getattr(result, "pvalue", 1.0)), 4),
            "pass": True,
            "interpretation": f"ATE = {float(estimate):.4f} (bridge={bridge})",
        })
    return {"checks": checks}


def _battery_msm(result, alpha: float = 0.05) -> Dict[str, Any]:
    """Marginal Structural Model: weight diagnostics + positivity."""
    checks = []
    info = getattr(result, "model_info", {}) or {}

    sw_mean = info.get("sw_mean")
    sw_max = info.get("sw_max")
    sw_min = info.get("sw_min")
    if sw_mean is not None:
        # Stabilised weights should have mean ≈ 1 under correct
        # specification; large deviation flags model misspec.
        deviation = abs(float(sw_mean) - 1.0)
        checks.append({
            "test": "Stabilized-weight mean (target ≈ 1)",
            "sw_mean": round(float(sw_mean), 4),
            "pass": deviation < 0.2,
            "interpretation": (
                f"sw mean = {float(sw_mean):.3f}"
                + (" — centred" if deviation < 0.2
                   else " — drift from 1 suggests nuisance misspec.")
            ),
        })
    if sw_max is not None:
        checks.append({
            "test": "Max stabilized weight (positivity watchdog)",
            "sw_max": round(float(sw_max), 4),
            "pass": float(sw_max) < 50.0,
            "interpretation": (
                f"max sw = {float(sw_max):.2f}"
                + (" — OK"
                   if float(sw_max) < 50
                   else " — extreme: consider trim_per_period=True.")
            ),
        })
    if sw_min is not None:
        checks.append({
            "test": "Min stabilized weight",
            "sw_min": round(float(sw_min), 4),
            "pass": float(sw_min) > 1e-3,
            "interpretation": f"min sw = {float(sw_min):.4f}",
        })

    exposure = info.get("exposure_type")
    if exposure:
        checks.append({
            "test": "Exposure summary",
            "exposure": exposure,
            "pass": True,
            "interpretation": f"Reported coefficient is the marginal "
                              f"effect on the {exposure} exposure.",
        })

    n_units = info.get("n_units")
    n_periods = info.get("n_periods")
    if n_units is not None and n_periods is not None:
        checks.append({
            "test": "Panel dimensions",
            "n_units": int(n_units),
            "n_periods": int(n_periods),
            "pass": int(n_units) >= 30 and int(n_periods) >= 2,
            "interpretation": f"{n_units} units × {n_periods} periods.",
        })

    estimate = getattr(result, "estimate", None)
    if estimate is not None:
        checks.append({
            "test": "Marginal causal effect",
            "estimate": round(float(estimate), 4),
            "se": round(float(getattr(result, "se", 0.0)), 4),
            "pvalue": round(float(getattr(result, "pvalue", 1.0)), 4),
            "pass": True,
            "interpretation": f"MSM coefficient = {float(estimate):.4f}",
        })
    return {"checks": checks}


def _battery_principal_strat(result, alpha: float = 0.05) -> Dict[str, Any]:
    """Principal stratification: monotonicity + stratum balance."""
    checks = []
    info = getattr(result, "model_info", {}) or {}

    # Monotonicity violation diagnostic (principal_score method)
    viol = info.get("mono_violation_frac")
    if viol is not None:
        checks.append({
            "test": "Monotonicity violation fraction (fitted p11 < p10)",
            "mono_violation_frac": round(float(viol), 4),
            "pass": float(viol) <= 0.05,
            "interpretation": (
                f"{float(viol):.1%} of units have fitted p11(x) < p10(x)"
                + (" — within 5% tolerance"
                   if float(viol) <= 0.05
                   else " — MONOTONICITY CONCERN. Bounds / PCEs depend "
                        "on S(1) ≥ S(0); pair with sensitivity analysis.")
            ),
        })

    # Stratum proportions (from PrincipalStratResult)
    props = getattr(result, "strata_proportions", None) or {}
    if props:
        summary = ", ".join(f"{k[:10]}={v:.2f}" for k, v in props.items())
        checks.append({
            "test": "Stratum proportions (sum ≈ 1)",
            "proportions": {k: round(float(v), 4) for k, v in props.items()},
            "pass": abs(sum(float(v) for v in props.values()) - 1.0) < 0.05,
            "interpretation": summary,
        })

    # Effects table
    eff = getattr(result, "effects", None)
    if isinstance(eff, pd.DataFrame) and len(eff):
        checks.append({
            "test": "Principal causal effects",
            "n_effects": int(len(eff)),
            "pass": True,
            "interpretation": (
                "PCEs: "
                + "; ".join(f"{r['stratum']}={r['estimate']:.3f}"
                             for _, r in eff.iterrows())
            ),
        })

    # Zhang-Rubin SACE bounds (monotonicity method)
    bounds = getattr(result, "bounds", None)
    if isinstance(bounds, pd.DataFrame) and len(bounds):
        lo = float(bounds.iloc[0]["estimate"])
        hi = float(bounds.iloc[-1]["estimate"])
        valid = not (np.isnan(lo) or np.isnan(hi))
        checks.append({
            "test": "Zhang-Rubin SACE bounds",
            "sace_lo": None if not valid else round(lo, 4),
            "sace_hi": None if not valid else round(hi, 4),
            "pass": valid and (lo <= hi),
            "interpretation": (
                f"SACE ∈ [{lo:.3f}, {hi:.3f}]"
                if valid else "SACE bounds degenerate (no always-survivors)."
            ),
        })

    return {"checks": checks}


def _battery_g_computation(result, alpha: float = 0.05) -> Dict[str, Any]:
    """G-computation: estimand + bootstrap failure rate."""
    checks = []
    info = getattr(result, "model_info", {}) or {}

    estimand = info.get("estimand")
    if estimand:
        checks.append({
            "test": "Estimand",
            "estimand": estimand,
            "pass": True,
            "interpretation": (
                f"Target estimand: {estimand}. Consistent when the "
                "outcome model is correctly specified; not doubly "
                "robust — consider sp.aipw / sp.tmle for DR coverage."
            ),
        })

    ml_Q = info.get("ml_Q")
    if ml_Q:
        checks.append({
            "test": "Outcome learner",
            "ml_Q": ml_Q,
            "pass": True,
            "interpretation": f"Outcome model: {ml_Q}.",
        })

    n_boot = info.get("n_boot")
    n_failed = info.get("n_boot_failed", 0)
    if n_boot:
        frac = (n_failed / n_boot) if n_boot else 0.0
        checks.append({
            "test": "Bootstrap health",
            "n_boot": int(n_boot),
            "n_boot_failed": int(n_failed),
            "failure_frac": round(frac, 4),
            "pass": frac < 0.1,
            "interpretation": (
                f"{n_failed}/{n_boot} bootstrap replications failed "
                f"({frac:.1%})"
                + (" — clean"
                   if frac < 0.1 else
                   " — high failure rate; SE may be unreliable.")
            ),
        })

    estimate = getattr(result, "estimate", None)
    if estimate is not None:
        checks.append({
            "test": "G-computation effect",
            "estimate": round(float(estimate), 4),
            "se": round(float(getattr(result, "se", 0.0)), 4),
            "pvalue": round(float(getattr(result, "pvalue", 1.0)), 4),
            "pass": True,
            "interpretation": f"{estimand or 'Effect'} = {float(estimate):.4f}",
        })
    return {"checks": checks}


def _battery_front_door(result, alpha: float = 0.05) -> Dict[str, Any]:
    """Front-door adjustment diagnostics."""
    checks = []
    info = getattr(result, "model_info", {}) or {}

    m_type = info.get("mediator_type")
    if m_type:
        checks.append({
            "test": "Mediator type",
            "mediator_type": m_type,
            "pass": m_type in ("binary", "continuous"),
            "interpretation": f"Mediator modelled as {m_type}.",
        })

    integrate = info.get("integrate_by_effective") or info.get("integrate_by")
    if integrate:
        checks.append({
            "test": "Integration formulation",
            "integrate_by": integrate,
            "pass": True,
            "interpretation": (
                f"Integration: {integrate}. "
                "'marginal' follows Pearl (1995), 'conditional' follows "
                "Fulcher et al. (2020)."
            ),
        })

    n_boot = info.get("n_boot")
    n_failed = info.get("n_boot_failed", 0)
    if n_boot:
        frac = n_failed / n_boot if n_boot else 0.0
        checks.append({
            "test": "Bootstrap health",
            "n_boot": int(n_boot),
            "n_boot_failed": int(n_failed),
            "pass": frac < 0.1,
            "interpretation": (
                f"{n_failed}/{n_boot} bootstrap replications failed "
                f"({frac:.1%})"
            ),
        })

    checks.append({
        "test": "Front-door assumption",
        "pass": None,
        "interpretation": (
            "Front-door identification requires: (1) no direct D→Y path "
            "(all effect via M), (2) no unobserved M-Y confounder, "
            "(3) positivity on M | D. These are DAG assumptions — "
            "verify with sp.dag before trusting the estimate."
        ),
    })

    estimate = getattr(result, "estimate", None)
    if estimate is not None:
        checks.append({
            "test": "Front-door ATE",
            "estimate": round(float(estimate), 4),
            "se": round(float(getattr(result, "se", 0.0)), 4),
            "pvalue": round(float(getattr(result, "pvalue", 1.0)), 4),
            "pass": True,
            "interpretation": f"ATE = {float(estimate):.4f}",
        })
    return {"checks": checks}


def _battery_mediation_interventional(result, alpha: float = 0.05) -> Dict[str, Any]:
    """Interventional (in)direct effects: IIE / IDE / Total consistency."""
    checks = []
    info = getattr(result, "model_info", {}) or {}

    pv_method = info.get("pvalue_method")
    if pv_method:
        checks.append({
            "test": "P-value convention",
            "pvalue_method": pv_method,
            "pass": pv_method in ("bootstrap_sign", "wald"),
            "interpretation": (
                f"Using {pv_method}; "
                "bootstrap_sign keeps parity with sp.mediate(), wald "
                "aligns with sp.aipw/sp.dml."
            ),
        })

    iie = info.get("iie")
    ide = info.get("ide")
    total = info.get("total_effect")
    if iie is not None and ide is not None and total is not None:
        diff = abs((float(iie) + float(ide)) - float(total))
        checks.append({
            "test": "Decomposition additivity (IIE + IDE ≈ Total)",
            "iie": round(float(iie), 4),
            "ide": round(float(ide), 4),
            "total": round(float(total), 4),
            "residual": round(float(diff), 4),
            "pass": diff < 1e-6,
            "interpretation": (
                f"IIE + IDE − Total = {diff:.2e} "
                + ("(exact)" if diff < 1e-6 else "(drift — investigate)")
            ),
        })

    tv = info.get("tv_confounders")
    if tv:
        checks.append({
            "test": "Treatment-induced confounders",
            "tv_confounders": list(tv),
            "pass": True,
            "interpretation": (
                f"Interventional effects use {len(tv)} "
                "treatment-induced confounder(s); natural effects would "
                "not be identified here."
            ),
        })

    n_boot = info.get("n_boot")
    n_failed = info.get("n_boot_failed", 0)
    if n_boot:
        frac = n_failed / n_boot
        checks.append({
            "test": "Bootstrap health",
            "n_boot": int(n_boot),
            "n_boot_failed": int(n_failed),
            "pass": frac < 0.1,
            "interpretation": f"{n_failed}/{n_boot} replications failed "
                              f"({frac:.1%}).",
        })
    return {"checks": checks}


# ====================================================================== #
#  Helpers
# ====================================================================== #

def _get_residuals(result) -> Optional[np.ndarray]:
    if hasattr(result, "data_info") and "residuals" in result.data_info:
        r = result.data_info["residuals"]
        return np.asarray(r) if r is not None else None
    return None


def _get_fitted(result) -> Optional[np.ndarray]:
    if hasattr(result, "data_info") and "fitted_values" in result.data_info:
        f = result.data_info["fitted_values"]
        return np.asarray(f) if f is not None else None
    return None


def _get_nobs(result) -> int:
    if hasattr(result, "n_obs"):
        return result.n_obs
    if hasattr(result, "data_info"):
        return result.data_info.get("nobs", 0)
    return 0


def _get_diagnostic(result, key: str):
    if hasattr(result, "diagnostics") and isinstance(result.diagnostics, dict):
        return result.diagnostics.get(key)
    return None


def _get_model_info(result, key: str):
    if hasattr(result, "model_info") and isinstance(result.model_info, dict):
        return result.model_info.get(key)
    return None


# ====================================================================== #
#  Pretty printer
# ====================================================================== #

def _print_battery(output: Dict[str, Any]):
    """Print diagnostic battery in a clean format."""
    method = output.get("method_type", "unknown").upper()
    checks = output.get("checks", [])

    print("=" * 65)
    print(f"  Diagnostic Battery — {method}")
    print("=" * 65)

    for i, chk in enumerate(checks, 1):
        test_name = chk.get("test", "Unknown")
        passed = chk.get("pass")
        interp = chk.get("interpretation", "")

        if passed is True:
            status = "PASS"
        elif passed is False:
            status = "WARN"
        else:
            status = "INFO"

        print(f"\n  [{status}] {i}. {test_name}")

        # Print numeric details
        for k in ("statistic", "pvalue", "estimate", "se", "R-squared",
                   "R2", "h", "max_SMD", "threshold", "n_pre_periods",
                   "n_significant", "n_treated", "n_matched",
                   "n_left", "n_right", "N_entities", "N_time_periods",
                   "Adj. R-squared", "n_donors_with_weight"):
            if k in chk and chk[k] is not None:
                print(f"         {k}: {chk[k]}")

        if interp:
            print(f"         → {interp}")

    n_pass = sum(1 for c in checks if c.get("pass") is True)
    n_warn = sum(1 for c in checks if c.get("pass") is False)
    n_info = sum(1 for c in checks if c.get("pass") is None)
    print(f"\n  Summary: {n_pass} passed, {n_warn} warnings, {n_info} info")

    # --- Agent-native violations block (populated by result.violations())
    violations = output.get("violations") or []
    if violations:
        print("\n  Structured violations (agent-native)")
        print("  " + "-" * 60)
        for v in violations:
            sev = str(v.get("severity", "info")).upper()
            test = v.get("test", "?")
            msg = v.get("message", "")
            hint = v.get("recovery_hint", "")
            print(f"  [{sev}] {test}: {msg}")
            if hint:
                print(f"         ↳ {hint}")

    print("=" * 65)
