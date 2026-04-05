"""
Agent-native workflow guidance for StatsPAI.

Provides method-aware ``next_steps()`` recommendations so that both
human researchers and AI agents know exactly what to do after fitting
a model — diagnostics, robustness checks, alternative specifications,
and export options.

Each recommendation is a dict with keys:

- ``action`` : str — the Python code to run
- ``reason`` : str — why this step matters
- ``priority`` : str — "essential", "recommended", or "optional"
- ``category`` : str — "diagnostics", "robustness", "export", "alternative", "sensitivity"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ====================================================================== #
#  Step data structure
# ====================================================================== #

class Step:
    """One recommended next step."""

    __slots__ = ("action", "reason", "priority", "category")

    def __init__(self, action: str, reason: str,
                 priority: str = "recommended",
                 category: str = "diagnostics"):
        self.action = action
        self.reason = reason
        self.priority = priority
        self.category = category

    def __repr__(self):
        icon = {"essential": "❗", "recommended": "→",
                "optional": "○"}.get(self.priority, "→")
        return f"{icon} {self.action}\n   {self.reason}"

    def to_dict(self) -> Dict[str, str]:
        return {
            "action": self.action,
            "reason": self.reason,
            "priority": self.priority,
            "category": self.category,
        }


# ====================================================================== #
#  Formatters
# ====================================================================== #

def _format_steps(steps: List[Step], title: str = "Suggested Next Steps") -> str:
    """Pretty-print a list of steps."""
    lines = [
        "━" * 65,
        f"  {title}",
        "━" * 65,
    ]

    by_priority = {"essential": [], "recommended": [], "optional": []}
    for s in steps:
        by_priority.get(s.priority, by_priority["recommended"]).append(s)

    idx = 1
    for pri, label in [("essential", "Essential"),
                       ("recommended", "Recommended"),
                       ("optional", "Optional")]:
        group = by_priority[pri]
        if not group:
            continue
        lines.append(f"\n  [{label}]")
        for s in group:
            lines.append(f"  {idx}. {s.action}")
            lines.append(f"     {s.reason}")
            idx += 1

    lines.append("━" * 65)
    return "\n".join(lines)


def _steps_repr_html(steps: List[Step], title: str = "Suggested Next Steps") -> str:
    """HTML rendering for Jupyter."""
    css = (
        '<style scoped>'
        '.sp-ns{font-family:"Helvetica Neue",Arial,sans-serif;max-width:700px;'
        'border:1px solid #E5E7EB;border-radius:8px;overflow:hidden;margin:8px 0}'
        '.sp-ns-hdr{background:linear-gradient(135deg,#0f766e 0%,#115e59 100%);'
        'color:#fff;padding:10px 16px;font-size:14px;font-weight:600}'
        '.sp-ns-body{padding:8px 0}'
        '.sp-ns-cat{padding:4px 16px;font-size:11px;font-weight:700;'
        'color:#64748B;text-transform:uppercase;letter-spacing:0.5px;'
        'border-bottom:1px solid #F1F5F9;margin-top:4px}'
        '.sp-ns-item{padding:6px 16px 6px 24px;font-size:12px;'
        'border-bottom:1px solid #F8FAFC}'
        '.sp-ns-item:hover{background:#F0FDFA}'
        '.sp-ns-code{font-family:"SF Mono",Menlo,monospace;font-size:12px;'
        'color:#0f766e;font-weight:600}'
        '.sp-ns-why{color:#64748B;font-size:11px;margin-top:2px}'
        '.sp-ns-badge{display:inline-block;font-size:9px;padding:1px 6px;'
        'border-radius:3px;font-weight:600;margin-left:6px}'
        '.sp-ns-ess{background:#FEE2E2;color:#991B1B}'
        '.sp-ns-rec{background:#DBEAFE;color:#1E40AF}'
        '.sp-ns-opt{background:#F3F4F6;color:#6B7280}'
        '</style>'
    )

    badge_cls = {"essential": "sp-ns-ess", "recommended": "sp-ns-rec",
                 "optional": "sp-ns-opt"}
    badge_txt = {"essential": "essential", "recommended": "recommended",
                 "optional": "optional"}

    h = [css, '<div class="sp-ns">',
         f'<div class="sp-ns-hdr">{title}</div>',
         '<div class="sp-ns-body">']

    by_cat: Dict[str, List[Step]] = {}
    for s in steps:
        by_cat.setdefault(s.category, []).append(s)

    cat_labels = {
        "diagnostics": "📋 Diagnostics",
        "robustness": "🔍 Robustness",
        "sensitivity": "🎯 Sensitivity",
        "export": "📄 Export",
        "alternative": "🔄 Alternatives",
        "workflow": "⚡ Workflow",
    }

    for cat, items in by_cat.items():
        label = cat_labels.get(cat, cat.title())
        h.append(f'<div class="sp-ns-cat">{label}</div>')
        for s in items:
            bcls = badge_cls.get(s.priority, "sp-ns-rec")
            btxt = badge_txt.get(s.priority, s.priority)
            h.append('<div class="sp-ns-item">')
            h.append(f'<div><span class="sp-ns-code">{s.action}</span>'
                     f'<span class="sp-ns-badge {bcls}">{btxt}</span></div>')
            h.append(f'<div class="sp-ns-why">{s.reason}</div>')
            h.append('</div>')

    h.append('</div></div>')
    return '\n'.join(h)


# ====================================================================== #
#  EconometricResults next_steps
# ====================================================================== #

def econometric_next_steps(result) -> List[Step]:
    """Generate next steps for an EconometricResults object."""
    steps: List[Step] = []
    model_type = (result.model_info.get("model_type", "") or "").lower()
    robust = result.model_info.get("robust", "nonrobust")
    has_residuals = result.data_info.get("residuals") is not None
    has_X = result.data_info.get("X") is not None
    n_obs = result.data_info.get("nobs", 0)
    dep_var = result.data_info.get("dependent_var", "y")

    # ── Diagnostics ────────────────────────────────────────────────── #
    if has_residuals and has_X:
        steps.append(Step(
            "sp.estat(result, 'all')",
            "Run all post-estimation diagnostics (heteroskedasticity, "
            "normality, VIF, RESET, serial correlation)",
            priority="essential",
            category="diagnostics",
        ))
    else:
        steps.append(Step(
            "sp.estat(result, 'ic')",
            "Compare information criteria (AIC, BIC) across specifications",
            priority="recommended",
            category="diagnostics",
        ))

    # Robust SEs suggestion
    if robust == "nonrobust":
        steps.append(Step(
            f"sp.regress('{dep_var} ~ ...', data=df, robust='hc1')",
            "Re-estimate with heteroskedasticity-robust standard errors",
            priority="recommended",
            category="robustness",
        ))

    # ── Post-estimation ────────────────────────────────────────────── #
    steps.append(Step(
        "sp.margins(result, data=df)",
        "Compute average marginal effects for substantive interpretation",
        priority="recommended",
        category="diagnostics",
    ))

    # ── Robustness ─────────────────────────────────────────────────── #
    steps.append(Step(
        "sp.oster_delta(data=df, y='...', x_base=[...], x_controls=[...])",
        "Oster (2019) coefficient stability — assess sensitivity to unobservables",
        priority="recommended",
        category="sensitivity",
    ))

    # Variable selection check
    n_params = len(result.params)
    if n_params >= 5:
        steps.append(Step(
            "sp.stepwise(data=df, y='...', x=[...], criterion='bic')",
            f"Variable selection: {n_params} regressors — "
            "check if BIC prefers a more parsimonious model",
            priority="optional",
            category="robustness",
        ))

    # ── Export ─────────────────────────────────────────────────────── #
    steps.append(Step(
        "sp.esttab(result1, result2, result3)",
        "Create publication-quality comparison table (text/LaTeX/Word/Excel)",
        priority="recommended",
        category="export",
    ))
    steps.append(Step(
        "sp.outreg2(result, filename='table.xlsx')",
        "Export formatted results to Excel",
        priority="optional",
        category="export",
    ))

    # ── IV-specific ────────────────────────────────────────────────── #
    if "iv" in model_type or "2sls" in model_type:
        steps.insert(0, Step(
            "sp.estat(result, 'firststage')",
            "Check first-stage F-statistic (Stock-Yogo threshold: F > 10)",
            priority="essential",
            category="diagnostics",
        ))
        steps.insert(1, Step(
            "sp.estat(result, 'overid')",
            "Sargan/Hansen J test for over-identification (instrument validity)",
            priority="essential",
            category="diagnostics",
        ))
        steps.insert(2, Step(
            "sp.estat(result, 'endogenous')",
            "Durbin-Wu-Hausman test: is IV actually needed vs OLS?",
            priority="essential",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.kitagawa_test(data=df, y='...', treatment='...', instrument='...')",
            "Test LATE validity (Kitagawa 2015 — necessary conditions for LATE)",
            priority="recommended",
            category="sensitivity",
        ))

    # ── Panel-specific ─────────────────────────────────────────────── #
    if any(k in model_type for k in ("panel", "fe", "fixed")):
        steps.append(Step(
            "sp.hausman_test(result_fe, result_re)",
            "Hausman test: fixed effects vs random effects",
            priority="recommended",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.twoway_cluster(result, data=df, cluster1='...', cluster2='...')",
            "Two-way clustered SEs (e.g., firm × year)",
            priority="optional",
            category="robustness",
        ))

    return steps


# ====================================================================== #
#  CausalResult next_steps
# ====================================================================== #

def causal_next_steps(result) -> List[Step]:
    """Generate next steps for a CausalResult object."""
    steps: List[Step] = []
    method = (result.method or "").lower()
    model_info = result.model_info or {}
    has_event_study = (
        model_info.get("event_study") is not None
        or (result.detail is not None and "relative_time" in
            (result.detail.columns if result.detail is not None else []))
    )

    method_family = _detect_family(method)

    # ── DID-family ─────────────────────────────────────────────────── #
    if method_family == "did":
        if has_event_study:
            steps.append(Step(
                "sp.pretrends_test(result)",
                "Joint test of pre-treatment coefficients (parallel trends)",
                priority="essential",
                category="diagnostics",
            ))
            steps.append(Step(
                "sp.pretrends_power(result)",
                "Roth (2022) — is the pre-trend test powerful enough "
                "to detect relevant violations?",
                priority="essential",
                category="diagnostics",
            ))
            steps.append(Step(
                "sp.sensitivity_rr(result)",
                "Rambachan & Roth (2023) — honest CI under bounded "
                "parallel trends violations",
                priority="essential",
                category="sensitivity",
            ))
        else:
            steps.append(Step(
                "sp.event_study(data=df, ...)",
                "Estimate event study to visualize pre-trends and dynamic effects",
                priority="essential",
                category="diagnostics",
            ))

        steps.append(Step(
            "result.plot()",
            "Plot event study / treatment effects",
            priority="recommended",
            category="diagnostics",
        ))

        # Suggest alternative DID estimators
        if "2x2" in method or "twfe" in method.replace(" ", ""):
            steps.append(Step(
                "sp.bacon_decomposition(data=df, ...)",
                "Goodman-Bacon decomposition: check for negative weights in TWFE",
                priority="recommended",
                category="diagnostics",
            ))
            steps.append(Step(
                "sp.did_imputation(data=df, ...)",
                "Try BJS (2024) imputation estimator — robust to heterogeneous effects",
                priority="recommended",
                category="alternative",
            ))
            steps.append(Step(
                "sp.callaway_santanna(data=df, ...)",
                "Try Callaway-Sant'Anna — group-time ATT with flexible aggregation",
                priority="recommended",
                category="alternative",
            ))

        # Multiple outcomes
        steps.append(Step(
            "sp.romano_wolf(data=df, y=[...], x=[...], n_boot=1000)",
            "If testing multiple outcomes: Romano-Wolf FWER-adjusted p-values",
            priority="optional",
            category="robustness",
        ))

    # ── RD-family ──────────────────────────────────────────────────── #
    elif method_family == "rd":
        steps.append(Step(
            "sp.mccrary_test(data=df, x='...', c=cutoff)",
            "McCrary density test: check for manipulation at the cutoff",
            priority="essential",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.rd_honest(data=df, y='...', x='...', c=cutoff)",
            "Armstrong-Kolesár honest CI — valid uniformly over smooth functions",
            priority="essential",
            category="sensitivity",
        ))
        steps.append(Step(
            "sp.rdbalance(data=df, covariates=[...], x='...', c=cutoff)",
            "Covariate balance test at the cutoff",
            priority="essential",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.rdbwsensitivity(data=df, y='...', x='...', c=cutoff)",
            "Bandwidth sensitivity: check stability at h/2, h, 2h",
            priority="recommended",
            category="robustness",
        ))
        steps.append(Step(
            "sp.rdplacebo(data=df, y='...', x='...', c=cutoff)",
            "Placebo cutoff test: should find no effect at false cutoffs",
            priority="recommended",
            category="robustness",
        ))
        steps.append(Step(
            "result.plot()",
            "RD plot with confidence intervals",
            priority="recommended",
            category="diagnostics",
        ))

    # ── IV-family ──────────────────────────────────────────────────── #
    elif method_family == "iv":
        steps.append(Step(
            "sp.estat(result, 'firststage')",
            "First-stage F: rule of thumb F > 10 (Stock-Yogo)",
            priority="essential",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.anderson_rubin_test(data=df, ...)",
            "Anderson-Rubin test: valid inference even with weak instruments",
            priority="recommended",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.kitagawa_test(data=df, y='...', treatment='...', instrument='...')",
            "Kitagawa (2015) LATE validity test",
            priority="recommended",
            category="sensitivity",
        ))
        steps.append(Step(
            "sp.iv_bounds(data=df, y='...', treatment='...', instrument='...')",
            "Nevo-Rosen (2012) bounds under imperfect instruments",
            priority="optional",
            category="sensitivity",
        ))

    # ── Matching / IPW ─────────────────────────────────────────────── #
    elif method_family == "matching":
        steps.append(Step(
            "sp.ps_balance(data=df, treatment='...', covariates=[...])",
            "Check covariate balance after matching/weighting (SMD < 0.1)",
            priority="essential",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.overlap_plot(data=df, treatment='...', covariates=[...])",
            "Propensity score overlap plot — check common support",
            priority="essential",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.love_plot(data=df, treatment='...', covariates=[...])",
            "Love plot: before/after balance visualization",
            priority="recommended",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.trimming(data=df, treatment='...', covariates=[...])",
            "Crump (2009) optimal trimming for improved overlap",
            priority="recommended",
            category="robustness",
        ))
        steps.append(Step(
            "sp.sensemakr(data=df, ...)",
            "Cinelli & Hazlett (2020) sensitivity to unobserved confounding",
            priority="recommended",
            category="sensitivity",
        ))

    # ── Synthetic Control ──────────────────────────────────────────── #
    elif method_family == "synth":
        steps.append(Step(
            "result.plot()",
            "Plot treated vs synthetic control unit",
            priority="essential",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.diagnose_result(result)",
            "Pre-treatment RMSPE and donor weight diagnostics",
            priority="essential",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.conformal_synth(data=df, ...)",
            "Conformal inference for valid p-values (Chernozhukov et al.)",
            priority="recommended",
            category="sensitivity",
        ))

    # ── DML / ML causal ────────────────────────────────────────────── #
    elif method_family == "dml":
        steps.append(Step(
            "sp.diagnose_result(result)",
            "Check outcome/treatment model cross-validated R² — "
            "both must predict reasonably well",
            priority="essential",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.cate_summary(result)",
            "Summarize treatment effect heterogeneity across subgroups",
            priority="recommended",
            category="diagnostics",
        ))

    # ── Meta-learners / CATE ───────────────────────────────────────── #
    elif method_family == "hte":
        steps.append(Step(
            "sp.cate_summary(result)",
            "Summarize CATE distribution and key subgroups",
            priority="essential",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.gate_test(result)",
            "GATES test: are treatment effect differences across groups real?",
            priority="essential",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.blp_test(result)",
            "BLP test: is there systematic heterogeneity? (Chernozhukov et al.)",
            priority="essential",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.cate_plot(result)",
            "Visualize CATE distribution",
            priority="recommended",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.compare_metalearners(data=df, ...)",
            "Compare S/T/X/R/DR-Learners for robustness",
            priority="recommended",
            category="alternative",
        ))

    # ── Mediation ──────────────────────────────────────────────────── #
    elif method_family == "mediation":
        steps.append(Step(
            "result.plot()",
            "Path diagram with direct/indirect/total effects",
            priority="recommended",
            category="diagnostics",
        ))
        steps.append(Step(
            "sp.sensemakr(data=df, ...)",
            "Sensitivity to unobserved confounding of the mediator",
            priority="recommended",
            category="sensitivity",
        ))

    # ── Universal steps (all causal methods) ───────────────────────── #

    # Bounds / sensitivity
    if method_family not in ("synth",):  # synth has own sensitivity
        steps.append(Step(
            "sp.evalue(result.estimate, result.se)",
            "E-value: minimum confounding strength to explain away the effect",
            priority="recommended",
            category="sensitivity",
        ))

    # Subgroup analysis
    steps.append(Step(
        "sp.subgroup_analysis(data=df, ...)",
        "Test for heterogeneous effects across subgroups (with Wald test)",
        priority="optional",
        category="robustness",
    ))

    # Export
    steps.append(Step(
        "sp.regtable(result1, result2, ...)",
        "Publication-quality comparison table (LaTeX / Word / Excel)",
        priority="optional",
        category="export",
    ))
    steps.append(Step(
        "result.cite()",
        "Get BibTeX citation for the method used",
        priority="optional",
        category="export",
    ))

    return steps


# ====================================================================== #
#  Method family detection
# ====================================================================== #

def _detect_family(method: str) -> str:
    """Classify a method string into a family."""
    m = method.lower()

    if any(k in m for k in ("did", "diff", "callaway", "sun_abraham",
                             "stagger", "imputation", "wooldridge",
                             "chaisemartin", "changes-in-changes",
                             "cic", "stacked")):
        return "did"

    if any(k in m for k in ("rd", "discontinuity", "rdrobust", "kink",
                             "rdit")):
        return "rd"

    if any(k in m for k in ("iv ", "2sls", "instrumental", "bartik",
                             "deepiv", "shift-share")):
        return "iv"

    if any(k in m for k in ("match", "psm", "cem", "mahalanobis",
                             "ipw", "entropy", "aipw", "tmle")):
        return "matching"

    if any(k in m for k in ("synth", "scm", "sdid", "augsynth")):
        return "synth"

    if any(k in m for k in ("dml", "double", "debiased")):
        return "dml"

    if any(k in m for k in ("metalearner", "slearner", "tlearner",
                             "xlearner", "rlearner", "drlearner",
                             "causal forest", "bcf", "cate",
                             "tarnet", "cfrnet", "dragonnet")):
        return "hte"

    if any(k in m for k in ("mediat",)):
        return "mediation"

    if any(k in m for k in ("causal impact", "structural time")):
        return "synth"

    return "generic"
