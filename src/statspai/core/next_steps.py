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
- ``executable`` : bool — ``True`` when ``action`` runs as written with
  ``result`` bound to the fitted result and ``df`` to the data it was fitted
  on. Column names the fit recorded are filled in; a step that still needs
  something the result does not know (another fit, a cohort column, the
  covariates to balance on) keeps a ``'...'`` placeholder and is ``False``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

# ====================================================================== #
#  Step data structure
# ====================================================================== #


class Step:
    """One recommended next step."""

    __slots__ = ("action", "reason", "priority", "category", "executable")

    def __init__(
        self,
        action: str,
        reason: str,
        priority: str = "recommended",
        category: str = "diagnostics",
        executable: Optional[bool] = None,
    ):
        self.action = action
        self.reason = reason
        self.priority = priority
        self.category = category
        self.executable = ("..." not in action) if executable is None else executable

    def __repr__(self) -> str:
        icon = {"essential": "❗", "recommended": "→", "optional": "○"}.get(
            self.priority, "→"
        )
        return f"{icon} {self.action}\n   {self.reason}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "priority": self.priority,
            "category": self.category,
            "executable": self.executable,
        }


# ====================================================================== #
#  Filling actions from the fit's own call
# ====================================================================== #

#: Recorded argument names that mean the same column, mapped to one key.
_CTX_ALIASES = {
    "y": ("y", "outcome", "dep_var", "depvar"),
    "treat": ("treat", "treatment", "d", "D"),
    "covariates": ("covariates", "controls", "x_controls", "X"),
    "time": ("time", "t", "tvar"),
    "id": ("id", "i", "unit", "ivar"),
    "cohort": ("g", "gvar", "cohort", "first_treat"),
    "running": ("x", "running", "running_var"),
    "cutoff": ("c", "cutoff"),
    "covs": ("covs",),
    "formula": ("formula", "fml"),
    "treated_unit": ("treated_unit",),
    "treatment_time": ("treatment_time",),
}


def _is_column_name(value: Any) -> bool:
    # Internal helper columns (``sp.did`` routes through a synthetic
    # ``_statspai_cohort``) are not in the caller's frame.
    return isinstance(value, str) and bool(value) and not value.startswith("_statspai")


def _fit_context(result: Any) -> Dict[str, Any]:
    """Arguments the fit recorded, keyed by role (``y``, ``treat``, ...)."""
    prov = getattr(result, "_provenance", None)
    params = dict(getattr(prov, "params", None) or {})
    mi = getattr(result, "model_info", None) or {}
    ctx: Dict[str, Any] = {}
    for role, names in _CTX_ALIASES.items():
        for name in names:
            value = params.get(name)
            if value is None:
                continue
            if role == "cutoff":
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    ctx[role] = float(value)
                    break
                continue
            if role in ("covariates", "covs"):
                if (
                    isinstance(value, (list, tuple))
                    and value
                    and all(_is_column_name(v) for v in value)
                ):
                    ctx[role] = list(value)
                    break
                continue
            if role == "treatment_time":
                if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                    ctx[role] = value
                    break
                continue
            if _is_column_name(value):
                ctx[role] = value
                break
    if "running" in ctx and "cutoff" not in ctx:
        c = mi.get("cutoff", mi.get("c"))
        if isinstance(c, (int, float)) and not isinstance(c, bool):
            ctx["cutoff"] = float(c)
    if "y" not in ctx:
        dep = (getattr(result, "data_info", None) or {}).get("dependent_var")
        if _is_column_name(dep):
            ctx["y"] = dep
    return ctx


def _q(ctx: Dict[str, Any], role: str, *, kind: str = "str") -> str:
    """Python literal for ``ctx[role]``, or a ``'...'`` placeholder."""
    value = ctx.get(role)
    if value is None or value == [] or value == "":
        return "[...]" if kind == "list" else "'...'"
    if kind == "list":
        return repr(list(value))
    if kind == "num" and isinstance(value, float) and value.is_integer():
        return repr(int(value)) if abs(value) < 1e15 else repr(value)
    return repr(value)


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

    by_priority: Dict[str, List[Step]] = {
        "essential": [],
        "recommended": [],
        "optional": [],
    }
    for s in steps:
        by_priority.get(s.priority, by_priority["recommended"]).append(s)

    idx = 1
    for pri, label in [
        ("essential", "Essential"),
        ("recommended", "Recommended"),
        ("optional", "Optional"),
    ]:
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
        "<style scoped>"
        '.sp-ns{font-family:"Helvetica Neue",Arial,sans-serif;max-width:700px;'
        "border:1px solid #E5E7EB;border-radius:8px;overflow:hidden;margin:8px 0}"
        ".sp-ns-hdr{background:linear-gradient(135deg,#0f766e 0%,#115e59 100%);"
        "color:#fff;padding:10px 16px;font-size:14px;font-weight:600}"
        ".sp-ns-body{padding:8px 0}"
        ".sp-ns-cat{padding:4px 16px;font-size:11px;font-weight:700;"
        "color:#64748B;text-transform:uppercase;letter-spacing:0.5px;"
        "border-bottom:1px solid #F1F5F9;margin-top:4px}"
        ".sp-ns-item{padding:6px 16px 6px 24px;font-size:12px;"
        "border-bottom:1px solid #F8FAFC}"
        ".sp-ns-item:hover{background:#F0FDFA}"
        '.sp-ns-code{font-family:"SF Mono",Menlo,monospace;font-size:12px;'
        "color:#0f766e;font-weight:600}"
        ".sp-ns-why{color:#64748B;font-size:11px;margin-top:2px}"
        ".sp-ns-badge{display:inline-block;font-size:9px;padding:1px 6px;"
        "border-radius:3px;font-weight:600;margin-left:6px}"
        ".sp-ns-ess{background:#FEE2E2;color:#991B1B}"
        ".sp-ns-rec{background:#DBEAFE;color:#1E40AF}"
        ".sp-ns-opt{background:#F3F4F6;color:#6B7280}"
        "</style>"
    )

    badge_cls = {
        "essential": "sp-ns-ess",
        "recommended": "sp-ns-rec",
        "optional": "sp-ns-opt",
    }
    badge_txt = {
        "essential": "essential",
        "recommended": "recommended",
        "optional": "optional",
    }

    h = [
        css,
        '<div class="sp-ns">',
        f'<div class="sp-ns-hdr">{title}</div>',
        '<div class="sp-ns-body">',
    ]

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
            h.append(
                f'<div><span class="sp-ns-code">{s.action}</span>'
                f'<span class="sp-ns-badge {bcls}">{btxt}</span></div>'
            )
            h.append(f'<div class="sp-ns-why">{s.reason}</div>')
            h.append("</div>")

    h.append("</div></div>")
    return "\n".join(h)


# ====================================================================== #
#  EconometricResults next_steps
# ====================================================================== #


def _has_log_likelihood(result: Any) -> bool:
    """Whether ``sp.estat(result, 'ic')`` has a likelihood to work from."""
    mi = getattr(result, "model_info", None) or {}
    diag = getattr(result, "diagnostics", None) or {}
    ll = mi.get("ll", diag.get("Log-Likelihood", diag.get("log_likelihood")))
    try:
        return ll is not None and bool(np.isfinite(float(ll)))
    except (TypeError, ValueError):
        return False


_IV_MODEL = re.compile(r"\b(?:iv|2sls|tsls|liml|fuller|gmm)\b")
_CLASSICAL_SE = {"", "none", "nonrobust", "iid", "classical", "ols", "unadjusted"}
_CLUSTER_SE = {"crv", "crv1", "crv2", "crv3", "cluster", "twoway", "cr1", "cr2", "cr3"}


def _is_iv_fit(model_type: str) -> bool:
    """IV fit, judged on whole words (``"OLS (StatsPAI native)"`` holds "iv")."""
    return bool(_IV_MODEL.search(model_type))


def _se_kind(mi: Dict[str, Any]) -> str:
    """``'cluster'``, ``'robust'`` or ``'classical'`` from what the fit recorded.

    Estimators record the variance choice under different keys:
    ``sp.regress`` keeps ``robust`` and puts a cluster in ``cluster``,
    ``sp.feols`` uses ``vcov_type`` / ``clustervar``.
    """
    clusters = mi.get("clustervar") or mi.get("cluster") or mi.get("cluster_var")
    vcov_type = str(mi.get("vcov_type") or "").strip().lower()
    if clusters or vcov_type in _CLUSTER_SE or vcov_type.startswith("cr"):
        return "cluster"
    robust = str(mi.get("robust") or "").strip().lower()
    if robust not in _CLASSICAL_SE or vcov_type not in _CLASSICAL_SE:
        return "robust"
    return "classical"


def _variance_step(result: Any, ctx: Dict[str, Any], model_type: str) -> Optional[Step]:
    """The standard-error follow-up that fits how the model was estimated.

    Suggesting heteroskedasticity-robust SEs on a fit that already clusters
    (the 2026-10-19 audit found it on every ``sp.feols`` result) points the
    user backwards; a clustered fit with few clusters needs a wild cluster
    bootstrap instead.
    """
    from ._agent_summary import _FEW_CLUSTERS_MIN

    mi = result.model_info or {}
    kind = _se_kind(mi)
    prov = getattr(result, "_provenance", None)
    fn = str(getattr(prov, "function", "") or "")
    if kind == "classical":
        formula = (
            ctx.get("formula") or f"{result.data_info.get('dependent_var', 'y')} ~ ..."
        )
        if fn == "sp.feols":
            action = f"sp.feols({formula!r}, data=df, vcov='hetero')"
        else:
            refit = "sp.iv" if _is_iv_fit(model_type) else "sp.regress"
            action = f"{refit}({formula!r}, data=df, robust='hc1')"
        return Step(
            action,
            "Re-estimate with heteroskedasticity-robust standard errors",
            priority="recommended",
            category="robustness",
        )
    if kind == "cluster":
        n_clusters = mi.get("n_clusters")
        try:
            few = n_clusters is not None and int(n_clusters) < _FEW_CLUSTERS_MIN
        except (TypeError, ValueError):
            few = False
        if few:
            cluster = mi.get("clustervar") or mi.get("cluster")
            if isinstance(cluster, (list, tuple)):
                cluster = cluster[0] if len(cluster) == 1 else None
            cl = repr(cluster) if isinstance(cluster, str) else "'...'"
            return Step(
                f"sp.wild_cluster_boot(result, data=df, cluster={cl}, "
                "variable='...')",
                f"Only {int(n_clusters)} clusters: cluster-robust SEs over-reject; "
                "check with a wild cluster bootstrap",
                priority="recommended",
                category="robustness",
            )
    return None


def econometric_next_steps(result: Any) -> List[Step]:
    """Generate next steps for an EconometricResults object."""
    steps: List[Step] = []
    ctx = _fit_context(result)
    model_type = (result.model_info.get("model_type", "") or "").lower()
    has_residuals = result.data_info.get("residuals") is not None
    has_X = result.data_info.get("X") is not None

    # ── Design guardrails recorded at fit time come first ─────────── #
    from ._design_guardrails import stored_design_violations

    for v in stored_design_violations(result):
        if v.get("python_call") and v.get("severity") in ("warning", "error"):
            steps.append(
                Step(
                    str(v["python_call"]),
                    str(v.get("message", "")),
                    priority="essential",
                    category="alternative",
                )
            )

    # ── Diagnostics ────────────────────────────────────────────────── #
    if has_residuals and has_X:
        steps.append(
            Step(
                "sp.estat(result, 'all')",
                "Run all post-estimation diagnostics (heteroskedasticity, "
                "normality, VIF, RESET, serial correlation)",
                priority="essential",
                category="diagnostics",
            )
        )
    elif _has_log_likelihood(result):
        steps.append(
            Step(
                "sp.estat(result, 'ic')",
                "Compare information criteria (AIC, BIC) across specifications",
                priority="recommended",
                category="diagnostics",
            )
        )

    # Standard errors: robust when classical, a wild cluster bootstrap when
    # clustered on few clusters, nothing when already appropriate. An IV
    # fit is re-run through its own estimator (sp.regress would drop the
    # instruments).
    variance_step = _variance_step(result, ctx, model_type)
    if variance_step is not None:
        steps.append(variance_step)

    # ── Post-estimation ────────────────────────────────────────────── #
    steps.append(
        Step(
            "sp.margins(result, data=df)",
            "Compute average marginal effects for substantive interpretation",
            priority="recommended",
            category="diagnostics",
        )
    )

    # ── Robustness ─────────────────────────────────────────────────── #
    steps.append(
        Step(
            f"sp.oster_delta(data=df, y={_q(ctx, 'y')}, x_base=[...], "
            "x_controls=[...])",
            "Oster (2019) coefficient stability — assess sensitivity to unobservables",
            priority="recommended",
            category="sensitivity",
        )
    )

    # Variable selection check
    n_params = len(result.params)
    if n_params >= 5:
        steps.append(
            Step(
                f"sp.stepwise(data=df, y={_q(ctx, 'y')}, x=[...], criterion='bic')",
                f"Variable selection: {n_params} regressors — "
                "check if BIC prefers a more parsimonious model",
                priority="optional",
                category="robustness",
            )
        )

    # ── Export ─────────────────────────────────────────────────────── #
    steps.append(
        Step(
            "sp.esttab(result)",
            "Create publication-quality comparison table (text/LaTeX/Word/Excel)",
            priority="recommended",
            category="export",
        )
    )
    steps.append(
        Step(
            "sp.outreg2(result, filename='table.xlsx')",
            "Export formatted results to Excel",
            priority="optional",
            category="export",
        )
    )

    # ── IV-specific ────────────────────────────────────────────────── #
    if _is_iv_fit(model_type):
        steps.insert(
            0,
            Step(
                "sp.estat(result, 'firststage')",
                "Check first-stage F-statistic (Stock-Yogo threshold: F > 10)",
                priority="essential",
                category="diagnostics",
            ),
        )
        # The J test has (excluded instruments - endogenous regressors)
        # degrees of freedom; on a just-identified fit it does not exist,
        # so recommending it would send the user after a non-test. When
        # the counts are unknown the step is kept (the audit does the same).
        from ..smart.audit import _overid_degree

        overid_view = dict(result.model_info or {})
        for k, v in (getattr(result, "diagnostics", None) or {}).items():
            overid_view.setdefault(k, v)
        overid_degree = _overid_degree(overid_view)
        pos = 1
        if overid_degree is None or overid_degree > 0:
            steps.insert(
                pos,
                Step(
                    "sp.estat(result, 'overid')",
                    "Sargan/Hansen J test of the over-identifying restrictions",
                    priority="essential",
                    category="diagnostics",
                ),
            )
            pos += 1
        steps.insert(
            pos,
            Step(
                "sp.estat(result, 'endogenous')",
                "Durbin-Wu-Hausman test: is IV actually needed vs OLS?",
                priority="essential",
                category="diagnostics",
            ),
        )
        steps.append(
            Step(
                f"sp.kitagawa_test(data=df, y={_q(ctx, 'y')}, treatment='...', "
                "instrument='...')",
                "Test LATE validity (Kitagawa 2015 — necessary conditions for LATE)",
                priority="recommended",
                category="sensitivity",
            )
        )

    # ── Panel-specific ─────────────────────────────────────────────── #
    if any(k in model_type for k in ("panel", "fe", "fixed")):
        steps.append(
            Step(
                f"sp.hausman_test(data=df, y={_q(ctx, 'y')}, x=[...], "
                f"id={_q(ctx, 'id')}, time={_q(ctx, 'time')})",
                "Hausman test: fixed effects vs random effects",
                priority="recommended",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                "sp.twoway_cluster(result, data=df, cluster1='...', cluster2='...')",
                "Two-way clustered SEs (e.g., firm × year)",
                priority="optional",
                category="robustness",
            )
        )

    return steps


# ====================================================================== #
#  CausalResult next_steps
# ====================================================================== #


def causal_next_steps(result: Any) -> List[Step]:
    """Generate next steps for a CausalResult object."""
    steps: List[Step] = []
    method = (result.method or "").lower()
    model_info = result.model_info or {}
    has_event_study = model_info.get("event_study") is not None or (
        result.detail is not None
        and "relative_time"
        in (result.detail.columns if result.detail is not None else [])
    )

    method_family = _detect_family(method)
    ctx = _fit_context(result)

    # ── DID-family ─────────────────────────────────────────────────── #
    if method_family == "did":
        if has_event_study:
            steps.append(
                Step(
                    "sp.pretrends_test(result)",
                    "Joint test of pre-treatment coefficients (parallel trends)",
                    priority="essential",
                    category="diagnostics",
                )
            )
            steps.append(
                Step(
                    "sp.pretrends_power(result)",
                    "Roth (2022) — is the pre-trend test powerful enough "
                    "to detect relevant violations?",
                    priority="essential",
                    category="diagnostics",
                )
            )
            steps.append(
                Step(
                    "sp.honest_did(result)",
                    "Rambachan & Roth (2023) — honest CI under bounded "
                    "parallel trends violations",
                    priority="essential",
                    category="sensitivity",
                )
            )
            steps.append(
                Step(
                    "sp.breakdown_m(result)",
                    "Rambachan & Roth (2023) — largest violation of parallel "
                    "trends under which the effect stays significant",
                    priority="recommended",
                    category="sensitivity",
                )
            )
        else:
            steps.append(
                Step(
                    f"sp.event_study(data=df, y={_q(ctx, 'y')}, treat_time='...', "
                    f"time={_q(ctx, 'time')}, unit={_q(ctx, 'id')})",
                    "Estimate event study to visualize pre-trends and dynamic effects",
                    priority="essential",
                    category="diagnostics",
                )
            )

        steps.append(
            Step(
                "result.plot()",
                "Plot event study / treatment effects",
                priority="recommended",
                category="diagnostics",
            )
        )

        # Suggest alternative DID estimators
        if "2x2" in method or "twfe" in method.replace(" ", ""):
            steps.append(
                Step(
                    f"sp.bacon_decomposition(data=df, y={_q(ctx, 'y')}, "
                    f"treat={_q(ctx, 'treat')}, time={_q(ctx, 'time')}, "
                    f"id={_q(ctx, 'id')})",
                    "Goodman-Bacon decomposition: check for negative weights in TWFE",
                    priority="recommended",
                    category="diagnostics",
                )
            )
            steps.append(
                Step(
                    f"sp.did_imputation(data=df, y={_q(ctx, 'y')}, "
                    f"group={_q(ctx, 'id')}, time={_q(ctx, 'time')}, "
                    "first_treat='...')",
                    "Try BJS (2024) imputation estimator — robust to heterogeneous "
                    "effects",
                    priority="recommended",
                    category="alternative",
                )
            )
            steps.append(
                Step(
                    f"sp.callaway_santanna(data=df, y={_q(ctx, 'y')}, g='...', "
                    f"t={_q(ctx, 'time')}, i={_q(ctx, 'id')})",
                    "Try Callaway-Sant'Anna — group-time ATT with flexible aggregation",
                    priority="recommended",
                    category="alternative",
                )
            )

        # Multiple outcomes
        steps.append(
            Step(
                "sp.romano_wolf(data=df, y=[...], x=[...], n_boot=1000)",
                "If testing multiple outcomes: Romano-Wolf FWER-adjusted p-values",
                priority="optional",
                category="robustness",
            )
        )

    # ── RD-family ──────────────────────────────────────────────────── #
    elif method_family == "rd":
        steps.append(
            Step(
                f"sp.mccrary_test(data=df, x={_q(ctx, 'running')}, "
                f"c={_q(ctx, 'cutoff', kind='num')})",
                "McCrary density test: check for manipulation at the cutoff",
                priority="essential",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                f"sp.rd_honest(data=df, y={_q(ctx, 'y')}, x={_q(ctx, 'running')}, "
                f"c={_q(ctx, 'cutoff', kind='num')})",
                "Armstrong-Kolesár honest CI — valid uniformly over smooth functions",
                priority="essential",
                category="sensitivity",
            )
        )
        steps.append(
            Step(
                f"sp.rdbalance(data=df, x={_q(ctx, 'running')}, "
                f"c={_q(ctx, 'cutoff', kind='num')}, covs={_q(ctx, 'covs', kind='list')})",
                "Covariate balance test at the cutoff",
                priority="essential",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                f"sp.rdbwsensitivity(data=df, y={_q(ctx, 'y')}, "
                f"x={_q(ctx, 'running')}, c={_q(ctx, 'cutoff', kind='num')})",
                "Bandwidth sensitivity: check stability at h/2, h, 2h",
                priority="recommended",
                category="robustness",
            )
        )
        steps.append(
            Step(
                f"sp.rdplacebo(data=df, y={_q(ctx, 'y')}, x={_q(ctx, 'running')}, "
                f"c={_q(ctx, 'cutoff', kind='num')})",
                "Placebo cutoff test: should find no effect at false cutoffs",
                priority="recommended",
                category="robustness",
            )
        )
        steps.append(
            Step(
                "result.plot()",
                "RD plot with confidence intervals",
                priority="recommended",
                category="diagnostics",
            )
        )

    # ── IV-family ──────────────────────────────────────────────────── #
    elif method_family == "iv":
        steps.append(
            Step(
                "sp.estat(result, 'firststage')",
                "First-stage F: rule of thumb F > 10 (Stock-Yogo)",
                priority="essential",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                f"sp.anderson_rubin_test(data=df, y={_q(ctx, 'y')}, endog='...', "
                "instruments=[...])",
                "Anderson-Rubin test: valid inference even with weak instruments",
                priority="recommended",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                f"sp.kitagawa_test(data=df, y={_q(ctx, 'y')}, "
                f"treatment={_q(ctx, 'treat')}, instrument='...')",
                "Kitagawa (2015) LATE validity test",
                priority="recommended",
                category="sensitivity",
            )
        )
        steps.append(
            Step(
                f"sp.iv_bounds(data=df, y={_q(ctx, 'y')}, "
                f"treatment={_q(ctx, 'treat')}, instrument='...')",
                "Nevo-Rosen (2012) bounds under imperfect instruments",
                priority="optional",
                category="sensitivity",
            )
        )

    # ── Matching / IPW ─────────────────────────────────────────────── #
    elif method_family == "matching":
        steps.append(
            Step(
                f"sp.ps_balance(data=df, treatment={_q(ctx, 'treat')}, "
                f"covariates={_q(ctx, 'covariates', kind='list')})",
                "Check covariate balance after matching/weighting (SMD < 0.1)",
                priority="essential",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                f"sp.overlap_plot(data=df, treatment={_q(ctx, 'treat')}, "
                f"covariates={_q(ctx, 'covariates', kind='list')})",
                "Propensity score overlap plot — check common support",
                priority="essential",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                f"sp.love_plot(data=df, treatment={_q(ctx, 'treat')}, "
                f"covariates={_q(ctx, 'covariates', kind='list')})",
                "Love plot: before/after balance visualization",
                priority="recommended",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                f"sp.trimming(data=df, treatment={_q(ctx, 'treat')}, "
                f"covariates={_q(ctx, 'covariates', kind='list')})",
                "Crump (2009) optimal trimming for improved overlap",
                priority="recommended",
                category="robustness",
            )
        )
        steps.append(
            Step(
                f"sp.sensemakr(data=df, y={_q(ctx, 'y')}, treat={_q(ctx, 'treat')}, "
                f"controls={_q(ctx, 'covariates', kind='list')})",
                "Cinelli & Hazlett (2020) sensitivity to unobserved confounding",
                priority="recommended",
                category="sensitivity",
            )
        )

    # ── Synthetic Control ──────────────────────────────────────────── #
    elif method_family == "synth":
        steps.append(
            Step(
                "result.plot()",
                "Plot treated vs synthetic control unit",
                priority="essential",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                "sp.diagnose_result(result)",
                "Pre-treatment RMSPE and donor weight diagnostics",
                priority="essential",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                f"sp.conformal_synth(data=df, outcome={_q(ctx, 'y')}, "
                f"unit={_q(ctx, 'id')}, time={_q(ctx, 'time')}, "
                f"treated_unit={_q(ctx, 'treated_unit')}, "
                f"treatment_time={_q(ctx, 'treatment_time', kind='num')})",
                "Conformal inference for valid p-values (Chernozhukov et al.)",
                priority="recommended",
                category="sensitivity",
            )
        )

    # ── DML / ML causal ────────────────────────────────────────────── #
    elif method_family == "dml":
        steps.append(
            Step(
                "sp.diagnose_result(result)",
                "Check outcome/treatment model cross-validated R² — "
                "both must predict reasonably well",
                priority="essential",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                "sp.dml_sensitivity(result)",
                "Chernozhukov et al. (2022) omitted-variable-bias bounds "
                "for the DML estimate",
                priority="recommended",
                category="sensitivity",
            )
        )

    # ── Meta-learners / CATE ───────────────────────────────────────── #
    elif method_family == "hte":
        steps.append(
            Step(
                "sp.cate_summary(result)",
                "Summarize CATE distribution and key subgroups",
                priority="essential",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                "sp.gate_test(result, data=df, by='...')",
                "GATES test: are treatment effect differences across groups real?",
                priority="essential",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                f"sp.blp_test(result, data=df, y={_q(ctx, 'y')}, "
                f"treat={_q(ctx, 'treat')}, "
                f"covariates={_q(ctx, 'covariates', kind='list')})",
                "BLP test: is there systematic heterogeneity? (Chernozhukov et al.)",
                priority="essential",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                "sp.cate_plot(result)",
                "Visualize CATE distribution",
                priority="recommended",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                f"sp.compare_metalearners(data=df, y={_q(ctx, 'y')}, "
                f"treat={_q(ctx, 'treat')}, "
                f"covariates={_q(ctx, 'covariates', kind='list')})",
                "Compare S/T/X/R/DR-Learners for robustness",
                priority="recommended",
                category="alternative",
            )
        )

    # ── Mediation ──────────────────────────────────────────────────── #
    elif method_family == "mediation":
        steps.append(
            Step(
                "result.plot()",
                "Path diagram with direct/indirect/total effects",
                priority="recommended",
                category="diagnostics",
            )
        )
        steps.append(
            Step(
                "sp.sensemakr(data=df, y='...', treat='...', controls=[...])",
                "Sensitivity to unobserved confounding of the mediator",
                priority="recommended",
                category="sensitivity",
            )
        )

    # ── Universal steps (all causal methods) ───────────────────────── #

    # E-value: a bound on unmeasured confounding, so only for designs that
    # assume no unmeasured confounding. DiD / RD / IV / synth rest on
    # other assumptions, and a mean difference needs the outcome SD to be
    # put on the risk-ratio scale (sp.evalue_from_result refuses without).
    if method_family in ("matching", "dml", "hte", "mediation", "generic"):
        steps.append(
            Step(
                f"sp.evalue_from_result(result, sd=df[{_q(ctx, 'y')}].std())",
                "E-value: minimum confounding strength to explain away the effect",
                priority="recommended",
                category="sensitivity",
            )
        )

    # Subgroup analysis
    steps.append(
        Step(
            "sp.subgroup_analysis(data=df, formula='...', x='...', by={...})",
            "Test for heterogeneous effects across subgroups (with Wald test)",
            priority="optional",
            category="robustness",
        )
    )

    # Export
    steps.append(
        Step(
            "sp.regtable(result)",
            "Publication-quality comparison table (LaTeX / Word / Excel)",
            priority="optional",
            category="export",
        )
    )
    steps.append(
        Step(
            "result.cite()",
            "Get BibTeX citation for the method used",
            priority="optional",
            category="export",
        )
    )

    return steps


# ====================================================================== #
#  Method family detection
# ====================================================================== #


def _detect_family(method: str) -> str:
    """Classify a method string into a family.

    Order matters: more specific patterns are checked before broader ones
    (e.g. ``sdid`` → synth before ``did`` → did).
    """
    m = method.lower()

    # --- Synthetic control (check before DID: "sdid" contains "did") ---
    if any(
        k in m
        for k in (
            "synth",
            "scm",
            "sdid",
            "augsynth",
            "causal impact",
            "structural time",
        )
    ):
        return "synth"

    # --- DID ---
    if any(
        k in m
        for k in (
            "did",
            "diff",
            "callaway",
            "sun_abraham",
            "stagger",
            "imputation",
            "wooldridge",
            "chaisemartin",
            "changes-in-changes",
            "twfe",
            "cic",
            "stacked",
        )
    ):
        return "did"

    # --- RD ---
    if any(k in m for k in ("rd", "discontinuity", "rdrobust", "kink", "rdit")):
        return "rd"

    # --- IV (match bare "iv" and "iv ..." via word boundary) ---
    if any(k in m for k in ("2sls", "instrumental", "bartik", "deepiv", "shift-share")):
        return "iv"
    if re.search(r"\biv\b", m):
        return "iv"

    # --- Matching / IPW ---
    if any(
        k in m
        for k in (
            "match",
            "psm",
            "cem",
            "mahalanobis",
            "ipw",
            "entropy",
            "aipw",
            "tmle",
        )
    ):
        return "matching"

    # --- DML ---
    if any(k in m for k in ("dml", "double", "debiased")):
        return "dml"

    # --- HTE / Meta-learners ---
    # Compare without spaces / hyphens: methods print as
    # "Meta-Learner (T-Learner)".
    compact = re.sub(r"[\s_\-]", "", m)
    if any(
        k in m or k.replace(" ", "") in compact
        for k in (
            "metalearner",
            "slearner",
            "tlearner",
            "xlearner",
            "rlearner",
            "drlearner",
            "causal forest",
            "bcf",
            "cate",
            "tarnet",
            "cfrnet",
            "dragonnet",
        )
    ):
        return "hte"

    # --- Mediation ---
    if any(k in m for k in ("mediat",)):
        return "mediation"

    return "generic"
