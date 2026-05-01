"""``sp.causal()`` — end-to-end causal-inference orchestrator.

The workflow object runs the canonical pipeline:

    1. ``.diagnose()``     — sp.check_identification (design-level blockers)
    2. ``.recommend()``    — sp.recommend         (pick estimator)
    3. ``.estimate()``     — fit the recommended model
    4. ``.robustness()``   — method-specific robustness suite
    5. ``.report(path)``   — one-page HTML summary with every output

Every stage is cached; re-invoking ``.report()`` does not re-fit.
Each stage is also independently callable so advanced users can skip
or override steps.

Usage
-----
>>> import statspai as sp
>>> w = sp.causal(df, y='wage', treatment='training',
...               id='worker', time='year', design='did')
>>> w.report('analysis.html')
>>> w.result      # the fitted CausalResult
>>> w.diagnostics # IdentificationReport
"""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Workflow object
# ---------------------------------------------------------------------------

@dataclass
class CausalWorkflow:
    """Holds state across the diagnose -> estimate -> report pipeline."""

    data: pd.DataFrame
    y: str
    treatment: Optional[str]
    covariates: List[str]
    id: Optional[str]
    time: Optional[str]
    running_var: Optional[str]
    instrument: Optional[str]
    cutoff: Optional[float]
    cohort: Optional[str]
    cluster: Optional[str]
    design: Optional[str]
    dag: Optional[Any]
    strict: bool
    # Sprint-B (0.9.6) causal-method context — opt-in.
    mediator: Optional[str] = None
    tv_confounders: Optional[List[str]] = None
    proxy_z: Optional[List[str]] = None
    proxy_w: Optional[List[str]] = None
    post_treat_strata: Optional[str] = None

    # Outputs (filled as stages run)
    diagnostics: Optional[Any] = None           # IdentificationReport
    recommendation: Optional[Any] = None        # RecommendationResult
    result: Optional[Any] = None                # CausalResult / EconometricResults
    robustness_findings: Dict[str, Any] = field(default_factory=dict)

    # Execution stats
    stages_completed: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Stage 1: diagnose
    # ------------------------------------------------------------------

    def diagnose(self):
        """Run sp.check_identification, cache the report."""
        from ..smart.identification import check_identification
        self.diagnostics = check_identification(
            self.data, y=self.y, treatment=self.treatment,
            covariates=self.covariates, id=self.id, time=self.time,
            running_var=self.running_var, instrument=self.instrument,
            cluster=self.cluster, cutoff=self.cutoff,
            design=self.design, cohort=self.cohort, dag=self.dag,
            strict=self.strict,
        )
        # Auto-adopt design if detection was left to check_identification
        if self.design is None:
            self.design = self.diagnostics.design
        self._mark('diagnose')
        return self.diagnostics

    # ------------------------------------------------------------------
    # Stage 2: recommend
    # ------------------------------------------------------------------

    def recommend(self):
        """Run sp.recommend() to pick an estimator."""
        from ..smart.recommend import recommend as _rec
        self.recommendation = _rec(
            data=self.data, y=self.y, treatment=self.treatment,
            covariates=self.covariates, id=self.id, time=self.time,
            running_var=self.running_var, instrument=self.instrument,
            cutoff=self.cutoff, design=self.design, dag=self.dag,
            # Forward Sprint-B context so the recommender can surface
            # MSM / proximal / principal-strat / mediation candidates.
            mediator=self.mediator,
            tv_confounders=self.tv_confounders,
            proxy_z=self.proxy_z,
            proxy_w=self.proxy_w,
            post_treat_strata=self.post_treat_strata,
        )
        self._mark('recommend')
        return self.recommendation

    # ------------------------------------------------------------------
    # Stage 3: estimate
    # ------------------------------------------------------------------

    def estimate(self):
        """Fit the top-recommended estimator.

        Returns the result object (``CausalResult`` or ``EconometricResults``).
        Uses the workflow's dataset and column mappings; when the top
        recommendation is plain OLS with a treatment-only formula,
        enrich the formula with the user's covariates so confounders
        are actually adjusted for (sp.recommend by default leaves this
        to the caller; the workflow takes responsibility).

        Sprint-B preference: when the caller supplied one of the new
        causal-method hints (proxy_z/proxy_w, tv_confounders,
        post_treat_strata, mediator), that hint wins over the
        design-based default in the top recommendation — the user
        signalled the estimand explicitly.
        """
        if self.recommendation is None:
            self.recommend()

        # If the user passed a Sprint-B hint, route via the fallback
        # which picks the matching Sprint-B estimator directly. This
        # keeps the user's explicit estimand choice authoritative.
        if any([self.proxy_z and self.proxy_w,
                self.tv_confounders and self.id and self.time,
                self.post_treat_strata,
                self.mediator]):
            try:
                self.result = self._fallback_estimate(
                    error="Sprint-B causal-method hint supplied"
                )
                self._mark('estimate')
                return self.result
            except Exception as _exc:
                # IMPORTANT: never swallow the failure silently.
                # The user asked for a specific Sprint-B estimator
                # (proximal / msm / principal_strat / mediation) by
                # passing the matching hint; if it blew up, they need
                # to see the reason before the recommender silently
                # hands back an OLS regression instead.
                import warnings as _warnings
                _warnings.warn(
                    f"Sprint-B causal-method hint failed to execute "
                    f"({type(_exc).__name__}: {_exc}); falling back to "
                    f"the design-based top recommendation. Check the "
                    f"corresponding hint args (proxy_z/proxy_w, "
                    f"tv_confounders, post_treat_strata, mediator) and "
                    f"the data columns they reference.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                # fall through to the normal recommendation path

        top = (self.recommendation.recommendations[0]
               if self.recommendation.recommendations else None)

        # OLS/IV recommendations from recommend() ship a formula that omits
        # the user's covariates; enrich here so the workflow doesn't fit
        # a deliberately under-specified model.
        if (top and top.get('function') in ('regress',) and
                self.covariates and self.treatment):
            rhs = self.treatment + ' + ' + ' + '.join(self.covariates)
            formula = f"{self.y} ~ {rhs}"
            try:
                import statspai as sp
                self.result = sp.regress(formula, data=self.data,
                                         robust='hc1')
                self._mark('estimate')
                return self.result
            except Exception:
                pass  # fall through to generic path below

        # Run the top recommendation via RecommendationResult.run()
        try:
            self.result = self.recommendation.run()
        except Exception as e:
            # Fallback: call the estimator directly with a safe default
            # for the detected design.  This mirrors recommend()'s
            # .run() behaviour but avoids blowing up on param mismatches.
            self.result = self._fallback_estimate(error=e)
        self._mark('estimate')
        return self.result

    def _fallback_estimate(self, error):
        """Direct fallback when recommendation.run() fails."""
        import statspai as sp
        d = self.design

        # Sprint-B (0.9.6) fallbacks — triggered by the advanced kwargs.
        # These run BEFORE the classical design branches so that e.g. a
        # proxy-variable panel still gets proximal even if the detected
        # design is 'observational'.
        if self.proxy_z and self.proxy_w and self.treatment:
            exog = [c for c in (self.covariates or [])
                    if c not in self.proxy_z + self.proxy_w]
            return sp.proximal(
                self.data, y=self.y, treat=self.treatment,
                proxy_z=list(self.proxy_z), proxy_w=list(self.proxy_w),
                covariates=exog,
            )
        if self.tv_confounders and self.treatment and self.id and self.time:
            baseline = [c for c in (self.covariates or [])
                        if c not in self.tv_confounders]
            return sp.msm(
                self.data, y=self.y, treat=self.treatment,
                id=self.id, time=self.time,
                time_varying=list(self.tv_confounders),
                baseline=baseline,
            )
        if self.post_treat_strata and self.treatment:
            if self.covariates:
                return sp.principal_strat(
                    self.data, y=self.y, treat=self.treatment,
                    strata=self.post_treat_strata,
                    covariates=list(self.covariates),
                    method='principal_score',
                )
            return sp.principal_strat(
                self.data, y=self.y, treat=self.treatment,
                strata=self.post_treat_strata,
                method='monotonicity',
            )
        if self.mediator and self.treatment:
            if self.tv_confounders:
                return sp.mediate_interventional(
                    self.data, y=self.y, treat=self.treatment,
                    mediator=self.mediator,
                    tv_confounders=list(self.tv_confounders),
                    covariates=list(self.covariates or []) or None,
                )
            return sp.mediate(
                self.data, y=self.y, treat=self.treatment,
                mediator=self.mediator,
                covariates=list(self.covariates or []) or None,
            )

        # Classical design-based fallbacks.
        if d == 'did':
            if self.time and self.id and self.cohort:
                return sp.callaway_santanna(
                    self.data, y=self.y, g=self.cohort,
                    t=self.time, i=self.id, estimator='reg',
                )
            return sp.did(self.data, y=self.y,
                          treat=self.treatment, time=self.time)
        if d == 'rd' and self.running_var:
            return sp.rdrobust(self.data, y=self.y,
                               x=self.running_var, c=self.cutoff or 0.0)
        if d == 'iv' and self.instrument and self.treatment:
            formula = f"{self.y} ~ ({self.treatment} ~ {self.instrument})"
            return sp.ivreg(formula, data=self.data, robust='hc1')
        if d in ('observational', 'rct'):
            rhs = self.treatment or '1'
            if self.covariates:
                rhs += ' + ' + ' + '.join(self.covariates[:5])
            return sp.regress(f"{self.y} ~ {rhs}", data=self.data,
                              robust='hc1')
        raise RuntimeError(
            f"Cannot fallback-estimate design='{d}' "
            f"(original error: {error})"
        )

    # ------------------------------------------------------------------
    # Stage 4: robustness
    # ------------------------------------------------------------------

    def robustness(self):
        """Run design-appropriate robustness checks on the fitted result.

        Delegates to :func:`statspai.workflow._robustness.run_robustness_battery`
        — the shared battery that powers both the natural-language path
        (``sp.paper(data, question, ...)``) and the estimand-first path
        (``sp.paper(CausalQuestion(...))``). The battery never raises;
        per-check failures land as ``severity='check_failed'`` findings.

        Backwards compatibility: ``self.robustness_findings`` keeps the
        flat ``Dict[str, Any]`` shape that callers (and the existing
        :class:`PaperDraft` renderer) expect — populated from
        :meth:`RobustnessReport.to_dict`. The structured per-finding
        records are reachable via ``self.robustness_findings['_findings']``
        when a caller wants severity-aware rendering.
        """
        if self.result is None:
            self.estimate()

        from ._robustness import run_robustness_battery
        report = run_robustness_battery(
            self.result,
            design=self.design,
            data=self.data,
            treatment=self.treatment,
            outcome=self.y,
            covariates=self.covariates,
        )
        # Cache the structured report for callers that prefer it.
        self._robustness_report = report
        self.robustness_findings = report.to_dict()
        self._mark('robustness')
        return self.robustness_findings

    # ------------------------------------------------------------------
    # Stage 5: report
    # ------------------------------------------------------------------

    def report(self, path: Optional[str] = None, fmt: str = 'html') -> str:
        """Generate an end-to-end report and optionally write to disk.

        Parameters
        ----------
        path : str, optional
            Output path.  If omitted, only returns the string.
        fmt : str
            One of 'html' (default) or 'markdown'.

        Returns
        -------
        str
            The report content.
        """
        # Ensure all stages have run
        if self.diagnostics is None:
            self.diagnose()
        if self.recommendation is None:
            self.recommend()
        if self.result is None:
            self.estimate()
        if not self.robustness_findings:
            self.robustness()

        if fmt == 'markdown':
            content = self._render_markdown()
        elif fmt == 'html':
            content = self._render_html()
        else:
            raise ValueError(f"Unknown fmt: {fmt!r}. Use 'html' or 'markdown'.")

        if path is not None:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        return content

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# Causal Analysis Report")
        lines.append("")
        lines.append(f"- Outcome: `{self.y}`")
        if self.treatment:
            lines.append(f"- Treatment: `{self.treatment}`")
        lines.append(f"- Design: `{self.design}`")
        lines.append(f"- N obs: {len(self.data):,}")
        lines.append("")

        lines.append("## 1. Identification diagnostics")
        lines.append("")
        lines.append(f"**Verdict: {self.diagnostics.verdict}**")
        lines.append("")
        if self.diagnostics.findings:
            for f in self.diagnostics.findings:
                lines.append(f"- [{f.severity.upper()}] "
                             f"*{f.category}* — {f.message}")
                if f.suggestion:
                    lines.append(f"    - Fix: {f.suggestion}")
        else:
            lines.append("No issues detected.")
        lines.append("")

        lines.append("## 2. Recommended estimator")
        lines.append("")
        top = (self.recommendation.recommendations[0]
               if self.recommendation.recommendations else None)
        if top:
            lines.append(f"- **Method**: {top['method']}")
            lines.append(f"- **Function**: `sp.{top['function']}()`")
            lines.append(f"- **Rationale**: {top['reason']}")
            if top.get('assumptions'):
                lines.append("- **Key assumptions**: "
                             + ", ".join(top['assumptions']))
        lines.append("")

        lines.append("## 3. Main estimate")
        lines.append("")
        r = self.result
        if hasattr(r, 'estimate') and hasattr(r, 'se'):
            stars = ''
            pv = getattr(r, 'pvalue', np.nan)
            if pd.notna(pv):
                if pv < 0.01: stars = '***'
                elif pv < 0.05: stars = '**'
                elif pv < 0.1: stars = '*'
            lines.append(f"- **{getattr(r, 'estimand', 'Effect')}**: "
                         f"{r.estimate:.4f} {stars}")
            lines.append(f"- **SE**: {r.se:.4f}")
            if hasattr(r, 'ci'):
                lines.append(f"- **95% CI**: "
                             f"[{r.ci[0]:.4f}, {r.ci[1]:.4f}]")
            lines.append(f"- **p-value**: {pv:.4f}")
        elif hasattr(r, 'params'):
            main = self.treatment or r.params.index[0]
            if main in r.params.index:
                lines.append(f"- **{main}**: {r.params[main]:.4f}")
                lines.append(f"- **SE**: {r.std_errors[main]:.4f}")
        lines.append("")

        lines.append("## 4. Robustness")
        lines.append("")
        if self.robustness_findings:
            for k, v in self.robustness_findings.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    lines.append(f"- {k.replace('_', ' ').title()}: "
                                 f"{float(v):.4f}")
                elif isinstance(v, dict):
                    lines.append(f"- {k.replace('_', ' ').title()}:")
                    for kk, vv in v.items():
                        lines.append(f"    - {kk}: {vv}")
                else:
                    lines.append(f"- {k.replace('_', ' ').title()}: {v}")
        else:
            lines.append("No robustness findings.")
        lines.append("")

        if isinstance(self.estimator_comparison, pd.DataFrame) \
                and not self.estimator_comparison.empty:
            lines.append("## 4b. Multi-estimator comparison")
            lines.append("")
            lines.append(self.estimator_comparison.round(4).to_markdown(index=False))
            lines.append("")

        if isinstance(self.sensitivity_panel_result, pd.DataFrame) \
                and not self.sensitivity_panel_result.empty:
            lines.append("## 4c. Sensitivity triad")
            lines.append("")
            lines.append(self.sensitivity_panel_result.round(4).to_markdown(index=False))
            lines.append("")

        if isinstance(self.cate_summary_table, pd.DataFrame) \
                and not self.cate_summary_table.empty:
            lines.append("## 4d. Heterogeneity (CATE)")
            lines.append("")
            lines.append(self.cate_summary_table.round(4).to_markdown(index=False))
            lines.append("")

        lines.append("## 5. Reproducibility")
        lines.append("")
        if top:
            lines.append("```python")
            lines.append(top.get('code', '# (see recommendation.code)'))
            lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("Generated by `sp.causal(...).report()`.")
        return "\n".join(lines)

    def _render_html(self) -> str:
        md = self._render_markdown()
        # Minimal markdown -> html conversion (no external deps).
        # Good enough for the one-page summary use-case.
        lines = md.split("\n")
        out: List[str] = [
            "<!DOCTYPE html>", "<html lang='en'>", "<head>",
            "<meta charset='utf-8'>",
            "<title>Causal Analysis Report</title>",
            "<style>",
            "body{font-family:-apple-system,'Helvetica Neue',Arial,sans-serif;"
            "max-width:860px;margin:40px auto;padding:0 20px;color:#1a1a2e;"
            "line-height:1.55}",
            "h1{border-bottom:2px solid #1a1a2e;padding-bottom:8px}",
            "h2{border-bottom:1px solid #E5E7EB;padding-bottom:4px;"
            "margin-top:32px;color:#2C3E50}",
            "ul{padding-left:24px}",
            "li{margin:4px 0}",
            "code{background:#F3F4F6;padding:2px 6px;border-radius:3px;"
            "font-family:'SF Mono',Menlo,Consolas,monospace;font-size:0.92em}",
            "pre{background:#1a1a2e;color:#F3F4F6;padding:16px;"
            "border-radius:6px;overflow-x:auto}",
            "pre code{background:transparent;color:inherit}",
            "strong{color:#1a1a2e}",
            "hr{border:none;border-top:1px solid #E5E7EB;margin:32px 0}",
            "</style>", "</head>", "<body>",
        ]
        in_list = False
        in_code = False
        for ln in lines:
            if ln.startswith("```"):
                if in_code:
                    out.append("</code></pre>")
                    in_code = False
                else:
                    out.append("<pre><code>")
                    in_code = True
                continue
            if in_code:
                out.append(escape(ln))
                continue
            if ln.startswith("# "):
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append(f"<h1>{escape(ln[2:])}</h1>")
            elif ln.startswith("## "):
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append(f"<h2>{escape(ln[3:])}</h2>")
            elif ln.startswith("- ") or ln.startswith("    - "):
                if not in_list:
                    out.append("<ul>")
                    in_list = True
                depth = 1 if ln.startswith("    - ") else 0
                text = ln.lstrip("- ").lstrip()
                text = _inline_md(text)
                indent_prefix = "    " if depth else ""
                out.append(f"{indent_prefix}<li>{text}</li>")
            elif ln.strip() == "---":
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append("<hr>")
            elif ln.strip() == "":
                if in_list:
                    out.append("</ul>")
                    in_list = False
            else:
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append(f"<p>{_inline_md(ln)}</p>")
        if in_list:
            out.append("</ul>")
        out.extend(["</body>", "</html>"])
        return "\n".join(out)

    # ------------------------------------------------------------------
    # Stage 4b: cross-estimator comparison
    # ------------------------------------------------------------------

    estimator_comparison: Optional[pd.DataFrame] = None
    sensitivity_panel_result: Optional[pd.DataFrame] = None
    cate_summary_table: Optional[pd.DataFrame] = None

    def compare_estimators(self) -> pd.DataFrame:
        """Run a design-appropriate panel of estimators for robustness.

        For DiD: CS + SA + BJS + Wooldridge (+ Stacked when feasible).
        For IV:  2SLS + LIML + JIVE + GMM.
        For observational:  OLS + Entropy Balancing + CBPS + SBW + DML.
        For RD:  Sharp RDD (rdrobust) at MSE-optimal + CERD ±50% bandwidths.

        Rows with a failed estimator carry NaN and an error string.
        """
        if self.result is None:
            self.estimate()

        import statspai as sp
        rows: List[Dict[str, Any]] = []
        d = self.design

        def _row(label, estimate, se, ci=None, extra=""):
            z = 1.96
            if ci is None and se == se and not np.isnan(se):
                ci = (estimate - z * se, estimate + z * se)
            elif ci is None:
                ci = (np.nan, np.nan)
            rows.append({
                "estimator": label,
                "estimate": float(estimate) if estimate is not None else np.nan,
                "se": float(se) if (se is not None and se == se) else np.nan,
                "ci_lower": float(ci[0]) if ci else np.nan,
                "ci_upper": float(ci[1]) if ci else np.nan,
                "note": extra,
            })

        def _extract_effect(r):
            """Pull estimate + SE whether r is CausalResult or EconometricResults."""
            # Preferred: CausalResult
            if hasattr(r, "estimate") and r.estimate is not None \
                    and not (isinstance(r.estimate, float) and np.isnan(r.estimate)):
                return (float(r.estimate),
                        float(getattr(r, "se", np.nan)),
                        getattr(r, "ci", None))
            # Fallback: EconometricResults with params / std_errors
            if hasattr(r, "params") and self.treatment is not None:
                try:
                    idx = r.params.index
                    if self.treatment in idx:
                        key = self.treatment
                    else:
                        # find a match that contains the treatment name
                        cand = [k for k in idx if self.treatment in str(k)]
                        if not cand:
                            # Treatment column genuinely missing — don't
                            # silently report the intercept as "the effect".
                            return (np.nan, np.nan, None)
                        key = cand[0]
                    est = float(r.params[key])
                    se = float(r.std_errors[key]) if hasattr(r, "std_errors") else np.nan
                    ci_arr = getattr(r, "conf_int", None)
                    if callable(ci_arr):
                        try:
                            ci_df = ci_arr()
                            ci = (float(ci_df.loc[key, 0]), float(ci_df.loc[key, 1]))
                        except Exception:
                            ci = None
                    else:
                        ci = None
                    return est, se, ci
                except Exception:
                    pass
            return (np.nan, np.nan, None)

        def _safe_call(label, fn):
            try:
                r = fn()
                est, se, ci = _extract_effect(r)
                _row(label, est, se, ci)
            except Exception as exc:
                _row(label, np.nan, np.nan, extra=f"ERROR: {type(exc).__name__}: {exc}")

        if d == "did" and self.time and self.id and self.cohort:
            _safe_call("Callaway–Sant'Anna (CS)", lambda: sp.callaway_santanna(
                self.data, y=self.y, g=self.cohort, t=self.time, i=self.id,
                estimator="reg"))
            _safe_call("Sun–Abraham", lambda: sp.sun_abraham(
                self.data, y=self.y, g=self.cohort, t=self.time, i=self.id))
            _safe_call("Borusyak–Jaravel–Spiess (imputation)",
                       lambda: sp.did_imputation(
                           self.data, y=self.y, g=self.cohort,
                           t=self.time, i=self.id))
            _safe_call("Wooldridge (2021)", lambda: sp.wooldridge_did(
                self.data, y=self.y, g=self.cohort, t=self.time, i=self.id))
        elif d == "did":
            _safe_call("2x2 DiD",
                       lambda: sp.did(self.data, y=self.y,
                                      treat=self.treatment, time=self.time))
        elif d == "iv" and self.instrument and self.treatment:
            formula = f"{self.y} ~ ({self.treatment} ~ {self.instrument})"
            if self.covariates:
                formula += " + " + " + ".join(self.covariates)
            _safe_call("2SLS", lambda: sp.ivreg(formula, data=self.data,
                                                robust="hc1"))
            _safe_call("LIML", lambda: sp.liml(formula, data=self.data))
        elif d == "rd" and self.running_var:
            c = self.cutoff or 0.0
            _safe_call("RDD (MSE-optimal)",
                       lambda: sp.rdrobust(self.data, y=self.y,
                                           x=self.running_var, c=c))
        elif d in ("observational", "rct") and self.treatment and self.covariates:
            cov = list(self.covariates)
            # OLS w/ controls
            rhs = self.treatment + " + " + " + ".join(cov)
            _safe_call("OLS (HC1)", lambda: sp.regress(
                f"{self.y} ~ {rhs}", data=self.data, robust="hc1"))
            _safe_call("Entropy balancing (ATT)", lambda: sp.ebalance(
                self.data, y=self.y, treat=self.treatment, covariates=cov))
            _safe_call("CBPS", lambda: sp.cbps(
                self.data, y=self.y, treat=self.treatment, covariates=cov))
            _safe_call("SBW (δ=0.02)", lambda: sp.sbw(
                self.data, y=self.y, treat=self.treatment, covariates=cov,
                delta=0.02))
            # DML (partially-linear)
            _safe_call("DML-PLR", lambda: sp.dml(
                self.data, y=self.y, d=self.treatment, X=cov))

        # Always add the primary estimate for context
        if self.result is not None and hasattr(self.result, "estimate"):
            _row("→ primary (this workflow)", self.result.estimate,
                 getattr(self.result, "se", np.nan),
                 getattr(self.result, "ci", None), extra="primary")

        self.estimator_comparison = pd.DataFrame(rows)
        self._mark("compare_estimators")
        return self.estimator_comparison

    # ------------------------------------------------------------------
    # Stage 4c: sensitivity-analysis triad
    # ------------------------------------------------------------------

    def sensitivity_panel(self) -> pd.DataFrame:
        """Stack E-value + Rosenbaum Γ + Oster δ for one-page sensitivity.

        Only runs the tests whose required inputs are present:
        * E-value requires a risk-ratio or the primary estimate + SE.
        * Rosenbaum Γ requires a matched/weighted comparison.
        * Oster δ requires OLS residual variances.
        """
        if self.result is None:
            self.estimate()
        import statspai as sp

        rows: List[Dict[str, Any]] = []

        # ── E-value ──────────────────────────────────────────────────
        try:
            est, se = None, None
            r = self.result
            if hasattr(r, "estimate") and r.estimate is not None \
                    and not (isinstance(r.estimate, float) and np.isnan(r.estimate)):
                est = float(r.estimate)
                se = float(getattr(r, "se", np.nan))
            elif hasattr(r, "params") and self.treatment \
                    and self.treatment in r.params.index:
                est = float(r.params[self.treatment])
                se = float(r.std_errors[self.treatment])
            if est is not None and se is not None and se > 0:
                ev = sp.evalue(estimate=est, se=se, measure="RR",
                               rare_outcome=True)
                val = None
                if isinstance(ev, dict):
                    val = (ev.get("evalue") or ev.get("evalue_estimate")
                           or ev.get("evalue_point"))
                else:
                    val = getattr(ev, "evalue_estimate",
                                  getattr(ev, "evalue", None))
                if val is not None:
                    rows.append({
                        "method": "E-value (VanderWeele–Ding 2017)",
                        "statistic": float(val),
                        "interpretation":
                            "unmeasured confounder must relate to both T and Y "
                            f"on the RR scale ≥ {float(val):.2f} to null out "
                            "the effect",
                    })
        except Exception as exc:  # pragma: no cover
            rows.append({"method": "E-value", "statistic": np.nan,
                         "interpretation": f"ERROR: {exc}"})

        # ── Oster δ (requires OLS-style result with R²) ───────────────
        if self.treatment and self.covariates:
            try:
                ob = sp.oster_bounds(
                    data=self.data, y=self.y, treat=self.treatment,
                    controls=list(self.covariates),
                )
                # The critical δ is delta_for_zero (δ* that drives the
                # coefficient to zero). `delta` is the plug-in value the
                # caller ran Oster at (defaults to 1.0 in sp.oster_bounds).
                delta_star = None
                if isinstance(ob, dict):
                    delta_star = (ob.get("delta_for_zero")
                                  or ob.get("delta_star"))
                else:
                    delta_star = getattr(ob, "delta_for_zero",
                                         getattr(ob, "delta_star", None))
                if delta_star is not None:
                    rows.append({
                        "method": "Oster δ* (2019 JBES)",
                        "statistic": float(delta_star),
                        "interpretation":
                            f"selection on unobservables must be "
                            f"{float(delta_star):.2f}× selection on observables "
                            "to drive β to zero",
                    })
            except Exception as exc:  # pragma: no cover
                rows.append({"method": "Oster δ", "statistic": np.nan,
                             "interpretation": f"ERROR: {exc}"})

        # ── Rosenbaum Γ (requires matched/weighted design) ────────────
        try:
            from ..diagnostics.rosenbaum import rosenbaum_gamma
            d = self.design
            if d in ("observational",) and self.treatment and self.covariates:
                gamma = rosenbaum_gamma(
                    self.data, y=self.y, treat=self.treatment,
                    covariates=list(self.covariates),
                )
                val = getattr(gamma, "gamma", None)
                if val is None and isinstance(gamma, dict):
                    val = gamma.get("gamma")
                if val is not None:
                    rows.append({
                        "method": "Rosenbaum Γ (bound)",
                        "statistic": float(val),
                        "interpretation":
                            f"odds-of-treatment differential ≥ {float(val):.2f} "
                            "flips the conclusion",
                    })
        except Exception:  # pragma: no cover
            pass

        self.sensitivity_panel_result = pd.DataFrame(rows)
        self._mark("sensitivity_panel")
        return self.sensitivity_panel_result

    # ------------------------------------------------------------------
    # Stage 4d: heterogeneity / CATE pass
    # ------------------------------------------------------------------

    def cate(self) -> pd.DataFrame:
        """Quick heterogeneity pass: X-Learner + Causal Forest summary.

        Skipped if the design is IV / RD / DiD with only the canonical
        columns and no covariates — heterogeneity needs an X matrix.
        """
        if self.result is None:
            self.estimate()
        if not (self.treatment and self.covariates):
            self.cate_summary_table = pd.DataFrame()
            self._mark("cate")
            return self.cate_summary_table

        import statspai as sp
        rows: List[Dict[str, Any]] = []

        def _summarise(label, result):
            # Extract per-unit CATE from any of several locations
            cate_vec = None
            try:
                if hasattr(result, "effect") and callable(result.effect):
                    cate_vec = result.effect()
                elif hasattr(result, "cate"):
                    cate_vec = result.cate
                elif hasattr(result, "model_info"):
                    cate_vec = result.model_info.get("cate")
            except Exception:
                cate_vec = None
            if cate_vec is None:
                # Fall back to the summary ATE alone
                if hasattr(result, "estimate"):
                    rows.append({
                        "learner": label,
                        "cate_mean": float(result.estimate),
                        "cate_sd": float("nan"),
                        "cate_q10": float("nan"),
                        "cate_q50": float("nan"),
                        "cate_q90": float("nan"),
                    })
                return
            cate_vec = np.asarray(cate_vec, dtype=float).ravel()
            rows.append({
                "learner": label,
                "cate_mean": float(np.mean(cate_vec)),
                "cate_sd": float(np.std(cate_vec)),
                "cate_q10": float(np.quantile(cate_vec, 0.10)),
                "cate_q50": float(np.quantile(cate_vec, 0.50)),
                "cate_q90": float(np.quantile(cate_vec, 0.90)),
            })

        try:
            xl = sp.xlearner(self.data, y=self.y, d=self.treatment,
                             X=list(self.covariates))
            _summarise("X-Learner", xl)
        except Exception as exc:  # pragma: no cover
            rows.append({"learner": "X-Learner", "cate_mean": np.nan,
                         "error": f"{type(exc).__name__}: {exc}"})

        try:
            Y_arr = self.data[self.y].values.astype(float)
            T_arr = self.data[self.treatment].values.astype(float)
            X_arr = self.data[list(self.covariates)].values.astype(float)
            cf = sp.causal_forest(Y=Y_arr, T=T_arr, X=X_arr,
                                  n_estimators=200, random_state=0)
            cate_vec = cf.effect(X_arr) if hasattr(cf, "effect") else None
            if cate_vec is None:
                cate_vec = getattr(cf, "cate", None)
            if cate_vec is not None:
                cate_vec = np.asarray(cate_vec, dtype=float).ravel()
                rows.append({
                    "learner": "Causal Forest",
                    "cate_mean": float(np.mean(cate_vec)),
                    "cate_sd": float(np.std(cate_vec)),
                    "cate_q10": float(np.quantile(cate_vec, 0.10)),
                    "cate_q50": float(np.quantile(cate_vec, 0.50)),
                    "cate_q90": float(np.quantile(cate_vec, 0.90)),
                })
        except Exception as exc:  # pragma: no cover
            rows.append({"learner": "Causal Forest", "cate_mean": np.nan,
                         "error": f"{type(exc).__name__}: {exc}"})

        self.cate_summary_table = pd.DataFrame(rows)
        self._mark("cate")
        return self.cate_summary_table

    # ------------------------------------------------------------------
    # Orchestration entry point
    # ------------------------------------------------------------------

    def run(self, full: bool = True):
        """Run the causal pipeline.

        Parameters
        ----------
        full : bool, default True
            When True, also runs the v0.9.17 extended stages:
            ``compare_estimators``, ``sensitivity_panel``, and ``cate``.
            Set to False for a quick single-estimator pass.
        """
        self.diagnose()
        self.recommend()
        self.estimate()
        self.robustness()
        if full:
            try:
                self.compare_estimators()
            except Exception:  # pragma: no cover
                pass
            try:
                self.sensitivity_panel()
            except Exception:  # pragma: no cover
                pass
            try:
                self.cate()
            except Exception:  # pragma: no cover
                pass
        return self

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def _mark(self, stage: str):
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)

    def __repr__(self) -> str:
        done = ",".join(self.stages_completed) or 'not-started'
        est = ''
        if self.result is not None and hasattr(self.result, 'estimate'):
            est = f" est={self.result.estimate:.4f}"
        return (f"<CausalWorkflow design={self.design} "
                f"stages=[{done}]{est}>")


# ---------------------------------------------------------------------------
# Inline-markdown helper
# ---------------------------------------------------------------------------

def _inline_md(text: str) -> str:
    """Minimal inline markdown (bold, code) to HTML."""
    text = escape(text)
    # **bold**
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # `code`
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def causal(
    data: pd.DataFrame,
    y: str,
    treatment: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    id: Optional[str] = None,
    time: Optional[str] = None,
    running_var: Optional[str] = None,
    instrument: Optional[str] = None,
    cutoff: Optional[float] = None,
    cohort: Optional[str] = None,
    cluster: Optional[str] = None,
    design: Optional[str] = None,
    dag=None,
    strict: bool = False,
    auto_run: bool = True,
    # --- Sprint-B causal-method context (all opt-in). When supplied,
    # these parameters extend the recommender's candidate set and
    # the fallback-estimator dispatch to msm / proximal / principal_strat
    # / mediate / mediate_interventional / front_door.
    mediator: Optional[str] = None,
    tv_confounders: Optional[List[str]] = None,
    proxy_z: Optional[List[str]] = None,
    proxy_w: Optional[List[str]] = None,
    post_treat_strata: Optional[str] = None,
) -> CausalWorkflow:
    """End-to-end causal-inference workflow.

    One call that diagnoses identification, picks an estimator, fits
    it, runs the canonical robustness suite, and produces a report.

    **Unique to StatsPAI** — no other Python/R/Stata package ships
    this orchestration.

    Parameters
    ----------
    data, y, treatment, covariates, id, time, running_var, instrument,
    cutoff, cohort, cluster, design, dag, strict
        Passed through to ``sp.check_identification`` and ``sp.recommend``.
    auto_run : bool, default True
        If True (default), immediately runs all 5 stages and returns
        the fully-populated workflow.  If False, returns the workflow
        object with no stages executed — call ``.diagnose()``,
        ``.estimate()``, etc. manually for finer control.

    Returns
    -------
    CausalWorkflow
        A workflow object with ``.diagnostics``, ``.recommendation``,
        ``.result``, ``.robustness_findings``, and ``.report()``.

    Examples
    --------
    One-call full analysis:

    >>> import statspai as sp
    >>> w = sp.causal(df, y='wage', treatment='training',
    ...               id='worker', time='year', design='did')
    >>> w.report('analysis.html')

    Fine-grained control:

    >>> w = sp.causal(df, y='y', treatment='d', auto_run=False)
    >>> w.diagnose()        # -> IdentificationReport
    >>> if w.diagnostics.verdict == 'BLOCKERS':
    ...     raise SystemExit(1)
    >>> w.estimate()
    >>> print(w.report(fmt='markdown'))
    """
    workflow = CausalWorkflow(
        data=data,
        y=y,
        treatment=treatment,
        covariates=covariates or [],
        id=id,
        time=time,
        running_var=running_var,
        instrument=instrument,
        cutoff=cutoff,
        cohort=cohort,
        cluster=cluster,
        design=design,
        dag=dag,
        strict=strict,
        mediator=mediator,
        tv_confounders=list(tv_confounders) if tv_confounders else None,
        proxy_z=list(proxy_z) if proxy_z else None,
        proxy_w=list(proxy_w) if proxy_w else None,
        post_treat_strata=post_treat_strata,
    )
    if auto_run:
        workflow.run()
    return workflow
