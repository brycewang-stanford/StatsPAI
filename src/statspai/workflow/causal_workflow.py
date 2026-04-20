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
        """
        if self.recommendation is None:
            self.recommend()

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
        """Run design-appropriate robustness checks on the fitted result."""
        if self.result is None:
            self.estimate()

        findings: Dict[str, Any] = {}

        # Design-specific robustness
        d = self.design
        try:
            if d == 'did' and hasattr(self.result, 'model_info'):
                if 'pretrend_test' in self.result.model_info:
                    pt = self.result.model_info['pretrend_test']
                    findings['pretrend_test'] = pt
            if d == 'iv' and hasattr(self.result, 'diagnostics'):
                keys = [k for k in self.result.diagnostics
                        if 'first' in k.lower() and 'f' in k.lower()]
                for k in keys:
                    findings[k] = self.result.diagnostics[k]
        except Exception:
            pass

        # Generic: always report confidence interval width
        if self.result is not None:
            try:
                if hasattr(self.result, 'ci'):
                    lo, hi = self.result.ci
                    findings['ci_width'] = float(hi - lo)
                    findings['estimate'] = float(self.result.estimate)
                elif hasattr(self.result, 'params'):
                    main = (self.treatment or self.result.params.index[0])
                    if main in self.result.params.index:
                        coef = float(self.result.params[main])
                        se = float(self.result.std_errors[main])
                        findings['estimate'] = coef
                        findings['ci_width'] = 2 * 1.96 * se
            except Exception:
                pass

        # E-value (if applicable & available)
        try:
            import statspai as sp
            if (d in ('observational', 'did') and
                    hasattr(self.result, 'estimate')):
                ev = sp.evalue_from_result(self.result)
                if hasattr(ev, 'evalue'):
                    findings['evalue'] = float(ev.evalue)
        except Exception:
            pass

        self.robustness_findings = findings
        self._mark('robustness')
        return findings

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
    # Orchestration entry point
    # ------------------------------------------------------------------

    def run(self):
        """Run all 5 stages in order and return self."""
        self.diagnose()
        self.recommend()
        self.estimate()
        self.robustness()
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
    )
    if auto_run:
        workflow.run()
    return workflow
