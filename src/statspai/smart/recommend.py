"""
Estimator Recommendation Engine.

Given a research question, data structure, and optionally a causal DAG,
recommends the appropriate estimator(s) with reasoning.

**No other econometrics package does this.**

This is the "smart advisor" that bridges the gap between knowing your
research question and knowing which estimator to use.

Usage
-----
>>> import statspai as sp
>>> rec = sp.recommend(
...     data=df, y='wage', treatment='training',
...     design='observational',  # or 'rct', 'panel', 'iv', 'rd', 'did'
...     dag=my_dag,  # optional
... )
>>> print(rec.summary())
>>> result = rec.run()  # execute the recommended estimator
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd


class RecommendationResult:
    """Result from the estimator recommendation engine."""

    def __init__(self, recommendations, data_profile, design,
                 warnings, data, y, treatment):
        self.recommendations = recommendations  # list of dicts
        self.data_profile = data_profile
        self.design = design
        self.warnings = warnings
        self._data = data
        self._y = y
        self._treatment = treatment

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "StatsPAI Estimator Recommendation",
            "=" * 70,
            "",
            "DATA PROFILE",
            f"  N obs:       {self.data_profile['n_obs']:,}",
            f"  Variables:   {self.data_profile['n_vars']}",
            f"  Outcome:     {self._y} ({self.data_profile['y_type']})",
            f"  Treatment:   {self._treatment} ({self.data_profile['treat_type']})",
        ]
        if self.data_profile.get('panel'):
            lines.append(f"  Panel:       {self.data_profile['n_units']} units × "
                         f"{self.data_profile['n_periods']} periods")
        if self.data_profile.get('missing_pct', 0) > 0:
            lines.append(f"  Missing:     {self.data_profile['missing_pct']:.1%}")

        lines.append(f"\n  Design:      {self.design.upper()}")

        if self.warnings:
            lines.append("\n⚠ WARNINGS")
            for w in self.warnings:
                lines.append(f"  • {w}")

        lines.append(f"\n{'─' * 70}")
        lines.append("RECOMMENDED ESTIMATORS (ranked by appropriateness)")
        lines.append(f"{'─' * 70}")

        for i, rec in enumerate(self.recommendations):
            star = "★" if i == 0 else "○"
            lines.append(f"\n  {star} #{i+1}: {rec['method']}")
            lines.append(f"    Function: sp.{rec['function']}()")
            lines.append(f"    Why: {rec['reason']}")
            if rec.get('assumptions'):
                lines.append(f"    Assumptions: {', '.join(rec['assumptions'])}")
            if rec.get('robustness'):
                lines.append(f"    Robustness: {rec['robustness']}")
            if rec.get('code'):
                lines.append(f"    Code: {rec['code']}")
            v = rec.get('verify')
            if v:
                if v.get('error'):
                    lines.append(f"    Stability: skipped ({v['error']})")
                elif np.isfinite(v.get('score', np.nan)):
                    stab = v.get('stability', {}).get('score', np.nan)
                    plac = v.get('placebo', {}).get('score', np.nan)
                    subs = v.get('subsample', {}).get('score', np.nan)
                    lines.append(
                        f"    Stability: score={v['score']:.0f}/100  "
                        f"(resample={stab:.0f}, placebo={plac:.0f}, "
                        f"subsample={subs:.0f}, B={v.get('B_used','?')}, "
                        f"{v.get('elapsed_s',0):.1f}s)  "
                        f"[measures resampling stability, NOT identification validity]"
                    )

        lines.append(f"\n{'─' * 70}")
        lines.append("SUGGESTED WORKFLOW")
        lines.append(f"{'─' * 70}")
        for i, step in enumerate(self._workflow_steps()):
            lines.append(f"  {i+1}. {step}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def to_latex(self, caption: Optional[str] = None,
                 label: str = "tab:recommendation") -> str:
        r"""Export recommendations as a booktabs LaTeX table.

        If ``verify=True`` was used when calling ``recommend()``, the
        table includes the stability-check columns (composite score,
        bootstrap stability, placebo pass-rate, subsample agreement).

        IMPORTANT CAVEAT FOR AUTHORS:
        The stability score measures whether a method gives **consistent**
        estimates under resampling on the observed data — it does NOT
        establish identification validity or protect against unobserved
        confounding. A biased OLS on observational data will typically
        score high because the bias is stable across resamples.  Do not
        cite this score as evidence that a method is "correct" for a
        given design; use it only to compare the stability of methods
        that already satisfy the design's identification assumptions.

        Parameters
        ----------
        caption : str, optional
            Table caption. Defaults to the detected design.
        label : str
            LaTeX label for cross-referencing.

        Returns
        -------
        str
            LaTeX source (booktabs + threeparttable).
        """
        has_verify = any(
            isinstance(r.get("verify"), dict)
            and np.isfinite(r["verify"].get("score", np.nan))
            for r in self.recommendations
        )
        if caption is None:
            caption = (
                f"StatsPAI recommended estimators for "
                f"{self.design.replace('_', ' ')} design"
                + (" (with empirical verification)" if has_verify else "")
            )

        def _esc(s: str) -> str:
            return (str(s).replace("\\", r"\textbackslash{}")
                         .replace("_", r"\_")
                         .replace("&", r"\&")
                         .replace("%", r"\%")
                         .replace("#", r"\#"))

        if has_verify:
            header = (r"Rank & Method & Function & "
                      r"Stab.\ Score & Resample & Plac. & Subs. \\")
            col_spec = "rllrrrr"
        else:
            header = r"Rank & Method & Function & Reason \\"
            col_spec = "rllp{6cm}"

        lines = [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\begin{threeparttable}",
            rf"\caption{{{_esc(caption)}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header,
            r"\midrule",
        ]

        for i, rec in enumerate(self.recommendations, 1):
            method = _esc(rec["method"])
            func = rf"\texttt{{sp.{_esc(rec['function'])}()}}"
            if has_verify:
                v = rec.get("verify") or {}
                score = v.get("score", np.nan)
                stab = (v.get("stability") or {}).get("score", np.nan)
                plac = (v.get("placebo") or {}).get("score", np.nan)
                subs = (v.get("subsample") or {}).get("score", np.nan)
                err = v.get("error")

                def _fmt(x):
                    return f"{x:.0f}" if isinstance(x, float) and np.isfinite(x) else "--"

                marker = f" {{\\tiny\\textit{{({_esc(err)})}}}}" if err else ""
                lines.append(
                    f"{i} & {method}{marker} & {func} & "
                    f"{_fmt(score)} & {_fmt(stab)} & {_fmt(plac)} & {_fmt(subs)} \\\\"
                )
            else:
                reason = _esc(rec.get("reason", ""))
                lines.append(f"{i} & {method} & {func} & {reason} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\footnotesize",
        ])
        if has_verify:
            lines.extend([
                r"\item \textbf{Stab.\ Score}: weighted composite in [0, 100]"
                r" measuring estimate \emph{stability} under resampling;"
                r" \emph{not} a measure of identification validity.",
                r"\item \textbf{Resample}: bootstrap stability "
                r"($100 \times (1 - \text{CV})$ of point estimate across $B$ resamples).",
                r"\item \textbf{Plac.}: permutation placebo pass rate "
                r"(\% of permuted-treatment runs with $p > 0.10$). "
                r"Note: unconditional permutation destroys confounder "
                r"structure and thus has limited power for "
                r"selection-on-observables designs.",
                r"\item \textbf{Subs.}: sign agreement across 50\% subsamples.",
                r"\item \textbf{Caveat}: high stability is necessary but not"
                r" sufficient; a biased estimator can be perfectly stable.",
                r"\item Data: " + _esc(f"N={self.data_profile['n_obs']:,}") +
                (f", {self.data_profile.get('n_units', '?')} units "
                 f"$\\times$ {self.data_profile.get('n_periods', '?')} periods"
                 if self.data_profile.get('panel') else "") + ".",
            ])
        else:
            lines.append(
                r"\item Rankings are rule-based; call with "
                r"\texttt{verify=True} for empirical scores."
            )
        lines.extend([
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def _workflow_steps(self):
        """Generate a recommended workflow."""
        steps = []
        rec = self.recommendations[0] if self.recommendations else None
        if not rec:
            return ["Insufficient information to recommend"]

        # Pre-estimation
        steps.append(f"Run sp.sumstats(df) to check data quality")
        if self.data_profile.get('missing_pct', 0) > 5:
            steps.append(f"Handle missing data: sp.mice(df, m=5) — {self.data_profile['missing_pct']:.0%} missing")

        if self._treatment:
            steps.append(f"Check balance: sp.balance_check(df, treatment='{self._treatment}', "
                         f"covariates=[...])")

        # Main estimation
        steps.append(f"Estimate: result = {rec['code']}")

        # Post-estimation
        steps.append(f"Diagnostics: sp.diagnose_result(result)")

        if rec['function'] in ['regress', 'iv', 'panel']:
            steps.append(f"Sensitivity: sp.sensemakr(result) or sp.oster_bounds(result)")
        if rec['function'] in ['did', 'callaway_santanna']:
            steps.append(f"Pre-trends: sp.pretrends_test(result)")
            steps.append(f"Event study: sp.event_study(df, ...)")
        if rec['function'] == 'rdrobust':
            steps.append(f"McCrary test: sp.rddensity(df, x='running_var')")

        steps.append(f"Robustness: sp.robustness_report(result)")
        steps.append(f"Export: sp.outreg2(result, filename='results.xlsx')")

        return steps

    def run(self, which: int = 0, **kwargs):
        """Execute the recommended estimator.

        Parameters
        ----------
        which : int, default 0
            Which recommendation to run (0 = top recommendation).
        **kwargs
            Override any parameters.
        """
        import statspai as sp

        rec = self.recommendations[which]
        func = getattr(sp, rec['function'])
        params = rec.get('params', {})
        params.update(kwargs)
        return func(**params)

    def run_all(self, **kwargs):
        """Run all recommended estimators and return a comparison."""
        results = {}
        for i, rec in enumerate(self.recommendations):
            try:
                results[rec['method']] = self.run(which=i, **kwargs)
            except Exception as e:
                results[rec['method']] = f"Error: {e}"
        return results


def _profile_data(data, y, treatment, id_col, time_col):
    """Profile the dataset to understand its structure."""
    profile = {
        'n_obs': len(data),
        'n_vars': len(data.columns),
    }

    # Outcome type
    y_data = data[y].dropna()
    if y_data.nunique() == 2:
        profile['y_type'] = 'binary'
    elif pd.api.types.is_integer_dtype(y_data) and y_data.min() >= 0:
        if y_data.max() <= 50:
            profile['y_type'] = 'count'
        else:
            profile['y_type'] = 'continuous'
    elif len(y_data) > 0 and y_data.min() >= 0 and y_data.max() <= 1:
        profile['y_type'] = 'fractional'
    else:
        profile['y_type'] = 'continuous'

    # Treatment type
    if treatment:
        t_data = data[treatment]
        if t_data.nunique() == 2:
            profile['treat_type'] = 'binary'
        elif t_data.nunique() <= 10:
            profile['treat_type'] = 'categorical'
        else:
            profile['treat_type'] = 'continuous'
    else:
        profile['treat_type'] = 'none'

    # Panel structure
    if id_col and time_col:
        profile['panel'] = True
        profile['n_units'] = data[id_col].nunique()
        profile['n_periods'] = data[time_col].nunique()
        profile['balanced'] = data.groupby(id_col).size().nunique() == 1
    else:
        profile['panel'] = False

    # Missing data
    profile['missing_pct'] = data.isna().any(axis=1).mean()

    return profile


def _detect_design(data, y, treatment, id_col, time_col, running_var,
                   instrument, profile):
    """Auto-detect the likely research design."""
    if running_var:
        return 'rd'
    if instrument:
        return 'iv'
    if id_col and time_col and treatment:
        # Check if treatment varies over time → DID
        treat_varies = data.groupby(id_col)[treatment].nunique().max() > 1
        if treat_varies:
            return 'did'
        else:
            return 'panel'
    if id_col and time_col:
        return 'panel'
    if treatment:
        return 'observational'
    return 'cross-section'


def recommend(
    data: pd.DataFrame,
    y: str,
    treatment: str = None,
    covariates: List[str] = None,
    id: str = None,
    time: str = None,
    running_var: str = None,
    instrument: str = None,
    cutoff: float = None,
    design: str = None,
    dag=None,
    verify: bool = False,
    verify_B: int = 50,
    verify_budget_s: float = 30.0,
    verify_top_k: int = 3,
) -> RecommendationResult:
    """
    Recommend the appropriate estimator(s) for your research question.

    **Unique to StatsPAI** — no other econometrics package does this.

    Given your data, outcome, treatment, and research design, this function:
    1. Profiles your data (type, structure, missing patterns)
    2. Detects your research design (RCT, DID, RD, IV, observational)
    3. Recommends ranked estimators with reasoning
    4. Generates a complete workflow (pre-estimation → estimation → robustness)
    5. Provides executable code via `.run()`

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treatment : str, optional
        Treatment/exposure variable.
    covariates : list of str, optional
        Control variables.
    id : str, optional
        Unit identifier (for panel data).
    time : str, optional
        Time variable (for panel/DID).
    running_var : str, optional
        Running variable (for RD designs).
    instrument : str, optional
        Instrumental variable.
    cutoff : float, optional
        RD cutoff value.
    design : str, optional
        Override design detection: 'rct', 'did', 'rd', 'iv',
        'observational', 'panel', 'cross-section'.
    dag : DAG, optional
        Causal DAG for identification analysis.
    verify : bool, default False
        If True, run *resampling-stability* checks on the top-k
        recommendations (bootstrap CV, permutation placebo, 50%-subsample
        sign agreement) and attach a composite ``score`` to each
        recommendation's ``verify`` dict. The score is used to re-rank
        the top-k. Opt-in because it costs extra compute.

        IMPORTANT: this score measures whether estimates are **stable**
        under resampling, NOT whether the method satisfies the design's
        identification assumptions. A biased method on observational
        data can score near 100. See ``statspai.smart.verify`` docstring.
    verify_B : int, default 50
        Bootstrap replications per recommendation (auto-capped by budget).
    verify_budget_s : float, default 30.0
        Wall-clock budget (seconds) per verified recommendation.
    verify_top_k : int, default 3
        Number of top recommendations to verify.

    Returns
    -------
    RecommendationResult
        With .summary(), .run(), .run_all() methods.

    Examples
    --------
    >>> import statspai as sp
    >>> rec = sp.recommend(df, y='wage', treatment='training',
    ...                    id='worker', time='year')
    >>> print(rec.summary())  # see recommendations
    >>> result = rec.run()    # execute top recommendation
    """
    if covariates is None:
        covariates = [c for c in data.columns
                      if c not in [y, treatment, id, time, running_var, instrument]
                      and pd.api.types.is_numeric_dtype(data[c])]

    profile = _profile_data(data, y, treatment, id, time)

    if design is None:
        design = _detect_design(data, y, treatment, id, time,
                                running_var, instrument, profile)

    warnings_list = []
    recommendations = []

    # DAG-based recommendations
    dag_adjustment = None
    if dag is not None:
        try:
            adj_sets = dag.adjustment_sets(treatment, y)
            if adj_sets:
                dag_adjustment = list(adj_sets[0])
                bad = dag.bad_controls(treatment, y)
                if bad:
                    warnings_list.append(
                        f"DAG flags bad controls (do NOT include): {bad}")
        except Exception as e:
            warnings_list.append(f"DAG analysis failed: {e}")

    controls = dag_adjustment if dag_adjustment else covariates
    ctrl_str = ", ".join(f"'{c}'" for c in controls[:5])
    if len(controls) > 5:
        ctrl_str += ", ..."

    # Missing data warning
    if profile['missing_pct'] > 0.1:
        warnings_list.append(
            f"{profile['missing_pct']:.0%} observations have missing values. "
            f"Consider sp.mice() before estimation.")

    # ===== DESIGN-SPECIFIC RECOMMENDATIONS =====

    if design == 'rct':
        recommendations.append({
            'method': 'OLS with robust SE (primary)',
            'function': 'regress',
            'reason': 'RCT: simple difference in means is unbiased. '
                      'Add covariates for precision.',
            'assumptions': ['Random assignment', 'SUTVA', 'No attrition bias'],
            'robustness': 'Check sp.balance_check() and sp.attrition_test()',
            'code': f"sp.regress('{y} ~ {treatment} + {ctrl_str}', data=df, robust='hc1')",
            'params': {'formula': f'{y} ~ {treatment}', 'data': data, 'robust': 'hc1'},
        })
        if profile['y_type'] == 'binary':
            recommendations.append({
                'method': 'Logit (for binary outcome)',
                'function': 'logit',
                'reason': 'Binary outcome → logit for correct functional form.',
                'code': f"sp.logit(data=df, y='{y}', x=['{treatment}'] + controls)",
                'params': {'data': data, 'y': y, 'x': [treatment] + controls[:5]},
            })

    elif design == 'did':
        # Staggered vs 2-period
        if time and data[time].nunique() > 2:
            cohort_col = f"_cohort_{treatment}"

            def _derive_cohort(df_in, _treat=treatment, _id=id, _time=time,
                                _col=cohort_col):
                """Attach cohort column = first treated period per unit."""
                out = df_in.copy()
                if _id and _id in out.columns:
                    treated = out[out[_treat] == 1]
                    cmap = treated.groupby(_id)[_time].min()
                    out[_col] = out[_id].map(cmap).fillna(0).astype(int)
                else:
                    out[_col] = 0
                return out

            did_data = _derive_cohort(data)
            recommendations.append({
                'method': 'Callaway-Sant\'Anna (2021) — staggered DID',
                'function': 'callaway_santanna',
                'reason': 'Multiple time periods with staggered treatment adoption. '
                          'Robust to heterogeneous treatment effects (unlike TWFE).',
                'assumptions': ['Parallel trends', 'No anticipation', 'Staggered adoption'],
                'robustness': 'Run sp.pretrends_test(), sp.honest_did(), sp.event_study()',
                'code': f"# Derived cohort column = first period treated\n"
                        f"sp.callaway_santanna(df, y='{y}', g='{cohort_col}', "
                        f"t='{time}', i='{id}')",
                'params': {'data': did_data, 'y': y, 'g': cohort_col,
                           't': time, 'i': id},
                'prep': _derive_cohort,
                'raw_treat': treatment,
            })
            recommendations.append({
                'method': 'Sun-Abraham (2021) — interaction-weighted',
                'function': 'sun_abraham',
                'reason': 'Alternative heterogeneity-robust DID estimator.',
                'code': f"sp.sun_abraham(df, y='{y}', g='{cohort_col}', "
                        f"t='{time}', i='{id}')",
                'params': {'data': did_data, 'y': y, 'g': cohort_col,
                           't': time, 'i': id},
                'prep': _derive_cohort,
                'raw_treat': treatment,
            })
        else:
            recommendations.append({
                'method': 'Classic 2×2 DID',
                'function': 'did',
                'reason': 'Two groups, two periods — classic DID is appropriate.',
                'assumptions': ['Parallel trends', 'No anticipation', 'SUTVA'],
                'code': f"sp.did(df, y='{y}', treat='{treatment}', time='{time}')",
                'params': {'data': data, 'y': y, 'treat': treatment, 'time': time},
            })

    elif design == 'rd':
        rv = running_var or 'running_var'
        c = cutoff or 0
        recommendations.append({
            'method': 'Local polynomial RD (CCT 2014)',
            'function': 'rdrobust',
            'reason': 'Sharp/fuzzy RD with MSE-optimal bandwidth and bias correction.',
            'assumptions': ['Continuity of potential outcomes at cutoff',
                            'No manipulation of running variable'],
            'robustness': 'Run sp.rddensity(), sp.rdbwsensitivity(), sp.rdplacebo()',
            'code': f"sp.rdrobust(df, y='{y}', x='{rv}', c={c})",
            'params': {'data': data, 'y': y, 'x': rv, 'c': c},
        })

    elif design == 'iv':
        z = instrument or 'instrument'
        exog_controls = [c for c in controls if c not in (treatment, z)]
        exog_str = " + ".join(exog_controls[:5]) if exog_controls else ""
        iv_formula = (
            f"{y} ~ {exog_str} + ({treatment} ~ {z})"
            if exog_str else f"{y} ~ ({treatment} ~ {z})"
        )
        recommendations.append({
            'method': '2SLS (two-stage least squares)',
            'function': 'ivreg',
            'reason': 'Standard IV estimator for endogenous treatment.',
            'assumptions': ['Instrument relevance (F > 10)',
                            'Exclusion restriction', 'Monotonicity (for LATE)'],
            'robustness': 'Check first-stage F, sp.anderson_rubin_test(), sp.kitagawa_test()',
            'code': f"sp.ivreg('{iv_formula}', data=df, robust='hc1')",
            'params': {'formula': iv_formula, 'data': data, 'robust': 'hc1'},
        })
        recommendations.append({
            'method': 'LIML (robust to weak instruments)',
            'function': 'liml',
            'reason': 'Less biased than 2SLS when instruments are weak.',
            'code': f"sp.liml(data=df, y='{y}', x_endog=['{treatment}'], z=['{z}'])",
            'params': {'data': data, 'y': y, 'x_endog': [treatment], 'z': [z]},
        })

    elif design == 'observational':
        recommendations.append({
            'method': 'OLS with robust SE (baseline)',
            'function': 'regress',
            'reason': 'Start with OLS as baseline. If endogeneity is a concern, '
                      'follow up with matching or IV.',
            'assumptions': ['E[ε|X]=0 (exogeneity)', 'Correct functional form'],
            'robustness': 'Run sp.sensemakr(), sp.oster_bounds(), sp.spec_curve()',
            'code': f"sp.regress('{y} ~ {treatment} + {ctrl_str}', data=df, robust='hc1')",
            'params': {'formula': f'{y} ~ {treatment}', 'data': data, 'robust': 'hc1'},
        })
        recommendations.append({
            'method': 'Propensity Score Matching (selection on observables)',
            'function': 'match',
            'reason': 'Nonparametric causal effect under unconfoundedness.',
            'assumptions': ['Unconfoundedness (CIA)', 'Common support (overlap)'],
            'code': f"sp.match(df, y='{y}', treat='{treatment}', "
                    f"covariates=[{ctrl_str}])",
            'params': {'data': data, 'y': y, 'treat': treatment,
                       'covariates': controls[:10]},
        })
        recommendations.append({
            'method': 'Double ML (high-dimensional controls)',
            'function': 'dml',
            'reason': 'Handles many controls without overfitting via cross-fitting.',
            'code': f"sp.dml(df, y='{y}', treat='{treatment}', "
                    f"covariates=[{ctrl_str}])",
            'params': {'data': data, 'y': y, 'treat': treatment,
                       'covariates': controls[:20]},
        })

    elif design == 'panel':
        panel_rhs = treatment if treatment else '1'
        panel_controls = [c for c in controls if c != treatment][:5]
        if panel_controls:
            panel_rhs += " + " + " + ".join(panel_controls)
        panel_formula = f"{y} ~ {panel_rhs}"
        recommendations.append({
            'method': 'Panel FE (within estimator)',
            'function': 'panel',
            'reason': 'Controls for time-invariant unobservables.',
            'assumptions': ['Strict exogeneity', 'No time-varying confounders'],
            'code': f"sp.panel(df, '{panel_formula}', "
                    f"entity='{id}', time='{time}', method='fe')",
            'params': {'data': data, 'formula': panel_formula,
                       'entity': id, 'time': time, 'method': 'fe'},
        })
        recommendations.append({
            'method': 'Correlated Random Effects (Mundlak)',
            'function': 'panel',
            'reason': 'Mundlak projection allows RE efficiency with FE consistency.',
            'code': f"sp.panel(df, '{panel_formula}', "
                    f"entity='{id}', time='{time}', method='mundlak')",
            'params': {'data': data, 'formula': panel_formula,
                       'entity': id, 'time': time, 'method': 'mundlak'},
        })

    else:
        # Cross-section
        if treatment:
            formula = f'{y} ~ {treatment}'
        elif controls:
            formula = f'{y} ~ {controls[0]}'
        else:
            formula = f'{y} ~ 1'
        recommendations.append({
            'method': 'OLS with robust SE',
            'function': 'regress',
            'reason': 'Cross-sectional data with continuous outcome.',
            'code': f"sp.regress('{y} ~ {ctrl_str or '...'}', data=df, robust='hc1')",
            'params': {'formula': formula, 'data': data, 'robust': 'hc1'},
        })

    # Outcome-type-specific additions
    if profile['y_type'] == 'binary' and design not in ['rd', 'did']:
        recommendations.append({
            'method': 'Logit (binary outcome)',
            'function': 'logit',
            'reason': 'Binary dependent variable → logit for correct likelihood.',
            'code': f"sp.logit(data=df, y='{y}', x=['{treatment}'] + controls[:5])",
            'params': {'data': data, 'y': y, 'x': [treatment] + controls[:5]
                       if treatment else controls[:5]},
        })
    elif profile['y_type'] == 'count':
        recommendations.append({
            'method': 'Poisson regression (count outcome)',
            'function': 'poisson',
            'reason': 'Count outcome → Poisson with robust SE is consistent.',
            'code': f"sp.poisson(data=df, y='{y}', x=['{treatment}'] + controls[:5])",
            'params': {'data': data, 'y': y, 'x': [treatment] + controls[:5]
                       if treatment else controls[:5]},
        })
    elif profile['y_type'] == 'fractional':
        recommendations.append({
            'method': 'Fractional logit (outcome in [0,1])',
            'function': 'fracreg',
            'reason': 'Proportional outcome → fractional logit (Papke-Wooldridge).',
            'code': f"sp.fracreg(data=df, y='{y}', x=['{treatment}'] + controls[:5])",
            'params': {'data': data, 'y': y, 'x': [treatment] + controls[:5]
                       if treatment else controls[:5]},
        })

    # Optional empirical verification (Plan 3: rule prior + empirical posterior)
    if verify and recommendations:
        from .verify import verify_recommendation

        k = min(verify_top_k, len(recommendations))
        for rec in recommendations[:k]:
            try:
                rec["verify"] = verify_recommendation(
                    rec, data,
                    B=verify_B,
                    budget_s=verify_budget_s,
                )
            except Exception as e:
                rec["verify"] = {"score": np.nan, "error": str(e)}

        # Re-rank top-k by verify score (stable sort; rest of list untouched).
        # Sort descending on score; NaN/error scores fall to the BOTTOM of
        # the head (keyed to +inf), not the middle — this preserves
        # determinism and makes a score=0.0 method (runnable but unstable)
        # strictly outrank a NaN one (not runnable).
        head = recommendations[:k]
        tail = recommendations[k:]

        def _sort_key(rec):
            v = rec.get("verify") or {}
            s = v.get("score")
            if s is None or not np.isfinite(s):
                return float("inf")  # push NaN / missing to the end
            return -float(s)         # primary: descending score

        head.sort(key=_sort_key)
        recommendations = head + tail

    return RecommendationResult(
        recommendations=recommendations,
        data_profile=profile,
        design=design,
        warnings=warnings_list,
        data=data,
        y=y,
        treatment=treatment,
    )
