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

        lines.append(f"\n{'─' * 70}")
        lines.append("SUGGESTED WORKFLOW")
        lines.append(f"{'─' * 70}")
        for i, step in enumerate(self._workflow_steps()):
            lines.append(f"  {i+1}. {step}")

        lines.append("\n" + "=" * 70)
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
    y_data = data[y]
    if y_data.nunique() == 2:
        profile['y_type'] = 'binary'
    elif y_data.dtype in ['int64', 'int32'] and y_data.min() >= 0:
        if y_data.max() <= 50 and (y_data == y_data.astype(int)).all():
            profile['y_type'] = 'count'
        else:
            profile['y_type'] = 'continuous'
    elif y_data.min() >= 0 and y_data.max() <= 1:
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
    if treatment and profile['treat_type'] == 'binary':
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
                      and data[c].dtype in ['int64', 'float64', 'int32', 'float32']]

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
        except Exception:
            pass

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
            recommendations.append({
                'method': 'Callaway-Sant\'Anna (2021) — staggered DID',
                'function': 'callaway_santanna',
                'reason': 'Multiple time periods with staggered treatment adoption. '
                          'Robust to heterogeneous treatment effects (unlike TWFE).',
                'assumptions': ['Parallel trends', 'No anticipation', 'Staggered adoption'],
                'robustness': 'Run sp.pretrends_test(), sp.honest_did(), sp.event_study()',
                'code': f"sp.callaway_santanna(df, y='{y}', treat='{treatment}', "
                        f"time='{time}', id='{id}')",
                'params': {'data': data, 'y': y, 'treat': treatment,
                           'time': time, 'id': id},
            })
            recommendations.append({
                'method': 'Sun-Abraham (2021) — interaction-weighted',
                'function': 'sun_abraham',
                'reason': 'Alternative heterogeneity-robust DID estimator.',
                'code': f"sp.sun_abraham(df, y='{y}', treat='{treatment}', "
                        f"time='{time}', id='{id}')",
                'params': {'data': data, 'y': y, 'treat': treatment,
                           'time': time, 'id': id},
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
        recommendations.append({
            'method': '2SLS (two-stage least squares)',
            'function': 'iv',
            'reason': 'Standard IV estimator for endogenous treatment.',
            'assumptions': ['Instrument relevance (F > 10)',
                            'Exclusion restriction', 'Monotonicity (for LATE)'],
            'robustness': 'Check first-stage F, sp.anderson_rubin_test(), sp.kitagawa_test()',
            'code': f"sp.iv(data=df, y='{y}', x_endog=['{treatment}'], z=['{z}'])",
            'params': {'data': data, 'y': y, 'x_endog': [treatment], 'z': [z]},
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
            'code': f"sp.match(df, y='{y}', treatment='{treatment}', "
                    f"covariates=[{ctrl_str}])",
            'params': {'data': data, 'y': y, 'treatment': treatment,
                       'covariates': controls[:10]},
        })
        recommendations.append({
            'method': 'Double ML (high-dimensional controls)',
            'function': 'dml',
            'reason': 'Handles many controls without overfitting via cross-fitting.',
            'code': f"sp.dml(df, y='{y}', treatment='{treatment}', "
                    f"covariates=[{ctrl_str}])",
            'params': {'data': data, 'y': y, 'treatment': treatment,
                       'covariates': controls[:20]},
        })

    elif design == 'panel':
        recommendations.append({
            'method': 'Panel FE (within estimator)',
            'function': 'panel',
            'reason': 'Controls for time-invariant unobservables.',
            'assumptions': ['Strict exogeneity', 'No time-varying confounders'],
            'code': f"sp.panel(df, y='{y}', x=['{treatment}', {ctrl_str}], "
                    f"id='{id}', time='{time}', method='fe')",
            'params': {'data': data, 'y': y, 'x': [treatment] + controls[:5],
                       'id': id, 'time': time, 'method': 'fe'},
        })
        recommendations.append({
            'method': 'Correlated Random Effects (Mundlak)',
            'function': 'panel',
            'reason': 'Mundlak projection allows RE efficiency with FE consistency.',
            'code': f"sp.panel(df, y='{y}', x=['{treatment}'], "
                    f"id='{id}', time='{time}', method='mundlak')",
            'params': {'data': data, 'y': y, 'x': [treatment] + controls[:5],
                       'id': id, 'time': time, 'method': 'mundlak'},
        })

    else:
        # Cross-section
        recommendations.append({
            'method': 'OLS with robust SE',
            'function': 'regress',
            'reason': 'Cross-sectional data with continuous outcome.',
            'code': f"sp.regress('{y} ~ {ctrl_str}', data=df, robust='hc1')",
            'params': {'formula': f'{y} ~ {treatment}' if treatment else f'{y} ~ {controls[0]}',
                       'data': data, 'robust': 'hc1'},
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

    return RecommendationResult(
        recommendations=recommendations,
        data_profile=profile,
        design=design,
        warnings=warnings_list,
        data=data,
        y=y,
        treatment=treatment,
    )
