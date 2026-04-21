"""
Multi-Estimator Comparison Engine.

Run multiple causal inference estimators on the same dataset,
compare estimates, diagnose each, and assess robustness.

**No other package does this.** Researchers typically run one method.
This runs 3-5 methods, compares, and tells you which to trust.

Usage
-----
>>> import statspai as sp
>>> comp = sp.compare_estimators(
...     data=df, y='wage', treatment='training',
...     methods=['ols', 'matching', 'ipw', 'dml'],
...     covariates=['age', 'education', 'experience'],
... )
>>> print(comp.summary())
>>> comp.plot()
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
import warnings


class ComparisonResult:
    """Results from multi-estimator comparison."""

    def __init__(self, results, estimates_table, agreement, n_obs):
        self.results = results  # dict: method -> result object
        self.estimates_table = estimates_table  # DataFrame
        self.agreement = agreement  # dict with agreement metrics
        self.n_obs = n_obs

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "Multi-Estimator Comparison",
            "=" * 70,
            "",
            f"{'Method':<25s} {'Estimate':>10s} {'SE':>10s} "
            f"{'95% CI':>22s} {'p-value':>10s}",
            "─" * 70,
        ]
        for _, row in self.estimates_table.iterrows():
            ci = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
            sig = "***" if row['p_value'] < 0.01 else "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.1 else ""
            lines.append(f"{row['method']:<25s} {row['estimate']:>10.4f} "
                         f"{row['se']:>10.4f} {ci:>22s} "
                         f"{row['p_value']:>9.4f}{sig}")

        lines.append("─" * 70)

        # Agreement metrics
        lines.append("\nAGREEMENT DIAGNOSTICS")
        lines.append(f"  Sign agreement:        {self.agreement['sign_agree']:.0%}")
        lines.append(f"  Significance agreement: {self.agreement['sig_agree']:.0%}")
        lines.append(f"  Estimate range:        [{self.agreement['min_est']:.4f}, "
                     f"{self.agreement['max_est']:.4f}]")
        lines.append(f"  Coefficient of variation: {self.agreement['cv']:.2f}")

        if self.agreement['cv'] < 0.25:
            lines.append("\n  ✓ ROBUST: Estimates are stable across methods.")
        elif self.agreement['cv'] < 0.5:
            lines.append("\n  ~ MODERATE: Some variation across methods. "
                         "Investigate assumptions.")
        else:
            lines.append("\n  ✗ FRAGILE: Large variation across methods. "
                         "Results are sensitive to identification strategy.")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Forest plot comparing estimates across methods."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(3, len(self.estimates_table) * 0.6)))

        df = self.estimates_table.sort_values('estimate')
        y_pos = range(len(df))

        ax.errorbar(df['estimate'].values, y_pos,
                     xerr=[df['estimate'].values - df['ci_lower'].values,
                           df['ci_upper'].values - df['estimate'].values],
                     fmt='o', capsize=4, color='steelblue', elinewidth=2, markersize=8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(df['method'].values)
        ax.axvline(0, color='gray', ls='--', lw=0.5)

        # Shade the range of estimates
        ax.axvspan(df['estimate'].min(), df['estimate'].max(),
                    alpha=0.1, color='orange')

        ax.set_xlabel('Treatment Effect Estimate')
        ax.set_title('Multi-Estimator Comparison')
        ax.get_figure().tight_layout()
        return ax


def compare_estimators(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    methods: List[str] = None,
    covariates: List[str] = None,
    id: str = None,
    time: str = None,
    instrument: str = None,
    alpha: float = 0.05,
    method_hints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ComparisonResult:
    """
    Run multiple estimators on the same data and compare.

    **Unique to StatsPAI.** No other package provides automated
    multi-method comparison with agreement diagnostics.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treatment : str
        Treatment variable (binary).
    methods : list of str, optional
        Estimators to compare. Default auto-selects based on data.
        Classical options: ``'ols'``, ``'matching'``, ``'ipw'``,
        ``'aipw'``, ``'dml'``, ``'g_computation'``,
        ``'causal_forest'``, ``'did'``, ``'panel_fe'``.

        Hint-driven Sprint-B options (require ``method_hints``):
        ``'proximal'``, ``'msm'``, ``'principal_strat'``, ``'mediate'``,
        ``'mediate_interventional'``, ``'front_door'``. Each needs
        method-specific kwargs the shared signature does not expose
        (proxy_z/proxy_w, time_varying, strata, mediator, etc.) — pass
        them through ``method_hints``.
    method_hints : dict, optional
        Per-method keyword overrides, merged with the shared kwargs
        when dispatching each estimator. Structure::

            {'proximal': {'proxy_z': ['z'], 'proxy_w': ['w']},
             'msm':      {'time_varying': ['L_lag']},
             'principal_strat': {'strata': 's'}}

        **Collision rule** (docs/ROADMAP.md §6): per-method hints
        take precedence over the shared kwargs for the method they
        name. If the top-level ``covariates=['age']`` disagrees with
        ``method_hints={'proximal': {'covariates': ['age', 'educ']}}``,
        proximal uses the hint and every other method uses the
        shared arg. A ``UserWarning`` fires on conflict so the
        override is visible in the log.
    covariates : list of str, optional
    id : str, optional
        Panel unit ID.
    time : str, optional
        Time variable.
    instrument : str, optional
    alpha : float, default 0.05

    Returns
    -------
    ComparisonResult
        With .summary(), .plot(), .results (dict of individual results).

    Examples
    --------
    >>> import statspai as sp
    >>> comp = sp.compare_estimators(
    ...     data=df, y='wage', treatment='training',
    ...     methods=['ols', 'matching', 'ipw', 'dml'],
    ...     covariates=['age', 'education'],
    ... )
    >>> print(comp.summary())
    >>> comp.plot()
    """
    import statspai as sp

    if covariates is None:
        covariates = [c for c in data.columns
                      if c not in [y, treatment, id, time, instrument]
                      and pd.api.types.is_numeric_dtype(data[c])]

    if methods is None:
        methods = ['ols', 'matching', 'ipw']
        if len(covariates) > 5:
            methods.append('dml')

    df = data.dropna(subset=[y, treatment] + covariates)
    n = len(df)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Resolve per-method hints with the ROADMAP §6 collision rule:
    # hint-supplied args win over shared args for the method they name,
    # with a UserWarning on explicit conflict.
    method_hints = method_hints or {}

    def _values_differ(a, b):
        """
        Compare two hint values, with list equality being set-based so
        that pure reordering (e.g. ``['x1','x2']`` vs ``['x2','x1']``)
        does not trigger a spurious conflict warning. Tuple / set
        callers get the same treatment; everything else falls back
        to standard ``!=``.
        """
        if isinstance(a, (list, tuple, set)) and isinstance(b, (list, tuple, set)):
            try:
                return set(a) != set(b)
            except TypeError:
                # Unhashable elements (e.g. nested lists) — fall back
                # to strict element-wise comparison, still
                # order-sensitive, but better than crashing.
                return list(a) != list(b)
        return a != b

    def _resolve_hints(method_name, shared_kwargs):
        """Merge shared kwargs with per-method hints; hint wins on conflict."""
        hint = method_hints.get(method_name, {}) or {}
        merged = dict(shared_kwargs)
        for k, v in hint.items():
            if k in merged and _values_differ(merged[k], v):
                warnings.warn(
                    f"compare_estimators: method_hints['{method_name}']"
                    f"['{k}'] overrides the shared {k}={shared_kwargs[k]!r} "
                    f"with {v!r}.",
                    UserWarning, stacklevel=2,
                )
            merged[k] = v
        return merged

    results = {}
    rows = []

    for method in methods:
        try:
            if method == 'ols':
                x_vars = [treatment] + covariates[:10]
                formula = f"{y} ~ {' + '.join(x_vars)}"
                r = sp.regress(formula, data=df, robust='hc1')
                est = r.params[treatment]
                se = r.std_errors[treatment]
                results['OLS'] = r
                name = 'OLS (robust SE)'

            elif method == 'matching':
                r = sp.match(df, y=y, treatment=treatment,
                             covariates=covariates[:10])
                est = r.estimate
                se = r.se
                results['Matching'] = r
                name = 'Propensity Score Matching'

            elif method == 'ipw':
                r = sp.ipw(df, y=y, treatment=treatment,
                           covariates=covariates[:10])
                est = r.estimate if hasattr(r, 'estimate') else r.params.iloc[0]
                se = r.se if hasattr(r, 'se') else r.std_errors.iloc[0]
                results['IPW'] = r
                name = 'Inverse Probability Weighting'

            elif method == 'aipw':
                r = sp.aipw(df, y=y, treatment=treatment,
                            covariates=covariates[:10])
                est = r.estimate if hasattr(r, 'estimate') else r.params.iloc[0]
                se = r.se if hasattr(r, 'se') else r.std_errors.iloc[0]
                results['AIPW'] = r
                name = 'Augmented IPW (DR)'

            elif method == 'g_computation':
                # sp.g_computation uses the 'treat=' kwarg (not
                # 'treatment=') to match the rest of the Sprint-B
                # causal-inference surface. Handle that mapping here
                # so the shared ``treatment`` parameter works.
                # Explicit binary guard: the default ATE path in
                # g_computation rejects non-binary D. Without this
                # guard the outer ``except Exception`` below would
                # swallow the error in a 400-line ValueError dump of
                # unique treatment values. Raising a clean
                # ValueError here keeps the warning compact.
                treat_vals = df[treatment].dropna().unique()
                if not set(treat_vals).issubset({0, 1, 0.0, 1.0}):
                    raise ValueError(
                        f"g_computation requires a binary treatment "
                        f"(0/1); '{treatment}' has "
                        f"{len(treat_vals)} unique values "
                        f"(first: {sorted(treat_vals)[:3]}). "
                        f"Use estimand='dose_response' directly via "
                        f"sp.g_computation() for continuous D."
                    )
                r = sp.g_computation(df, y=y, treat=treatment,
                                     covariates=covariates[:10],
                                     seed=0)
                est = r.estimate
                se = r.se
                results['G-computation'] = r
                name = 'G-computation (parametric g-formula)'

            elif method == 'dml':
                r = sp.dml(df, y=y, treatment=treatment,
                           covariates=covariates[:20])
                est = r.estimate
                se = r.se
                results['DML'] = r
                name = 'Double/Debiased ML'

            elif method == 'causal_forest':
                r = sp.causal_forest(data=df, y=y, treatment=treatment,
                                     covariates=covariates[:20])
                est = r.estimate if hasattr(r, 'estimate') else r.ate
                se = r.se if hasattr(r, 'se') else r.ate_se
                results['Causal Forest'] = r
                name = 'Causal Forest (GRF)'

            elif method == 'panel_fe' and id and time:
                r = sp.panel(df, y=y, x=[treatment] + covariates[:5],
                             id=id, time=time, method='fe')
                est = r.params[treatment]
                se = r.std_errors[treatment]
                results['Panel FE'] = r
                name = 'Panel Fixed Effects'

            elif method == 'did' and id and time:
                r = sp.did(df, y=y, treat=treatment, time=time, id=id)
                est = r.estimate
                se = r.se
                results['DID'] = r
                name = 'Difference-in-Differences'

            # ----- Sprint-B hint-driven branches -------------------- #
            # Each requires per-method kwargs supplied via
            # method_hints[method]; we raise a clean ValueError if the
            # mandatory hint is missing so the outer try/except turns
            # it into a compact UserWarning rather than silent skip.

            elif method == 'proximal':
                hint = method_hints.get('proximal', {}) or {}
                proxy_z = hint.get('proxy_z')
                proxy_w = hint.get('proxy_w')
                if not proxy_z or not proxy_w:
                    raise ValueError(
                        "method='proximal' requires "
                        "method_hints['proximal']={'proxy_z': [...], "
                        "'proxy_w': [...]}."
                    )
                merged = _resolve_hints('proximal', {
                    'covariates': [c for c in covariates[:10]
                                   if c not in proxy_z + proxy_w],
                })
                r = sp.proximal(df, y=y, treat=treatment,
                                proxy_z=list(proxy_z),
                                proxy_w=list(proxy_w),
                                covariates=merged['covariates'])
                est = r.estimate
                se = r.se
                results['Proximal'] = r
                name = 'Proximal Causal Inference (bridge 2SLS)'

            elif method == 'msm':
                hint = method_hints.get('msm', {}) or {}
                tv = hint.get('time_varying')
                if not tv or not id or not time:
                    raise ValueError(
                        "method='msm' requires id + time + "
                        "method_hints['msm']={'time_varying': [...]}."
                    )
                merged = _resolve_hints('msm', {
                    'baseline': [c for c in covariates[:5] if c not in tv],
                })
                r = sp.msm(df, y=y, treat=treatment, id=id, time=time,
                           time_varying=list(tv),
                           baseline=merged.get('baseline'))
                est = r.estimate
                se = r.se
                results['MSM'] = r
                name = 'Marginal Structural Model (IPTW)'

            elif method == 'principal_strat':
                hint = method_hints.get('principal_strat', {}) or {}
                strata = hint.get('strata')
                if not strata:
                    raise ValueError(
                        "method='principal_strat' requires "
                        "method_hints['principal_strat']={'strata': <col>}."
                    )
                merged = _resolve_hints('principal_strat', {
                    'covariates': covariates[:10],
                    'method': 'principal_score' if covariates else 'monotonicity',
                })
                r = sp.principal_strat(
                    df, y=y, treat=treatment, strata=strata,
                    covariates=merged['covariates'] if merged['method'] == 'principal_score' else None,
                    method=merged['method'],
                )
                # PrincipalStratResult is not a CausalResult; pull
                # the complier PCE (= LATE) by stratum NAME rather
                # than .iloc[0] so an upstream reordering of the
                # effects table can't silently make us report the
                # always-taker effect instead.
                effects = getattr(r, 'effects', None)
                if effects is not None and len(effects):
                    _complier = effects['stratum'].astype(str).str.lower().str.contains('complier')
                    _row = effects[_complier].iloc[0] if _complier.any() else effects.iloc[0]
                    est = float(_row['estimate'])
                    se = float(_row['se']) if 'se' in effects.columns else 0.0
                else:
                    est, se = np.nan, np.nan
                results['Principal Strat'] = r
                name = 'Principal Stratification (complier PCE)'

            elif method == 'front_door':
                hint = method_hints.get('front_door', {}) or {}
                mediator = hint.get('mediator')
                if not mediator:
                    raise ValueError(
                        "method='front_door' requires "
                        "method_hints['front_door']={'mediator': <col>}."
                    )
                merged = _resolve_hints('front_door', {
                    'covariates': [c for c in covariates[:10] if c != mediator],
                })
                r = sp.front_door(df, y=y, treat=treatment,
                                  mediator=mediator,
                                  covariates=merged['covariates'])
                est = r.estimate
                se = r.se
                results['Front-door'] = r
                name = 'Front-door adjustment'

            elif method in ('mediate', 'mediate_interventional'):
                hint = method_hints.get(method, {}) or {}
                mediator = hint.get('mediator')
                if not mediator:
                    raise ValueError(
                        f"method='{method}' requires "
                        f"method_hints['{method}']={{'mediator': <col>}}."
                    )
                merged = _resolve_hints(method, {
                    'covariates': [c for c in covariates[:10] if c != mediator],
                })
                if method == 'mediate':
                    r = sp.mediate(df, y=y, treat=treatment,
                                   mediator=mediator,
                                   covariates=merged['covariates'] or None)
                    label = 'Causal Mediation (ACME)'
                    key = 'Mediation (natural)'
                else:
                    tv_c = hint.get('tv_confounders')
                    r = sp.mediate_interventional(
                        df, y=y, treat=treatment, mediator=mediator,
                        covariates=merged['covariates'] or None,
                        tv_confounders=list(tv_c) if tv_c else None,
                    )
                    label = 'Interventional Mediation (IIE)'
                    key = 'Mediation (interventional)'
                est = r.estimate
                se = r.se
                results[key] = r
                name = label

            else:
                continue

            p_val = 2 * (1 - stats.norm.cdf(abs(est / se))) if se > 0 else np.nan
            rows.append({
                'method': name,
                'estimate': est,
                'se': se,
                'ci_lower': est - z_crit * se,
                'ci_upper': est + z_crit * se,
                'p_value': p_val,
            })

        except Exception as e:
            warnings.warn(f"{method} failed: {e}")

    estimates_table = pd.DataFrame(rows)

    # Agreement metrics
    if len(rows) > 1:
        ests = np.array([r['estimate'] for r in rows])
        ses = np.array([r['se'] for r in rows])
        pvals = np.array([r['p_value'] for r in rows])

        sign_agree = np.mean(np.sign(ests) == np.sign(ests[0]))
        sig_agree = np.mean((pvals < 0.05) == (pvals[0] < 0.05))
        est_range = ests.max() - ests.min()
        mean_est = np.mean(ests)
        cv = np.std(ests) / abs(mean_est) if abs(mean_est) > 1e-10 else np.inf

        agreement = {
            'sign_agree': sign_agree,
            'sig_agree': sig_agree,
            'min_est': ests.min(),
            'max_est': ests.max(),
            'range': est_range,
            'mean': mean_est,
            'cv': cv,
        }
    else:
        agreement = {'sign_agree': 1, 'sig_agree': 1, 'min_est': 0,
                     'max_est': 0, 'range': 0, 'mean': 0, 'cv': 0}

    return ComparisonResult(
        results=results,
        estimates_table=estimates_table,
        agreement=agreement,
        n_obs=n,
    )
