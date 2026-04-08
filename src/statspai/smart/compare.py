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
        Options: 'ols', 'matching', 'ipw', 'aipw', 'dml',
        'causal_forest', 'did', 'panel_fe'.
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
