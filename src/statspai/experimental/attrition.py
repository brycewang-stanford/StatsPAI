"""
Attrition analysis for RCTs.

Tests for differential attrition and computes bounds on treatment effects
under various attrition assumptions.

References
----------
Lee, D.S. (2009).
"Training, Wages, and Sample Selection: Estimating Sharp Bounds
on Treatment Effects." *RES*, 76(3), 1071-1102. [@lee2009training]
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats


class AttritionResult:
    """Results from attrition analysis."""

    def __init__(self, overall_rate, treat_rate, control_rate,
                 diff_test_stat, diff_p_value, covariate_tests,
                 n_total, n_attrit):
        self.overall_rate = overall_rate
        self.treat_rate = treat_rate
        self.control_rate = control_rate
        self.diff_test_stat = diff_test_stat
        self.diff_p_value = diff_p_value
        self.covariate_tests = covariate_tests
        self.n_total = n_total
        self.n_attrit = n_attrit

    def summary(self) -> str:
        lines = [
            "Attrition Analysis",
            "=" * 55,
            f"Total sample: {self.n_total}   Attrited: {self.n_attrit} "
            f"({self.overall_rate:.1%})",
            f"Treatment attrition: {self.treat_rate:.1%}",
            f"Control attrition:   {self.control_rate:.1%}",
            f"Differential attrition test: chi2 = {self.diff_test_stat:.3f}, "
            f"p = {self.diff_p_value:.4f}",
        ]
        if self.diff_p_value < 0.05:
            lines.append("WARNING: Significant differential attrition detected!")
        if self.covariate_tests is not None:
            lines.append("\nCovariate predictors of attrition:")
            lines.append(f"{'Variable':<20s} {'Coef':>10s} {'p-value':>10s}")
            lines.append("-" * 42)
            for _, row in self.covariate_tests.iterrows():
                lines.append(f"{row['variable']:<20s} {row['coef']:>10.4f} "
                             f"{row['p_value']:>10.4f}")
        lines.append("=" * 55)
        return "\n".join(lines)


def attrition_test(
    data: pd.DataFrame,
    treatment: str,
    observed: str,
    covariates: List[str] = None,
) -> AttritionResult:
    """
    Test for differential attrition in an RCT.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
        Treatment indicator (0/1).
    observed : str
        Indicator for whether outcome is observed (1) or missing (0).
    covariates : list of str, optional
        Baseline covariates to test as predictors of attrition.

    Returns
    -------
    AttritionResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.attrition_test(df, treatment='treated', observed='endline_observed',
    ...                            covariates=['age', 'income', 'education'])
    >>> print(result.summary())
    """
    n = len(data)
    attrit = 1 - data[observed].values
    treat = data[treatment].values

    overall_rate = attrit.mean()
    treat_rate = attrit[treat == 1].mean()
    control_rate = attrit[treat == 0].mean()

    # Chi-squared test for differential attrition
    table = pd.crosstab(data[treatment], data[observed])
    chi2, p_val = stats.chi2_contingency(table)[:2]

    # Covariate predictors of attrition
    cov_tests = None
    if covariates is not None:
        rows = []
        for var in covariates:
            x = data[var].values
            valid = np.isfinite(x)
            x_v, a_v = x[valid], attrit[valid]
            X = np.column_stack([np.ones(len(x_v)), x_v])
            try:
                beta = np.linalg.lstsq(X, a_v, rcond=None)[0]
                resid = a_v - X @ beta
                se = np.sqrt(np.sum(resid**2) / (len(a_v) - 2) *
                             np.linalg.inv(X.T @ X)[1, 1])
                t = beta[1] / se
                p = 2 * (1 - stats.t.cdf(abs(t), len(a_v) - 2))
                rows.append({'variable': var, 'coef': beta[1], 'se': se, 'p_value': p})
            except Exception:
                rows.append({'variable': var, 'coef': np.nan, 'se': np.nan, 'p_value': np.nan})
        cov_tests = pd.DataFrame(rows)

    return AttritionResult(
        overall_rate=overall_rate, treat_rate=treat_rate,
        control_rate=control_rate, diff_test_stat=chi2,
        diff_p_value=p_val, covariate_tests=cov_tests,
        n_total=n, n_attrit=int(attrit.sum()),
    )


def attrition_bounds(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    observed: str = None,
    method: str = "lee",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute bounds on treatment effects under attrition.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treatment : str
        Treatment indicator (0/1).
    observed : str, optional
        Indicator for observed outcome. If None, uses non-missing y.
    method : str, default 'lee'
        Bounding method: 'lee' (Lee 2009), 'manski' (worst-case).
    alpha : float, default 0.05

    Returns
    -------
    dict
        Keys: 'lower_bound', 'upper_bound', 'naive_ate', 'method', 'n_obs'.
    """
    df = data.copy()
    if observed is None:
        df['_observed'] = df[y].notna().astype(int)
        observed = '_observed'

    obs_mask = df[observed] == 1
    treat_mask = df[treatment] == 1

    # Naive ATE (complete cases only)
    y_treat = df.loc[obs_mask & treat_mask, y].values
    y_control = df.loc[obs_mask & ~treat_mask, y].values
    naive_ate = y_treat.mean() - y_control.mean()

    if method == 'lee':
        # Lee (2009) trimming bounds
        p_treat = obs_mask[treat_mask].mean()
        p_control = obs_mask[~treat_mask].mean()

        if p_treat > p_control:
            # Trim treatment group from top/bottom
            trim_frac = 1 - p_control / p_treat
            y_sorted = np.sort(y_treat)
            n_trim = int(np.ceil(len(y_sorted) * trim_frac))

            # Lower bound: trim from top
            y_trim_low = y_sorted[:len(y_sorted) - n_trim]
            lower = y_trim_low.mean() - y_control.mean()

            # Upper bound: trim from bottom
            y_trim_up = y_sorted[n_trim:]
            upper = y_trim_up.mean() - y_control.mean()
        elif p_control > p_treat:
            trim_frac = 1 - p_treat / p_control
            y_sorted = np.sort(y_control)
            n_trim = int(np.ceil(len(y_sorted) * trim_frac))

            y_trim_low = y_sorted[n_trim:]
            lower = y_treat.mean() - y_trim_low.mean()

            y_trim_up = y_sorted[:len(y_sorted) - n_trim]
            upper = y_treat.mean() - y_trim_up.mean()
        else:
            lower = upper = naive_ate

    elif method == 'manski':
        # Manski worst-case bounds
        y_all = df.loc[obs_mask, y].values
        y_min, y_max = y_all.min(), y_all.max()

        n_treat_miss = (~obs_mask & treat_mask).sum()
        n_control_miss = (~obs_mask & ~treat_mask).sum()
        n_treat_obs = (obs_mask & treat_mask).sum()
        n_control_obs = (obs_mask & ~treat_mask).sum()

        # Lower bound: missing treated get y_min, missing control get y_max
        lower = ((y_treat.sum() + n_treat_miss * y_min) / (n_treat_obs + n_treat_miss) -
                 (y_control.sum() + n_control_miss * y_max) / (n_control_obs + n_control_miss))

        # Upper bound: missing treated get y_max, missing control get y_min
        upper = ((y_treat.sum() + n_treat_miss * y_max) / (n_treat_obs + n_treat_miss) -
                 (y_control.sum() + n_control_miss * y_min) / (n_control_obs + n_control_miss))
    else:
        raise ValueError(f"Unknown method: {method}")

    if lower > upper:
        lower, upper = upper, lower

    return {
        'lower_bound': lower,
        'upper_bound': upper,
        'naive_ate': naive_ate,
        'method': method,
        'n_obs': obs_mask.sum(),
        'n_total': len(df),
        'attrition_rate': 1 - obs_mask.mean(),
    }
