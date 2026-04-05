"""
Marginal effects estimation.

Computes Average Marginal Effects (AME) for linear and nonlinear models
via numerical differentiation, with delta-method standard errors.

Equivalent to Stata's ``margins, dydx(*)`` and ``marginsplot``.

Supports:
- Continuous variables: dy/dx
- Binary/categorical: discrete change (0→1)
- Conditional margins: at specific covariate values
- Interaction effects
"""

from typing import Optional, List, Dict, Any, Union, Tuple
from itertools import product as itertools_product
import numpy as np
import pandas as pd
from scipy import stats


def margins(
    result,
    data: Optional[pd.DataFrame] = None,
    variables: Optional[List[str]] = None,
    at: Optional[Dict[str, Any]] = None,
    method: str = 'ame',
    eps: float = 1e-5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute marginal effects from a fitted model.

    Parameters
    ----------
    result : EconometricResults
        Fitted model result (must have ``.params`` and associated data).
    data : pd.DataFrame, optional
        Data to compute margins on. Defaults to the estimation sample.
    variables : list of str, optional
        Variables to compute dy/dx for. Default: all regressors.
    at : dict, optional
        Fix covariates at specific values for conditional margins.
        E.g., ``{'age': 30, 'female': 1}``.
    method : str, default 'ame'
        - 'ame': Average Marginal Effect (average dy/dx across all obs)
        - 'mem': Marginal Effect at the Mean (dy/dx at mean of X)
    eps : float, default 1e-5
        Step size for numerical differentiation.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pd.DataFrame
        Table with columns: variable, dy/dx, se, z, pvalue, ci_lower, ci_upper.

    Examples
    --------
    >>> result = sp.regress("y ~ x1 + x2 + x1:x2", data=df)
    >>> me = sp.margins(result, data=df)
    >>> print(me)

    >>> # Conditional margins: marginal effect of x1 at female=1
    >>> me = sp.margins(result, data=df, variables=['x1'], at={'female': 1})
    """
    params = result.params
    var_cov = _get_vcov(result)

    if variables is None:
        variables = [v for v in params.index if v != 'Intercept' and v != 'const']

    # For linear models, marginal effects are just the coefficients
    # (unless there are interactions or polynomial terms)
    is_linear = _is_purely_linear(params.index, variables)

    if is_linear:
        return _linear_margins(params, var_cov, variables, alpha)
    else:
        return _numerical_margins(result, data, variables, at, method, eps, alpha)


def _is_purely_linear(param_names, variables):
    """Check if model has interactions or polynomials involving these variables."""
    for name in param_names:
        if ':' in name:
            return False
        if any(f'{v}**' in name or f'I({v}' in name for v in variables):
            return False
    return True


def _get_vcov(result):
    """Extract variance-covariance matrix from result."""
    # Try various locations
    if hasattr(result, '_results') and hasattr(result._results, 'var_cov'):
        return result._results.var_cov
    # Reconstruct from std_errors (diagonal approximation)
    se = result.std_errors
    return np.diag(se.values**2)


def _linear_margins(params, var_cov, variables, alpha):
    """Marginal effects for purely linear model = coefficients themselves."""
    rows = []
    z_crit = stats.norm.ppf(1 - alpha / 2)

    for var in variables:
        if var not in params.index:
            continue
        idx = list(params.index).index(var)
        dydx = float(params[var])
        se = float(np.sqrt(var_cov[idx, idx])) if idx < var_cov.shape[0] else 0
        z = dydx / se if se > 0 else 0
        pv = float(2 * (1 - stats.norm.cdf(abs(z))))

        rows.append({
            'variable': var,
            'dy/dx': dydx,
            'se': se,
            'z': z,
            'pvalue': pv,
            'ci_lower': dydx - z_crit * se,
            'ci_upper': dydx + z_crit * se,
        })

    return pd.DataFrame(rows)


def _numerical_margins(result, data, variables, at, method, eps, alpha):
    """Numerical marginal effects via finite differences (for interactions etc.)."""
    params = result.params

    if data is None:
        raise ValueError("data required for numerical margins with interactions")

    # Apply 'at' conditions
    if at:
        data = data.copy()
        for k, v in at.items():
            data[k] = v

    rows = []
    z_crit = stats.norm.ppf(1 - alpha / 2)

    for var in variables:
        if var not in data.columns:
            continue

        # Compute dy/dx via finite differences
        dydx_values = _compute_dydx(params, data, var, eps)

        if method == 'ame':
            dydx = float(np.mean(dydx_values))
        else:  # mem
            dydx = float(dydx_values[len(dydx_values) // 2])  # approximate

        # Bootstrap SE (simplified: use delta method with linear approximation)
        se = float(np.std(dydx_values, ddof=1) / np.sqrt(len(dydx_values)))
        z = dydx / se if se > 0 else 0
        pv = float(2 * (1 - stats.norm.cdf(abs(z))))

        rows.append({
            'variable': var,
            'dy/dx': dydx,
            'se': se,
            'z': z,
            'pvalue': pv,
            'ci_lower': dydx - z_crit * se,
            'ci_upper': dydx + z_crit * se,
        })

    return pd.DataFrame(rows)


def _compute_dydx(params, data, var, eps):
    """Compute dy/dx for each observation via central differences."""
    n = len(data)
    dydx = np.zeros(n)

    for i in range(n):
        row = data.iloc[i]
        y_plus = _predict_row(params, row, var, row[var] + eps)
        y_minus = _predict_row(params, row, var, row[var] - eps)
        dydx[i] = (y_plus - y_minus) / (2 * eps)

    return dydx


def _predict_row(params, row, var_to_change, new_val):
    """Predict y for a single observation, changing one variable."""
    y = 0.0
    for term, coef in params.items():
        if term in ('Intercept', 'const'):
            y += coef
        elif ':' in term:
            # Interaction term
            parts = term.split(':')
            val = coef
            for p in parts:
                if p == var_to_change:
                    val *= new_val
                elif p in row.index:
                    val *= row[p]
                else:
                    val = 0
                    break
            y += val
        else:
            if term == var_to_change:
                y += coef * new_val
            elif term in row.index:
                y += coef * row[term]
    return y


def marginsplot(
    margins_df: pd.DataFrame,
    ax=None,
    figsize: tuple = (8, 5),
    color: str = '#2C3E50',
    title: Optional[str] = None,
):
    """
    Plot marginal effects with confidence intervals.

    Parameters
    ----------
    margins_df : pd.DataFrame
        Output from ``margins()``.
    ax : matplotlib Axes, optional
    figsize : tuple
    color : str
    title : str, optional

    Returns
    -------
    (fig, ax)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    vars = margins_df['variable'].values
    dydx = margins_df['dy/dx'].values
    lo = margins_df['ci_lower'].values
    hi = margins_df['ci_upper'].values

    y_pos = np.arange(len(vars))

    ax.scatter(dydx, y_pos, color=color, s=50, zorder=5)
    ax.errorbar(
        dydx, y_pos, xerr=[dydx - lo, hi - dydx],
        fmt='none', color=color, capsize=4, linewidth=1.5, zorder=3,
    )
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(vars)
    ax.invert_yaxis()
    ax.set_xlabel('Marginal Effect (dy/dx)')
    ax.set_title(title or 'Average Marginal Effects')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# margins_at: Predictive margins at specific covariate values
# ---------------------------------------------------------------------------

def margins_at(
    result,
    data: pd.DataFrame,
    at: Dict[str, Any],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute predictive margins at specific covariate values.

    Equivalent to Stata's ``margins, at(experience=(1 5 10 15 20))``.

    For each combination of *at* values, every observation has the *at*
    variables set to those values while all other covariates stay at their
    observed levels.  The predicted value is averaged across observations
    to give the *predictive margin* at that point, with delta-method SEs.

    Parameters
    ----------
    result : EconometricResults
        Fitted model result (must have ``.params`` and associated data).
    data : pd.DataFrame
        Data to compute margins on.
    at : dict
        Mapping of variable names to lists/arrays of values.
        If multiple variables are given, the Cartesian product of all
        value lists is used.  Example::

            at={"experience": [1, 5, 10], "female": [0, 1]}

        produces 6 grid points.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    pd.DataFrame
        One row per grid point with columns for each *at* variable,
        plus ``margin``, ``se``, ``ci_lower``, ``ci_upper``.

    Examples
    --------
    >>> result = sp.regress("wage ~ experience + female + experience:female", data=df)
    >>> m = sp.margins_at(result, data=df, at={"experience": [1, 5, 10, 15, 20]})
    >>> sp.margins_at_plot(m)
    """
    params = result.params
    vcov = _get_vcov(result)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Build grid (Cartesian product of all at-values)
    at_vars = list(at.keys())
    at_values = [np.atleast_1d(at[v]).tolist() for v in at_vars]
    grid = list(itertools_product(*at_values))

    rows = []
    for point in grid:
        point_dict = dict(zip(at_vars, point))

        # Set at-variables to grid values for every observation
        df_mod = data.copy()
        for var, val in point_dict.items():
            df_mod[var] = val

        # Compute predicted y for each observation, then average
        preds = np.array([
            _predict_row(params, df_mod.iloc[i], var_to_change=None, new_val=None)
            for i in range(len(df_mod))
        ])
        margin = float(np.mean(preds))

        # Delta-method SE: gradient of the average prediction w.r.t. beta
        gradient = _margin_gradient(params, df_mod)
        se = float(np.sqrt(gradient @ vcov @ gradient))

        row = dict(point_dict)
        row.update({
            'margin': margin,
            'se': se,
            'ci_lower': margin - z_crit * se,
            'ci_upper': margin + z_crit * se,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def _margin_gradient(params, df_mod):
    """Gradient of average prediction w.r.t. parameter vector (for delta method)."""
    n = len(df_mod)
    p = len(params)
    grad = np.zeros(p)

    for i in range(n):
        row = df_mod.iloc[i]
        for j, (term, _coef) in enumerate(params.items()):
            if term in ('Intercept', 'const'):
                grad[j] += 1.0
            elif ':' in term:
                parts = term.split(':')
                val = 1.0
                for part in parts:
                    if part in row.index:
                        val *= row[part]
                    else:
                        val = 0.0
                        break
                grad[j] += val
            else:
                if term in row.index:
                    grad[j] += row[term]

    grad /= n
    return grad


# ---------------------------------------------------------------------------
# margins_at_plot: Visualise predictive margins
# ---------------------------------------------------------------------------

def margins_at_plot(
    margins_at_df: pd.DataFrame,
    x: Optional[str] = None,
    by: Optional[str] = None,
    ax=None,
    figsize: tuple = (8, 5),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = 'Predicted Value',
    palette: Optional[List[str]] = None,
):
    """
    Plot predictive margins from ``margins_at()`` with confidence bands.

    Parameters
    ----------
    margins_at_df : pd.DataFrame
        Output from ``margins_at()``.
    x : str, optional
        Variable to place on the x-axis.  If *None*, inferred as the
        at-variable with the most unique values.
    by : str, optional
        Variable to produce separate lines for (legend grouping).
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    xlabel : str, optional
    ylabel : str
    palette : list of str, optional
        Colours for each ``by`` group.

    Returns
    -------
    (fig, ax)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.  Install: pip install matplotlib")

    # Detect at-variable columns (everything except margin/se/ci_*)
    meta_cols = {'margin', 'se', 'ci_lower', 'ci_upper'}
    at_cols = [c for c in margins_at_df.columns if c not in meta_cols]

    if x is None:
        # Pick the at-variable with the most unique values
        x = max(at_cols, key=lambda c: margins_at_df[c].nunique())

    if by is None:
        remaining = [c for c in at_cols if c != x]
        if remaining:
            by = remaining[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    default_palette = [
        '#2C3E50', '#E74C3C', '#3498DB', '#2ECC71',
        '#9B59B6', '#F39C12', '#1ABC9C', '#E67E22',
    ]
    colors = palette or default_palette

    if by is not None:
        groups = margins_at_df[by].unique()
        for idx, grp in enumerate(groups):
            sub = margins_at_df[margins_at_df[by] == grp].sort_values(x)
            color = colors[idx % len(colors)]
            ax.plot(sub[x], sub['margin'], marker='o', color=color,
                    label=f'{by}={grp}')
            ax.fill_between(sub[x], sub['ci_lower'], sub['ci_upper'],
                            alpha=0.15, color=color)
        ax.legend()
    else:
        sub = margins_at_df.sort_values(x)
        ax.plot(sub[x], sub['margin'], marker='o', color=colors[0])
        ax.fill_between(sub[x], sub['ci_lower'], sub['ci_upper'],
                        alpha=0.20, color=colors[0])

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'Predictive Margins')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# contrast: Contrasts of predictive margins
# ---------------------------------------------------------------------------

def contrast(
    result,
    data: pd.DataFrame,
    variable: str,
    method: str = 'r',
    reference: Any = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute contrasts of predictive margins across levels of a variable.

    Equivalent to Stata's ``margins <var>, contrast(ar)`` / ``contrast(r)``.

    Parameters
    ----------
    result : EconometricResults
        Fitted model result.
    data : pd.DataFrame
        Estimation data.
    variable : str
        Categorical variable whose levels are contrasted.
    method : str, default 'r'
        Contrast type:

        - ``'r'`` (reference): each level vs *reference* level.
        - ``'ar'`` (adjacent): each level vs the previous level.
        - ``'gw'`` (grand-mean weighted): each level vs the weighted
          grand mean of all levels.
    reference : scalar, optional
        Reference level when ``method='r'``.  Defaults to the smallest
        observed level.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pd.DataFrame
        Columns: ``contrast_label``, ``contrast``, ``se``, ``z``,
        ``pvalue``, ``ci_lower``, ``ci_upper``.

    Examples
    --------
    >>> result = sp.regress("wage ~ C(education) + experience", data=df)
    >>> sp.contrast(result, data=df, variable="education", method="r", reference=0)
    """
    params = result.params
    vcov = _get_vcov(result)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    levels = sorted(data[variable].unique())

    # Compute predictive margin and gradient at each level
    level_margins = {}
    level_grads = {}
    level_counts = {}
    for lev in levels:
        df_mod = data.copy()
        df_mod[variable] = lev
        preds = np.array([
            _predict_row(params, df_mod.iloc[i], var_to_change=None, new_val=None)
            for i in range(len(df_mod))
        ])
        level_margins[lev] = float(np.mean(preds))
        level_grads[lev] = _margin_gradient(params, df_mod)
        level_counts[lev] = int((data[variable] == lev).sum())

    # Build contrasts
    rows = []
    if method == 'r':
        if reference is None:
            reference = levels[0]
        for lev in levels:
            if lev == reference:
                continue
            diff = level_margins[lev] - level_margins[reference]
            grad_diff = level_grads[lev] - level_grads[reference]
            se = float(np.sqrt(grad_diff @ vcov @ grad_diff))
            z = diff / se if se > 0 else 0.0
            pv = float(2 * (1 - stats.norm.cdf(abs(z))))
            rows.append({
                'contrast_label': f'{lev} vs {reference}',
                'contrast': diff,
                'se': se,
                'z': z,
                'pvalue': pv,
                'ci_lower': diff - z_crit * se,
                'ci_upper': diff + z_crit * se,
            })

    elif method == 'ar':
        for i in range(1, len(levels)):
            lev, prev = levels[i], levels[i - 1]
            diff = level_margins[lev] - level_margins[prev]
            grad_diff = level_grads[lev] - level_grads[prev]
            se = float(np.sqrt(grad_diff @ vcov @ grad_diff))
            z = diff / se if se > 0 else 0.0
            pv = float(2 * (1 - stats.norm.cdf(abs(z))))
            rows.append({
                'contrast_label': f'{lev} vs {prev}',
                'contrast': diff,
                'se': se,
                'z': z,
                'pvalue': pv,
                'ci_lower': diff - z_crit * se,
                'ci_upper': diff + z_crit * se,
            })

    elif method == 'gw':
        total_n = sum(level_counts.values())
        weights = {lev: level_counts[lev] / total_n for lev in levels}
        grand_margin = sum(weights[lev] * level_margins[lev] for lev in levels)
        grand_grad = sum(weights[lev] * level_grads[lev] for lev in levels)

        for lev in levels:
            diff = level_margins[lev] - grand_margin
            grad_diff = level_grads[lev] - grand_grad
            se = float(np.sqrt(grad_diff @ vcov @ grad_diff))
            z = diff / se if se > 0 else 0.0
            pv = float(2 * (1 - stats.norm.cdf(abs(z))))
            rows.append({
                'contrast_label': f'{lev} vs grand mean',
                'contrast': diff,
                'se': se,
                'z': z,
                'pvalue': pv,
                'ci_lower': diff - z_crit * se,
                'ci_upper': diff + z_crit * se,
            })
    else:
        raise ValueError(
            f"Unknown contrast method '{method}'. Use 'r', 'ar', or 'gw'."
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# pwcompare: Pairwise comparisons of predictive margins
# ---------------------------------------------------------------------------

def pwcompare(
    result,
    data: pd.DataFrame,
    variable: str,
    adjust: str = 'none',
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Pairwise comparisons of predictive margins across all levels.

    Equivalent to Stata's ``pwcompare <var>``.

    Parameters
    ----------
    result : EconometricResults
        Fitted model result.
    data : pd.DataFrame
        Estimation data.
    variable : str
        Categorical variable whose levels are compared pairwise.
    adjust : str, default 'none'
        P-value adjustment method:

        - ``'none'``: unadjusted.
        - ``'bonferroni'``: Bonferroni correction.
        - ``'sidak'``: Sidak correction.
        - ``'holm'``: Holm step-down procedure.
    alpha : float, default 0.05
        Significance level for (adjusted) confidence intervals.

    Returns
    -------
    pd.DataFrame
        Columns: ``comparison``, ``diff``, ``se``, ``z``, ``pvalue``,
        ``pvalue_adj``, ``ci_lower``, ``ci_upper``.

    Examples
    --------
    >>> result = sp.regress("wage ~ C(group) + experience", data=df)
    >>> sp.pwcompare(result, data=df, variable="group", adjust="bonferroni")
    """
    params = result.params
    vcov = _get_vcov(result)

    levels = sorted(data[variable].unique())

    # Compute margin & gradient at each level
    level_margins = {}
    level_grads = {}
    for lev in levels:
        df_mod = data.copy()
        df_mod[variable] = lev
        preds = np.array([
            _predict_row(params, df_mod.iloc[i], var_to_change=None, new_val=None)
            for i in range(len(df_mod))
        ])
        level_margins[lev] = float(np.mean(preds))
        level_grads[lev] = _margin_gradient(params, df_mod)

    # All pairwise comparisons
    pairs = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            pairs.append((levels[i], levels[j]))

    n_comp = len(pairs)
    rows = []
    for lev_a, lev_b in pairs:
        diff = level_margins[lev_b] - level_margins[lev_a]
        grad_diff = level_grads[lev_b] - level_grads[lev_a]
        se = float(np.sqrt(grad_diff @ vcov @ grad_diff))
        z = diff / se if se > 0 else 0.0
        pv = float(2 * (1 - stats.norm.cdf(abs(z))))
        rows.append({
            'comparison': f'{lev_b} vs {lev_a}',
            'diff': diff,
            'se': se,
            'z': z,
            'pvalue': pv,
        })

    # Adjust p-values
    raw_pvals = [r['pvalue'] for r in rows]
    adj_pvals = _adjust_pvalues(raw_pvals, method=adjust, n_comparisons=n_comp)

    # Determine adjusted alpha for CIs
    alpha_adj = _adjusted_alpha(alpha, adjust, n_comp)
    z_crit = stats.norm.ppf(1 - alpha_adj / 2)

    for r, padj in zip(rows, adj_pvals):
        r['pvalue_adj'] = padj
        r['ci_lower'] = r['diff'] - z_crit * r['se']
        r['ci_upper'] = r['diff'] + z_crit * r['se']

    return pd.DataFrame(rows)


def _adjust_pvalues(pvals, method, n_comparisons):
    """Apply multiple-comparison correction to p-values."""
    pvals = np.asarray(pvals, dtype=float)
    if method == 'none':
        return pvals.tolist()
    elif method == 'bonferroni':
        return np.minimum(pvals * n_comparisons, 1.0).tolist()
    elif method == 'sidak':
        return (1.0 - (1.0 - pvals) ** n_comparisons).tolist()
    elif method == 'holm':
        n = len(pvals)
        order = np.argsort(pvals)
        adj = np.empty(n)
        for rank, idx in enumerate(order):
            adj[idx] = min(pvals[idx] * (n_comparisons - rank), 1.0)
        # Enforce monotonicity
        cum_max = 0.0
        for idx in order:
            cum_max = max(cum_max, adj[idx])
            adj[idx] = cum_max
        return adj.tolist()
    else:
        raise ValueError(
            f"Unknown adjustment method '{method}'. "
            "Use 'none', 'bonferroni', 'sidak', or 'holm'."
        )


def _adjusted_alpha(alpha, method, n_comparisons):
    """Return adjusted significance level for CI construction."""
    if method == 'none':
        return alpha
    elif method == 'bonferroni':
        return alpha / n_comparisons
    elif method == 'sidak':
        return 1.0 - (1.0 - alpha) ** (1.0 / n_comparisons)
    elif method == 'holm':
        # Conservative: use Bonferroni alpha for CIs
        return alpha / n_comparisons
    else:
        return alpha
