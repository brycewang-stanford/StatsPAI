"""
CATE diagnostics and visualisation for Meta-Learners.

Provides tools for analysing and presenting heterogeneous treatment
effect estimates:
- ``cate_summary``: descriptive statistics of CATE distribution
- ``cate_plot``: histogram / density of individual-level effects
- ``cate_by_group``: group-level ATE with CIs (e.g., by quartile)
- ``cate_importance``: variable importance for CATE heterogeneity
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def cate_summary(result: CausalResult) -> pd.DataFrame:
    """
    Descriptive statistics of the CATE distribution.

    Parameters
    ----------
    result : CausalResult
        Result from ``metalearner()`` containing ``model_info['cate']``.

    Returns
    -------
    pd.DataFrame
        Summary statistics: mean, sd, min, q25, median, q75, max,
        fraction positive, fraction significant (> 2*SE away from 0).
    """
    cate = _extract_cate(result)

    summary = {
        'Mean (ATE)': np.mean(cate),
        'Std. Dev.': np.std(cate, ddof=1),
        'Min': np.min(cate),
        'Q25': np.percentile(cate, 25),
        'Median': np.median(cate),
        'Q75': np.percentile(cate, 75),
        'Max': np.max(cate),
        'Frac. Positive': np.mean(cate > 0),
        'IQR': np.percentile(cate, 75) - np.percentile(cate, 25),
        'N': len(cate),
    }
    return pd.DataFrame(summary, index=['CATE']).T


def cate_by_group(
    result: CausalResult,
    data: pd.DataFrame,
    by: str,
    n_groups: int = 4,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Group-level average treatment effects.

    Splits the CATE distribution by a covariate (or by CATE quartiles
    if ``by='cate'``) and reports group means with confidence intervals.

    Parameters
    ----------
    result : CausalResult
        Result from ``metalearner()``.
    data : pd.DataFrame
        Original data (same rows as the estimation sample).
    by : str
        Column name to group by, or 'cate' to group by CATE quartiles.
    n_groups : int, default 4
        Number of quantile groups when ``by='cate'`` or when the
        grouping variable is continuous.
    alpha : float, default 0.05
        Significance level for CIs.

    Returns
    -------
    pd.DataFrame
        Columns: group, n, mean_cate, se, ci_lower, ci_upper.
    """
    cate = _extract_cate(result)

    if by == 'cate':
        labels = pd.qcut(cate, q=n_groups, labels=False, duplicates='drop')
        group_name = 'CATE Quartile'
    else:
        if by not in data.columns:
            raise ValueError(f"Column '{by}' not found in data")
        col = data[by].values[:len(cate)]
        if pd.api.types.is_numeric_dtype(col) and len(np.unique(col)) > n_groups:
            labels = pd.qcut(col, q=n_groups, labels=False, duplicates='drop')
        else:
            labels = col
        group_name = by

    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for g in sorted(np.unique(labels)):
        mask = labels == g
        c = cate[mask]
        n = len(c)
        m = float(np.mean(c))
        se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        rows.append({
            'group': g,
            'n': n,
            'mean_cate': m,
            'se': se,
            'ci_lower': m - z * se,
            'ci_upper': m + z * se,
        })

    df = pd.DataFrame(rows)
    df.index.name = group_name
    return df


def cate_plot(
    result: CausalResult,
    kind: str = 'hist',
    ax=None,
    figsize: tuple = (8, 5),
    color: str = '#2C3E50',
    title: Optional[str] = None,
    **kwargs,
):
    """
    Plot the CATE distribution.

    Parameters
    ----------
    result : CausalResult
        Result from ``metalearner()``.
    kind : str, default 'hist'
        'hist' for histogram, 'kde' for kernel density, 'both'.
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
        raise ImportError("matplotlib required for plotting")

    cate = _extract_cate(result)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if kind in ('hist', 'both'):
        ax.hist(cate, bins=kwargs.get('bins', 40), density=True,
                alpha=0.6, color=color, edgecolor='white', linewidth=0.5)
    if kind in ('kde', 'both'):
        from scipy.stats import gaussian_kde
        xs = np.linspace(cate.min(), cate.max(), 300)
        kde = gaussian_kde(cate)
        ax.plot(xs, kde(xs), color=color, linewidth=2)

    # Mark ATE
    ate = np.mean(cate)
    ax.axvline(ate, color='#E74C3C', linestyle='--', linewidth=1.5,
               label=f'ATE = {ate:.3f}')
    ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6)

    learner_name = result.model_info.get('learner', 'Meta-Learner')
    ax.set_xlabel('Conditional Average Treatment Effect (CATE)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(title or f'CATE Distribution ({learner_name})', fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, ax


def cate_group_plot(
    group_df: pd.DataFrame,
    ax=None,
    figsize: tuple = (8, 5),
    color: str = '#2C3E50',
    title: Optional[str] = None,
):
    """
    Plot group-level CATEs with confidence intervals.

    Parameters
    ----------
    group_df : pd.DataFrame
        Output from ``cate_by_group()``.
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
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    groups = group_df['group'].values
    means = group_df['mean_cate'].values
    ci_lo = group_df['ci_lower'].values
    ci_hi = group_df['ci_upper'].values

    x = np.arange(len(groups))
    ax.bar(x, means, color=color, alpha=0.7, edgecolor='white')
    ax.errorbar(x, means,
                yerr=[means - ci_lo, ci_hi - means],
                fmt='none', color='black', capsize=4, linewidth=1.2)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_ylabel('Mean CATE', fontsize=11)
    ax.set_title(title or 'Group-Level Treatment Effects', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, ax


def blp_test(
    result: CausalResult,
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    n_folds: int = 5,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Best Linear Predictor (BLP) test for CATE heterogeneity.

    Implements the calibration test from Chernozhukov et al. (2018,
    *Econometrica*) "Generic Machine Learning Inference on Heterogeneous
    Treatment Effects." Equivalent to ``grf::test_calibration()`` in R.

    Fits via OLS:
        Y_i = alpha + beta_1 * (D_i - e(X_i)) + beta_2 * (D_i - e(X_i)) * (S(X_i) - mean(S)) + eps

    where S(X) is the CATE proxy from the meta-learner and e(X) the
    propensity score (estimated via cross-fitting).

    - **beta_1**: tests whether ATE != 0 (mean forest prediction)
    - **beta_2**: tests whether CATE heterogeneity is real
      (if beta_2 > 0 and significant, the learner has found genuine
      heterogeneity rather than noise)

    Parameters
    ----------
    result : CausalResult
        Result from ``metalearner()``.
    data : pd.DataFrame
        Original data.
    y, treat : str
        Outcome and treatment column names.
    covariates : list of str
        Covariate column names.
    n_folds : int, default 5
        Folds for propensity cross-fitting.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys: 'beta1' (ATE signal), 'beta1_se', 'beta1_pvalue',
        'beta2' (heterogeneity signal), 'beta2_se', 'beta2_pvalue',
        'heterogeneity_significant' (bool).

    References
    ----------
    Chernozhukov, V., Demirer, M., Duflo, E., & Fernandez-Val, I. (2018).
    Generic Machine Learning Inference on Heterogeneous Treatment Effects
    in Randomized Experiments. *Econometrica* (forthcoming as of 2018 NBER WP).
    """
    import statsmodels.api as sm
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    cate = _extract_cate(result)
    n = len(cate)

    # Extract arrays
    Y = data[y].values[:n].astype(float)
    D = data[treat].values[:n].astype(float)
    X = data[covariates].values[:n].astype(float)

    # Cross-fit propensity score
    from sklearn.ensemble import GradientBoostingClassifier
    prop_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42,
    )
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    e_hat = np.zeros(n)
    for train_idx, test_idx in kf.split(X):
        m = clone(prop_model)
        m.fit(X[train_idx], D[train_idx])
        e_hat[test_idx] = m.predict_proba(X[test_idx])[:, 1]
    e_hat = np.clip(e_hat, 0.01, 0.99)

    # BLP regression
    D_centered = D - e_hat
    S_centered = cate - np.mean(cate)

    # Y = alpha + beta1 * (D - e(X)) + beta2 * (D - e(X)) * (S(X) - S_bar) + eps
    Z = np.column_stack([
        np.ones(n),
        D_centered,
        D_centered * S_centered,
    ])

    ols = sm.OLS(Y, Z).fit(cov_type='HC1')

    beta1 = float(ols.params[1])
    beta1_se = float(ols.bse[1])
    beta1_pv = float(ols.pvalues[1])
    beta2 = float(ols.params[2])
    beta2_se = float(ols.bse[2])
    beta2_pv = float(ols.pvalues[2])

    return {
        'beta1': beta1,
        'beta1_se': beta1_se,
        'beta1_pvalue': beta1_pv,
        'beta2': beta2,
        'beta2_se': beta2_se,
        'beta2_pvalue': beta2_pv,
        'heterogeneity_significant': beta2_pv < alpha,
    }


def gate_test(
    result: CausalResult,
    data: pd.DataFrame,
    by: str,
    n_groups: int = 4,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Test for significant heterogeneity across GATE (Group ATE) groups.

    Performs two tests:
    1. **Omnibus F-test**: are all group CATEs equal? (ANOVA)
    2. **Top-vs-bottom**: is the highest-CATE group significantly
       different from the lowest?

    Parameters
    ----------
    result : CausalResult
        Result from ``metalearner()``.
    data : pd.DataFrame
        Original data (same rows as estimation).
    by : str
        Column name to group by, or 'cate' for CATE quartiles.
    n_groups : int, default 4
        Number of groups.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys: 'gate_table' (DataFrame), 'omnibus_F', 'omnibus_pvalue',
        'top_vs_bottom_diff', 'top_vs_bottom_se', 'top_vs_bottom_pvalue'.
    """
    cate = _extract_cate(result)
    gate_df = cate_by_group(result, data, by=by, n_groups=n_groups, alpha=alpha)

    # Omnibus: one-way ANOVA across groups
    if by == 'cate':
        labels = pd.qcut(cate, q=n_groups, labels=False, duplicates='drop')
    else:
        col = data[by].values[:len(cate)]
        if pd.api.types.is_numeric_dtype(col) and len(np.unique(col)) > n_groups:
            labels = pd.qcut(col, q=n_groups, labels=False, duplicates='drop')
        else:
            labels = col

    groups_list = [cate[labels == g] for g in sorted(np.unique(labels))]
    if len(groups_list) >= 2:
        f_stat, f_pvalue = stats.f_oneway(*groups_list)
    else:
        f_stat, f_pvalue = np.nan, np.nan

    # Top vs bottom group
    sorted_gate = gate_df.sort_values('mean_cate')
    bottom = sorted_gate.iloc[0]
    top = sorted_gate.iloc[-1]
    diff = top['mean_cate'] - bottom['mean_cate']
    se_diff = np.sqrt(top['se'] ** 2 + bottom['se'] ** 2)
    if se_diff > 0:
        z = diff / se_diff
        tvb_pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))
    else:
        tvb_pvalue = np.nan

    return {
        'gate_table': gate_df,
        'omnibus_F': float(f_stat),
        'omnibus_pvalue': float(f_pvalue),
        'top_vs_bottom_diff': float(diff),
        'top_vs_bottom_se': float(se_diff),
        'top_vs_bottom_pvalue': float(tvb_pvalue),
    }


def compare_metalearners(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    learners: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Fit multiple meta-learners and compare their ATE estimates.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariate / effect modifier variables.
    learners : list of str, optional
        Which learners to compare. Default: all five ('s','t','x','r','dr').
    **kwargs
        Additional arguments passed to ``metalearner()``.

    Returns
    -------
    pd.DataFrame
        Comparison table with columns: learner, ate, se, ci_lower,
        ci_upper, pvalue, cate_std, cate_iqr.

    Examples
    --------
    >>> import statspai as sp
    >>> comp = sp.compare_metalearners(df, y='wage', treat='training',
    ...                                 covariates=['age', 'edu'])
    >>> print(comp)
    """
    from .metalearners import metalearner as _metalearner

    if learners is None:
        learners = ['s', 't', 'x', 'r', 'dr']

    learner_names = {
        's': 'S-Learner', 't': 'T-Learner', 'x': 'X-Learner',
        'r': 'R-Learner', 'dr': 'DR-Learner',
    }

    rows = []
    for lr in learners:
        result = _metalearner(
            data, y=y, treat=treat, covariates=covariates,
            learner=lr, **kwargs,
        )
        cate = result.model_info['cate']
        rows.append({
            'learner': learner_names.get(lr, lr),
            'ate': result.estimate,
            'se': result.se,
            'ci_lower': result.ci[0],
            'ci_upper': result.ci[1],
            'pvalue': result.pvalue,
            'se_method': result.model_info.get('se_method', 'bootstrap'),
            'cate_std': float(np.std(cate)),
            'cate_iqr': float(np.percentile(cate, 75) - np.percentile(cate, 25)),
        })

    return pd.DataFrame(rows)


def predict_cate(
    result: CausalResult,
    new_data: pd.DataFrame,
) -> np.ndarray:
    """
    Predict CATE on new (out-of-sample) data.

    Parameters
    ----------
    result : CausalResult
        Result from ``metalearner()`` containing a fitted estimator.
    new_data : pd.DataFrame
        New data with the same covariate columns used in estimation.

    Returns
    -------
    np.ndarray
        Predicted CATE for each row of new_data.
    """
    if '_estimator' not in result.model_info:
        raise ValueError(
            "Result does not contain a fitted estimator. "
            "Use metalearner() to produce a result."
        )
    covariates = result.model_info.get('covariates')
    if covariates is None:
        raise ValueError("Result does not contain covariate names.")
    for c in covariates:
        if c not in new_data.columns:
            raise ValueError(f"Column '{c}' not found in new_data")

    X_new = new_data[covariates].values.astype(float)
    est = result.model_info['_estimator']
    return est.effect(X_new)


# ======================================================================
# Internal
# ======================================================================

def _extract_cate(result: CausalResult) -> np.ndarray:
    """Extract CATE array from CausalResult."""
    if not hasattr(result, 'model_info') or 'cate' not in result.model_info:
        raise ValueError(
            "Result does not contain CATE estimates. "
            "Use metalearner() to produce a result with individual effects."
        )
    return np.asarray(result.model_info['cate'])
