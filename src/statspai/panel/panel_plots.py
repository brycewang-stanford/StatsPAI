"""
Panel-specific visualization functions.

All plots follow StatsPAI's academic theme conventions and return (fig, ax).

Functions
---------
- ``plot_coef``             Coefficient forest plot for a panel result
- ``plot_effects``          Distribution of estimated entity fixed effects
- ``plot_residuals``        Residual diagnostics (by entity, by time, QQ)
- ``plot_within_between``   Within vs between variation scatter
- ``plot_compare``          Side-by-side coefficient comparison across methods
"""

from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .panel_reg import PanelResults

# Common academic color palette
_COLORS = [
    '#2C3E50', '#E74C3C', '#3498DB', '#2ECC71',
    '#9B59B6', '#F39C12', '#1ABC9C', '#E67E22',
]


def _get_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. "
            "Install: pip install matplotlib"
        )


# ======================================================================
# Coefficient plot
# ======================================================================

def plot_coef(
    result: 'PanelResults',
    variables: Optional[List[str]] = None,
    ax=None,
    figsize: tuple = (8, 5),
    color: str = '#2C3E50',
    title: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple:
    """
    Coefficient forest plot for a panel regression result.

    Parameters
    ----------
    result : PanelResults
    variables : list of str, optional
        Which variables to plot. Default: all.
    ax : matplotlib Axes, optional
    figsize : tuple
    color : str
    title : str, optional
    alpha : float
        Significance level for CIs.

    Returns
    -------
    (fig, ax)
    """
    plt = _get_plt()
    from scipy import stats as sp_stats

    params = result.params
    se = result.std_errors

    if variables is not None:
        params = params[[v for v in variables if v in params.index]]
        se = se[[v for v in variables if v in se.index]]

    # Filter out Mundlak/Chamberlain internal vars for cleaner plots
    mask = ~params.index.str.startswith('_')
    params = params[mask]
    se = se[mask]

    var_names = list(params.index)
    coefs = params.values
    ses = se.values
    n_vars = len(var_names)

    t_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lo = coefs - t_crit * ses
    ci_hi = coefs + t_crit * ses

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_pos = np.arange(n_vars)
    ax.errorbar(
        coefs, y_pos,
        xerr=[coefs - ci_lo, ci_hi - coefs],
        fmt='o', color=color, capsize=4, markersize=6,
        linewidth=1.5, markeredgewidth=1.5,
    )
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(var_names)
    ax.invert_yaxis()

    model_name = result.model_info.get('model_type', 'Panel')
    ax.set_title(title or f'{model_name}: Coefficients', fontsize=13)
    ax.set_xlabel('Estimate', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ======================================================================
# Entity fixed effects distribution
# ======================================================================

def plot_effects(
    result: 'PanelResults',
    ax=None,
    figsize: tuple = (8, 5),
    color: str = '#2C3E50',
    bins: int = 30,
    title: Optional[str] = None,
    kind: str = 'hist',
) -> Tuple:
    """
    Distribution of estimated entity fixed effects.

    Only available when the model was estimated with FE or two-way FE
    via linearmodels (which stores estimated effects on the result).

    Parameters
    ----------
    result : PanelResults
    ax : matplotlib Axes, optional
    figsize : tuple
    color : str
    bins : int
    title : str, optional
    kind : str
        ``'hist'`` for histogram, ``'kde'`` for kernel density,
        ``'both'`` for overlaid histogram + KDE.

    Returns
    -------
    (fig, ax)
    """
    plt = _get_plt()

    lm_result = result._lm_result
    if lm_result is None:
        raise ValueError(
            "Entity effects not available — this method does not "
            "produce entity-level effects (only FE/twoway via linearmodels)."
        )

    # Extract estimated effects
    if hasattr(lm_result, 'estimated_effects'):
        effects = lm_result.estimated_effects.values.ravel()
    else:
        raise ValueError(
            "Entity effects not available for this model type. "
            "Use method='fe' or method='twoway'."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if kind in ('hist', 'both'):
        ax.hist(effects, bins=bins, color=color, alpha=0.6,
                edgecolor='white', linewidth=0.5, density=(kind == 'both'))

    if kind in ('kde', 'both'):
        from scipy.stats import gaussian_kde
        x_grid = np.linspace(effects.min(), effects.max(), 200)
        kde = gaussian_kde(effects)
        ax.plot(x_grid, kde(x_grid), color=color, linewidth=2)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('Estimated Entity Effect', fontsize=11)
    ax.set_ylabel('Density' if kind != 'hist' else 'Frequency', fontsize=11)
    ax.set_title(title or 'Distribution of Entity Fixed Effects', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotation
    ax.text(
        0.98, 0.95,
        f'N = {len(effects):,}\nMean = {np.mean(effects):.3f}\nSD = {np.std(effects):.3f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )

    fig.tight_layout()
    return fig, ax


# ======================================================================
# Residual diagnostics
# ======================================================================

def plot_residuals(
    result: 'PanelResults',
    figsize: tuple = (14, 10),
    color: str = '#2C3E50',
    title: Optional[str] = None,
) -> Tuple:
    """
    Panel residual diagnostic plots (2x2 grid).

    - Top-left: Residuals vs fitted values
    - Top-right: Residual distribution (histogram + KDE)
    - Bottom-left: Mean residual by entity (top/bottom 20)
    - Bottom-right: Mean residual by time period

    Parameters
    ----------
    result : PanelResults
    figsize : tuple
    color : str
    title : str, optional

    Returns
    -------
    (fig, axes)
    """
    plt = _get_plt()
    from scipy.stats import gaussian_kde

    resids = result.data_info.get('residuals')
    fitted = result.data_info.get('fitted_values')
    if resids is None or fitted is None:
        raise ValueError("Residuals/fitted values not available.")

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # --- (0,0) Residuals vs Fitted ---
    ax = axes[0, 0]
    ax.scatter(fitted, resids, alpha=0.3, s=10, color=color)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Fitted Values', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    ax.set_title('Residuals vs Fitted', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- (0,1) Residual distribution ---
    ax = axes[0, 1]
    ax.hist(resids, bins=40, color=color, alpha=0.6,
            edgecolor='white', linewidth=0.5, density=True)
    x_grid = np.linspace(resids.min(), resids.max(), 200)
    try:
        kde = gaussian_kde(resids)
        ax.plot(x_grid, kde(x_grid), color='#E74C3C', linewidth=1.5)
    except Exception:
        pass
    ax.set_xlabel('Residual', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Residual Distribution', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- (1,0) Mean residual by entity (top/bottom 20) ---
    ax = axes[1, 0]
    if result._panel_data is not None and result._entity is not None:
        panel_df = result._panel_data.copy()
        panel_df['_resid'] = np.nan
        # Align residuals back to panel data
        # Residuals are aligned to the fitted model's index
        if result._lm_result is not None:
            try:
                resid_series = result._lm_result.resids
                panel_indexed = panel_df.set_index([result._entity, result._time])
                panel_indexed['_resid'] = resid_series
                mean_by_entity = panel_indexed.groupby(level=0)['_resid'].mean().dropna()
                # Show top/bottom 20
                n_show = min(20, len(mean_by_entity))
                extreme = pd.concat([
                    mean_by_entity.nsmallest(n_show // 2),
                    mean_by_entity.nlargest(n_show // 2),
                ]).sort_values()
                y_pos = np.arange(len(extreme))
                colors_bar = [('#E74C3C' if v < 0 else '#3498DB') for v in extreme.values]
                ax.barh(y_pos, extreme.values, color=colors_bar, alpha=0.7, height=0.7)
                ax.set_yticks(y_pos)
                labels = [str(x)[:12] for x in extreme.index]
                ax.set_yticklabels(labels, fontsize=8)
                ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
            except Exception:
                ax.text(0.5, 0.5, 'Entity residuals\nnot available',
                        transform=ax.transAxes, ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'Entity residuals\nnot available',
                    transform=ax.transAxes, ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'Entity data\nnot stored',
                transform=ax.transAxes, ha='center', va='center')
    ax.set_xlabel('Mean Residual', fontsize=10)
    ax.set_title('Mean Residual by Entity (extremes)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- (1,1) Mean residual by time ---
    ax = axes[1, 1]
    if result._panel_data is not None and result._time is not None and result._lm_result is not None:
        try:
            resid_series = result._lm_result.resids
            panel_indexed = result._panel_data.set_index([result._entity, result._time])
            panel_indexed['_resid'] = resid_series
            mean_by_time = panel_indexed.groupby(level=1)['_resid'].mean().dropna()
            mean_by_time = mean_by_time.sort_index()
            ax.plot(range(len(mean_by_time)), mean_by_time.values,
                    marker='o', color=color, markersize=5, linewidth=1.2)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_xticks(range(len(mean_by_time)))
            labels = [str(x) for x in mean_by_time.index]
            ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        except Exception:
            ax.text(0.5, 0.5, 'Time residuals\nnot available',
                    transform=ax.transAxes, ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'Time data\nnot stored',
                transform=ax.transAxes, ha='center', va='center')
    ax.set_xlabel('Time Period', fontsize=10)
    ax.set_ylabel('Mean Residual', fontsize=10)
    ax.set_title('Mean Residual by Time', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    model_name = result.model_info.get('model_type', 'Panel')
    fig.suptitle(title or f'{model_name}: Residual Diagnostics',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes


# ======================================================================
# Within vs Between variation
# ======================================================================

def plot_within_between(
    data: pd.DataFrame,
    variables: List[str],
    entity: str,
    ax=None,
    figsize: tuple = (8, 6),
    color: str = '#2C3E50',
    title: Optional[str] = None,
) -> Tuple:
    """
    Bar chart comparing within vs between variation for each variable.

    Helps assess which estimator is appropriate:
    - High between / low within → BE may be efficient
    - High within / low between → FE captures the action
    - Similar → both capture similar information

    Parameters
    ----------
    data : pd.DataFrame
    variables : list of str
        Variables to decompose.
    entity : str
        Entity identifier column.
    ax : matplotlib Axes, optional
    figsize : tuple
    color : str
    title : str, optional

    Returns
    -------
    (fig, ax)
    """
    plt = _get_plt()

    rows = []
    for var in variables:
        vals = data[var].astype(float)
        total_var = vals.var()
        if total_var == 0:
            rows.append({'Variable': var, 'Between': 0, 'Within': 0})
            continue
        group_means = data.groupby(entity)[var].transform('mean')
        between_var = group_means.var()
        within_var = (vals - group_means).var()
        rows.append({
            'Variable': var,
            'Between': between_var,
            'Within': within_var,
        })

    df_var = pd.DataFrame(rows)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x = np.arange(len(df_var))
    width = 0.35
    ax.bar(x - width / 2, df_var['Between'], width, label='Between',
           color='#3498DB', alpha=0.8)
    ax.bar(x + width / 2, df_var['Within'], width, label='Within',
           color='#E74C3C', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(df_var['Variable'], fontsize=10)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title(title or 'Within vs Between Variation', fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ======================================================================
# Multi-method coefficient comparison
# ======================================================================

def plot_compare(
    results: Dict[str, 'PanelResults'],
    variables: Optional[List[str]] = None,
    ax=None,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple:
    """
    Side-by-side coefficient comparison across multiple panel methods.

    Parameters
    ----------
    results : dict
        ``{name: PanelResults}`` for each method.
    variables : list of str, optional
        Which variables to plot. Default: all shared (excluding internals).
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    alpha : float

    Returns
    -------
    (fig, ax)
    """
    plt = _get_plt()
    from scipy import stats as sp_stats

    names = list(results.keys())
    n_models = len(names)

    # Gather all variables (excluding Mundlak/Chamberlain internals)
    if variables is None:
        all_vars = []
        for r in results.values():
            for v in r.params.index:
                if not v.startswith('_') and v != 'const' and v not in all_vars:
                    all_vars.append(v)
        variables = all_vars

    n_vars = len(variables)
    t_crit = sp_stats.norm.ppf(1 - alpha / 2)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    offsets = np.linspace(-0.15, 0.15, n_models)

    for i, (name, r) in enumerate(results.items()):
        coefs = []
        ci_lo = []
        ci_hi = []
        for var in variables:
            c = r.params.get(var, np.nan)
            s = r.std_errors.get(var, np.nan)
            coefs.append(c)
            ci_lo.append(c - t_crit * s)
            ci_hi.append(c + t_crit * s)

        coefs = np.array(coefs)
        ci_lo = np.array(ci_lo)
        ci_hi = np.array(ci_hi)
        y_pos = np.arange(n_vars) + offsets[i]

        ax.errorbar(
            coefs, y_pos,
            xerr=[coefs - ci_lo, ci_hi - coefs],
            fmt='o', color=_COLORS[i % len(_COLORS)],
            capsize=3, markersize=5, linewidth=1.2,
            label=name,
        )

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_yticks(np.arange(n_vars))
    ax.set_yticklabels(variables)
    ax.invert_yaxis()
    ax.set_xlabel('Estimate', fontsize=11)
    ax.set_title(title or 'Coefficient Comparison Across Methods', fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ======================================================================
# Hausman visual comparison
# ======================================================================

def plot_hausman(
    result: 'PanelResults',
    ax=None,
    figsize: tuple = (8, 5),
    title: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple:
    """
    Visual Hausman test: FE vs RE coefficients with CIs.

    Plots both FE and RE estimates for each variable side by side.
    Large differences suggest FE is needed.

    Parameters
    ----------
    result : PanelResults
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    alpha : float

    Returns
    -------
    (fig, ax)
    """
    plt = _get_plt()
    from .panel_reg import panel

    # Get both FE and RE results
    fe_r = panel(result._panel_data, result._formula,
                 result._entity, result._time, method='fe')
    re_r = panel(result._panel_data, result._formula,
                 result._entity, result._time, method='re')

    plot_results = {'FE (Within)': fe_r, 'RE (GLS)': re_r}

    # Filter to shared variables
    shared_vars = [v for v in fe_r.params.index
                   if v in re_r.params.index and v != 'const']

    fig, ax_out = plot_compare(plot_results, variables=shared_vars,
                               ax=ax, figsize=figsize, title=title, alpha=alpha)

    # Add Hausman test result as annotation
    try:
        h = result.hausman_test(alpha=alpha)
        ax_out.text(
            0.02, 0.02,
            f"Hausman: χ²({h['df']}) = {h['statistic']:.2f}, "
            f"p = {h['pvalue']:.4f} → {h['recommendation']}",
            transform=ax_out.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9),
        )
    except Exception:
        pass

    return fig, ax_out
