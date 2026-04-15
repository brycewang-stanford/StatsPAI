"""
DID Visualisation Suite.

Publication-quality plots for Difference-in-Differences analysis,
inspired by Stata's ``event_plot``, ``bacondecomp``, and R's
``did`` / ``fixest::iplot()`` / ``HonestDiD`` packages.

Available plots
---------------
- ``parallel_trends_plot``    — raw outcome means over time by group
- ``bacon_plot``              — Goodman-Bacon decomposition scatter
- ``group_time_plot``         — Callaway-Sant'Anna (g,t) ATT dot/heatmap
- ``did_plot``                — classic DID 2×2 diagram with counterfactual
- ``event_study_plot``        — enhanced event study with pre/post shading
- ``treatment_rollout_plot``  — staggered treatment timing visualisation
- ``sensitivity_plot``        — Rambachan-Roth honest DID sensitivity
- ``cohort_event_study_plot`` — per-cohort event study overlay

All functions return ``(fig, ax)`` and accept an optional ``ax`` argument
for embedding in multi-panel figures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _ensure_mpl():
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        return plt, matplotlib
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install: pip install matplotlib"
        )


def _style_ax(ax):
    """Apply clean academic styling to axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)


# ======================================================================
# 1. Parallel Trends Plot
# ======================================================================

def parallel_trends_plot(
    data: pd.DataFrame,
    y: str,
    time: str,
    treat: str,
    id: Optional[str] = None,
    treat_time: Optional[Union[int, float]] = None,
    agg: str = 'mean',
    labels: Optional[Dict] = None,
    colors: Optional[Tuple[str, str]] = None,
    ci: bool = True,
    ax=None,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    **kwargs,
):
    """
    Plot raw outcome means over time for treatment and control groups.

    The workhorse pre-analysis plot: shows whether parallel trends
    is plausible before running DID.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable.
    time : str
        Time period variable.
    treat : str
        Treatment group indicator. Binary (0/1) for 2×2, or
        first-treatment-period for staggered (0 = never treated).
    id : str, optional
        Unit identifier (for panel data).
    treat_time : int or float, optional
        Treatment onset time. Draws a vertical line if provided.
    agg : str, default 'mean'
        Aggregation function: 'mean' or 'median'.
    labels : dict, optional
        Custom labels, e.g. ``{'treat': 'New Jersey', 'control': 'Pennsylvania'}``.
    colors : tuple of str, optional
        Colors for (treatment, control). Default: ('#E74C3C', '#2C3E50').
    ci : bool, default True
        Show 95% confidence intervals (±1.96 SE of mean).
    ax : matplotlib Axes, optional
        Existing axes to plot on.
    figsize : tuple, default (10, 6)
        Figure size.
    title : str, optional
        Plot title.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> parallel_trends_plot(df, y='wage', time='year', treat='treated',
    ...                      treat_time=2010)
    """
    plt, _ = _ensure_mpl()
    colors = colors or ('#E74C3C', '#2C3E50')
    labels = labels or {}
    treat_label = labels.get('treat', 'Treatment')
    ctrl_label = labels.get('control', 'Control')

    df = data.copy()

    # Binarize treatment for staggered designs
    if set(df[treat].dropna().unique()) - {0, 1, True, False}:
        # Staggered: treat column is first treatment period, 0 = never
        df['_treat_group'] = (df[treat] > 0).astype(int)
    else:
        df['_treat_group'] = df[treat].astype(int)

    # Aggregate by (time, group)
    agg_func = agg if agg in ('mean', 'median') else 'mean'
    grouped = df.groupby([time, '_treat_group'])[y]

    if agg_func == 'mean':
        means = grouped.mean().reset_index()
        if ci:
            sems = grouped.sem().reset_index()
            sems.columns = [time, '_treat_group', '_se']
            means = means.merge(sems, on=[time, '_treat_group'])
    else:
        means = grouped.median().reset_index()
        ci = False  # no SE for median

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for grp, color, label in [(1, colors[0], treat_label),
                               (0, colors[1], ctrl_label)]:
        mask = means['_treat_group'] == grp
        sub = means[mask].sort_values(time)
        t_vals = sub[time].values
        y_vals = sub[y].values

        ax.plot(t_vals, y_vals, color=color, linewidth=2,
                marker='o', markersize=5, label=label, zorder=5)

        if ci and '_se' in sub.columns:
            se_vals = sub['_se'].values
            ax.fill_between(
                t_vals,
                y_vals - 1.96 * se_vals,
                y_vals + 1.96 * se_vals,
                alpha=0.12, color=color,
            )

    # Treatment onset line
    if treat_time is not None:
        ax.axvline(
            x=treat_time, color='gray', linestyle='--',
            linewidth=1, alpha=0.7, label='Treatment',
        )

    ax.set_xlabel(time.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel(y.replace('_', ' ').title(), fontsize=11)
    ax.set_title(title or 'Parallel Trends', fontsize=13)
    _style_ax(ax)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    return fig, ax


# ======================================================================
# 2. Bacon Decomposition Plot
# ======================================================================

def bacon_plot(
    bacon_result: Dict[str, Any],
    ax=None,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    **kwargs,
):
    """
    Scatter plot of Goodman-Bacon decomposition.

    Each point is a 2×2 sub-comparison: x = weight, y = DD estimate.
    Color distinguishes comparison types (Treated vs Never-treated,
    Earlier vs Later, Later vs Already-treated).

    Parameters
    ----------
    bacon_result : dict
        Output from ``bacon_decomposition()``.
        Must contain ``'decomposition'`` DataFrame and ``'beta_twfe'``.
    ax : matplotlib Axes, optional
    figsize : tuple, default (10, 6)
    title : str, optional
    colors : dict, optional
        Map comparison type → color. Defaults provided.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> bacon = sp.bacon_decomposition(df, y='y', treat='d',
    ...                                 time='t', id='i')
    >>> bacon_plot(bacon)
    """
    plt, _ = _ensure_mpl()

    decomp = bacon_result.get('decomposition')
    if decomp is None or len(decomp) == 0:
        raise ValueError("Bacon decomposition has no sub-comparisons to plot.")

    beta_twfe = bacon_result.get('beta_twfe', None)

    default_colors = {
        'Treated vs Never-treated': '#2C3E50',
        'Earlier vs Later treated': '#27AE60',
        'Later vs Already-treated': '#E74C3C',
    }
    colors = colors or default_colors

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for comp_type in decomp['type'].unique():
        sub = decomp[decomp['type'] == comp_type]
        color = colors.get(comp_type, '#7F8C8D')
        ax.scatter(
            sub['weight'], sub['estimate'],
            color=color, s=80, alpha=0.7, edgecolors='white',
            linewidth=0.5, label=comp_type, zorder=5,
        )

    # TWFE estimate line
    if beta_twfe is not None:
        ax.axhline(
            y=beta_twfe, color='gray', linestyle='--',
            linewidth=1, alpha=0.7,
            label=f'TWFE = {beta_twfe:.4f}',
        )

    ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Weight', fontsize=11)
    ax.set_ylabel('2×2 DD Estimate', fontsize=11)
    ax.set_title(title or 'Goodman-Bacon Decomposition', fontsize=13)
    _style_ax(ax)
    ax.legend(fontsize=9, frameon=False, loc='best')
    fig.tight_layout()
    return fig, ax


# ======================================================================
# 3. Group-Time ATT Plot (Callaway-Sant'Anna)
# ======================================================================

def group_time_plot(
    result,
    plot_type: str = 'dot',
    ax=None,
    figsize: tuple = (12, 7),
    title: Optional[str] = None,
    color: str = '#2C3E50',
    sig_color: str = '#E74C3C',
    insig_color: str = '#BDC3C7',
    alpha_level: float = 0.05,
    **kwargs,
):
    """
    Plot group-time ATT estimates from Callaway-Sant'Anna.

    Two modes:
    - ``'dot'``  — dot plot with CI error bars, colored by significance
    - ``'heatmap'`` — (group × time) heatmap of ATT magnitudes

    Parameters
    ----------
    result : CausalResult
        Result from ``callaway_santanna()`` or ``did(method='cs')``.
        Must have ``detail`` DataFrame with 'group', 'time', 'att' columns.
    plot_type : str, default 'dot'
        'dot' or 'heatmap'.
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    color : str
        Default color for dot plot.
    sig_color : str
        Color for significant estimates.
    insig_color : str
        Color for insignificant estimates.
    alpha_level : float, default 0.05
        Significance threshold.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> result = sp.did(df, y='y', treat='g', time='t', id='i', method='cs')
    >>> group_time_plot(result)
    >>> group_time_plot(result, plot_type='heatmap')
    """
    plt, mpl = _ensure_mpl()

    detail = result.detail
    if detail is None or 'group' not in detail.columns:
        raise ValueError(
            "Result must contain group-time detail from "
            "Callaway-Sant'Anna. Use did(method='cs')."
        )

    gt = detail.copy()

    if plot_type == 'heatmap':
        return _group_time_heatmap(gt, ax, figsize, title, plt, mpl)

    # ── Dot plot ──────────────────────────────────────────────────── #
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    gt = gt.sort_values(['group', 'time']).reset_index(drop=True)
    is_sig = gt['pvalue'] < alpha_level

    # Create label for each (g, t)
    labels = [f"g={int(r['group'])}, t={int(r['time'])}" for _, r in gt.iterrows()]
    y_pos = np.arange(len(gt))

    # Plot insignificant
    insig = ~is_sig
    if insig.any():
        ax.errorbar(
            gt.loc[insig, 'att'], y_pos[insig],
            xerr=[
                gt.loc[insig, 'att'] - gt.loc[insig, 'ci_lower'],
                gt.loc[insig, 'ci_upper'] - gt.loc[insig, 'att'],
            ],
            fmt='o', color=insig_color, capsize=3,
            markersize=5, linewidth=1, label='Not significant',
        )

    # Plot significant
    if is_sig.any():
        ax.errorbar(
            gt.loc[is_sig, 'att'], y_pos[is_sig],
            xerr=[
                gt.loc[is_sig, 'att'] - gt.loc[is_sig, 'ci_lower'],
                gt.loc[is_sig, 'ci_upper'] - gt.loc[is_sig, 'att'],
            ],
            fmt='o', color=sig_color, capsize=3,
            markersize=5, linewidth=1, label=f'Significant (p < {alpha_level})',
        )

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('ATT Estimate', fontsize=11)
    ax.set_title(title or 'Group-Time ATT Estimates', fontsize=13)
    _style_ax(ax)
    ax.legend(fontsize=9, frameon=False)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax


def _group_time_heatmap(gt, ax, figsize, title, plt, mpl):
    """Internal: heatmap of group-time ATTs."""
    pivot = gt.pivot_table(
        values='att', index='group', columns='time', aggfunc='mean',
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    im = ax.imshow(
        pivot.values, cmap='RdBu_r', aspect='auto',
        vmin=-vmax, vmax=vmax,
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), fontsize=8, rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int), fontsize=9)
    ax.set_xlabel('Time Period', fontsize=11)
    ax.set_ylabel('Treatment Cohort', fontsize=11)
    ax.set_title(title or 'Group-Time ATT Heatmap', fontsize=13)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(
                    j, i, f'{val:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if abs(val) > vmax * 0.6 else 'black',
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('ATT', fontsize=10)
    fig.tight_layout()
    return fig, ax


# ======================================================================
# 4. Classic DID Diagram with Counterfactual
# ======================================================================

def did_plot(
    data: pd.DataFrame,
    y: str,
    time: str,
    treat: str,
    treat_time: Optional[Union[int, float]] = None,
    show_counterfactual: bool = True,
    labels: Optional[Dict] = None,
    colors: Optional[Tuple[str, str, str]] = None,
    ax=None,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    annotate_effect: bool = True,
    **kwargs,
):
    """
    Classic DID diagram showing treatment effect with counterfactual.

    Plots group means over time and adds a dashed counterfactual line
    for the treatment group (extrapolated from pre-treatment trend
    parallel to the control group).

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable.
    time : str
        Time period variable.
    treat : str
        Binary treatment group indicator (0/1).
    treat_time : int or float, optional
        Treatment onset time. If None, inferred as the midpoint.
    show_counterfactual : bool, default True
        Draw the dashed counterfactual line.
    labels : dict, optional
        Custom labels: ``{'treat': ..., 'control': ..., 'counterfactual': ...}``.
    colors : tuple, optional
        (treatment, control, counterfactual) colors.
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    annotate_effect : bool, default True
        Annotate the treatment effect arrow on the plot.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> did_plot(df, y='wage', time='year', treat='treated',
    ...          treat_time=2010)
    """
    plt, _ = _ensure_mpl()
    colors = colors or ('#E74C3C', '#2C3E50', '#E74C3C')
    labels = labels or {}
    treat_label = labels.get('treat', 'Treatment')
    ctrl_label = labels.get('control', 'Control')
    cf_label = labels.get('counterfactual', 'Counterfactual')

    df = data.copy()

    # Binarize
    if set(df[treat].dropna().unique()) - {0, 1, True, False}:
        df['_tg'] = (df[treat] > 0).astype(int)
    else:
        df['_tg'] = df[treat].astype(int)

    means = df.groupby([time, '_tg'])[y].mean().reset_index()

    # Infer treat_time
    time_vals = sorted(means[time].unique())
    if treat_time is None:
        treat_time = time_vals[len(time_vals) // 2]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot control
    ctrl = means[means['_tg'] == 0].sort_values(time)
    ax.plot(ctrl[time], ctrl[y], color=colors[1], linewidth=2,
            marker='s', markersize=5, label=ctrl_label, zorder=5)

    # Plot treatment
    tr = means[means['_tg'] == 1].sort_values(time)
    ax.plot(tr[time], tr[y], color=colors[0], linewidth=2,
            marker='o', markersize=5, label=treat_label, zorder=5)

    # Counterfactual
    if show_counterfactual:
        # Pre-treatment: same as actual treatment line
        pre_treat = tr[tr[time] < treat_time]
        post_treat = tr[tr[time] >= treat_time]
        pre_ctrl = ctrl[ctrl[time] < treat_time]
        post_ctrl = ctrl[ctrl[time] >= treat_time]

        if len(pre_ctrl) > 0 and len(post_ctrl) > 0 and len(pre_treat) > 0:
            # Counterfactual = last pre-treatment value of treatment group
            # + change in control group from that point
            last_pre_t = pre_treat.iloc[-1][time]
            last_pre_y = pre_treat.iloc[-1][y]

            # Control group change from last_pre_t onward
            ctrl_at_pre = ctrl[ctrl[time] == last_pre_t]
            if len(ctrl_at_pre) > 0:
                ctrl_base = ctrl_at_pre.iloc[0][y]
                cf_times = post_ctrl[time].values
                cf_vals = last_pre_y + (post_ctrl[y].values - ctrl_base)

                # Connect from last pre-treatment point
                cf_times_full = np.concatenate([[last_pre_t], cf_times])
                cf_vals_full = np.concatenate([[last_pre_y], cf_vals])

                ax.plot(
                    cf_times_full, cf_vals_full,
                    color=colors[2], linewidth=1.5, linestyle='--',
                    marker='', alpha=0.7, label=cf_label, zorder=4,
                )

                # Annotate treatment effect
                if annotate_effect and len(post_treat) > 0 and len(cf_vals) > 0:
                    # Use last post-treatment point
                    last_post_t = post_treat.iloc[-1][time]
                    last_post_y = post_treat.iloc[-1][y]
                    # Find matching counterfactual
                    cf_idx = np.where(cf_times == last_post_t)[0]
                    if len(cf_idx) > 0:
                        cf_y = cf_vals[cf_idx[0]]
                        effect = last_post_y - cf_y
                        mid_y = (last_post_y + cf_y) / 2
                        ax.annotate(
                            '', xy=(last_post_t, last_post_y),
                            xytext=(last_post_t, cf_y),
                            arrowprops=dict(
                                arrowstyle='<->',
                                color='#8E44AD', lw=1.5,
                            ),
                        )
                        ax.text(
                            last_post_t + 0.1 * (time_vals[-1] - time_vals[0]),
                            mid_y, f'ATT ≈ {effect:.2f}',
                            fontsize=10, color='#8E44AD',
                            ha='left', va='center',
                        )

    # Treatment onset
    ax.axvline(
        x=treat_time, color='gray', linestyle=':',
        linewidth=1, alpha=0.5,
    )
    ax.text(
        treat_time, ax.get_ylim()[1],
        ' Treatment', fontsize=8, color='gray',
        ha='left', va='top',
    )

    ax.set_xlabel(time.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel(y.replace('_', ' ').title(), fontsize=11)
    ax.set_title(title or 'Difference-in-Differences', fontsize=13)
    _style_ax(ax)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    return fig, ax


# ======================================================================
# 5. Enhanced Event Study Plot
# ======================================================================

def event_study_plot(
    result,
    ax=None,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    color: str = '#2C3E50',
    sig_color: Optional[str] = '#E74C3C',
    ci_alpha: float = 0.15,
    shade_pre: bool = True,
    shade_post: bool = True,
    pre_color: str = '#EBF5FB',
    post_color: str = '#FDEDEC',
    show_zero: bool = True,
    marker: str = 'o',
    markersize: int = 6,
    alpha_level: float = 0.05,
    **kwargs,
):
    """
    Enhanced event study plot with pre/post shading and significance coloring.

    Improvement over the basic CausalResult.event_study_plot() —
    adds optional background shading for pre/post periods and
    colors significant coefficients differently.

    Parameters
    ----------
    result : CausalResult
        DID result with event study in ``model_info['event_study']``.
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    color : str
        Default color for estimates.
    sig_color : str or None
        Color for significant estimates. None disables coloring.
    ci_alpha : float
        Confidence band transparency.
    shade_pre : bool, default True
        Shade pre-treatment region.
    shade_post : bool, default True
        Shade post-treatment region.
    pre_color : str
        Pre-treatment shading color.
    post_color : str
        Post-treatment shading color.
    show_zero : bool, default True
        Show horizontal zero line.
    marker : str
    markersize : int
    alpha_level : float

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> result = sp.did(df, y='y', treat='g', time='t', id='i')
    >>> event_study_plot(result, shade_pre=True)
    """
    plt, _ = _ensure_mpl()

    mi = result.model_info or {}
    if 'event_study' not in mi:
        raise ValueError(
            "Result has no event study estimates. "
            "Use a staggered DID estimator or event_study()."
        )

    es = mi['event_study'].copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    e = es['relative_time'].values
    att = es['att'].values
    lo = es['ci_lower'].values
    hi = es['ci_upper'].values

    # Background shading
    if shade_pre and (e < 0).any():
        ax.axvspan(
            e.min() - 0.5, -0.5,
            color=pre_color, alpha=0.5, zorder=0,
        )
    if shade_post and (e >= 0).any():
        ax.axvspan(
            -0.5, e.max() + 0.5,
            color=post_color, alpha=0.5, zorder=0,
        )

    # CI band
    ax.fill_between(e, lo, hi, alpha=ci_alpha, color=color, zorder=2)

    # Line
    ax.plot(e, att, color=color, linewidth=1, alpha=0.6, zorder=3)

    # Points — color by significance
    if sig_color and 'pvalue' in es.columns:
        sig_mask = es['pvalue'].values < alpha_level
        if sig_mask.any():
            ax.scatter(
                e[sig_mask], att[sig_mask],
                color=sig_color, s=markersize ** 2, marker=marker,
                zorder=6, edgecolors='white', linewidth=0.5,
                label=f'Significant (p < {alpha_level})',
            )
        if (~sig_mask).any():
            ax.scatter(
                e[~sig_mask], att[~sig_mask],
                color=color, s=markersize ** 2, marker=marker,
                zorder=6, edgecolors='white', linewidth=0.5,
                label='Not significant',
            )
    else:
        ax.scatter(
            e, att, color=color, s=markersize ** 2,
            marker=marker, zorder=6, edgecolors='white', linewidth=0.5,
        )

    # Error bars
    ax.errorbar(
        e, att,
        yerr=[att - lo, hi - att],
        fmt='none', color=color, capsize=3,
        linewidth=0.8, zorder=4,
    )

    # Reference lines
    if show_zero:
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, zorder=1)
    ax.axvline(
        x=-0.5, color='#7F8C8D', linestyle=':',
        linewidth=1, alpha=0.7, zorder=1,
    )

    # Pre-trend test annotation
    pretrend = mi.get('pretrend_test') or mi.get('pretrend_pvalue')
    if isinstance(pretrend, dict):
        p = pretrend.get('pvalue')
    elif isinstance(pretrend, (int, float)):
        p = pretrend
    else:
        p = None

    if p is not None:
        ax.text(
            0.02, 0.98,
            f'Pre-trend p = {p:.3f}',
            transform=ax.transAxes,
            fontsize=9, va='top', ha='left',
            color='#27AE60' if p >= 0.05 else '#E74C3C',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='lightgray', alpha=0.8),
        )

    ax.set_xlabel('Periods Relative to Treatment', fontsize=11)
    ax.set_ylabel('Estimated Effect', fontsize=11)
    ax.set_title(title or f'Event Study: {result.method}', fontsize=13)
    _style_ax(ax)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    return fig, ax


# ======================================================================
# 6. Treatment Rollout / Timing Plot
# ======================================================================

def treatment_rollout_plot(
    data: pd.DataFrame,
    time: str,
    treat: str,
    id: str,
    ax=None,
    figsize: tuple = (12, 7),
    title: Optional[str] = None,
    treated_color: str = '#E74C3C',
    untreated_color: str = '#ECF0F1',
    never_color: str = '#BDC3C7',
    sort_by: str = 'treat_time',
    show_cohort_labels: bool = True,
    **kwargs,
):
    """
    Visualise staggered treatment adoption timing.

    Draws a tile/heatmap where each row is a unit and each column is
    a time period.  Treated periods are shaded, making the staggered
    rollout pattern immediately visible.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with unit, time, and treatment columns.
    time : str
        Time period variable.
    treat : str
        First-treatment-period column (0 = never treated), or binary
        treatment indicator.
    id : str
        Unit identifier.
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    treated_color : str
        Color for treated unit-periods.
    untreated_color : str
        Color for untreated unit-periods.
    never_color : str
        Color for never-treated units.
    sort_by : str, default 'treat_time'
        Sort units by: 'treat_time' (earliest first), 'id', or 'random'.
    show_cohort_labels : bool, default True
        Annotate cohort boundaries on the y-axis.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> treatment_rollout_plot(df, time='year', treat='first_treat', id='state')
    """
    plt, mpl = _ensure_mpl()
    from matplotlib.colors import ListedColormap

    df = data.copy()
    time_periods = sorted(df[time].unique())
    T = len(time_periods)
    time_map = {t: i for i, t in enumerate(time_periods)}

    # Determine each unit's first treatment period
    unit_info = df.groupby(id)[treat].first().reset_index()
    treat_vals = set(unit_info[treat].unique())

    # Check if treat is binary or first-treat-period
    if treat_vals <= {0, 1, True, False}:
        # Binary: infer first treatment period from data
        unit_treat_time = (
            df[df[treat].astype(bool)]
            .groupby(id)[time].min()
            .reset_index()
            .rename(columns={time: '_first_treat'})
        )
        unit_info = unit_info.merge(unit_treat_time, on=id, how='left')
        unit_info['_first_treat'] = unit_info['_first_treat'].fillna(0)
    else:
        unit_info['_first_treat'] = unit_info[treat]

    # Sort
    if sort_by == 'treat_time':
        # Never-treated last, then by treatment time
        unit_info['_sort'] = unit_info['_first_treat'].replace(0, 9999)
        unit_info = unit_info.sort_values('_sort').reset_index(drop=True)
    elif sort_by == 'id':
        unit_info = unit_info.sort_values(id).reset_index(drop=True)

    n_units = len(unit_info)
    unit_ids = unit_info[id].values
    first_treats = unit_info['_first_treat'].values

    # Build tile matrix: 0 = untreated, 1 = treated, -1 = never-treated
    tiles = np.zeros((n_units, T))
    for i, (uid, ft) in enumerate(zip(unit_ids, first_treats)):
        if ft == 0:
            tiles[i, :] = -1  # never treated
        else:
            for j, t_val in enumerate(time_periods):
                if t_val >= ft:
                    tiles[i, j] = 1

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Custom colormap: never-treated, untreated, treated
    from matplotlib.colors import LinearSegmentedColormap
    cmap = ListedColormap([never_color, untreated_color, treated_color])

    ax.imshow(
        tiles, cmap=cmap, aspect='auto',
        vmin=-1, vmax=1, interpolation='nearest',
    )

    # X-axis: time periods
    if T <= 20:
        ax.set_xticks(range(T))
        ax.set_xticklabels([str(t) for t in time_periods], fontsize=8, rotation=45)
    else:
        step = max(1, T // 10)
        ticks = list(range(0, T, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(time_periods[t]) for t in ticks], fontsize=8, rotation=45)

    # Y-axis: cohort labels
    if show_cohort_labels and n_units <= 60:
        # Show cohort boundaries
        cohorts = []
        prev_ft = None
        for i, ft in enumerate(first_treats):
            if ft != prev_ft:
                label = f'g={int(ft)}' if ft > 0 else 'Never'
                cohorts.append((i, label))
                prev_ft = ft

        cohort_ticks = [c[0] for c in cohorts]
        cohort_labels = [c[1] for c in cohorts]
        ax.set_yticks(cohort_ticks)
        ax.set_yticklabels(cohort_labels, fontsize=9)

        # Draw cohort boundary lines
        for i in range(1, len(cohort_ticks)):
            ax.axhline(
                y=cohort_ticks[i] - 0.5,
                color='white', linewidth=1.5,
            )
    elif n_units <= 30:
        ax.set_yticks(range(n_units))
        ax.set_yticklabels(unit_ids, fontsize=7)
    else:
        ax.set_yticks([])
        ax.set_ylabel(f'Units (n={n_units})', fontsize=11)

    ax.set_xlabel(time.replace('_', ' ').title(), fontsize=11)
    ax.set_title(title or 'Treatment Rollout', fontsize=13)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=treated_color, label='Treated'),
        Patch(facecolor=untreated_color, label='Not yet treated'),
        Patch(facecolor=never_color, label='Never treated'),
    ]
    ax.legend(
        handles=legend_elements, fontsize=9, frameon=False,
        loc='upper right', bbox_to_anchor=(1.15, 1),
    )

    fig.tight_layout()
    return fig, ax


# ======================================================================
# 7. Honest DID Sensitivity Plot
# ======================================================================

def sensitivity_plot(
    sensitivity: pd.DataFrame,
    original_ci: Optional[Tuple[float, float]] = None,
    original_estimate: Optional[float] = None,
    ax=None,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    color: str = '#2C3E50',
    breakdown_color: str = '#E74C3C',
    original_color: str = '#27AE60',
    **kwargs,
):
    """
    Plot Rambachan & Roth (2023) sensitivity analysis.

    Shows how the robust confidence interval changes as the maximum
    allowed parallel trends violation (M) increases.

    Parameters
    ----------
    sensitivity : pd.DataFrame
        Output from ``honest_did()``.
        Columns: M, ci_lower, ci_upper, rejects_zero.
    original_ci : tuple of (float, float), optional
        Original CI (at M=0) for comparison.
    original_estimate : float, optional
        Original point estimate.
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    color : str
        CI band color.
    breakdown_color : str
        Color for the breakdown point marker.
    original_color : str
        Color for original estimate marker.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> sens = sp.honest_did(result, e=0)
    >>> sensitivity_plot(sens, original_estimate=result.estimate,
    ...                  original_ci=result.ci)
    """
    plt, _ = _ensure_mpl()

    if sensitivity is None or len(sensitivity) == 0:
        raise ValueError("Empty sensitivity DataFrame.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    m_vals = sensitivity['M'].values
    ci_lo = sensitivity['ci_lower'].values
    ci_hi = sensitivity['ci_upper'].values
    rejects = sensitivity['rejects_zero'].values

    # CI band
    ax.fill_between(
        m_vals, ci_lo, ci_hi,
        alpha=0.15, color=color, zorder=2,
    )

    # CI boundaries
    ax.plot(m_vals, ci_lo, color=color, linewidth=1.5, zorder=3)
    ax.plot(m_vals, ci_hi, color=color, linewidth=1.5, zorder=3)

    # Zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, zorder=1)

    # Breakdown point: where rejects_zero switches from True to False
    breakdown_idx = None
    for i in range(len(rejects)):
        if rejects[i] and (i + 1 >= len(rejects) or not rejects[i + 1]):
            breakdown_idx = i
            break

    if breakdown_idx is not None:
        m_star = m_vals[breakdown_idx]
        ax.axvline(
            x=m_star, color=breakdown_color, linestyle=':',
            linewidth=1.5, alpha=0.7, zorder=4,
        )
        ax.scatter(
            [m_star], [(ci_lo[breakdown_idx] + ci_hi[breakdown_idx]) / 2],
            color=breakdown_color, s=80, marker='D', zorder=6,
            label=f'Breakdown M* = {m_star:.3f}',
        )

    # Original estimate
    if original_estimate is not None:
        ax.axhline(
            y=original_estimate, color=original_color,
            linestyle='-.', linewidth=1, alpha=0.6,
            label=f'Point estimate = {original_estimate:.3f}',
        )

    # Original CI at M=0
    if original_ci is not None:
        ax.plot(
            [0, 0], [original_ci[0], original_ci[1]],
            color=original_color, linewidth=3, alpha=0.5, zorder=5,
        )

    ax.set_xlabel('M (Maximum Violation Magnitude)', fontsize=11)
    ax.set_ylabel('Robust Confidence Interval', fontsize=11)
    ax.set_title(
        title or 'Sensitivity to Parallel Trends Violations\n(Rambachan & Roth, 2023)',
        fontsize=13,
    )
    _style_ax(ax)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    return fig, ax


# ======================================================================
# 8. Cohort-Specific Event Study Plot
# ======================================================================

def cohort_event_study_plot(
    result,
    ax=None,
    figsize: tuple = (12, 7),
    title: Optional[str] = None,
    palette: Optional[List[str]] = None,
    show_aggregate: bool = True,
    aggregate_color: str = '#2C3E50',
    ci: bool = True,
    ci_alpha: float = 0.08,
    **kwargs,
):
    """
    Per-cohort event study plot (overlay).

    Plots a separate event study line for each treatment cohort,
    showing heterogeneity in treatment effects across cohorts.
    Optionally overlays the aggregate event study.

    Parameters
    ----------
    result : CausalResult
        Result from ``callaway_santanna()`` or ``did(method='cs')``.
        Must have ``detail`` with 'group', 'relative_time', 'att' columns,
        and ``model_info['event_study']`` for aggregate.
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    palette : list of str, optional
        Colors for each cohort. Auto-generated if None.
    show_aggregate : bool, default True
        Overlay the aggregate event study line.
    aggregate_color : str
        Color for aggregate line.
    ci : bool, default True
        Show confidence intervals for each cohort.
    ci_alpha : float
        CI band transparency.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> result = sp.did(df, y='y', treat='g', time='t', id='i', method='cs')
    >>> cohort_event_study_plot(result)
    """
    plt, _ = _ensure_mpl()

    detail = result.detail
    if detail is None or 'group' not in detail.columns:
        raise ValueError(
            "Result must have group-time detail. Use did(method='cs')."
        )

    gt = detail.copy()
    if 'relative_time' not in gt.columns:
        raise ValueError("Detail must have 'relative_time' column.")

    cohorts = sorted(gt['group'].unique())
    cohorts = [c for c in cohorts if c > 0]  # exclude never-treated

    if not cohorts:
        raise ValueError("No treated cohorts found.")

    # Default palette
    if palette is None:
        default_colors = [
            '#E74C3C', '#3498DB', '#27AE60', '#F39C12',
            '#9B59B6', '#1ABC9C', '#E67E22', '#2980B9',
            '#C0392B', '#16A085',
        ]
        palette = default_colors[:len(cohorts)]
        while len(palette) < len(cohorts):
            palette.append(f'#{np.random.randint(0, 0xFFFFFF):06X}')

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot each cohort
    for i, cohort in enumerate(cohorts):
        coh = gt[gt['group'] == cohort].sort_values('relative_time')
        color = palette[i % len(palette)]
        e = coh['relative_time'].values
        att = coh['att'].values

        ax.plot(
            e, att, color=color, linewidth=1.2, alpha=0.7,
            marker='o', markersize=4,
            label=f'Cohort {int(cohort)}', zorder=4,
        )

        if ci and 'ci_lower' in coh.columns:
            ax.fill_between(
                e, coh['ci_lower'].values, coh['ci_upper'].values,
                alpha=ci_alpha, color=color, zorder=2,
            )

    # Aggregate event study
    mi = result.model_info or {}
    if show_aggregate and 'event_study' in mi:
        agg = mi['event_study'].copy()
        ax.plot(
            agg['relative_time'], agg['att'],
            color=aggregate_color, linewidth=2.5, alpha=0.9,
            marker='s', markersize=6,
            label='Aggregate', zorder=6,
        )
        if ci and 'ci_lower' in agg.columns:
            ax.fill_between(
                agg['relative_time'],
                agg['ci_lower'], agg['ci_upper'],
                alpha=0.12, color=aggregate_color, zorder=3,
            )

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, zorder=1)
    ax.axvline(x=-0.5, color='#7F8C8D', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel('Periods Relative to Treatment', fontsize=11)
    ax.set_ylabel('Estimated Effect', fontsize=11)
    ax.set_title(title or 'Event Study by Cohort', fontsize=13)
    _style_ax(ax)
    ax.legend(fontsize=9, frameon=False, ncol=min(3, len(cohorts) + 1))
    fig.tight_layout()
    return fig, ax


# ======================================================================
# 9. ggdid — aggte() result visualiser with uniform bands
# ======================================================================

def ggdid(
    result,
    ax=None,
    figsize=(10, 6),
    title: Optional[str] = None,
    point_color: str = '#2E86AB',
    band_color: str = '#F18F01',
    show_pointwise: bool = True,
    show_uniform: bool = True,
):
    """Plot an ``aggte()`` result, mirroring R :func:`did::ggdid`.

    Automatically dispatches on ``result.model_info['aggregation']``:

    - ``simple``   : a single point with pointwise CI
    - ``dynamic``  : event-study line with pointwise CI **and** uniform band
    - ``group``    : horizontal bars of θ̂(g) per cohort
    - ``calendar`` : time-series of θ̂(t) per calendar period

    Uniform bands (sup-t simultaneous confidence bands) are drawn from the
    ``cband_lower`` / ``cband_upper`` columns created by :func:`aggte`.

    Parameters
    ----------
    result : CausalResult
        Output of :func:`aggte`.
    ax : matplotlib Axes, optional
    figsize : tuple, default (10, 6)
    title : str, optional
    point_color, band_color : str
        Colours for the pointwise estimate and the uniform band.
    show_pointwise : bool, default True
        Draw pointwise CI lines.
    show_uniform : bool, default True
        Draw uniform band (shaded region).

    Returns
    -------
    (fig, ax)
    """
    plt, _ = _ensure_mpl()

    info = result.model_info or {}
    agg = info.get('aggregation', 'dynamic')
    df = result.detail
    if df is None or len(df) == 0:
        raise ValueError("result has empty detail table.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    has_cband = 'cband_lower' in df.columns and 'cband_upper' in df.columns

    if agg == 'simple':
        est = df['att'].iloc[0]
        lo, hi = df['ci_lower'].iloc[0], df['ci_upper'].iloc[0]
        ax.errorbar(
            [0], [est], yerr=[[est - lo], [hi - est]],
            fmt='o', color=point_color, capsize=6, markersize=8,
            linewidth=2, label=f'Overall ATT = {est:.3f}',
        )
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks([])
        ax.set_ylabel('ATT')
        ax.set_title(title or "Callaway-Sant'Anna — simple aggregation")
        ax.legend(frameon=False)

    elif agg == 'dynamic':
        x = df['relative_time'].values
        est = df['att'].values
        # Uniform band first (behind lines).
        if show_uniform and has_cband:
            ax.fill_between(
                x, df['cband_lower'], df['cband_upper'],
                color=band_color, alpha=0.25,
                label=f"Uniform {int(100*(1-result.alpha))}% band",
            )
        if show_pointwise:
            ax.fill_between(
                x, df['ci_lower'], df['ci_upper'],
                color=point_color, alpha=0.18,
                label=f"Pointwise {int(100*(1-result.alpha))}% CI",
            )
        post = x >= 0
        pre = x < 0
        ax.plot(x[pre], est[pre], 'o-', color='#7F8C8D',
                markersize=6, linewidth=1.5, label='Pre-treatment')
        ax.plot(x[post], est[post], 'o-', color=point_color,
                markersize=7, linewidth=2, label='Post-treatment')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(-0.5, color='#7F8C8D', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Event time e = t − g')
        ax.set_ylabel('ATT(e)')
        ax.set_title(title or "Callaway-Sant'Anna — dynamic (event study)")
        ax.legend(frameon=False, fontsize=9)

    elif agg == 'group':
        groups = df['group'].values
        est = df['att'].values
        yvals = np.arange(len(groups))
        ax.errorbar(
            est, yvals,
            xerr=[est - df['ci_lower'], df['ci_upper'] - est],
            fmt='o', color=point_color, capsize=5, markersize=7,
            label=f"Pointwise {int(100*(1-result.alpha))}% CI",
        )
        if show_uniform and has_cband:
            ax.errorbar(
                est, yvals,
                xerr=[est - df['cband_lower'], df['cband_upper'] - est],
                fmt='none', color=band_color, capsize=10, linewidth=2,
                alpha=0.6,
                label=f"Uniform {int(100*(1-result.alpha))}% band",
            )
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_yticks(yvals)
        ax.set_yticklabels([f'g = {g}' for g in groups])
        ax.set_xlabel('ATT(g)')
        ax.set_title(title or "Callaway-Sant'Anna — group aggregation")
        ax.legend(frameon=False, fontsize=9)

    elif agg == 'calendar':
        x = df['time'].values
        est = df['att'].values
        if show_uniform and has_cband:
            ax.fill_between(
                x, df['cband_lower'], df['cband_upper'],
                color=band_color, alpha=0.25,
                label=f"Uniform {int(100*(1-result.alpha))}% band",
            )
        if show_pointwise:
            ax.fill_between(
                x, df['ci_lower'], df['ci_upper'],
                color=point_color, alpha=0.18,
                label=f"Pointwise {int(100*(1-result.alpha))}% CI",
            )
        ax.plot(x, est, 'o-', color=point_color, linewidth=2, markersize=7)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Calendar time t')
        ax.set_ylabel('ATT(t)')
        ax.set_title(title or "Callaway-Sant'Anna — calendar aggregation")
        ax.legend(frameon=False, fontsize=9)

    else:
        raise ValueError(
            f"Unsupported aggregation type in result.model_info: {agg!r}"
        )

    _style_ax(ax)
    fig.tight_layout()
    return fig, ax


# ======================================================================
# N. Forest Plot for did_summary()
# ======================================================================

def did_summary_plot(
    result,
    ax=None,
    figsize: tuple = (9, 5),
    color: str = "#2C3E50",
    highlight_color: str = "#C0392B",
    reference: Optional[float] = None,
    title: Optional[str] = None,
    sort_by: Optional[str] = None,
):
    """
    Forest plot of DID method-robustness summary.

    Plots each method's point estimate with its confidence interval as a
    horizontal errorbar. Designed to consume the ``CausalResult`` returned
    by :func:`statspai.did.did_summary`.

    Parameters
    ----------
    result : CausalResult
        Output of :func:`did_summary`. Must have a ``detail`` DataFrame
        with columns ``estimate``, ``ci_low``, ``ci_high``, and either
        ``method`` or ``estimator``.
    ax : matplotlib Axes, optional
        Existing axes to draw on. If ``None`` a new figure is created.
    figsize : tuple, default ``(9, 5)``
        Figure size when creating a new figure.
    color : str, default ``"#2C3E50"``
        Color for point estimates and CIs.
    highlight_color : str, default ``"#C0392B"``
        Color for the cross-method mean line.
    reference : float, optional
        Horizontal reference value (e.g. 0 for 'no effect'). Defaults
        to ``0``.
    title : str, optional
        Plot title. Defaults to ``"DID Method-Robustness Summary"``.
    sort_by : {'estimate', None}, optional
        If ``'estimate'``, sort methods by point estimate ascending.
        Otherwise keep the order in ``result.detail``.

    Returns
    -------
    (fig, ax) : matplotlib figure and axes.

    Examples
    --------
    >>> out = sp.did_summary(df, y='y', time='time',
    ...                      first_treat='first_treat', group='unit')
    >>> fig, ax = sp.did_summary_plot(out)
    """
    plt, _mpl = _ensure_mpl()

    detail = getattr(result, "detail", None)
    if not isinstance(detail, pd.DataFrame) or "estimate" not in detail.columns:
        raise ValueError(
            "result must come from sp.did_summary() — its .detail "
            "DataFrame needs an 'estimate' column."
        )

    df = detail.copy()
    df = df.loc[df["estimate"].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError("No successfully-fit methods to plot.")

    if sort_by == "estimate":
        df = df.sort_values("estimate").reset_index(drop=True)

    labels = df["estimator"] if "estimator" in df.columns else df["method"]
    ests = df["estimate"].values
    lo = df["ci_low"].values
    hi = df["ci_high"].values

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y_pos = np.arange(len(df))
    # CI error bars
    lo_err = ests - lo
    hi_err = hi - ests
    ax.errorbar(
        ests, y_pos,
        xerr=[lo_err, hi_err],
        fmt="o", color=color, ecolor=color, markersize=7,
        capsize=4, linewidth=1.8, elinewidth=1.2, zorder=3,
    )

    # Reference line (usually at 0)
    ref = 0.0 if reference is None else reference
    ax.axvline(ref, color="grey", linestyle=":", linewidth=1, zorder=1)

    # Cross-method mean (if >1 method)
    if len(df) > 1 and getattr(result, "estimate", None) is not None:
        try:
            mean_est = float(result.estimate)
            ax.axvline(
                mean_est, color=highlight_color, linestyle="--",
                linewidth=1.2, zorder=2,
                label=f"Mean across methods = {mean_est:.3f}",
            )
            ax.legend(loc="best", frameon=False, fontsize=9)
        except (TypeError, ValueError):
            pass

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # first method at top
    ax.set_xlabel("Overall ATT estimate (95% CI)")
    ax.set_title(title or "DID Method-Robustness Summary")

    _style_ax(ax)
    fig.tight_layout()
    return fig, ax
