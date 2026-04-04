"""
Specification Curve Analysis (Multiverse Analysis)

Run all "reasonable" specifications of a regression model and visualize
how the key estimate varies across analytical choices.

Reference
---------
Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020).
Specification curve analysis. *Nature Human Behaviour*, 4(11), 1208–1214.

Usage
-----
>>> import statspai as sp
>>> result = sp.spec_curve(
...     data=df,
...     y='wage',
...     x='education',
...     controls=[[], ['experience'], ['experience', 'female']],
...     se_types=['nonrobust', 'hc1'],
... )
>>> result.plot()
>>> result.summary()
"""

from __future__ import annotations

from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union,
)
from itertools import product
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SpecCurveResult:
    """Holds all specification curve outputs."""

    results_df: pd.DataFrame
    """One row per specification with columns:
    spec_id, estimate, se, ci_lower, ci_upper, pvalue, significant,
    plus one column per choice dimension."""

    x: str
    y: str
    n_specs: int
    median_estimate: float
    share_significant: float
    share_positive: float
    choice_dims: List[str]

    # ---- pretty-print --------------------------------------------------

    def summary(self, alpha: float = 0.05) -> str:
        """Return a formatted text summary."""
        lines = [
            "=" * 72,
            "  Specification Curve Analysis",
            "  (Simonsohn, Simmons & Nelson, 2020)",
            "=" * 72,
            "",
            f"  Key variable:        {self.x}",
            f"  Outcome:             {self.y}",
            f"  Total specifications: {self.n_specs}",
            "",
            f"  Median estimate:     {self.median_estimate: .4f}",
            f"  Mean estimate:       {self.results_df['estimate'].mean(): .4f}",
            f"  Min estimate:        {self.results_df['estimate'].min(): .4f}",
            f"  Max estimate:        {self.results_df['estimate'].max(): .4f}",
            f"  Std. dev:            {self.results_df['estimate'].std(): .4f}",
            "",
            f"  Share significant (p<{alpha}): "
            f"{self.share_significant:.1%}  "
            f"({int(self.share_significant * self.n_specs)}/{self.n_specs})",
            f"  Share positive:      {self.share_positive:.1%}",
            "",
        ]

        # Per-dimension breakdown
        lines.append("-" * 72)
        lines.append("  Estimates by analytical choice")
        lines.append("-" * 72)
        for dim in self.choice_dims:
            lines.append(f"\n  Dimension: {dim}")
            grp = self.results_df.groupby(dim)['estimate']
            tbl = grp.agg(['count', 'mean', 'median', 'std']).reset_index()
            tbl.columns = [dim, 'n', 'mean', 'median', 'std']
            for _, row in tbl.iterrows():
                lines.append(
                    f"    {row[dim]:<30s}  "
                    f"n={int(row['n']):>4d}  "
                    f"mean={row['mean']:>8.4f}  "
                    f"med={row['median']:>8.4f}"
                )

        lines.append("")
        lines.append("=" * 72)
        lines.append("  * p<0.1, ** p<0.05, *** p<0.01")
        return "\n".join(lines)

    # ---- export ---------------------------------------------------------

    def to_latex(self, caption: str = "Specification Curve Summary") -> str:
        """Export summary to a LaTeX table."""
        df = self.results_df
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            r"\label{tab:spec_curve}",
            r"\begin{tabular}{lc}",
            r"\hline\hline",
            f"Key variable & {self.x} \\\\",
            f"Outcome & {self.y} \\\\",
            r"\hline",
            f"Total specifications & {self.n_specs} \\\\",
            f"Median estimate & {self.median_estimate:.4f} \\\\",
            f"Mean estimate & {df['estimate'].mean():.4f} \\\\",
            f"[Min, Max] & [{df['estimate'].min():.4f}, "
            f"{df['estimate'].max():.4f}] \\\\",
            f"Share significant (p<0.05) & {self.share_significant:.1%} \\\\",
            f"Share positive & {self.share_positive:.1%} \\\\",
            r"\hline\hline",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item Simonsohn, Simmons \& Nelson (2020). "
            r"Specification curve analysis. \textit{Nature Human Behaviour}.",
            r"\end{tablenotes}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full results DataFrame."""
        return self.results_df.copy()

    # ---- citation -------------------------------------------------------

    def cite(self) -> str:
        return (
            "@article{simonsohn2020specification,\n"
            "  title={Specification curve analysis},\n"
            "  author={Simonsohn, Uri and Simmons, Joseph P "
            "and Nelson, Leif D},\n"
            "  journal={Nature Human Behaviour},\n"
            "  volume={4},\n"
            "  number={11},\n"
            "  pages={1208--1214},\n"
            "  year={2020},\n"
            "  publisher={Nature Publishing Group}\n"
            "}"
        )

    # ---- plotting -------------------------------------------------------

    def plot(
        self,
        alpha: float = 0.05,
        color_sig: str = "#2C3E50",
        color_nonsig: str = "#BDC3C7",
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        sort_by: str = "estimate",
    ):
        """
        Draw the canonical two-panel specification curve plot.

        Top panel:  sorted point estimates with 95 % CIs,
                    coloured by significance.
        Bottom panel: indicator matrix showing which analytical
                      choices produced each specification.

        Parameters
        ----------
        alpha : float
            Significance threshold for colouring.
        color_sig, color_nonsig : str
            Colours for significant / non-significant estimates.
        figsize : tuple, optional
            Figure size (width, height). Auto-sized if *None*.
        title : str, optional
            Title for the top panel.
        sort_by : str
            Column to sort specifications by. Default ``'estimate'``.

        Returns
        -------
        fig, axes : matplotlib Figure and array of Axes
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib.lines import Line2D
        except ImportError:
            raise ImportError(
                "matplotlib required for plotting.  pip install matplotlib"
            )

        df = self.results_df.sort_values(sort_by).reset_index(drop=True)
        n = len(df)
        n_dims = len(self.choice_dims)

        # Auto-size
        if figsize is None:
            w = max(8, n * 0.06)
            h = 4 + n_dims * 0.55
            figsize = (min(w, 18), h)

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            2, 1,
            height_ratios=[3, max(n_dims, 1)],
            hspace=0.05,
        )

        # ---- Top panel: estimates + CIs --------------------------------
        ax_top = fig.add_subplot(gs[0])
        sig_mask = df['pvalue'] < alpha

        xs = np.arange(n)
        for i in range(n):
            c = color_sig if sig_mask.iloc[i] else color_nonsig
            ax_top.plot(
                [i, i],
                [df['ci_lower'].iloc[i], df['ci_upper'].iloc[i]],
                color=c, linewidth=0.6, alpha=0.7,
            )
            ax_top.plot(i, df['estimate'].iloc[i], 'o',
                        color=c, markersize=2.0)

        ax_top.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax_top.axhline(
            self.median_estimate,
            color='#E74C3C', linewidth=0.8, linestyle=':',
            label=f'Median = {self.median_estimate:.4f}',
        )
        ax_top.set_ylabel(f"Estimate of '{self.x}'")
        ax_top.set_xlim(-0.5, n - 0.5)
        ax_top.set_xticks([])
        ax_top.legend(fontsize=8, loc='upper left')
        if title:
            ax_top.set_title(title, fontsize=12)
        else:
            ax_top.set_title("Specification Curve Analysis", fontsize=12)

        for spine in ['top', 'right', 'bottom']:
            ax_top.spines[spine].set_visible(False)

        # ---- Bottom panel: indicator matrix ----------------------------
        ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

        # Build category mapping per dimension
        y_ticks = []
        y_labels = []
        y_offset = 0

        for dim_idx, dim in enumerate(self.choice_dims):
            categories = sorted(df[dim].unique(), key=str)
            for cat_idx, cat in enumerate(categories):
                row_y = y_offset + cat_idx
                mask = df[dim].astype(str) == str(cat)
                idxs = xs[mask]
                colors = [
                    color_sig if sig_mask.iloc[i] else color_nonsig
                    for i in idxs
                ]
                ax_bot.scatter(
                    idxs,
                    [row_y] * len(idxs),
                    c=colors, s=4, marker='|', linewidths=0.6,
                )
                y_ticks.append(row_y)
                y_labels.append(str(cat))
            y_offset += len(categories) + 0.5  # gap between dims

        ax_bot.set_yticks(y_ticks)
        ax_bot.set_yticklabels(y_labels, fontsize=7)
        ax_bot.set_ylim(-0.5, y_offset)
        ax_bot.invert_yaxis()
        ax_bot.set_xlabel(f"Specifications (sorted by {sort_by})")

        for spine in ['top', 'right']:
            ax_bot.spines[spine].set_visible(False)

        # Add dimension labels on the right
        y_offset_label = 0
        for dim in self.choice_dims:
            cats = sorted(df[dim].unique(), key=str)
            mid = y_offset_label + (len(cats) - 1) / 2
            ax_bot.text(
                n + n * 0.01, mid, dim,
                fontsize=8, fontweight='bold',
                va='center', ha='left', clip_on=False,
            )
            y_offset_label += len(cats) + 0.5

        try:
            fig.tight_layout()
        except Exception:
            pass  # sharex axes can conflict with tight_layout
        return fig, (ax_top, ax_bot)


# ---------------------------------------------------------------------------
# Internal: run a single OLS specification
# ---------------------------------------------------------------------------

def _run_one_spec(
    data: pd.DataFrame,
    y: str,
    x: str,
    controls: List[str],
    se_type: str,
    cluster_var: Optional[str],
    subset_mask: Optional[pd.Series],
    subset_label: str,
    model_type: str,
    transform_y: Optional[Callable],
    transform_label: str,
) -> Dict[str, Any]:
    """Run a single specification and return results dict."""
    df = data.copy()

    # Apply subset
    if subset_mask is not None:
        df = df.loc[subset_mask].copy()

    if len(df) < 10:
        return None  # too few observations

    # Apply y-transform
    y_col = y
    if transform_y is not None:
        y_col = f"__{y}_transformed"
        df[y_col] = transform_y(df[y])
        # Drop NaN/inf from transform
        valid = np.isfinite(df[y_col])
        df = df.loc[valid]

    if len(df) < 10:
        return None

    all_vars = [y_col, x] + controls
    if cluster_var:
        all_vars.append(cluster_var)
    df_clean = df[all_vars].dropna()

    if len(df_clean) < len(controls) + 3:
        return None

    Y = df_clean[y_col].values.astype(float)
    X_vars = [x] + controls
    X_mat = np.column_stack([
        np.ones(len(df_clean)),
        df_clean[X_vars].values.astype(float),
    ])

    n, k = X_mat.shape

    # Estimate OLS
    try:
        XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    except np.linalg.LinAlgError:
        return None

    params = XtX_inv @ X_mat.T @ Y
    resid = Y - X_mat @ params

    # Standard errors
    if se_type == 'nonrobust':
        sigma2 = np.sum(resid ** 2) / (n - k)
        vcov = sigma2 * XtX_inv
    elif se_type in ('hc1', 'robust'):
        # HC1 robust SE
        u2 = resid ** 2
        meat = X_mat.T @ np.diag(u2) @ X_mat * n / (n - k)
        vcov = XtX_inv @ meat @ XtX_inv
    elif se_type == 'cluster' and cluster_var:
        clusters = df_clean[cluster_var].values
        unique_c = np.unique(clusters)
        G = len(unique_c)
        meat = np.zeros((k, k))
        for c in unique_c:
            idx = clusters == c
            Xc = X_mat[idx]
            uc = resid[idx]
            score = Xc.T @ uc
            meat += np.outer(score, score)
        # Small-sample correction
        correction = (G / (G - 1)) * ((n - 1) / (n - k))
        vcov = correction * XtX_inv @ meat @ XtX_inv
    else:
        # Fallback to HC1
        u2 = resid ** 2
        meat = X_mat.T @ np.diag(u2) @ X_mat * n / (n - k)
        vcov = XtX_inv @ meat @ XtX_inv

    se = np.sqrt(np.diag(vcov))

    # Key variable is index 1 (index 0 is intercept)
    beta_x = params[1]
    se_x = se[1]
    df_resid = n - k
    t_stat = beta_x / se_x if se_x > 0 else np.inf
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_resid))
    t_crit = stats.t.ppf(0.975, df_resid)

    # R-squared
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'estimate': beta_x,
        'se': se_x,
        'ci_lower': beta_x - t_crit * se_x,
        'ci_upper': beta_x + t_crit * se_x,
        'pvalue': p_val,
        'tstat': t_stat,
        'nobs': n,
        'r_squared': r2,
        'controls': ", ".join(controls) if controls else "(none)",
        'se_type': se_type,
        'subset': subset_label,
        'model': model_type,
        'y_transform': transform_label,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spec_curve(
    data: pd.DataFrame,
    y: str,
    x: str,
    controls: Optional[List[List[str]]] = None,
    se_types: Optional[List[str]] = None,
    subsets: Optional[Dict[str, Optional[pd.Series]]] = None,
    cluster_var: Optional[str] = None,
    y_transforms: Optional[Dict[str, Callable]] = None,
    alpha: float = 0.05,
) -> SpecCurveResult:
    """
    Run a specification curve analysis.

    Enumerate all combinations of analytical choices and estimate the
    effect of *x* on *y* under each.  Returns a :class:`SpecCurveResult`
    with a ``.plot()`` method producing the canonical two-panel figure
    from Simonsohn, Simmons & Nelson (2020).

    Parameters
    ----------
    data : DataFrame
        Analysis dataset.
    y : str
        Outcome variable column name.
    x : str
        Key explanatory variable of interest.
    controls : list of list of str, optional
        Each inner list is one control-set specification.
        Example: ``[[], ['age'], ['age', 'female']]``.
        If *None*, defaults to ``[[]]`` (no controls only).
    se_types : list of str, optional
        Standard-error types to iterate over.
        Options: ``'nonrobust'``, ``'hc1'`` (or ``'robust'``),
        ``'cluster'`` (requires *cluster_var*).
        Default: ``['nonrobust', 'hc1']``.
    subsets : dict[str, Series | None], optional
        Named sub-samples.  Keys are labels; values are boolean
        ``pd.Series`` masks aligned with *data*.  ``None`` means the
        full sample.  Default: ``{'Full Sample': None}``.
    cluster_var : str, optional
        Column for clustered SEs.  Automatically adds ``'cluster'``
        to *se_types* if provided and not already present.
    y_transforms : dict[str, callable], optional
        Named outcome transformations.
        Example: ``{'Level': None, 'Log': np.log}``.
        Default: ``{'Level': None}``.
        ``None`` value means no transform.
    alpha : float, default 0.05
        Significance threshold.

    Returns
    -------
    SpecCurveResult
        Container with ``.plot()``, ``.summary()``, ``.to_latex()``,
        ``.results_df`` (DataFrame of all specifications).

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.spec_curve(
    ...     data=df,
    ...     y='wage',
    ...     x='education',
    ...     controls=[
    ...         [],
    ...         ['experience'],
    ...         ['experience', 'female'],
    ...     ],
    ...     se_types=['nonrobust', 'hc1'],
    ...     subsets={
    ...         'Full': None,
    ...         'Male': df['female'] == 0,
    ...     },
    ... )
    >>> result.plot()
    >>> print(result.summary())
    """
    # ---- defaults -------------------------------------------------------
    if controls is None:
        controls = [[]]
    if se_types is None:
        se_types = ['nonrobust', 'hc1']
    if subsets is None:
        subsets = {'Full Sample': None}
    if y_transforms is None:
        y_transforms = {'Level': None}

    # Auto-add cluster SE type
    if cluster_var and 'cluster' not in se_types:
        se_types = list(se_types) + ['cluster']

    # Remove 'cluster' if no cluster_var
    if not cluster_var:
        se_types = [s for s in se_types if s != 'cluster']

    # ---- enumerate all combinations -------------------------------------
    all_results = []
    spec_id = 0

    for ctrl_set, se_type, (sub_label, sub_mask), (tf_label, tf_func) in product(
        controls, se_types, subsets.items(), y_transforms.items(),
    ):
        res = _run_one_spec(
            data=data,
            y=y,
            x=x,
            controls=ctrl_set,
            se_type=se_type,
            cluster_var=cluster_var,
            subset_mask=sub_mask,
            subset_label=sub_label,
            model_type='OLS',
            transform_y=tf_func,
            transform_label=tf_label,
        )
        if res is not None:
            res['spec_id'] = spec_id
            all_results.append(res)
            spec_id += 1

    if not all_results:
        raise ValueError(
            "No valid specification could be estimated. "
            "Check that your data, variable names, and subsets are correct."
        )

    results_df = pd.DataFrame(all_results)

    # Significance flag
    results_df['significant'] = results_df['pvalue'] < alpha

    # ---- determine which choice dimensions vary -------------------------
    choice_dims = []
    for col in ['controls', 'se_type', 'subset', 'y_transform']:
        if results_df[col].nunique() > 1:
            choice_dims.append(col)
    # Always show at least controls
    if not choice_dims:
        choice_dims = ['controls']

    n_specs = len(results_df)

    return SpecCurveResult(
        results_df=results_df,
        x=x,
        y=y,
        n_specs=n_specs,
        median_estimate=float(results_df['estimate'].median()),
        share_significant=float(results_df['significant'].mean()),
        share_positive=float((results_df['estimate'] > 0).mean()),
        choice_dims=choice_dims,
    )
