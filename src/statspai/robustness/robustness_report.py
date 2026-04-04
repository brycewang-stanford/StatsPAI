"""
Automated Robustness Report

Given a baseline regression, automatically run a battery of robustness
checks and present them in a unified table / report.

Checks include:
- Different standard error types (OLS / HC1 / clustered)
- Adding or removing control variables
- Winsorizing the outcome
- Dropping influential observations
- Subsample analysis

Usage
-----
>>> import statspai as sp
>>> baseline = sp.regress("wage ~ education + experience", data=df)
>>> report = sp.robustness_report(
...     data=df,
...     formula="wage ~ education + experience",
...     x='education',
...     cluster_var='region',
...     extra_controls=['female', 'age'],
...     drop_controls=['experience'],
...     winsor_levels=[0.01, 0.05],
... )
>>> report.plot()
>>> report.summary()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RobustnessResult:
    """Container for robustness report results."""

    results_df: pd.DataFrame
    """One row per robustness check."""

    x: str
    baseline_estimate: float
    baseline_se: float
    n_checks: int

    def summary(self) -> str:
        """Formatted text summary."""
        df = self.results_df
        lines = [
            "=" * 78,
            "  Robustness Report",
            "=" * 78,
            "",
            f"  Key variable:        {self.x}",
            f"  Baseline estimate:   {self.baseline_estimate: .4f}",
            f"  Baseline SE:         ({self.baseline_se:.4f})",
            f"  Total checks:        {self.n_checks}",
            "",
            "-" * 78,
            f"  {'Check':<35s} {'Estimate':>10s} {'SE':>10s} "
            f"{'p-value':>10s} {'N':>8s}",
            "-" * 78,
        ]
        for _, row in df.iterrows():
            stars = _stars(row['pvalue'])
            lines.append(
                f"  {row['check']:<35s} "
                f"{row['estimate']:>9.4f}{stars:<3s} "
                f"({row['se']:>8.4f}) "
                f"{row['pvalue']:>9.4f}  "
                f"{int(row['nobs']):>7d}"
            )
        lines.append("-" * 78)

        # Stability assessment
        ests = df['estimate']
        pct_change = ((ests - self.baseline_estimate) /
                      abs(self.baseline_estimate) * 100)
        max_change = pct_change.abs().max()
        all_same_sign = (ests > 0).all() or (ests < 0).all()
        all_sig = (df['pvalue'] < 0.05).all()

        lines.append("")
        lines.append("  Stability Assessment")
        lines.append(f"  - Max % change from baseline: {max_change:.1f}%")
        lines.append(
            f"  - Sign consistency: "
            f"{'Yes' if all_same_sign else 'NO — sign flips detected'}"
        )
        lines.append(
            f"  - All significant at 5%: "
            f"{'Yes' if all_sig else 'No'} "
            f"({(df['pvalue'] < 0.05).sum()}/{len(df)})"
        )
        lines.append(
            f"  - Estimate range: "
            f"[{ests.min():.4f}, {ests.max():.4f}]"
        )
        lines.append("=" * 78)
        return "\n".join(lines)

    def to_latex(self, caption: str = "Robustness Checks") -> str:
        """Export to LaTeX table."""
        df = self.results_df
        n_cols = 5
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            r"\label{tab:robustness}",
            r"\begin{tabular}{lcccc}",
            r"\hline\hline",
            r"Check & Estimate & Std.\ Error & P-value & N \\",
            r"\hline",
        ]
        for _, row in df.iterrows():
            stars = _stars_latex(row['pvalue'])
            lines.append(
                f"{row['check']} & "
                f"{row['estimate']:.4f}{stars} & "
                f"({row['se']:.4f}) & "
                f"{row['pvalue']:.4f} & "
                f"{int(row['nobs']):,d} \\\\"
            )
        lines += [
            r"\hline\hline",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item *** p<0.01; ** p<0.05; * p<0.1",
            r"\end{tablenotes}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def plot(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        color: str = "#2C3E50",
        baseline_color: str = "#E74C3C",
    ):
        """
        Forest-plot style visualization of robustness checks.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required. pip install matplotlib")

        df = self.results_df.copy()
        n = len(df)
        if figsize is None:
            figsize = (8, max(3, n * 0.4))

        fig, ax = plt.subplots(figsize=figsize)

        ys = np.arange(n)
        for i, (_, row) in enumerate(df.iterrows()):
            c = baseline_color if row['check'] == 'Baseline' else color
            ax.errorbar(
                row['estimate'], i,
                xerr=1.96 * row['se'],
                fmt='o', color=c, markersize=5,
                capsize=3, linewidth=1.2,
            )

        ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.axvline(
            self.baseline_estimate,
            color=baseline_color, linewidth=0.8, linestyle=':',
            alpha=0.6,
        )
        ax.set_yticks(ys)
        ax.set_yticklabels(df['check'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(f"Estimate of '{self.x}'")
        ax.set_title(title or "Robustness Checks", fontsize=12)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        return fig, ax


def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""


def _stars_latex(p: float) -> str:
    if p < 0.01:
        return "^{***}"
    elif p < 0.05:
        return "^{**}"
    elif p < 0.1:
        return "^{*}"
    return ""


# ---------------------------------------------------------------------------
# Internal: quick OLS for a single spec
# ---------------------------------------------------------------------------

def _quick_ols(
    data: pd.DataFrame,
    y_col: str,
    x_col: str,
    control_cols: List[str],
    se_type: str = 'hc1',
    cluster_col: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run OLS and return dict with key-variable stats."""
    all_cols = [y_col, x_col] + control_cols
    if cluster_col:
        all_cols.append(cluster_col)
    df = data[list(set(all_cols))].dropna()

    if len(df) < len(control_cols) + 5:
        return None

    Y = df[y_col].values.astype(float)
    X_vars = [x_col] + control_cols
    X = np.column_stack([np.ones(len(df)), df[X_vars].values.astype(float)])
    n, k = X.shape

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None

    params = XtX_inv @ X.T @ Y
    resid = Y - X @ params

    if se_type == 'nonrobust':
        sigma2 = np.sum(resid ** 2) / (n - k)
        vcov = sigma2 * XtX_inv
    elif se_type == 'cluster' and cluster_col:
        clusters = df[cluster_col].values
        unique_c = np.unique(clusters)
        G = len(unique_c)
        if G < 2:
            return None
        meat = np.zeros((k, k))
        for c in unique_c:
            idx = clusters == c
            Xc = X[idx]
            uc = resid[idx]
            score = Xc.T @ uc
            meat += np.outer(score, score)
        correction = (G / (G - 1)) * ((n - 1) / (n - k))
        vcov = correction * XtX_inv @ meat @ XtX_inv
    else:
        # HC1
        u2 = resid ** 2
        meat = X.T @ np.diag(u2) @ X * n / (n - k)
        vcov = XtX_inv @ meat @ XtX_inv

    se = np.sqrt(np.diag(vcov))
    beta_x = params[1]
    se_x = se[1]
    df_resid = n - k
    t_stat = beta_x / se_x if se_x > 0 else np.inf
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_resid))
    t_crit = stats.t.ppf(0.975, df_resid)
    r2 = 1 - np.sum(resid ** 2) / np.sum((Y - Y.mean()) ** 2)

    return {
        'estimate': beta_x,
        'se': se_x,
        'ci_lower': beta_x - t_crit * se_x,
        'ci_upper': beta_x + t_crit * se_x,
        'pvalue': p_val,
        'nobs': n,
        'r_squared': r2,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def robustness_report(
    data: pd.DataFrame,
    formula: str,
    x: str,
    cluster_var: Optional[str] = None,
    extra_controls: Optional[List[str]] = None,
    drop_controls: Optional[List[str]] = None,
    winsor_levels: Optional[List[float]] = None,
    trim_pct: Optional[float] = None,
    subsets: Optional[Dict[str, pd.Series]] = None,
) -> RobustnessResult:
    """
    Run an automated battery of robustness checks.

    Given a baseline regression specified by *formula*, this function
    automatically runs the following checks and compiles results:

    1. **Baseline** — the original specification.
    2. **SE variants** — OLS / Robust (HC1) / Clustered.
    3. **Add controls** — each variable in *extra_controls* added one by one.
    4. **Drop controls** — each variable in *drop_controls* removed.
    5. **Winsorize Y** — at each level in *winsor_levels*.
    6. **Trim outliers** — drop obs where |Y - mean| > *trim_pct* × range.
    7. **Subsamples** — user-defined subsample masks.

    Parameters
    ----------
    data : DataFrame
        Analysis dataset.
    formula : str
        Baseline regression formula, e.g. ``"y ~ x1 + x2 + x3"``.
    x : str
        Key explanatory variable whose estimate stability is assessed.
    cluster_var : str, optional
        Column for clustered SE check.
    extra_controls : list of str, optional
        Additional controls to add (one-by-one) beyond baseline.
    drop_controls : list of str, optional
        Baseline controls to drop (one-by-one) for sensitivity.
    winsor_levels : list of float, optional
        Winsorization percentiles, e.g. ``[0.01, 0.05]``.
    trim_pct : float, optional
        Drop observations beyond this percentile from both tails.
        E.g. ``0.01`` trims top and bottom 1 %.
    subsets : dict[str, Series], optional
        Named boolean masks for subsample checks.

    Returns
    -------
    RobustnessResult
        Container with ``.summary()``, ``.plot()``, ``.to_latex()``,
        ``.results_df``.

    Examples
    --------
    >>> import statspai as sp
    >>> report = sp.robustness_report(
    ...     data=df,
    ...     formula="wage ~ education + experience",
    ...     x='education',
    ...     cluster_var='region',
    ...     extra_controls=['female', 'age'],
    ...     winsor_levels=[0.01, 0.05],
    ... )
    >>> report.plot()
    >>> print(report.summary())
    """
    # Parse formula → y, controls
    from ..core.utils import parse_formula
    parsed = parse_formula(formula)
    y_col = parsed['dependent']
    all_rhs = parsed['exogenous']

    # Separate x from controls
    controls_base = [v for v in all_rhs if v != x]
    if x not in all_rhs:
        raise ValueError(
            f"Key variable '{x}' not found in formula RHS. "
            f"Found: {all_rhs}"
        )

    rows: List[Dict[str, Any]] = []

    # ---- (1) Baseline with HC1 ----------------------------------------
    res = _quick_ols(data, y_col, x, controls_base, se_type='hc1')
    if res is None:
        raise ValueError("Baseline regression failed. Check data/formula.")
    res['check'] = 'Baseline'
    baseline_est = res['estimate']
    baseline_se = res['se']
    rows.append(res)

    # ---- (2) SE variants -----------------------------------------------
    for se_label, se_type, cl in [
        ('SE: OLS (non-robust)', 'nonrobust', None),
        ('SE: Robust (HC1)', 'hc1', None),
    ]:
        if se_label == 'SE: Robust (HC1)':
            continue  # baseline already uses HC1
        r = _quick_ols(data, y_col, x, controls_base, se_type=se_type)
        if r:
            r['check'] = se_label
            rows.append(r)

    if cluster_var:
        r = _quick_ols(
            data, y_col, x, controls_base,
            se_type='cluster', cluster_col=cluster_var,
        )
        if r:
            r['check'] = f'SE: Clustered ({cluster_var})'
            rows.append(r)

    # ---- (3) Add controls one-by-one -----------------------------------
    if extra_controls:
        for ctrl in extra_controls:
            if ctrl in controls_base or ctrl == x or ctrl == y_col:
                continue
            r = _quick_ols(
                data, y_col, x, controls_base + [ctrl], se_type='hc1',
            )
            if r:
                r['check'] = f'+ {ctrl}'
                rows.append(r)

    # ---- (4) Drop controls one-by-one ----------------------------------
    drop_list = drop_controls or controls_base
    for ctrl in drop_list:
        if ctrl not in controls_base:
            continue
        reduced = [c for c in controls_base if c != ctrl]
        r = _quick_ols(data, y_col, x, reduced, se_type='hc1')
        if r:
            r['check'] = f'- {ctrl}'
            rows.append(r)

    # ---- (5) Winsorize Y -----------------------------------------------
    if winsor_levels:
        for level in winsor_levels:
            df_w = data.copy()
            lo = df_w[y_col].quantile(level)
            hi = df_w[y_col].quantile(1 - level)
            df_w[y_col] = df_w[y_col].clip(lo, hi)
            r = _quick_ols(df_w, y_col, x, controls_base, se_type='hc1')
            if r:
                r['check'] = f'Winsorize Y ({level:.0%})'
                rows.append(r)

    # ---- (6) Trim outliers ---------------------------------------------
    if trim_pct:
        lo = data[y_col].quantile(trim_pct)
        hi = data[y_col].quantile(1 - trim_pct)
        df_t = data[(data[y_col] >= lo) & (data[y_col] <= hi)]
        r = _quick_ols(df_t, y_col, x, controls_base, se_type='hc1')
        if r:
            r['check'] = f'Trim Y ({trim_pct:.0%} tails)'
            rows.append(r)

    # ---- (7) Subsamples ------------------------------------------------
    if subsets:
        for label, mask in subsets.items():
            r = _quick_ols(
                data.loc[mask], y_col, x, controls_base, se_type='hc1',
            )
            if r:
                r['check'] = f'Sub: {label}'
                rows.append(r)

    results_df = pd.DataFrame(rows)
    return RobustnessResult(
        results_df=results_df,
        x=x,
        baseline_estimate=baseline_est,
        baseline_se=baseline_se,
        n_checks=len(results_df),
    )
