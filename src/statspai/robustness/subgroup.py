"""
Subgroup Heterogeneity Analysis with Forest Plot

Run treatment-effect estimation across user-defined subgroups and
present results in a publication-ready forest plot.

This solves a common pain point: researchers manually split samples,
re-run regressions, and hand-assemble forest plots.  Now one call
does it all and includes an interaction-based test for heterogeneity.

Usage
-----
>>> import statspai as sp
>>> result = sp.subgroup_analysis(
...     data=df,
...     formula="wage ~ education + experience",
...     x='education',
...     by={'Gender': 'female', 'Region': 'region'},
... )
>>> result.plot()
>>> result.summary()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SubgroupResult:
    """Container for subgroup heterogeneity analysis."""

    results_df: pd.DataFrame
    """Columns: group_var, group_val, estimate, se, ci_lower, ci_upper,
    pvalue, nobs, label."""

    x: str
    overall_estimate: float
    overall_se: float
    het_tests: Dict[str, Dict[str, float]]
    """Per group_var: chi2, pvalue, df."""

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "  Subgroup Heterogeneity Analysis",
            "=" * 72,
            "",
            f"  Key variable:       {self.x}",
            f"  Overall estimate:   {self.overall_estimate: .4f} "
            f"({self.overall_se:.4f})",
            "",
        ]

        for gvar in self.results_df['group_var'].unique():
            sub = self.results_df[self.results_df['group_var'] == gvar]
            lines.append("-" * 72)
            lines.append(f"  Subgroups by: {gvar}")
            lines.append(
                f"  {'Group':<25s} {'Estimate':>10s} {'SE':>10s} "
                f"{'95% CI':>22s} {'N':>8s}"
            )
            lines.append("-" * 72)
            for _, row in sub.iterrows():
                stars = _stars(row['pvalue'])
                lines.append(
                    f"  {str(row['group_val']):<25s} "
                    f"{row['estimate']:>9.4f}{stars:<3s} "
                    f"({row['se']:>8.4f}) "
                    f"[{row['ci_lower']:>7.4f}, {row['ci_upper']:>7.4f}]  "
                    f"{int(row['nobs']):>7d}"
                )
            # Heterogeneity test
            if gvar in self.het_tests:
                ht = self.het_tests[gvar]
                lines.append(
                    f"\n  Heterogeneity test: "
                    f"chi2({ht['df']:.0f}) = {ht['chi2']:.3f}, "
                    f"p = {ht['pvalue']:.4f}"
                )
                if ht['pvalue'] < 0.05:
                    lines.append(
                        "  → Significant heterogeneity detected (p<0.05)"
                    )
                else:
                    lines.append(
                        "  → No significant heterogeneity (p≥0.05)"
                    )
            lines.append("")

        lines.append("=" * 72)
        lines.append("  * p<0.1, ** p<0.05, *** p<0.01")
        return "\n".join(lines)

    def to_latex(
        self, caption: str = "Subgroup Heterogeneity Analysis",
    ) -> str:
        """Export to LaTeX."""
        df = self.results_df
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            r"\label{tab:heterogeneity}",
            r"\begin{tabular}{llcccr}",
            r"\hline\hline",
            r"Dimension & Subgroup & Estimate & Std.\ Error "
            r"& 95\% CI & N \\",
            r"\hline",
        ]
        for gvar in df['group_var'].unique():
            sub = df[df['group_var'] == gvar]
            first = True
            for _, row in sub.iterrows():
                stars = _stars_latex(row['pvalue'])
                dim_col = gvar if first else ""
                first = False
                lines.append(
                    f"{dim_col} & {row['group_val']} & "
                    f"{row['estimate']:.4f}{stars} & "
                    f"({row['se']:.4f}) & "
                    f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] & "
                    f"{int(row['nobs']):,d} \\\\"
                )
            # Add het test row
            if gvar in self.het_tests:
                ht = self.het_tests[gvar]
                lines.append(
                    f"& \\multicolumn{{4}}{{l}}"
                    f"{{Heterogeneity: $\\chi^2({ht['df']:.0f})"
                    f"={ht['chi2']:.3f}$, $p={ht['pvalue']:.4f}$}} "
                    r"& \\"
                )
            lines.append(r"\hline")

        lines += [
            r"\hline",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item *** p<0.01; ** p<0.05; * p<0.1",
            r"\item Heterogeneity tested via interaction-based Wald test.",
            r"\end{tablenotes}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def plot(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        color: str = "#2C3E50",
        overall_color: str = "#E74C3C",
    ):
        """
        Forest plot of subgroup estimates.

        Returns
        -------
        fig, ax
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch
        except ImportError:
            raise ImportError("matplotlib required. pip install matplotlib")

        df = self.results_df
        groups = df['group_var'].unique()

        # Build label list with section headers
        labels = []
        positions = []
        is_header = []
        estimates = []
        ci_lo = []
        ci_hi = []
        colors = []

        pos = 0
        for gvar in groups:
            # Section header
            labels.append(f"By {gvar}")
            positions.append(pos)
            is_header.append(True)
            estimates.append(None)
            ci_lo.append(None)
            ci_hi.append(None)
            colors.append(None)
            pos += 1

            sub = df[df['group_var'] == gvar]
            for _, row in sub.iterrows():
                labels.append(f"  {row['group_val']}")
                positions.append(pos)
                is_header.append(False)
                estimates.append(row['estimate'])
                ci_lo.append(row['ci_lower'])
                ci_hi.append(row['ci_upper'])
                colors.append(color)
                pos += 1
            pos += 0.5  # gap between groups

        # Add overall
        labels.append("Overall")
        positions.append(pos)
        is_header.append(False)
        estimates.append(self.overall_estimate)
        t_crit = 1.96
        ci_lo.append(self.overall_estimate - t_crit * self.overall_se)
        ci_hi.append(self.overall_estimate + t_crit * self.overall_se)
        colors.append(overall_color)

        n_rows = len(labels)
        if figsize is None:
            figsize = (8, max(3, n_rows * 0.4))

        fig, ax = plt.subplots(figsize=figsize)

        for i in range(n_rows):
            if is_header[i]:
                ax.text(
                    ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else -0.5,
                    positions[i], labels[i],
                    fontsize=9, fontweight='bold', va='center',
                )
            else:
                ax.errorbar(
                    estimates[i], positions[i],
                    xerr=[[estimates[i] - ci_lo[i]], [ci_hi[i] - estimates[i]]],
                    fmt='D' if labels[i] == "Overall" else 'o',
                    color=colors[i],
                    markersize=6 if labels[i] == "Overall" else 5,
                    capsize=3, linewidth=1.2,
                )

        ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.axvline(
            self.overall_estimate, color=overall_color,
            linewidth=0.6, linestyle=':', alpha=0.5,
        )

        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(f"Estimate of '{self.x}'")
        ax.set_title(
            title or "Subgroup Heterogeneity Analysis", fontsize=12,
        )
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


def _quick_ols_full(
    data: pd.DataFrame,
    y_col: str,
    x_col: str,
    control_cols: List[str],
) -> Optional[Dict[str, Any]]:
    """Quick OLS with HC1, return key variable stats."""
    all_cols = list(set([y_col, x_col] + control_cols))
    df = data[all_cols].dropna()
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

    # HC1
    u2 = resid ** 2
    meat = X.T @ np.diag(u2) @ X * n / (n - k)
    vcov = XtX_inv @ meat @ XtX_inv

    se = np.sqrt(np.diag(vcov))
    beta_x = params[1]
    se_x = se[1]
    df_resid = n - k
    t_crit = stats.t.ppf(0.975, df_resid)
    p_val = 2 * (1 - stats.t.cdf(abs(beta_x / se_x), df_resid))

    return {
        'estimate': beta_x,
        'se': se_x,
        'ci_lower': beta_x - t_crit * se_x,
        'ci_upper': beta_x + t_crit * se_x,
        'pvalue': p_val,
        'nobs': n,
    }


def _interaction_het_test(
    data: pd.DataFrame,
    y_col: str,
    x_col: str,
    control_cols: List[str],
    group_col: str,
) -> Optional[Dict[str, float]]:
    """
    Wald test for heterogeneity via interaction terms.

    Runs Y ~ x + controls + group_dummies + x * group_dummies
    and tests H0: all interaction coefficients = 0.
    """
    df = data[[y_col, x_col, group_col] + control_cols].dropna()
    if len(df) < 20:
        return None

    groups = sorted(df[group_col].unique(), key=str)
    if len(groups) < 2:
        return None

    # Create dummies (drop first)
    ref = groups[0]
    interaction_names = []
    for g in groups[1:]:
        col_d = f"_d_{g}"
        col_int = f"_int_{g}"
        df[col_d] = (df[group_col] == g).astype(float)
        df[col_int] = df[col_d] * df[x_col]
        interaction_names.append(col_int)

    dummy_names = [f"_d_{g}" for g in groups[1:]]
    all_x = [x_col] + control_cols + dummy_names + interaction_names

    Y = df[y_col].values.astype(float)
    X = np.column_stack([np.ones(len(df)), df[all_x].values.astype(float)])
    n, k = X.shape

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None

    params = XtX_inv @ X.T @ Y
    resid = Y - X @ params

    # HC1 vcov
    u2 = resid ** 2
    meat = X.T @ np.diag(u2) @ X * n / (n - k)
    vcov = XtX_inv @ meat @ XtX_inv

    # Indices of interaction terms (last len(interaction_names) columns)
    n_int = len(interaction_names)
    int_idx = list(range(k - n_int, k))

    beta_int = params[int_idx]
    V_int = vcov[np.ix_(int_idx, int_idx)]

    try:
        V_int_inv = np.linalg.inv(V_int)
    except np.linalg.LinAlgError:
        return None

    chi2 = float(beta_int @ V_int_inv @ beta_int)
    df_test = n_int
    p_val = 1 - stats.chi2.cdf(chi2, df_test)

    return {'chi2': chi2, 'pvalue': p_val, 'df': float(df_test)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def subgroup_analysis(
    data: pd.DataFrame,
    formula: str,
    x: str,
    by: Dict[str, str],
    robust: str = 'hc1',
) -> SubgroupResult:
    """
    Run subgroup heterogeneity analysis with forest plot.

    Estimate the effect of *x* on *y* within each subgroup defined
    by the variables in *by*, and test for heterogeneity using
    interaction-based Wald tests.

    Parameters
    ----------
    data : DataFrame
        Analysis dataset.
    formula : str
        Regression formula, e.g. ``"wage ~ education + experience"``.
    x : str
        Key explanatory variable.
    by : dict[str, str]
        Mapping of *display name* → *column name* for grouping.
        Example: ``{'Gender': 'female', 'Region': 'region'}``.
    robust : str, default 'hc1'
        Standard error type for subgroup regressions.

    Returns
    -------
    SubgroupResult
        Container with ``.plot()``, ``.summary()``, ``.to_latex()``,
        ``.results_df``.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.subgroup_analysis(
    ...     data=df,
    ...     formula="wage ~ education + experience",
    ...     x='education',
    ...     by={'Gender': 'female', 'Region': 'region'},
    ... )
    >>> result.plot()
    >>> print(result.summary())
    """
    from ..core.utils import parse_formula
    parsed = parse_formula(formula)
    y_col = parsed['dependent']
    all_rhs = parsed['exogenous']
    controls_base = [v for v in all_rhs if v != x]

    # Overall estimate
    overall = _quick_ols_full(data, y_col, x, controls_base)
    if overall is None:
        raise ValueError("Overall regression failed.")

    rows = []
    het_tests = {}

    for display_name, col_name in by.items():
        if col_name not in data.columns:
            raise ValueError(f"Column '{col_name}' not found in data.")

        # Remove group column from controls if present
        ctrl_clean = [c for c in controls_base if c != col_name]

        groups = sorted(data[col_name].dropna().unique(), key=str)

        for g in groups:
            mask = data[col_name] == g
            sub_data = data.loc[mask]
            res = _quick_ols_full(sub_data, y_col, x, ctrl_clean)
            if res is not None:
                res['group_var'] = display_name
                res['group_val'] = str(g)
                res['label'] = f"{display_name}: {g}"
                rows.append(res)

        # Heterogeneity test
        ht = _interaction_het_test(
            data, y_col, x, ctrl_clean, col_name,
        )
        if ht is not None:
            het_tests[display_name] = ht

    if not rows:
        raise ValueError("No valid subgroup regressions succeeded.")

    results_df = pd.DataFrame(rows)

    return SubgroupResult(
        results_df=results_df,
        x=x,
        overall_estimate=overall['estimate'],
        overall_se=overall['se'],
        het_tests=het_tests,
    )
