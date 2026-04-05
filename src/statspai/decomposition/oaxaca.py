"""
Oaxaca-Blinder and Gelbach decomposition estimators.

Implements:
- Threefold Oaxaca-Blinder decomposition (Blinder 1973, Oaxaca 1973)
  with Neumark (1988), Cotton (1988), and Reimers (1983) reference weights.
- Gelbach (2016) sequential decomposition of omitted variable bias.

All estimation uses numpy/scipy only (no statsmodels dependency).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats


# ════════════════════════════════════════════════════════════════════════
# Internal OLS helper
# ════════════════════════════════════════════════════════════════════════

def _ols(
    y: np.ndarray,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OLS via the normal equations with HC1 robust variance.

    Parameters
    ----------
    y : (n,) array
    X : (n, k) array — must include constant if desired.

    Returns
    -------
    beta : (k,) coefficient vector
    vcov : (k, k) HC1 (White) robust variance-covariance matrix
    resid : (n,) residuals
    """
    n, k = X.shape
    # QR decomposition for numerical stability
    Q, R = np.linalg.qr(X, mode='reduced')
    beta = np.linalg.solve(R, Q.T @ y)
    resid = y - X @ beta

    # HC1 robust variance: (X'X)^{-1} X' diag(e^2) X (X'X)^{-1} * n/(n-k)
    XtX_inv = np.linalg.inv(R.T @ R)
    e2 = resid ** 2
    meat = (X * e2[:, None]).T @ X  # X' diag(e^2) X
    vcov = XtX_inv @ meat @ XtX_inv * (n / (n - k))

    return beta, vcov, resid


def _add_constant(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones."""
    n = X.shape[0]
    return np.column_stack([np.ones(n), X])


def _significance_stars(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    if pval < 0.1:
        return "+"
    return ""


# ════════════════════════════════════════════════════════════════════════
# OaxacaResult
# ════════════════════════════════════════════════════════════════════════

class OaxacaResult:
    """
    Result container for Oaxaca-Blinder decomposition.

    Attributes
    ----------
    overall : dict
        Keys: ``'gap'``, ``'explained'``, ``'unexplained'``,
        ``'explained_se'``, ``'unexplained_se'``,
        ``'unexplained_a'``, ``'unexplained_b'`` (threefold components).
    detailed : pd.DataFrame
        Variable-level decomposition with columns ``contribution``,
        ``se``, ``pct_of_explained``.
    group_stats : dict
        Per-group means, coefficients, standard errors, sample sizes.
    reference : str or int
        Reference weight specification used.
    """

    def __init__(
        self,
        overall: Dict[str, float],
        detailed: pd.DataFrame,
        group_stats: Dict[str, Any],
        reference: Union[str, int],
        var_names: List[str],
    ):
        self.overall = overall
        self.detailed = detailed
        self.group_stats = group_stats
        self.reference = reference
        self.var_names = var_names

    # ── Pretty text summary ──────────────────────────────────────────

    def summary(self) -> str:
        """Return formatted decomposition summary."""
        o = self.overall
        gs = self.group_stats

        gap = o['gap']
        expl = o['explained']
        unex = o['unexplained']
        expl_se = o['explained_se']
        unex_se = o['unexplained_se']

        expl_z = expl / expl_se if expl_se > 0 else 0.0
        unex_z = unex / unex_se if unex_se > 0 else 0.0
        expl_p = 2 * (1 - stats.norm.cdf(abs(expl_z)))
        unex_p = 2 * (1 - stats.norm.cdf(abs(unex_z)))

        expl_pct = (expl / gap * 100) if gap != 0 else float('nan')
        unex_pct = (unex / gap * 100) if gap != 0 else float('nan')

        ref_label = (
            f"Group {self.reference}"
            if isinstance(self.reference, int)
            else self.reference.capitalize()
        )

        lines: list[str] = []
        w = 58
        lines.append("━" * w)
        lines.append("  Oaxaca-Blinder Decomposition")
        lines.append("━" * w)
        lines.append(
            f"  Group A ({gs['group_var']}=0): "
            f"mean {gs['y_var']} = {gs['mean_a']:.4f}    N = {gs['n_a']}"
        )
        lines.append(
            f"  Group B ({gs['group_var']}=1): "
            f"mean {gs['y_var']} = {gs['mean_b']:.4f}    N = {gs['n_b']}"
        )
        lines.append(f"  Raw gap:            {gap:.4f}")
        lines.append("")
        lines.append(f"  Decomposition (reference: {ref_label}):")
        lines.append("  " + "─" * (w - 4))

        lines.append(
            f"  Explained:          {expl:>10.4f}{_significance_stars(expl_p):<4s}"
            f"  ({expl_pct:>5.1f}% of gap)   SE={expl_se:.4f}"
        )
        lines.append(
            f"  Unexplained:        {unex:>10.4f}{_significance_stars(unex_p):<4s}"
            f"  ({unex_pct:>5.1f}% of gap)   SE={unex_se:.4f}"
        )

        # Threefold pieces if present
        if 'unexplained_a' in o and 'unexplained_b' in o:
            ua = o['unexplained_a']
            ub = o['unexplained_b']
            lines.append("")
            lines.append("  Threefold decomposition:")
            lines.append("  " + "─" * (w - 4))
            lines.append(f"  Endowments:         {expl:>10.4f}")
            lines.append(f"  Coefficients (A):   {ua:>10.4f}")
            lines.append(f"  Coefficients (B):   {ub:>10.4f}")

        # Detailed table
        if not self.detailed.empty:
            lines.append("")
            lines.append("  Detailed explained component:")
            lines.append("  " + "─" * (w - 4))
            lines.append(
                f"  {'Variable':<20s}{'Contribution':>14s}  {'% of Explained':>14s}"
            )
            for _, row in self.detailed.iterrows():
                pval_j = row.get('pvalue', 1.0)
                stars = _significance_stars(pval_j)
                pct_str = f"{row['pct_of_explained']:.1f}%"
                lines.append(
                    f"  {row['variable']:<20s}"
                    f"{row['contribution']:>12.4f}{stars:<3s}"
                    f"  {pct_str:>13s}"
                )

        lines.append("━" * w)
        text = "\n".join(lines)
        print(text)
        return text

    # ── Plotting ─────────────────────────────────────────────────────

    def plot(self, figsize=(8, 5), color_pos="#2196F3", color_neg="#FF5722"):
        """
        Horizontal bar chart of variable-level explained contributions.

        Returns
        -------
        (fig, ax)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )

        df = self.detailed.copy()
        if df.empty:
            raise ValueError(
                "No detailed decomposition available. "
                "Re-run oaxaca() with detail=True."
            )

        df = df.sort_values("contribution")
        colors = [color_pos if v >= 0 else color_neg for v in df["contribution"]]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df["variable"], df["contribution"], color=colors, edgecolor="white")
        ax.set_xlabel("Contribution to explained gap")
        ax.set_title("Oaxaca-Blinder: Detailed Decomposition")
        ax.axvline(0, color="black", linewidth=0.8)
        fig.tight_layout()
        return fig, ax

    # ── LaTeX ────────────────────────────────────────────────────────

    def to_latex(self) -> str:
        """Return a LaTeX-formatted decomposition table."""
        o = self.overall
        lines: list[str] = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Oaxaca-Blinder Decomposition}")
        lines.append(r"\begin{tabular}{lcc}")
        lines.append(r"\toprule")
        lines.append(r"Component & Estimate & Std.\ Error \\")
        lines.append(r"\midrule")
        lines.append(
            f"Raw gap & {o['gap']:.4f} & \\\\"
        )
        lines.append(
            f"Explained & {o['explained']:.4f} & {o['explained_se']:.4f} \\\\"
        )
        lines.append(
            f"Unexplained & {o['unexplained']:.4f} & {o['unexplained_se']:.4f} \\\\"
        )

        if not self.detailed.empty:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{3}{l}{\textit{Detailed explained:}} \\")
            for _, row in self.detailed.iterrows():
                lines.append(
                    f"\\quad {row['variable']} & "
                    f"{row['contribution']:.4f} & {row['se']:.4f} \\\\"
                )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    # ── Jupyter HTML repr ────────────────────────────────────────────

    def _repr_html_(self) -> str:
        o = self.overall
        gs = self.group_stats
        gap = o['gap']
        expl = o['explained']
        unex = o['unexplained']
        expl_pct = (expl / gap * 100) if gap != 0 else float('nan')
        unex_pct = (unex / gap * 100) if gap != 0 else float('nan')

        html = (
            "<div style='font-family: monospace; padding: 10px;'>"
            "<h3>Oaxaca-Blinder Decomposition</h3>"
            "<table style='border-collapse: collapse;'>"
            f"<tr><td>Group A ({gs['group_var']}=0)</td>"
            f"<td>mean = {gs['mean_a']:.4f}, N = {gs['n_a']}</td></tr>"
            f"<tr><td>Group B ({gs['group_var']}=1)</td>"
            f"<td>mean = {gs['mean_b']:.4f}, N = {gs['n_b']}</td></tr>"
            f"<tr><td><b>Raw gap</b></td><td><b>{gap:.4f}</b></td></tr>"
            f"<tr><td>Explained</td>"
            f"<td>{expl:.4f} ({expl_pct:.1f}%)</td></tr>"
            f"<tr><td>Unexplained</td>"
            f"<td>{unex:.4f} ({unex_pct:.1f}%)</td></tr>"
            "</table>"
        )

        if not self.detailed.empty:
            html += (
                "<h4>Detailed Explained Component</h4>"
                "<table style='border-collapse: collapse;'>"
                "<tr><th style='text-align:left;'>Variable</th>"
                "<th>Contribution</th><th>SE</th>"
                "<th>% of Explained</th></tr>"
            )
            for _, row in self.detailed.iterrows():
                html += (
                    f"<tr><td style='text-align:left;'>{row['variable']}</td>"
                    f"<td>{row['contribution']:.4f}</td>"
                    f"<td>{row['se']:.4f}</td>"
                    f"<td>{row['pct_of_explained']:.1f}%</td></tr>"
                )
            html += "</table>"

        html += "</div>"
        return html

    def __repr__(self) -> str:
        return (
            f"OaxacaResult(gap={self.overall['gap']:.4f}, "
            f"explained={self.overall['explained']:.4f}, "
            f"unexplained={self.overall['unexplained']:.4f}, "
            f"reference={self.reference!r})"
        )


# ════════════════════════════════════════════════════════════════════════
# GelbachResult
# ════════════════════════════════════════════════════════════════════════

class GelbachResult:
    """
    Result container for Gelbach (2016) decomposition.

    Attributes
    ----------
    total_change : float
        Total change in the base coefficient when added controls are
        included: beta_base - beta_full.
    decomposition : pd.DataFrame
        Per-variable contributions with columns ``delta``, ``se``,
        ``pct_of_change``.
    base_coef : float
        Coefficient of interest from the base (short) regression.
    full_coef : float
        Coefficient of interest from the full (long) regression.
    base_var : str
        Name of the variable of interest.
    """

    def __init__(
        self,
        total_change: float,
        decomposition: pd.DataFrame,
        base_coef: float,
        full_coef: float,
        base_var: str,
    ):
        self.total_change = total_change
        self.decomposition = decomposition
        self.base_coef = base_coef
        self.full_coef = full_coef
        self.base_var = base_var

    def summary(self) -> str:
        """Return formatted Gelbach decomposition summary."""
        lines: list[str] = []
        w = 58
        lines.append("━" * w)
        lines.append("  Gelbach (2016) Decomposition")
        lines.append("━" * w)
        lines.append(
            f"  Base coefficient on '{self.base_var}':  {self.base_coef:.4f}"
        )
        lines.append(
            f"  Full coefficient on '{self.base_var}':  {self.full_coef:.4f}"
        )
        lines.append(
            f"  Total change (base - full):       {self.total_change:.4f}"
        )
        lines.append("")
        lines.append("  Decomposition of the change:")
        lines.append("  " + "─" * (w - 4))
        lines.append(
            f"  {'Added variable':<22s}{'delta':>10s}{'SE':>10s}"
            f"{'% of change':>14s}"
        )
        lines.append("  " + "─" * (w - 4))
        for _, row in self.decomposition.iterrows():
            pval = row.get('pvalue', 1.0)
            stars = _significance_stars(pval)
            pct_str = f"{row['pct_of_change']:.1f}%"
            lines.append(
                f"  {row['variable']:<22s}"
                f"{row['delta']:>8.4f}{stars:<3s}"
                f"{row['se']:>9.4f}"
                f"{pct_str:>13s}"
            )
        lines.append("━" * w)
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, figsize=(8, 5), color="#4CAF50"):
        """
        Horizontal bar chart of Gelbach contributions.

        Returns
        -------
        (fig, ax)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )

        df = self.decomposition.sort_values("delta")
        colors = [color if v >= 0 else "#FF5722" for v in df["delta"]]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df["variable"], df["delta"], color=colors, edgecolor="white")
        ax.set_xlabel(f"Contribution to change in '{self.base_var}' coefficient")
        ax.set_title("Gelbach Decomposition")
        ax.axvline(0, color="black", linewidth=0.8)
        fig.tight_layout()
        return fig, ax

    def to_latex(self) -> str:
        """Return LaTeX table of the decomposition."""
        lines: list[str] = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Gelbach Decomposition}")
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\toprule")
        lines.append(r"Added Variable & $\hat\delta$ & SE & \% of Change \\")
        lines.append(r"\midrule")
        for _, row in self.decomposition.iterrows():
            lines.append(
                f"{row['variable']} & {row['delta']:.4f} & "
                f"{row['se']:.4f} & {row['pct_of_change']:.1f}\\% \\\\"
            )
        lines.append(r"\midrule")
        lines.append(
            f"Total & {self.total_change:.4f} & & 100.0\\% \\\\"
        )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        html = (
            "<div style='font-family: monospace; padding: 10px;'>"
            "<h3>Gelbach (2016) Decomposition</h3>"
            f"<p>Base coef on <b>{self.base_var}</b>: {self.base_coef:.4f} "
            f"&rarr; Full coef: {self.full_coef:.4f} "
            f"(change = {self.total_change:.4f})</p>"
            "<table style='border-collapse: collapse;'>"
            "<tr><th style='text-align:left;'>Variable</th>"
            "<th>delta</th><th>SE</th><th>% of Change</th></tr>"
        )
        for _, row in self.decomposition.iterrows():
            html += (
                f"<tr><td style='text-align:left;'>{row['variable']}</td>"
                f"<td>{row['delta']:.4f}</td>"
                f"<td>{row['se']:.4f}</td>"
                f"<td>{row['pct_of_change']:.1f}%</td></tr>"
            )
        html += (
            f"<tr style='border-top: 1px solid #ccc;'>"
            f"<td><b>Total</b></td>"
            f"<td><b>{self.total_change:.4f}</b></td>"
            f"<td></td><td>100.0%</td></tr>"
            "</table></div>"
        )
        return html

    def __repr__(self) -> str:
        return (
            f"GelbachResult(base_var='{self.base_var}', "
            f"base_coef={self.base_coef:.4f}, "
            f"full_coef={self.full_coef:.4f}, "
            f"total_change={self.total_change:.4f})"
        )


# ════════════════════════════════════════════════════════════════════════
# oaxaca() — Main entry point
# ════════════════════════════════════════════════════════════════════════

def oaxaca(
    data: pd.DataFrame,
    y: str,
    group: str,
    x: Sequence[str],
    reference: Union[int, str] = 0,
    detail: bool = True,
    alpha: float = 0.05,
) -> OaxacaResult:
    """
    Oaxaca-Blinder decomposition of mean outcome gaps.

    Decomposes the mean difference in ``y`` between two groups defined
    by ``group`` into an "explained" (endowment) component and an
    "unexplained" (coefficient / discrimination) component.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    group : str
        Binary group indicator (0 = Group A, 1 = Group B).
    x : list of str
        Covariate names.
    reference : {0, 1, 'pooled', 'cotton', 'reimers'}, default 0
        Reference coefficient vector beta*:

        - ``0`` — Group A coefficients (beta_A). The "explained" part
          uses Group A's returns as the benchmark.
        - ``1`` — Group B coefficients (beta_B).
        - ``'pooled'`` — Pooled OLS (Neumark 1988).
        - ``'cotton'`` — Sample-size weighted average (Cotton 1988).
        - ``'reimers'`` — Equal-weighted average (Reimers 1983).
    detail : bool, default True
        If True, compute variable-level contributions to the explained
        component.
    alpha : float, default 0.05
        Significance level for p-values.

    Returns
    -------
    OaxacaResult
        Result object with ``.summary()``, ``.plot()``, ``.to_latex()``.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.oaxaca(
    ...     data=df, y="wage", group="female",
    ...     x=["education", "experience", "tenure"],
    ...     reference=0, detail=True,
    ... )
    >>> result.summary()
    """
    # ── Validate inputs ──────────────────────────────────────────────
    data = data.copy()
    x = list(x)

    required_cols = [y, group] + x
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    # Drop missing values
    data = data[required_cols].dropna()

    group_vals = sorted(data[group].unique())
    if set(group_vals) - {0, 1, 0.0, 1.0, True, False}:
        raise ValueError(
            f"Group variable '{group}' must be binary (0/1). "
            f"Found values: {group_vals}"
        )

    # ── Split into two groups ────────────────────────────────────────
    mask_a = data[group].astype(int) == 0
    mask_b = ~mask_a

    y_a = data.loc[mask_a, y].values.astype(float)
    y_b = data.loc[mask_b, y].values.astype(float)
    X_a_raw = data.loc[mask_a, x].values.astype(float)
    X_b_raw = data.loc[mask_b, x].values.astype(float)

    n_a, n_b = len(y_a), len(y_b)
    if n_a < 2 or n_b < 2:
        raise ValueError(
            f"Each group must have at least 2 observations. "
            f"Group A: {n_a}, Group B: {n_b}."
        )

    # Add constant
    X_a = _add_constant(X_a_raw)
    X_b = _add_constant(X_b_raw)

    # ── OLS for each group ───────────────────────────────────────────
    beta_a, vcov_a, _ = _ols(y_a, X_a)
    beta_b, vcov_b, _ = _ols(y_b, X_b)

    # Means (with constant = 1)
    mean_X_a = np.concatenate([[1.0], X_a_raw.mean(axis=0)])
    mean_X_b = np.concatenate([[1.0], X_b_raw.mean(axis=0)])

    mean_y_a = y_a.mean()
    mean_y_b = y_b.mean()
    gap = mean_y_a - mean_y_b

    # ── Reference coefficients (beta*) ───────────────────────────────
    if reference == 0:
        beta_star = beta_a.copy()
        vcov_star = vcov_a.copy()
    elif reference == 1:
        beta_star = beta_b.copy()
        vcov_star = vcov_b.copy()
    elif reference == 'pooled':
        X_all = _add_constant(data[x].values.astype(float))
        y_all = data[y].values.astype(float)
        beta_star, vcov_star, _ = _ols(y_all, X_all)
    elif reference == 'cotton':
        w_a = n_a / (n_a + n_b)
        w_b = n_b / (n_a + n_b)
        beta_star = w_a * beta_a + w_b * beta_b
        vcov_star = w_a**2 * vcov_a + w_b**2 * vcov_b
    elif reference == 'reimers':
        beta_star = 0.5 * beta_a + 0.5 * beta_b
        vcov_star = 0.25 * vcov_a + 0.25 * vcov_b
    else:
        raise ValueError(
            f"Invalid reference: {reference!r}. "
            "Choose from 0, 1, 'pooled', 'cotton', 'reimers'."
        )

    # ── Decomposition ────────────────────────────────────────────────
    diff_X = mean_X_a - mean_X_b  # (k,) including constant
    diff_beta_a = beta_a - beta_star
    diff_beta_b = beta_star - beta_b

    explained = diff_X @ beta_star
    unexplained_a = mean_X_a @ diff_beta_a
    unexplained_b = mean_X_b @ diff_beta_b
    unexplained = unexplained_a + unexplained_b

    # ── Standard errors (delta method) ───────────────────────────────
    # Var(explained) = diff_X' Var(beta*) diff_X
    var_explained = diff_X @ vcov_star @ diff_X

    # Var(unexplained): depends on reference
    # For reference=0 (beta*=beta_a): unexplained = X_a'(beta_a - beta_a) + X_b'(beta_a - beta_b)
    #   = X_b'(beta_a - beta_b) → Var = X_b' (Vcov_a + Vcov_b) X_b
    # General approximation: use combined variance
    if reference == 0:
        var_unexplained = mean_X_b @ (vcov_a + vcov_b) @ mean_X_b
    elif reference == 1:
        var_unexplained = mean_X_a @ (vcov_a + vcov_b) @ mean_X_a
    else:
        # Conservative: sum of both group variances, weighted
        var_unexplained = (
            mean_X_a @ (vcov_a + vcov_star) @ mean_X_a
            + mean_X_b @ (vcov_star + vcov_b) @ mean_X_b
        )

    se_explained = np.sqrt(max(var_explained, 0.0))
    se_unexplained = np.sqrt(max(var_unexplained, 0.0))

    # ── Detailed decomposition ───────────────────────────────────────
    detailed_df = pd.DataFrame()
    if detail:
        # Variable-level contributions (skip the constant at index 0)
        contributions = []
        for j, var in enumerate(x):
            idx = j + 1  # skip constant
            contrib_j = diff_X[idx] * beta_star[idx]
            se_j = abs(diff_X[idx]) * np.sqrt(vcov_star[idx, idx])
            z_j = contrib_j / se_j if se_j > 0 else 0.0
            p_j = 2 * (1 - stats.norm.cdf(abs(z_j)))
            pct_j = (contrib_j / explained * 100) if explained != 0 else float('nan')
            contributions.append({
                'variable': var,
                'contribution': contrib_j,
                'se': se_j,
                'zvalue': z_j,
                'pvalue': p_j,
                'pct_of_explained': pct_j,
            })
        detailed_df = pd.DataFrame(contributions)

    # ── Group stats ──────────────────────────────────────────────────
    var_names_full = ['_cons'] + x
    group_stats = {
        'y_var': y,
        'group_var': group,
        'n_a': n_a,
        'n_b': n_b,
        'mean_a': mean_y_a,
        'mean_b': mean_y_b,
        'beta_a': pd.Series(beta_a, index=var_names_full),
        'beta_b': pd.Series(beta_b, index=var_names_full),
        'se_a': pd.Series(np.sqrt(np.diag(vcov_a)), index=var_names_full),
        'se_b': pd.Series(np.sqrt(np.diag(vcov_b)), index=var_names_full),
        'beta_star': pd.Series(beta_star, index=var_names_full),
        'mean_X_a': pd.Series(mean_X_a, index=var_names_full),
        'mean_X_b': pd.Series(mean_X_b, index=var_names_full),
    }

    overall = {
        'gap': gap,
        'explained': explained,
        'unexplained': unexplained,
        'explained_se': se_explained,
        'unexplained_se': se_unexplained,
        'unexplained_a': unexplained_a,
        'unexplained_b': unexplained_b,
    }

    return OaxacaResult(
        overall=overall,
        detailed=detailed_df,
        group_stats=group_stats,
        reference=reference,
        var_names=x,
    )


# ════════════════════════════════════════════════════════════════════════
# gelbach() — Gelbach (2016) decomposition
# ════════════════════════════════════════════════════════════════════════

def gelbach(
    data: pd.DataFrame,
    y: str,
    base_x: Sequence[str],
    added_x: Sequence[str],
    var_of_interest: Optional[str] = None,
    alpha: float = 0.05,
) -> GelbachResult:
    """
    Gelbach (2016) decomposition of omitted variable bias.

    When controls are added to a regression, the coefficient on a
    variable of interest may change.  This function decomposes that
    change into contributions from each added variable, answering:
    "Which added controls explain the change, and by how much?"

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    base_x : list of str
        Variables in the base (short) specification.
    added_x : list of str
        Variables added to obtain the full (long) specification.
    var_of_interest : str, optional
        Which base variable's coefficient change to decompose.
        Defaults to the first element of ``base_x``.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    GelbachResult
        Result object with ``.summary()``, ``.plot()``, ``.to_latex()``.

    Notes
    -----
    The Gelbach identity:

    .. math::

        \\hat\\beta^{\\text{base}}_k - \\hat\\beta^{\\text{full}}_k
        = \\sum_{j \\in \\text{added}} \\tilde\\gamma_{kj} \\hat\\beta^{\\text{full}}_j

    where :math:`\\tilde\\gamma_{kj}` is the coefficient from regressing
    added variable *j* on all base variables (including constant).

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.gelbach(
    ...     data=df, y="wage",
    ...     base_x=["education"],
    ...     added_x=["experience", "tenure", "union"],
    ... )
    >>> result.summary()
    """
    # ── Validate ─────────────────────────────────────────────────────
    data = data.copy()
    base_x = list(base_x)
    added_x = list(added_x)

    if var_of_interest is None:
        var_of_interest = base_x[0]
    if var_of_interest not in base_x:
        raise ValueError(
            f"var_of_interest='{var_of_interest}' is not in base_x."
        )

    overlap = set(base_x) & set(added_x)
    if overlap:
        raise ValueError(
            f"Variables appear in both base_x and added_x: {overlap}"
        )

    all_x = base_x + added_x
    required_cols = [y] + all_x
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    data = data[required_cols].dropna()
    n = len(data)
    if n < len(all_x) + 2:
        raise ValueError(
            f"Not enough observations ({n}) for {len(all_x)} variables."
        )

    y_vec = data[y].values.astype(float)
    X_base_raw = data[base_x].values.astype(float)
    X_full_raw = data[all_x].values.astype(float)

    X_base = _add_constant(X_base_raw)
    X_full = _add_constant(X_full_raw)

    # ── Base and full regressions ────────────────────────────────────
    beta_base, vcov_base, _ = _ols(y_vec, X_base)
    beta_full, vcov_full, _ = _ols(y_vec, X_full)

    # Index of var_of_interest in base spec (offset by 1 for constant)
    voi_idx_base = base_x.index(var_of_interest) + 1
    voi_idx_full = all_x.index(var_of_interest) + 1

    base_coef = beta_base[voi_idx_base]
    full_coef = beta_full[voi_idx_full]
    total_change = base_coef - full_coef

    # ── Auxiliary regressions: regress each added_x on base_x ────────
    # gamma_tilde[j] = coef on var_of_interest from regressing added_x_j on base_x
    base_var_names = ['_cons'] + base_x
    full_var_names = ['_cons'] + all_x

    decomp_rows = []
    delta_vec = np.zeros(len(added_x))
    delta_var = np.zeros(len(added_x))

    for j, av in enumerate(added_x):
        z_j = data[av].values.astype(float)
        gamma_j, vcov_gamma_j, _ = _ols(z_j, X_base)
        gamma_tilde_j = gamma_j[voi_idx_base]  # coef on var_of_interest

        # Full-model coefficient on this added variable
        av_idx_full = all_x.index(av) + 1  # offset for constant
        beta_full_j = beta_full[av_idx_full]

        delta_j = gamma_tilde_j * beta_full_j
        delta_vec[j] = delta_j

        # Standard error via delta method:
        # Var(delta_j) = gamma^2 * Var(beta_full_j) + beta_full_j^2 * Var(gamma_j)
        #              + Var(gamma_j) * Var(beta_full_j)  [conservative]
        var_beta_full_j = vcov_full[av_idx_full, av_idx_full]
        var_gamma_j = vcov_gamma_j[voi_idx_base, voi_idx_base]

        var_delta_j = (
            gamma_tilde_j**2 * var_beta_full_j
            + beta_full_j**2 * var_gamma_j
        )
        se_delta_j = np.sqrt(max(var_delta_j, 0.0))
        delta_var[j] = var_delta_j

        z_stat = delta_j / se_delta_j if se_delta_j > 0 else 0.0
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        pct = (delta_j / total_change * 100) if total_change != 0 else float('nan')

        decomp_rows.append({
            'variable': av,
            'delta': delta_j,
            'se': se_delta_j,
            'zvalue': z_stat,
            'pvalue': p_val,
            'pct_of_change': pct,
            'gamma_tilde': gamma_tilde_j,
            'beta_full': beta_full_j,
        })

    decomposition = pd.DataFrame(decomp_rows)

    # Sanity check: sum of deltas should ≈ total_change
    sum_delta = delta_vec.sum()
    if abs(sum_delta - total_change) > 1e-6 * (abs(total_change) + 1e-8):
        warnings.warn(
            f"Sum of Gelbach deltas ({sum_delta:.6f}) does not match "
            f"total coefficient change ({total_change:.6f}). "
            f"Difference: {abs(sum_delta - total_change):.2e}. "
            "This may indicate collinearity or numerical issues.",
            stacklevel=2,
        )

    return GelbachResult(
        total_change=total_change,
        decomposition=decomposition,
        base_coef=base_coef,
        full_coef=full_coef,
        base_var=var_of_interest,
    )
