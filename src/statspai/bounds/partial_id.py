"""
Advanced Partial Identification and Treatment Effect Bounds.

Methods
-------
- **Horowitz-Manski Bounds** : Tighter bounds conditioning on covariates (2000)
- **IV Bounds** : Bounds under imperfect instruments (Nevo & Rosen 2012)
- **Oster Delta** : Coefficient stability / identified set (Oster 2019)
- **Selection Bounds** : Lee bounds with covariates (Lee 2009, conditional)
- **Breakdown Frontier** : Assumption robustness frontier (Masten & Poirier 2021)

References
----------
Horowitz, J. L. & Manski, C. F. (2000).
"Nonparametric Analysis of Randomized Experiments with Missing
Covariate and Outcome Data." JASA, 95(449), 77-84.

Nevo, A. & Rosen, A. M. (2012).
"Identification with Imperfect Instruments." RES, 79(3), 1104-1127.

Oster, E. (2019).
"Unobservable Selection and Coefficient Stability: Theory and Evidence."
Journal of Business & Economic Statistics, 37(2), 187-204.

Lee, D. S. (2009).
"Training, Wages, and Sample Selection." RES, 76(3), 1071-1102.

Masten, M. A. & Poirier, A. (2021).
"Salvaging Falsified Instrumental Variable Models."
Econometrica, 89(3), 1449-1469.
"""

from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from dataclasses import dataclass, field

from ..core.results import CausalResult


# ======================================================================
# BoundsResult — Shared result class for all bounding methods
# ======================================================================

@dataclass
class BoundsResult:
    """Result container for partial identification / bounds estimation.

    Attributes
    ----------
    lower : float
        Lower bound of the identified set.
    upper : float
        Upper bound of the identified set.
    se_lower : float
        Standard error of the lower bound.
    se_upper : float
        Standard error of the upper bound.
    ci_lower : tuple
        Confidence interval for the lower bound (lower_lo, lower_hi).
    ci_upper : tuple
        Confidence interval for the upper bound (upper_lo, upper_hi).
    method : str
        Name of the bounding method.
    alpha : float
        Significance level used for confidence intervals.
    n_obs : int
        Number of observations used.
    model_info : dict
        Additional method-specific information.
    """

    lower: float
    upper: float
    se_lower: float
    se_upper: float
    ci_lower: Tuple[float, float]
    ci_upper: Tuple[float, float]
    method: str
    alpha: float = 0.05
    n_obs: int = 0
    model_info: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> float:
        """Width of the identified set."""
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        """Midpoint of the identified set."""
        return (self.lower + self.upper) / 2.0

    def includes_zero(self) -> bool:
        """Check whether the identified set includes zero."""
        return self.lower <= 0 <= self.upper

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary string."""
        z = sp_stats.norm.ppf(1 - self.alpha / 2)
        pct = int((1 - self.alpha) * 100)

        # Imbens-Manski CI for the identified set
        im_lo = self.lower - z * self.se_lower
        im_hi = self.upper + z * self.se_upper

        lines = [
            "",
            "\u2501" * 56,
            f"  {self.method}",
            "\u2501" * 56,
            f"  Lower bound:  {self.lower:>10.4f}   (SE: {self.se_lower:.4f})",
            f"  Upper bound:  {self.upper:>10.4f}   (SE: {self.se_upper:.4f})",
            f"  {pct}% CI:       [{im_lo:.4f}, {im_hi:.4f}]",
            "",
        ]

        if self.includes_zero():
            lines.append("  \u2192 Bounds include zero: cannot reject null")
        else:
            direction = "positive" if self.lower > 0 else "negative"
            lines.append(f"  \u2192 Bounds exclude zero: treatment effect is {direction}")

        # Method-specific extras
        if 'selection_rate_treated' in self.model_info:
            lines.append(
                f"  \u2192 Selection rate: "
                f"{self.model_info['selection_rate_treated']:.1%} (treated), "
                f"{self.model_info['selection_rate_control']:.1%} (control)"
            )
        if 'delta_star' in self.model_info:
            lines.append(f"  \u2192 \u03b4* for \u03b2=0: {self.model_info['delta_star']:.4f}")
        if 'r_max' in self.model_info:
            lines.append(f"  \u2192 R\u00b2_max assumption: {self.model_info['r_max']:.4f}")

        lines.append("\u2501" * 56)
        lines.append("")

        text = "\n".join(lines)
        print(text)
        return text

    def _repr_html_(self) -> str:
        """Rich HTML for Jupyter notebooks."""
        z = sp_stats.norm.ppf(1 - self.alpha / 2)
        pct = int((1 - self.alpha) * 100)
        im_lo = self.lower - z * self.se_lower
        im_hi = self.upper + z * self.se_upper

        if self.includes_zero():
            verdict = '<span style="color:#e67e22">Bounds include zero</span>'
        else:
            direction = "positive" if self.lower > 0 else "negative"
            verdict = f'<span style="color:#27ae60">Bounds exclude zero ({direction})</span>'

        return (
            f'<div style="font-family:monospace; padding:12px; '
            f'border:1px solid #ddd; border-radius:6px; max-width:520px">'
            f'<h4 style="margin:0 0 8px 0">{self.method}</h4>'
            f'<table style="border-collapse:collapse">'
            f'<tr><td style="padding:2px 12px 2px 0">Lower bound</td>'
            f'<td><b>{self.lower:.4f}</b> (SE {self.se_lower:.4f})</td></tr>'
            f'<tr><td style="padding:2px 12px 2px 0">Upper bound</td>'
            f'<td><b>{self.upper:.4f}</b> (SE {self.se_upper:.4f})</td></tr>'
            f'<tr><td style="padding:2px 12px 2px 0">{pct}% CI</td>'
            f'<td>[{im_lo:.4f}, {im_hi:.4f}]</td></tr>'
            f'</table>'
            f'<p style="margin:8px 0 0 0">{verdict}</p>'
            f'</div>'
        )

    def plot(self, ax=None, **kwargs):
        """Interval plot showing the identified set with confidence intervals.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to draw on; created if *None*.
        **kwargs
            Passed to ``ax.errorbar``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))

        z = sp_stats.norm.ppf(1 - self.alpha / 2)

        # Identified set as thick bar
        ax.barh(
            0, self.upper - self.lower, left=self.lower, height=0.3,
            color='#3498db', alpha=0.6, label='Identified set',
        )

        # CI whiskers
        ci_lo = self.lower - z * self.se_lower
        ci_hi = self.upper + z * self.se_upper
        ax.plot([ci_lo, ci_hi], [0, 0], 'k-', linewidth=1)
        ax.plot([ci_lo, ci_lo], [-0.1, 0.1], 'k-', linewidth=1)
        ax.plot([ci_hi, ci_hi], [-0.1, 0.1], 'k-', linewidth=1)

        # Zero line
        ax.axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)

        ax.set_yticks([])
        ax.set_xlabel('Treatment Effect')
        ax.set_title(self.method)
        ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        return ax

    def __repr__(self) -> str:
        return (
            f"BoundsResult(method='{self.method}', "
            f"lower={self.lower:.4f}, upper={self.upper:.4f})"
        )


# ======================================================================
# Helper: bootstrap utilities
# ======================================================================

def _bootstrap_bounds(
    compute_fn, data: pd.DataFrame, n_boot: int, alpha: float,
    random_state: int, **kwargs,
) -> Tuple[float, float, float, float, Tuple, Tuple]:
    """Generic bootstrap for a function that returns (lower, upper).

    Returns
    -------
    se_lower, se_upper, ci_lower_tuple, ci_upper_tuple
    """
    rng = np.random.RandomState(random_state)
    n = len(data)
    boot_lb = np.zeros(n_boot)
    boot_ub = np.zeros(n_boot)

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        df_b = data.iloc[idx].reset_index(drop=True)
        try:
            boot_lb[b], boot_ub[b] = compute_fn(df_b, **kwargs)
        except Exception:
            boot_lb[b], boot_ub[b] = np.nan, np.nan

    boot_lb = boot_lb[~np.isnan(boot_lb)]
    boot_ub = boot_ub[~np.isnan(boot_ub)]

    se_lb = float(np.std(boot_lb, ddof=1)) if len(boot_lb) > 1 else 0.0
    se_ub = float(np.std(boot_ub, ddof=1)) if len(boot_ub) > 1 else 0.0

    z = sp_stats.norm.ppf(1 - alpha / 2)
    lb_point = float(np.median(boot_lb)) if len(boot_lb) > 0 else 0.0
    ub_point = float(np.median(boot_ub)) if len(boot_ub) > 0 else 0.0

    ci_lb = (lb_point - z * se_lb, lb_point + z * se_lb)
    ci_ub = (ub_point - z * se_ub, ub_point + z * se_ub)

    return se_lb, se_ub, ci_lb, ci_ub


# ======================================================================
# 1. Horowitz-Manski Bounds with Covariates
# ======================================================================

def horowitz_manski(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    covariates: List[str],
    y_lower: Optional[float] = None,
    y_upper: Optional[float] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> BoundsResult:
    """
    Horowitz-Manski (2000) bounds conditioning on covariates.

    Tighter than unconditional Manski bounds by averaging conditional
    bounds over the covariate distribution.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treatment : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariates to condition on (discretised via quartiles for
        continuous variables).
    y_lower : float, optional
        Known lower bound of Y. Defaults to observed min.
    y_upper : float, optional
        Known upper bound of Y. Defaults to observed max.
    n_boot : int, default 500
        Bootstrap replications.
    alpha : float, default 0.05
    random_state : int, default 42

    Returns
    -------
    BoundsResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.horowitz_manski(
    ...     data=df, y="wage", treatment="trained",
    ...     covariates=["age", "education"],
    ...     y_lower=0, y_upper=100,
    ... )
    >>> result.summary()
    """
    _check_cols(data, [y, treatment] + covariates)
    df = data.dropna(subset=[y, treatment] + covariates).copy()

    Y = df[y].values.astype(np.float64)
    D = df[treatment].values.astype(np.float64)

    if y_lower is None:
        y_lower = float(Y.min())
    if y_upper is None:
        y_upper = float(Y.max())

    # Discretise continuous covariates into quartile bins
    strata_col = _create_strata(df, covariates)

    lb, ub = _hm_point(df, y, treatment, strata_col, y_lower, y_upper)

    # Bootstrap
    def _compute(df_b, **kw):
        sc = _create_strata(df_b, covariates)
        return _hm_point(df_b, y, treatment, sc, y_lower, y_upper)

    se_lb, se_ub, ci_lb, ci_ub = _bootstrap_bounds(
        _compute, df, n_boot, alpha, random_state,
    )

    return BoundsResult(
        lower=lb,
        upper=ub,
        se_lower=se_lb,
        se_upper=se_ub,
        ci_lower=ci_lb,
        ci_upper=ci_ub,
        method='Horowitz-Manski Bounds (2000)',
        alpha=alpha,
        n_obs=len(df),
        model_info={
            'y_lower': y_lower,
            'y_upper': y_upper,
            'n_strata': int(strata_col.nunique()),
            'covariates': covariates,
        },
    )


def _hm_point(df, y, treatment, strata, y_lo, y_hi):
    """Compute Horowitz-Manski conditional bounds."""
    Y = df[y].values.astype(np.float64)
    D = df[treatment].values.astype(np.float64)
    S = strata.values
    n = len(Y)

    lb_total = 0.0
    ub_total = 0.0

    for s in np.unique(S):
        mask = S == s
        w = mask.sum() / n  # P(X = s)

        Y_s = Y[mask]
        D_s = D[mask]

        Y1 = Y_s[D_s == 1]
        Y0 = Y_s[D_s == 0]

        if len(Y1) == 0 or len(Y0) == 0:
            continue

        p1 = np.mean(D_s)   # P(D=1 | X=s)
        p0 = 1 - p1         # P(D=0 | X=s)

        e1 = np.mean(Y1)
        e0 = np.mean(Y0)

        # Conditional Manski bounds for ATE at stratum s
        lb_s = e1 * p1 + y_lo * p0 - e0 * p0 - y_hi * p1
        ub_s = e1 * p1 + y_hi * p0 - e0 * p0 - y_lo * p1

        lb_total += w * lb_s
        ub_total += w * ub_s

    return float(lb_total), float(ub_total)


def _create_strata(df: pd.DataFrame, covariates: List[str]) -> pd.Series:
    """Create strata by discretising continuous covariates into quartiles."""
    parts = []
    for c in covariates:
        col = df[c]
        if col.dtype.kind in ('f',) or col.nunique() > 10:
            parts.append(pd.qcut(col, q=4, labels=False, duplicates='drop').astype(str))
        else:
            parts.append(col.astype(str))
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + '_' + p
    return combined


# ======================================================================
# 2. IV Bounds (Nevo & Rosen 2012)
# ======================================================================

def iv_bounds(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    instrument: str,
    controls: Optional[List[str]] = None,
    assumption: str = 'monotone_iv',
    alpha: float = 0.05,
    n_boot: int = 500,
    random_state: int = 42,
) -> BoundsResult:
    """
    Nevo-Rosen (2012) bounds for LATE under imperfect instruments.

    When the exclusion restriction may be violated, the LATE is no longer
    point-identified.  Under weaker monotonicity assumptions on the
    direction of the violation, informative bounds can still be obtained.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treatment : str
        Endogenous treatment (binary).
    instrument : str
        Instrument variable (binary).
    controls : list of str, optional
        Control variables (residualized out via OLS).
    assumption : str, default 'monotone_iv'
        - ``'monotone_iv'``: instrument has same-sign direct effect as
          through the treatment (Nevo-Rosen Proposition 2).
        - ``'less_than_late'``: direct effect of Z on Y is weakly less
          than the indirect effect (tighter).
    alpha : float, default 0.05
    n_boot : int, default 500
    random_state : int, default 42

    Returns
    -------
    BoundsResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.iv_bounds(
    ...     data=df, y="wage", treatment="trained",
    ...     instrument="lottery", assumption="monotone_iv",
    ... )
    >>> result.summary()
    """
    cols = [y, treatment, instrument]
    if controls:
        cols += controls
    _check_cols(data, cols)

    df = data.dropna(subset=cols).copy()

    # Residualize if controls are provided
    Y = df[y].values.astype(np.float64)
    D = df[treatment].values.astype(np.float64)
    Z = df[instrument].values.astype(np.float64)

    if controls and len(controls) > 0:
        X = df[controls].values.astype(np.float64)
        X_aug = np.column_stack([np.ones(len(X)), X])
        # Residualize Y, D, Z
        for arr_name in ['Y', 'D', 'Z']:
            arr = locals()[arr_name]
            beta = np.linalg.lstsq(X_aug, arr, rcond=None)[0]
            resid = arr - X_aug @ beta
            if arr_name == 'Y':
                Y = resid
            elif arr_name == 'D':
                D = resid + np.mean(D)  # Keep mean for proportions
            else:
                Z = resid + np.mean(Z)

    lb, ub = _iv_bounds_point(Y, D, Z, assumption)

    # Bootstrap
    def _compute(df_b, **kw):
        Yb = df_b[y].values.astype(np.float64)
        Db = df_b[treatment].values.astype(np.float64)
        Zb = df_b[instrument].values.astype(np.float64)
        if controls and len(controls) > 0:
            Xb = df_b[controls].values.astype(np.float64)
            Xb_aug = np.column_stack([np.ones(len(Xb)), Xb])
            beta_y = np.linalg.lstsq(Xb_aug, Yb, rcond=None)[0]
            Yb = Yb - Xb_aug @ beta_y
            beta_d = np.linalg.lstsq(Xb_aug, Db, rcond=None)[0]
            Db = Db - Xb_aug @ beta_d + np.mean(Db)
            beta_z = np.linalg.lstsq(Xb_aug, Zb, rcond=None)[0]
            Zb = Zb - Xb_aug @ beta_z + np.mean(Zb)
        return _iv_bounds_point(Yb, Db, Zb, assumption)

    se_lb, se_ub, ci_lb, ci_ub = _bootstrap_bounds(
        _compute, df, n_boot, alpha, random_state,
    )

    # Wald estimate for reference
    cov_zy = np.cov(Z, Y)[0, 1]
    cov_zd = np.cov(Z, D)[0, 1]
    wald = cov_zy / cov_zd if abs(cov_zd) > 1e-12 else np.nan

    return BoundsResult(
        lower=lb,
        upper=ub,
        se_lower=se_lb,
        se_upper=se_ub,
        ci_lower=ci_lb,
        ci_upper=ci_ub,
        method=f'Nevo-Rosen IV Bounds ({assumption})',
        alpha=alpha,
        n_obs=len(df),
        model_info={
            'assumption': assumption,
            'wald_estimate': float(wald) if np.isfinite(wald) else None,
            'cov_zy': float(cov_zy),
            'cov_zd': float(cov_zd),
            'first_stage_f': float(cov_zd ** 2 / (np.var(Z) * np.var(D) / len(Z)))
            if np.var(D) > 0 else 0.0,
        },
    )


def _iv_bounds_point(Y, D, Z, assumption):
    """Compute Nevo-Rosen bounds for given arrays."""
    n = len(Y)

    cov_zy = np.cov(Z, Y, ddof=1)[0, 1]
    cov_zd = np.cov(Z, D, ddof=1)[0, 1]
    var_z = np.var(Z, ddof=1)
    var_d = np.var(D, ddof=1)

    # Standard Wald/IV estimate
    wald = cov_zy / cov_zd if abs(cov_zd) > 1e-12 else 0.0

    # OLS estimate for reference
    cov_dy = np.cov(D, Y, ddof=1)[0, 1]
    ols = cov_dy / var_d if var_d > 1e-12 else 0.0

    if assumption == 'monotone_iv':
        # Nevo-Rosen Proposition 2: if instrument has same-sign direct
        # effect on Y as through D, LATE is bounded between OLS and Wald.
        # Under positive correlation: [min(OLS, Wald), max(OLS, Wald)]
        lb = min(ols, wald)
        ub = max(ols, wald)

    elif assumption == 'less_than_late':
        # Tighter: the direct effect of Z on Y is less than the indirect effect.
        # Bounds: [Wald - |OLS - Wald|, Wald + |OLS - Wald|]
        # but the upper bound is capped at OLS direction.
        diff = abs(ols - wald)
        lb = wald - diff
        ub = wald + diff
        # Ensure sensible ordering
        if lb > ub:
            lb, ub = ub, lb
    else:
        raise ValueError(
            f"Unknown assumption '{assumption}'. "
            f"Use 'monotone_iv' or 'less_than_late'."
        )

    return float(lb), float(ub)


# ======================================================================
# 3. Oster Delta (2019) — Coefficient stability / identified set
# ======================================================================

def oster_delta(
    data: pd.DataFrame,
    y: str,
    x_base: List[str],
    x_controls: List[str],
    r_max: float = 1.3,
    delta_range: Tuple[float, float] = (-2.0, 2.0),
    n_grid: int = 200,
    alpha: float = 0.05,
    n_boot: int = 500,
    random_state: int = 42,
) -> BoundsResult:
    """
    Oster (2019) coefficient stability bounds and delta* computation.

    Computes the identified set for the treatment coefficient beta under
    the assumption that selection on unobservables is proportional to
    selection on observables (delta), and R-squared would be at most
    ``r_max`` if all unobservables were included.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    x_base : list of str
        Key treatment/variable(s) of interest.
    x_controls : list of str
        Additional controls whose inclusion tightens identification.
    r_max : float, default 1.3
        Maximum R-squared assumption. Oster recommends 1.3 * R-squared
        from the fully controlled regression. If <= 0, it is set to
        1.3 * R_full automatically.
    delta_range : tuple, default (-2, 2)
        Range of proportional selection parameter delta.
    n_grid : int, default 200
        Grid points for delta in the identified set computation.
    alpha : float, default 0.05
    n_boot : int, default 500
    random_state : int, default 42

    Returns
    -------
    BoundsResult
        lower/upper give the identified set for beta at delta=1 (equal
        selection) and the given r_max.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.oster_delta(
    ...     data=df, y="wage",
    ...     x_base=["education"],
    ...     x_controls=["experience", "tenure"],
    ...     r_max=1.3,
    ... )
    >>> result.summary()
    >>> result.plot()
    """
    all_cols = [y] + x_base + x_controls
    _check_cols(data, all_cols)
    df = data.dropna(subset=all_cols).copy()
    n = len(df)

    Y = df[y].values.astype(np.float64)

    # Short regression: y ~ x_base
    X_short = np.column_stack([np.ones(n), df[x_base].values.astype(np.float64)])
    beta_short_all = np.linalg.lstsq(X_short, Y, rcond=None)[0]
    beta_short = beta_short_all[1]  # coefficient on first x_base variable
    resid_short = Y - X_short @ beta_short_all
    r2_short = 1 - np.var(resid_short) / np.var(Y)

    # Full regression: y ~ x_base + x_controls
    X_full = np.column_stack([
        np.ones(n),
        df[x_base].values.astype(np.float64),
        df[x_controls].values.astype(np.float64),
    ])
    beta_full_all = np.linalg.lstsq(X_full, Y, rcond=None)[0]
    beta_full = beta_full_all[1]  # coefficient on first x_base variable
    resid_full = Y - X_full @ beta_full_all
    r2_full = 1 - np.var(resid_full) / np.var(Y)

    # Set r_max
    if r_max <= 0:
        r_max = min(1.3 * r2_full, 1.0)
    elif r_max <= r2_full:
        r_max = min(1.3 * r2_full, 1.0)

    # Oster's bias-adjusted estimate at delta=1
    beta_star = _oster_bias_adjusted(beta_short, beta_full, r2_short, r2_full, r_max, delta=1.0)

    # Identified set: [min(beta_full, beta_star), max(beta_full, beta_star)]
    lb = min(beta_full, beta_star)
    ub = max(beta_full, beta_star)

    # Compute delta* (the delta that makes beta_star = 0)
    delta_star = _oster_delta_star(beta_short, beta_full, r2_short, r2_full, r_max)

    # Grid of (delta, beta_star) for plotting
    deltas = np.linspace(delta_range[0], delta_range[1], n_grid)
    betas_grid = np.array([
        _oster_bias_adjusted(beta_short, beta_full, r2_short, r2_full, r_max, d)
        for d in deltas
    ])

    # Bootstrap
    def _compute(df_b, **kw):
        nb = len(df_b)
        Yb = df_b[y].values.astype(np.float64)
        Xs = np.column_stack([np.ones(nb), df_b[x_base].values.astype(np.float64)])
        bs_all = np.linalg.lstsq(Xs, Yb, rcond=None)[0]
        rs = Y_var_ratio(Yb, Xs, bs_all)

        Xf = np.column_stack([
            np.ones(nb),
            df_b[x_base].values.astype(np.float64),
            df_b[x_controls].values.astype(np.float64),
        ])
        bf_all = np.linalg.lstsq(Xf, Yb, rcond=None)[0]
        rf = Y_var_ratio(Yb, Xf, bf_all)

        rm = min(1.3 * rf, 1.0) if r_max <= 0 or r_max <= rf else r_max
        bs_star = _oster_bias_adjusted(bs_all[1], bf_all[1], rs, rf, rm, 1.0)
        return float(min(bf_all[1], bs_star)), float(max(bf_all[1], bs_star))

    def Y_var_ratio(Yv, Xv, bv):
        resid = Yv - Xv @ bv
        return 1 - np.var(resid) / np.var(Yv) if np.var(Yv) > 0 else 0.0

    se_lb, se_ub, ci_lb, ci_ub = _bootstrap_bounds(
        _compute, df, n_boot, alpha, random_state,
    )

    return BoundsResult(
        lower=float(lb),
        upper=float(ub),
        se_lower=se_lb,
        se_upper=se_ub,
        ci_lower=ci_lb,
        ci_upper=ci_ub,
        method='Oster (2019) Coefficient Stability Bounds',
        alpha=alpha,
        n_obs=n,
        model_info={
            'beta_short': float(beta_short),
            'beta_full': float(beta_full),
            'r2_short': float(r2_short),
            'r2_full': float(r2_full),
            'r_max': float(r_max),
            'beta_star_delta1': float(beta_star),
            'delta_star': float(delta_star) if np.isfinite(delta_star) else None,
            'delta_grid': deltas.tolist(),
            'beta_grid': betas_grid.tolist(),
        },
    )


def _oster_bias_adjusted(beta_short, beta_full, r2_short, r2_full, r_max, delta):
    """Oster's bias-adjusted beta* at a given delta and R_max.

    Formula (Oster 2019, eq. 3):
        beta* = beta_full - delta * (beta_short - beta_full) *
                (r_max - r2_full) / (r2_full - r2_short)
    """
    denom = r2_full - r2_short
    if abs(denom) < 1e-12:
        return beta_full
    bias = delta * (beta_short - beta_full) * (r_max - r2_full) / denom
    return beta_full - bias


def _oster_delta_star(beta_short, beta_full, r2_short, r2_full, r_max):
    """Compute delta* such that beta*(delta*, r_max) = 0.

    From beta* = 0:
        delta* = beta_full * (r2_full - r2_short) /
                 ((beta_short - beta_full) * (r_max - r2_full))
    """
    numer = beta_full * (r2_full - r2_short)
    denom = (beta_short - beta_full) * (r_max - r2_full)
    if abs(denom) < 1e-12:
        return np.inf
    return numer / denom


# ======================================================================
# 4. Selection Bounds — Lee (2009) with covariates
# ======================================================================

def selection_bounds(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    selection: str,
    covariates: Optional[List[str]] = None,
    method: str = 'conditional',
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> BoundsResult:
    """
    Lee (2009) bounds for ATE under sample selection, optionally
    conditioning on covariates for tighter bounds.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable (may have NaN when selection=0).
    treatment : str
        Binary treatment (0/1).
    selection : str
        Binary indicator: 1 = outcome observed, 0 = missing.
    covariates : list of str, optional
        Covariates to condition on for tighter (conditional) bounds.
    method : str, default 'conditional'
        - ``'conditional'``: compute Lee bounds within covariate strata
          and average (tighter).
        - ``'unconditional'``: standard Lee bounds ignoring covariates.
    n_boot : int, default 500
    alpha : float, default 0.05
    random_state : int, default 42

    Returns
    -------
    BoundsResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.selection_bounds(
    ...     data=df, y="wage", treatment="trained",
    ...     selection="employed",
    ...     covariates=["age", "education"],
    ...     method="conditional",
    ... )
    >>> result.summary()
    """
    cols = [y, treatment, selection]
    if covariates:
        cols += covariates
    _check_cols(data, cols)

    df = data.copy()
    D = df[treatment].values.astype(float)
    S = df[selection].values.astype(float)

    p1 = np.mean(S[D == 1])
    p0 = np.mean(S[D == 0])

    if method == 'unconditional' or covariates is None or len(covariates) == 0:
        lb, ub = _lee_bounds_core(df, y, treatment, selection)
    else:
        lb, ub = _conditional_lee_bounds(df, y, treatment, selection, covariates)

    # Bootstrap
    def _compute(df_b, **kw):
        if method == 'unconditional' or covariates is None or len(covariates) == 0:
            return _lee_bounds_core(df_b, y, treatment, selection)
        else:
            return _conditional_lee_bounds(df_b, y, treatment, selection, covariates)

    se_lb, se_ub, ci_lb, ci_ub = _bootstrap_bounds(
        _compute, df, n_boot, alpha, random_state,
    )

    return BoundsResult(
        lower=lb,
        upper=ub,
        se_lower=se_lb,
        se_upper=se_ub,
        ci_lower=ci_lb,
        ci_upper=ci_ub,
        method=f'Lee (2009) Bounds ({method})',
        alpha=alpha,
        n_obs=int(np.sum(S)),
        model_info={
            'method': method,
            'selection_rate_treated': float(p1),
            'selection_rate_control': float(p0),
            'trimming_fraction': float(abs(p1 - p0) / max(p1, p0)) if max(p1, p0) > 0 else 0.0,
            'covariates': covariates or [],
        },
    )


def _lee_bounds_core(df, y, treatment, selection):
    """Core unconditional Lee bounds computation."""
    D = df[treatment].values.astype(float)
    S = df[selection].values.astype(float)

    p1 = np.mean(S[D == 1])
    p0 = np.mean(S[D == 0])

    if p1 == 0 or p0 == 0:
        return (np.nan, np.nan)

    obs = S == 1
    Y_obs = df.loc[obs, y].values.astype(float)
    D_obs = D[obs]

    Y1 = Y_obs[D_obs == 1]
    Y0 = Y_obs[D_obs == 0]

    if len(Y1) == 0 or len(Y0) == 0:
        return (np.nan, np.nan)

    mean_y0 = np.mean(Y0)

    if p1 > p0:
        q = p0 / p1
        n1 = len(Y1)
        k = max(int(np.floor(q * n1)), 1)
        Y1s = np.sort(Y1)
        lb = np.mean(Y1s[:k]) - mean_y0
        ub = np.mean(Y1s[n1 - k:]) - mean_y0
    elif p0 > p1:
        q = p1 / p0
        n0 = len(Y0)
        k = max(int(np.floor(q * n0)), 1)
        Y0s = np.sort(Y0)
        mean_y1 = np.mean(Y1)
        lb = mean_y1 - np.mean(Y0s[n0 - k:])
        ub = mean_y1 - np.mean(Y0s[:k])
    else:
        lb = np.mean(Y1) - mean_y0
        ub = lb

    return float(lb), float(ub)


def _conditional_lee_bounds(df, y, treatment, selection, covariates):
    """Conditional Lee bounds — average stratum-specific bounds."""
    strata = _create_strata(df, covariates)

    D = df[treatment].values.astype(float)
    S = df[selection].values.astype(float)
    strata_arr = strata.values
    n = len(df)

    lb_total = 0.0
    ub_total = 0.0
    w_total = 0.0

    for s in np.unique(strata_arr):
        mask = strata_arr == s
        w = mask.sum() / n

        df_s = df.loc[mask].copy()
        if df_s[treatment].nunique() < 2:
            continue

        D_s = df_s[treatment].values.astype(float)
        S_s = df_s[selection].values.astype(float)

        if np.sum(D_s == 1) == 0 or np.sum(D_s == 0) == 0:
            continue
        if np.mean(S_s[D_s == 1]) == 0 or np.mean(S_s[D_s == 0]) == 0:
            continue

        lb_s, ub_s = _lee_bounds_core(df_s, y, treatment, selection)
        if np.isnan(lb_s) or np.isnan(ub_s):
            continue

        lb_total += w * lb_s
        ub_total += w * ub_s
        w_total += w

    if w_total > 0:
        lb_total /= w_total
        ub_total /= w_total

    return float(lb_total), float(ub_total)


# ======================================================================
# 5. Breakdown Frontier (Masten & Poirier 2021)
# ======================================================================

def breakdown_frontier(
    estimate: float,
    se: float,
    assumption: str = 'parallel_trends',
    max_violation: float = 0.1,
    n_grid: int = 100,
    alpha: float = 0.05,
) -> BoundsResult:
    """
    Masten-Poirier (2021) breakdown frontier for qualitative conclusions.

    For a given point estimate and standard error, computes how much an
    identifying assumption can be violated before a qualitative conclusion
    (e.g., "positive treatment effect") breaks down.

    Parameters
    ----------
    estimate : float
        Point estimate of the treatment effect.
    se : float
        Standard error of the estimate.
    assumption : str, default 'parallel_trends'
        Label for the identifying assumption being relaxed.  Currently
        supports a generic linear violation model applicable to
        ``'parallel_trends'``, ``'exclusion_restriction'``, or
        ``'selection_on_observables'``.
    max_violation : float, default 0.1
        Maximum magnitude of the assumption violation to explore.
    n_grid : int, default 100
        Grid resolution for the frontier.
    alpha : float, default 0.05

    Returns
    -------
    BoundsResult
        lower/upper give the identified set at the maximum violation.
        ``model_info`` contains ``'breakdown_point'`` (the violation
        magnitude at which the conclusion reverses) and
        ``'frontier_grid'`` / ``'frontier_bounds'`` for plotting.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.breakdown_frontier(
    ...     estimate=0.05, se=0.02,
    ...     assumption="parallel_trends",
    ...     max_violation=0.1,
    ... )
    >>> result.summary()
    >>> result.plot()
    """
    if se <= 0:
        raise ValueError("Standard error must be positive.")

    # Breakdown point: the violation c such that sign could flip
    # Under linear violation: identified set = [estimate - c, estimate + c]
    # Conclusion reverses when estimate - c <= 0 (if estimate > 0)
    # => breakdown point c* = |estimate|
    breakdown_point = abs(estimate)

    # Grid of violations
    violations = np.linspace(0, max_violation, n_grid)
    lower_bounds = estimate - violations
    upper_bounds = estimate + violations

    # At max_violation
    lb = float(estimate - max_violation)
    ub = float(estimate + max_violation)

    # SE of the bounds = se (violation is deterministic, not estimated)
    z = sp_stats.norm.ppf(1 - alpha / 2)

    # Breakdown point with CI: includes statistical uncertainty
    # Conclusion breaks down when lower CI crosses zero
    breakdown_ci = max(abs(estimate) - z * se, 0)

    return BoundsResult(
        lower=lb,
        upper=ub,
        se_lower=se,
        se_upper=se,
        ci_lower=(lb - z * se, lb + z * se),
        ci_upper=(ub - z * se, ub + z * se),
        method=f'Breakdown Frontier ({assumption})',
        alpha=alpha,
        n_obs=0,
        model_info={
            'estimate': estimate,
            'se': se,
            'assumption': assumption,
            'breakdown_point': float(breakdown_point),
            'breakdown_point_ci': float(breakdown_ci),
            'max_violation': max_violation,
            'frontier_grid': violations.tolist(),
            'frontier_lower': lower_bounds.tolist(),
            'frontier_upper': upper_bounds.tolist(),
            'conclusion_at_zero_violation': 'positive' if estimate > 0 else 'negative',
            'robust_at_max_violation': bool(lb > 0) if estimate > 0 else bool(ub < 0),
        },
    )


# ======================================================================
# Utilities
# ======================================================================

def _check_cols(data: pd.DataFrame, cols: List[str]):
    """Raise if any columns are missing from the DataFrame."""
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")


# ======================================================================
# Citations
# ======================================================================

CausalResult._CITATIONS['horowitz_manski'] = (
    "@article{horowitz2000nonparametric,\n"
    "  title={Nonparametric Analysis of Randomized Experiments with "
    "Missing Covariate and Outcome Data},\n"
    "  author={Horowitz, Joel L and Manski, Charles F},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={95},\n"
    "  number={449},\n"
    "  pages={77--84},\n"
    "  year={2000}\n"
    "}"
)

CausalResult._CITATIONS['nevo_rosen'] = (
    "@article{nevo2012identification,\n"
    "  title={Identification with Imperfect Instruments},\n"
    "  author={Nevo, Aviv and Rosen, Adam M},\n"
    "  journal={The Review of Economic Studies},\n"
    "  volume={79},\n"
    "  number={3},\n"
    "  pages={1104--1127},\n"
    "  year={2012}\n"
    "}"
)

CausalResult._CITATIONS['oster_delta'] = (
    "@article{oster2019unobservable,\n"
    "  title={Unobservable Selection and Coefficient Stability: "
    "Theory and Evidence},\n"
    "  author={Oster, Emily},\n"
    "  journal={Journal of Business \\& Economic Statistics},\n"
    "  volume={37},\n"
    "  number={2},\n"
    "  pages={187--204},\n"
    "  year={2019}\n"
    "}"
)

CausalResult._CITATIONS['selection_bounds'] = (
    "@article{lee2009training,\n"
    "  title={Training, Wages, and Sample Selection: Estimating Sharp "
    "Bounds on Treatment Effects},\n"
    "  author={Lee, David S},\n"
    "  journal={The Review of Economic Studies},\n"
    "  volume={76},\n"
    "  number={3},\n"
    "  pages={1071--1102},\n"
    "  year={2009}\n"
    "}"
)

CausalResult._CITATIONS['breakdown_frontier'] = (
    "@article{masten2021salvaging,\n"
    "  title={Salvaging Falsified Instrumental Variable Models},\n"
    "  author={Masten, Matthew A and Poirier, Alexandre},\n"
    "  journal={Econometrica},\n"
    "  volume={89},\n"
    "  number={3},\n"
    "  pages={1449--1469},\n"
    "  year={2021}\n"
    "}"
)
