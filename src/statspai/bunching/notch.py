"""
Bunching at Notches (Kleven & Waseem 2013).

Estimates behavioural responses to notches — discontinuous jumps in a
tax/benefit schedule (e.g., income tax brackets with discrete jumps,
social program eligibility thresholds).

At a notch there is:
1. Excess bunching just below the notch (people cluster to avoid the jump).
2. A hole (missing mass) in the distribution just above the notch.

The algorithm iteratively finds the marginal buncher x* such that
the excess mass below equals the missing mass above, then converts
this to a structural elasticity estimate.

References
----------
Kleven, H. J. and Waseem, M. (2013).
"Using Notches to Uncover Optimization Frictions and Structural
Elasticities: Theory and Evidence from Pakistan."
QJE, 128(2), 669-723.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


# ======================================================================
# NotchResult
# ======================================================================

class NotchResult:
    """
    Result container for notch bunching analysis.

    Attributes
    ----------
    excess_bunching : float
        Excess mass B in the bunching region (below the notch).
    missing_mass : float
        Missing mass H above the notch (the "hole").
    marginal_buncher : float
        Upper bound of the dominated region (x*).
    elasticity : float or None
        Structural elasticity estimate (if notch_size provided).
    se_bunching : float
        Bootstrap standard error of excess bunching.
    se_elasticity : float or None
        Bootstrap standard error of elasticity (if notch_size provided).
    pvalue : float
        P-value for H0: excess_bunching = 0.
    ci : tuple of (float, float)
        Confidence interval for excess bunching.
    notch_point : float
        Location of the notch.
    bin_centers : np.ndarray
        Bin centres of the histogram.
    observed : np.ndarray
        Observed bin counts.
    counterfactual : np.ndarray
        Counterfactual polynomial counts.
    n_obs : int
        Number of observations.
    causal_result : CausalResult
        Full CausalResult object for interoperability.
    """

    def __init__(
        self,
        excess_bunching: float,
        missing_mass: float,
        marginal_buncher: float,
        elasticity,
        se_bunching: float,
        se_elasticity,
        pvalue: float,
        ci: Tuple[float, float],
        alpha: float,
        notch_point: float,
        bin_centers: np.ndarray,
        observed: np.ndarray,
        counterfactual: np.ndarray,
        n_obs: int,
        causal_result: CausalResult,
        notch_size=None,
    ):
        self.excess_bunching = excess_bunching
        self.missing_mass = missing_mass
        self.marginal_buncher = marginal_buncher
        self.elasticity = elasticity
        self.se_bunching = se_bunching
        self.se_elasticity = se_elasticity
        self.pvalue = pvalue
        self.ci = ci
        self.alpha = alpha
        self.notch_point = notch_point
        self.bin_centers = bin_centers
        self.observed = observed
        self.counterfactual = counterfactual
        self.n_obs = n_obs
        self.causal_result = causal_result
        self.notch_size = notch_size

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Return a formatted summary table."""
        lines = [
            "=" * 65,
            "  Bunching at Notch Estimator (Kleven & Waseem 2013)",
            "=" * 65,
            f"  Notch point           : {self.notch_point:,.2f}",
            f"  Observations          : {self.n_obs:,}",
            f"  Significance level    : {self.alpha}",
            "-" * 65,
            f"  Excess bunching (B)   : {self.excess_bunching:,.2f}"
            f"   (SE = {self.se_bunching:,.2f})",
            f"  Missing mass (H)      : {self.missing_mass:,.2f}",
            f"  Marginal buncher (x*) : {self.marginal_buncher:,.2f}",
        ]

        if self.elasticity is not None:
            lines.append(
                f"  Elasticity            : {self.elasticity:.4f}"
                + (f"   (SE = {self.se_elasticity:.4f})"
                   if self.se_elasticity is not None else "")
            )
            if self.notch_size is not None:
                lines.append(
                    f"  Notch size (dt)       : {self.notch_size}"
                )

        ci_lo, ci_hi = self.ci
        lines += [
            "-" * 65,
            f"  p-value               : {self.pvalue:.4f}",
            f"  {100*(1-self.alpha):.0f}% CI (B)          "
            f"  : [{ci_lo:,.2f}, {ci_hi:,.2f}]",
            "=" * 65,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # plot
    # ------------------------------------------------------------------
    def plot(self, figsize=(10, 6), title=None):
        """
        Plot observed histogram with counterfactual overlay.

        Returns
        -------
        (fig, ax) : matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        fig, ax = plt.subplots(figsize=figsize)

        # Bar plot of observed
        bw = np.median(np.diff(self.bin_centers)) if len(self.bin_centers) > 1 else 1
        ax.bar(self.bin_centers, self.observed, width=bw * 0.9,
               color="#B0C4DE", edgecolor="white", alpha=0.85,
               label="Observed")

        # Counterfactual line
        ax.plot(self.bin_centers, self.counterfactual,
                color="#E74C3C", linewidth=2, label="Counterfactual")

        # Shade excess bunching (below notch, observed > counterfactual)
        below = self.bin_centers <= self.notch_point
        excess_mask = below & (self.observed > self.counterfactual)
        if np.any(excess_mask):
            ax.fill_between(
                self.bin_centers[excess_mask],
                self.counterfactual[excess_mask],
                self.observed[excess_mask],
                alpha=0.4, color="#3498DB", label="Excess bunching",
            )

        # Shade missing mass (above notch, counterfactual > observed)
        above = self.bin_centers > self.notch_point
        hole_mask = above & (self.counterfactual > self.observed)
        if np.any(hole_mask):
            ax.fill_between(
                self.bin_centers[hole_mask],
                self.observed[hole_mask],
                self.counterfactual[hole_mask],
                alpha=0.4, color="#E74C3C", label="Missing mass",
            )

        # Vertical line at notch
        ax.axvline(self.notch_point, color="#2C3E50", linestyle="--",
                   linewidth=1.5, label=f"Notch ({self.notch_point:,.0f})")

        # Marginal buncher marker
        if self.marginal_buncher > self.notch_point:
            ax.axvline(self.marginal_buncher, color="#27AE60",
                       linestyle=":", linewidth=1.2,
                       label=f"x* = {self.marginal_buncher:,.0f}")

        ax.set_xlabel("Running variable")
        ax.set_ylabel("Count")
        ax.set_title(title or "Bunching at Notch Analysis")
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Jupyter HTML
    # ------------------------------------------------------------------
    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        rows = [
            ("Notch point", f"{self.notch_point:,.2f}"),
            ("Excess bunching (B)", f"{self.excess_bunching:,.2f}"),
            ("Missing mass (H)", f"{self.missing_mass:,.2f}"),
            ("Marginal buncher (x*)", f"{self.marginal_buncher:,.2f}"),
        ]
        if self.elasticity is not None:
            rows.append(("Elasticity", f"{self.elasticity:.4f}"))
        rows += [
            ("SE (B)", f"{self.se_bunching:,.4f}"),
            ("p-value", f"{self.pvalue:.4f}"),
            (f"{100*(1-self.alpha):.0f}% CI",
             f"[{self.ci[0]:,.2f}, {self.ci[1]:,.2f}]"),
            ("N", f"{self.n_obs:,}"),
        ]

        html = (
            '<div style="font-family:Helvetica Neue,Arial,sans-serif;'
            'max-width:520px;border:1px solid #E5E7EB;border-radius:8px;'
            'overflow:hidden;margin:6px 0">'
            '<div style="background:linear-gradient(135deg,#1a1a2e 0%,'
            '#16213e 100%);color:#fff;padding:12px 16px">'
            '<h3 style="margin:0;font-size:15px">Bunching at Notch</h3>'
            '<div style="font-size:11px;color:#94A3B8;margin-top:2px">'
            'Kleven &amp; Waseem (2013)</div></div>'
            '<table style="width:100%;border-collapse:collapse;'
            'font-size:13px">'
        )
        for label, val in rows:
            html += (
                f'<tr style="border-bottom:1px solid #F1F5F9">'
                f'<td style="padding:6px 14px;color:#64748B">{label}</td>'
                f'<td style="padding:6px 14px;text-align:right;'
                f'font-weight:600;color:#1a1a2e">{val}</td></tr>'
            )
        html += '</table></div>'
        return html


# ======================================================================
# Core estimation helpers
# ======================================================================

def _fit_counterfactual(bin_centers, counts, exclude_mask, poly_order,
                        notch_point, bin_width):
    """Fit counterfactual polynomial excluding the notch region."""
    x_out = bin_centers[~exclude_mask]
    y_out = counts[~exclude_mask]

    # Normalise for numerical stability
    x_norm = (x_out - notch_point) / bin_width
    x_all_norm = (bin_centers - notch_point) / bin_width

    coeffs = np.polyfit(x_norm, y_out, poly_order)
    cf = np.polyval(coeffs, x_all_norm)
    cf = np.maximum(cf, 0)
    return cf, coeffs


def _find_marginal_buncher(bin_centers, counts, counterfactual,
                           notch_point, bin_width):
    """
    Iteratively find x* such that excess bunching = missing mass.

    Starting just above the notch, shift the upper bound until the
    integral of the counterfactual above the notch (up to x*) minus
    observed equals the excess bunching below.
    """
    below = bin_centers <= notch_point
    excess_B = float(np.sum(counts[below] - counterfactual[below]))
    excess_B = max(excess_B, 0)

    above_idx = np.where(bin_centers > notch_point)[0]
    if len(above_idx) == 0:
        return notch_point + bin_width, excess_B, 0.0

    cumulative_hole = 0.0
    x_star = bin_centers[above_idx[-1]]

    for idx in above_idx:
        gap = counterfactual[idx] - counts[idx]
        if gap > 0:
            cumulative_hole += gap
        if cumulative_hole >= excess_B:
            x_star = bin_centers[idx]
            break

    missing_H = cumulative_hole
    return float(x_star), excess_B, float(missing_H)


# ======================================================================
# Public API
# ======================================================================

def notch(
    data: pd.DataFrame,
    x: str,
    notch_point: float,
    notch_size=None,
    bin_width: float = 500,
    poly_order: int = 7,
    exclude_range: Optional[Tuple[float, float]] = None,
    n_boot: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> NotchResult:
    """
    Bunching at Notches estimator (Kleven & Waseem 2013).

    Estimates behavioural responses to a notch (discontinuous jump)
    in a tax or benefit schedule by comparing the observed
    distribution to a counterfactual polynomial.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    x : str
        Running variable name (e.g., 'income').
    notch_point : float
        Location of the notch in the running variable.
    notch_size : float, optional
        Size of the discontinuous jump (delta-tau). If provided,
        a structural elasticity is estimated.
    bin_width : float, default 500
        Bin width for the histogram.
    poly_order : int, default 7
        Polynomial order for the counterfactual distribution.
    exclude_range : tuple of (float, float), optional
        Range around the notch to exclude from the counterfactual fit.
        If None, defaults to (notch_point - 3*bin_width,
        notch_point + 5*bin_width).
    n_boot : int, default 500
        Number of bootstrap replications for standard errors.
    seed : int, default 42
        Random seed for reproducibility.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    NotchResult
        Result object with excess_bunching, missing_mass,
        marginal_buncher, elasticity (if notch_size given),
        bootstrap SEs, and methods for .summary(), .plot(), and
        Jupyter HTML display.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.notch(df, x='income', notch_point=50000,
    ...                   notch_size=0.10, bin_width=500)
    >>> result.summary()
    >>> result.plot()

    Notes
    -----
    The algorithm:
    1. Bin the running variable with given bin_width.
    2. Exclude the notch region and fit a polynomial counterfactual.
    3. Excess bunching B = observed - counterfactual below the notch.
    4. Missing mass H = counterfactual - observed above the notch.
    5. Find marginal buncher x* where cumulative H >= B.
    6. Elasticity e = (x* - z*) / (z* * dt) if notch_size given.
    """
    if x not in data.columns:
        raise ValueError(f"Column '{x}' not found in data.")

    z = data[x].dropna().values.astype(np.float64)
    n = len(z)

    # ------------------------------------------------------------------
    # 1. Create histogram
    # ------------------------------------------------------------------
    # Range: extend well beyond the notch
    z_min = np.floor(z.min() / bin_width) * bin_width
    z_max = np.ceil(z.max() / bin_width) * bin_width
    bins = np.arange(z_min, z_max + bin_width / 2, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(z, bins=bins)
    counts = counts.astype(np.float64)

    # ------------------------------------------------------------------
    # 2. Define excluded region
    # ------------------------------------------------------------------
    if exclude_range is None:
        exc_lo = notch_point - 3 * bin_width
        exc_hi = notch_point + 5 * bin_width
    else:
        exc_lo, exc_hi = exclude_range

    exclude_mask = (bin_centers >= exc_lo) & (bin_centers <= exc_hi)

    # ------------------------------------------------------------------
    # 3. Fit counterfactual
    # ------------------------------------------------------------------
    cf, coeffs = _fit_counterfactual(
        bin_centers, counts, exclude_mask, poly_order,
        notch_point, bin_width,
    )

    # ------------------------------------------------------------------
    # 4. Compute excess bunching, missing mass, marginal buncher
    # ------------------------------------------------------------------
    x_star, excess_B, missing_H = _find_marginal_buncher(
        bin_centers, counts, cf, notch_point, bin_width,
    )

    # ------------------------------------------------------------------
    # 5. Elasticity
    # ------------------------------------------------------------------
    elasticity = None
    if notch_size is not None and notch_size > 0 and notch_point > 0:
        dx = x_star - notch_point
        if dx > 0:
            elasticity = dx / (notch_point * notch_size)
        else:
            elasticity = 0.0

    # ------------------------------------------------------------------
    # 6. Bootstrap standard errors
    # ------------------------------------------------------------------
    rng = np.random.RandomState(seed)
    boot_B = np.zeros(n_boot)
    boot_elast = np.zeros(n_boot) if elasticity is not None else None

    for b in range(n_boot):
        z_b = rng.choice(z, size=n, replace=True)
        counts_b, _ = np.histogram(z_b, bins=bins)
        counts_b = counts_b.astype(np.float64)

        cf_b, _ = _fit_counterfactual(
            bin_centers, counts_b, exclude_mask, poly_order,
            notch_point, bin_width,
        )
        x_star_b, B_b, _ = _find_marginal_buncher(
            bin_centers, counts_b, cf_b, notch_point, bin_width,
        )
        boot_B[b] = B_b

        if boot_elast is not None and notch_size > 0 and notch_point > 0:
            dx_b = x_star_b - notch_point
            boot_elast[b] = max(dx_b / (notch_point * notch_size), 0)

    se_B = float(np.std(boot_B, ddof=1))
    se_elast = float(np.std(boot_elast, ddof=1)) if boot_elast is not None else None

    # ------------------------------------------------------------------
    # 7. Inference
    # ------------------------------------------------------------------
    if se_B > 0:
        z_stat = excess_B / se_B
        pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(z_stat))))
    else:
        pvalue = 0.0

    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (excess_B - z_crit * se_B, excess_B + z_crit * se_B)

    # ------------------------------------------------------------------
    # 8. Build CausalResult for interop
    # ------------------------------------------------------------------
    detail = pd.DataFrame({
        'bin_center': bin_centers,
        'observed': counts,
        'counterfactual': cf,
        'excess': counts - cf,
        'in_excluded_region': exclude_mask,
    })

    model_info = {
        'notch_point': notch_point,
        'notch_size': notch_size,
        'bin_width': bin_width,
        'poly_order': poly_order,
        'exclude_range': (exc_lo, exc_hi),
        'excess_bunching': float(excess_B),
        'missing_mass': float(missing_H),
        'marginal_buncher': float(x_star),
        'design': 'notch',
    }
    if elasticity is not None:
        model_info['elasticity'] = float(elasticity)

    cr = CausalResult(
        method='Bunching at Notch (Kleven & Waseem 2013)',
        estimand='Excess Mass at Notch',
        estimate=float(excess_B),
        se=se_B,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='notch',
    )

    return NotchResult(
        excess_bunching=float(excess_B),
        missing_mass=float(missing_H),
        marginal_buncher=float(x_star),
        elasticity=elasticity,
        se_bunching=se_B,
        se_elasticity=se_elast,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        notch_point=notch_point,
        bin_centers=bin_centers,
        observed=counts,
        counterfactual=cf,
        n_obs=n,
        causal_result=cr,
        notch_size=notch_size,
    )


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS['notch'] = (
    "@article{kleven2013using,\n"
    "  title={Using Notches to Uncover Optimization Frictions and "
    "Structural Elasticities: Theory and Evidence from Pakistan},\n"
    "  author={Kleven, Henrik Jacobsen and Waseem, Mazhar},\n"
    "  journal={The Quarterly Journal of Economics},\n"
    "  volume={128},\n"
    "  number={2},\n"
    "  pages={669--723},\n"
    "  year={2013},\n"
    "  publisher={MIT Press}\n"
    "}"
)
