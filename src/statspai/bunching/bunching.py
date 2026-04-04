"""
Bunching Estimator: Estimating elasticities from kinks and notches.

The bunching approach estimates behavioural responses to policy kinks
(changes in marginal incentives) or notches (discontinuous changes in
average incentives) by:

1. Bin the running variable around the threshold.
2. Fit a counterfactual polynomial to the distribution excluding
   the bunching region.
3. Excess mass B = observed - counterfactual in the bunching region.
4. Convert excess mass to an elasticity estimate.

For a kink with tax rate change dt at threshold z*:
    elasticity = B / (z* * dt / (1 + dt))

References
----------
Kleven, H. J. & Waseem, M. (2013).
"Using Notches to Uncover Optimization Frictions and Structural
Elasticities." QJE, 128(2), 669-723.

Chetty, R., Friedman, J. N., Olsen, T., & Pistaferri, L. (2011).
"Adjustment Costs, Firm Responses, and Micro vs. Macro Labor Supply
Elasticities." QJE, 126(2), 749-804.
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

def bunching(
    data: pd.DataFrame,
    running_var: str,
    threshold: float,
    bin_width: Optional[float] = None,
    n_bins: int = 50,
    poly_order: int = 7,
    bunch_region: Optional[Tuple[float, float]] = None,
    exclude_region: Optional[Tuple[float, float]] = None,
    dt: Optional[float] = None,
    design: str = 'kink',
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> CausalResult:
    """
    Estimate bunching at a policy threshold.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    running_var : str
        Name of the running variable (e.g., income).
    threshold : float
        Policy threshold (kink/notch point).
    bin_width : float, optional
        Width of bins. If None, computed from data range / n_bins.
    n_bins : int, default 50
        Number of bins on each side of the threshold.
    poly_order : int, default 7
        Order of the counterfactual polynomial.
    bunch_region : tuple of (float, float), optional
        (lower, upper) bounds of the bunching region.
        If None, uses [threshold - 2*bin_width, threshold + 2*bin_width].
    exclude_region : tuple of (float, float), optional
        Same as bunch_region unless otherwise specified.
    dt : float, optional
        Change in marginal tax rate at the kink (for elasticity).
        E.g., 0.10 for a 10pp increase.
    design : str, default 'kink'
        'kink' or 'notch'.
    n_bootstrap : int, default 200
        Bootstrap iterations for standard errors.
    alpha : float, default 0.05
        Significance level.
    random_state : int, default 42

    Returns
    -------
    CausalResult
        'estimate' is the excess mass (bunching) estimate B.
        model_info contains 'elasticity' if dt is provided,
        'counterfactual' histogram, and 'observed' histogram.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.bunching(df, running_var='income',
    ...                      threshold=50000, dt=0.10)
    >>> print(result.summary())
    >>> print(f"Elasticity: {result.model_info['elasticity']:.4f}")
    """
    est = BunchingEstimator(
        data=data, running_var=running_var, threshold=threshold,
        bin_width=bin_width, n_bins=n_bins, poly_order=poly_order,
        bunch_region=bunch_region, exclude_region=exclude_region,
        dt=dt, design=design, n_bootstrap=n_bootstrap,
        alpha=alpha, random_state=random_state,
    )
    return est.fit()


# ======================================================================
# BunchingEstimator class
# ======================================================================

class BunchingEstimator:
    """
    Bunching estimator for kink/notch designs.

    Parameters
    ----------
    data : pd.DataFrame
    running_var : str
    threshold : float
    bin_width : float, optional
    n_bins : int
    poly_order : int
    bunch_region : tuple, optional
    exclude_region : tuple, optional
    dt : float, optional
    design : str
    n_bootstrap : int
    alpha : float
    random_state : int
    """

    def __init__(
        self,
        data: pd.DataFrame,
        running_var: str,
        threshold: float,
        bin_width: Optional[float] = None,
        n_bins: int = 50,
        poly_order: int = 7,
        bunch_region: Optional[Tuple[float, float]] = None,
        exclude_region: Optional[Tuple[float, float]] = None,
        dt: Optional[float] = None,
        design: str = 'kink',
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        self.data = data
        self.running_var = running_var
        self.threshold = threshold
        self.bin_width = bin_width
        self.n_bins = n_bins
        self.poly_order = poly_order
        self.bunch_region = bunch_region
        self.exclude_region = exclude_region
        self.dt = dt
        self.design = design
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state

    def fit(self) -> CausalResult:
        """Run bunching estimation."""
        if self.running_var not in self.data.columns:
            raise ValueError(f"Column '{self.running_var}' not found")

        z = self.data[self.running_var].dropna().values.astype(np.float64)
        n = len(z)

        # Determine bin width
        if self.bin_width is None:
            z_range = z.max() - z.min()
            self.bin_width = z_range / (2 * self.n_bins)

        # Create bins centered around threshold
        bin_lo = self.threshold - self.n_bins * self.bin_width
        bin_hi = self.threshold + self.n_bins * self.bin_width
        bins = np.arange(bin_lo, bin_hi + self.bin_width / 2, self.bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Histogram
        counts, _ = np.histogram(z, bins=bins)
        counts = counts.astype(np.float64)

        # Bunching region
        if self.bunch_region is None:
            bl = self.threshold - 2 * self.bin_width
            bu = self.threshold + 2 * self.bin_width
        else:
            bl, bu = self.bunch_region

        if self.exclude_region is None:
            self.exclude_region = (bl, bu)

        # Identify bins in/out of bunching region
        in_bunch = (bin_centers >= bl) & (bin_centers <= bu)
        out_bunch = ~in_bunch

        # Fit counterfactual polynomial on non-bunching bins
        x_out = bin_centers[out_bunch]
        y_out = counts[out_bunch]

        # Normalise x for numerical stability
        x_norm = (x_out - self.threshold) / self.bin_width
        x_all_norm = (bin_centers - self.threshold) / self.bin_width

        # Fit polynomial
        coeffs = np.polyfit(x_norm, y_out, self.poly_order)
        counterfactual = np.polyval(coeffs, x_all_norm)
        counterfactual = np.maximum(counterfactual, 0)

        # Excess mass
        excess = np.sum(counts[in_bunch]) - np.sum(counterfactual[in_bunch])
        # Normalised bunching: B = excess / counterfactual_at_threshold
        cf_at_threshold = np.polyval(coeffs, 0)
        if cf_at_threshold > 0:
            B_normalised = excess / cf_at_threshold
        else:
            B_normalised = 0.0

        # Elasticity (for kink design)
        elasticity = None
        if self.dt is not None and self.dt > 0:
            if self.design == 'kink':
                # e = B / (z* * dt / (1 + dt))
                dz_star = self.threshold * self.dt / (1 + self.dt)
                if dz_star > 0:
                    elasticity = B_normalised * self.bin_width / dz_star
            elif self.design == 'notch':
                elasticity = B_normalised  # simplified

        # Bootstrap SE
        rng = np.random.RandomState(self.random_state)
        boot_B = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            z_b = rng.choice(z, size=n, replace=True)
            counts_b, _ = np.histogram(z_b, bins=bins)
            counts_b = counts_b.astype(np.float64)

            y_out_b = counts_b[out_bunch]
            coeffs_b = np.polyfit(x_norm, y_out_b, self.poly_order)
            cf_b = np.polyval(coeffs_b, x_all_norm)
            cf_b = np.maximum(cf_b, 0)

            excess_b = np.sum(counts_b[in_bunch]) - np.sum(cf_b[in_bunch])
            cf_thresh_b = np.polyval(coeffs_b, 0)
            boot_B[b] = excess_b / max(cf_thresh_b, 1)

        se = float(np.std(boot_B, ddof=1))

        if se > 0:
            z_stat = B_normalised / se
            pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(z_stat))))
        else:
            pvalue = 0.0

        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)
        ci = (B_normalised - z_crit * se, B_normalised + z_crit * se)

        detail = pd.DataFrame({
            'bin_center': bin_centers,
            'observed': counts,
            'counterfactual': counterfactual,
            'excess': counts - counterfactual,
            'in_bunching_region': in_bunch,
        })

        model_info = {
            'threshold': self.threshold,
            'bin_width': self.bin_width,
            'bunch_region': (bl, bu),
            'excess_mass_raw': float(excess),
            'excess_mass_normalised': float(B_normalised),
            'counterfactual_at_threshold': float(cf_at_threshold),
            'poly_order': self.poly_order,
            'design': self.design,
            'dt': self.dt,
        }

        if elasticity is not None:
            model_info['elasticity'] = float(elasticity)

        return CausalResult(
            method=f'Bunching Estimator ({self.design.title()} Design)',
            estimand='Excess Mass (Normalised)',
            estimate=float(B_normalised),
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=detail,
            model_info=model_info,
            _citation_key='bunching',
        )


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS['bunching'] = (
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
