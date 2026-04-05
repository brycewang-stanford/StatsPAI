"""
Optimal experimental design.

Optimal sample allocation, cluster size determination, and
stratification design for RCTs.

References
----------
Duflo, E., Glennerster, R. & Kremer, M. (2007).
"Using Randomization in Development Economics Research: A Toolkit."
*Handbook of Development Economics*, 4, 3895-3962.
"""

from typing import Optional, List, Dict, Any
import numpy as np
from scipy import stats


class OptimalDesignResult:
    """Results from optimal design calculation."""

    def __init__(self, n_total, n_per_arm, n_clusters, cluster_size,
                 icc, mde, power, alpha, design_type):
        self.n_total = n_total
        self.n_per_arm = n_per_arm
        self.n_clusters = n_clusters
        self.cluster_size = cluster_size
        self.icc = icc
        self.mde = mde
        self.power = power
        self.alpha = alpha
        self.design_type = design_type

    def summary(self) -> str:
        lines = [
            "Optimal Experimental Design",
            "=" * 50,
            f"Design: {self.design_type}",
            f"Total sample: {self.n_total}",
            f"Per arm: {self.n_per_arm}",
            f"MDE: {self.mde:.4f}",
            f"Power: {self.power:.1%}",
            f"Alpha: {self.alpha}",
        ]
        if self.n_clusters:
            lines.append(f"Clusters: {self.n_clusters}")
            lines.append(f"Cluster size: {self.cluster_size}")
            lines.append(f"ICC: {self.icc:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)


def optimal_design(
    design: str = "individual",
    sigma: float = 1.0,
    mde: float = None,
    power: float = 0.8,
    alpha: float = 0.05,
    n_arms: int = 2,
    prop_treat: float = 0.5,
    icc: float = 0.0,
    cluster_size: int = None,
    n_clusters: int = None,
    cost_per_cluster: float = None,
    cost_per_unit: float = None,
    r2: float = 0.0,
    baseline_mean: float = 0.0,
) -> OptimalDesignResult:
    """
    Compute optimal sample size and design parameters.

    Parameters
    ----------
    design : str, default 'individual'
        'individual', 'cluster', 'stratified'.
    sigma : float, default 1.0
        Standard deviation of the outcome.
    mde : float, optional
        Minimum detectable effect. If None, compute MDE given n.
    power : float, default 0.8
        Statistical power (1 - Type II error).
    alpha : float, default 0.05
        Significance level.
    n_arms : int, default 2
        Number of treatment arms.
    prop_treat : float, default 0.5
        Proportion assigned to treatment.
    icc : float, default 0.0
        Intra-cluster correlation (for cluster designs).
    cluster_size : int, optional
        Average cluster size.
    n_clusters : int, optional
        Number of clusters (if fixed).
    cost_per_cluster : float, optional
        Cost of adding a cluster (for optimal allocation).
    cost_per_unit : float, optional
        Cost per individual unit.
    r2 : float, default 0.0
        R-squared from baseline covariates (variance reduction).
    baseline_mean : float, default 0.0

    Returns
    -------
    OptimalDesignResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.optimal_design(mde=0.2, sigma=1.0, icc=0.05, cluster_size=20)
    >>> print(result.summary())
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    variance_factor = 1 - r2  # Variance reduction from covariates

    if design == 'individual':
        if mde is not None:
            # Compute required n
            n_per_arm = int(np.ceil(
                ((z_alpha + z_beta)**2 * sigma**2 * variance_factor) /
                (mde**2 * prop_treat * (1 - prop_treat))
            ))
            n_total = int(np.ceil(n_per_arm / prop_treat))
        else:
            n_per_arm = None
            n_total = None
            mde = (z_alpha + z_beta) * sigma * np.sqrt(variance_factor) / np.sqrt(
                prop_treat * (1 - prop_treat))

        return OptimalDesignResult(
            n_total=n_total, n_per_arm=n_per_arm,
            n_clusters=None, cluster_size=None,
            icc=0, mde=mde, power=power, alpha=alpha,
            design_type='Individual RCT',
        )

    elif design == 'cluster':
        # Design effect
        if cluster_size is None:
            cluster_size = 20  # default

        deff = 1 + (cluster_size - 1) * icc

        if mde is not None:
            n_ind_per_arm = int(np.ceil(
                ((z_alpha + z_beta)**2 * sigma**2 * variance_factor * deff) /
                (mde**2 * prop_treat * (1 - prop_treat))
            ))
            n_clusters_per_arm = int(np.ceil(n_ind_per_arm / cluster_size))
            n_clusters_total = n_clusters_per_arm * n_arms
            n_total = n_clusters_total * cluster_size
        else:
            n_clusters_total = n_clusters or 100
            n_total = n_clusters_total * cluster_size
            n_ind_per_arm = int(n_total * prop_treat)
            mde = (z_alpha + z_beta) * sigma * np.sqrt(
                variance_factor * deff / (prop_treat * (1 - prop_treat) * n_ind_per_arm))

        # Optimal cluster size given costs
        if cost_per_cluster is not None and cost_per_unit is not None:
            optimal_m = np.sqrt(
                (cost_per_cluster / cost_per_unit) * ((1 - icc) / icc)
            )
            cluster_size = max(1, int(np.round(optimal_m)))

        return OptimalDesignResult(
            n_total=n_total, n_per_arm=n_total // n_arms,
            n_clusters=n_clusters_total, cluster_size=cluster_size,
            icc=icc, mde=mde, power=power, alpha=alpha,
            design_type='Cluster RCT',
        )

    elif design == 'stratified':
        # Stratified design reduces variance by (1-R²_strata)
        if mde is not None:
            n_per_arm = int(np.ceil(
                ((z_alpha + z_beta)**2 * sigma**2 * variance_factor) /
                (mde**2 * prop_treat * (1 - prop_treat))
            ))
            n_total = int(np.ceil(n_per_arm / prop_treat))
        else:
            n_per_arm = None
            n_total = None

        return OptimalDesignResult(
            n_total=n_total, n_per_arm=n_per_arm,
            n_clusters=None, cluster_size=None,
            icc=0, mde=mde, power=power, alpha=alpha,
            design_type='Stratified RCT',
        )

    else:
        raise ValueError(f"Unknown design: {design}")
