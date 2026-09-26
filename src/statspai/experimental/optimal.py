"""
Optimal experimental design.

Optimal sample allocation, cluster size determination, and
stratification design for RCTs.

References
----------
Duflo, E., Glennerster, R. & Kremer, M. (2007).
"Using Randomization in Development Economics Research: A Toolkit."
*Handbook of Development Economics*, 4, 3895-3962. [@duflo2007chapter]
"""

import warnings
from typing import List, Optional

import numpy as np
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility


class OptimalDesignResult(ResultProtocolMixin):
    """Results from optimal design calculation.

    Returned by :func:`optimal_design`. Carries the required total / per-arm
    sample size, cluster counts and size (for cluster designs), the intra-
    cluster correlation, the minimum detectable effect, and the target power.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.optimal_design(
    ...     design="cluster", mde=0.2, sigma=1.0, icc=0.05, cluster_size=20
    ... )
    >>> type(result).__name__
    'OptimalDesignResult'
    >>> result.design_type
    'Cluster RCT'
    >>> bool(result.n_total > 0 and result.n_clusters > 0)
    True
    """

    def __init__(
        self,
        n_total: Optional[int],
        n_per_arm: Optional[int],
        n_clusters: Optional[int],
        cluster_size: Optional[int],
        icc: float,
        mde: Optional[float],
        power: float,
        alpha: float,
        design_type: str,
    ) -> None:
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
        mde_text = f"{self.mde:.4f}" if self.mde is not None else "not solved"
        lines: List[str] = [
            "Optimal Experimental Design",
            "=" * 50,
            f"Design: {self.design_type}",
            f"Total sample: {self.n_total}",
            f"Per arm: {self.n_per_arm}",
            f"MDE: {mde_text}",
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
    mde: Optional[float] = None,
    power: float = 0.8,
    alpha: float = 0.05,
    n_arms: int = 2,
    prop_treat: float = 0.5,
    icc: float = 0.0,
    cluster_size: Optional[int] = None,
    n_clusters: Optional[int] = None,
    cost_per_cluster: Optional[float] = None,
    cost_per_unit: Optional[float] = None,
    r2: float = 0.0,
    baseline_mean: float = 0.0,
    n: Optional[int] = None,
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
    n : int, optional
        Total number of individuals, used to solve for the MDE when ``mde`` is
        None (individual and stratified designs; cluster designs use
        ``n_clusters`` and ``cluster_size``).

    Returns
    -------
    OptimalDesignResult
        ``n_total`` is the total sample (individuals); ``n_per_arm`` the size
        of the largest arm (all arms are equal under ``prop_treat=0.5``).

    Notes
    -----
    Two arms with treated share ``p`` and total sample ``N`` give
    ``Var(tau_hat) = sigma^2 (1 - r2) deff / (N p (1 - p))``, so
    ``N = (z_{1-alpha/2} + z_{power})^2 sigma^2 (1 - r2) deff / (mde^2 p (1 - p))``
    individuals in total, ``p N`` treated and ``(1 - p) N`` control
    (``deff = 1 + (cluster_size - 1) icc`` for cluster designs, 1 otherwise).
    With ``n_arms > 2`` every arm is sized for its pairwise comparison with
    control at equal allocation. When ``cost_per_cluster`` and
    ``cost_per_unit`` are given, the cost-optimal cluster size
    ``sqrt((cost_per_cluster / cost_per_unit) (1 - icc) / icc)`` is used for the
    sample-size calculation itself.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.optimal_design(
    ...     design="cluster", mde=0.2, sigma=1.0, icc=0.05, cluster_size=20
    ... )
    >>> result.design_type, result.cluster_size, result.icc
    ('Cluster RCT', 20, 0.05)
    >>> text = result.summary()
    """
    if design not in {"individual", "cluster", "stratified"}:
        raise MethodIncompatibility(f"Unknown design: {design}")
    if n_arms < 2:
        raise MethodIncompatibility(f"n_arms must be at least 2, got {n_arms}.")
    if not 0 < prop_treat < 1:
        raise MethodIncompatibility(f"prop_treat must lie in (0, 1), got {prop_treat}.")
    if n_arms > 2 and prop_treat != 0.5:
        raise MethodIncompatibility(
            "prop_treat applies to two-arm designs; with n_arms > 2 every arm "
            "is sized for an equal-allocation comparison with control."
        )
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    variance_factor = 1 - r2  # variance reduction from baseline covariates

    if design != "cluster":
        if icc or cluster_size is not None:
            warnings.warn(
                f"optimal_design(design={design!r}) ignores icc / cluster_size; "
                "use design='cluster' for clustered assignment.",
                UserWarning,
                stacklevel=2,
            )
        label = "Individual RCT" if design == "individual" else "Stratified RCT"
        if mde is not None:
            arms = _arm_sizes(z, sigma, variance_factor, mde, n_arms, prop_treat)
            return OptimalDesignResult(
                n_total=int(sum(arms)),
                n_per_arm=int(max(arms)),
                n_clusters=None,
                cluster_size=None,
                icc=0,
                mde=mde,
                power=power,
                alpha=alpha,
                design_type=label,
            )
        if n is None:
            raise MethodIncompatibility(
                "optimal_design needs either mde= or n= (total sample)."
            )
        return OptimalDesignResult(
            n_total=int(n),
            n_per_arm=None,
            n_clusters=None,
            cluster_size=None,
            icc=0,
            mde=_mde(z, sigma, variance_factor, n, n_arms, prop_treat),
            power=power,
            alpha=alpha,
            design_type=label,
        )

    if cost_per_cluster is not None and cost_per_unit is not None:
        if not 0 < icc < 1:
            raise MethodIncompatibility(
                "The cost-optimal cluster size needs 0 < icc < 1, got " f"icc={icc}."
            )
        optimal_m = max(
            1,
            int(
                np.round(np.sqrt((cost_per_cluster / cost_per_unit) * (1 - icc) / icc))
            ),
        )
        if cluster_size is not None and cluster_size != optimal_m:
            warnings.warn(
                f"cluster_size={cluster_size} replaced by the cost-optimal "
                f"cluster size {optimal_m}.",
                UserWarning,
                stacklevel=2,
            )
        cluster_size = optimal_m
    elif cluster_size is None:
        cluster_size = 20
        warnings.warn(
            "optimal_design(design='cluster') without cluster_size assumes 20 "
            "individuals per cluster; pass cluster_size= explicitly.",
            UserWarning,
            stacklevel=2,
        )
    deff = 1 + (cluster_size - 1) * icc

    if mde is not None:
        arms = _arm_sizes(z, sigma, variance_factor * deff, mde, n_arms, prop_treat)
        clusters = [int(np.ceil(a / cluster_size)) for a in arms]
        return OptimalDesignResult(
            n_total=int(sum(clusters) * cluster_size),
            n_per_arm=int(max(clusters) * cluster_size),
            n_clusters=int(sum(clusters)),
            cluster_size=cluster_size,
            icc=icc,
            mde=mde,
            power=power,
            alpha=alpha,
            design_type="Cluster RCT",
        )
    if n_clusters is None:
        raise MethodIncompatibility(
            "optimal_design(design='cluster') needs either mde= or n_clusters= "
            "(total clusters)."
        )
    n_total = int(n_clusters * cluster_size)
    return OptimalDesignResult(
        n_total=n_total,
        n_per_arm=None,
        n_clusters=int(n_clusters),
        cluster_size=cluster_size,
        icc=icc,
        mde=_mde(z, sigma, variance_factor * deff, n_total, n_arms, prop_treat),
        power=power,
        alpha=alpha,
        design_type="Cluster RCT",
    )


def _arm_sizes(
    z: float,
    sigma: float,
    variance_factor: float,
    mde: float,
    n_arms: int,
    prop_treat: float,
) -> List[int]:
    """Individuals per arm needed to detect ``mde`` (see optimal_design Notes)."""
    k = z**2 * sigma**2 * variance_factor / mde**2
    if n_arms == 2:
        # N = k / (p (1 - p)); treated p N = k / (1 - p), control (1 - p) N = k / p.
        return [int(np.ceil(k / (1 - prop_treat))), int(np.ceil(k / prop_treat))]
    return [int(np.ceil(2 * k))] * n_arms


def _mde(
    z: float,
    sigma: float,
    variance_factor: float,
    n_total: int,
    n_arms: int,
    prop_treat: float,
) -> float:
    """Minimum detectable effect for ``n_total`` individuals."""
    if n_arms == 2:
        return float(
            z
            * sigma
            * np.sqrt(variance_factor / (n_total * prop_treat * (1 - prop_treat)))
        )
    return float(z * sigma * np.sqrt(2 * variance_factor * n_arms / n_total))
