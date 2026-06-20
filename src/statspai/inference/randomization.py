"""
Randomization Inference (Fisher's Exact Test for causal effects).

Computes exact p-values by permuting treatment assignment and
comparing the observed test statistic to the permutation distribution.
Increasingly required by top journals (Young 2019, QJE).

This module provides both the original ``ri_test`` for quick usage and
the enhanced ``fisher_exact`` with FisherResult class for richer output
including Hodges-Lehmann confidence intervals and plotting.

References
----------
Fisher, R.A. (1935).
*The Design of Experiments*. Oliver and Boyd.

Young, A. (2019).
"Channeling Fisher: Randomization Tests and the Statistical
Insignificance of Seemingly Significant Experimental Results."
*Quarterly Journal of Economics*, 134(2), 557-598. [@young2019channeling]

Hodges, J.L. and Lehmann, E.L. (1963).
"Estimates of Location Based on Rank Tests."
*Annals of Mathematical Statistics*, 34(2), 598-611. [@hodges1963estimates]
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult

StatFn = Callable[[np.ndarray, np.ndarray], float]
Permuter = Callable[[], np.ndarray]


# ======================================================================
# FisherResult class
# ======================================================================


class FisherResult:
    """
    Result container for Fisher's exact permutation test.

    Attributes
    ----------
    statistic : float
        Observed test statistic.
    p_value : float
        Two-sided permutation p-value.
    p_one_sided : float
        One-sided (greater) permutation p-value.
    ci : tuple of float
        Confidence interval from Hodges-Lehmann inversion.
    perm_dist : np.ndarray
        Full permutation distribution of the test statistic.
    statistic_type : str
        Name of the test statistic used.
    n_perm : int
        Number of permutations performed.
    n_obs : int
        Number of observations.
    n_treated : int
        Number of treated units.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> d = rng.integers(0, 2, 80)
    >>> y = 0.5 * d + rng.normal(size=80)
    >>> df = pd.DataFrame({"outcome": y, "treated": d})
    >>> res = sp.fisher_exact(data=df, y="outcome", treatment="treated",
    ...                       statistic="ate", n_perm=2000, seed=42)
    >>> type(res).__name__
    'FisherResult'
    >>> bool(0.0 <= res.p_value <= 1.0)
    True
    >>> res.n_obs
    80
    """

    def __init__(
        self,
        statistic: float,
        p_value: float,
        p_one_sided: float,
        ci: Tuple[float, float],
        perm_dist: np.ndarray,
        statistic_type: str,
        n_perm: int,
        n_obs: int,
        n_treated: int,
    ) -> None:
        self.statistic = statistic
        self.p_value = p_value
        self.p_one_sided = p_one_sided
        self.ci = ci
        self.perm_dist = perm_dist
        self.statistic_type = statistic_type
        self.n_perm = n_perm
        self.n_obs = n_obs
        self.n_treated = n_treated

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 60,
            "Fisher's Exact Randomization Test",
            "=" * 60,
            f"  Test statistic ({self.statistic_type}):  {self.statistic:.6f}",
            f"  Two-sided p-value:           {self.p_value:.4f}",
            f"  One-sided p-value (greater):  {self.p_one_sided:.4f}",
            f"  95% CI (Hodges-Lehmann):      [{self.ci[0]:.4f}, {self.ci[1]:.4f}]",
            "-" * 60,
            f"  Permutations:  {self.n_perm:,}",
            f"  N (total):     {self.n_obs:,}",
            f"  N (treated):   {self.n_treated:,}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def plot(
        self,
        ax: Optional[Any] = None,
        figsize: Tuple[int, int] = (8, 5),
    ) -> Tuple[Any, Any]:
        """
        Plot the permutation distribution with observed value.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates a new figure.
        figsize : tuple, default (8, 5)
            Figure size if creating a new figure.

        Returns
        -------
        tuple of (fig, ax)
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.hist(
            self.perm_dist,
            bins=min(50, max(20, self.n_perm // 20)),
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
            label="Permutation distribution",
        )
        ax.axvline(
            self.statistic,
            color="red",
            linewidth=2,
            linestyle="--",
            label=f"Observed = {self.statistic:.4f}",
        )
        ax.axvline(
            -abs(self.statistic),
            color="red",
            linewidth=1,
            linestyle=":",
            alpha=0.5,
        )
        ax.axvline(
            abs(self.statistic),
            color="red",
            linewidth=1,
            linestyle=":",
            alpha=0.5,
        )

        ax.set_xlabel("Test Statistic")
        ax.set_ylabel("Density")
        ax.set_title(f"Fisher Randomization Test (p = {self.p_value:.4f})")
        ax.legend()

        return fig, ax

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        sig_color = "#d32f2f" if self.p_value < 0.05 else "#388e3c"
        box_style = (
            "font-family: monospace; padding: 10px; border: 1px solid #ddd; "
            "border-radius: 5px; max-width: 500px;"
        )
        cell_style = "padding: 4px 8px;"
        right_style = f"{cell_style} text-align: right;"
        p_style = f"{right_style} color: {sig_color}; font-weight: bold;"
        return "\n".join(
            [
                f'<div style="{box_style}">',
                '<h3 style="margin-top: 0;">Fisher\'s Exact Randomization Test</h3>',
                '<table style="border-collapse: collapse; width: 100%;">',
                (
                    f'<tr><td style="{cell_style}"><b>Test statistic '
                    f"({self.statistic_type})</b></td>"
                    f'<td style="{right_style}">{self.statistic:.6f}</td></tr>'
                ),
                (
                    f'<tr><td style="{cell_style}"><b>p-value (two-sided)</b></td>'
                    f'<td style="{p_style}">{self.p_value:.4f}</td></tr>'
                ),
                (
                    f'<tr><td style="{cell_style}"><b>p-value (one-sided)</b></td>'
                    f'<td style="{right_style}">{self.p_one_sided:.4f}</td></tr>'
                ),
                (
                    f'<tr><td style="{cell_style}"><b>95% CI</b></td>'
                    f'<td style="{right_style}">'
                    f"[{self.ci[0]:.4f}, {self.ci[1]:.4f}]</td></tr>"
                ),
                (
                    f'<tr><td style="{cell_style}"><b>Permutations</b></td>'
                    f'<td style="{right_style}">{self.n_perm:,}</td></tr>'
                ),
                (
                    f'<tr><td style="{cell_style}"><b>N</b></td>'
                    f'<td style="{right_style}">{self.n_obs:,} '
                    f"(treated: {self.n_treated:,})</td></tr>"
                ),
                "</table>",
                "</div>",
            ]
        )

    def __repr__(self) -> str:
        return (
            f"FisherResult(statistic={self.statistic:.4f}, "
            f"p_value={self.p_value:.4f}, ci={self.ci})"
        )


# ======================================================================
# Main functions
# ======================================================================


def fisher_exact(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    statistic: str = "ate",
    controls: Optional[List[str]] = None,
    n_perm: int = 10000,
    stratify: Optional[str] = None,
    cluster: Optional[str] = None,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> FisherResult:
    """
    Fisher's exact randomization test with enhanced features.

    Computes a permutation-based p-value and Hodges-Lehmann confidence
    interval for the treatment effect under the sharp null hypothesis.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable name.
    treatment : str
        Binary treatment variable name (0/1).
    statistic : str, default 'ate'
        Test statistic to use:
        - ``'ate'``: Average treatment effect (difference in means).
        - ``'ks'``: Kolmogorov-Smirnov statistic.
        - ``'rank_sum'``: Wilcoxon rank-sum statistic.
    controls : list of str, optional
        Control variables for covariate-adjusted inference.
        When provided, the test statistic is computed on residuals
        from regressing Y on controls.
    n_perm : int, default 10000
        Number of random permutations.
    stratify : str, optional
        Variable for stratified permutation (permute within strata).
    cluster : str, optional
        Variable for cluster-level randomization (permute by cluster).
    seed : int, optional
        Random seed for reproducibility.
    alpha : float, default 0.05
        Significance level for the Hodges-Lehmann confidence interval.

    Returns
    -------
    FisherResult
        Object with statistic, p_value, ci, perm_dist, and methods
        for summary() and plot().

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> d = rng.integers(0, 2, 80)
    >>> y = 0.5 * d + rng.normal(size=80)
    >>> df = pd.DataFrame({"outcome": y, "treated": d})
    >>> result = sp.fisher_exact(
    ...     data=df, y="outcome", treatment="treated",
    ...     statistic="ate", n_perm=2000, seed=42)
    >>> type(result).__name__
    'FisherResult'
    >>> bool(0.0 <= result.p_value <= 1.0)
    True
    >>> result.n_obs
    80

    Notes
    -----
    Under the sharp null H0: Y_i(1) = Y_i(0) for all i, treatment
    assignment is the only source of randomness. The p-value is the
    proportion of permuted statistics at least as extreme as observed.

    The Hodges-Lehmann CI is constructed by inverting the permutation
    test: find the set of tau_0 values for which the test of
    Y - tau_0 * D does not reject at level alpha.

    For stratified experiments, treatment is permuted within strata.
    For cluster-randomized experiments, treatment is permuted at
    the cluster level.

    See Young (2019, *QJE*) for why randomization p-values should
    accompany asymptotic p-values in experimental papers.
    """
    rng = np.random.default_rng(seed)

    # Prepare data
    keep_cols = [y, treatment]
    if controls:
        keep_cols += controls
    if stratify:
        keep_cols.append(stratify)
    if cluster:
        keep_cols.append(cluster)
    keep_cols = list(dict.fromkeys(keep_cols))  # deduplicate preserving order

    df = data[keep_cols].dropna()
    Y = df[y].values.astype(float)
    D = df[treatment].values.astype(float)
    n = len(Y)
    n_treated = int(D.sum())

    # Covariate adjustment: residualize Y on controls
    if controls:
        X_ctrl = df[controls].values.astype(float)
        X_ctrl = np.column_stack([np.ones(n), X_ctrl])
        beta_ctrl = np.linalg.lstsq(X_ctrl, Y, rcond=None)[0]
        Y = Y - X_ctrl @ beta_ctrl

    # Select test statistic function
    stat_fn = _get_stat_fn(statistic)

    # Observed statistic
    obs_stat = float(stat_fn(Y, D))

    # Build the permutation function
    perm_fn: Permuter
    if cluster is not None:
        perm_fn = _make_cluster_permuter(df, treatment, cluster, rng)
    elif stratify is not None:
        perm_fn = _make_stratified_permuter(df, treatment, stratify, rng)
    else:

        def unrestricted_permute() -> np.ndarray:
            return np.asarray(rng.permutation(D), dtype=float)

        perm_fn = unrestricted_permute

    # Permutation distribution
    perm_stats = np.zeros(n_perm)
    for b in range(n_perm):
        D_perm = perm_fn()
        perm_stats[b] = stat_fn(Y, D_perm)

    # P-values
    p_two_sided = float(np.mean(np.abs(perm_stats) >= np.abs(obs_stat)))
    p_one_sided = float(np.mean(perm_stats >= obs_stat))

    # Hodges-Lehmann confidence interval (only for ATE)
    if statistic == "ate":
        ci = _hodges_lehmann_ci(Y, D, stat_fn, perm_fn, n_perm, alpha, rng)
    else:
        # For non-ATE statistics, use percentile CI from permutation dist
        ci = (
            float(np.percentile(perm_stats, 100 * alpha / 2)),
            float(np.percentile(perm_stats, 100 * (1 - alpha / 2))),
        )

    _result = FisherResult(
        statistic=obs_stat,
        p_value=p_two_sided,
        p_one_sided=p_one_sided,
        ci=ci,
        perm_dist=perm_stats,
        statistic_type=statistic,
        n_perm=n_perm,
        n_obs=n,
        n_treated=n_treated,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.inference.fisher_exact",
            params={
                "y": y,
                "treatment": treatment,
                "statistic": statistic,
                "controls": list(controls) if controls else None,
                "n_perm": n_perm,
                "stratify": stratify,
                "cluster": cluster,
                "seed": seed,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def ri_test(
    data: pd.DataFrame,
    y: str,
    treat: str,
    stat: str = "diff_means",
    n_perms: int = 1000,
    cluster: Optional[str] = None,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Randomization inference p-value.

    Computes the Fisher exact p-value by permuting the treatment
    vector and recalculating the test statistic.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Binary treatment indicator (0/1).
    stat : str or callable, default 'diff_means'
        Test statistic:
        - ``'diff_means'``: difference in means (Y_bar_1 - Y_bar_0)
        - ``'ks'``: Kolmogorov-Smirnov statistic
        - ``'t'``: t-statistic
        - A callable ``f(Y, D) -> float`` for custom statistics.
    n_perms : int, default 1000
        Number of random permutations. Use 10000+ for publications.
    cluster : str, optional
        Cluster-level permutation (permute treatment at cluster level).
    seed : int, optional
        Random seed.
    alpha : float, default 0.05

    Returns
    -------
    dict
        ``'observed'``: observed test statistic
        ``'p_value'``: two-sided randomization p-value
        ``'p_one_sided'``: one-sided (greater) p-value
        ``'n_perms'``: number of permutations used
        ``'perm_distribution'``: array of permuted statistics

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> d = rng.integers(0, 2, 80)
    >>> y = 0.5 * d + rng.normal(size=80)
    >>> df = pd.DataFrame({"outcome": y, "treatment": d})
    >>> result = sp.ri_test(df, y='outcome', treat='treatment',
    ...                     n_perms=2000, seed=42)
    >>> sorted(result.keys())
    ['n_perms', 'observed', 'p_one_sided', 'p_value', 'perm_distribution']
    >>> bool(0.0 <= result['p_value'] <= 1.0)
    True
    >>> result['n_perms']
    2000

    Notes
    -----
    Under the sharp null H0: Y_i(1) = Y_i(0) for all i, the treatment
    assignment is the only source of randomness. The RI p-value is the
    fraction of permutation statistics at least as extreme as the
    observed statistic.

    For cluster-randomized experiments, treatment is permuted at the
    cluster level (all units in a cluster get the same permuted status).

    See Young (2019, *QJE*) for why RI p-values should be reported
    alongside asymptotic p-values in experimental papers.
    """
    rng = np.random.default_rng(seed)

    df = data[[y, treat]].copy()
    if cluster:
        df["_cluster"] = data[cluster].values
    df = df.dropna()

    Y = df[y].values.astype(float)
    D = df[treat].values.astype(float)
    n = len(Y)

    # Select test statistic function
    stat_fn: StatFn
    if callable(stat):
        stat_fn = cast(StatFn, stat)
    elif stat == "diff_means":

        def diff_means_stat(y_: np.ndarray, d_: np.ndarray) -> float:
            return float(np.mean(y_[d_ == 1]) - np.mean(y_[d_ == 0]))

        stat_fn = diff_means_stat
    elif stat == "t":
        stat_fn = _t_stat
    elif stat == "ks":

        def ks_stat(y_: np.ndarray, d_: np.ndarray) -> float:
            return float(stats.ks_2samp(y_[d_ == 1], y_[d_ == 0]).statistic)

        stat_fn = ks_stat
    else:
        raise ValueError(
            f"Unknown stat: '{stat}'. Use 'diff_means', 't', 'ks', or callable."
        )

    # Observed statistic
    obs_stat = float(stat_fn(Y, D))

    # Permutation distribution
    perm_stats = np.zeros(n_perms)

    if cluster:
        # Cluster-level permutation
        cl = df["_cluster"].values
        unique_cl = np.unique(cl)
        # Get treatment per cluster (first obs)
        cl_treat = np.array([D[cl == c][0] for c in unique_cl])
        for b in range(n_perms):
            cl_perm = rng.permutation(cl_treat)
            D_perm = np.zeros(n)
            for i, c in enumerate(unique_cl):
                D_perm[cl == c] = cl_perm[i]
            perm_stats[b] = stat_fn(Y, D_perm)
    else:
        for b in range(n_perms):
            D_perm = rng.permutation(D)
            perm_stats[b] = stat_fn(Y, D_perm)

    # P-values
    p_two_sided = float(np.mean(np.abs(perm_stats) >= np.abs(obs_stat)))
    p_one_sided = float(np.mean(perm_stats >= obs_stat))

    return {
        "observed": obs_stat,
        "p_value": p_two_sided,
        "p_one_sided": p_one_sided,
        "n_perms": n_perms,
        "perm_distribution": perm_stats,
    }


# ======================================================================
# Helper functions
# ======================================================================


def _t_stat(y: np.ndarray, d: np.ndarray) -> float:
    """Two-sample t-statistic."""
    y1, y0 = y[d == 1], y[d == 0]
    n1, n0 = len(y1), len(y0)
    if n1 < 2 or n0 < 2:
        return 0.0
    mean_diff = np.mean(y1) - np.mean(y0)
    se = np.sqrt(np.var(y1, ddof=1) / n1 + np.var(y0, ddof=1) / n0)
    return mean_diff / se if se > 0 else 0.0


def _get_stat_fn(statistic: str) -> StatFn:
    """Return the test statistic function by name."""
    if statistic == "ate":

        def ate_stat(y: np.ndarray, d: np.ndarray) -> float:
            return float(np.mean(y[d == 1]) - np.mean(y[d == 0]))

        return ate_stat
    elif statistic == "ks":

        def ks_stat(y: np.ndarray, d: np.ndarray) -> float:
            return float(stats.ks_2samp(y[d == 1], y[d == 0]).statistic)

        return ks_stat
    elif statistic == "rank_sum":

        def rank_sum_stat(y: np.ndarray, d: np.ndarray) -> float:
            return float(stats.ranksums(y[d == 1], y[d == 0]).statistic)

        return rank_sum_stat
    else:
        raise ValueError(
            f"Unknown statistic: '{statistic}'. " f"Use 'ate', 'ks', or 'rank_sum'."
        )


def _make_cluster_permuter(
    df: pd.DataFrame,
    treatment: str,
    cluster: str,
    rng: np.random.Generator,
) -> Permuter:
    """Create a function that permutes treatment at the cluster level."""
    cl = df[cluster].values
    D = df[treatment].values.astype(float)
    unique_cl = np.unique(cl)
    n = len(D)
    # Get treatment per cluster (first obs)
    cl_treat = np.array([D[cl == c][0] for c in unique_cl])

    def permute() -> np.ndarray:
        cl_perm = rng.permutation(cl_treat)
        D_perm = np.zeros(n)
        for i, c in enumerate(unique_cl):
            D_perm[cl == c] = cl_perm[i]
        return D_perm

    return permute


def _make_stratified_permuter(
    df: pd.DataFrame,
    treatment: str,
    stratify: str,
    rng: np.random.Generator,
) -> Permuter:
    """Create a function that permutes treatment within strata."""
    D = df[treatment].values.astype(float)
    strata = df[stratify].values
    unique_strata = np.unique(strata)
    n = len(D)

    # Pre-compute indices for each stratum
    strata_indices = {s: np.where(strata == s)[0] for s in unique_strata}

    def permute() -> np.ndarray:
        D_perm = np.zeros(n)
        for s in unique_strata:
            idx = strata_indices[s]
            D_perm[idx] = rng.permutation(D[idx])
        return D_perm

    return permute


def _hodges_lehmann_ci(
    Y: np.ndarray,
    D: np.ndarray,
    stat_fn: StatFn,
    perm_fn: Permuter,
    n_perm: int,
    alpha: float,
    rng: np.random.Generator,
    n_grid: int = 101,
) -> Tuple[float, float]:
    """
    Hodges-Lehmann confidence interval by inverting the permutation test.

    For each candidate tau_0, test the sharp null Y_i(1) - Y_i(0) = tau_0
    by computing the permutation test on Y - tau_0 * D. The CI is the
    set of tau_0 values for which the test does not reject.

    Uses a coarse grid search followed by refinement.
    """
    # Observed ATE for centering the grid
    ate_obs = float(np.mean(Y[D == 1]) - np.mean(Y[D == 0]))
    y_std = float(np.std(Y)) if np.std(Y) > 0 else 1.0

    # Coarse grid: search over a wide range
    grid_half = max(3 * y_std, abs(ate_obs) * 3)
    tau_grid = np.linspace(ate_obs - grid_half, ate_obs + grid_half, n_grid)

    # Use fewer permutations for the grid search
    n_perm_grid = min(n_perm, 500)

    p_values = np.zeros(n_grid)
    for i, tau_0 in enumerate(tau_grid):
        # Adjust outcome: Y_adj = Y - tau_0 * D
        Y_adj = Y - tau_0 * D
        obs_adj = float(stat_fn(Y_adj, D))

        # Quick permutation test
        count = 0
        for b in range(n_perm_grid):
            D_perm = perm_fn()
            t_perm = stat_fn(Y_adj, D_perm)
            if abs(t_perm) >= abs(obs_adj):
                count += 1
        p_values[i] = count / n_perm_grid

    # CI = set of tau_0 where p >= alpha
    in_ci = tau_grid[p_values >= alpha]
    if len(in_ci) == 0:
        # If no values are in CI, return point estimate +/- small margin
        return (ate_obs, ate_obs)

    return (float(in_ci[0]), float(in_ci[-1]))


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS["randomization_inference"] = (
    "@article{young2019channeling,\n"
    "  title={Channeling Fisher: Randomization Tests and the Statistical "
    "Insignificance of Seemingly Significant Experimental Results},\n"
    "  author={Young, Alwyn},\n"
    "  journal={Quarterly Journal of Economics},\n"
    "  volume={134},\n"
    "  number={2},\n"
    "  pages={557--598},\n"
    "  year={2019},\n"
    "  publisher={Oxford University Press}\n"
    "}"
)
