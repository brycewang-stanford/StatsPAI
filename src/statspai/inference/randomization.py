"""
Randomization Inference (Fisher's Exact Test for causal effects).

Computes exact p-values by permuting treatment assignment and
comparing the observed test statistic to the permutation distribution.
Increasingly required by top journals (Young 2019, QJE).

References
----------
Fisher, R.A. (1935).
*The Design of Experiments*. Oliver and Boyd.

Young, A. (2019).
"Channeling Fisher: Randomization Tests and the Statistical
Insignificance of Seemingly Significant Experimental Results."
*Quarterly Journal of Economics*, 134(2), 557-598.
"""

from typing import Optional, List, Dict, Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def ri_test(
    data: pd.DataFrame,
    y: str,
    treat: str,
    stat: str = 'diff_means',
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
        - ``'diff_means'``: difference in means (Y̅₁ - Y̅₀)
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
    >>> result = sp.ri_test(df, y='outcome', treat='treatment',
    ...                     n_perms=5000, seed=42)
    >>> print(f"RI p-value: {result['p_value']:.4f}")

    Notes
    -----
    Under the sharp null H���: Y_i(1) = Y_i(0) for all i, the treatment
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
        df['_cluster'] = data[cluster].values
    df = df.dropna()

    Y = df[y].values.astype(float)
    D = df[treat].values.astype(float)
    n = len(Y)

    # Select test statistic function
    if callable(stat):
        stat_fn = stat
    elif stat == 'diff_means':
        stat_fn = lambda y, d: np.mean(y[d == 1]) - np.mean(y[d == 0])
    elif stat == 't':
        stat_fn = lambda y, d: _t_stat(y, d)
    elif stat == 'ks':
        stat_fn = lambda y, d: stats.ks_2samp(y[d == 1], y[d == 0]).statistic
    else:
        raise ValueError(f"Unknown stat: '{stat}'. Use 'diff_means', 't', 'ks', or callable.")

    # Observed statistic
    obs_stat = float(stat_fn(Y, D))

    # Permutation distribution
    perm_stats = np.zeros(n_perms)

    if cluster:
        # Cluster-level permutation
        cl = df['_cluster'].values
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
        'observed': obs_stat,
        'p_value': p_two_sided,
        'p_one_sided': p_one_sided,
        'n_perms': n_perms,
        'perm_distribution': perm_stats,
    }


def _t_stat(y, d):
    """Two-sample t-statistic."""
    y1, y0 = y[d == 1], y[d == 0]
    n1, n0 = len(y1), len(y0)
    if n1 < 2 or n0 < 2:
        return 0.0
    mean_diff = np.mean(y1) - np.mean(y0)
    se = np.sqrt(np.var(y1, ddof=1) / n1 + np.var(y0, ddof=1) / n0)
    return mean_diff / se if se > 0 else 0.0


# Citation
CausalResult._CITATIONS['randomization_inference'] = (
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
