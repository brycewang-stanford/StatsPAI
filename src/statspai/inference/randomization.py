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

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility

StatFn = Callable[[np.ndarray, np.ndarray], float]
Permuter = Callable[[], np.ndarray]

# Relative slack when counting permutations that tie with the observed
# statistic.  Ties count as extreme (the ri2 / randomizr convention), so the
# comparison has to survive round-off: a statistic that is mathematically a
# tie can come back one ULP below the observed value, and dropping it from
# the count moves the p-value by a full 1/n_perm.  ``statistic="ks"`` is the
# case that bites.  On n=12 the KS statistic takes six exact values j/6, but
# ``scipy.stats.ks_2samp`` builds them as a max over differences of two
# ECDFs: since SciPy 1.18 those six values arrive as eleven distinct floats.
# Under a strict ``>=`` the enumerated count over all 924 assignments falls
# from 438 to 384 and the two-sided p-value moves 0.4740 -> 0.4156, breaking
# parity with R's ri2 on a SciPy upgrade alone.  1e-12 is far below any
# difference an applied statistic can resolve and far above accumulated
# floating-point noise.
_TIE_RTOL = 1e-12


def _share_at_least(
    perm_stats: np.ndarray, obs_stat: float, *, two_sided: bool
) -> float:
    """Share of permutations at least as extreme as ``obs_stat``.

    Ties count as extreme, up to ``_TIE_RTOL`` scaled by the magnitude of the
    observed statistic (round-off is proportional to it).
    """
    perm = np.asarray(perm_stats, dtype=float)
    obs = float(obs_stat)
    if two_sided:
        perm = np.abs(perm)
        obs = abs(obs)
    tol = _TIE_RTOL * max(1.0, abs(obs))
    return float(np.mean(perm >= obs - tol))


# ======================================================================
# FisherResult class
# ======================================================================


class FisherResult(ResultProtocolMixin):
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


@accepts_aliases(covariates="controls", treat="treatment")
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

    Computes a permutation-based p-value under the sharp null hypothesis
    and, for the difference in means, the confidence interval for a
    constant effect obtained by inverting the test.

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
        - ``'t'``: studentized difference in means (unequal variances).
    controls : list of str, optional
        Control variables for covariate-adjusted inference.
        When provided, the test statistic is computed on residuals
        from regressing Y on controls.
    n_perm : int, default 10000
        Number of random permutations. When the design admits no more than
        ``n_perm`` distinct assignments (complete, cluster or stratified
        randomization) they are all enumerated instead, so the p-value is
        exact (the R ``ri2`` / ``randomizr`` rule), and ``n_perm`` on the
        result reports the number of assignments.
    stratify : str, optional
        Variable for stratified permutation (permute within strata).
    cluster : str, optional
        Variable for cluster-level randomization (permute by cluster).
    seed : int, optional
        Random seed for reproducibility.
    alpha : float, default 0.05
        Significance level for the test-inversion confidence interval.

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

    The confidence interval inverts the permutation test: it is the set of
    tau_0 for which the test of ``Y - tau_0 * D`` does not reject at level
    alpha. It is computed on the same assignments as the p-value (the
    difference in means is linear in the outcome, so no new draws are
    needed) and each end is found by bisection; with ``controls`` the
    shifted outcome is residualized, as the null requires. An end is
    infinite, with a warning, when no hypothesised effect is rejected,
    which happens when the design has too few assignments for ``alpha``.
    Through 1.38.0 the ends were read off a 101-point grid spanning six
    standard deviations of the outcome, each point tested on its own 500
    draws, and with ``controls`` the unresidualized ``tau_0 * D`` was
    subtracted from the residualized outcome.
    For ``statistic='t'``, ``'ks'`` and ``'rank_sum'`` the interval is the
    same inversion, in outcome units: the statistic is recomputed on
    ``Y - tau_0 * D`` for a stored set of assignments. Through 1.38.0 those
    statistics returned the 2.5% and 97.5% percentiles of the null
    distribution of the statistic, which is not an interval for the effect.

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

    # Covariate adjustment: residualize Y on controls. Under the sharp null
    # of a constant effect tau_0 the fixed quantity is Y - tau_0 * D, whose
    # residual is resid(Y) - tau_0 * resid(D); the interval needs the second
    # residual too.
    D_shift = D
    if controls:
        X_ctrl = df[controls].values.astype(float)
        X_ctrl = np.column_stack([np.ones(n), X_ctrl])
        beta_ctrl = np.linalg.lstsq(X_ctrl, Y, rcond=None)[0]
        Y = Y - X_ctrl @ beta_ctrl
        D_shift = D - X_ctrl @ np.linalg.lstsq(X_ctrl, D, rcond=None)[0]

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

    # Permutation distribution. When the design has no more distinct
    # assignments than ``n_perm`` they are enumerated (the ri2 / randomizr
    # rule), which makes the randomization distribution -- and the p-value --
    # exact rather than a Monte-Carlo estimate.
    assignments = _enumerate_assignments(
        D,
        clusters=df[cluster].values if cluster is not None else None,
        strata=(
            df[stratify].values if (stratify is not None and cluster is None) else None
        ),
        max_count=n_perm,
    )
    # ``shift_stats`` is the same difference in means applied to ``D_shift``:
    # with it the statistic of every assignment at any hypothesised effect is
    # ``perm_stats - tau_0 * shift_stats``, so the interval is inverted on
    # the very draws that gave the p-value.
    want_ci = statistic == "ate"
    # The other statistics are not linear in the outcome: their interval is
    # inverted on stored assignments (at most ``_MAX_DRAW_CELLS / n`` rows).
    draws: Optional[np.ndarray] = None
    if assignments is not None:
        perm_stats = np.array([stat_fn(Y, a) for a in assignments])
        shift_stats = (
            np.array([stat_fn(D_shift, a) for a in assignments]) if want_ci else None
        )
        if not want_ci:
            draws = np.asarray(assignments, dtype=float)
    else:
        perm_stats = np.zeros(n_perm)
        shift_stats = np.zeros(n_perm) if want_ci else None
        if not want_ci:
            draws = np.empty((min(n_perm, max(_MAX_DRAW_CELLS // n, 99)), n))
        for b in range(n_perm):
            D_perm = perm_fn()
            perm_stats[b] = stat_fn(Y, D_perm)
            if shift_stats is not None:
                shift_stats[b] = stat_fn(D_shift, D_perm)
            if draws is not None and b < draws.shape[0]:
                draws[b] = D_perm

    # P-values
    p_two_sided = _share_at_least(perm_stats, obs_stat, two_sided=True)
    p_one_sided = _share_at_least(perm_stats, obs_stat, two_sided=False)

    # Confidence interval by test inversion (only for ATE)
    if shift_stats is not None:
        ci = _invert_constant_effect(
            perm_stats, shift_stats, obs_stat, float(stat_fn(D_shift, D)), alpha
        )
        if not np.all(np.isfinite(ci)):
            warnings.warn(
                "fisher_exact: the randomization test cannot reject at level "
                f"{alpha:g} however large the hypothesised effect "
                f"({perm_stats.size} assignments; the smallest attainable "
                "two-sided p-value is above alpha), so the confidence "
                "interval is unbounded.",
                UserWarning,
                stacklevel=2,
            )
    else:
        # Same inversion for the statistics that are not linear in the
        # outcome, on the stored assignments.
        assert draws is not None
        ci = _invert_constant_effect_generic(
            Y, D, D_shift, draws[: min(draws.shape[0], n_perm)], statistic, alpha
        )

    _result = FisherResult(
        statistic=obs_stat,
        p_value=p_two_sided,
        p_one_sided=p_one_sided,
        ci=ci,
        perm_dist=perm_stats,
        statistic_type=statistic,
        n_perm=int(perm_stats.size),
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
    strata: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    absorb: Optional[Union[str, List[str]]] = None,
    interact: Optional[str] = None,
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
        Treatment. ``'diff_means'`` / ``'t'`` / ``'ks'`` need a binary 0/1
        indicator; ``'ols'`` / ``'ols_t'`` and callables also take a
        continuous one (an exposure or dose), whose observed values are
        permuted. Under ``cluster`` it must be constant within clusters
        (the permuted unit is the cluster); a treatment that switches on
        within a cluster, such as ``exposure x post``, is passed as
        ``treat='exposure', interact='post'``.
    stat : str or callable, default 'diff_means'
        Test statistic:
        - ``'diff_means'``: difference in means (Y_bar_1 - Y_bar_0)
        - ``'ks'``: Kolmogorov-Smirnov statistic
        - ``'t'``: the difference in means divided by its unequal-variance
          (Neyman) standard error. Unlike ``'diff_means'``, the test based
          on it also has the right size in large samples for the weak null
          of a zero *average* effect when effects vary across units
        - ``'rank_sum'``: Wilcoxon rank sum of the treated outcomes
          (mid-ranks), centred at its null mean ``n_1 (n + 1) / 2``
        - ``'lin'``: coefficient on the treatment in the regression of
          ``y`` on the treatment, the centred ``covariates`` and their
          interactions with the treatment (:func:`sp.lm_lin`)
        - ``'lin_t'``: that coefficient divided by its HC2 standard error,
          the covariate-adjusted statistic with the same large-sample
          guarantee as ``'t'``
        - ``'ols'``: coefficient on the treatment in the OLS regression of
          ``y`` on the treatment, ``covariates`` and (with ``strata``)
          stratum fixed effects -- the regression-adjusted statistic of
          stratified experiments (R ``ri2::conduct_ri(y ~ Z + x + block)``)
        - ``'ols_t'``: its t-statistic (HC1, or CR1 by ``cluster``)
        - A callable ``f(Y, D) -> float`` for custom statistics.
    n_perms : int, default 1000
        Number of random permutations. Use 10000+ for publications. When
        the design admits no more than ``n_perms`` distinct assignments
        (``C(n, n_1)``, or ``C(G, G_1)`` under ``cluster``) every assignment
        is enumerated instead and the p-value is exact -- the rule R
        ``ri2`` / ``randomizr::obtain_permutation_matrix`` apply.
    cluster : str, optional
        Cluster-level permutation (permute treatment at cluster level).
    seed : int, optional
        Random seed.
    alpha : float, default 0.05
    strata : str, optional
        Randomization strata (blocks): treatment is re-randomized *within*
        each stratum, keeping its number of treated units (or clusters,
        with ``cluster``). Permuting across strata when the experiment
        assigned within them tests the wrong null distribution and loses
        power (UCT, QJE 2016: education p 0.70 unstratified against 0.17).
    covariates : list of str, optional
        Controls for ``stat='ols'`` / ``'ols_t'`` / ``'lin'`` / ``'lin_t'``.
    absorb : str or list of str, optional
        Fixed effects for ``stat='ols'`` / ``'ols_t'``, as ``reghdfe``'s
        ``absorb()`` (column names, ``"a^b"`` for combinations; singletons
        dropped). They are swept out by the ``sp.hdfe_ols`` absorber
        (Frisch-Waugh-Lovell). Under ``cluster`` the regressor of any
        permutation is ``R v`` with ``R`` the swept cluster indicators
        (times ``interact``) and ``v`` the permuted cluster values, so only
        ``G`` columns are swept once and each permutation is a small matrix
        product -- the statistic equals the ``sp.hdfe_ols`` coefficient
        (or its cluster t) refitted on the permuted treatment. Without
        ``cluster`` every permutation re-sweeps the regressor.
    interact : str, optional
        For ``stat='ols'`` / ``'ols_t'``: the regressor is
        ``treat * data[interact]`` while ``treat`` is what is permuted --
        the intensity DiD ``exposure x post`` with exposure permuted across
        clusters.

    Returns
    -------
    dict
        ``'observed'``: observed test statistic
        ``'p_value'``: two-sided randomization p-value
        ``'p_one_sided'``: one-sided (greater) p-value
        ``'n_perms'``: number of permutations used (the number of distinct
        assignments when they were enumerated)
        ``'exact'``: True when every assignment of the design was enumerated
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
    ['exact', 'n_perms', 'observed', 'p_one_sided', 'p_value', 'perm_distribution']
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

    covariates = list(covariates or [])
    fe_terms = [absorb] if isinstance(absorb, str) else list(absorb or [])
    fe_cols = [c for t in fe_terms for c in t.split("^")]
    regress_stat = isinstance(stat, str) and stat in ("ols", "ols_t")
    if (fe_terms or interact) and not regress_stat:
        raise MethodIncompatibility(
            "absorb= and interact= are used by stat='ols' / 'ols_t' only.",
            recovery_hint="Pass stat='ols' (coefficient) or 'ols_t' (its t).",
        )
    extra = list(dict.fromkeys(fe_cols + ([interact] if interact else [])))
    extra = [c for c in extra if c not in (y, treat, *covariates)]
    df = data[[y, treat] + covariates + extra].copy()
    if cluster:
        df["_cluster"] = data[cluster].values
    if strata:
        df["_strata"] = data[strata].values
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped:
        cols = [y, treat] + ([cluster] if cluster else [])
        na_counts = {c: int(data[c].isna().sum()) for c in cols if data[c].isna().any()}
        warnings.warn(
            f"ri_test: dropped {n_dropped} of {n_before} rows with NaN in an "
            f"estimation column (NaN counts by column: {na_counts}). The "
            "randomization distribution is built on the remaining rows, so "
            "the observed statistic refers to that subsample — passing "
            "cluster= can shrink the sample further when the cluster id is "
            "itself missing.",
            UserWarning,
            stacklevel=2,
        )

    if fe_terms:
        from ..panel.hdfe import Absorber, _factorize_multi

        fe_mat = np.column_stack(
            [
                _factorize_multi([df[c].to_numpy() for c in t.split("^")])[0]
                for t in fe_terms
            ]
            + ([pd.factorize(df["_strata"])[0]] if strata else [])
        )
        absorber = Absorber(fe_mat, drop_singletons=True, tol=1e-8)
        df = df.iloc[np.flatnonzero(absorber.keep_mask)]

    Y = df[y].values.astype(float)
    D = df[treat].values.astype(float)
    n = len(Y)
    st = df["_strata"].values if strata else None
    binary = bool(np.all(np.isin(D, (0.0, 1.0))))
    if (
        not binary
        and isinstance(stat, str)
        and stat in ("diff_means", "t", "ks", "rank_sum", "lin", "lin_t")
    ):
        raise MethodIncompatibility(
            f"stat={stat!r} compares treated and control units and needs a "
            f"binary 0/1 treatment; {treat!r} takes other values.",
            recovery_hint="For a continuous treatment use stat='ols' / 'ols_t' "
            "(with absorb= for fixed effects) or a callable.",
            diagnostics={"n_distinct": int(np.unique(D).size)},
        )
    if cluster:
        cl_codes_chk = pd.factorize(df["_cluster"])[0]
        first = np.zeros(int(cl_codes_chk.max()) + 1)
        first[cl_codes_chk[::-1]] = D[::-1]
        if not np.allclose(D, first[cl_codes_chk], rtol=0, atol=0):
            raise MethodIncompatibility(
                f"{treat!r} varies within {cluster!r} clusters, but the "
                "permutation assigns one value per cluster.",
                recovery_hint="Pass the cluster-level variable as treat= and the "
                "within-cluster pattern as interact= (e.g. treat='exposure', "
                "interact='post'), or permute at the row level (no cluster=).",
                diagnostics={"cluster": cluster},
            )
    lin_stat = isinstance(stat, str) and stat in ("lin", "lin_t")
    if covariates and not (regress_stat or lin_stat):
        raise MethodIncompatibility(
            "covariates= is used by stat='ols' / 'ols_t' / 'lin' / 'lin_t' only.",
            recovery_hint="Pass one of those statistics with covariates.",
        )
    if lin_stat and not covariates:
        raise MethodIncompatibility(
            f"stat={stat!r} needs covariates=.",
            recovery_hint="Without covariates use stat='diff_means' or 't'.",
        )
    if lin_stat and cluster:
        raise MethodIncompatibility(
            f"stat={stat!r} uses the HC2 variance of independently assigned "
            "units and is not available with cluster=.",
            recovery_hint="Use stat='ols_t' (CR1) with cluster=.",
        )

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
    elif stat == "rank_sum":
        stat_fn = _rank_sum_stat
    elif stat in ("lin", "lin_t"):
        stat_fn = _lin_stat_factory(
            df[covariates].to_numpy(dtype=float), t_stat=stat == "lin_t"
        )
    elif stat in ("ols", "ols_t") and (fe_terms or interact):
        stat_fn = _absorbed_ols_stat_factory(
            df,
            y,
            covariates,
            absorber if fe_terms else None,
            df[interact].to_numpy(dtype=float) if interact else None,
            df["_cluster"].values if cluster else None,
            None if fe_terms else st,
            t_stat=stat == "ols_t",
        )
    elif stat in ("ols", "ols_t"):
        stat_fn = _ols_stat_factory(
            df,
            covariates,
            st,
            df["_cluster"].values if cluster else None,
            t_stat=stat == "ols_t",
        )
    else:
        raise ValueError(
            f"Unknown stat: '{stat}'. Use 'diff_means', 't', 'ks', 'rank_sum', "
            "'lin', 'lin_t', 'ols', 'ols_t' or a callable."
        )

    # Observed statistic
    obs_stat = float(stat_fn(Y, D))

    # Permutation distribution: exact enumeration when the design has no more
    # distinct assignments than ``n_perms`` (the ri2 / randomizr rule),
    # otherwise ``n_perms`` random re-randomizations.
    cl = df["_cluster"].values if cluster else None
    assignments = _enumerate_assignments(D, clusters=cl, strata=st, max_count=n_perms)
    if assignments is not None:
        perm_stats = np.array([stat_fn(Y, a) for a in assignments])
    elif cluster:
        perm_stats = np.zeros(n_perms)
        # Cluster-level permutation (within strata when given)
        unique_cl = np.unique(cl)
        # Get treatment per cluster (first obs)
        cl_treat = np.array([D[cl == c][0] for c in unique_cl])
        cl_rows = [np.flatnonzero(cl == c) for c in unique_cl]
        if st is not None:
            cl_st = np.array([st[r[0]] for r in cl_rows])
            groups = [np.flatnonzero(cl_st == s_) for s_ in np.unique(cl_st)]
        else:
            groups = [np.arange(unique_cl.size)]
        for b in range(n_perms):
            cl_perm = cl_treat.copy()
            for gidx in groups:
                cl_perm[gidx] = rng.permutation(cl_treat[gidx])
            on_values = getattr(stat_fn, "on_cluster_values", None)
            if on_values is not None:
                # the statistic needs only the cluster-level values (in
                # np.unique order); skip expanding them to rows
                perm_stats[b] = on_values(cl_perm)
                continue
            D_perm = np.zeros(n)
            for i, rows in enumerate(cl_rows):
                D_perm[rows] = cl_perm[i]
            perm_stats[b] = stat_fn(Y, D_perm)
    else:
        perm_stats = np.zeros(n_perms)
        groups = (
            [np.flatnonzero(st == s_) for s_ in np.unique(st)]
            if st is not None
            else [np.arange(n)]
        )
        for b in range(n_perms):
            D_perm = D.copy()
            for gidx in groups:
                D_perm[gidx] = rng.permutation(D[gidx])
            perm_stats[b] = stat_fn(Y, D_perm)

    # P-values (ties with the observed statistic count as extreme; the
    # enumerated set contains the observed assignment itself)
    p_two_sided = _share_at_least(perm_stats, obs_stat, two_sided=True)
    p_one_sided = _share_at_least(perm_stats, obs_stat, two_sided=False)

    return {
        "observed": obs_stat,
        "p_value": p_two_sided,
        "p_one_sided": p_one_sided,
        "n_perms": int(perm_stats.size),
        "exact": assignments is not None,
        "perm_distribution": perm_stats,
    }


# ======================================================================
# Helper functions
# ======================================================================


def _ols_stat_factory(
    df: pd.DataFrame,
    covariates: List[str],
    strata: Optional[np.ndarray],
    clusters: Optional[np.ndarray],
    t_stat: bool,
) -> StatFn:
    """Treatment coefficient (or its t) after partialling out the controls.

    By Frisch-Waugh-Lovell only ``M_W d`` changes across permutations, so the
    projection onto the controls ``W`` (intercept, covariates, stratum
    dummies) is factored once.
    """
    n = len(df)
    parts = [np.ones((n, 1))]
    if covariates:
        parts.append(df[covariates].to_numpy(dtype=float))
    if strata is not None:
        parts.append(
            pd.get_dummies(pd.Series(strata), drop_first=True, dtype=float).to_numpy()
        )
    Wm = np.column_stack(parts)
    Q, _ = np.linalg.qr(Wm)
    k = int(np.linalg.matrix_rank(Wm)) + 1
    cl_codes = None if clusters is None else pd.factorize(clusters)[0]

    def fn(y_: np.ndarray, d_: np.ndarray) -> float:
        dt = d_ - Q @ (Q.T @ d_)
        yt = y_ - Q @ (Q.T @ y_)
        dd = float(dt @ dt)
        if dd <= 0:
            return 0.0
        b = float(dt @ yt) / dd
        if not t_stat:
            return b
        e = yt - b * dt
        if cl_codes is None:
            var = (n / (n - k)) * float(np.sum((dt * e) ** 2)) / dd**2
        else:
            G = int(cl_codes.max()) + 1
            sc = np.bincount(cl_codes, weights=dt * e, minlength=G)
            var = (G / (G - 1)) * ((n - 1) / (n - k)) * float(sc @ sc) / dd**2
        return b / np.sqrt(var) if var > 0 else 0.0

    return fn


def _absorbed_ols_stat_factory(
    df: pd.DataFrame,
    y: str,
    covariates: List[str],
    absorber: Any,
    interact: Optional[np.ndarray],
    clusters: Optional[np.ndarray],
    strata: Optional[np.ndarray],
    t_stat: bool,
) -> StatFn:
    """``ols`` / ``ols_t`` with absorbed fixed effects and an interaction.

    The regressor is ``M (d * s)`` with ``M`` the FE sweep (then the
    covariates partialled out) and ``s`` the interaction (or 1). Under
    cluster permutation ``d`` is ``v[cluster]``, so the regressor is
    ``R v`` with ``R = M [1_g * s]``: ``G`` columns are swept once, and
    ``b = v'a / v'Bv`` with ``a = R'y~``, ``B = R'R``; the cluster scores are
    ``A_g v - b v'B_g v`` from per-cluster blocks. Without clusters each call
    sweeps its regressor.
    """
    n = len(df)
    s = np.ones(n) if interact is None else interact

    def sweep(A: np.ndarray) -> np.ndarray:
        return absorber.demean(A, copy=True, already_masked=True) if absorber else A

    parts = [] if absorber is not None else [np.ones((n, 1))]
    if covariates:
        parts.append(df[covariates].to_numpy(dtype=float))
    if strata is not None:
        parts.append(
            pd.get_dummies(pd.Series(strata), drop_first=True, dtype=float).to_numpy()
        )
    W = (
        sweep(np.column_stack(parts))
        if parts and absorber is not None
        else (np.column_stack(parts) if parts else np.empty((n, 0)))
    )
    if W.shape[1]:
        Q, _ = np.linalg.qr(W)
    else:
        Q = np.empty((n, 0))

    def resid(A: np.ndarray) -> np.ndarray:
        return A - Q @ (Q.T @ A) if Q.shape[1] else A

    k = Q.shape[1] + 1
    # sorted codes: the permutation loop hands over values in np.unique order
    cl_codes = None if clusters is None else np.unique(clusters, return_inverse=True)[1]
    y_raw = df[y].to_numpy(dtype=float)
    key = None
    if cl_codes is not None:
        key = _sweep_key(y_raw, s, absorber, cl_codes, parts)
        hit = _SWEEP_CACHE.get(key)
        if hit is not None:
            return _cluster_stat(*hit, t_stat=t_stat)
    yt = resid(sweep(y_raw))

    def finish(dt: np.ndarray) -> float:
        dd = float(dt @ dt)
        if dd <= 0:
            return 0.0
        b = float(dt @ yt) / dd
        if not t_stat:
            return b
        e = yt - b * dt
        if cl_codes is None:
            var = (n / (n - k)) * float(np.sum((dt * e) ** 2)) / dd**2
        else:
            G = int(cl_codes.max()) + 1
            sc = np.bincount(cl_codes, weights=dt * e, minlength=G)
            var = (G / (G - 1)) * ((n - 1) / (n - k)) * float(sc @ sc) / dd**2
        return b / np.sqrt(var) if var > 0 else 0.0

    if cl_codes is None:
        return lambda y_, d_: finish(resid(sweep(np.asarray(d_, float) * s)))

    G = int(cl_codes.max()) + 1
    first_row = np.full(G, -1)
    first_row[cl_codes[::-1]] = np.arange(n)[::-1]
    R = np.zeros((n, G))
    R[np.arange(n), cl_codes] = s
    R = resid(sweep(R))
    a = R.T @ yt
    B = R.T @ R
    if G > 400:  # per-cluster G x G blocks would not fit; sweep-free matvec
        return lambda y_, d_: finish(R @ np.asarray(d_, float)[first_row])
    order = np.argsort(cl_codes, kind="stable")
    bounds = np.searchsorted(cl_codes[order], np.arange(G + 1))
    Ag = np.zeros((G, G))
    Bg = np.zeros((G, G, G))
    for g in range(G):
        rows = order[bounds[g] : bounds[g + 1]]
        rg = R[rows]
        Ag[g] = rg.T @ yt[rows]
        Bg[g] = rg.T @ rg
    blocks = (a, B, Ag, Bg, first_row, n, k)
    _SWEEP_CACHE.clear()  # one entry: the ols / ols_t pair on the same design
    _SWEEP_CACHE[key] = blocks
    return _cluster_stat(*blocks, t_stat=t_stat)


#: The swept cluster blocks of the last design, so ``stat='ols'`` followed by
#: ``stat='ols_t'`` on the same data sweeps the ``G`` columns once.
_SWEEP_CACHE: Dict[Any, tuple] = {}


def _sweep_key(y: Any, s: Any, absorber: Any, cl_codes: Any, parts: Any) -> str:
    """Fingerprint of everything the swept cluster blocks depend on."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    arrays = [y, s, cl_codes] + list(parts)
    if absorber is not None:
        arrays += list(absorber.fe_codes)
        h.update(repr((absorber.tol, absorber.n_kept)).encode())
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        h.update(repr((arr.dtype.str, arr.shape)).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _cluster_stat(
    a: Any, B: Any, Ag: Any, Bg: Any, first_row: Any, n: Any, k: Any, *, t_stat: bool
) -> StatFn:
    """The statistic from the swept cluster blocks, for row- or cluster-level input."""
    G = B.shape[0]

    def on_values(v: np.ndarray) -> float:
        v = np.asarray(v, float)
        den = float(v @ B @ v)
        if den <= 0:
            return 0.0
        b = float(v @ a) / den
        if not t_stat:
            return b
        sc = Ag @ v - b * np.einsum("i,gij,j->g", v, Bg, v)
        var = (G / (G - 1)) * ((n - 1) / (n - k)) * float(sc @ sc) / den**2
        return b / np.sqrt(var) if var > 0 else 0.0

    def fast(y_: np.ndarray, d_: np.ndarray) -> float:
        return on_values(np.asarray(d_, float)[first_row])

    fast.on_cluster_values = on_values  # type: ignore[attr-defined]
    return fast


def _t_stat(y: np.ndarray, d: np.ndarray) -> float:
    """Two-sample t-statistic."""
    y1, y0 = y[d == 1], y[d == 0]
    n1, n0 = len(y1), len(y0)
    if n1 < 2 or n0 < 2:
        return 0.0
    mean_diff = np.mean(y1) - np.mean(y0)
    se = np.sqrt(np.var(y1, ddof=1) / n1 + np.var(y0, ddof=1) / n0)
    return mean_diff / se if se > 0 else 0.0


def _rank_sum_stat(y: np.ndarray, d: np.ndarray) -> float:
    """Wilcoxon rank sum of the treated (mid-ranks) minus its null mean."""
    ranks = stats.rankdata(y)
    n1 = float(np.sum(d == 1))
    return float(np.sum(ranks[d == 1]) - n1 * (len(y) + 1) / 2.0)


def _lin_stat_factory(X: np.ndarray, t_stat: bool) -> StatFn:
    """Lin's (2013) interacted-regression estimate, or its HC2 t-ratio."""
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    ones = np.ones((n, 1))

    def fn(y_: np.ndarray, d_: np.ndarray) -> float:
        design = np.column_stack([ones, d_, Xc, Xc * d_[:, None]])
        gram_inv = np.linalg.pinv(design.T @ design)
        beta = gram_inv @ (design.T @ y_)
        if not t_stat:
            return float(beta[1])
        resid = y_ - design @ beta
        lev = np.einsum("ij,jk,ik->i", design, gram_inv, design)
        w = resid**2 / np.clip(1.0 - lev, 1e-12, None)
        var = float((gram_inv @ ((design.T * w) @ design) @ gram_inv)[1, 1])
        return float(beta[1] / np.sqrt(var)) if var > 0 else 0.0

    return fn


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
    elif statistic == "t":
        return _t_stat
    else:
        raise ValueError(
            f"Unknown statistic: '{statistic}'. Use 'ate', 't', 'ks', or 'rank_sum'."
        )


def _enumerate_assignments(
    D: np.ndarray,
    clusters: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
    max_count: int = 10000,
) -> Optional[np.ndarray]:
    """Every treatment vector the design could have produced, or ``None``.

    Complete randomization of ``n_1`` treated among ``n`` units (optionally
    within ``strata``, keeping each stratum's treated count), or of treated
    *clusters* when ``clusters`` is given (units inherit their cluster's
    assignment). Returns an array of shape ``(N_assign, n)`` when
    ``N_assign <= max_count`` -- the rule R ``randomizr::
    obtain_permutation_matrix`` uses to switch from sampling to exact
    enumeration -- and ``None`` otherwise. The observed assignment is one of
    the rows.
    """
    from itertools import combinations, product
    from math import comb, prod

    D = np.asarray(D, dtype=float)
    if not np.all(np.isin(D, (0.0, 1.0))):
        return None
    if clusters is not None:
        _, inv = np.unique(clusters, return_inverse=True)
        n_units = int(inv.max()) + 1
        unit_D = np.array([D[inv == g][0] for g in range(n_units)])
        if not all(np.all(D[inv == g] == unit_D[g]) for g in range(n_units)):
            return None  # treatment varies within a cluster
        if strata is not None:
            cl_strata = np.array([strata[inv == g][0] for g in range(n_units)])
            blocks = [np.where(cl_strata == s_)[0] for s_ in np.unique(cl_strata)]
        else:
            blocks = [np.arange(n_units)]
    else:
        inv = None
        unit_D = D
        if strata is not None:
            blocks = [np.where(strata == s_)[0] for s_ in np.unique(strata)]
        else:
            blocks = [np.arange(D.size)]
    m = [int(unit_D[b].sum()) for b in blocks]
    total = prod(comb(len(b), k) for b, k in zip(blocks, m))
    if total > max_count:
        return None
    per_block = [
        [b[list(c)] for c in combinations(range(len(b)), k)] for b, k in zip(blocks, m)
    ]
    out = np.zeros((total, unit_D.size))
    for r, choice in enumerate(product(*per_block)):
        for idx in choice:
            out[r, idx] = 1.0
    if inv is not None:
        out = out[:, inv]
    return out


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


def _invert_constant_effect(
    perm_stats: np.ndarray,
    shift_stats: np.ndarray,
    obs_stat: float,
    obs_shift: float,
    alpha: float,
) -> Tuple[float, float]:
    """Confidence interval for a constant effect by inverting the test.

    The sharp null ``Y_i(1) - Y_i(0) = tau_0`` is tested on ``Y - tau_0 D``.
    The difference in means is linear in the outcome, so under assignment
    ``b`` it equals ``perm_stats[b] - tau_0 * shift_stats[b]`` and the
    observed one ``obs_stat - tau_0 * obs_shift``: the p-value at any
    ``tau_0`` comes from the stored draws, with no new permutations. Each
    end of the interval is bracketed by stepping away from the point
    estimate and then bisected, so it is exact for the assignments used
    (all of them when the design was enumerated) up to floating point.
    An end is infinite when the test never rejects in that direction.
    """
    perm = np.asarray(perm_stats, dtype=float)
    shift = np.asarray(shift_stats, dtype=float)

    def pval(tau0: float) -> float:
        return _share_at_least(
            perm - tau0 * shift, obs_stat - tau0 * obs_shift, two_sided=True
        )

    if obs_shift == 0 or not np.isfinite(obs_stat):
        return (float("nan"), float("nan"))
    center = obs_stat / obs_shift  # the observed statistic is zero here
    scale = float(np.std(perm)) / abs(obs_shift)
    if not np.isfinite(scale) or scale <= 0:
        scale = max(abs(center), 1.0)

    def end(sign: float) -> float:
        inside, step = center, scale
        for _ in range(200):
            outside = inside + sign * step
            if pval(outside) < alpha:
                break
            inside, step = outside, step * 2.0
        else:
            return sign * float("inf")
        for _ in range(200):
            mid = 0.5 * (inside + outside)
            if mid == inside or mid == outside:
                break
            if pval(mid) < alpha:
                outside = mid
            else:
                inside = mid
        return float(inside)

    return (end(-1.0), end(1.0))


#: Cap on ``rows * n`` of the assignments stored to invert a statistic that
#: is not linear in the outcome.
_MAX_DRAW_CELLS = 20_000_000


def _stat_all(A: np.ndarray, y: np.ndarray, statistic: str) -> np.ndarray:
    """``statistic`` of ``y`` under every assignment (row) of ``A``.

    Vectorised versions of the functions :func:`_get_stat_fn` returns, with
    the same definitions (``scipy.stats.ranksums`` and ``ks_2samp``
    statistics, the Welch ``t``).
    """
    n = A.shape[1]
    n1 = A.sum(axis=1)
    n0 = n - n1
    s1 = A @ y
    s0 = y.sum() - s1
    if statistic == "ate":
        out: np.ndarray = s1 / n1 - s0 / n0
        return out
    if statistic == "t":
        q1 = A @ (y * y)
        q0 = float(y @ y) - q1
        with np.errstate(divide="ignore", invalid="ignore"):
            v1 = (q1 - s1 * s1 / n1) / (n1 - 1.0)
            v0 = (q0 - s0 * s0 / n0) / (n0 - 1.0)
            se = np.sqrt(np.clip(v1, 0.0, None) / n1 + np.clip(v0, 0.0, None) / n0)
            out = (s1 / n1 - s0 / n0) / se
        out = np.where((n1 < 2) | (n0 < 2) | ~(se > 0), 0.0, out)
        return out
    if statistic == "rank_sum":
        ranks = stats.rankdata(y)
        out = (A @ ranks - n1 * (n + 1) / 2.0) / np.sqrt(n0 * n1 * (n + 1) / 12.0)
        return out
    if statistic == "ks":
        order = np.argsort(y, kind="stable")
        ys = y[order]
        As = A[:, order]
        gap = np.cumsum(As, axis=1) / n1[:, None] - np.cumsum(1.0 - As, axis=1) / (
            n0[:, None]
        )
        # The ECDFs are compared at distinct outcome values only.
        last = np.r_[ys[1:] != ys[:-1], True]
        out = np.abs(gap[:, last]).max(axis=1)
        return out
    raise MethodIncompatibility(f"Unknown statistic: {statistic!r}.")


def _invert_constant_effect_generic(
    Y: np.ndarray,
    D: np.ndarray,
    D_shift: np.ndarray,
    draws: np.ndarray,
    statistic: str,
    alpha: float,
) -> Tuple[float, float]:
    """:func:`_invert_constant_effect` for a statistic that is not linear.

    The test of ``tau_0`` is the randomization test applied to
    ``Y - tau_0 * D_shift``, with any ``statistic``; the interval is in
    outcome units. The statistic has to be recomputed at every ``tau_0``,
    which is done for all stored assignments at once. Ends are bracketed
    from the difference-in-means estimate and bisected; an end is infinite
    when the test never rejects in that direction, and the interval is
    ``(nan, nan)`` when the estimate itself is rejected.
    """
    obs_row = D[None, :]

    def pval(tau0: float) -> float:
        y_adj = Y - tau0 * D_shift
        return _share_at_least(
            _stat_all(draws, y_adj, statistic),
            float(_stat_all(obs_row, y_adj, statistic)[0]),
            two_sided=True,
        )

    shift = float(_stat_all(obs_row, D_shift, "ate")[0])
    if shift == 0:
        return (float("nan"), float("nan"))
    center = float(_stat_all(obs_row, Y, "ate")[0]) / shift
    if pval(center) < alpha:
        return (float("nan"), float("nan"))
    scale = float(np.std(_stat_all(draws, Y, "ate"))) / abs(shift)
    if not np.isfinite(scale) or scale <= 0:
        scale = max(abs(center), 1.0)

    def end(sign: float) -> float:
        inside, step = center, scale
        for _ in range(60):
            outside = inside + sign * step
            if pval(outside) < alpha:
                break
            inside, step = outside, step * 2.0
        else:
            return sign * float("inf")
        for _ in range(50):
            mid = 0.5 * (inside + outside)
            if mid == inside or mid == outside:
                break
            if pval(mid) < alpha:
                outside = mid
            else:
                inside = mid
        return float(inside)

    return (end(-1.0), end(1.0))


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
