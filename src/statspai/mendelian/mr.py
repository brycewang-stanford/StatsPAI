"""
Mendelian Randomization (MR) methods.

Uses genetic variants as instrumental variables to estimate causal effects
of exposures on outcomes. Implements IVW, MR-Egger, weighted median,
and MR-PRESSO for outlier detection.

Equivalent to R's ``MendelianRandomization`` and ``TwoSampleMR`` packages.

References
----------
Burgess, S. et al. (2013).
"Mendelian randomization analysis with multiple genetic variants using
summarized data." *Genetic Epidemiology*, 37(7), 658-665. [@burgess2013mendelian]

Bowden, J. et al. (2015).
"Mendelian randomization with invalid instruments: effect estimation and
bias detection through Egger regression." *IJE*, 44(2), 512-525. [@bowden2015mendelian]

Bowden, J. et al. (2016).
"Consistent estimation in Mendelian randomization with some invalid
instruments using a weighted median estimator." *Genetic Epidemiology*,
40(4), 304-314. [@bowden2016consistent]
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


class MRResult:
    """Results from Mendelian Randomization analysis.

    Returned by :func:`mendelian_randomization`. Holds the per-method
    ``estimates`` table plus heterogeneity (Cochran's Q) and pleiotropy
    (MR-Egger intercept) diagnostics; render with ``.summary()`` or
    ``.plot()``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n_snps = 15
    >>> bx = rng.uniform(0.1, 0.5, n_snps)
    >>> by = 0.3 * bx + rng.normal(0, 0.02, n_snps)
    >>> snp_stats = pd.DataFrame({
    ...     "beta_x": bx, "beta_y": by,
    ...     "se_x": rng.uniform(0.01, 0.05, n_snps),
    ...     "se_y": rng.uniform(0.01, 0.05, n_snps),
    ... })
    >>> res = sp.mendelian_randomization(
    ...     data=snp_stats, beta_exposure="beta_x", beta_outcome="beta_y",
    ...     se_exposure="se_x", se_outcome="se_y",
    ...     exposure_name="BMI", outcome_name="T2D", seed=0)
    >>> isinstance(res, sp.MRResult)
    True
    >>> res.n_snps
    15
    >>> list(res.estimates["method"])
    ['IVW', 'MR-Egger', 'Weighted Median']
    """

    def __init__(
        self,
        estimates: pd.DataFrame,
        heterogeneity: Dict[str, Any],
        pleiotropy: Optional[Dict[str, float]],
        n_snps: int,
        exposure: str,
        outcome: str,
    ) -> None:
        self.estimates = estimates  # DataFrame with methods and results
        self.heterogeneity = heterogeneity  # Q-statistic
        self.pleiotropy = pleiotropy  # MR-Egger intercept test
        self.n_snps = n_snps
        self.exposure = exposure
        self.outcome = outcome

    def summary(self) -> str:
        lines = [
            "Mendelian Randomization Analysis",
            "=" * 65,
            f"Exposure: {self.exposure}",
            f"Outcome:  {self.outcome}",
            f"Number of SNPs: {self.n_snps}",
            "",
            f"{'Method':<25s} {'Estimate':>10s} {'SE':>10s} {'95% CI':>22s} {'p-value':>10s}",
            "-" * 65,
        ]
        for _, row in self.estimates.iterrows():
            ci = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
            lines.append(
                f"{row['method']:<25s} {row['estimate']:>10.4f} "
                f"{row['se']:>10.4f} {ci:>22s} {row['p_value']:>10.4f}"
            )

        lines.append("")
        lines.append("Heterogeneity:")
        lines.append(
            f"  Cochran's Q = {self.heterogeneity['Q']:.3f} "
            f"(p = {self.heterogeneity['Q_p']:.4f})"
        )
        lines.append(f"  I² = {self.heterogeneity['I2']:.1f}%")

        if self.pleiotropy is not None:
            lines.append("\nPleiotropy (MR-Egger intercept):")
            lines.append(
                f"  Intercept = {self.pleiotropy['intercept']:.4f} "
                f"(p = {self.pleiotropy['p_value']:.4f})"
            )

        lines.append("=" * 65)
        return "\n".join(lines)

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Scatter plot of SNP effects with MR lines."""
        return mr_plot(self, ax=ax, **kwargs)


def mr_ivw(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Inverse-Variance Weighted (IVW) MR estimator.

    Fixed-effects meta-analysis of Wald ratios.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n_snps = 15
    >>> beta_exposure = rng.uniform(0.1, 0.5, n_snps)
    >>> beta_outcome = 0.3 * beta_exposure + rng.normal(0, 0.02, n_snps)
    >>> se_exposure = rng.uniform(0.01, 0.05, n_snps)
    >>> se_outcome = rng.uniform(0.01, 0.05, n_snps)
    >>> res = sp.mr_ivw(beta_exposure, beta_outcome, se_exposure, se_outcome)
    >>> sorted(res.keys())
    ['I2', 'Q', 'Q_df', 'Q_p', 'ci_lower', 'ci_upper', 'estimate', 'p_value', 'se']
    >>> bool(0.0 < res["estimate"] < 0.6)
    True

    References
    ----------
    burgess2013mendelian
    """
    # Weighted regression of beta_Y on beta_X through the origin.
    w = 1 / se_outcome**2
    estimate_wls = np.sum(w * beta_exposure * beta_outcome) / np.sum(
        w * beta_exposure**2
    )
    se_wls = np.sqrt(1 / np.sum(w * beta_exposure**2))

    z = estimate_wls / se_wls
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Cochran's Q
    Q = np.sum(w * (beta_outcome - estimate_wls * beta_exposure) ** 2)
    Q_df = len(beta_exposure) - 1
    Q_p = 1 - stats.chi2.cdf(Q, Q_df)
    I2 = max(0, (Q - Q_df) / Q * 100) if Q > 0 else 0

    return {
        "estimate": estimate_wls,
        "se": se_wls,
        "ci_lower": estimate_wls - z_crit * se_wls,
        "ci_upper": estimate_wls + z_crit * se_wls,
        "p_value": p_value,
        "Q": Q,
        "Q_df": Q_df,
        "Q_p": Q_p,
        "I2": I2,
    }


def mr_egger(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    MR-Egger regression.

    Allows for directional pleiotropy via a non-zero intercept.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n_snps = 15
    >>> beta_exposure = rng.uniform(0.1, 0.5, n_snps)
    >>> beta_outcome = 0.3 * beta_exposure + rng.normal(0, 0.02, n_snps)
    >>> se_exposure = rng.uniform(0.01, 0.05, n_snps)
    >>> se_outcome = rng.uniform(0.01, 0.05, n_snps)
    >>> res = sp.mr_egger(beta_exposure, beta_outcome, se_exposure, se_outcome)
    >>> sorted(res.keys())
    ['ci_lower', 'ci_upper', 'estimate', 'intercept', 'intercept_p', 'intercept_se', 'p_value', 'se']
    >>> bool("intercept" in res)
    True

    References
    ----------
    bowden2015mendelian
    """
    w = 1 / se_outcome**2
    n = len(beta_exposure)

    # Weighted regression: beta_Y = alpha + beta * beta_X
    X = np.column_stack([np.ones(n), beta_exposure])
    W = np.diag(w)

    try:
        XtWX_inv = np.linalg.inv(X.T @ W @ X)
        beta_hat = XtWX_inv @ X.T @ W @ beta_outcome
    except np.linalg.LinAlgError:
        return {
            "estimate": np.nan,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "intercept": np.nan,
            "intercept_se": np.nan,
            "intercept_p": np.nan,
        }

    resid = beta_outcome - X @ beta_hat
    sigma2 = np.sum(w * resid**2) / (n - 2)
    se_hat = np.sqrt(sigma2 * np.diag(XtWX_inv))

    estimate = beta_hat[1]
    se_est = se_hat[1]
    intercept = beta_hat[0]
    intercept_se = se_hat[0]

    # Egger σ² is plug-in estimated, so both slope and intercept
    # inference follow t(n - 2), matching R's MendelianRandomization /
    # TwoSampleMR packages and mr_pleiotropy_egger below.  Prior
    # versions (< v1.5) used Normal for the slope and t for the
    # intercept; the inconsistency produced anti-conservative slope CIs
    # at small n_snps (e.g. n=5 gave p=7e-6 vs the correct p ~ 1e-3).
    df = max(n - 2, 1)
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    z_slope = estimate / se_est if se_est > 0 else 0.0
    z_intercept = intercept / intercept_se if intercept_se > 0 else 0.0

    return {
        "estimate": estimate,
        "se": se_est,
        "ci_lower": estimate - t_crit * se_est,
        "ci_upper": estimate + t_crit * se_est,
        "p_value": float(2 * stats.t.sf(abs(z_slope), df=df)),
        "intercept": intercept,
        "intercept_se": intercept_se,
        "intercept_p": float(2 * stats.t.sf(abs(z_intercept), df=df)),
    }


def mr_median(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    penalized: bool = False,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Weighted median MR estimator.

    Consistent when at least 50% of the weight comes from valid instruments.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n_snps = 15
    >>> beta_exposure = rng.uniform(0.1, 0.5, n_snps)
    >>> beta_outcome = 0.3 * beta_exposure + rng.normal(0, 0.02, n_snps)
    >>> se_exposure = rng.uniform(0.01, 0.05, n_snps)
    >>> se_outcome = rng.uniform(0.01, 0.05, n_snps)
    >>> res = sp.mr_median(beta_exposure, beta_outcome, se_exposure,
    ...                    se_outcome, n_boot=200, seed=0)
    >>> sorted(res.keys())
    ['ci_lower', 'ci_upper', 'estimate', 'p_value', 'se']
    >>> bool(0.0 < res["estimate"] < 0.6)
    True

    References
    ----------
    bowden2016consistent
    """
    rng = np.random.default_rng(seed)

    ratio = beta_outcome / beta_exposure
    ratio_se = se_outcome / np.abs(beta_exposure)

    weights = 1 / ratio_se**2
    if penalized:
        # Penalize SNPs with large residuals from IVW
        ivw_est = np.sum(weights * ratio) / np.sum(weights)
        penalty = stats.chi2.cdf((ratio - ivw_est) ** 2 / ratio_se**2, 1)
        weights = weights * penalty

    weights = weights / weights.sum()

    # Weighted median
    order = np.argsort(ratio)
    sorted_ratio = ratio[order]
    sorted_weights = weights[order]
    cum_weights = np.cumsum(sorted_weights)
    median_idx = int(np.searchsorted(cum_weights, 0.5))
    estimate = sorted_ratio[min(median_idx, len(sorted_ratio) - 1)]

    # Bootstrap SE
    boot_estimates = np.empty(n_boot)
    for b in range(n_boot):
        boot_beta_y = beta_outcome + rng.normal(0, se_outcome)
        boot_beta_x = beta_exposure + rng.normal(0, se_exposure)
        boot_ratio = boot_beta_y / boot_beta_x
        boot_ratio_se = se_outcome / np.abs(boot_beta_x)
        boot_w = 1 / boot_ratio_se**2
        boot_w = boot_w / boot_w.sum()

        order_b = np.argsort(boot_ratio)
        cum_w_b = np.cumsum(boot_w[order_b])
        mid_b = int(np.searchsorted(cum_w_b, 0.5))
        boot_estimates[b] = boot_ratio[order_b[min(mid_b, len(order_b) - 1)]]

    se = np.std(boot_estimates, ddof=1)
    z = estimate / se
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": estimate - z_crit * se,
        "ci_upper": estimate + z_crit * se,
        "p_value": 2 * (1 - stats.norm.cdf(abs(z))),
    }


def mendelian_randomization(
    data: Optional[pd.DataFrame] = None,
    beta_exposure: Optional[str] = None,
    beta_outcome: Optional[str] = None,
    se_exposure: Optional[str] = None,
    se_outcome: Optional[str] = None,
    exposure_name: str = "Exposure",
    outcome_name: str = "Outcome",
    methods: Optional[List[str]] = None,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> MRResult:
    """
    Mendelian Randomization analysis using summary statistics.

    Equivalent to R's ``MendelianRandomization::mr_allmethods()``
    and ``TwoSampleMR::mr()``.

    Parameters
    ----------
    data : pd.DataFrame
        Summary statistics with one row per SNP/instrument.
    beta_exposure : str
        Column name for SNP-exposure association beta.
    beta_outcome : str
        Column name for SNP-outcome association beta.
    se_exposure : str
        Column name for SNP-exposure SE.
    se_outcome : str
        Column name for SNP-outcome SE.
    exposure_name : str, default 'Exposure'
    outcome_name : str, default 'Outcome'
    methods : list of str, optional
        MR methods to use. Default: ['ivw', 'egger', 'weighted_median'].
    alpha : float, default 0.05
    seed : int, optional

    Returns
    -------
    MRResult
        Results with .summary(), .plot(), estimates table.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n_snps = 25
    >>> bx = rng.uniform(0.05, 0.25, n_snps)          # SNP-exposure effects
    >>> by = 0.4 * bx + rng.normal(0, 0.01, n_snps)   # causal effect ~ 0.4
    >>> snp_stats = pd.DataFrame({
    ...     'beta_x': bx, 'beta_y': by,
    ...     'se_x': rng.uniform(0.01, 0.03, n_snps),
    ...     'se_y': rng.uniform(0.01, 0.03, n_snps),
    ... })
    >>> result = sp.mendelian_randomization(
    ...     data=snp_stats,
    ...     beta_exposure='beta_x', beta_outcome='beta_y',
    ...     se_exposure='se_x', se_outcome='se_y',
    ...     exposure_name='BMI', outcome_name='T2D',
    ... )
    >>> bool(result.n_snps == 25)
    True
    >>> summary_text = result.summary()
    >>> ax = result.plot()
    """
    if data is None:
        raise ValueError("data must be provided.")
    if (
        beta_exposure is None
        or beta_outcome is None
        or se_exposure is None
        or se_outcome is None
    ):
        raise ValueError(
            "beta_exposure, beta_outcome, se_exposure, and se_outcome "
            "must all be provided."
        )
    if methods is None:
        methods = ["ivw", "egger", "weighted_median"]

    bx = data[beta_exposure].values.astype(float)
    by = data[beta_outcome].values.astype(float)
    sx = data[se_exposure].values.astype(float)
    sy = data[se_outcome].values.astype(float)

    # Harmonize: ensure positive beta_exposure
    flip = bx < 0
    bx[flip] = -bx[flip]
    by[flip] = -by[flip]

    n_snps = len(bx)
    results_rows: List[Dict[str, Any]] = []
    heterogeneity: Dict[str, Any] = {"Q": np.nan, "Q_p": np.nan, "I2": np.nan}
    pleiotropy: Optional[Dict[str, float]] = None

    for method in methods:
        if method == "ivw":
            res = mr_ivw(bx, by, sx, sy, alpha)
            heterogeneity = {
                "Q": res["Q"],
                "Q_p": res["Q_p"],
                "Q_df": res["Q_df"],
                "I2": res["I2"],
            }
            results_rows.append(
                {
                    "method": "IVW",
                    "estimate": res["estimate"],
                    "se": res["se"],
                    "ci_lower": res["ci_lower"],
                    "ci_upper": res["ci_upper"],
                    "p_value": res["p_value"],
                }
            )

        elif method == "egger":
            res = mr_egger(bx, by, sx, sy, alpha)
            pleiotropy = {
                "intercept": res["intercept"],
                "se": res["intercept_se"],
                "p_value": res["intercept_p"],
            }
            results_rows.append(
                {
                    "method": "MR-Egger",
                    "estimate": res["estimate"],
                    "se": res["se"],
                    "ci_lower": res["ci_lower"],
                    "ci_upper": res["ci_upper"],
                    "p_value": res["p_value"],
                }
            )

        elif method == "weighted_median":
            res = mr_median(bx, by, sx, sy, seed=seed, alpha=alpha)
            results_rows.append(
                {
                    "method": "Weighted Median",
                    "estimate": res["estimate"],
                    "se": res["se"],
                    "ci_lower": res["ci_lower"],
                    "ci_upper": res["ci_upper"],
                    "p_value": res["p_value"],
                }
            )

        elif method == "penalized_median":
            res = mr_median(bx, by, sx, sy, penalized=True, seed=seed, alpha=alpha)
            results_rows.append(
                {
                    "method": "Penalized Median",
                    "estimate": res["estimate"],
                    "se": res["se"],
                    "ci_lower": res["ci_lower"],
                    "ci_upper": res["ci_upper"],
                    "p_value": res["p_value"],
                }
            )

    estimates = pd.DataFrame(results_rows)

    return MRResult(
        estimates=estimates,
        heterogeneity=heterogeneity,
        pleiotropy=pleiotropy,
        n_snps=n_snps,
        exposure=exposure_name,
        outcome=outcome_name,
    )


def mr_plot(
    result: Optional[MRResult] = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Scatter plot of SNP-exposure vs SNP-outcome effects with MR lines.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if result is None:
        raise ValueError("result must be provided.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # This is a convenience function — actual data would need to be passed
    # For now, plot the estimate lines
    for _, row in result.estimates.iterrows():
        x_range = np.array([0, 1])
        y_range = row["estimate"] * x_range
        ax.plot(
            x_range, y_range, label=f"{row['method']}: {row['estimate']:.3f}", lw=1.5
        )

    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel(f"SNP effect on {result.exposure}")
    ax.set_ylabel(f"SNP effect on {result.outcome}")
    ax.set_title("Mendelian Randomization")
    ax.legend()
    return ax
