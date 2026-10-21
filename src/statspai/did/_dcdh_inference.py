"""Joint inference across horizons for ``sp.did_multiplegt``.

Both helpers work on the cluster-bootstrap draws, so the covariance across
placebo lags or dynamic horizons is the bootstrap covariance.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats


def _joint_placebo_test(
    placebo_results: List[Dict[str, Any]],
    boot_placebo: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """Joint Wald test of H0: all placebo lags equal zero.

    Uses the bootstrap draws (cluster-level) to estimate the covariance
    across lags, so the test accounts for correlation between consecutive
    pre-periods.  Follows dCDH (2024, §3.3) "joint placebo" recommendation.
    """
    if not placebo_results or boot_placebo is None:
        return None

    est = np.array([r["estimate"] for r in placebo_results], dtype=float)
    boot = np.asarray(boot_placebo, dtype=float)  # (n_boot, k)
    # Drop all-NaN columns / rows before covariance.
    mask_cols = ~np.all(np.isnan(boot), axis=0)
    if not mask_cols.any():
        return None
    est = est[mask_cols]
    boot = boot[:, mask_cols]
    valid_rows = ~np.any(np.isnan(boot), axis=1)
    if valid_rows.sum() < boot.shape[1] + 1:
        return None
    boot = boot[valid_rows]

    cov = np.cov(boot, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    k = est.shape[0]
    cov_reg = cov + np.eye(k) * 1e-10

    try:
        W = float(est @ np.linalg.solve(cov_reg, est))
    except np.linalg.LinAlgError:
        W = float(est @ np.linalg.pinv(cov_reg) @ est)

    pval = float(stats.chi2.sf(W, k))
    return {"statistic": W, "df": int(k), "pvalue": pval}


def _avg_cumulative_effect(
    dynamic_results: List[Dict[str, Any]],
    boot_dynamic: Optional[np.ndarray],
    alpha: float,
) -> Optional[Dict[str, Any]]:
    """Unweighted mean of the dynamic estimates DID_0..DID_L.

    Bootstrap SE is computed over the per-draw average across horizons,
    preserving the cross-horizon covariance.

    Every horizon counts equally and nothing is divided by the change in
    treatment, so this is not Stata's ``e(effect_average)`` (switcher-
    weighted and scaled by the first stage, see :func:`_effect_average`)
    nor ``sp.did_multiplegt_dyn``'s average total effect.
    """
    if not dynamic_results or boot_dynamic is None:
        return None
    est_vec = np.array([r["estimate"] for r in dynamic_results], dtype=float)
    if not np.isfinite(est_vec).any():
        return None
    avg_est = float(np.nanmean(est_vec))

    boot = np.asarray(boot_dynamic, dtype=float)
    per_draw_avg = np.nanmean(boot, axis=1)
    per_draw_avg = per_draw_avg[np.isfinite(per_draw_avg)]
    if per_draw_avg.size < 2:
        return {
            "estimate": avg_est,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "pvalue": np.nan,
            "n_horizons": int(np.isfinite(est_vec).sum()),
        }
    se = float(np.std(per_draw_avg, ddof=1))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = avg_est / se if se > 0 else 0.0
    return {
        "estimate": avg_est,
        "se": se,
        "ci_lower": avg_est - z_crit * se,
        "ci_upper": avg_est + z_crit * se,
        "pvalue": float(2 * stats.norm.sf(abs(z))),
        "n_horizons": int(np.isfinite(est_vec).sum()),
    }


def _effect_average(
    dynamic_results: List[Dict[str, Any]],
    boot_draws: Optional[np.ndarray],
    alpha: float,
) -> Optional[Dict[str, Any]]:
    """Stata ``did_multiplegt_old, robust_dynamic``'s ``e(effect_average)``.

    ``sum_l N_l DID_l / sum_l N_l DID^D_l`` over the first-time-switcher
    estimates l = 0..L, where ``N_l`` is the number of switchers DID_l
    applies to and ``DID^D_l`` is DID_l with the treatment as outcome. Since
    ``N_l DID_l`` is DID_l's numerator, this is the ratio of the summed
    numerators (``num`` over ``num_d``, from
    :func:`._dcdh_first_switch.first_switch_effect`): the effect of the
    first switches on the outcome per unit of treatment they cause.

    ``boot_draws`` holds the ratio in each bootstrap replication; the SE is
    their standard deviation, as for every other estimate of the function.
    """
    if not dynamic_results:
        return None
    num = float(sum(r["num"] for r in dynamic_results))
    num_d = float(sum(r["num_d"] for r in dynamic_results))
    den = float(sum(r["den"] for r in dynamic_results))
    nan = float("nan")
    est = num / num_d if num_d != 0 else nan
    draws = (
        np.asarray(boot_draws, dtype=float) if boot_draws is not None else np.empty(0)
    )
    draws = draws[np.isfinite(draws)]
    se = float(np.std(draws, ddof=1)) if np.isfinite(est) and draws.size >= 2 else nan
    z_crit = stats.norm.ppf(1 - alpha / 2)
    if np.isfinite(se):
        z = est / se if se > 0 else 0.0
        pvalue = float(2 * stats.norm.sf(abs(z)))
        ci = (est - z_crit * se, est + z_crit * se)
    else:
        pvalue, ci = nan, (nan, nan)
    return {
        "estimate": est,
        "se": se,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "pvalue": pvalue,
        # sum_l N_l DID_l / sum_l N_l and sum_l N_l DID^D_l / sum_l N_l
        "itt": num / den if den else nan,
        "first_stage": num_d / den if den else nan,
        "n_switchers": int(round(den)),
        "n_horizons": int(sum(r["den"] > 0 for r in dynamic_results)),
    }
