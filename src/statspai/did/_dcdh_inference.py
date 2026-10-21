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
    """Average cumulative dynamic effect: mean of dynamic[0..L].

    Bootstrap SE is computed over the per-draw average across horizons,
    preserving the cross-horizon covariance (dCDH 2024 §3.4).
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
