"""
Stratified analysis for confounding control in 2x2xK tables.

Implements:
  - Mantel-Haenszel pooled OR / RR / IRR
  - Breslow-Day test for homogeneity of OR across strata
  - Cochran's Q test as a simpler homogeneity check

References
----------
Mantel, N. & Haenszel, W. (1959). "Statistical aspects of the analysis of
data from retrospective studies of disease." *JNCI*, 22(4), 719-748.

Breslow, N.E. & Day, N.E. (1980). *Statistical Methods in Cancer
Research. Volume I - The Analysis of Case-Control Studies*. IARC
Scientific Publications No. 32.

Tarone, R.E. (1985). "On heterogeneity tests based on efficient scores."
*Biometrika*, 72(1), 91-95. [@tarone1985heterogeneity]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np
from scipy import stats


__all__ = [
    "MantelHaenszelResult",
    "mantel_haenszel",
    "breslow_day_test",
]


@dataclass
class MantelHaenszelResult:
    estimate: float
    ci: tuple[float, float]
    p_value: float
    se_log: float
    measure: str  # "OR" | "RR" | "IRR"
    n_strata: int
    stratum_estimates: List[float]
    homogeneity_statistic: float
    homogeneity_p: float
    homogeneity_method: str

    def summary(self) -> str:
        lo, hi = self.ci
        lines = [
            f"Mantel-Haenszel pooled {self.measure}",
            f"  Estimate = {self.estimate:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   "
            f"p = {self.p_value:.4g}",
            f"  Strata: {self.n_strata}",
            f"  Per-stratum {self.measure}: "
            + ", ".join(f"{x:.3f}" for x in self.stratum_estimates),
            f"  Homogeneity ({self.homogeneity_method}): "
            f"chi2 = {self.homogeneity_statistic:.3f}, p = {self.homogeneity_p:.4g}",
        ]
        return "\n".join(lines)


def _validate_strata(tables) -> np.ndarray:
    arr = np.asarray(tables, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] != (2, 2):
        raise ValueError(
            "Expected an array-like of shape (K, 2, 2); got %s" % (arr.shape,)
        )
    if (arr < 0).any():
        raise ValueError("Counts must be non-negative.")
    return arr


def mantel_haenszel(
    tables: Union[Sequence, np.ndarray],
    *,
    measure: str = "OR",
    alpha: float = 0.05,
) -> MantelHaenszelResult:
    """Mantel-Haenszel pooled OR / RR across ``K`` strata.

    Parameters
    ----------
    tables : array-like, shape ``(K, 2, 2)``
        Each stratum's 2x2 table with layout
        ``[[a_k, b_k], [c_k, d_k]]`` (exposure x outcome).
    measure : {"OR", "RR"}
        Pooled measure.  Use :func:`mantel_haenszel_rate` for
        person-time IRR.
    alpha : float
        Two-sided CI level.

    Returns
    -------
    MantelHaenszelResult
    """
    arr = _validate_strata(tables)
    K = arr.shape[0]
    measure = measure.upper()
    if measure not in ("OR", "RR"):
        raise ValueError("measure must be 'OR' or 'RR'")

    a = arr[:, 0, 0]
    b = arr[:, 0, 1]
    c = arr[:, 1, 0]
    d = arr[:, 1, 1]
    n1 = a + b
    n0 = c + d
    n = n1 + n0

    if (n == 0).any():
        raise ValueError("Encountered an empty stratum.")

    # Per-stratum estimates (with Haldane if empty cells)
    stratum_est: list[float] = []
    for k in range(K):
        ak, bk, ck, dk = a[k], b[k], c[k], d[k]
        if min(ak, bk, ck, dk) == 0:
            ak_c, bk_c, ck_c, dk_c = ak + 0.5, bk + 0.5, ck + 0.5, dk + 0.5
        else:
            ak_c, bk_c, ck_c, dk_c = ak, bk, ck, dk
        if measure == "OR":
            stratum_est.append(float((ak_c * dk_c) / (bk_c * ck_c)))
        else:
            p1k = ak_c / (ak_c + bk_c)
            p0k = ck_c / (ck_c + dk_c)
            stratum_est.append(float(p1k / p0k))

    if measure == "OR":
        num = np.sum(a * d / n)
        den = np.sum(b * c / n)
        or_mh = num / den if den > 0 else np.inf
        # Robins-Breslow-Greenland variance for log(OR_MH)
        P = (a + d) / n
        Q = (b + c) / n
        R = (a * d) / n
        S = (b * c) / n
        sum_R = R.sum()
        sum_S = S.sum()
        var_log = (
            np.sum(P * R) / (2 * sum_R ** 2)
            + np.sum(P * S + Q * R) / (2 * sum_R * sum_S)
            + np.sum(Q * S) / (2 * sum_S ** 2)
        )
        se_log = float(np.sqrt(var_log))
        est = float(or_mh)
        log_est = np.log(est) if est > 0 else 0.0
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (float(np.exp(log_est - z * se_log)),
              float(np.exp(log_est + z * se_log)))
        # M-H chi-square test (with continuity correction)
        expected_a = n1 * (a + c) / n
        var_a = (n1 * n0 * (a + c) * (b + d)) / (n ** 2 * (n - 1 + (n == 1)))
        chi2_mh = (abs(a.sum() - expected_a.sum()) - 0.5) ** 2 / var_a.sum()
        p = float(1 - stats.chi2.cdf(chi2_mh, 1))

    else:  # RR
        num = np.sum(a * n0 / n)
        den = np.sum(c * n1 / n)
        rr_mh = num / den if den > 0 else np.inf
        # Greenland-Robins variance for log(RR_MH)
        var_log = np.sum(
            (n1 * n0 * (a + c) - a * c * n) / n ** 2
        ) / (num * den)
        se_log = float(np.sqrt(var_log)) if var_log > 0 else 0.0
        est = float(rr_mh)
        log_est = np.log(est) if est > 0 else 0.0
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (float(np.exp(log_est - z * se_log)),
              float(np.exp(log_est + z * se_log)))
        # Test via pooled RR == 1
        z_stat = log_est / se_log if se_log > 0 else 0.0
        p = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # Homogeneity check (inverse-variance weighted Cochran's Q on
    # log-scale per-stratum estimates).  This is the standard
    # meta-analysis test for heterogeneity:
    #   Q = sum_k  w_k * (log(theta_k) - log(theta_bar))^2 ~ chi^2(K-1)
    # where w_k = 1 / var(log(theta_k)) from each stratum's 2x2.
    log_ks = np.log(np.asarray(stratum_est))
    var_log_ks = np.empty(K)
    for k in range(K):
        ak, bk, ck_, dk = a[k], b[k], c[k], d[k]
        # Haldane correction for empty cells
        if min(ak, bk, ck_, dk) == 0:
            ak_c, bk_c, ck_c, dk_c = ak + 0.5, bk + 0.5, ck_ + 0.5, dk + 0.5
        else:
            ak_c, bk_c, ck_c, dk_c = ak, bk, ck_, dk
        if measure == "OR":
            # Var(log OR_k) = 1/a + 1/b + 1/c + 1/d
            var_log_ks[k] = 1.0 / ak_c + 1.0 / bk_c + 1.0 / ck_c + 1.0 / dk_c
        else:  # RR
            # Var(log RR_k) = 1/a - 1/(a+b) + 1/c - 1/(c+d)
            var_log_ks[k] = (
                1.0 / ak_c - 1.0 / (ak_c + bk_c)
                + 1.0 / ck_c - 1.0 / (ck_c + dk_c)
            )
    w_meta = 1.0 / var_log_ks
    pooled_log = float(np.sum(w_meta * log_ks) / np.sum(w_meta))
    q_chi2 = float(np.sum(w_meta * (log_ks - pooled_log) ** 2))
    q_df = max(K - 1, 1)
    q_p = float(1 - stats.chi2.cdf(q_chi2, q_df))

    return MantelHaenszelResult(
        estimate=est,
        ci=ci,
        p_value=p,
        se_log=se_log,
        measure=measure,
        n_strata=K,
        stratum_estimates=stratum_est,
        homogeneity_statistic=q_chi2,
        homogeneity_p=q_p,
        homogeneity_method="inverse-variance Cochran Q (log-scale)",
    )


def breslow_day_test(
    tables: Union[Sequence, np.ndarray],
    *,
    tarone_correction: bool = True,
) -> tuple[float, float]:
    """Breslow-Day test for homogeneity of the odds ratio across strata.

    Parameters
    ----------
    tables : array-like, shape ``(K, 2, 2)``
    tarone_correction : bool, default True
        Apply Tarone's correction (recommended; Tarone 1985).

    Returns
    -------
    chi2 : float
    p_value : float
    """
    arr = _validate_strata(tables)
    K = arr.shape[0]

    # Compute common MH-OR
    mh = mantel_haenszel(arr, measure="OR")
    or_mh = mh.estimate

    chi2 = 0.0
    sum_a = 0.0
    sum_ea = 0.0
    sum_va = 0.0

    for k in range(K):
        a, b = arr[k, 0, 0], arr[k, 0, 1]
        c, d = arr[k, 1, 0], arr[k, 1, 1]
        n1 = a + b
        n0 = c + d
        m1 = a + c  # outcome = 1 total
        n = n1 + n0

        if or_mh == 1.0:
            expected_a = n1 * m1 / n
        else:
            # Solve quadratic for expected a under common OR
            # (e_a)(d+e_a-m1+n1) = OR * (n1-e_a)(m1-e_a)
            coef_a_sq = 1 - or_mh
            coef_a_b = or_mh * (m1 + n1) + n - m1 - n1
            coef_a_c = -or_mh * m1 * n1
            if abs(coef_a_sq) < 1e-12:
                expected_a = -coef_a_c / coef_a_b if coef_a_b else n1 * m1 / n
            else:
                disc = coef_a_b ** 2 - 4 * coef_a_sq * coef_a_c
                disc = max(disc, 0.0)
                expected_a = (-coef_a_b + np.sqrt(disc)) / (2 * coef_a_sq)
                if expected_a < 0 or expected_a > min(m1, n1):
                    expected_a = (-coef_a_b - np.sqrt(disc)) / (2 * coef_a_sq)

        # Variance of a under common OR (Breslow-Day)
        e_b = n1 - expected_a
        e_c = m1 - expected_a
        e_d = n - n1 - e_c
        if min(expected_a, e_b, e_c, e_d) <= 0:
            continue
        var_a = 1 / (1 / expected_a + 1 / e_b + 1 / e_c + 1 / e_d)
        chi2 += (a - expected_a) ** 2 / var_a
        sum_a += a
        sum_ea += expected_a
        sum_va += var_a

    if tarone_correction and sum_va > 0:
        chi2 -= (sum_a - sum_ea) ** 2 / sum_va

    df = max(K - 1, 1)
    p_value = float(1 - stats.chi2.cdf(chi2, df))
    return float(chi2), p_value
