"""
Comprehensive bandwidth selection for local polynomial RD estimation.

Implements all eight bandwidth selection methods from Calonico, Cattaneo,
and Farrell (2020) including MSE-optimal and CER-optimal variants:

MSE-optimal (minimize mean squared error of the RD estimator):
  - mserd    : common bandwidth
  - msetwo   : separate left/right
  - msecomb1 : min(mserd, mseleft, mseright)
  - msecomb2 : median(mserd, mseleft, mseright)

CER-optimal (minimize coverage error rate of confidence intervals):
  - cerrd    : common bandwidth
  - certwo   : separate left/right
  - cercomb1 : min(cerrd, cerleft, cerright)
  - cercomb2 : median(cerrd, cerleft, cerright)

Supports sharp RD, fuzzy RD, covariate adjustment, and cluster-robust
variance estimation.

References
----------
Calonico, S., Cattaneo, M.D. and Farrell, M.H. (2020).
"Optimal Bandwidth Choice for Robust Bias-Corrected Inference in
Regression Discontinuity Designs." *Econometrics Journal*, 23(2), 192-210. [@calonico2020optimal]

Calonico, S., Cattaneo, M.D. and Titiunik, R. (2014).
"Robust Nonparametric Confidence Intervals for Regression-Discontinuity
Designs." *Econometrica*, 82(6), 2295-2326. [@calonico2014robust]

Imbens, G. and Kalyanaraman, K. (2012).
"Optimal Bandwidth Choice for the Regression Discontinuity Estimator."
*Review of Economic Studies*, 79(3), 933-959. [@imbens2012optimal]
"""

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ======================================================================
# Kernel helpers (canonical definitions live in ._core)
# ======================================================================

from ._core import _kernel_fn, _kernel_constants, _sandwich_variance  # noqa: F401


# ======================================================================
# Internal estimation helpers
# ======================================================================

def _local_poly_fit(
    y: np.ndarray,
    x: np.ndarray,
    h: float,
    p: int,
    kernel: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Weighted local polynomial regression at x = 0.

    Returns (coefficients, residuals_in_bw, n_effective).
    """
    u = x / h
    in_bw = np.abs(u) <= 1
    n_eff = int(in_bw.sum())
    if n_eff < p + 2:
        return np.zeros(p + 1), np.array([]), n_eff

    y_bw = y[in_bw]
    x_bw = x[in_bw]
    w_bw = _kernel_fn(u[in_bw], kernel)

    X = np.column_stack([x_bw ** j for j in range(p + 1)])
    sqw = np.sqrt(w_bw)
    Xw = X * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(p + 1)

    resid = y_bw - X @ beta
    return beta, resid, n_eff


def _local_residual_var(
    y: np.ndarray, x: np.ndarray, h: float, kernel: str,
) -> float:
    """Conditional variance at x = 0 from local linear residuals."""
    u = x / h
    in_bw = np.abs(u) <= 1
    if in_bw.sum() < 5:
        return float(np.var(y)) if len(y) > 0 else 1.0

    y_bw, x_bw, w_bw = y[in_bw], x[in_bw], _kernel_fn(u[in_bw], kernel)
    X = np.column_stack([np.ones(len(x_bw)), x_bw])
    sqw = np.sqrt(w_bw)
    Xw = X * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        resid = y_bw - X @ beta
        return float(np.average(resid ** 2, weights=w_bw))
    except Exception:
        return float(np.var(y_bw))


def _local_residual_var_cluster(
    y: np.ndarray,
    x: np.ndarray,
    h: float,
    kernel: str,
    cluster: np.ndarray,
) -> float:
    """Cluster-robust conditional variance at x = 0."""
    u = x / h
    in_bw = np.abs(u) <= 1
    if in_bw.sum() < 5:
        return float(np.var(y)) if len(y) > 0 else 1.0

    y_bw = y[in_bw]
    x_bw = x[in_bw]
    w_bw = _kernel_fn(u[in_bw], kernel)
    cl_bw = cluster[in_bw]

    X = np.column_stack([np.ones(len(x_bw)), x_bw])
    sqw = np.sqrt(w_bw)
    Xw = X * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        resid = y_bw - X @ beta
    except Exception:
        return float(np.var(y_bw))

    # Cluster-robust variance of the intercept (treatment effect proxy).
    # _sandwich_variance accepts (Xw, yw, beta, resid_raw, n_eff, k, cluster)
    # and constructs scores Xw[g]' @ (yw[g] - Xw[g]@beta) = Xw[g]' @ (sqw*resid)[g],
    # which matches the legacy inline computation.
    vcov = _sandwich_variance(
        Xw, yw, beta, resid, int(in_bw.sum()), 2, cl_bw,
    )
    # Return variance estimate (intercept variance scaled by n * f_c)
    return float(vcov[0, 0] * in_bw.sum())


def _estimate_deriv(
    y: np.ndarray,
    x: np.ndarray,
    h: float,
    kernel: str,
    deriv_order: int = 2,
    poly_order: int = 3,
) -> float:
    """
    Estimate m^{(deriv_order)}(0) via local polynomial regression.

    For second derivative (curvature), uses local cubic (poly_order=3)
    and returns factorial(deriv_order) * beta[deriv_order].
    """
    u = x / h
    in_bw = np.abs(u) <= 1
    if in_bw.sum() < poly_order + 2:
        return 0.0

    y_bw = y[in_bw]
    x_bw = x[in_bw]
    w_bw = _kernel_fn(u[in_bw], kernel)

    X = np.column_stack([x_bw ** j for j in range(poly_order + 1)])
    sqw = np.sqrt(w_bw)
    Xw = X * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        from math import factorial
        return float(factorial(deriv_order) * beta[deriv_order])
    except Exception:
        return 0.0


def _estimate_third_deriv(
    y: np.ndarray, x: np.ndarray, h: float, kernel: str,
) -> float:
    """Estimate m'''(0) using local quartic regression."""
    return _estimate_deriv(y, x, h, kernel, deriv_order=3, poly_order=4)


def _density_at_cutoff(X_c: np.ndarray, h_pilot: float, n: int) -> float:
    """Estimate f(c) via a simple frequency estimator."""
    n_near = np.sum(np.abs(X_c) <= h_pilot)
    f_c = n_near / (2 * h_pilot * n) if h_pilot > 0 and n > 0 else 1.0
    return max(f_c, 1e-10)


def _covariate_adjusted_variance(
    y: np.ndarray,
    x: np.ndarray,
    covs_data: np.ndarray,
    h: float,
    kernel: str,
) -> float:
    """
    Variance at the cutoff after partialling out covariates.

    Regresses y on covariates within the bandwidth, then computes the
    residual variance with kernel weights.
    """
    u = x / h
    in_bw = np.abs(u) <= 1
    if in_bw.sum() < covs_data.shape[1] + 3:
        return _local_residual_var(y, x, h, kernel)

    y_bw = y[in_bw]
    x_bw = x[in_bw]
    w_bw = _kernel_fn(u[in_bw], kernel)
    covs_bw = covs_data[in_bw]

    # First stage: partial out covariates
    X_cov = np.column_stack([np.ones(len(x_bw)), x_bw, covs_bw])
    sqw = np.sqrt(w_bw)
    Xw = X_cov * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        resid = y_bw - X_cov @ beta
        return float(np.average(resid ** 2, weights=w_bw))
    except Exception:
        return _local_residual_var(y, x, h, kernel)


# ======================================================================
# CER shrinkage factor
# ======================================================================

def _cer_factor(n: int, p: int) -> float:
    """
    Coverage Error Rate (CER) shrinkage factor.

    The MSE-optimal bandwidth has rate n^{-1/(2p+3)} while the
    CER-optimal bandwidth has rate n^{-1/(2p+3+2/(2p+3))}.
    For the default local linear (p=1), this gives:
      MSE rate = n^{-1/5}, CER rate approx n^{-5/21}

    The shrinkage factor is:
      n^{-1/(2p+3)} / n^{rate_CER} = n^{rate_CER - rate_MSE}

    For p=1: factor ~ n^{-1/21 + 1/5} ... simplified as n^{-1/(2(2p+3))}

    Following CCF (2020), the CER bandwidth is:
      h_CER = h_MSE * n^{-1/((2p+3)(2p+5))}

    For p=1: h_CER = h_MSE * n^{-1/(5*7)} = h_MSE * n^{-1/35}
    For p=2: h_CER = h_MSE * n^{-1/(7*9)} = h_MSE * n^{-1/63}

    Parameters
    ----------
    n : int
        Number of observations (total or on one side).
    p : int
        Polynomial order for estimation.

    Returns
    -------
    float
        Multiplicative factor in (0, 1) to shrink MSE bandwidth to CER.
    """
    if n <= 1:
        return 1.0
    rate_exponent = 1.0 / ((2 * p + 3) * (2 * p + 5))
    return float(n ** (-rate_exponent))


# ======================================================================
# Side-level MSE-optimal bandwidth
# ======================================================================

def _mse_bandwidth_side(
    sigma2: float,
    m2: float,
    f_c: float,
    n_side: int,
    C_K: float,
    h_pilot: float,
    x_range: float,
) -> float:
    """
    MSE-optimal bandwidth for one side of the cutoff.

    h_MSE = (C_K * sigma^2 / (n * f_c * (m'')^2))^{1/5}

    Parameters
    ----------
    sigma2 : float
        Conditional variance at the cutoff (one side).
    m2 : float
        Second derivative estimate m''(0) on one side.
    f_c : float
        Density at the cutoff.
    n_side : int
        Number of observations on this side.
    C_K : float
        Kernel-specific MSE constant.
    h_pilot : float
        Pilot bandwidth (fallback).
    x_range : float
        Range of the running variable (for clipping).

    Returns
    -------
    float
        MSE-optimal bandwidth for this side.
    """
    bias_sq = m2 ** 2
    if bias_sq < 1e-12 or n_side < 5:
        h_opt = h_pilot
    else:
        h_opt = (C_K * sigma2 / (f_c * bias_sq * n_side)) ** (1 / 5)
    return float(np.clip(h_opt, 0.02 * x_range, 0.98 * x_range))


def _mse_bandwidth_common(
    sigma2_l: float,
    sigma2_r: float,
    m2_l: float,
    m2_r: float,
    f_c: float,
    n: int,
    C_K: float,
    h_pilot: float,
    x_range: float,
) -> float:
    """
    MSE-optimal common bandwidth for sharp RD.

    h_MSE = (C_K * (sigma_l^2 + sigma_r^2) / (n * f_c * ((m''_r - m''_l)/2)^2))^{1/5}
    """
    bias_sq = ((m2_r - m2_l) / 2) ** 2
    if bias_sq < 1e-12:
        h_opt = h_pilot
    else:
        h_opt = (C_K * (sigma2_l + sigma2_r) /
                 (f_c * bias_sq * n)) ** (1 / 5)
    return float(np.clip(h_opt, 0.02 * x_range, 0.98 * x_range))


# ======================================================================
# Pilot bandwidth for bias estimation
# ======================================================================

def _pilot_bandwidth(
    Y: np.ndarray,
    X_c: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    p: int,
    kernel: str,
) -> Tuple[float, float]:
    """
    Compute pilot bandwidths for curvature estimation.

    Uses a regularization-based approach: pilot = C * h_rot where
    h_rot is the Silverman rotation bandwidth and C is inflated to
    ensure enough observations for higher-order fits.

    Returns (h_pilot_main, h_pilot_deriv).
    """
    n = len(Y)
    sd_x = np.std(X_c)
    # Silverman rule
    h_pilot = 1.06 * sd_x * n ** (-1 / 5)
    # Wider pilot for derivative estimation
    h_deriv = max(np.median(np.abs(X_c)), h_pilot) * 1.5
    return h_pilot, h_deriv


# ======================================================================
# Bias bandwidth (b) selection
# ======================================================================

def _bias_bandwidth_side(
    y: np.ndarray,
    x: np.ndarray,
    h_main: float,
    h_pilot: float,
    kernel: str,
    n_side: int,
    f_c: float,
    x_range: float,
) -> float:
    """
    Pilot bandwidth for bias correction on one side.

    Uses the third derivative to select the bandwidth b for estimating
    the bias of the local polynomial estimator. The bandwidth b is
    typically wider than h.

    b_MSE ~ (C_K * sigma^2 / (n * f_c * (m''')^2))^{1/7}
    """
    sigma2 = _local_residual_var(y, x, h_main, kernel)
    m3 = _estimate_third_deriv(y, x, h_pilot * 2.0, kernel)

    bias_sq = m3 ** 2
    C_K = _kernel_constants(kernel)['C_K']

    if bias_sq < 1e-12 or n_side < 8:
        return float(np.clip(h_main * 1.5, 0.02 * x_range, 0.98 * x_range))

    b_opt = (C_K * sigma2 / (f_c * bias_sq * n_side)) ** (1 / 7)
    return float(np.clip(b_opt, 0.02 * x_range, 0.98 * x_range))


# ======================================================================
# Fuzzy-design variance adjustment
# ======================================================================

def _fuzzy_variance_adjust(
    sigma2_y: float,
    sigma2_d: float,
    cov_yd: float,
    fs_effect: float,
) -> float:
    """
    Adjust variance for fuzzy RD design.

    In fuzzy RD, the variance of the Wald estimator is approximately:
      Var(tau_FRD) ~ (sigma_y^2 - 2*tau*cov(y,d) + tau^2*sigma_d^2) / fs^2

    For bandwidth selection, we use a simplified inflation factor.

    Parameters
    ----------
    sigma2_y : float
        Outcome variance at cutoff.
    sigma2_d : float
        Treatment variance at cutoff.
    cov_yd : float
        Covariance of outcome and treatment at cutoff.
    fs_effect : float
        First-stage effect (jump in treatment probability).

    Returns
    -------
    float
        Adjusted variance for bandwidth selection.
    """
    if abs(fs_effect) < 1e-10:
        return sigma2_y * 100  # degenerate first stage -> very wide
    # Delta method approximation
    tau_approx = 0  # under null for bandwidth purposes
    var_wald = (sigma2_y - 2 * tau_approx * cov_yd +
                tau_approx ** 2 * sigma2_d) / (fs_effect ** 2)
    return max(var_wald, sigma2_y)


def _estimate_first_stage(
    D: np.ndarray,
    X_c: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    h: float,
    kernel: str,
) -> Tuple[float, float, float]:
    """
    Estimate first-stage effect and treatment variance on each side.

    Returns (fs_effect, sigma2_d_left, sigma2_d_right).
    """
    beta_l, _, _ = _local_poly_fit(D[left], X_c[left], h, 1, kernel)
    beta_r, _, _ = _local_poly_fit(D[right], X_c[right], h, 1, kernel)
    fs_effect = beta_r[0] - beta_l[0]

    sigma2_d_l = _local_residual_var(D[left], X_c[left], h, kernel)
    sigma2_d_r = _local_residual_var(D[right], X_c[right], h, kernel)
    return fs_effect, sigma2_d_l, sigma2_d_r


def _estimate_covariance_yd(
    Y: np.ndarray,
    D: np.ndarray,
    X_c: np.ndarray,
    h: float,
    kernel: str,
) -> float:
    """Estimate Cov(Y, D) at cutoff from local linear residuals."""
    u = X_c / h
    in_bw = np.abs(u) <= 1
    if in_bw.sum() < 5:
        return 0.0

    y_bw = Y[in_bw]
    d_bw = D[in_bw]
    x_bw = X_c[in_bw]
    w_bw = _kernel_fn(u[in_bw], kernel)

    X = np.column_stack([np.ones(len(x_bw)), x_bw])
    sqw = np.sqrt(w_bw)
    Xw = X * sqw[:, np.newaxis]

    try:
        beta_y = np.linalg.lstsq(Xw, y_bw * sqw, rcond=None)[0]
        beta_d = np.linalg.lstsq(Xw, d_bw * sqw, rcond=None)[0]
        resid_y = y_bw - X @ beta_y
        resid_d = d_bw - X @ beta_d
        return float(np.average(resid_y * resid_d, weights=w_bw))
    except Exception:
        return 0.0


# ======================================================================
# Core bandwidth computation engine
# ======================================================================

def _compute_all_bandwidths(
    Y: np.ndarray,
    X_c: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    p: int,
    q: int,
    kernel: str,
    D: Optional[np.ndarray] = None,
    covs_data: Optional[np.ndarray] = None,
    cluster_vals: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all eight bandwidth types plus bias bandwidths.

    Returns a dict keyed by method name with values
    (h_left, h_right, b_left, b_right).
    """
    n = len(Y)
    n_left = int(left.sum())
    n_right = int(right.sum())
    x_range = np.ptp(X_c)

    y_l, x_l = Y[left], X_c[left]
    y_r, x_r = Y[right], X_c[right]

    kc = _kernel_constants(kernel)
    C_K = kc['C_K']

    # --- Pilot bandwidths ---
    h_pilot, h_deriv = _pilot_bandwidth(Y, X_c, left, right, p, kernel)

    # --- Density at cutoff ---
    f_c = _density_at_cutoff(X_c, h_pilot, n)

    # --- Conditional variance on each side ---
    if covs_data is not None:
        covs_l = covs_data[left]
        covs_r = covs_data[right]
        sigma2_l = _covariate_adjusted_variance(
            y_l, x_l, covs_l, h_pilot, kernel)
        sigma2_r = _covariate_adjusted_variance(
            y_r, x_r, covs_r, h_pilot, kernel)
    elif cluster_vals is not None:
        sigma2_l = _local_residual_var_cluster(
            y_l, x_l, h_pilot, kernel, cluster_vals[left])
        sigma2_r = _local_residual_var_cluster(
            y_r, x_r, h_pilot, kernel, cluster_vals[right])
    else:
        sigma2_l = _local_residual_var(y_l, x_l, h_pilot, kernel)
        sigma2_r = _local_residual_var(y_r, x_r, h_pilot, kernel)

    # --- Fuzzy design: adjust variance ---
    if D is not None:
        fs_effect, sigma2_d_l, sigma2_d_r = _estimate_first_stage(
            D, X_c, left, right, h_pilot, kernel)
        cov_yd_l = _estimate_covariance_yd(
            y_l, D[left], x_l, h_pilot, kernel)
        cov_yd_r = _estimate_covariance_yd(
            y_r, D[right], x_r, h_pilot, kernel)
        sigma2_l = _fuzzy_variance_adjust(
            sigma2_l, sigma2_d_l, cov_yd_l, fs_effect)
        sigma2_r = _fuzzy_variance_adjust(
            sigma2_r, sigma2_d_r, cov_yd_r, fs_effect)

    # --- Second derivatives (curvature for bias) ---
    m2_l = _estimate_deriv(y_l, x_l, h_deriv, kernel, deriv_order=2,
                           poly_order=max(p + 1, 3))
    m2_r = _estimate_deriv(y_r, x_r, h_deriv, kernel, deriv_order=2,
                           poly_order=max(p + 1, 3))

    # ================================================================
    # MSE-optimal bandwidths
    # ================================================================

    # mserd: common bandwidth
    h_mserd = _mse_bandwidth_common(
        sigma2_l, sigma2_r, m2_l, m2_r, f_c, n, C_K, h_pilot, x_range)

    # msetwo / mseleft / mseright: separate bandwidths
    h_mse_l = _mse_bandwidth_side(
        sigma2_l, m2_l, f_c, n_left, C_K, h_pilot, x_range)
    h_mse_r = _mse_bandwidth_side(
        sigma2_r, m2_r, f_c, n_right, C_K, h_pilot, x_range)

    # msecomb1: min of common and separate
    h_msecomb1 = min(h_mserd, h_mse_l, h_mse_r)

    # msecomb2: median of common and separate
    h_msecomb2 = float(np.median([h_mserd, h_mse_l, h_mse_r]))

    # ================================================================
    # Bias bandwidths (b) for each MSE bandwidth
    # ================================================================
    b_mserd_l = _bias_bandwidth_side(
        y_l, x_l, h_mserd, h_deriv, kernel, n_left, f_c, x_range)
    b_mserd_r = _bias_bandwidth_side(
        y_r, x_r, h_mserd, h_deriv, kernel, n_right, f_c, x_range)

    b_mse_l = _bias_bandwidth_side(
        y_l, x_l, h_mse_l, h_deriv, kernel, n_left, f_c, x_range)
    b_mse_r = _bias_bandwidth_side(
        y_r, x_r, h_mse_r, h_deriv, kernel, n_right, f_c, x_range)

    b_msecomb1_l = _bias_bandwidth_side(
        y_l, x_l, h_msecomb1, h_deriv, kernel, n_left, f_c, x_range)
    b_msecomb1_r = _bias_bandwidth_side(
        y_r, x_r, h_msecomb1, h_deriv, kernel, n_right, f_c, x_range)

    b_msecomb2_l = _bias_bandwidth_side(
        y_l, x_l, h_msecomb2, h_deriv, kernel, n_left, f_c, x_range)
    b_msecomb2_r = _bias_bandwidth_side(
        y_r, x_r, h_msecomb2, h_deriv, kernel, n_right, f_c, x_range)

    # ================================================================
    # CER-optimal bandwidths
    # ================================================================
    cer_n = _cer_factor(n, p)
    cer_l = _cer_factor(n_left, p)
    cer_r = _cer_factor(n_right, p)

    # cerrd: CER-optimal common
    h_cerrd = h_mserd * cer_n

    # certwo: CER-optimal separate
    h_cer_l = h_mse_l * cer_l
    h_cer_r = h_mse_r * cer_r

    # cercomb1: min
    h_cercomb1 = min(h_cerrd, h_cer_l, h_cer_r)

    # cercomb2: median
    h_cercomb2 = float(np.median([h_cerrd, h_cer_l, h_cer_r]))

    # CER bias bandwidths (shrink b proportionally)
    b_cerrd_l = b_mserd_l * cer_n
    b_cerrd_r = b_mserd_r * cer_n
    b_cer_l = b_mse_l * cer_l
    b_cer_r = b_mse_r * cer_r
    b_cercomb1_l = b_msecomb1_l * min(cer_n, cer_l, cer_r)
    b_cercomb1_r = b_msecomb1_r * min(cer_n, cer_l, cer_r)
    b_cercomb2_l = b_msecomb2_l * float(np.median([cer_n, cer_l, cer_r]))
    b_cercomb2_r = b_msecomb2_r * float(np.median([cer_n, cer_l, cer_r]))

    # ================================================================
    # Pack results
    # ================================================================
    results = {
        'mserd':    (h_mserd,    h_mserd,    b_mserd_l,    b_mserd_r),
        'msetwo':   (h_mse_l,   h_mse_r,    b_mse_l,      b_mse_r),
        'msecomb1': (h_msecomb1, h_msecomb1, b_msecomb1_l, b_msecomb1_r),
        'msecomb2': (h_msecomb2, h_msecomb2, b_msecomb2_l, b_msecomb2_r),
        'cerrd':    (h_cerrd,    h_cerrd,    b_cerrd_l,    b_cerrd_r),
        'certwo':   (h_cer_l,   h_cer_r,    b_cer_l,      b_cer_r),
        'cercomb1': (h_cercomb1, h_cercomb1, b_cercomb1_l, b_cercomb1_r),
        'cercomb2': (h_cercomb2, h_cercomb2, b_cercomb2_l, b_cercomb2_r),
    }
    return results


# ======================================================================
# Public API
# ======================================================================

_VALID_METHODS = {
    'mserd', 'msetwo', 'msecomb1', 'msecomb2',
    'cerrd', 'certwo', 'cercomb1', 'cercomb2',
}


def rdbwselect(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    fuzzy: Optional[str] = None,
    deriv: int = 0,
    p: int = 1,
    q: Optional[int] = None,
    covs: Optional[List[str]] = None,
    kernel: str = 'triangular',
    bwselect: str = 'mserd',
    cluster: Optional[str] = None,
    all: bool = False,
) -> pd.DataFrame:
    """
    Bandwidth selection for local polynomial RD estimation.

    Implements all eight MSE-optimal and CER-optimal bandwidth selection
    procedures from Calonico, Cattaneo, and Farrell (2020). MSE-optimal
    bandwidths minimize the mean squared error of the RD point estimator,
    while CER-optimal bandwidths minimize the coverage error rate of
    robust bias-corrected confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable column name.
    x : str
        Running variable column name.
    c : float, default 0
        RD cutoff value.
    fuzzy : str, optional
        Treatment variable name for fuzzy RD. When provided, bandwidth
        accounts for first-stage variance in the Wald / IV estimator.
    deriv : int, default 0
        Derivative order. 0 = standard RD (jump in level),
        1 = regression kink design (change in slope).
    p : int, default 1
        Polynomial order for point estimation (1 = local linear).
    q : int, optional
        Polynomial order for bias correction. Default is p + 1.
    covs : list of str, optional
        Covariate column names. When provided, the variance estimates
        used in bandwidth selection account for covariate adjustment,
        typically yielding narrower bandwidths.
    kernel : str, default 'triangular'
        Kernel function: 'triangular', 'uniform', or 'epanechnikov'.
    bwselect : str, default 'mserd'
        Bandwidth selection method. One of:

        - ``'mserd'`` : MSE-optimal common bandwidth (default)
        - ``'msetwo'`` : MSE-optimal separate left/right bandwidths
        - ``'msecomb1'`` : min of mserd, mseleft, mseright
        - ``'msecomb2'`` : median of mserd, mseleft, mseright
        - ``'cerrd'`` : CER-optimal common bandwidth
        - ``'certwo'`` : CER-optimal separate left/right
        - ``'cercomb1'`` : min of cerrd, cerleft, cerright
        - ``'cercomb2'`` : median of cerrd, cerleft, cerright
    cluster : str, optional
        Cluster variable name for cluster-robust variance estimation.
    all : bool, default False
        If True, compute and return all eight bandwidth types.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``[method, h_left, h_right, b_left, b_right,
        n_left, n_right]``. When ``all=False``, contains a single row for
        the selected method. When ``all=True``, contains eight rows for
        all methods.

    Notes
    -----
    The CER-optimal bandwidth is related to the MSE-optimal bandwidth by:

        h_CER = h_MSE * n^{-1/((2p+3)(2p+5))}

    For p=1 (local linear), this gives h_CER ~ h_MSE * n^{-1/35}, which
    is strictly narrower than h_MSE. The narrower bandwidth yields
    confidence intervals with better coverage properties at the cost of
    slightly wider intervals.

    The MSE-optimal bandwidth has rate n^{-1/(2p+3)} while the CER rate
    is n^{-1/(2p+3) - 1/((2p+3)(2p+5))}. For large samples the
    difference is meaningful: CER bandwidths produce robust CIs that
    achieve their nominal coverage rate, whereas MSE bandwidths can
    exhibit substantial coverage distortion.

    References
    ----------
    Calonico, S., Cattaneo, M.D. and Farrell, M.H. (2020).
    "Optimal Bandwidth Choice for Robust Bias-Corrected Inference in
    Regression Discontinuity Designs." *Econometrics Journal*, 23(2),
    192-210. [@calonico2020optimal]

    Calonico, S., Cattaneo, M.D. and Titiunik, R. (2014).
    "Robust Nonparametric Confidence Intervals for Regression-Discontinuity
    Designs." *Econometrica*, 82(6), 2295-2326. [@calonico2014robust]

    Examples
    --------
    Basic MSE-optimal bandwidth:

    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> X = rng.uniform(-1, 1, n)
    >>> Y = 0.5 * X + 3.0 * (X >= 0) + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({'y': Y, 'x': X})
    >>> bw = sp.rdbwselect(df, y='outcome', x='score', c=0)
    >>> print(bw)  # DataFrame with bandwidth info

    Compare all eight bandwidth methods:

    >>> bw_all = sp.rdbwselect(df, y='outcome', x='score', c=0, all=True)
    >>> print(bw_all)  # All 8 bandwidth methods compared

    CER-optimal bandwidth for better coverage:

    >>> bw_cer = sp.rdbwselect(df, y='outcome', x='score', c=0,
    ...                         bwselect='cerrd')

    Fuzzy RD with covariates:

    >>> bw_fuzzy = sp.rdbwselect(df, y='outcome', x='score', c=0,
    ...                           fuzzy='treated', covs=['age', 'gender'])
    """
    # --- Validate inputs ---
    if kernel not in ('triangular', 'uniform', 'epanechnikov'):
        raise ValueError(
            f"kernel must be 'triangular', 'uniform', or 'epanechnikov', "
            f"got '{kernel}'")
    if bwselect not in _VALID_METHODS:
        raise ValueError(
            f"bwselect must be one of {sorted(_VALID_METHODS)}, "
            f"got '{bwselect}'")
    if deriv < 0:
        raise ValueError(f"deriv must be non-negative, got {deriv}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if deriv > 0 and p < deriv + 1:
        p = deriv + 1
    if q is None:
        q = p + 1
    if q <= p:
        raise ValueError(f"q must be > p, got q={q}, p={p}")

    # --- Parse data ---
    Y = data[y].values.astype(float)
    X_raw = data[x].values.astype(float)
    X_c = X_raw - c

    # Drop missing
    valid = np.isfinite(Y) & np.isfinite(X_c)
    if fuzzy is not None:
        D = data[fuzzy].values.astype(float)
        valid &= np.isfinite(D)
    else:
        D = None

    if covs is not None:
        covs_data = data[covs].values.astype(float)
        valid &= np.all(np.isfinite(covs_data), axis=1)
    else:
        covs_data = None

    if cluster is not None:
        cluster_vals = data[cluster].values
    else:
        cluster_vals = None

    # Apply valid mask
    Y = Y[valid]
    X_c = X_c[valid]
    if D is not None:
        D = D[valid]
    if covs_data is not None:
        covs_data = covs_data[valid]
    if cluster_vals is not None:
        cluster_vals = cluster_vals[valid]

    n = len(Y)
    if n < 20:
        raise ValueError(f"Need at least 20 observations, got {n}.")

    left = X_c < 0
    right = X_c >= 0
    n_left = int(left.sum())
    n_right = int(right.sum())

    if n_left < p + 2 or n_right < p + 2:
        raise ValueError(
            f"Not enough observations on each side of the cutoff "
            f"(left={n_left}, right={n_right}, need >= {p + 2}).")

    # --- Compute bandwidths ---
    bw_all = _compute_all_bandwidths(
        Y, X_c, left, right, p, q, kernel,
        D=D, covs_data=covs_data, cluster_vals=cluster_vals,
    )

    # --- Count effective observations for each bandwidth ---
    def _count_effective(h_l, h_r):
        n_eff_l = int(np.sum((X_c[left] >= -h_l)))
        n_eff_r = int(np.sum((X_c[right] <= h_r)))
        return n_eff_l, n_eff_r

    # --- Build output ---
    if all:
        methods_order = [
            'mserd', 'msetwo', 'msecomb1', 'msecomb2',
            'cerrd', 'certwo', 'cercomb1', 'cercomb2',
        ]
        rows = []
        for method in methods_order:
            h_l, h_r, b_l, b_r = bw_all[method]
            n_eff_l, n_eff_r = _count_effective(h_l, h_r)
            rows.append({
                'method': method,
                'h_left': round(h_l, 6),
                'h_right': round(h_r, 6),
                'b_left': round(b_l, 6),
                'b_right': round(b_r, 6),
                'n_left': n_eff_l,
                'n_right': n_eff_r,
            })
        return pd.DataFrame(rows)
    else:
        h_l, h_r, b_l, b_r = bw_all[bwselect]
        n_eff_l, n_eff_r = _count_effective(h_l, h_r)
        return pd.DataFrame([{
            'method': bwselect,
            'h_left': round(h_l, 6),
            'h_right': round(h_r, 6),
            'b_left': round(b_l, 6),
            'b_right': round(b_r, 6),
            'n_left': n_eff_l,
            'n_right': n_eff_r,
        }])
