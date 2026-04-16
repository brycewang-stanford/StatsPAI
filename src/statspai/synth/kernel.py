"""
Kernel-based Nonlinear Synthetic Control Method.

Relaxes the standard SCM linearity assumption by mapping donor outcomes
into a reproducing kernel Hilbert space (RKHS).  The synthetic control
weights are found by minimising the MMD (Maximum Mean Discrepancy) in
that feature space, allowing the method to capture nonlinear donor
relationships.

Two estimators are provided:

* ``kernel_synth``  — Constrained (w >= 0, sum(w) = 1) kernel SCM.
* ``kernel_ridge_synth`` — Kernel ridge regression (no constraints,
  regularised).

Supported kernels: RBF (Gaussian), polynomial, Laplacian.

References
----------
Scholkopf, B. and Smola, A.J. (2002).
"Learning with Kernels: Support Vector Machines, Regularization,
Optimization, and Beyond." MIT Press.

Kloft, M. and Blanchard, G. (2011).
"The Local Rademacher Complexity of Lp-Norm Multiple Kernel Learning."
*Advances in Neural Information Processing Systems (NeurIPS)*.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats as sp_stats

from ..core.results import CausalResult

# ====================================================================== #
#  Kernel functions
# ====================================================================== #


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """RBF (Gaussian) kernel: k(x, y) = exp(-||x - y||^2 / (2 sigma^2)).

    Parameters
    ----------
    X : array of shape (n, d)
    Y : array of shape (m, d)
    sigma : bandwidth

    Returns
    -------
    K : array of shape (n, m)
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    sq_dists = (
        np.sum(X ** 2, axis=1, keepdims=True)
        - 2.0 * X @ Y.T
        + np.sum(Y ** 2, axis=1, keepdims=True).T
    )
    # Clamp negative values from floating-point noise
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def _polynomial_kernel(
    X: np.ndarray, Y: np.ndarray, degree: int = 2, c: float = 1.0,
) -> np.ndarray:
    """Polynomial kernel: k(x, y) = (x'y + c)^d.

    Parameters
    ----------
    X : array of shape (n, d)
    Y : array of shape (m, d)
    degree : polynomial degree
    c : additive constant

    Returns
    -------
    K : array of shape (n, m)
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    return (X @ Y.T + c) ** degree


def _laplacian_kernel(
    X: np.ndarray, Y: np.ndarray, sigma: float,
) -> np.ndarray:
    """Laplacian kernel: k(x, y) = exp(-||x - y||_1 / sigma).

    Parameters
    ----------
    X : array of shape (n, d)
    Y : array of shape (m, d)
    sigma : bandwidth

    Returns
    -------
    K : array of shape (n, m)
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    # L1 pairwise distances
    l1_dists = np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
    return np.exp(-l1_dists / sigma)


def _median_heuristic(X: np.ndarray) -> float:
    """Compute median pairwise Euclidean distance (bandwidth heuristic).

    Parameters
    ----------
    X : array of shape (n, d)

    Returns
    -------
    sigma : float
        Median of all pairwise Euclidean distances (clamped >= 1e-6).
    """
    X = np.atleast_2d(X)
    n = X.shape[0]
    if n < 2:
        return 1.0
    sq_dists = (
        np.sum(X ** 2, axis=1, keepdims=True)
        - 2.0 * X @ X.T
        + np.sum(X ** 2, axis=1, keepdims=True).T
    )
    sq_dists = np.maximum(sq_dists, 0.0)
    # Extract upper triangle (no diagonal)
    triu_idx = np.triu_indices(n, k=1)
    dists = np.sqrt(sq_dists[triu_idx])
    med = float(np.median(dists))
    return max(med, 1e-6)


# ====================================================================== #
#  Kernel matrix builders
# ====================================================================== #

_KERNEL_DISPATCH = {
    "rbf": lambda X, Y, s, d: _rbf_kernel(X, Y, s),
    "polynomial": lambda X, Y, s, d: _polynomial_kernel(X, Y, degree=d),
    "laplacian": lambda X, Y, s, d: _laplacian_kernel(X, Y, s),
}


def _compute_kernel_matrix(
    Y0_pre: np.ndarray,
    kernel: str,
    sigma: float,
    degree: int,
) -> np.ndarray:
    """Compute donor-donor kernel matrix K  (J x J).

    Parameters
    ----------
    Y0_pre : array of shape (J, T0)  — donor pre-treatment outcomes
    kernel : kernel name
    sigma : bandwidth
    degree : polynomial degree (ignored for non-polynomial)

    Returns
    -------
    K : array of shape (J, J)
    """
    fn = _KERNEL_DISPATCH[kernel]
    return fn(Y0_pre, Y0_pre, sigma, degree)


def _compute_kernel_vector(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    kernel: str,
    sigma: float,
    degree: int,
) -> np.ndarray:
    """Compute treated-vs-donor kernel vector k(Y1)  (J,).

    Parameters
    ----------
    Y1_pre : array of shape (T0,) or (1, T0) — treated unit pre-treatment
    Y0_pre : array of shape (J, T0) — donor pre-treatment
    kernel : kernel name
    sigma : bandwidth
    degree : polynomial degree

    Returns
    -------
    k_vec : array of shape (J,)
    """
    fn = _KERNEL_DISPATCH[kernel]
    Y1_2d = np.atleast_2d(Y1_pre)
    return fn(Y0_pre, Y1_2d, sigma, degree).ravel()


def _kernel_weights(K: np.ndarray, k_vec: np.ndarray) -> np.ndarray:
    """Solve constrained QP for kernel SCM weights.

    Minimise  K(Y1,Y1) - 2 w' k(Y1) + w' K w
    subject to  w >= 0,  sum(w) = 1.

    Since K(Y1,Y1) is constant w.r.t. w, we minimise:
        w' K w - 2 w' k_vec

    Parameters
    ----------
    K : array of shape (J, J) — donor kernel matrix
    k_vec : array of shape (J,) — treated-vs-donor kernel vector

    Returns
    -------
    w : array of shape (J,) — optimal weights
    """
    J = K.shape[0]

    # Regularise K slightly for numerical stability
    K_reg = K + 1e-8 * np.eye(J)

    def objective(w: np.ndarray) -> float:
        return float(w @ K_reg @ w - 2.0 * w @ k_vec)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * K_reg @ w - 2.0 * k_vec

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * J
    w0 = np.ones(J) / J

    res = optimize.minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = res.x
    # Project onto simplex (clamp tiny negatives)
    w = np.maximum(w, 0.0)
    w /= w.sum() if w.sum() > 0 else 1.0
    return w


# ====================================================================== #
#  Panel data reshaping
# ====================================================================== #


def _reshape_panel(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit,
    treatment_time,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    List, List, List,
]:
    """Reshape long-format panel into treated/donor matrices.

    Returns
    -------
    Y1_pre, Y1_post, Y0_pre, Y0_post, donors, pre_times, post_times
    """
    panel = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(panel.columns)
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError(
            f"Need at least 2 pre-treatment periods, got {len(pre_times)}"
        )
    if len(post_times) < 1:
        raise ValueError("Need at least 1 post-treatment period")

    if treated_unit not in panel.index:
        raise ValueError(f"Treated unit '{treated_unit}' not found in data")

    Y1_pre = panel.loc[treated_unit, pre_times].values.astype(np.float64)
    Y1_post = panel.loc[treated_unit, post_times].values.astype(np.float64)

    donors = [u for u in panel.index if u != treated_unit]
    if len(donors) < 2:
        raise ValueError(
            f"Need at least 2 donor units, got {len(donors)}"
        )

    Y0_pre = panel.loc[donors, pre_times].values.astype(np.float64)   # (J, T0)
    Y0_post = panel.loc[donors, post_times].values.astype(np.float64)  # (J, T1)

    return Y1_pre, Y1_post, Y0_pre, Y0_post, donors, pre_times, post_times


# ====================================================================== #
#  Placebo inference (shared)
# ====================================================================== #


def _placebo_inference(
    Y0_pre: np.ndarray,
    Y0_post: np.ndarray,
    donors: List,
    kernel: str,
    sigma: float,
    degree: int,
    att: float,
    use_ridge: bool = False,
    ridge_lambda: float = 0.01,
) -> Tuple[float, float, np.ndarray]:
    """In-space placebo permutation inference.

    For each donor j, treat it as the pseudo-treated unit and compute
    its placebo ATT using the remaining donors.

    Returns
    -------
    se, pvalue, placebo_effects
    """
    J = Y0_pre.shape[0]
    placebo_effects: List[float] = []

    for j in range(J):
        other_idx = [i for i in range(J) if i != j]
        y1_pre_plac = Y0_pre[j]
        y1_post_plac = Y0_post[j]
        y0_pre_plac = Y0_pre[other_idx]
        y0_post_plac = Y0_post[other_idx]

        if use_ridge:
            K_plac = _compute_kernel_matrix(y0_pre_plac, kernel, sigma, degree)
            k_plac = _compute_kernel_vector(
                y1_pre_plac, y0_pre_plac, kernel, sigma, degree,
            )
            J_plac = K_plac.shape[0]
            beta_plac = np.linalg.solve(
                K_plac + ridge_lambda * np.eye(J_plac), k_plac,
            )
            synth_post = y0_post_plac.T @ beta_plac
        else:
            K_plac = _compute_kernel_matrix(y0_pre_plac, kernel, sigma, degree)
            k_plac = _compute_kernel_vector(
                y1_pre_plac, y0_pre_plac, kernel, sigma, degree,
            )
            w_plac = _kernel_weights(K_plac, k_plac)
            synth_post = y0_post_plac.T @ w_plac

        plac_att = float(np.mean(y1_post_plac - synth_post))
        placebo_effects.append(plac_att)

    placebo_arr = np.array(placebo_effects)
    se = float(np.std(placebo_arr, ddof=1)) if J > 1 else 0.0
    pvalue = float(np.mean(np.abs(placebo_arr) >= abs(att)))
    pvalue = max(pvalue, 1.0 / (J + 1))  # minimum p-value bound
    return se, pvalue, placebo_arr


# ====================================================================== #
#  Public API — Kernel Synthetic Control
# ====================================================================== #


def kernel_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit,
    treatment_time,
    kernel: str = "rbf",
    sigma: Optional[float] = None,
    degree: int = 2,
    covariates: Optional[List[str]] = None,
    placebo: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """Kernel-based Nonlinear Synthetic Control Method.

    Standard SCM assumes the counterfactual is a *linear* combination of
    donors.  This estimator lifts the donor panel into a reproducing kernel
    Hilbert space (RKHS) and solves for synthetic control weights in that
    feature space, capturing nonlinear donor relationships.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format with columns for unit, time, and outcome.
    outcome : str
        Name of the outcome variable.
    unit : str
        Column identifying panel units.
    time : str
        Column identifying time periods.
    treated_unit
        Identifier of the treated unit.
    treatment_time
        First treatment period (inclusive).
    kernel : ``{'rbf', 'polynomial', 'laplacian'}``, default ``'rbf'``
        Kernel function to use.
    sigma : float or None
        Bandwidth for RBF / Laplacian kernels.  If *None*, the median
        heuristic is used (recommended).
    degree : int, default 2
        Degree for the polynomial kernel (ignored otherwise).
    covariates : list of str or None
        Additional pre-treatment covariates to include in the feature
        vector.  If provided, each donor row is ``[outcomes | covariates]``.
    placebo : bool, default True
        Whether to run in-space placebo permutation for inference.
    alpha : float, default 0.05
        Significance level for the confidence interval.

    Returns
    -------
    CausalResult
        Unified result with ATT estimate, SE, p-value, CI, and
        period-level effects in ``detail``.

    Notes
    -----
    The optimisation solved is:

    .. math::

        \\min_{w \\ge 0,\\, \\sum w = 1}
        \\bigl[K(Y_1, Y_1) - 2\\,w^\\top k(Y_1) + w^\\top K\\,w\\bigr]

    where :math:`K_{ij} = k(Y_{0,i},\\, Y_{0,j})` is the donor kernel
    matrix and :math:`k(Y_1)_j = k(Y_1,\\, Y_{0,j})`.

    References
    ----------
    Scholkopf, B. and Smola, A.J. (2002). "Learning with Kernels."
    """
    if kernel not in _KERNEL_DISPATCH:
        raise ValueError(
            f"Unknown kernel '{kernel}'. "
            f"Choose from {list(_KERNEL_DISPATCH.keys())}."
        )

    # --- Reshape panel ---
    Y1_pre, Y1_post, Y0_pre, Y0_post, donors, pre_times, post_times = (
        _reshape_panel(data, outcome, unit, time, treated_unit, treatment_time)
    )

    # Append covariates if provided
    if covariates:
        panel_cov = data.pivot_table(index=unit, columns=time, values=outcome)
        cov_means = data.groupby(unit)[covariates].mean()
        Y1_cov = cov_means.loc[treated_unit].values.astype(np.float64)
        Y0_cov = cov_means.loc[donors].values.astype(np.float64)
        Y1_pre = np.concatenate([Y1_pre, Y1_cov])
        Y0_pre = np.hstack([Y0_pre, Y0_cov])

    J = len(donors)
    T1 = len(post_times)

    # --- Bandwidth ---
    if sigma is None:
        # Stack all units (treated + donors) for median heuristic
        all_units = np.vstack([Y1_pre.reshape(1, -1), Y0_pre])
        sigma = _median_heuristic(all_units)

    # --- Kernel matrices ---
    K = _compute_kernel_matrix(Y0_pre, kernel, sigma, degree)
    k_vec = _compute_kernel_vector(Y1_pre, Y0_pre, kernel, sigma, degree)

    # --- Solve for weights ---
    w = _kernel_weights(K, k_vec)

    # --- Counterfactual & effects ---
    synth_pre = Y0_pre[:, :len(pre_times)].T @ w   # (T0,)
    synth_post = Y0_post.T @ w                       # (T1,)
    # Use only outcome columns for pre-RMSPE (exclude appended covariates)
    pre_rmspe = float(
        np.sqrt(np.mean((Y1_pre[:len(pre_times)] - synth_pre) ** 2))
    )

    effects = Y1_post - synth_post
    att = float(np.mean(effects))
    post_rmspe = float(np.sqrt(np.mean(effects ** 2)))

    # --- Inference ---
    if placebo and J >= 3:
        se, pvalue, placebo_effects = _placebo_inference(
            Y0_pre[:, :len(pre_times)], Y0_post, donors,
            kernel, sigma, degree, att,
        )
    else:
        se = 0.0
        pvalue = np.nan
        placebo_effects = np.array([])

    t_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (att - t_crit * se, att + t_crit * se) if se > 0 else (np.nan, np.nan)

    # --- Effects DataFrame ---
    effects_df = pd.DataFrame({
        "time": post_times,
        "treated": Y1_post,
        "counterfactual": synth_post,
        "effect": effects,
    })

    return CausalResult(
        method="Kernel Synthetic Control",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info={
            "model_type": "Kernel Synthetic Control",
            "weights": dict(zip(donors, w.tolist())),
            "kernel_type": kernel,
            "sigma": sigma,
            "degree": degree if kernel == "polynomial" else None,
            "kernel_matrix": K,
            "pre_rmspe": pre_rmspe,
            "post_rmspe": post_rmspe,
            "n_donors": J,
            "n_pre_periods": len(pre_times),
            "n_post_periods": T1,
            "effects_by_period": effects_df,
            "placebo_distribution": placebo_effects,
        },
    )


# ====================================================================== #
#  Public API — Kernel Ridge Synthetic Control
# ====================================================================== #


def kernel_ridge_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit,
    treatment_time,
    kernel: str = "rbf",
    sigma: Optional[float] = None,
    degree: int = 2,
    ridge_lambda: float = 0.01,
    covariates: Optional[List[str]] = None,
    placebo: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """Kernel Ridge Regression Synthetic Control.

    Instead of constrained simplex weights, this estimator uses kernel
    ridge regression to learn the mapping from donors to the treated unit.
    The ridge penalty ``lambda`` prevents overfitting when the number of
    donors is small relative to pre-treatment periods.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    outcome : str
        Outcome variable name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit
        Identifier of the treated unit.
    treatment_time
        First treatment period (inclusive).
    kernel : ``{'rbf', 'polynomial', 'laplacian'}``, default ``'rbf'``
        Kernel function.
    sigma : float or None
        Bandwidth (None = median heuristic).
    degree : int, default 2
        Polynomial kernel degree.
    ridge_lambda : float, default 0.01
        Regularisation parameter.  Larger values shrink the coefficient
        vector toward zero.
    covariates : list of str or None
        Additional pre-treatment covariates.
    placebo : bool, default True
        Run placebo permutation inference.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult

    Notes
    -----
    The solution is:

    .. math::

        \\beta = (K + \\lambda I)^{-1}\\, k(Y_1)

    and the counterfactual is :math:`\\hat{Y}_{1,\\text{post}} =
    Y_{0,\\text{post}}^\\top \\beta`.

    No non-negativity or sum-to-one constraints are imposed, which gives
    the estimator more flexibility but may produce extrapolation.
    """
    if kernel not in _KERNEL_DISPATCH:
        raise ValueError(
            f"Unknown kernel '{kernel}'. "
            f"Choose from {list(_KERNEL_DISPATCH.keys())}."
        )
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")

    # --- Reshape panel ---
    Y1_pre, Y1_post, Y0_pre, Y0_post, donors, pre_times, post_times = (
        _reshape_panel(data, outcome, unit, time, treated_unit, treatment_time)
    )

    # Append covariates
    if covariates:
        cov_means = data.groupby(unit)[covariates].mean()
        Y1_cov = cov_means.loc[treated_unit].values.astype(np.float64)
        Y0_cov = cov_means.loc[donors].values.astype(np.float64)
        Y1_pre = np.concatenate([Y1_pre, Y1_cov])
        Y0_pre = np.hstack([Y0_pre, Y0_cov])

    J = len(donors)
    T1 = len(post_times)

    # --- Bandwidth ---
    if sigma is None:
        all_units = np.vstack([Y1_pre.reshape(1, -1), Y0_pre])
        sigma = _median_heuristic(all_units)

    # --- Kernel matrices ---
    K = _compute_kernel_matrix(Y0_pre, kernel, sigma, degree)
    k_vec = _compute_kernel_vector(Y1_pre, Y0_pre, kernel, sigma, degree)

    # --- Solve kernel ridge ---
    beta = np.linalg.solve(K + ridge_lambda * np.eye(J), k_vec)

    # --- Counterfactual & effects ---
    synth_pre = Y0_pre[:, :len(pre_times)].T @ beta
    synth_post = Y0_post.T @ beta

    pre_rmspe = float(
        np.sqrt(np.mean((Y1_pre[:len(pre_times)] - synth_pre) ** 2))
    )

    effects = Y1_post - synth_post
    att = float(np.mean(effects))
    post_rmspe = float(np.sqrt(np.mean(effects ** 2)))

    # --- Inference ---
    if placebo and J >= 3:
        se, pvalue, placebo_effects = _placebo_inference(
            Y0_pre[:, :len(pre_times)], Y0_post, donors,
            kernel, sigma, degree, att,
            use_ridge=True, ridge_lambda=ridge_lambda,
        )
    else:
        se = 0.0
        pvalue = np.nan
        placebo_effects = np.array([])

    t_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (att - t_crit * se, att + t_crit * se) if se > 0 else (np.nan, np.nan)

    # --- Effects DataFrame ---
    effects_df = pd.DataFrame({
        "time": post_times,
        "treated": Y1_post,
        "counterfactual": synth_post,
        "effect": effects,
    })

    return CausalResult(
        method="Kernel Ridge Synthetic Control",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info={
            "model_type": "Kernel Ridge Synthetic Control",
            "weights": dict(zip(donors, beta.tolist())),
            "kernel_type": kernel,
            "sigma": sigma,
            "degree": degree if kernel == "polynomial" else None,
            "ridge_lambda": ridge_lambda,
            "kernel_matrix": K,
            "pre_rmspe": pre_rmspe,
            "post_rmspe": post_rmspe,
            "n_donors": J,
            "n_pre_periods": len(pre_times),
            "n_post_periods": T1,
            "effects_by_period": effects_df,
            "placebo_distribution": placebo_effects,
        },
    )
