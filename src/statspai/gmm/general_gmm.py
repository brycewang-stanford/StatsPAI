"""
General GMM (Generalized Method of Moments) framework.

Provides a flexible GMM estimator for arbitrary moment conditions,
with one-step, two-step, and continuously-updated (CUE) options.

Equivalent to Stata's ``gmm`` and R's ``gmm::gmm()``.

References
----------
Hansen, L.P. (1982).
"Large Sample Properties of Generalized Method of Moments Estimators."
*Econometrica*, 50(4), 1029-1054. [@hansen1982large]

Hansen, L.P., Heaton, J. & Yaron, A. (1996).
"Finite-Sample Properties of Some Alternative GMM Estimators."
*Journal of Business & Economic Statistics*, 14(3), 262-280. [@hansen1996finite]
"""

from typing import Any, Callable, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.results import EconometricResults


def gmm(
    moment_fn: Callable[[np.ndarray, Optional[pd.DataFrame]], Any],
    theta0: np.ndarray,
    data: Optional[pd.DataFrame] = None,
    W: Optional[np.ndarray] = None,
    method: str = "twostep",
    se: str = "robust",
    maxiter: int = 200,
    tol: float = 1e-8,
    param_names: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    General GMM estimator for arbitrary moment conditions.

    Minimizes Q(θ) = ḡ(θ)' W ḡ(θ) where ḡ(θ) = (1/n) Σ g_i(θ).

    Parameters
    ----------
    moment_fn : callable
        Function g(theta, data) -> np.ndarray of shape (n, q)
        returning moment conditions for each observation.
    theta0 : np.ndarray
        Initial parameter vector.
    data : pd.DataFrame, optional
        Data passed to moment_fn.
    W : np.ndarray, optional
        Weighting matrix (q x q). If None, uses identity (one-step)
        then optimal (two-step).
    method : str, default 'twostep'
        'onestep', 'twostep', 'iterative', 'cue' (continuously updated).
    se : str, default 'robust'
        'robust' (HAC-type), 'unadjusted'.
    maxiter : int, default 200
    tol : float, default 1e-8
    param_names : list of str, optional
        Names for parameters.
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> z1, z2, u = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    >>> x1 = 0.7 * z1 + 0.5 * z2 + u + rng.normal(size=n)
    >>> y = 1.0 + 2.0 * x1 + u + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'x1': x1, 'z1': z1, 'z2': z2})
    >>>
    >>> # IV-GMM example
    >>> def moment_fn(theta, data):
    ...     y, X, Z = data['y'].values, data[['x1']].values, data[['z1', 'z2']].values
    ...     X_full = np.column_stack([np.ones(len(y)), X])
    ...     resid = y - X_full @ theta
    ...     Z_full = np.column_stack([np.ones(len(y)), Z])
    ...     return resid[:, np.newaxis] * Z_full  # n x q moment conditions
    >>>
    >>> result = sp.gmm(moment_fn, theta0=np.zeros(2), data=df,
    ...                 param_names=['_cons', 'x1'])
    >>> bool(result is not None)
    True
    """
    k = len(theta0)
    G0 = np.asarray(moment_fn(theta0, data), dtype=float)
    n = int(G0.shape[0])
    q = int(G0.shape[1])

    def g_bar(theta: np.ndarray) -> np.ndarray:
        """Average moment conditions."""
        G = G_mat(theta)
        return np.asarray(G.mean(axis=0), dtype=float)

    def G_mat(theta: np.ndarray) -> np.ndarray:
        """Individual moment conditions (n x q)."""
        return np.asarray(moment_fn(theta, data), dtype=float)

    def objective(theta: np.ndarray, W_mat: np.ndarray) -> float:
        """GMM objective function."""
        gb = g_bar(theta)
        return float(gb @ W_mat @ gb)

    if method == "cue":
        # Continuously Updated Estimator
        def cue_objective(theta: np.ndarray) -> float:
            G = G_mat(theta)
            gb = G.mean(axis=0)
            S = G.T @ G / n
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.eye(q)
            return float(gb @ S_inv @ gb)

        result = minimize(
            cue_objective,
            theta0,
            method="BFGS",
            options={"maxiter": maxiter, "gtol": tol},
        )
        theta_hat = result.x
        G_hat = G_mat(theta_hat)
        S_hat = G_hat.T @ G_hat / n
        try:
            W_opt = np.linalg.inv(S_hat)
        except np.linalg.LinAlgError:
            W_opt = np.eye(q)
    else:
        # Step 1: Identity weighting (or provided W)
        if W is None:
            W1 = np.eye(q)
        else:
            W1 = W

        result1 = minimize(
            lambda t: objective(t, W1),
            theta0,
            method="BFGS",
            options={"maxiter": maxiter, "gtol": tol},
        )
        theta1 = result1.x

        if method == "onestep":
            theta_hat = theta1
            W_opt = W1
        else:
            # Optimal weighting matrix from first-step residuals
            G1 = G_mat(theta1)
            S1 = G1.T @ G1 / n
            try:
                W_opt = np.linalg.inv(S1)
            except np.linalg.LinAlgError:
                W_opt = np.eye(q)

            result2 = minimize(
                lambda t: objective(t, W_opt),
                theta1,
                method="BFGS",
                options={"maxiter": maxiter, "gtol": tol},
            )
            theta_hat = result2.x

            if method == "iterative":
                for _ in range(maxiter):
                    theta_old = theta_hat.copy()
                    G_iter = G_mat(theta_hat)
                    S_iter = G_iter.T @ G_iter / n
                    try:
                        W_opt = np.linalg.inv(S_iter)
                    except np.linalg.LinAlgError:
                        break
                    result_iter = minimize(
                        lambda t: objective(t, W_opt),
                        theta_hat,
                        method="BFGS",
                        options={"maxiter": 50},
                    )
                    theta_hat = result_iter.x
                    if np.max(np.abs(theta_hat - theta_old)) < tol:
                        break

    # Standard errors
    G_hat = G_mat(theta_hat)

    # Jacobian: D = (1/n) Σ ∂g_i/∂θ' (numerical)
    eps = 1e-6
    D = np.zeros((q, k))
    for j in range(k):
        ej = np.zeros(k)
        ej[j] = eps
        D[:, j] = (g_bar(theta_hat + ej) - g_bar(theta_hat - ej)) / (2 * eps)

    # Variance: V = (1/n) (D'WD)^{-1} D'W S W D (D'WD)^{-1}
    S_hat = G_hat.T @ G_hat / n
    DtW = D.T @ W_opt
    DtWD = DtW @ D
    try:
        DtWD_inv = np.linalg.inv(DtWD)
    except np.linalg.LinAlgError:
        DtWD_inv = np.eye(k)

    if se == "robust":
        V = DtWD_inv @ DtW @ S_hat @ DtW.T @ DtWD_inv / n
    else:
        V = DtWD_inv / n

    se_hat = np.sqrt(np.abs(np.diag(V)))

    # J-test (overidentification)
    J_stat = n * objective(theta_hat, W_opt)
    J_df = q - k
    J_p = 1 - stats.chi2.cdf(J_stat, J_df) if J_df > 0 else np.nan

    if param_names is None:
        param_names = [f"theta_{i}" for i in range(k)]

    params = pd.Series(theta_hat, index=param_names)
    std_errors = pd.Series(se_hat, index=param_names)

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            "model_type": f"GMM ({method})",
            "n_moments": q,
            "n_params": k,
            "overidentified": q > k,
        },
        data_info={
            "n_obs": n,
            "df_resid": n - k,
        },
        diagnostics={
            "J_stat": J_stat,
            "J_df": J_df,
            "J_p": J_p,
            "gmm_objective": objective(theta_hat, W_opt),
        },
    )
