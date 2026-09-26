"""
Interactive Fixed Effects estimator (Bai 2009).

Estimates panel models with unobserved interactive effects:

    Y_{it} = X_{it}'β + λ_i' f_t + ε_{it}

where λ_i are factor loadings and f_t are common factors.
This generalizes additive FE (α_i + γ_t) to allow unit-specific
responses to common shocks.

The point estimate is the least-squares fixed point computed by R's
``phtt::Eup(additive.effects = "none")`` and Stata's ``regife ...,
noconstant`` (SSC); the standard errors are Bai's (2009) ``D0`` sandwich,
see :func:`interactive_fe`.

References
----------
Bai, J. (2009).
"Panel Data Models with Interactive Fixed Effects."
*Econometrica*, 77(4), 1229-1279.

Moon, H.R. & Weidner, M. (2015).
"Linear Regression for Panel with Unknown Number of Factors as
Interactive Fixed Effects." *Econometrica*, 83(4), 1543-1579. [@moon2015linear]
"""

import warnings
from typing import List

import numpy as np
import pandas as pd

from ..core._validate import require_bool_flag
from ..core.results import EconometricResults
from ..exceptions import MethodIncompatibility


def interactive_fe(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id: str = "id",
    time: str = "time",
    n_factors: int = 1,
    method: str = "iterative",
    maxiter: int = 10000,
    tol: float = 1e-10,
    robust: bool = True,
    alpha: float = 0.05,
    dof: str = "exact",
) -> EconometricResults:
    """
    Interactive fixed effects estimator (Bai 2009).

    Estimates Y_{it} = X_{it}'β + λ_i' f_t + ε_{it}
    where λ_i (N×r) are unit loadings and f_t (T×r) are time factors.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel data.
    y : str
        Dependent variable.
    x : list of str
        Regressors.
    id : str, default 'id'
        Unit identifier.
    time : str, default 'time'
        Time identifier.
    n_factors : int, default 1
        Number of interactive factors (r).
    method : str, default 'iterative'
        Bai's (2009) iterated least squares: alternate principal components
        of ``Y − Xβ`` (factors) and least squares of ``Y M_F`` on ``X M_F``
        (slopes) until the slopes move by less than ``tol``.  ``'pca'`` is
        accepted for backward compatibility and runs the same algorithm (it
        always did); it is deprecated.
    maxiter : int, default 10000
    tol : float, default 1e-10
        Convergence criterion on max |Δβ|.  The iteration converges
        linearly, so the default is tight (``regife``'s own default is
        1e-9).
    robust : bool, default True
        ``True``: cluster-robust by unit (Bai's ``D0⁻¹ D_Z D0⁻¹`` with the
        Stata CR1 factor ``G/(G−1)·(NT−1)/(NT−K)``); ``False``: Bai's
        homoskedastic ``σ̂² D0⁻¹``.  In both, the regressors enter through
        ``Z_j = M_Λ X_j M_F`` -- projected off *both* the factor space and
        the loading space.
    alpha : float, default 0.05
    dof : {'exact', 'regife'}, default 'exact'
        Parameters absorbed by the factor structure.  ``'exact'`` counts
        ``r(N + T − r)`` (``ΛF'`` is identified only up to an r×r rotation);
        ``'regife'`` counts ``r(N + T)``, as Stata ``regife`` does through
        ``reghdfe`` (which flags that it could not detect the redundant
        ones).  Enters ``σ̂²`` and ``K`` in the cluster factor.

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> N, T = 30, 10
    >>> f = rng.normal(size=(T, 2))           # 2 common factors
    >>> lam = rng.normal(size=(N, 2))         # unit loadings
    >>> rows = []
    >>> for i in range(N):
    ...     for t in range(T):
    ...         inv, trade = rng.normal(), rng.normal()
    ...         gdp = 0.5 * inv + 0.3 * trade + lam[i] @ f[t] + rng.normal(0, 0.3)
    ...         rows.append({"country": i, "year": t, "gdp": gdp,
    ...                      "investment": inv, "trade": trade})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.interactive_fe(df, y='gdp', x=['investment', 'trade'],
    ...                            id='country', time='year', n_factors=2)
    >>> bool(result.params['investment'] > 0)
    True

    References
    ----------
    Bai, J. (2009). Panel data models with interactive fixed effects.
    *Econometrica*. [@bai2009panel]
    """
    robust = require_bool_flag(robust, argument="robust")
    if method not in ("iterative", "pca"):
        raise MethodIncompatibility(f"method must be 'iterative'; got {method!r}")
    if method == "pca":
        warnings.warn(
            "interactive_fe(method='pca') runs the same iterated least-squares "
            "algorithm as method='iterative' (it always has); the option is "
            "deprecated and will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
    if dof not in ("exact", "regife"):
        raise MethodIncompatibility(f"dof must be 'exact' or 'regife'; got {dof!r}")
    df = data.copy()
    units = df[id].unique()
    times = df[time].unique()
    N = len(units)
    T = len(times)
    k = len(x)
    r = n_factors

    # Create unit and time indices
    unit_map = {u: i for i, u in enumerate(units)}
    time_map = {t: i for i, t in enumerate(times)}
    df["_uid"] = df[id].map(unit_map)
    df["_tid"] = df[time].map(time_map)

    # Reshape to panel matrices (N x T)
    Y_mat = np.full((N, T), np.nan)
    X_mats = [np.full((N, T), np.nan) for _ in range(k)]
    ui = df["_uid"].to_numpy(dtype=int)
    ti = df["_tid"].to_numpy(dtype=int)
    Y_mat[ui, ti] = df[y].to_numpy(dtype=float)
    for j, xvar in enumerate(x):
        X_mats[j][ui, ti] = df[xvar].to_numpy(dtype=float)

    # Bai's estimator needs a balanced panel: units with any missing cell
    # are dropped, loudly.
    x_valid = np.all(
        np.array([np.all(np.isfinite(xm), axis=1) for xm in X_mats]), axis=0
    )
    valid = np.all(np.isfinite(Y_mat), axis=1) & x_valid
    if not np.all(valid):
        warnings.warn(
            f"interactive_fe: dropping {int((~valid).sum())} of {N} units with "
            "missing periods or values; the estimator requires a balanced "
            "panel.",
            RuntimeWarning,
            stacklevel=2,
        )
        Y_mat = Y_mat[valid]
        X_mats = [xm[valid] for xm in X_mats]
        N = Y_mat.shape[0]

    def _factors(beta_):
        E_ = Y_mat.copy()
        for j_ in range(k):
            E_ -= beta_[j_] * X_mats[j_]
        U_, S_, Vt_ = np.linalg.svd(E_, full_matrices=False)
        return E_, U_[:, :r] * S_[:r], Vt_[:r, :].T, S_

    # Iterated least squares (Bai 2009): start from pooled OLS.
    Y_vec = Y_mat.ravel()
    X_pool = np.column_stack([xm.ravel() for xm in X_mats])
    beta = np.linalg.lstsq(X_pool, Y_vec, rcond=None)[0]

    converged = False
    iteration = 0
    for iteration in range(maxiter):
        beta_old = beta.copy()
        _, Lambda, F, _ = _factors(beta)
        M_F = np.eye(T) - F @ np.linalg.solve(F.T @ F, F.T)
        X_stacked = np.column_stack([(xm @ M_F).ravel() for xm in X_mats])
        beta = np.linalg.lstsq(X_stacked, (Y_mat @ M_F).ravel(), rcond=None)[0]
        if np.max(np.abs(beta - beta_old)) < tol:
            converged = True
            break
    if not converged:
        warnings.warn(
            f"interactive_fe: not converged after {maxiter} iterations "
            f"(max |dbeta| = {np.max(np.abs(beta - beta_old)):.2e}).",
            RuntimeWarning,
            stacklevel=2,
        )

    # Factors, loadings and residuals at the final beta.
    E, Lambda, F, S = _factors(beta)
    M_F = np.eye(T) - F @ np.linalg.solve(F.T @ F, F.T)
    M_L = np.eye(N) - Lambda @ np.linalg.solve(Lambda.T @ Lambda, Lambda.T)
    resid_mat = E @ M_F  # = E - Lambda F'
    residuals = resid_mat.ravel()

    # Bai (2009): the slope's influence runs through Z_j = M_L X_j M_F.
    Z_mats = [M_L @ xm @ M_F for xm in X_mats]
    Z = np.column_stack([zm.ravel() for zm in Z_mats])
    ZZ_inv = np.linalg.inv(Z.T @ Z)
    n_obs = N * T
    n_factor_params = r * (N + T - r) if dof == "exact" else r * (N + T)
    df_resid = n_obs - k - n_factor_params
    sigma2 = float(np.sum(residuals**2) / df_resid)

    if robust:
        meat = np.zeros((k, k))
        for i in range(N):
            z_i = np.column_stack([zm[i] for zm in Z_mats])  # T x k
            s_i = z_i.T @ resid_mat[i]
            meat += np.outer(s_i, s_i)
        cr1 = (N / (N - 1)) * ((n_obs - 1) / (n_obs - k - n_factor_params))
        var_cov = cr1 * ZZ_inv @ meat @ ZZ_inv
    else:
        var_cov = sigma2 * ZZ_inv

    se = np.sqrt(np.diag(var_cov))

    params = pd.Series(beta, index=x)
    std_errors = pd.Series(se, index=x)

    # R-squared
    tss = np.sum((Y_mat - Y_mat.mean()) ** 2)
    rss = np.sum(residuals**2)
    r2 = 1 - rss / tss

    # Eigenvalues of E'E / (NT) for factor diagnostics
    eigenvalues = (S[: min(10, len(S))] ** 2) / (N * T)

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            "alpha": alpha,
            "model_type": "Interactive Fixed Effects (Bai 2009)",
            "n_factors": r,
            "method": "iterative",
            "n_iterations": iteration + 1,
            "converged": converged,
            "vce": "cluster(unit), CR1" if robust else "homoskedastic (Bai D0)",
            "dof": dof,
        },
        data_info={
            "n_obs": N * T,
            "n_units": N,
            "n_periods": T,
            "dep_var": y,
            "df_resid": df_resid,
            "var_cov": var_cov,
        },
        diagnostics={
            "r_squared": r2,
            "sigma2": sigma2,
            "eigenvalues": eigenvalues.tolist(),
            "factors": F,
            "loadings": Lambda,
        },
    )
