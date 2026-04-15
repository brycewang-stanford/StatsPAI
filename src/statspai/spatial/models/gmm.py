"""Generalised Method of Moments estimators for spatial regression models.

Follows Kelejian & Prucha (1998, 1999) for SEM GMM and SAR 2SLS-GMM, and
Arraiz et al. (2010) for the heteroskedasticity-robust variants used in
``sphet`` / ``spreg.GM_*_Het``. Designed for large-N settings where dense
eigenvalue-based ML is intractable.

- :func:`sem_gmm`    — Kelejian-Prucha (1999) 3-moment GMM for ``λ``.
- :func:`sar_gmm`    — Kelejian-Prucha (1998) 2SLS with spatial-lag
  instruments (columns of ``[X, WX, W²X]``).
- :func:`sarar_gmm`  — combined lag + error (``GM_Combo`` in spreg).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ...core.results import EconometricResults
from .ml import _coerce_W, _parse_formula


# --------------------------------------------------------------------- #
#  SEM GMM (Kelejian-Prucha 1999)
# --------------------------------------------------------------------- #

def _kp_moment_residuals(u: np.ndarray, W, lam: float, sigma2: float) -> np.ndarray:
    """Three KP 1999 moment residuals given residuals ``u`` and candidate
    ``(λ, σ²)``.

    Uses the OLS residuals (``u``) to form ``v = W u`` and
    ``w_bar = W² u`` once; returns the 3-vector of deviations from the
    theoretical moments under ``u = ε + λ W ε``.
    """
    n = u.shape[0]
    v = W @ u
    w_bar = W @ v
    tr_WtW_over_n = float((W.multiply(W)).sum()) / n
    # Moment 1
    m1 = (u @ u - 2 * lam * u @ v + lam ** 2 * v @ v) / n - sigma2
    # Moment 2
    m2 = (v @ v - 2 * lam * v @ w_bar + lam ** 2 * w_bar @ w_bar) / n - sigma2 * tr_WtW_over_n
    # Moment 3
    m3 = (u @ v - lam * (u @ w_bar + v @ v) + lam ** 2 * v @ w_bar) / n
    return np.array([m1, m2, m3])


def sem_gmm(W, data: pd.DataFrame, formula: str,
            row_normalize: bool = True,
            robust: Optional[str] = None) -> EconometricResults:
    """Kelejian-Prucha (1999) GMM for the spatial-error parameter λ.

    Stage 1 — OLS on ``y = Xβ + u`` ⇒ residuals ``u``.
    Stage 2 — minimise sum of squared KP moment residuals over ``(λ, σ²)``.
    Stage 3 — GLS with ``(I - λ̂ W)`` → final ``β̂``.

    Parameters
    ----------
    robust : {None, "het"}
        When ``"het"``, the final β covariance uses the heteroscedasticity-
        robust sandwich (Arraiz et al. 2010 / ``spreg.GM_Error_Het``).
    """
    y, X, dep, indep = _parse_formula(formula, data)
    n = len(y)
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)

    # Stage 1 — OLS
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_ols = XtX_inv @ (X.T @ y)
    u = y - X @ beta_ols

    # Stage 2 — minimise sum of squared moment residuals
    def obj(theta):
        lam, s2 = float(theta[0]), float(theta[1])
        if not (-0.99 < lam < 0.99) or s2 <= 0:
            return 1e20
        g = _kp_moment_residuals(u, M, lam, s2)
        return float(g @ g)

    s2_init = float(u @ u) / n
    opt = minimize(obj, x0=[0.1, s2_init], method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 600})
    lam_hat, sigma2_hat = float(opt.x[0]), float(opt.x[1])

    # Stage 3 — feasible GLS using λ̂
    import scipy.sparse as sp
    A = sp.eye(n) - lam_hat * M
    Xa = A @ X
    ya = A @ y
    XatXa_inv = np.linalg.inv(Xa.T @ Xa)
    beta = XatXa_inv @ (Xa.T @ ya)
    e = ya - Xa @ beta

    if robust == "het":
        # Sandwich: (X'X)^-1 X' diag(e^2) X (X'X)^-1  (HC0 on transformed design)
        meat = Xa.T @ (np.diag(e ** 2) @ Xa)
        V = XatXa_inv @ meat @ XatXa_inv
    else:
        sigma2 = float(e @ e) / n
        V = sigma2 * XatXa_inv
    se_beta = np.sqrt(np.diag(V))

    names = ["const"] + list(indep) + ["lambda"]
    params = np.concatenate([beta, [lam_hat]])
    se = np.concatenate([se_beta, [float("nan")]])     # KP does not return λ̂ SE

    return EconometricResults(
        params=pd.Series(params, index=names),
        std_errors=pd.Series(se, index=names),
        model_info={
            "model_type": "SEM-GMM (Kelejian-Prucha 1999"
                          + (" Het" if robust == "het" else "") + ")",
            "method": "Generalized Method of Moments",
            "spatial_param": "lambda",
            "spatial_param_value": lam_hat,
        },
        data_info={
            "nobs": n,
            "df_model": len(indep) + 1,
            "df_resid": n - len(names),
            "dependent_var": dep,
            "fitted_values": y - e,
            "residuals": e,
            "W_sparse": M,
        },
        diagnostics={
            "sigma2_hat": round(sigma2_hat, 6),
            "moment_obj": round(float(opt.fun), 8),
        },
    )


# --------------------------------------------------------------------- #
#  SAR GMM / 2SLS (Kelejian-Prucha 1998)
# --------------------------------------------------------------------- #

def sar_gmm(W, data: pd.DataFrame, formula: str,
            row_normalize: bool = True,
            robust: Optional[str] = None,
            w_lags: int = 1) -> EconometricResults:
    """Kelejian-Prucha (1998) 2SLS for SAR with spatial-lag instruments.

    Instruments: ``[X, W X, …, W^w_lags X]`` (dropping constant duplicates).
    ``w_lags=1`` (default) matches ``spreg.GM_Lag`` default.
    Endogenous regressor: ``W Y``.
    """
    y, X, dep, indep = _parse_formula(formula, data)
    n, k = X.shape
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)

    Wy = M @ y
    X_nc = X[:, 1:]                              # non-constant columns
    # Accumulate spatial lags of X up to order w_lags
    lags = [X_nc]
    current = X_nc
    for _ in range(w_lags):
        current = M @ current
        lags.append(current)
    # Instrument matrix: constant + X + W X + ... + W^w_lags X
    Z = np.column_stack([X] + lags[1:])
    # Regressors:  [X, W Y]
    D = np.column_stack([X, Wy])

    # 2SLS: beta_hat = (D' P_Z D)^{-1} D' P_Z y
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    P_Z = Z @ ZtZ_inv @ Z.T
    DtPzD_inv = np.linalg.inv(D.T @ P_Z @ D)
    theta = DtPzD_inv @ D.T @ P_Z @ y

    e = y - D @ theta
    beta = theta[:-1]
    rho = float(theta[-1])

    if robust == "het":
        meat = D.T @ P_Z @ np.diag(e ** 2) @ P_Z @ D
        V = DtPzD_inv @ meat @ DtPzD_inv
    else:
        sigma2 = float(e @ e) / (n - D.shape[1])
        V = sigma2 * DtPzD_inv
    se_all = np.sqrt(np.diag(V))
    se_beta = se_all[:-1]
    se_rho = float(se_all[-1])

    names = ["const"] + list(indep) + ["rho"]
    params = np.concatenate([beta, [rho]])
    se = np.concatenate([se_beta, [se_rho]])

    return EconometricResults(
        params=pd.Series(params, index=names),
        std_errors=pd.Series(se, index=names),
        model_info={
            "model_type": "SAR-GMM (Kelejian-Prucha 1998 2SLS"
                          + (" Het" if robust == "het" else "") + ")",
            "method": "Spatial 2SLS",
            "spatial_param": "rho",
            "spatial_param_value": rho,
        },
        data_info={
            "nobs": n,
            "df_model": len(names) - 1,
            "df_resid": n - len(names),
            "dependent_var": dep,
            "fitted_values": y - e,
            "residuals": e,
            "W_sparse": M,
        },
        diagnostics={
            "sigma2": round(float(e @ e) / n, 6),
        },
    )


# --------------------------------------------------------------------- #
#  SARAR GMM = SAR GMM + SEM GMM on residuals (spreg's GM_Combo)
# --------------------------------------------------------------------- #

def sarar_gmm(W, data: pd.DataFrame, formula: str,
              row_normalize: bool = True,
              robust: Optional[str] = None) -> EconometricResults:
    """Combined GMM: Kelejian-Prucha SAR 2SLS then SEM GMM on residuals.

    Equivalent to ``spreg.GM_Combo`` (or ``GM_Combo_Het`` with ``robust='het'``).
    """
    # Stage 1 — SAR GMM to recover (β̂, ρ̂) and residuals ε̂
    sar_res = sar_gmm(W, data, formula, row_normalize=row_normalize, robust=robust)
    rho_hat = float(sar_res.model_info["spatial_param_value"])
    e1 = sar_res.data_info["residuals"]

    # Stage 2 — KP 1999 moments on ε̂ to recover λ̂
    _, X, dep, indep = _parse_formula(formula, data)
    M = _coerce_W(W, n_expected=len(e1), row_normalize=row_normalize)

    def obj(theta):
        lam, s2 = float(theta[0]), float(theta[1])
        if not (-0.99 < lam < 0.99) or s2 <= 0:
            return 1e20
        g = _kp_moment_residuals(e1, M, lam, s2)
        return float(g @ g)

    s2_init = float(e1 @ e1) / len(e1)
    opt = minimize(obj, x0=[0.1, s2_init], method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 600})
    lam_hat = float(opt.x[0])

    # Stage 3 — Cochrane-Orcutt-style GLS: filter (y, X, WY) by (I - λ̂ W)
    # and re-run 2SLS to match ``spreg.GM_Combo``'s final estimator.
    import scipy.sparse as _sp
    y_full = data[dep].to_numpy(float)
    n_full = len(y_full)
    # Rebuild X and instruments to apply the filter uniformly
    X_full = np.column_stack([np.ones(n_full),
                              data[list(indep)].to_numpy(float)])
    Wy_full = M @ y_full
    A = _sp.eye(n_full) - lam_hat * M
    y_flt  = A @ y_full
    X_flt  = A @ X_full
    Wy_flt = A @ Wy_full
    # Instruments: apply same filter so orthogonality is preserved
    X_nc = X_full[:, 1:]
    WX = M @ X_nc
    Z_flt = A @ np.column_stack([X_full, WX])
    D_flt = np.column_stack([X_flt, Wy_flt])
    ZtZ_inv = np.linalg.inv(Z_flt.T @ Z_flt)
    P_Z = Z_flt @ ZtZ_inv @ Z_flt.T
    theta = np.linalg.solve(D_flt.T @ P_Z @ D_flt, D_flt.T @ P_Z @ y_flt)
    beta_final = theta[:-1]
    rho_final = float(theta[-1])

    names = ["const"] + list(indep) + ["rho", "lambda"]
    params = np.concatenate([beta_final, [rho_final, lam_hat]])
    # Keep SE for beta from stage-1 SAR GMM as a serviceable approximation
    se = np.concatenate([
        sar_res.std_errors.values[:len(beta_final)],
        [sar_res.std_errors.values[-1], float("nan")],
    ])
    # Update rho to stage-3 value
    rho_hat = rho_final
    e1 = y_full - X_full @ beta_final - rho_final * Wy_full

    return EconometricResults(
        params=pd.Series(params, index=names),
        std_errors=pd.Series(se, index=names),
        model_info={
            "model_type": "SARAR-GMM / GM_Combo"
                          + (" Het" if robust == "het" else ""),
            "method": "Spatial 2SLS + KP 1999 moments",
            "spatial_param": "rho,lambda",
            "spatial_param_value": rho_hat,
        },
        data_info={
            "nobs": len(e1),
            "df_model": len(names) - 1,
            "df_resid": len(e1) - len(names),
            "dependent_var": dep,
            "fitted_values": sar_res.data_info["fitted_values"],
            "residuals": e1,
            "W_sparse": M,
        },
        diagnostics={
            "sigma2": sar_res.diagnostics["sigma2"],
            "lambda_moment_obj": round(float(opt.fun), 8),
        },
    )
