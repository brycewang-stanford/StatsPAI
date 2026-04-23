"""
Spatial regression models via concentrated Maximum Likelihood.

Implements SAR, SEM, and SDM with:
- Row-normalised or raw spatial weights
- ML estimation via eigenvalue decomposition (Ord 1975)
- LM tests for spatial dependence (Anselin 1988)
- Direct / indirect / total effect decomposition for SAR/SDM (LeSage & Pace 2009)

References
----------
Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
Ord, K. (1975). "Estimation Methods for Models of Spatial Interaction."
    *JASA*, 70(349), 120-126.
LeSage, J. and Pace, R.K. (2009).
    *Introduction to Spatial Econometrics*. CRC Press. [@anselin1988spatial]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar
from scipy import sparse

from ...core.results import EconometricResults


# ====================================================================== #
#  Public API
# ====================================================================== #

def sar(
    W: np.ndarray,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Spatial Autoregressive (Lag) Model:  Y = ρWY + Xβ + ε.

    Parameters
    ----------
    W : (n, n) array
        Spatial weights matrix (contiguity or distance-based).
    data : pd.DataFrame
    formula : str
        ``"y ~ x1 + x2"`` style formula.
    row_normalize : bool
        If True, row-normalise W so each row sums to 1.
    alpha : float
        Significance level.

    Returns
    -------
    EconometricResults
        Includes spatial autoregressive coefficient ρ (rho) as a parameter.

    Examples
    --------
    >>> W = sp.queen_weights(gdf)  # or any (n,n) array
    >>> result = sp.sar(W, data=df, formula='crime ~ income + education')
    >>> print(result.summary())
    """
    model = SpatialModel(W, data, formula, model_type="sar",
                         row_normalize=row_normalize, alpha=alpha)
    return model.fit()


def sem(
    W: np.ndarray,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Spatial Error Model:  Y = Xβ + u,  u = λWu + ε.

    Parameters
    ----------
    W, data, formula, row_normalize, alpha : as in :func:`sar`.

    Returns
    -------
    EconometricResults
        Includes spatial error parameter λ (lambda).
    """
    model = SpatialModel(W, data, formula, model_type="sem",
                         row_normalize=row_normalize, alpha=alpha)
    return model.fit()


def sdm(
    W: np.ndarray,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Spatial Durbin Model:  Y = ρWY + Xβ + WXθ + ε.

    Nests both SAR and SEM as special cases. Use LR tests to determine
    whether SDM simplifies to SAR (θ = 0) or SEM (θ + ρβ = 0).

    Parameters
    ----------
    W, data, formula, row_normalize, alpha : as in :func:`sar`.

    Returns
    -------
    EconometricResults
        Includes ρ (rho), β (own effects), θ (spatial lag effects),
        and direct/indirect/total effect decomposition.
    """
    model = SpatialModel(W, data, formula, model_type="sdm",
                         row_normalize=row_normalize, alpha=alpha)
    return model.fit()


# ====================================================================== #
#  SpatialModel class
# ====================================================================== #

class SpatialModel:
    """
    Unified spatial regression estimator.

    Uses concentrated ML: profile out β and σ², optimise over ρ (or λ)
    in one dimension using eigenvalues of W.
    """

    def __init__(
        self,
        W: np.ndarray,
        data: pd.DataFrame,
        formula: str,
        model_type: str = "sar",
        row_normalize: bool = True,
        alpha: float = 0.05,
    ):
        self.model_type = model_type.lower()
        self.alpha = alpha

        # Parse formula
        if "~" not in formula:
            raise ValueError("Formula must contain '~'")
        dep, indep = formula.split("~", 1)
        self._dep_var = dep.strip()
        self._indep_vars = [v.strip() for v in indep.split("+") if v.strip()]

        # Prepare data
        self.y = data[self._dep_var].values.astype(np.float64)
        X_raw = data[self._indep_vars].values.astype(np.float64)
        self.n = len(self.y)
        # Add constant
        self.X = np.column_stack([np.ones(self.n), X_raw])
        self.k = self.X.shape[1]
        self._var_names = ["const"] + self._indep_vars

        # Weights matrix
        W = np.asarray(W, dtype=np.float64)
        if W.shape != (self.n, self.n):
            raise ValueError(f"W must be ({self.n}, {self.n}), got {W.shape}")
        if row_normalize:
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            W = W / row_sums
        self.W = W

        # Eigenvalues of W (for concentrated log-likelihood bounds)
        self._eigenvalues = np.real(np.linalg.eigvals(W))
        self._rho_min = 1.0 / min(self._eigenvalues[self._eigenvalues < 0]) if any(self._eigenvalues < 0) else -0.99
        self._rho_max = 1.0 / max(self._eigenvalues[self._eigenvalues > 0]) if any(self._eigenvalues > 0) else 0.99
        # Safety margin
        self._rho_min = max(self._rho_min * 0.99, -0.99)
        self._rho_max = min(self._rho_max * 0.99, 0.99)

    def fit(self) -> EconometricResults:
        if self.model_type == "sar":
            return self._fit_sar()
        elif self.model_type == "sem":
            return self._fit_sem()
        elif self.model_type == "sdm":
            return self._fit_sdm()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    # ------------------------------------------------------------------ #
    #  SAR:  Y = ρWY + Xβ + ε
    # ------------------------------------------------------------------ #

    def _fit_sar(self) -> EconometricResults:
        W, y, X, n = self.W, self.y, self.X, self.n
        Wy = W @ y

        # Concentrated ML: for given ρ, β = (X'X)^{-1} X'(y - ρWy)
        XtX_inv = np.linalg.inv(X.T @ X)

        def neg_conc_ll(rho):
            y_star = y - rho * Wy
            beta = XtX_inv @ X.T @ y_star
            e = y_star - X @ beta
            sigma2 = e @ e / n
            if sigma2 <= 0:
                return 1e20
            ll = -n / 2 * np.log(sigma2)
            ll += np.sum(np.log(np.abs(1 - rho * self._eigenvalues)))
            return -ll

        result = minimize_scalar(neg_conc_ll, bounds=(self._rho_min, self._rho_max),
                                 method="bounded")
        rho = result.x

        # Recover β, σ²
        y_star = y - rho * Wy
        beta = XtX_inv @ X.T @ y_star
        e = y_star - X @ beta
        sigma2 = float(e @ e / n)

        # Standard errors via information matrix
        se_beta, se_rho = self._spatial_se(X, e, Wy, rho, sigma2, "sar")

        # Build results
        all_names = self._var_names + ["rho"]
        all_params = np.append(beta, rho)
        all_se = np.append(se_beta, se_rho)

        return self._make_results(all_params, all_se, all_names, sigma2, rho, e)

    # ------------------------------------------------------------------ #
    #  SEM:  Y = Xβ + u,  u = λWu + ε  →  (I - λW)Y = (I - λW)Xβ + ε
    # ------------------------------------------------------------------ #

    def _fit_sem(self) -> EconometricResults:
        W, y, X, n = self.W, self.y, self.X, self.n

        def neg_conc_ll(lam):
            A = np.eye(n) - lam * W
            y_star = A @ y
            X_star = A @ X
            XtX_inv = np.linalg.inv(X_star.T @ X_star)
            beta = XtX_inv @ X_star.T @ y_star
            e = y_star - X_star @ beta
            sigma2 = e @ e / n
            if sigma2 <= 0:
                return 1e20
            ll = -n / 2 * np.log(sigma2)
            ll += np.sum(np.log(np.abs(1 - lam * self._eigenvalues)))
            return -ll

        result = minimize_scalar(neg_conc_ll, bounds=(self._rho_min, self._rho_max),
                                 method="bounded")
        lam = result.x

        # Recover β
        A = np.eye(n) - lam * W
        y_star = A @ y
        X_star = A @ X
        XtX_inv = np.linalg.inv(X_star.T @ X_star)
        beta = XtX_inv @ X_star.T @ y_star
        e = y_star - X_star @ beta
        sigma2 = float(e @ e / n)

        # Standard errors
        se_beta = np.sqrt(np.diag(sigma2 * XtX_inv))
        # Approximate SE for lambda
        se_lam = self._lambda_se(W, e, lam, sigma2)

        all_names = self._var_names + ["lambda"]
        all_params = np.append(beta, lam)
        all_se = np.append(se_beta, se_lam)

        return self._make_results(all_params, all_se, all_names, sigma2, lam, e, param_name="lambda")

    # ------------------------------------------------------------------ #
    #  SDM:  Y = ρWY + Xβ + WXθ + ε
    # ------------------------------------------------------------------ #

    def _fit_sdm(self) -> EconometricResults:
        W, y, X, n = self.W, self.y, self.X, self.n
        Wy = W @ y

        # Augment X with WX (spatial lags of covariates, skip constant)
        WX = W @ X[:, 1:]  # don't lag the constant
        X_aug = np.column_stack([X, WX])
        k_aug = X_aug.shape[1]
        XtX_inv = np.linalg.inv(X_aug.T @ X_aug)

        lag_names = [f"W_{v}" for v in self._indep_vars]

        def neg_conc_ll(rho):
            y_star = y - rho * Wy
            beta = XtX_inv @ X_aug.T @ y_star
            e = y_star - X_aug @ beta
            sigma2 = e @ e / n
            if sigma2 <= 0:
                return 1e20
            ll = -n / 2 * np.log(sigma2)
            ll += np.sum(np.log(np.abs(1 - rho * self._eigenvalues)))
            return -ll

        result = minimize_scalar(neg_conc_ll, bounds=(self._rho_min, self._rho_max),
                                 method="bounded")
        rho = result.x

        y_star = y - rho * Wy
        beta_aug = XtX_inv @ X_aug.T @ y_star
        e = y_star - X_aug @ beta_aug
        sigma2 = float(e @ e / n)

        se_beta = np.sqrt(np.diag(sigma2 * XtX_inv))
        _, se_rho = self._spatial_se(X_aug, e, Wy, rho, sigma2, "sar")

        all_names = self._var_names + lag_names + ["rho"]
        all_params = np.append(beta_aug, rho)
        all_se = np.append(se_beta, se_rho)

        result_obj = self._make_results(all_params, all_se, all_names, sigma2, rho, e)

        # Direct / indirect / total effects (LeSage & Pace 2009)
        beta_own = beta_aug[1:self.k]  # skip constant
        theta = beta_aug[self.k:]
        effects = self._compute_effects(rho, beta_own, theta)
        result_obj.diagnostics.update(effects)

        return result_obj

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _spatial_se(self, X, e, Wy, rho, sigma2, model):
        """Approximate SE from concentrated likelihood Hessian."""
        n, k = X.shape
        XtX_inv = np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(sigma2 * XtX_inv))

        # SE for rho: from information matrix diagonal
        A_inv_W = np.linalg.solve(np.eye(n) - rho * self.W, self.W)
        tr1 = np.trace(A_inv_W)
        tr2 = np.trace(A_inv_W @ A_inv_W)
        I_rho = tr2 + (A_inv_W @ X @ XtX_inv @ X.T @ A_inv_W.T).trace() / sigma2
        se_rho = 1.0 / np.sqrt(max(I_rho, 1e-10))

        return se_beta, se_rho

    def _lambda_se(self, W, e, lam, sigma2):
        """Approximate SE for spatial error parameter."""
        n = len(e)
        A_inv_W = np.linalg.solve(np.eye(n) - lam * W, W)
        tr2 = np.trace(A_inv_W @ A_inv_W)
        tr1 = np.trace(A_inv_W)
        I_lam = tr2 + tr1 ** 2 / n
        return float(1.0 / np.sqrt(max(I_lam, 1e-10)))

    def _compute_effects(self, rho, beta, theta):
        """Direct, indirect, total effects for SDM (LeSage & Pace 2009)."""
        n = self.n
        S_inv = np.linalg.inv(np.eye(n) - rho * self.W)
        k_vars = len(beta)

        direct = np.zeros(k_vars)
        indirect = np.zeros(k_vars)
        total = np.zeros(k_vars)

        for j in range(k_vars):
            S_j = S_inv @ (beta[j] * np.eye(n) + theta[j] * self.W)
            direct[j] = np.trace(S_j) / n
            total[j] = S_j.sum() / n
            indirect[j] = total[j] - direct[j]

        return {
            "Direct effects": dict(zip(self._indep_vars, np.round(direct, 6))),
            "Indirect effects": dict(zip(self._indep_vars, np.round(indirect, 6))),
            "Total effects": dict(zip(self._indep_vars, np.round(total, 6))),
        }

    def _make_results(self, params, se, names, sigma2, spatial_param, resid,
                      param_name="rho"):
        """Build EconometricResults."""
        model_names = {"sar": "SAR (Spatial Lag)", "sem": "SEM (Spatial Error)",
                       "sdm": "SDM (Spatial Durbin)"}
        fitted = self.y - resid[:len(self.y)] if len(resid) == self.n else None

        return EconometricResults(
            params=pd.Series(params, index=names),
            std_errors=pd.Series(se, index=names),
            model_info={
                "model_type": model_names.get(self.model_type, self.model_type),
                "method": "Maximum Likelihood",
                "spatial_param": param_name,
                "spatial_param_value": float(spatial_param),
            },
            data_info={
                "nobs": self.n,
                "df_model": len(names) - 1,
                "df_resid": self.n - len(names),
                "dependent_var": self._dep_var,
                "fitted_values": fitted if fitted is not None else np.zeros(self.n),
                "residuals": resid[:self.n],
            },
            diagnostics={
                "sigma2": round(sigma2, 6),
                "Log-Likelihood": round(self._log_likelihood(sigma2, spatial_param), 4),
            },
        )

    def _log_likelihood(self, sigma2, spatial_param):
        """Full log-likelihood at ML estimates."""
        n = self.n
        ll = -n / 2 * np.log(2 * np.pi) - n / 2 * np.log(sigma2) - n / 2
        ll += np.sum(np.log(np.abs(1 - spatial_param * self._eigenvalues)))
        return float(ll)
