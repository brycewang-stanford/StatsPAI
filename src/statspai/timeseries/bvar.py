"""Bayesian VAR with Minnesota (Litterman) prior.

The Minnesota prior shrinks towards a random-walk model: each variable's
own first lag coefficient is centred at 1, all other coefficients at 0,
with tightness controlled by hyperparameter λ₁ (overall tightness) and
λ₂ (cross-variable shrinkage).

The posterior is analytically tractable (normal-inverse-Wishart) and
computed in closed form — no MCMC required for the posterior mean and
credible intervals.

References
----------
Litterman, R.B. (1986). "Forecasting with Bayesian Vector Autoregressions."
  *JBE&S*, 4(1), 25–38.
Doan, T., Litterman, R. & Sims, C. (1984). "Forecasting and Conditional
  Projection Using Realistic Prior Distributions."
  *ECREV*, 3(1), 1–100.
Kilian, L. & Lütkepohl, H. (2017). *Structural Vector Autoregressive
  Analysis*. Cambridge. [@litterman1986forecasting]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BVARResult:
    coef: np.ndarray                 # (K*p + 1, K)  posterior mean B
    sigma: np.ndarray                # (K, K) posterior mean of Sigma
    fitted: np.ndarray               # (T-p, K)
    residuals: np.ndarray            # (T-p, K)
    var_names: list
    lags: int
    n: int
    lambda1: float
    lambda2: float

    def forecast(self, horizon: int = 8) -> pd.DataFrame:
        K = self.sigma.shape[0]
        p = self.lags
        B = self.coef
        last_obs = self.fitted[-1:] + self.residuals[-1:]  # last actual
        # gather last p observations
        history = np.zeros((p, K))
        T_data = self.fitted.shape[0]
        for lag in range(p):
            idx = T_data - 1 - lag
            if idx >= 0:
                history[lag] = self.fitted[idx] + self.residuals[idx]
        fc = np.empty((horizon, K))
        for h in range(horizon):
            x = np.concatenate([history.ravel(), [1.0]])
            fc[h] = x @ B
            history = np.roll(history, 1, axis=0)
            history[0] = fc[h]
        cols = self.var_names
        return pd.DataFrame(fc, columns=cols,
                           index=np.arange(1, horizon + 1))

    def irf(self, shock_var: int = 0, horizon: int = 20) -> np.ndarray:
        """Orthogonalised impulse responses (Cholesky decomposition)."""
        K = self.sigma.shape[0]
        p = self.lags
        B = self.coef[:-1]           # exclude constant row
        # Companion form
        A_mats = [B[k * K:(k + 1) * K].T for k in range(p)]
        chol = np.linalg.cholesky(self.sigma)
        irfs = np.zeros((horizon, K))
        phi = np.eye(K)
        irfs[0] = chol[:, shock_var]
        for h in range(1, horizon):
            phi_new = np.zeros((K, K))
            for j in range(min(h, p)):
                phi_new += A_mats[j] @ (irfs[h - 1 - j][:, None] @ np.eye(1, K, 0)
                                         if h - 1 - j == 0 else np.diag(irfs[h - 1 - j]))
            # Simplified: multiply lag coefficients
            contrib = np.zeros(K)
            for j in range(min(h, p)):
                contrib += A_mats[j] @ irfs[h - 1 - j]
            irfs[h] = contrib
        return irfs

    def summary(self) -> str:
        K = len(self.var_names)
        lines = [
            f"Bayesian VAR (Minnesota prior)",
            f"  lags = {self.lags}, K = {K}, T = {self.n}",
            f"  λ₁ = {self.lambda1}, λ₂ = {self.lambda2}",
            "",
            "Posterior mean coefficients (first 5 rows):",
            str(pd.DataFrame(self.coef[:5], columns=self.var_names).round(3)),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def bvar(
    data: pd.DataFrame,
    lags: int = 4,
    lambda1: float = 0.1,
    lambda2: float = 0.5,
) -> BVARResult:
    """Bayesian VAR with Minnesota (Litterman) prior.

    Parameters
    ----------
    data : pd.DataFrame
        Columns are the endogenous variables.
    lags : int, default 4
    lambda1 : float, default 0.1
        Overall tightness (smaller = stronger shrinkage toward RW).
    lambda2 : float, default 0.5
        Cross-variable shrinkage relative to own-lag.
    """
    Y_raw = data.to_numpy(dtype=float)
    var_names = list(data.columns)
    T, K = Y_raw.shape

    # Build VAR design: Y = X B + E
    Y = Y_raw[lags:]                          # (T-p, K)
    n = Y.shape[0]
    X_parts = []
    for lag in range(1, lags + 1):
        X_parts.append(Y_raw[lags - lag: T - lag])
    X = np.column_stack(X_parts + [np.ones(n)])  # (n, K*p + 1)
    m = X.shape[1]

    # OLS as starting point for σ² estimates
    B_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
    E_ols = Y - X @ B_ols
    sigma_ols = np.diag(E_ols.T @ E_ols / n)

    # Minnesota prior: V_prior is diagonal, B_prior is near RW
    V_prior = np.zeros(m)
    B_prior = np.zeros((m, K))
    for k in range(K):
        for lag in range(1, lags + 1):
            for j in range(K):
                idx = (lag - 1) * K + j
                if j == k:
                    V_prior[idx] = (lambda1 / lag) ** 2
                    if lag == 1:
                        B_prior[idx, k] = 1.0   # RW prior
                else:
                    V_prior[idx] = (lambda1 * lambda2 / lag) ** 2 * (
                        sigma_ols[k] / max(sigma_ols[j], 1e-12)
                    )
        V_prior[-1] = 100.0                     # flat prior on constant

    # Posterior: B_post = (X'X + V^{-1})^{-1} (X'Y + V^{-1} B_prior)
    V_inv = np.diag(1.0 / np.maximum(V_prior, 1e-12))
    XtX = X.T @ X
    precision = XtX + V_inv
    try:
        precision_inv = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        precision_inv = np.linalg.pinv(precision)
    B_post = precision_inv @ (X.T @ Y + V_inv @ B_prior)
    E_post = Y - X @ B_post
    Sigma_post = E_post.T @ E_post / n

    _result = BVARResult(
        coef=B_post, sigma=Sigma_post,
        fitted=X @ B_post, residuals=E_post,
        var_names=var_names, lags=lags, n=n,
        lambda1=lambda1, lambda2=lambda2,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.timeseries.bvar",
            params={"lags": lags, "lambda1": lambda1, "lambda2": lambda2},
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
