"""
DYNOTEARS: continuous-optimisation causal discovery for time series.

Pamfil, Sriwattanaworachai, Desai, Pilgerstorfer, Georgatzis, Beaumont,
Aragam (2020, AISTATS) extend NOTEARS to a structural vector
autoregression (SVAR):

    X_t = X_t W + sum_{k=1..p} X_{t-k} A_k + epsilon_t

where ``W`` is the contemporaneous (intra-slice) adjacency (must be
acyclic) and ``A_1,...,A_p`` are the lagged (inter-slice) adjacencies.
Parameters are estimated by minimising

    L(W, A) = 0.5/N * || X - X W - sum_k X_{-k} A_k ||_F^2
           + lambda_W * ||W||_1 + lambda_A * sum_k ||A_k||_1
           + rho/2 * h(W)^2 + mu * h(W)

with ``h(W) = tr(exp(W*W)) - d`` the NOTEARS acyclicity function.

References
----------
Pamfil, R. et al. (2020).
"DYNOTEARS: Structure Learning from Time-Series Data." AISTATS 2020.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import linalg, optimize


__all__ = ["dynotears", "DYNOTEARSResult"]


@dataclass
class DYNOTEARSResult:
    """Output of :func:`dynotears`."""
    variables: List[str]
    W: np.ndarray      # (d, d) contemporaneous
    A: np.ndarray      # (p, d, d) lagged
    lag: int
    threshold: float
    loss: float

    def summary(self) -> str:
        d = len(self.variables)
        W_nz = int((np.abs(self.W) > self.threshold).sum())
        A_nz = int((np.abs(self.A) > self.threshold).sum())
        return "\n".join([
            "DYNOTEARS — SVAR Structure Learning",
            "=" * 60,
            f"  variables     : {self.variables}",
            f"  lags          : {self.lag}",
            f"  edges (W/cont): {W_nz}/{d*(d-1)}",
            f"  edges (A/lag) : {A_nz}/{self.lag * d * d}",
            f"  loss          : {self.loss:.6f}",
            f"  threshold     : {self.threshold}",
        ])

    def to_frame(self) -> pd.DataFrame:
        """Long-format edges DataFrame (|coef| > threshold)."""
        rows = []
        d = len(self.variables)
        # Contemporaneous
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                w = self.W[i, j]
                if abs(w) > self.threshold:
                    rows.append({
                        "lag": 0,
                        "from": self.variables[i],
                        "to": self.variables[j],
                        "coef": float(w),
                    })
        # Lagged
        for k in range(self.lag):
            for i in range(d):
                for j in range(d):
                    a = self.A[k, i, j]
                    if abs(a) > self.threshold:
                        rows.append({
                            "lag": k + 1,
                            "from": self.variables[i],
                            "to": self.variables[j],
                            "coef": float(a),
                        })
        return pd.DataFrame(rows)


def _h_acyclicity(W: np.ndarray) -> float:
    """``h(W) = tr(exp(W * W)) - d``."""
    d = W.shape[0]
    return float(np.trace(linalg.expm(W * W)) - d)


def _h_grad(W: np.ndarray) -> np.ndarray:
    """Gradient of the acyclicity constraint wrt ``W``."""
    return linalg.expm(W * W).T * 2 * W


def _loss_fn(
    params: np.ndarray,
    X_now: np.ndarray,
    X_lags: List[np.ndarray],
    d: int,
    p: int,
    lambda_W: float,
    lambda_A: float,
    rho: float,
    mu: float,
) -> tuple:
    """Compute augmented-Lagrangian loss + gradient."""
    W = params[: d * d].reshape(d, d)
    A = params[d * d:].reshape(p, d, d) if p > 0 else np.zeros((0, d, d))
    np.fill_diagonal(W, 0.0)  # no self-contemporaneous loops

    # Residuals
    resid = X_now - X_now @ W
    for k in range(p):
        resid -= X_lags[k] @ A[k]
    N = X_now.shape[0]
    mse = 0.5 / N * float((resid ** 2).sum())

    # Acyclicity
    h = _h_acyclicity(W)
    penalty = rho / 2.0 * h * h + mu * h

    # L1 (smoothed via |w| ≈ sqrt(w^2 + eps))
    eps = 1e-8
    l1_W = lambda_W * float(np.sqrt(W ** 2 + eps).sum()
                             - np.sqrt(np.diag(W) ** 2 + eps).sum())
    l1_A = lambda_A * float(np.sqrt(A ** 2 + eps).sum()) if p > 0 else 0.0

    total = mse + penalty + l1_W + l1_A

    # Gradients
    grad_W = -1.0 / N * (X_now.T @ resid)  # wrt the X @ W term (it subtracts)
    # Acyclicity gradient
    grad_W += (rho * h + mu) * _h_grad(W)
    # L1 gradients (smoothed)
    grad_W += lambda_W * W / np.sqrt(W ** 2 + eps)
    np.fill_diagonal(grad_W, 0.0)

    if p > 0:
        grad_A = np.zeros_like(A)
        for k in range(p):
            grad_A[k] = -1.0 / N * (X_lags[k].T @ resid)
            grad_A[k] += lambda_A * A[k] / np.sqrt(A[k] ** 2 + eps)
    else:
        grad_A = np.zeros((0, d, d))

    grad = np.concatenate([grad_W.ravel(), grad_A.ravel()])
    return total, grad


def dynotears(
    data: pd.DataFrame,
    *,
    variables: Optional[Sequence[str]] = None,
    lag: int = 1,
    lambda_w: float = 0.05,
    lambda_a: float = 0.05,
    max_iter: int = 60,
    threshold: float = 0.1,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    verbose: bool = False,
) -> DYNOTEARSResult:
    """DYNOTEARS — learn a structural VAR with acyclic contemporaneous part.

    Parameters
    ----------
    data : DataFrame
        Time-series panel, one row per time stamp.
    variables : sequence of str, optional
        Numeric columns to include. Defaults to all numeric columns.
    lag : int, default 1
        Number of lagged slices to include in the SVAR.
    lambda_w, lambda_a : float, default 0.05
        L1 regularisation for ``W`` and ``A``.
    max_iter : int, default 60
        Outer augmented-Lagrangian iterations.
    threshold : float, default 0.1
        Coefficients with ``|value| <= threshold`` are pruned to 0 after
        optimisation.
    h_tol : float, default 1e-8
        Acyclicity tolerance — outer loop stops when ``h(W) < h_tol``.
    rho_max : float, default 1e16
        Maximum augmented-Lagrangian penalty.
    verbose : bool, default False

    Returns
    -------
    DYNOTEARSResult

    References
    ----------
    Pamfil et al. (2020), AISTATS.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    if variables is None:
        variables = [c for c in data.columns if np.issubdtype(data[c].dtype, np.number)]
    variables = list(variables)
    d = len(variables)
    if d < 2:
        raise ValueError("Need at least 2 variables for DYNOTEARS.")
    if lag < 0:
        raise ValueError("`lag` must be >= 0.")
    X_full = data[variables].to_numpy(dtype=float)
    T = X_full.shape[0]
    if T <= lag + 5:
        raise ValueError(f"Time series too short (T={T}) for lag={lag}.")

    # Center (column-wise)
    X_full = X_full - X_full.mean(axis=0, keepdims=True)
    # Align: X_now is rows lag..T-1; X_lags[k] is rows (lag-k-1)..(T-k-2) wait
    # Let X_now_t, X_lag_k_t = X_{t-k-1}. We pick t in [lag, T-1].
    X_now = X_full[lag:]
    X_lags = [X_full[lag - k - 1: T - k - 1] for k in range(lag)]

    # Augmented Lagrangian
    rho, mu = 1.0, 0.0
    params = np.zeros(d * d + lag * d * d)
    h_prev = np.inf

    for it in range(max_iter):
        # Inner L-BFGS-B minimization
        result = optimize.minimize(
            _loss_fn, params,
            args=(X_now, X_lags, d, lag, lambda_w, lambda_a, rho, mu),
            jac=True, method="L-BFGS-B",
            options={"maxiter": 100, "disp": False},
        )
        params = result.x
        W = params[: d * d].reshape(d, d)
        np.fill_diagonal(W, 0.0)
        h = _h_acyclicity(W)
        if verbose:
            print(f"iter {it}: loss={result.fun:.4f}, h={h:.2e}, rho={rho:.2e}")
        if h < h_tol or rho >= rho_max:
            break
        if h >= 0.25 * h_prev:
            rho *= 10.0
        mu += rho * h
        h_prev = h

    W = params[: d * d].reshape(d, d)
    np.fill_diagonal(W, 0.0)
    A = params[d * d:].reshape(lag, d, d) if lag > 0 else np.zeros((0, d, d))
    # Threshold small coefficients
    W = np.where(np.abs(W) > threshold, W, 0.0)
    A = np.where(np.abs(A) > threshold, A, 0.0) if lag > 0 else A
    loss = float(result.fun)

    return DYNOTEARSResult(
        variables=variables,
        W=W,
        A=A,
        lag=lag,
        threshold=threshold,
        loss=loss,
    )
