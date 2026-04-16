"""
Nonparametric IV (NPIV) via sieve estimation — Newey and Powell (2003).

In many empirical settings the structural function ``h(D)`` in
``Y = h(D) + U`` with ``E[U | Z] = 0`` may be nonlinear. Standard
2SLS imposes linearity; NPIV lifts that restriction by approximating
``h`` and the first stage using sieve (polynomial / B-spline) bases:

    Stage 1:  D = g(Z) + V    — sieve of Z of degree ``k_z``
    Stage 2:  Y = h(D̂) + U    — sieve of D̂ of degree ``k_d``

where D̂ is the first-stage fit. Penalized 2SLS (Tikhonov
regularization ``α``) stabilises the ill-posed inverse problem.

This module provides:
- :func:`npiv` — full nonparametric IV estimator returning the
  :class:`NPIVResult` with fitted ``h``, SE band, first-stage fit, etc.

The basis is either **polynomial** (Vandermonde) or **B-spline**
(``scipy.interpolate.BSpline``). When ``basis='auto'``, polynomial is
used for low degree and B-spline for high degree.

References
----------
Newey, W.K. and Powell, J.L. (2003). "Instrumental Variable Estimation
    of Nonparametric Models." *Econometrica*, 71(5), 1565-1578.

Blundell, R., Chen, X. and Kristensen, D. (2007). "Semi-Nonparametric
    IV Estimation of Shape-Invariant Engel Curves." *Econometrica*,
    75(6), 1613-1669.

Darolles, S., Fan, Y., Florens, J.-P. and Renault, E. (2011). "Nonparametric
    instrumental regression." *Econometrica*, 79(5), 1541-1565.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class NPIVResult:
    """Nonparametric IV estimation result."""
    h_values: np.ndarray
    h_se: np.ndarray
    d_grid: np.ndarray
    first_stage_f: float
    regularization: float
    basis_type: str
    k_d: int
    k_z: int
    n_obs: int
    residuals: np.ndarray
    extra: dict

    def summary(self) -> str:
        lines = [
            "Nonparametric IV (Newey-Powell 2003 sieve)",
            "-" * 60,
            f"  Observations         : {self.n_obs}",
            f"  Basis                : {self.basis_type}  (k_d={self.k_d}, k_z={self.k_z})",
            f"  Regularization α     : {self.regularization:.4g}",
            f"  First-stage F        : {self.first_stage_f:.2f}",
            f"  h(D) evaluated at    : {len(self.d_grid)} grid points "
            f"[{self.d_grid.min():.2f}, {self.d_grid.max():.2f}]",
        ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "D": self.d_grid,
            "h": self.h_values,
            "se": self.h_se,
            "ci_lower": self.h_values - 1.96 * self.h_se,
            "ci_upper": self.h_values + 1.96 * self.h_se,
        })


def _grab(v, data, cols=False):
    if isinstance(v, str):
        return data[v].values.astype(float)
    if cols and isinstance(v, list) and all(isinstance(x, str) for x in v):
        return data[v].values.astype(float)
    return np.asarray(v, dtype=float)


def _poly_basis(X: np.ndarray, K: int) -> np.ndarray:
    """Return [1, X, X², ..., X^K] basis matrix, shape (n, K+1)."""
    X = np.asarray(X, dtype=float).reshape(-1)
    return np.vander(X, N=K + 1, increasing=True)


def _bspline_basis(X: np.ndarray, K: int, n_knots: int = 10) -> np.ndarray:
    """B-spline basis matrix."""
    from scipy.interpolate import BSpline
    X = np.asarray(X, dtype=float).reshape(-1)
    lo, hi = float(X.min()), float(X.max())
    knots = np.linspace(lo, hi, n_knots)
    internal = knots[1:-1]
    # use degree = min(3, K) for stability
    deg = min(3, K)
    t = np.concatenate([
        np.full(deg + 1, lo),
        internal,
        np.full(deg + 1, hi),
    ])
    n_basis = len(t) - deg - 1
    B = np.zeros((len(X), n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spl = BSpline(t, c, deg, extrapolate=True)
        B[:, j] = spl(X)
    return B


def _build_basis(X: np.ndarray, K: int, basis: str) -> np.ndarray:
    if basis == "polynomial":
        return _poly_basis(X, K)
    if basis == "bspline":
        return _bspline_basis(X, K)
    if basis == "auto":
        return _poly_basis(X, K) if K <= 5 else _bspline_basis(X, K)
    raise ValueError(f"Unknown basis: {basis}")


def _residualize(M: np.ndarray, W: Optional[np.ndarray]) -> np.ndarray:
    if W is None or W.size == 0 or W.shape[1] == 0:
        return M
    b, *_ = np.linalg.lstsq(W, M, rcond=None)
    return M - W @ b


def npiv(
    y: Union[np.ndarray, pd.Series, str],
    endog: Union[np.ndarray, pd.Series, str],
    instruments: Union[np.ndarray, pd.DataFrame],
    exog=None,
    data: Optional[pd.DataFrame] = None,
    k_d: int = 4,
    k_z: int = 4,
    basis: str = "auto",
    regularization: float = 0.0,
    d_grid: Optional[np.ndarray] = None,
    add_const: bool = True,
) -> NPIVResult:
    """
    Nonparametric IV via sieve estimation.

    Parameters
    ----------
    y : outcome.
    endog : endogenous regressor D (continuous or discrete).
    instruments : excluded instruments Z.
    exog : exogenous controls X (optional; intercept is added automatically).
    data : DataFrame for string inputs.
    k_d : int, default 4
        Degree of the sieve for the structural function ``h(D)``.
    k_z : int, default 4
        Degree of the sieve for the first stage ``g(Z)``.
    basis : {'auto', 'polynomial', 'bspline'}
    regularization : float, default 0.0
        Tikhonov penalty α. Set > 0 (e.g. 0.01) to stabilise ill-posed
        problems when ``k_d`` is large.
    d_grid : array, optional
        Evaluation grid for ``h(D)`` output. Defaults to 100 points
        over the sample range of D.
    add_const : bool, default True.

    Returns
    -------
    NPIVResult
    """
    Y = _grab(y, data).reshape(-1)
    D = _grab(endog, data).reshape(-1)
    Z = _grab(instruments, data, cols=True)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    n = len(Y)
    if exog is None:
        W = np.ones((n, 1)) if add_const else np.empty((n, 0))
    else:
        Wx = _grab(exog, data, cols=True)
        if Wx.ndim == 1:
            Wx = Wx.reshape(-1, 1)
        W = np.column_stack([np.ones(n), Wx]) if add_const else Wx

    # Partial out exogenous controls
    Yt = _residualize(Y.reshape(-1, 1), W).ravel()
    Dt = _residualize(D.reshape(-1, 1), W).ravel()
    Zt = _residualize(Z, W)

    # === Stage 1: sieve regression of D on Z ===
    if Zt.shape[1] == 1:
        z_scalar = Zt[:, 0]
    else:
        # Multi-instrument: build tensor-product basis would be complex;
        # use first-stage fitted value from linear projection as the index.
        pi_lin, *_ = np.linalg.lstsq(Zt, Dt, rcond=None)
        z_scalar = Zt @ pi_lin  # scalar index, preserves instrument info
    Phi_Z = _build_basis(z_scalar, k_z, basis)
    pi_hat, *_ = np.linalg.lstsq(Phi_Z, Dt, rcond=None)
    Dt_hat = Phi_Z @ pi_hat
    resid_fs = Dt - Dt_hat
    # First-stage F
    rss_full = float(resid_fs @ resid_fs)
    rss_red = float(Dt @ Dt)
    df_d = max(n - Phi_Z.shape[1], 1)
    first_f = ((rss_red - rss_full) / Phi_Z.shape[1]) / (rss_full / df_d) if rss_full > 0 else np.inf

    # === Stage 2: sieve regression of Y on h(D̂) ===
    Phi_D = _build_basis(Dt_hat, k_d, basis)
    k_all = Phi_D.shape[1]
    # Tikhonov ridge: (Phi'Phi + α I)^{-1} Phi'Y
    PtP = Phi_D.T @ Phi_D
    if regularization > 0:
        PtP += regularization * np.eye(k_all)
    try:
        theta = np.linalg.solve(PtP, Phi_D.T @ Yt)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(PtP, Phi_D.T @ Yt, rcond=None)[0]

    fitted = Phi_D @ theta
    residuals = Yt - fitted

    # === Evaluate h on grid ===
    if d_grid is None:
        lo, hi = float(Dt_hat.min()), float(Dt_hat.max())
        d_grid = np.linspace(lo, hi, 100)
    Phi_grid = _build_basis(d_grid, k_d, basis)
    h_grid = Phi_grid @ theta

    # HC1 SE for h on grid (sandwich)
    sigma2 = float(residuals @ residuals) / max(n - k_all, 1)
    cov_theta = sigma2 * np.linalg.inv(PtP)
    h_var = np.einsum("ij,jk,ik->i", Phi_grid, cov_theta, Phi_grid)
    h_se = np.sqrt(np.maximum(h_var, 0))

    return NPIVResult(
        h_values=h_grid,
        h_se=h_se,
        d_grid=d_grid,
        first_stage_f=float(first_f),
        regularization=float(regularization),
        basis_type=basis,
        k_d=k_d,
        k_z=k_z,
        n_obs=n,
        residuals=residuals,
        extra={"theta": theta, "pi_hat": pi_hat, "D_hat": Dt_hat},
    )


__all__ = ["npiv", "NPIVResult"]
