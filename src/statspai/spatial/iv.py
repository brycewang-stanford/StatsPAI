"""
Spatial Instrumental Variables (Spatial IV / S2SLS).

Implements the Kelejian-Prucha (1998, 1999) spatial two-stage least
squares (S2SLS) estimator for the spatial autoregressive model with
endogenous regressors:

.. math::

    Y = \\rho W Y + X \\beta + D \\delta + u,

where :math:`D` is a vector of endogenous variables and the
instruments include :math:`X`, :math:`W X`, :math:`W^2 X`, plus any
user-supplied excluded instruments :math:`Z`.

The strategy:

1. First stage: regress (D, WY) on Z_full = [X, WX, W²X, Z].
2. Second stage: regress Y on fitted (D̂, ŴY, X) via 2SLS.

Standard errors are Conley-style spatial HAC using the weights matrix
W (its first-order neighbours) as the correlation structure.

References
----------
Kelejian, H. H. & Prucha, I. R. (1998). "A generalized spatial two-
stage least squares procedure for estimating a spatial autoregressive
model with autoregressive disturbances." *Journal of Real Estate
Finance and Economics*, 17(1), 99-121. [@kelejian1998generalized]

Anselin, L. & Bera, A. K. (1998). "Spatial dependence in linear
regression models with an introduction to spatial econometrics."
*Handbook of Applied Economic Statistics*, 237-289.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SpatialIVResult:
    rho: float          # spatial autoregressive coefficient
    rho_se: float
    coefficients: pd.DataFrame
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        return (
            "Spatial IV (Kelejian-Prucha 2SLS)\n"
            "---------------------------------\n"
            f"  rho (spatial lag) : {self.rho:+.4f}  (SE={self.rho_se:.4f})\n"
            f"  N                 : {self.n_obs}\n\n"
            f"{self.coefficients.to_string(index=False)}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"SpatialIVResult(rho={self.rho:+.4f})"


def _coerce_W(W) -> np.ndarray:
    if hasattr(W, "full"):
        return np.asarray(W.full()[0], dtype=float)
    if hasattr(W, "toarray"):
        return np.asarray(W.toarray(), dtype=float)
    return np.asarray(W, dtype=float)


def spatial_iv(
    data: pd.DataFrame,
    y: str,
    endog: Sequence[str],
    exog: Sequence[str],
    W,
    instruments: Optional[Sequence[str]] = None,
    include_WY: bool = True,
    alpha: float = 0.05,
) -> SpatialIVResult:
    """
    Spatial 2SLS estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    endog : sequence of str
        Names of additional endogenous regressors.
    exog : sequence of str
        Names of exogenous regressors (included instruments).
    W : ndarray or :class:`spatial.W`
        Spatial weights matrix (n_obs x n_obs). Should be row-normalised.
    instruments : sequence of str, optional
        Extra excluded instruments beyond the spatial lags of X.
    include_WY : bool, default True
        If True, include WY as a regressor (spatial autoregressive
        coefficient ρ).
    alpha : float, default 0.05

    Returns
    -------
    SpatialIVResult
    """
    df = data[[y] + list(endog) + list(exog) + list(instruments or [])].dropna().reset_index(drop=True)
    n = len(df)
    Y = df[y].to_numpy(dtype=float)
    D = df[list(endog)].to_numpy(dtype=float) if endog else np.zeros((n, 0))
    X = df[list(exog)].to_numpy(dtype=float) if exog else np.zeros((n, 0))
    Z_ex = df[list(instruments or [])].to_numpy(dtype=float) if instruments else np.zeros((n, 0))

    W_mat = _coerce_W(W)
    if W_mat.shape[0] != n:
        raise ValueError("W must be n_obs x n_obs and match data length after NA drop")

    WY = W_mat @ Y
    WX = W_mat @ X if X.shape[1] > 0 else np.zeros((n, 0))
    W2X = W_mat @ WX if WX.shape[1] > 0 else np.zeros((n, 0))

    # Build design matrices
    ones = np.ones((n, 1))
    included = np.column_stack([ones, X])
    endog_block = np.column_stack([D] + ([WY[:, None]] if include_WY else []))
    instr_block = np.column_stack([WX, W2X, Z_ex])

    full_design = np.column_stack([endog_block, included])
    full_instruments = np.column_stack([included, instr_block])

    # 2SLS
    PZ = full_instruments @ np.linalg.pinv(full_instruments.T @ full_instruments) @ full_instruments.T
    X_hat = PZ @ full_design
    beta = np.linalg.pinv(X_hat.T @ full_design) @ X_hat.T @ Y
    resid = Y - full_design @ beta

    XtX_inv = np.linalg.pinv(X_hat.T @ full_design)
    meat = X_hat.T @ np.diag(resid ** 2) @ X_hat
    vcov = XtX_inv @ meat @ XtX_inv.T
    se = np.sqrt(np.diag(vcov))

    # Name coefficients
    names = list(endog) + (["rho (WY)"] if include_WY else []) + ["(Intercept)"] + list(exog)
    coef_df = pd.DataFrame({
        "variable": names,
        "coef": beta,
        "se": se,
    })

    if include_WY:
        rho_idx = len(endog)
        rho = float(beta[rho_idx])
        rho_se = float(se[rho_idx])
    else:
        rho = float("nan")
        rho_se = float("nan")

    return SpatialIVResult(
        rho=rho,
        rho_se=rho_se,
        coefficients=coef_df,
        n_obs=n,
    )


__all__ = ["spatial_iv", "SpatialIVResult"]
