"""
Plausibly exogenous IV — Conley, Hansen and Rossi (2012).

Sensitivity of 2SLS inference when the exclusion restriction is relaxed
to allow the instrument to have a *small* direct effect on the outcome.

Four approaches from CHR (2012):

1. **UCI (Union of Confidence Intervals)** — for each candidate direct
   effect γ in a user-supplied support, compute the 2SLS CI assuming
   ``y = X*beta + Z*gamma + u`` with γ known, then take the union.
2. **LTZ (Local-to-Zero)** — Bayesian-flavoured Gaussian prior on γ with
   mean zero and variance ``Omega``; delivers a closed-form variance
   inflation for beta.

Both are exceptionally useful as headline sensitivity checks and are
absent from Python's IV stack.

References
----------
Conley, T.G., Hansen, C.B. and Rossi, P.E. (2012).
    "Plausibly exogenous." *Review of Economics and Statistics*,
    94(1), 260-272.

van Kippersluis, H. and Rietveld, C.A. (2018).
    "Beyond plausibly exogenous." *Econometrics Journal*, 21(3),
    316-331.  [pragmatic guidance]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PlausiblyExogenousResult:
    method: str
    beta_hat: float
    se_hat: float
    gamma_grid: np.ndarray
    beta_at_gamma: np.ndarray
    se_at_gamma: np.ndarray
    ci_lower: float
    ci_upper: float
    ci_level: float
    extra: dict

    def summary(self) -> str:
        lines = [
            f"Plausibly exogenous IV — {self.method}",
            "-" * 48,
            f"  Point estimate (γ=0) : {self.beta_hat:>10.4f}   SE={self.se_hat:.4f}",
            f"  Sensitivity CI ({int(self.ci_level*100)}%): "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"  γ grid size          : {len(self.gamma_grid)}",
        ]
        if 'gamma_mean' in self.extra:
            lines.append(f"  γ prior mean         : {self.extra['gamma_mean']}")
            lines.append(f"  γ prior variance     : {self.extra['gamma_var']}")
        return "\n".join(lines)


def _as_matrix(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _prep(
    y, endog, instruments, exog, data, add_const,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def grab(v, prefix):
        if isinstance(v, str):
            return data[v].values.astype(float)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return data[v].values.astype(float)
        return np.asarray(v, dtype=float)

    Y = grab(y, "y").reshape(-1)
    D = grab(endog, "d")
    if D.ndim == 1:
        D = D.reshape(-1, 1)
    Z = grab(instruments, "z")
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    n = len(Y)
    if exog is None:
        W = np.ones((n, 1)) if add_const else np.empty((n, 0))
    else:
        Wmat = grab(exog, "w")
        if Wmat.ndim == 1:
            Wmat = Wmat.reshape(-1, 1)
        W = np.column_stack([np.ones(n), Wmat]) if add_const else Wmat
    return Y, D, Z, W


def _tsls(
    Y: np.ndarray, D: np.ndarray, Z: np.ndarray, W: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (beta_on_D, full_params, var_cov) for 2SLS of Y on [D, W] using [Z, W]."""
    X = np.column_stack([D, W]) if W.size else D
    ZZ = np.column_stack([Z, W]) if W.size else Z
    PZZ = ZZ @ np.linalg.solve(ZZ.T @ ZZ, ZZ.T)
    X_hat = PZZ @ X
    XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    params = XhXh_inv @ X_hat.T @ Y
    resid = Y - X @ params
    n, k = X.shape
    sigma2 = float(resid @ resid) / max(n - k, 1)
    var = sigma2 * XhXh_inv
    return params[: D.shape[1]], params, var


def plausibly_exogenous_uci(
    y: Union[np.ndarray, pd.Series, str],
    endog: Union[np.ndarray, pd.Series, str],
    instruments: Union[np.ndarray, pd.DataFrame, List[str]],
    gamma_grid: Union[np.ndarray, Iterable[float]],
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    add_const: bool = True,
    ci_level: float = 0.95,
) -> PlausiblyExogenousResult:
    """
    Union-of-CIs (UCI) plausibly exogenous bounds (Conley, Hansen, Rossi 2012).

    For each candidate γ in ``gamma_grid`` (one value per instrument), the
    model ``y = D*beta + Z*gamma + W*alpha + u`` is estimated by 2SLS after
    subtracting ``Z @ gamma`` from ``y``; the final confidence set for
    ``beta`` is the union over γ of the per-γ 2SLS CIs.

    Parameters
    ----------
    y, endog : array, Series or column name
        Outcome and a single endogenous regressor.
    instruments : array, DataFrame or list of str
    gamma_grid : array-like
        Candidate direct-effect vectors. If ``instruments`` has ``k`` columns,
        ``gamma_grid`` may be (m,) for k=1 or (m, k) for k>1.
    exog, data, add_const, ci_level : usual options.

    Returns
    -------
    PlausiblyExogenousResult
    """
    Y, D, Z, W = _prep(y, endog, instruments, exog, data, add_const)
    p_endog = D.shape[1]

    gamma_arr = np.asarray(list(gamma_grid), dtype=float)
    if gamma_arr.ndim == 1:
        if Z.shape[1] > 1:
            if len(gamma_arr) % Z.shape[1] != 0:
                raise ValueError(
                    f"gamma_grid length {len(gamma_arr)} is not divisible "
                    f"by n_instruments={Z.shape[1]}"
                )
            gamma_arr = gamma_arr.reshape(-1, Z.shape[1])
        else:
            gamma_arr = gamma_arr.reshape(-1, 1)

    beta_hat, _, var0 = _tsls(Y, D, Z, W)
    # When p_endog > 1, focus on the "target_endog" coefficient (default: column 0).
    target_col = 0
    se_hat = float(np.sqrt(max(var0[target_col, target_col], 0)))

    betas = np.empty(len(gamma_arr))
    ses = np.empty(len(gamma_arr))
    for i, g in enumerate(gamma_arr):
        y_adj = Y - Z @ g
        b, _, v = _tsls(y_adj, D, Z, W)
        betas[i] = float(b[target_col])
        ses[i] = float(np.sqrt(max(v[target_col, target_col], 0)))

    z_crit = stats.norm.ppf(0.5 + ci_level / 2)
    lowers = betas - z_crit * ses
    uppers = betas + z_crit * ses
    ci_lower = float(np.min(lowers))
    ci_upper = float(np.max(uppers))

    return PlausiblyExogenousResult(
        method="UCI (union of CIs)",
        beta_hat=float(beta_hat[target_col]),
        se_hat=se_hat,
        gamma_grid=gamma_arr,
        beta_at_gamma=betas,
        se_at_gamma=ses,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        extra={"target_col": target_col, "n_endog": p_endog},
    )


def plausibly_exogenous_ltz(
    y: Union[np.ndarray, pd.Series, str],
    endog: Union[np.ndarray, pd.Series, str],
    instruments: Union[np.ndarray, pd.DataFrame, List[str]],
    gamma_mean: Union[float, np.ndarray] = 0.0,
    gamma_var: Union[float, np.ndarray] = 0.0,
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    add_const: bool = True,
    ci_level: float = 0.95,
) -> PlausiblyExogenousResult:
    """
    Local-to-zero (LTZ) plausibly exogenous (Conley, Hansen, Rossi 2012).

    Places a Gaussian prior on γ with mean ``gamma_mean`` and covariance
    ``gamma_var * I`` (scalar) or the matrix ``gamma_var``, then integrates
    it out to yield a closed-form adjustment to β's asymptotic variance:

        beta_LTZ   = beta_2SLS - A * gamma_mean
        Var(beta)  = Var_2SLS + A * Omega * A'
    where A = (X'P_Z X)^{-1} X'P_Z Z restricted to β's row.

    Parameters
    ----------
    y, endog, instruments, exog, data, add_const : usual.
    gamma_mean : float or array of shape (k,)
    gamma_var : float or array of shape (k, k)
        Prior variance. ``0`` reproduces exact exogeneity → 2SLS result.
    ci_level : float

    Returns
    -------
    PlausiblyExogenousResult
    """
    Y, D, Z, W = _prep(y, endog, instruments, exog, data, add_const)
    k = Z.shape[1]
    p_endog = D.shape[1]
    target_col = 0  # report result for endog[0]; multi-endog supported

    gm = np.asarray(gamma_mean, dtype=float).reshape(-1)
    if gm.size == 1:
        gm = np.full(k, float(gm[0]))
    if gm.size != k:
        raise ValueError(f"gamma_mean length {gm.size} != n_instruments {k}")

    gv = np.asarray(gamma_var, dtype=float)
    if gv.ndim == 0:
        Omega = float(gv) * np.eye(k)
    elif gv.ndim == 1:
        if gv.size != k:
            raise ValueError("gamma_var length must equal n_instruments")
        Omega = np.diag(gv)
    else:
        if gv.shape != (k, k):
            raise ValueError("gamma_var must be (k, k) if matrix")
        Omega = gv

    beta_hat, _, var0 = _tsls(Y, D, Z, W)
    se0 = float(np.sqrt(max(var0[target_col, target_col], 0)))

    # A matrix (CHR 2012 notation): effect of γ on β via OLS-like projection
    ZZ = np.column_stack([Z, W]) if W.size else Z
    X = np.column_stack([D, W]) if W.size else D
    PZZ = ZZ @ np.linalg.solve(ZZ.T @ ZZ, ZZ.T)
    X_hat = PZZ @ X
    XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    # A is (k_X, k) where k_X = D.shape[1] + W.shape[1]
    A = XhXh_inv @ X_hat.T @ Z
    a_beta = A[target_col, :]  # row for target endogenous regressor

    beta_ltz = float(beta_hat[target_col] - a_beta @ gm)
    var_ltz = float(var0[target_col, target_col] + a_beta @ Omega @ a_beta)
    se_ltz = float(np.sqrt(max(var_ltz, 0)))

    z_crit = stats.norm.ppf(0.5 + ci_level / 2)
    ci_lower = beta_ltz - z_crit * se_ltz
    ci_upper = beta_ltz + z_crit * se_ltz

    return PlausiblyExogenousResult(
        method="LTZ (local-to-zero)",
        beta_hat=float(beta_hat[target_col]),
        se_hat=se0,
        gamma_grid=gm.reshape(1, -1),
        beta_at_gamma=np.array([beta_ltz]),
        se_at_gamma=np.array([se_ltz]),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_level=ci_level,
        extra={"gamma_mean": gm, "gamma_var": Omega, "beta_ltz": beta_ltz,
               "se_ltz": se_ltz, "target_col": target_col, "n_endog": p_endog},
    )


__all__ = [
    "plausibly_exogenous_uci",
    "plausibly_exogenous_ltz",
    "PlausiblyExogenousResult",
]
