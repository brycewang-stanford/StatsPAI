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

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ...core.results import EconometricResults
from .ml import _coerce_W, _parse_formula

# --------------------------------------------------------------------- #
#  SEM GMM (Kelejian-Prucha 1999)
# --------------------------------------------------------------------- #


def _kp_moment_residuals(
    u: np.ndarray,
    W: Any,
    lam: float,
    sigma2: float,
) -> np.ndarray:
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
    m1 = (u @ u - 2 * lam * u @ v + lam**2 * v @ v) / n - sigma2
    # Moment 2
    m2 = (
        v @ v - 2 * lam * v @ w_bar + lam**2 * w_bar @ w_bar
    ) / n - sigma2 * tr_WtW_over_n
    # Moment 3
    m3 = (u @ v - lam * (u @ w_bar + v @ v) + lam**2 * v @ w_bar) / n
    return np.array([m1, m2, m3])


def sem_gmm(
    W: Any,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    robust: Optional[str] = None,
) -> EconometricResults:
    """Kelejian-Prucha (1999) GMM for the spatial-error parameter λ.

    Stage 1 — OLS on ``y = Xβ + u`` ⇒ residuals ``u``.
    Stage 2 — minimise sum of squared KP moment residuals over ``(λ, σ²)``.
    Stage 3 — GLS with ``(I - λ̂ W)`` → final ``β̂``.

    Parameters
    ----------
    robust : {None, "het"}
        When ``"het"``, the final β covariance uses the heteroscedasticity-
        robust sandwich (Arraiz et al. 2010 / ``spreg.GM_Error_Het``).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 40
    >>> W = np.zeros((n, n))  # ring contiguity: each unit linked to neighbours
    >>> for i in range(n):
    ...     W[i, (i - 1) % n] = 1.0
    ...     W[i, (i + 1) % n] = 1.0
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1.0 + 0.8 * x + rng.normal(scale=0.5, size=n),
    ...                    "x": x})
    >>> res = sp.sem_gmm(W, data=df, formula="y ~ x")
    >>> list(res.params.index)
    ['const', 'x', 'lambda']
    """
    y, X, dep, indep = _parse_formula(formula, data)
    n = len(y)
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)

    # Stage 1 — OLS
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_ols = XtX_inv @ (X.T @ y)
    u = y - X @ beta_ols

    # Stage 2 — minimise sum of squared moment residuals
    def obj(theta: np.ndarray) -> float:
        lam, s2 = float(theta[0]), float(theta[1])
        if not (-0.99 < lam < 0.99) or s2 <= 0:
            return 1e20
        g = _kp_moment_residuals(u, M, lam, s2)
        return float(g @ g)

    s2_init = float(u @ u) / n
    opt = minimize(
        obj,
        x0=[0.1, s2_init],
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 600},
    )
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
        meat = (Xa * e[:, None]).T @ (Xa * e[:, None])
        V = XatXa_inv @ meat @ XatXa_inv
    else:
        sigma2 = float(e @ e) / n
        V = sigma2 * XatXa_inv
    se_beta = np.sqrt(np.diag(V))

    names = ["const"] + list(indep) + ["lambda"]
    params = np.concatenate([beta, [lam_hat]])
    se = np.concatenate([se_beta, [float("nan")]])  # KP does not return λ̂ SE

    return EconometricResults(
        params=pd.Series(params, index=names),
        std_errors=pd.Series(se, index=names),
        model_info={
            "model_type": "SEM-GMM (Kelejian-Prucha 1999"
            + (" Het" if robust == "het" else "")
            + ")",
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


def sar_gmm(
    W: Any,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    robust: Optional[str] = None,
    w_lags: int = 1,
) -> EconometricResults:
    """Kelejian-Prucha (1998) 2SLS for SAR with spatial-lag instruments.

    Instruments: ``[X, W X, …, W^w_lags X]`` (dropping constant duplicates).
    ``w_lags=1`` (default) matches ``spreg.GM_Lag`` default.
    Endogenous regressor: ``W Y``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 40
    >>> W = np.zeros((n, n))  # ring contiguity: each unit linked to neighbours
    >>> for i in range(n):
    ...     W[i, (i - 1) % n] = 1.0
    ...     W[i, (i + 1) % n] = 1.0
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1.0 + 0.8 * x + rng.normal(scale=0.5, size=n),
    ...                    "x": x})
    >>> res = sp.sar_gmm(W, data=df, formula="y ~ x")
    >>> list(res.params.index)
    ['const', 'x', 'rho']

    References
    ----------
    kelejian1998generalized
    """
    y, X, dep, indep = _parse_formula(formula, data)
    n, k = X.shape
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)

    Wy = M @ y
    X_nc = X[:, 1:]  # non-constant columns
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
        PZD_e = (P_Z @ D) * e[:, None]
        meat = PZD_e.T @ PZD_e
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
            + (" Het" if robust == "het" else "")
            + ")",
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


def _tsls(y: np.ndarray, Z: np.ndarray, Q: np.ndarray, robust: bool, sig2n_k: bool):
    """2SLS of ``y`` on ``Z`` with instrument matrix ``Q`` (``spatialreg:::tsls``).

    Returns ``(coef, vcov, resid)``. Homoskedastic variance
    ``s2 (Zp'Zp)^{-1}`` with ``s2 = e'e / n`` (``/ (n - k)`` when
    ``sig2n_k``); ``robust=True`` gives the HC0 sandwich
    ``(Zp'Zp)^{-1} Zp' diag(e^2) Zp (Zp'Zp)^{-1}``, where ``Zp`` is ``Z``
    with its endogenous columns replaced by their first-stage fits.
    """
    n = y.shape[0]
    coef_fs, *_ = np.linalg.lstsq(Q, Z, rcond=None)
    Zp = Q @ coef_fs
    ZpZp_inv = np.linalg.inv(Zp.T @ Zp)
    coef = ZpZp_inv @ (Zp.T @ y)
    e = y - Z @ coef
    if robust:
        meat = (Zp * (e**2)[:, None]).T @ Zp
        V = ZpZp_inv @ meat @ ZpZp_inv
    else:
        df = n - Z.shape[1] if sig2n_k else n
        V = float(e @ e) / df * ZpZp_inv
    return coef, V, e


def _kp_gm_lambda(u: np.ndarray, M: Any):
    """Kelejian-Prucha (1999) GM estimate of ``(lambda, sigma^2)`` from residuals.

    Unweighted nonlinear least squares on the three moment conditions, in
    the ``G [lambda, lambda^2, sigma^2]' - g`` form of
    ``spatialreg:::.kpwuwu`` / ``.kpgm``.

    The minimisation is exact rather than iterative: for fixed ``lambda``
    the objective is linear least squares in ``sigma^2``, and profiling it
    out leaves a quartic in ``lambda`` whose stationary points are the real
    roots of a cubic. The quartic can have a second, spurious minimum
    outside the parameter space (on the Columbus data it sits at
    ``lambda = 4.07`` with a *lower* objective than the admissible one), so
    the minimiser is taken over stationary points with ``|lambda| < 1``,
    the Kelejian-Prucha parameter space. ``gstsls`` runs ``nlminb`` from
    the residual autocorrelation instead, which finds the same admissible
    minimum but stops a few 1e-9 short of it on a flat objective.
    """
    n = u.shape[0]
    wu = M @ u
    wwu = M @ wu
    trwpw = float(M.multiply(M).sum())
    G = np.empty((3, 3))
    G[:, 0] = np.array([2 * u @ wu, 2 * wwu @ wu, u @ wwu + wu @ wu]) / n
    G[:, 1] = -np.array([wu @ wu, wwu @ wwu, wwu @ wu]) / n
    G[:, 2] = np.array([1.0, trwpw / n, 0.0])
    g = np.array([u @ u, wu @ wu, u @ wu]) / n

    c = G[:, 2]
    P = np.eye(3) - np.outer(c, c) / float(c @ c)
    A = [-g, G[:, 0], G[:, 1]]  # residual = A0 + A1 lam + A2 lam^2 + c s2
    # Profiled objective: sum_{i,j} lam^(i+j) A_i' P A_j  (quartic in lam)
    quart = np.zeros(5)
    for i in range(3):
        for j in range(3):
            quart[i + j] += float(A[i] @ P @ A[j])
    deriv = np.array([k * quart[k] for k in range(1, 5)])  # ascending
    roots = np.roots(deriv[::-1])
    real = roots[np.abs(roots.imag) <= 1e-12 * np.maximum(1.0, np.abs(roots))].real
    if real.size == 0:  # pragma: no cover - a quartic with positive leading term
        raise RuntimeError("KP GM moment objective has no stationary point")

    def profile(lam: float):
        a_vec = A[0] + A[1] * lam + A[2] * lam * lam
        s2 = -float(a_vec @ c) / float(c @ c)
        r = a_vec + c * s2
        return float(r @ r), s2

    real = real[np.abs(real) < 1.0]
    if real.size == 0:
        raise RuntimeError(
            "KP GM moment objective has no stationary point with |lambda| < 1; "
            "the spatial-error parameter is not identified on these residuals"
        )
    # Minima only (second derivative of the quartic > 0).
    d2 = np.array(
        [sum(k * (k - 1) * quart[k] * r ** (k - 2) for k in range(2, 5)) for r in real]
    )
    real = real[d2 > 0] if np.any(d2 > 0) else real
    vals = [profile(float(r)) for r in real]
    best = int(np.argmin([v[0] for v in vals]))
    lam_hat = float(real[best])
    obj, s2_hat = vals[best]
    return lam_hat, s2_hat, obj


def sarar_gmm(
    W: Any,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    robust: Optional[str] = None,
    sig2n_k: bool = False,
    w_lags: int = 2,
) -> EconometricResults:
    """SARAR (spatial lag + spatial error) by generalized spatial 2SLS.

    The Kelejian-Prucha (1998) GS2SLS estimator for
    ``y = rho W y + X beta + u,  u = lambda W u + e``, computed step for
    step as ``spatialreg::gstsls``:

    1. 2SLS of ``y`` on ``[W y, X]`` with instruments
       ``H = [X, W X, W^2 X]`` (lags of the non-constant columns).
    2. ``lambda`` from the Kelejian-Prucha (1999) generalized moments on the
       2SLS residuals (unweighted nonlinear least squares).
    3. Spatial Cochrane-Orcutt: 2SLS of ``y - lambda W y`` on
       ``[(I - lambda W) W y, (I - lambda W) X]``, instruments
       ``[(I - lambda W) X, W X, W^2 X]`` (the spatial-lag instruments
       are *not* filtered, as in ``gstsls``).

    ``beta``, ``rho`` and their standard errors come from step 3. The GM
    step does not deliver a standard error for ``lambda``; it is reported
    as NaN (as ``gstsls`` reports ``lambda.se = NULL``).

    Parameters
    ----------
    W : array-like, sparse matrix or :class:`W`
        Spatial weights.
    data : DataFrame
    formula : str
        ``"y ~ x1 + x2"``; a constant is always added.
    row_normalize : bool, default True
        Row-standardise ``W`` (``spdep`` style ``"W"``).
    robust : {None, "het"}
        ``"het"``: HC0 sandwich in both 2SLS steps
        (``gstsls(robust = TRUE)``). The ``lambda`` moments are still the
        homoskedastic Kelejian-Prucha (1999) ones; this is *not* the
        Arraiz et al. (2010) heteroskedastic GM estimator of
        ``sphet::spreg(het = TRUE)``.
    sig2n_k : bool, default False
        Divide the residual sum of squares by ``n - k`` instead of ``n``
        in the homoskedastic variance (``gstsls(sig2n_k = TRUE)``).
    w_lags : int, default 2
        Spatial lags of ``X`` used as instruments: ``[W X, ..., W^w_lags X]``.
        ``2`` is Kelejian-Prucha's ``H = [X, WX, W^2 X]`` and the only
        choice ``gstsls`` offers (its ``W2X`` argument is not used by the
        code); ``1`` is PySAL ``spreg.GM_Combo``'s default.

    Returns
    -------
    EconometricResults
        ``params`` ``[const, x..., rho, lambda]``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 40
    >>> W = np.zeros((n, n))  # ring contiguity: each unit linked to neighbours
    >>> for i in range(n):
    ...     W[i, (i - 1) % n] = 1.0
    ...     W[i, (i + 1) % n] = 1.0
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1.0 + 0.8 * x + rng.normal(scale=0.5, size=n),
    ...                    "x": x})
    >>> res = sp.sarar_gmm(W, data=df, formula="y ~ x")
    >>> list(res.params.index)
    ['const', 'x', 'rho', 'lambda']

    References
    ----------
    kelejian1998generalized
    """
    if robust not in (None, "het"):
        raise ValueError(f"robust must be None or 'het'; got {robust!r}")
    is_robust = robust == "het"
    y, X, dep, indep = _parse_formula(formula, data)
    n = len(y)
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)

    Wy = M @ y
    X_nc = X[:, 1:]
    if int(w_lags) < 1:
        raise ValueError(f"w_lags must be >= 1; got {w_lags!r}")
    lags = []
    cur = X_nc
    for _ in range(int(w_lags)):
        cur = M @ cur
        lags.append(cur)
    instr = np.column_stack(lags)

    # Step 1 -- 2SLS
    Z1 = np.column_stack([Wy, X])
    Q1 = np.column_stack([X, instr])
    _, _, u = _tsls(y, Z1, Q1, is_robust, sig2n_k)

    # Step 2 -- KP GM moments for lambda
    lam_hat, gm_s2, gm_obj = _kp_gm_lambda(u, M)

    # Step 3 -- spatial Cochrane-Orcutt 2SLS
    yt = y - lam_hat * (M @ y)
    Xt = X - lam_hat * (M @ X)
    Wyt = Wy - lam_hat * (M @ Wy)
    Z2 = np.column_stack([Wyt, Xt])
    Q2 = np.column_stack([Xt, instr])
    coef, V, e = _tsls(yt, Z2, Q2, is_robust, sig2n_k)
    se_all = np.sqrt(np.diag(V))
    rho_hat = float(coef[0])
    beta = coef[1:]

    names = ["const"] + list(indep) + ["rho", "lambda"]
    params = np.concatenate([beta, [rho_hat, lam_hat]])
    se = np.concatenate([se_all[1:], [se_all[0], float("nan")]])
    fitted = y - e  # gstsls convention: fit = y - (filtered-model residual)

    return EconometricResults(
        params=pd.Series(params, index=names),
        std_errors=pd.Series(se, index=names),
        model_info={
            "model_type": "SARAR GS2SLS (Kelejian-Prucha 1998"
            + (", HC0" if is_robust else "")
            + ")",
            "method": "Generalized spatial 2SLS + KP 1999 GM",
            "spatial_param": "rho,lambda",
            "spatial_param_value": rho_hat,
        },
        data_info={
            "nobs": n,
            "df_model": len(names) - 1,
            "df_resid": n - len(names),
            "dependent_var": dep,
            "fitted_values": fitted,
            "residuals": e,
            "W_sparse": M,
        },
        diagnostics={
            "sigma2": float(e @ e) / (n - Z2.shape[1] if sig2n_k else n),
            "gm_sigma2": gm_s2,
            "lambda_moment_obj": gm_obj,
        },
    )
