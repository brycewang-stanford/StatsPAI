"""Sparse-backed ML estimation for spatial regression models (SAR / SEM / SDM).

Sparse-backed companion to the legacy dense implementation in :mod:`._legacy`.
Accepts ndarray, scipy.sparse matrix, or :class:`statspai.spatial.weights.W`.

Numerics are designed to match the legacy reference implementation to within
``rtol < 1e-6`` for ``n < 5000`` (exact log-det path), and to scale to
``n > 10000`` via a stochastic Barry-Pace log-det (pulled in from
:mod:`._logdet`).

References
----------
Anselin (1988); Ord (1975); LeSage & Pace (2009). [@anselin1988spatial]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse, stats as sp_stats  # noqa: F401  (sp_stats reserved for future SE work)
from scipy.optimize import minimize_scalar

from ...core.results import EconometricResults
from ..weights.core import W as _W
from ._logdet import log_det_approx


ArrayOrW = Union[np.ndarray, sparse.spmatrix, "_W"]

# When the problem is small enough, prefer the exact eigenvalue-based
# log-determinant — it matches the legacy implementation bit-for-bit and
# allows sub-1e-6 numerical agreement on the backward-compat fixture.
_EXACT_LOGDET_MAX_N = 5000


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

def _coerce_W(W: ArrayOrW, n_expected: Optional[int],
              row_normalize: bool) -> sparse.csr_matrix:
    """Return W as a CSR matrix, optionally row-normalised.

    A ``W`` object's ``transform`` attribute already encodes whether it was
    row-standardised; we still re-normalise here when ``row_normalize=True``
    so behaviour is identical regardless of input type. (Re-normalising an
    already row-stochastic matrix is a no-op.)
    """
    if isinstance(W, _W):
        M = W.sparse
    elif sparse.issparse(W):
        M = W.tocsr()
    else:
        M = sparse.csr_matrix(np.asarray(W, dtype=float))
    if n_expected is not None and M.shape != (n_expected, n_expected):
        raise ValueError(
            f"W must be ({n_expected}, {n_expected}), got {M.shape}"
        )
    if row_normalize:
        rs = np.asarray(M.sum(axis=1)).ravel()
        rs = np.where(rs == 0, 1.0, rs)
        M = sparse.diags(1.0 / rs) @ M
    return M.tocsr()


def _parse_formula(formula: str, data: pd.DataFrame
                   ) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
    """Legacy-compatible formula parser.

    Returns ``(y, X, dep_var, indep_vars)`` where ``X`` includes a leading
    constant column. We deliberately mirror the legacy parser (rather than
    using formulaic) so column names match exactly — ``["const", x1, ...]``
    — and the backward-compat parameter vectors line up.
    """
    if "~" not in formula:
        raise ValueError("Formula must contain '~'")
    dep, indep = formula.split("~", 1)
    dep_var = dep.strip()
    indep_vars = [v.strip() for v in indep.split("+") if v.strip()]
    y = np.asarray(data[dep_var].values, dtype=np.float64)
    X_raw = np.asarray(data[indep_vars].values, dtype=np.float64)
    n = y.shape[0]
    X = np.column_stack([np.ones(n), X_raw])
    return y, X, dep_var, indep_vars


def _eigvals_for_bounds(M: sparse.csr_matrix) -> Optional[np.ndarray]:
    """Return real parts of W's eigenvalues for small n; None for large n."""
    n = M.shape[0]
    if n > _EXACT_LOGDET_MAX_N:
        return None
    return np.real(np.linalg.eigvals(M.toarray()))


def _rho_bounds(eigvals: Optional[np.ndarray]) -> Tuple[float, float]:
    if eigvals is None:
        return -0.999, 0.999
    rho_min = (1.0 / eigvals[eigvals < 0].min()) if np.any(eigvals < 0) else -0.99
    rho_max = (1.0 / eigvals[eigvals > 0].max()) if np.any(eigvals > 0) else 0.99
    rho_min = max(rho_min * 0.99, -0.99)
    rho_max = min(rho_max * 0.99, 0.99)
    return float(rho_min), float(rho_max)


def _logdet(M: sparse.csr_matrix, rho: float,
            eigvals: Optional[np.ndarray]) -> float:
    """log|I - rho W| via the cached eigenvalues when available."""
    if eigvals is not None:
        return float(np.sum(np.log(np.abs(1.0 - rho * eigvals))))
    return log_det_approx(M, rho, n_draws=100, order=40, seed=0)


def _full_loglik(n: int, sigma2: float, logdet_val: float) -> float:
    return float(-n / 2.0 * np.log(2.0 * np.pi)
                 - n / 2.0 * np.log(sigma2)
                 - n / 2.0
                 + logdet_val)


def _make_results(
    *,
    model_type: str,
    spatial_param_name: str,
    spatial_param_value: float,
    var_names: List[str],
    params_vec: np.ndarray,
    se_vec: np.ndarray,
    sigma2: float,
    resid: np.ndarray,
    fitted: np.ndarray,
    n: int,
    dep_var: str,
    log_lik: float,
    extra_diag: Optional[Dict[str, Any]] = None,
    _w_sparse: Optional[sparse.csr_matrix] = None,
) -> EconometricResults:
    model_pretty = {
        "sar": "SAR (Spatial Lag)",
        "sem": "SEM (Spatial Error)",
        "sdm": "SDM (Spatial Durbin)",
        "slx": "SLX (Spatial Lag of X)",
        "sac": "SAC / SARAR (Spatial Lag + Error)",
    }.get(model_type, model_type)
    diagnostics: Dict[str, Any] = {
        "sigma2": round(float(sigma2), 6),
        "Log-Likelihood": round(float(log_lik), 4),
    }
    if extra_diag:
        diagnostics.update(extra_diag)
    data_info = {
        "nobs": n,
        "df_model": len(var_names) - 1,
        "df_resid": n - len(var_names),
        "dependent_var": dep_var,
        "fitted_values": fitted,
        "residuals": resid,
    }
    if _w_sparse is not None:
        data_info["W_sparse"] = _w_sparse
    return EconometricResults(
        params=pd.Series(params_vec, index=var_names),
        std_errors=pd.Series(se_vec, index=var_names),
        model_info={
            "model_type": model_pretty,
            "method": "Maximum Likelihood",
            "spatial_param": spatial_param_name,
            "spatial_param_value": float(spatial_param_value),
        },
        data_info=data_info,
        diagnostics=diagnostics,
    )


def _spatial_se_rho(M: sparse.csr_matrix, X: np.ndarray, beta: np.ndarray,
                    rho: float, sigma2: float) -> float:
    """Asymptotic SE of the spatial-lag parameter ``rho``.

    Uses the SAR information matrix (Ord 1975; Anselin 1988, *Spatial
    Econometrics*). With ``G = W (I - rho W)^{-1}`` and ``Xb = X @ beta`` the
    ``(beta, rho, sigma2)`` information matrix is

        I_bb = X'X / s2
        I_br = X' G Xb / s2
        I_rr = tr(G G) + tr(G' G) + (G Xb)'(G Xb) / s2
        I_rs = tr(G) / s2
        I_ss = n / (2 s2^2)

    and ``Var(rho) = [I^{-1}]_{rho,rho}``. The earlier formula dropped the
    ``tr(G'G)`` term (a positive quantity) and replaced ``(G Xb)`` with a
    projection of ``X``, which understated the information and overstated the
    SE by ~50% (verified by Monte-Carlo coverage vs the empirical SD of rho).
    """
    n, k = X.shape
    Xb = X @ beta
    if n <= _EXACT_LOGDET_MAX_N:
        W_dense = M.toarray()
        G = W_dense @ np.linalg.inv(np.eye(n) - rho * W_dense)  # W A^{-1}
        GXb = G @ Xb
        Info = np.zeros((k + 2, k + 2))
        Info[:k, :k] = (X.T @ X) / sigma2
        Info[:k, k] = (X.T @ GXb) / sigma2
        Info[k, :k] = Info[:k, k]
        Info[k, k] = (np.trace(G @ G) + np.trace(G.T @ G)
                      + float(GXb @ GXb) / sigma2)
        Info[k, k + 1] = np.trace(G) / sigma2
        Info[k + 1, k] = Info[k, k + 1]
        Info[k + 1, k + 1] = n / (2.0 * sigma2 ** 2)
        try:
            return float(np.sqrt(max(np.linalg.inv(Info)[k, k], 1e-12)))
        except np.linalg.LinAlgError:
            return float("nan")
    # Large-n sparse path: 1/sqrt(I_rr) with all three terms (the small
    # beta/sigma2 cross-terms are dropped at scale). G v = W A^{-1} v.
    rng = np.random.default_rng(0)
    m_probes = 30
    A = sparse.eye(n) - rho * M
    from scipy.sparse.linalg import splu
    lu = splu(A.tocsc())

    def _G(V):  # rows v_i -> (W A^{-1} v_i)
        out = np.empty_like(V, dtype=float)
        for i in range(V.shape[0]):
            out[i] = M @ lu.solve(V[i])
        return out

    U = rng.choice([-1.0, 1.0], size=(m_probes, n))
    GU = _G(U)
    G2U = _G(GU)
    tr_G2 = float(np.mean(np.sum(U * G2U, axis=1)))
    tr_GtG = float(np.mean(np.sum(GU * GU, axis=1)))
    GXb = M @ lu.solve(Xb)
    I_rho = tr_G2 + tr_GtG + float(GXb @ GXb) / sigma2
    return float(1.0 / np.sqrt(max(I_rho, 1e-10)))


def _spatial_se_lambda(M: sparse.csr_matrix, lam: float) -> float:
    """Asymptotic SE of the spatial-error parameter ``lambda``.

    With ``G = W (I - lam W)^{-1}`` the spatial-error parameter is
    asymptotically independent of ``beta`` (Anselin 1988), so

        Var(lambda) = 1 / ( tr(G G) + tr(G' G) - 2 tr(G)^2 / n )

    after concentrating out ``sigma2`` (the ``sigma2`` factors cancel). The
    earlier formula used ``tr(G^2) + tr(G)^2 / n`` — missing ``tr(G'G)`` and
    with the wrong ``sigma2`` adjustment — which overstated the SE by ~40%.
    """
    n = M.shape[0]
    if n <= _EXACT_LOGDET_MAX_N:
        W_dense = M.toarray()
        G = W_dense @ np.linalg.inv(np.eye(n) - lam * W_dense)  # W A^{-1}
        trG = np.trace(G)
        I_lam = np.trace(G @ G) + np.trace(G.T @ G) - 2.0 * trG ** 2 / n
        return float(1.0 / np.sqrt(max(I_lam, 1e-10)))
    # Large-n sparse path. G v = W A^{-1} v.
    rng = np.random.default_rng(0)
    m_probes = 30
    A = sparse.eye(n) - lam * M
    from scipy.sparse.linalg import splu
    lu = splu(A.tocsc())

    def _G(V):
        out = np.empty_like(V, dtype=float)
        for i in range(V.shape[0]):
            out[i] = M @ lu.solve(V[i])
        return out

    U = rng.choice([-1.0, 1.0], size=(m_probes, n))
    GU = _G(U)
    G2U = _G(GU)
    tr_G2 = float(np.mean(np.sum(U * G2U, axis=1)))
    tr_GtG = float(np.mean(np.sum(GU * GU, axis=1)))
    trG = float(np.mean(np.sum(U * GU, axis=1)))
    I_lam = tr_G2 + tr_GtG - 2.0 * trG ** 2 / n
    return float(1.0 / np.sqrt(max(I_lam, 1e-10)))


# ====================================================================== #
#  SAR
# ====================================================================== #

def sar(W: ArrayOrW, data: pd.DataFrame, formula: str,
        row_normalize: bool = True, alpha: float = 0.05) -> EconometricResults:
    """Spatial Autoregressive (Lag) Model: ``Y = ρWY + Xβ + ε``.

    Parameters
    ----------
    W : ndarray, scipy.sparse matrix, or statspai.spatial.weights.W
    data, formula, row_normalize, alpha : see legacy implementation.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 30
    >>> W = np.zeros((n, n))  # ring contiguity: each unit linked to neighbours
    >>> for i in range(n):
    ...     W[i, (i - 1) % n] = 1.0
    ...     W[i, (i + 1) % n] = 1.0
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1.0 + 0.8 * x + rng.normal(scale=0.5, size=n),
    ...                    "x": x})
    >>> res = sp.sar(W, data=df, formula="y ~ x")
    >>> list(res.params.index)
    ['const', 'x', 'rho']

    References
    ----------
    anselin1988spatial
    """
    y, X, dep_var, indep = _parse_formula(formula, data)
    n, k = X.shape
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)
    Wy = M @ y

    eigvals = _eigvals_for_bounds(M)
    rho_min, rho_max = _rho_bounds(eigvals)

    XtX_inv = np.linalg.inv(X.T @ X)

    def neg_conc_ll(rho: float) -> float:
        y_star = y - rho * Wy
        beta = XtX_inv @ X.T @ y_star
        e = y_star - X @ beta
        sigma2 = float(e @ e) / n
        if sigma2 <= 0:
            return 1e20
        ll = -n / 2.0 * np.log(sigma2)
        ll += _logdet(M, rho, eigvals)
        return -ll

    opt = minimize_scalar(neg_conc_ll, bounds=(rho_min, rho_max),
                          method="bounded")
    rho_hat = float(opt.x)

    y_star = y - rho_hat * Wy
    beta = XtX_inv @ X.T @ y_star
    e = y_star - X @ beta
    sigma2 = float(e @ e) / n

    se_beta = np.sqrt(np.diag(sigma2 * XtX_inv))
    se_rho = _spatial_se_rho(M, X, beta, rho_hat, sigma2)

    var_names = ["const"] + list(indep) + ["rho"]
    params_vec = np.append(beta, rho_hat)
    se_vec = np.append(se_beta, se_rho)
    fitted = y - e
    log_lik = _full_loglik(n, sigma2, _logdet(M, rho_hat, eigvals))

    return _make_results(
        model_type="sar", spatial_param_name="rho",
        spatial_param_value=rho_hat,
        var_names=var_names, params_vec=params_vec, se_vec=se_vec,
        sigma2=sigma2, resid=e, fitted=fitted, n=n, dep_var=dep_var,
        log_lik=log_lik, _w_sparse=M,
    )


# ====================================================================== #
#  SEM
# ====================================================================== #

def sem(W: ArrayOrW, data: pd.DataFrame, formula: str,
        row_normalize: bool = True, alpha: float = 0.05) -> EconometricResults:
    """Spatial Error Model: ``Y = Xβ + u, u = λWu + ε``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 30
    >>> W = np.zeros((n, n))
    >>> for i in range(n):
    ...     W[i, (i - 1) % n] = 1.0
    ...     W[i, (i + 1) % n] = 1.0
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1.0 + 0.8 * x + rng.normal(scale=0.5, size=n),
    ...                    "x": x})
    >>> res = sp.sem(W, data=df, formula="y ~ x")
    >>> list(res.params.index)
    ['const', 'x', 'lambda']

    References
    ----------
    anselin1988spatial
    """
    y, X, dep_var, indep = _parse_formula(formula, data)
    n, k = X.shape
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)

    eigvals = _eigvals_for_bounds(M)
    lam_min, lam_max = _rho_bounds(eigvals)

    def neg_conc_ll(lam: float) -> float:
        # (I - lam W) is applied implicitly: y_star = y - lam * W y
        y_star = y - lam * (M @ y)
        X_star = X - lam * (M @ X)
        XtX_inv = np.linalg.inv(X_star.T @ X_star)
        beta = XtX_inv @ X_star.T @ y_star
        e = y_star - X_star @ beta
        sigma2 = float(e @ e) / n
        if sigma2 <= 0:
            return 1e20
        ll = -n / 2.0 * np.log(sigma2)
        ll += _logdet(M, lam, eigvals)
        return -ll

    opt = minimize_scalar(neg_conc_ll, bounds=(lam_min, lam_max),
                          method="bounded")
    lam_hat = float(opt.x)

    y_star = y - lam_hat * (M @ y)
    X_star = X - lam_hat * (M @ X)
    XtX_inv = np.linalg.inv(X_star.T @ X_star)
    beta = XtX_inv @ X_star.T @ y_star
    e = y_star - X_star @ beta
    sigma2 = float(e @ e) / n

    se_beta = np.sqrt(np.diag(sigma2 * XtX_inv))
    se_lam = _spatial_se_lambda(M, lam_hat)

    var_names = ["const"] + list(indep) + ["lambda"]
    params_vec = np.append(beta, lam_hat)
    se_vec = np.append(se_beta, se_lam)
    # Residuals for SEM are filtered residuals; the original y - X beta is the
    # untransformed residual vector. Mirror legacy choice (filtered).
    fitted = y - e  # placeholder (matches legacy convention)
    log_lik = _full_loglik(n, sigma2, _logdet(M, lam_hat, eigvals))

    return _make_results(
        model_type="sem", spatial_param_name="lambda",
        spatial_param_value=lam_hat,
        var_names=var_names, params_vec=params_vec, se_vec=se_vec,
        sigma2=sigma2, resid=e, fitted=fitted, n=n, dep_var=dep_var,
        log_lik=log_lik, _w_sparse=M,
    )


# ====================================================================== #
#  SDM
# ====================================================================== #

def sdm(W: ArrayOrW, data: pd.DataFrame, formula: str,
        row_normalize: bool = True, alpha: float = 0.05) -> EconometricResults:
    """Spatial Durbin Model: ``Y = ρWY + Xβ + WXθ + ε``.

    Reports LeSage & Pace direct / indirect / total effects in the
    diagnostics when ``n`` is small enough for the dense ``(I - ρW)^{-1}``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 30
    >>> W = np.zeros((n, n))
    >>> for i in range(n):
    ...     W[i, (i - 1) % n] = 1.0
    ...     W[i, (i + 1) % n] = 1.0
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1.0 + 0.8 * x + rng.normal(scale=0.5, size=n),
    ...                    "x": x})
    >>> res = sp.sdm(W, data=df, formula="y ~ x")
    >>> list(res.params.index)  # own + spatially-lagged covariate + rho
    ['const', 'x', 'W_x', 'rho']

    References
    ----------
    lesage2009introduction
    """
    y, X, dep_var, indep = _parse_formula(formula, data)
    n, k = X.shape
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)
    Wy = M @ y

    # Augment X with W X (skip the constant column)
    WX = M @ X[:, 1:]
    X_aug = np.column_stack([X, WX])
    XtX_inv = np.linalg.inv(X_aug.T @ X_aug)

    eigvals = _eigvals_for_bounds(M)
    rho_min, rho_max = _rho_bounds(eigvals)

    def neg_conc_ll(rho: float) -> float:
        y_star = y - rho * Wy
        beta = XtX_inv @ X_aug.T @ y_star
        e = y_star - X_aug @ beta
        sigma2 = float(e @ e) / n
        if sigma2 <= 0:
            return 1e20
        ll = -n / 2.0 * np.log(sigma2)
        ll += _logdet(M, rho, eigvals)
        return -ll

    opt = minimize_scalar(neg_conc_ll, bounds=(rho_min, rho_max),
                          method="bounded")
    rho_hat = float(opt.x)

    y_star = y - rho_hat * Wy
    beta_aug = XtX_inv @ X_aug.T @ y_star
    e = y_star - X_aug @ beta_aug
    sigma2 = float(e @ e) / n

    se_beta = np.sqrt(np.diag(sigma2 * XtX_inv))
    se_rho = _spatial_se_rho(M, X_aug, beta_aug, rho_hat, sigma2)

    lag_names = [f"W_{v}" for v in indep]
    var_names = ["const"] + list(indep) + lag_names + ["rho"]
    params_vec = np.append(beta_aug, rho_hat)
    se_vec = np.append(se_beta, se_rho)
    fitted = y - e
    log_lik = _full_loglik(n, sigma2, _logdet(M, rho_hat, eigvals))

    # Direct / indirect / total effects (LeSage & Pace 2009). Only computed in
    # the dense regime — at n > 5000 the (I - rho W)^-1 matrix is intractable.
    extra: Dict[str, Any] = {}
    if n <= _EXACT_LOGDET_MAX_N:
        beta_own = beta_aug[1:k]   # skip constant
        theta = beta_aug[k:]
        S_inv = np.linalg.inv(np.eye(n) - rho_hat * M.toarray())
        direct = np.zeros(len(beta_own))
        indirect = np.zeros(len(beta_own))
        total = np.zeros(len(beta_own))
        W_dense = M.toarray()
        for j in range(len(beta_own)):
            S_j = S_inv @ (beta_own[j] * np.eye(n) + theta[j] * W_dense)
            direct[j] = np.trace(S_j) / n
            total[j] = S_j.sum() / n
            indirect[j] = total[j] - direct[j]
        extra = {
            "Direct effects": dict(zip(indep, np.round(direct, 6))),
            "Indirect effects": dict(zip(indep, np.round(indirect, 6))),
            "Total effects": dict(zip(indep, np.round(total, 6))),
        }

    return _make_results(
        model_type="sdm", spatial_param_name="rho",
        spatial_param_value=rho_hat,
        var_names=var_names, params_vec=params_vec, se_vec=se_vec,
        sigma2=sigma2, resid=e, fitted=fitted, n=n, dep_var=dep_var,
        log_lik=log_lik, extra_diag=extra,
        _w_sparse=M,
    )


# --------------------------------------------------------------------- #
#  SLX:  Y = X β + W X θ + ε   — OLS on augmented design
# --------------------------------------------------------------------- #

def slx(W: ArrayOrW, data: pd.DataFrame, formula: str,
        row_normalize: bool = True, alpha: float = 0.05) -> EconometricResults:
    """Spatial Lag of X (SLX) model: ``Y = Xβ + WXθ + ε``.

    No autoregressive term. Estimated by ordinary least squares on the
    design matrix augmented with spatially-lagged covariates (skip the
    constant).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 30
    >>> W = np.zeros((n, n))
    >>> for i in range(n):
    ...     W[i, (i - 1) % n] = 1.0
    ...     W[i, (i + 1) % n] = 1.0
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1.0 + 0.8 * x + rng.normal(scale=0.5, size=n),
    ...                    "x": x})
    >>> res = sp.slx(W, data=df, formula="y ~ x")
    >>> list(res.params.index)
    ['const', 'x', 'W_x']

    References
    ----------
    anselin1988spatial
    """
    y, X, dep_var, indep = _parse_formula(formula, data)
    n, k = X.shape
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)

    WX = M @ X[:, 1:]
    X_aug = np.column_stack([X, WX])
    XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
    beta_aug = XtX_inv @ (X_aug.T @ y)
    e = y - X_aug @ beta_aug
    sigma2 = float(e @ e) / (n - X_aug.shape[1])

    se_beta = np.sqrt(np.diag(sigma2 * XtX_inv))
    lag_names = [f"W_{v}" for v in indep]
    var_names = ["const"] + list(indep) + lag_names
    fitted = y - e
    log_lik = (
        -n / 2 * np.log(2 * np.pi)
        - n / 2 * np.log(sigma2)
        - (e @ e) / (2 * sigma2)
    )

    return _make_results(
        model_type="slx", spatial_param_name="", spatial_param_value=0.0,
        var_names=var_names, params_vec=beta_aug, se_vec=se_beta,
        sigma2=sigma2, resid=e, fitted=fitted, n=n, dep_var=dep_var,
        log_lik=log_lik, extra_diag={},
        _w_sparse=M,
    )


# --------------------------------------------------------------------- #
#  SAC / SARAR:  Y = ρ W Y + X β + u,   u = λ W u + ε
# --------------------------------------------------------------------- #

def sac(W: ArrayOrW, data: pd.DataFrame, formula: str,
        row_normalize: bool = True, alpha: float = 0.05) -> EconometricResults:
    """SAC / SARAR: combined spatial lag + spatial error model.

    Jointly estimates the autoregressive coefficient ρ (on the dependent
    variable) and the spatial-error coefficient λ by concentrated
    maximum likelihood, profiling out (β, σ²) at each (ρ, λ) candidate.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 30
    >>> W = np.zeros((n, n))
    >>> for i in range(n):
    ...     W[i, (i - 1) % n] = 1.0
    ...     W[i, (i + 1) % n] = 1.0
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1.0 + 0.8 * x + rng.normal(scale=0.5, size=n),
    ...                    "x": x})
    >>> res = sp.sac(W, data=df, formula="y ~ x")
    >>> list(res.params.index)  # both spatial coefficients estimated jointly
    ['const', 'x', 'rho', 'lambda']

    References
    ----------
    anselin1988spatial
    """
    y, X, dep_var, indep = _parse_formula(formula, data)
    n, k = X.shape
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)
    Wy = M @ y
    eigvals = _eigvals_for_bounds(M)
    lo, hi = _rho_bounds(eigvals)
    I_n = np.eye(n)
    W_dense = M.toarray()

    def neg_conc_ll(theta: np.ndarray) -> float:
        rho, lam = float(theta[0]), float(theta[1])
        if not (lo < rho < hi and lo < lam < hi):
            return 1e20
        A_lam = I_n - lam * W_dense
        y_tilde = A_lam @ (y - rho * Wy)
        X_tilde = A_lam @ X
        try:
            XtX_inv = np.linalg.inv(X_tilde.T @ X_tilde)
        except np.linalg.LinAlgError:
            return 1e20
        beta = XtX_inv @ (X_tilde.T @ y_tilde)
        e = y_tilde - X_tilde @ beta
        sigma2 = float(e @ e) / n
        if sigma2 <= 0:
            return 1e20
        ll = -n / 2.0 * np.log(sigma2)
        ll += _logdet(M, rho, eigvals) + _logdet(M, lam, eigvals)
        return -ll

    from scipy.optimize import minimize, differential_evolution
    # Multi-start: coarse global search via differential evolution, then
    # local polish with Nelder-Mead. Single-start Nelder-Mead gets stuck
    # when ρ and λ are close to zero because the neighbourhood is very flat.
    bounds = [(lo, hi), (lo, hi)]
    de_opt = differential_evolution(
        neg_conc_ll, bounds=bounds, tol=1e-5, maxiter=60,
        seed=0, polish=False, popsize=12,
    )
    opt = minimize(neg_conc_ll, x0=de_opt.x, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 500})
    rho_hat, lam_hat = float(opt.x[0]), float(opt.x[1])

    A_lam = I_n - lam_hat * W_dense
    X_tilde = A_lam @ X
    y_tilde = A_lam @ (y - rho_hat * Wy)
    XtX_inv = np.linalg.inv(X_tilde.T @ X_tilde)
    beta = XtX_inv @ (X_tilde.T @ y_tilde)
    e = y_tilde - X_tilde @ beta
    sigma2 = float(e @ e) / n

    se_beta = np.sqrt(np.diag(sigma2 * XtX_inv))
    # Numerical Hessian for the two spatial params (2x2 block, diagonal used)
    h = 1e-4
    def _num_d2(f, x, i, j):
        e_i = np.zeros_like(x); e_j = np.zeros_like(x)
        e_i[i] = h; e_j[j] = h
        return (f(x + e_i + e_j) - f(x + e_i - e_j)
                - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * h * h)
    try:
        drr = _num_d2(neg_conc_ll, np.array([rho_hat, lam_hat]), 0, 0)
        dll = _num_d2(neg_conc_ll, np.array([rho_hat, lam_hat]), 1, 1)
        se_rho = float(1.0 / np.sqrt(max(drr, 1e-10)))
        se_lam = float(1.0 / np.sqrt(max(dll, 1e-10)))
    except Exception:
        se_rho = se_lam = float("nan")

    var_names = ["const"] + list(indep) + ["rho", "lambda"]
    params_vec = np.concatenate([beta, [rho_hat, lam_hat]])
    se_vec = np.concatenate([se_beta, [se_rho, se_lam]])
    fitted = y - e
    ldet_rho = _logdet(M, rho_hat, eigvals)
    ldet_lam = _logdet(M, lam_hat, eigvals)
    log_lik = _full_loglik(n, sigma2, ldet_rho + ldet_lam)
    return _make_results(
        model_type="sac", spatial_param_name="rho,lambda",
        spatial_param_value=rho_hat,
        var_names=var_names, params_vec=params_vec, se_vec=se_vec,
        sigma2=sigma2, resid=e, fitted=fitted, n=n, dep_var=dep_var,
        log_lik=log_lik,
        extra_diag={"lambda_hat": round(lam_hat, 6)},
        _w_sparse=M,
    )


__all__ = ["sar", "sem", "sdm", "slx", "sac"]
