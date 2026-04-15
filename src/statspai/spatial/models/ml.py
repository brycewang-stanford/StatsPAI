"""Sparse-backed ML estimation for spatial regression models (SAR / SEM / SDM).

Drop-in replacement for the legacy dense implementation in :mod:`._legacy`.
Accepts ndarray, scipy.sparse matrix, or :class:`statspai.spatial.weights.W`.

Numerics are designed to match the legacy reference implementation to within
``rtol < 1e-6`` for ``n < 5000`` (exact log-det path), and to scale to
``n > 10000`` via a stochastic Barry-Pace log-det (pulled in from
:mod:`._logdet`).

References
----------
Anselin (1988); Ord (1975); LeSage & Pace (2009).
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
    return EconometricResults(
        params=pd.Series(params_vec, index=var_names),
        std_errors=pd.Series(se_vec, index=var_names),
        model_info={
            "model_type": model_pretty,
            "method": "Maximum Likelihood",
            "spatial_param": spatial_param_name,
            "spatial_param_value": float(spatial_param_value),
        },
        data_info={
            "nobs": n,
            "df_model": len(var_names) - 1,
            "df_resid": n - len(var_names),
            "dependent_var": dep_var,
            "fitted_values": fitted,
            "residuals": resid,
        },
        diagnostics=diagnostics,
    )


def _spatial_se_rho(M: sparse.csr_matrix, X: np.ndarray, rho: float,
                    sigma2: float) -> float:
    """Approximate SE for rho — mirrors legacy ``_spatial_se`` block."""
    n = X.shape[0]
    # For small n we take the dense path (matches legacy exactly); otherwise
    # we fall back on a sparse solve.
    if n <= _EXACT_LOGDET_MAX_N:
        W_dense = M.toarray()
        A_inv_W = np.linalg.solve(np.eye(n) - rho * W_dense, W_dense)
        XtX_inv = np.linalg.inv(X.T @ X)
        tr2 = np.trace(A_inv_W @ A_inv_W)
        I_rho = tr2 + (A_inv_W @ X @ XtX_inv @ X.T @ A_inv_W.T).trace() / sigma2
    else:
        # Stochastic estimate of tr((A^-1 W)^2) via Hutchinson; cheap upper bound.
        rng = np.random.default_rng(0)
        m_probes = 30
        U = rng.choice([-1.0, 1.0], size=(m_probes, n))
        A = sparse.eye(n) - rho * M
        # Solve A z = W u  -->  z = A^-1 W u
        from scipy.sparse.linalg import splu
        lu = splu(A.tocsc())
        Wu = U @ M.T  # rows are W u_i^T
        Z = np.empty_like(Wu)
        for i in range(m_probes):
            Z[i] = lu.solve(Wu[i])
        # tr((A^-1 W)^2) ≈ mean_i u_i^T (A^-1 W)(A^-1 W) u_i
        WZ = Z @ M.T
        AinvWZ = np.empty_like(WZ)
        for i in range(m_probes):
            AinvWZ[i] = lu.solve(WZ[i])
        tr2 = float(np.mean(np.sum(U * AinvWZ, axis=1)))
        I_rho = tr2  # drop the dense X cross term at scale; conservative SE
    return float(1.0 / np.sqrt(max(I_rho, 1e-10)))


def _spatial_se_lambda(M: sparse.csr_matrix, lam: float) -> float:
    n = M.shape[0]
    if n <= _EXACT_LOGDET_MAX_N:
        W_dense = M.toarray()
        A_inv_W = np.linalg.solve(np.eye(n) - lam * W_dense, W_dense)
        tr1 = np.trace(A_inv_W)
        tr2 = np.trace(A_inv_W @ A_inv_W)
        I_lam = tr2 + tr1 ** 2 / n
        return float(1.0 / np.sqrt(max(I_lam, 1e-10)))
    # Stochastic tr2 only (drop tr1^2 term — small relative to tr2).
    rng = np.random.default_rng(0)
    m_probes = 30
    U = rng.choice([-1.0, 1.0], size=(m_probes, n))
    A = sparse.eye(n) - lam * M
    from scipy.sparse.linalg import splu
    lu = splu(A.tocsc())
    Wu = U @ M.T
    Z = np.empty_like(Wu)
    for i in range(m_probes):
        Z[i] = lu.solve(Wu[i])
    WZ = Z @ M.T
    AinvWZ = np.empty_like(WZ)
    for i in range(m_probes):
        AinvWZ[i] = lu.solve(WZ[i])
    tr2 = float(np.mean(np.sum(U * AinvWZ, axis=1)))
    return float(1.0 / np.sqrt(max(tr2, 1e-10)))


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
    se_rho = _spatial_se_rho(M, X, rho_hat, sigma2)

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
        log_lik=log_lik,
    )


# ====================================================================== #
#  SEM
# ====================================================================== #

def sem(W: ArrayOrW, data: pd.DataFrame, formula: str,
        row_normalize: bool = True, alpha: float = 0.05) -> EconometricResults:
    """Spatial Error Model: ``Y = Xβ + u, u = λWu + ε``."""
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
        log_lik=log_lik,
    )


# ====================================================================== #
#  SDM
# ====================================================================== #

def sdm(W: ArrayOrW, data: pd.DataFrame, formula: str,
        row_normalize: bool = True, alpha: float = 0.05) -> EconometricResults:
    """Spatial Durbin Model: ``Y = ρWY + Xβ + WXθ + ε``."""
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
    se_rho = _spatial_se_rho(M, X_aug, rho_hat, sigma2)

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
    log_lik = -float(opt.fun) - n / 2 * np.log(2 * np.pi) - n / 2
    return _make_results(
        model_type="sac", spatial_param_name="rho,lambda",
        spatial_param_value=rho_hat,
        var_names=var_names, params_vec=params_vec, se_vec=se_vec,
        sigma2=sigma2, resid=e, fitted=fitted, n=n, dep_var=dep_var,
        log_lik=log_lik,
        extra_diag={"lambda_hat": round(lam_hat, 6)},
    )


__all__ = ["sar", "sem", "sdm", "slx", "sac"]
