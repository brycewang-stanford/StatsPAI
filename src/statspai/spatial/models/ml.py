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
from scipy import sparse
from scipy import (  # noqa: F401  (sp_stats reserved for future SE work)
    stats as sp_stats,
)
from scipy.optimize import minimize_scalar

from ...core.results import EconometricResults
from ...exceptions import MethodIncompatibility
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


def _coerce_W(
    W: ArrayOrW, n_expected: Optional[int], row_normalize: bool
) -> sparse.csr_matrix:
    """Return W as a CSR matrix, optionally row-normalised.

    A ``W`` object's ``transform`` attribute already encodes whether it was
    row-standardised; we still re-normalise here when ``row_normalize=True``
    so behaviour is identical regardless of input type. (Re-normalising an
    already row-stochastic matrix is a no-op.)
    """
    M: sparse.csr_matrix
    if isinstance(W, _W):
        M = sparse.csr_matrix(W.sparse)
    elif sparse.issparse(W):
        M = sparse.csr_matrix(W)
    else:
        M = sparse.csr_matrix(np.asarray(W, dtype=float))
    if n_expected is not None and M.shape != (n_expected, n_expected):
        raise ValueError(f"W must be ({n_expected}, {n_expected}), got {M.shape}")
    if row_normalize:
        row_sums = np.asarray(M.sum(axis=1)).ravel()
        safe_row_sums = np.where(row_sums == 0, 1.0, row_sums)
        M = sparse.csr_matrix(sparse.diags(1.0 / safe_row_sums) @ M)
    return M


def _parse_formula(
    formula: str, data: pd.DataFrame
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
    terms = [dep_var] + indep_vars
    unknown = [t for t in terms if t not in data.columns]
    if unknown:
        if all(t.isidentifier() for t in unknown):
            from ...exceptions import ColumnNotFound

            raise ColumnNotFound.from_missing(
                unknown, list(data.columns), context="spatial model"
            )
        raise MethodIncompatibility(
            f"spatial model formula: {unknown} are not columns. The spatial "
            "fits take plain column names joined by '+' (no C(), I(), "
            "interactions or transformations).",
            recovery_hint="Create the columns first (e.g. dummies with "
            "pd.get_dummies(df['g'], drop_first=True)) and list them.",
        )
    bad = [t for t in terms if data[t].isna().any()]
    if bad:
        raise MethodIncompatibility(
            f"spatial model: missing values in {bad}. Rows cannot be dropped "
            "because W is indexed by row.",
            recovery_hint="Impute the values, or drop the units from both the "
            "data and W before fitting.",
        )
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
    """First search bracket: 0.99 times the admissible interval, inside
    (-0.99, 0.99). :func:`_line_search` widens it when the maximum sits on
    an edge that is not an edge of the admissible interval."""
    if eigvals is None:
        return -0.999, 0.999
    rho_min = (1.0 / eigvals[eigvals < 0].min()) if np.any(eigvals < 0) else -0.99
    rho_max = (1.0 / eigvals[eigvals > 0].max()) if np.any(eigvals > 0) else 0.99
    rho_min = max(rho_min * 0.99, -0.99)
    rho_max = min(rho_max * 0.99, 0.99)
    return float(rho_min), float(rho_max)


#: Distance kept from the singular ends ``1 / omega`` of the admissible
#: interval, where ``log|I - rho W|`` is minus infinity (as in ``_fiml``).
_EDGE_GAP = 1e-6


def _admissible_bounds(eigvals: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
    """``(1 / omega_min, 1 / omega_max)`` over the real parts of the
    eigenvalues of ``W``, pulled in by ``_EDGE_GAP``: the interval on which
    ``I - rho W`` stays nonsingular, and R ``spatialreg``'s default search
    interval. None when the eigenvalues are not computed (large ``n``) or
    one side has no eigenvalue of that sign (no finite end to search to)."""
    if eigvals is None or not (np.any(eigvals < 0) and np.any(eigvals > 0)):
        return None
    lo = 1.0 / float(eigvals.min()) + _EDGE_GAP
    hi = 1.0 / float(eigvals.max()) - _EDGE_GAP
    return lo, hi


def _line_search(f: Any, eigvals: Optional[np.ndarray]) -> float:
    """Minimise the negative concentrated log-likelihood over the spatial
    parameter.

    The search runs first on the bracket of :func:`_rho_bounds` and again on
    the whole admissible interval (:func:`_admissible_bounds`) when the
    first minimum sits on an end of the bracket that is inside that
    interval. Through 1.40.2 the bracket was the only search: an estimate
    below -0.99 (possible whenever the smallest eigenvalue of ``W`` is
    above -1, e.g. -0.5 gives the interval (-2, 1)) or within 1% of
    ``1 / omega_max`` came back pinned to the bracket end. Fits whose first
    minimum is interior are unchanged bit for bit.
    """
    lo, hi = _rho_bounds(eigvals)
    opt = minimize_scalar(
        f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10}
    )
    x = float(opt.x)
    wide = _admissible_bounds(eigvals)
    if wide is None:
        return x
    tol = 1e-6 * (hi - lo)
    if (x - lo < tol and wide[0] < lo) or (hi - x < tol and wide[1] > hi):
        opt = minimize_scalar(
            f, bounds=wide, method="bounded", options={"xatol": 1e-10}
        )
        x = float(opt.x)
    return x


def _logdet(M: sparse.csr_matrix, rho: float, eigvals: Optional[np.ndarray]) -> float:
    """log|I - rho W| via the cached eigenvalues when available."""
    if eigvals is not None:
        return float(np.sum(np.log(np.abs(1.0 - rho * eigvals))))
    return log_det_approx(M, rho, n_draws=100, order=40, seed=0)


def _full_loglik(n: int, sigma2: float, logdet_val: float) -> float:
    return float(
        -n / 2.0 * np.log(2.0 * np.pi) - n / 2.0 * np.log(sigma2) - n / 2.0 + logdet_val
    )


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
    alpha: float = 0.05,
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
        # Maximum-likelihood fits report z statistics and normal intervals,
        # as Stata spregress, ml and R spatialreg do (they used t(n - k):
        # on a 40-unit ring the SDEM _cons interval was [0.979, 1.645]
        # against Stata's [0.990, 1.634]). SLX is least squares and keeps t.
        **({} if model_type == "slx" else {"inference": "z"}),
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
            # Read by EconometricResults for every interval it reports;
            # through 1.39.6 the five ML fits accepted alpha= and dropped it.
            "alpha": alpha,
        },
        data_info=data_info,
        diagnostics=diagnostics,
    )


def _attach_vcov(
    res: EconometricResults, vcov: Optional[np.ndarray], names: List[str]
) -> None:
    """Store the joint covariance of ``params`` (read by ``vcov()`` and
    ``sp.impacts``); absent on the large-``n`` path, which has none."""
    if vcov is not None:
        res.model_info["vcov"] = pd.DataFrame(vcov, index=names, columns=names)


def _sar_information_ses(
    M: sparse.csr_matrix, X: np.ndarray, beta: np.ndarray, rho: float, sigma2: float
) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """Full-information asymptotic SEs of ``(beta, rho)`` for a SAR/SDM ML fit.

    Returns ``(se_beta, se_rho, V)`` where ``se_beta`` is a length-``k``
    array and ``V`` the ``(k + 1) x (k + 1)`` covariance of ``(beta, rho)``
    (the leading block of the inverted information matrix; ``None`` on the
    large-``n`` path, which has no joint covariance). ``sp.impacts`` draws
    from ``V``.

    Uses the SAR information matrix (Ord 1975; Anselin 1988, *Spatial
    Econometrics*). With ``G = W (I - rho W)^{-1}`` and ``Xb = X @ beta`` the
    ``(beta, rho, sigma2)`` information matrix is

        I_bb = X'X / s2
        I_br = X' G Xb / s2
        I_rr = tr(G G) + tr(G' G) + (G Xb)'(G Xb) / s2
        I_rs = tr(G) / s2
        I_ss = n / (2 s2^2)

    We invert it **once** and read both blocks of the inverse:
    ``Var(beta) = [I^{-1}]_{beta,beta}`` (the leading ``k x k`` block, which
    accounts for the beta-rho covariance) and ``Var(rho) = [I^{-1}]_{rho,rho}``.
    This is exactly the asymptotic covariance ``spatialreg::lagsarlm`` reports.
    The naive concentrated ``sigma2 (X'X)^{-1}`` understates ``Var(beta)``
    because it treats ``rho`` as known — for a row-standardised ``W`` it can
    halve the intercept SE. (The earlier ``rho``-only formula already fixed the
    ``tr(G'G)`` drop that overstated the spatial SE by ~50%.)
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
        Info[k, k] = np.trace(G @ G) + np.trace(G.T @ G) + float(GXb @ GXb) / sigma2
        Info[k, k + 1] = np.trace(G) / sigma2
        Info[k + 1, k] = Info[k, k + 1]
        Info[k + 1, k + 1] = n / (2.0 * sigma2**2)
        try:
            Vc = np.linalg.inv(Info)
            se_beta = np.sqrt(np.clip(np.diag(Vc)[:k], 1e-12, None))
            se_rho = float(np.sqrt(max(Vc[k, k], 1e-12)))
            return se_beta, se_rho, Vc[: k + 1, : k + 1]
        except np.linalg.LinAlgError:
            se_beta = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
            return se_beta, float("nan"), None
    # Large-n sparse path: the full-information Var(beta) needs a dense
    # (I - rho W)^{-1}, which is intractable at scale, so beta falls back to
    # the concentrated conditional SE (documented approximation). The rho SE
    # keeps all three information terms (the small beta/sigma2 cross-terms are
    # dropped at scale). G v = W A^{-1} v.
    se_beta = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    rng = np.random.default_rng(0)
    m_probes = 30
    A = sparse.eye(n) - rho * M
    from scipy.sparse.linalg import splu

    lu = splu(A.tocsc())

    def _G(V: np.ndarray) -> np.ndarray:  # rows v_i -> (W A^{-1} v_i)
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
    se_rho = float(1.0 / np.sqrt(max(I_rho, 1e-10)))
    return se_beta, se_rho, None


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
        I_lam = np.trace(G @ G) + np.trace(G.T @ G) - 2.0 * trG**2 / n
        return float(1.0 / np.sqrt(max(I_lam, 1e-10)))
    # Large-n sparse path. G v = W A^{-1} v.
    rng = np.random.default_rng(0)
    m_probes = 30
    A = sparse.eye(n) - lam * M
    from scipy.sparse.linalg import splu

    lu = splu(A.tocsc())

    def _G(V: np.ndarray) -> np.ndarray:
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
    I_lam = tr_G2 + tr_GtG - 2.0 * trG**2 / n
    return float(1.0 / np.sqrt(max(I_lam, 1e-10)))


# ====================================================================== #
#  SAR
# ====================================================================== #


def sar(
    W: ArrayOrW,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
    weights: Optional[Any] = None,
) -> EconometricResults:
    """Spatial Autoregressive (Lag) Model: ``Y = ρWY + Xβ + ε``.

    Parameters
    ----------
    W : ndarray, scipy.sparse matrix, or statspai.spatial.weights.W
        n x n spatial weights matrix, rows in the order of ``data``.
    data : pd.DataFrame
        Data with the variables named in ``formula``.
    formula : str
        Model formula ``"y ~ x1 + x2"``; a constant is added.
    row_normalize : bool, default True
        Divide each row of ``W`` by its sum before fitting (rows that sum
        to zero are left as they are).
    weights : None
        Not supported, and refused with ``MethodIncompatibility`` rather
        than ignored: the likelihood is a joint one over all areas linked
        by ``W``, not a sum over independently sampled units, so sampling
        weights do not define a weighted version of it.

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
    if weights is not None:
        from ...exceptions import MethodIncompatibility

        raise MethodIncompatibility(
            "sar: weights= is not supported. The spatial lag model is a joint "
            "model of every area linked by W (its likelihood has the "
            "log|I - rho W| term), not a sum over independently sampled "
            "units, so sampling weights do not define a weighted likelihood "
            "of it.",
            recovery_hint="Fit on all areas without weights; under "
            "heteroskedasticity use sp.sar_gmm, or enter the area size as a "
            "covariate.",
            diagnostics={"weights": repr(weights)[:80]},
        )
    y, X, dep_var, indep = _parse_formula(formula, data)
    n, k = X.shape
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)
    Wy = M @ y

    eigvals = _eigvals_for_bounds(M)

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
        return float(-ll)

    rho_hat = _line_search(neg_conc_ll, eigvals)

    y_star = y - rho_hat * Wy
    beta = XtX_inv @ X.T @ y_star
    e = y_star - X @ beta
    sigma2 = float(e @ e) / n

    se_beta, se_rho, vcov = _sar_information_ses(M, X, beta, rho_hat, sigma2)

    var_names = ["const"] + list(indep) + ["rho"]
    params_vec = np.append(beta, rho_hat)
    se_vec = np.append(se_beta, se_rho)
    fitted = y - e
    log_lik = _full_loglik(n, sigma2, _logdet(M, rho_hat, eigvals))

    res = _make_results(
        alpha=alpha,
        model_type="sar",
        spatial_param_name="rho",
        spatial_param_value=rho_hat,
        var_names=var_names,
        params_vec=params_vec,
        se_vec=se_vec,
        sigma2=sigma2,
        resid=e,
        fitted=fitted,
        n=n,
        dep_var=dep_var,
        log_lik=log_lik,
        _w_sparse=M,
    )
    _attach_vcov(res, vcov, var_names)
    return res


# ====================================================================== #
#  SEM
# ====================================================================== #


def sem(
    W: ArrayOrW,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """Spatial Error Model: ``Y = Xβ + u, u = λWu + ε``.

    Parameters
    ----------
    W : ndarray, scipy.sparse matrix, or statspai.spatial.weights.W
        n x n spatial weights matrix, rows in the order of ``data``.
    data : pd.DataFrame
        Data with the variables named in ``formula``.
    formula : str
        Model formula ``"y ~ x1 + x2"``; a constant is added.
    row_normalize : bool, default True
        Divide each row of ``W`` by its sum before fitting (rows that sum
        to zero are left as they are).

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
        return float(-ll)

    lam_hat = _line_search(neg_conc_ll, eigvals)

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
        alpha=alpha,
        model_type="sem",
        spatial_param_name="lambda",
        spatial_param_value=lam_hat,
        var_names=var_names,
        params_vec=params_vec,
        se_vec=se_vec,
        sigma2=sigma2,
        resid=e,
        fitted=fitted,
        n=n,
        dep_var=dep_var,
        log_lik=log_lik,
        _w_sparse=M,
    )


# ====================================================================== #
#  SDM
# ====================================================================== #


def sdm(
    W: ArrayOrW,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """Spatial Durbin Model: ``Y = ρWY + Xβ + WXθ + ε``.

    Reports LeSage & Pace direct / indirect / total effects in the
    diagnostics when ``n`` is small enough for the dense ``(I - ρW)^{-1}``.

    Parameters
    ----------
    W : ndarray, scipy.sparse matrix, or statspai.spatial.weights.W
        n x n spatial weights matrix, rows in the order of ``data``.
    data : pd.DataFrame
        Data with the variables named in ``formula``.
    formula : str
        Model formula ``"y ~ x1 + x2"``; a constant is added.
    row_normalize : bool, default True
        Divide each row of ``W`` by its sum before fitting (rows that sum
        to zero are left as they are).

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

    def neg_conc_ll(rho: float) -> float:
        y_star = y - rho * Wy
        beta = XtX_inv @ X_aug.T @ y_star
        e = y_star - X_aug @ beta
        sigma2 = float(e @ e) / n
        if sigma2 <= 0:
            return 1e20
        ll = -n / 2.0 * np.log(sigma2)
        ll += _logdet(M, rho, eigvals)
        return float(-ll)

    rho_hat = _line_search(neg_conc_ll, eigvals)

    y_star = y - rho_hat * Wy
    beta_aug = XtX_inv @ X_aug.T @ y_star
    e = y_star - X_aug @ beta_aug
    sigma2 = float(e @ e) / n

    se_beta, se_rho, vcov = _sar_information_ses(M, X_aug, beta_aug, rho_hat, sigma2)

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
        beta_own = beta_aug[1:k]  # skip constant
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

    res = _make_results(
        alpha=alpha,
        model_type="sdm",
        spatial_param_name="rho",
        spatial_param_value=rho_hat,
        var_names=var_names,
        params_vec=params_vec,
        se_vec=se_vec,
        sigma2=sigma2,
        resid=e,
        fitted=fitted,
        n=n,
        dep_var=dep_var,
        log_lik=log_lik,
        extra_diag=extra,
        _w_sparse=M,
    )
    _attach_vcov(res, vcov, var_names)
    return res


# --------------------------------------------------------------------- #
#  SLX:  Y = X β + W X θ + ε   — OLS on augmented design
# --------------------------------------------------------------------- #


def slx(
    W: ArrayOrW,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
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
    # Gaussian ML log-likelihood, sigma^2 = RSS / n (R logLik(lmSLX)). Through
    # 1.40.2 the df-corrected sigma2 above was plugged in instead.
    log_lik = -n / 2 * (np.log(2 * np.pi * float(e @ e) / n) + 1.0)

    res = _make_results(
        alpha=alpha,
        model_type="slx",
        spatial_param_name="",
        spatial_param_value=0.0,
        var_names=var_names,
        params_vec=beta_aug,
        se_vec=se_beta,
        sigma2=sigma2,
        resid=e,
        fitted=fitted,
        n=n,
        dep_var=dep_var,
        log_lik=log_lik,
        extra_diag={},
        _w_sparse=M,
    )
    # OLS covariance (n - k), as R lmSLX; sp.impacts reads it for the
    # indirect and total SEs, which involve cov(beta, theta).
    _attach_vcov(res, sigma2 * XtX_inv, var_names)
    return res


# --------------------------------------------------------------------- #
#  SAC / SARAR:  Y = ρ W Y + X β + u,   u = λ W u + ε
# --------------------------------------------------------------------- #


def sac(
    W: ArrayOrW,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """SAC / SARAR: combined spatial lag + spatial error model.

    ``y = rho W y + X b + u,  u = lambda W u + e``,  ``e ~ N(0, sigma^2 I)``.
    ``(b, sigma^2)`` are profiled out and ``(rho, lambda)`` maximise the
    concentrated log-likelihood. Standard errors invert the observed
    information of the full log-likelihood in ``(b, rho, lambda, sigma^2)``
    (Stata ``spregress, ml dvarlag(W) errorlag(W)``, default ``vce(oim)``),
    so they account for the covariance between the regression and the
    spatial parameters. Through 1.39 the regression SEs treated ``rho`` and
    ``lambda`` as known and the spatial SEs read the diagonal of a
    numerical Hessian of the concentrated likelihood; both were too small.

    Parameters
    ----------
    W : ndarray, scipy.sparse matrix, or statspai.spatial.weights.W
        n x n spatial weights, rows in the order of ``data``.
    data : pd.DataFrame
    formula : str
        ``"y ~ x1 + x2"`` with plain column names; a constant is added.
    row_normalize : bool, default True
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params`` ``[const, x..., rho, lambda]`` with OIM standard errors;
        ``model_info['vcov']`` (their covariance), ``model_info['sigma2']``
        and ``model_info['se_sigma2']``, ``model_info['loglik']``.

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
    from ...exceptions import NumericalInstability
    from ._fiml import _fit

    y, X, dep_var, indep = _parse_formula(formula, data)
    n, k = X.shape
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)
    if np.linalg.matrix_rank(X) < k:
        raise NumericalInstability(
            "sp.sac: the regressors (with the constant) are collinear.",
            recovery_hint="Drop collinear or constant regressors.",
        )
    fit = _fit(y, X, M.toarray(), lag=True)
    var_names = ["const"] + list(indep) + ["rho", "lambda"]
    m = len(var_names)
    params_vec = np.concatenate([fit["g"], [fit["rho"], fit["lam"]]])
    se_vec = np.sqrt(np.clip(np.diag(fit["V"])[:m], 0.0, None))
    se_sigma2 = float(np.sqrt(max(fit["V"][-1, -1], 0.0)))
    res = _make_results(
        alpha=alpha,
        model_type="sac",
        spatial_param_name="rho,lambda",
        spatial_param_value=fit["rho"],
        var_names=var_names,
        params_vec=params_vec,
        se_vec=se_vec,
        sigma2=fit["sigma2"],
        resid=fit["resid_e"],
        fitted=y - fit["resid_u"],
        n=n,
        dep_var=dep_var,
        log_lik=fit["loglik"],
        extra_diag={
            "lambda_hat": round(float(fit["lam"]), 6),
            "se_sigma2": se_sigma2,
            "vce": "oim",
        },
        _w_sparse=M,
    )
    res.model_info["sigma2"] = float(fit["sigma2"])
    res.model_info["se_sigma2"] = se_sigma2
    res.model_info["loglik"] = float(fit["loglik"])
    res.model_info["vcov"] = pd.DataFrame(
        fit["V"][:m, :m], index=var_names, columns=var_names
    )
    return res


__all__ = ["sar", "sem", "sdm", "slx", "sac"]
