"""
General GMM (Generalized Method of Moments) framework.

Estimates the parameter solving ``E[g(theta, data)] = 0`` for an arbitrary
user-supplied moment function, with one-step, two-step, iterated, and
continuously-updated weights.

Equivalent to Stata's ``gmm`` and R's ``gmm::gmm()``.

Design notes
------------
Three things in here are easy to get subtly wrong and are therefore
handled explicitly rather than left to a default:

**The moment covariance.** ``S`` can be estimated centred
(``E[(m - m̄)(m - m̄)']``, R's ``gmm`` default) or uncentred
(``E[m m']``, Stata's). At the optimum the two nearly coincide, but only
nearly, and the gap propagates into the weight, the standard errors, and
the J statistic. ``center`` exposes the choice; the default follows
Stata.

**The "unadjusted" variance.** ``(D'WD)^{-1}/n`` is the variance of the
GMM estimator *only when ``W`` is the efficient weight* ``S^{-1}``. With
any other weight the estimator's variance is the sandwich, and reporting
the efficient formula understates it. Asking for ``se='unadjusted'``
under an inefficient weight now warns instead of silently returning the
wrong number.

**Affine moment conditions.** When ``g`` is affine in ``theta`` — the
common case, covering every linear IV / 2SLS moment — the GMM problem has
a closed-form solution and running an optimiser on it only adds error.
The Jacobian is probed at two parameter values; if it does not move, the
closed form is used and ``diagnostics['n_iter']`` is 0.

References
----------
Hansen, L.P. (1982).
"Large Sample Properties of Generalized Method of Moments Estimators."
*Econometrica*, 50(4), 1029-1054. [@hansen1982large]

Hansen, L.P., Heaton, J. & Yaron, A. (1996).
"Finite-Sample Properties of Some Alternative GMM Estimators."
*Journal of Business & Economic Statistics*, 14(3), 262-280. [@hansen1996finite]

Newey, W.K. & West, K.D. (1987).
"A Simple, Positive Semi-Definite, Heteroskedasticity and
Autocorrelation Consistent Covariance Matrix."
*Econometrica*, 55(3), 703-708. [@newey1987simple]
"""

import warnings
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.results import EconometricResults

_VALID_METHODS = ("onestep", "twostep", "iterative", "cue")
_VALID_SE = ("robust", "unadjusted")
_VALID_VCOV = ("mds", "iid", "hac", "cluster")


def _as_moments(moment_fn, theta, data) -> np.ndarray:
    """Evaluate the moment function into a 2-D ``(n, q)`` array."""
    G = np.asarray(moment_fn(np.asarray(theta, dtype=float), data), dtype=float)
    if G.ndim == 1:
        G = G[:, None]
    return G


def _numeric_jacobian(
    g_bar: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, q: int
) -> np.ndarray:
    """Central-difference Jacobian ``D = d gbar / d theta'`` of shape (q, k)."""
    theta = np.asarray(theta, dtype=float)
    k = theta.size
    D = np.zeros((q, k))
    for j in range(k):
        step = 1e-6 * max(1.0, abs(float(theta[j])))
        ej = np.zeros(k)
        ej[j] = step
        D[:, j] = (g_bar(theta + ej) - g_bar(theta - ej)) / (2.0 * step)
    return D


def _omega(
    M: np.ndarray,
    *,
    vcov: str,
    center: bool,
    cluster: Optional[np.ndarray],
    hac_bandwidth: Optional[int],
) -> np.ndarray:
    """Moment covariance ``S``, by the requested convention.

    ``center`` subtracts the sample mean of the moments first: R's ``gmm``
    does this by default, Stata's does not. The choice changes ``S``, and
    through it the efficient weight, the standard errors and J.
    """
    n = M.shape[0]
    if center:
        M = M - M.mean(axis=0)

    if vcov == "cluster":
        codes = np.asarray(cluster)
        S = np.zeros((M.shape[1], M.shape[1]))
        for code in np.unique(codes):
            gc = M[codes == code].sum(axis=0)
            S += np.outer(gc, gc)
        return S / n

    if vcov == "hac":
        bw = int(hac_bandwidth) if hac_bandwidth else 1
        S = M.T @ M / n
        # Bartlett weights evaluated at lag / bandwidth, so the weight
        # vanishes at lag == bandwidth. This is R ``sandwich``'s
        # convention; Newey-West state it as 1 - lag/(q+1) for truncation
        # lag q, which is the same thing with bandwidth = q + 1. Getting
        # this off by one moves HAC standard errors by percent-level
        # amounts.
        for lag in range(1, bw):
            w = 1.0 - lag / bw
            gamma = M[lag:].T @ M[:-lag] / n
            S += w * (gamma + gamma.T)
        return S

    # 'mds' and 'iid' both estimate S by the moment outer product. For
    # cross-sectional moments the two assumptions coincide; they part
    # company only through the serial-correlation term, which is what the
    # 'hac' branch adds.
    return M.T @ M / n


def _safe_inv(A: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:  # pragma: no cover - near-singular guard
        return np.linalg.pinv(A)


def gmm(
    moment_fn: Callable[[np.ndarray, Optional[pd.DataFrame]], Any],
    theta0: np.ndarray,
    data: Optional[pd.DataFrame] = None,
    W: Optional[np.ndarray] = None,
    method: str = "twostep",
    se: str = "robust",
    maxiter: int = 200,
    tol: float = 1e-8,
    param_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    jacobian: Optional[Callable[[np.ndarray, Optional[pd.DataFrame]], Any]] = None,
    vcov: str = "mds",
    cluster: Optional[Any] = None,
    hac_bandwidth: Optional[int] = None,
    center: bool = False,
) -> EconometricResults:
    """
    General GMM estimator for arbitrary moment conditions.

    Minimizes ``Q(theta) = gbar(theta)' W gbar(theta)`` where
    ``gbar(theta) = (1/n) sum_i g_i(theta)``.

    Parameters
    ----------
    moment_fn : callable
        ``g(theta, data) -> ndarray`` of shape ``(n, q)``: the moment
        contribution of each observation.
    theta0 : np.ndarray
        Starting values. Also the expansion point for the closed form when
        the moments are affine.
    data : pd.DataFrame, optional
        Passed through to ``moment_fn``.
    W : np.ndarray, optional
        Weighting matrix ``(q, q)`` for the first step. Defaults to the
        identity. Supplying one and asking for ``se='unadjusted'`` warns
        unless it happens to be efficient.
    method : {'onestep', 'twostep', 'iterative', 'cue'}, default 'twostep'
    se : {'robust', 'unadjusted'}, default 'robust'
        ``'robust'`` returns the sandwich, valid for any ``W``.
        ``'unadjusted'`` returns the efficient-GMM variance
        ``(D'WD)^{-1}/n``, which describes the estimator *only* at the
        efficient weight; otherwise it warns.
    maxiter : int, default 200
    tol : float, default 1e-8
    param_names : list of str, optional
    alpha : float, default 0.05
    jacobian : callable, optional
        ``D(theta, data) -> ndarray`` of shape ``(q, k)``, the derivative
        of the *average* moment. Supplying it removes finite-difference
        error from the standard errors and from the affine test.
    vcov : {'mds', 'iid', 'hac', 'cluster'}, default 'mds'
        Estimator for the moment covariance ``S``.
    cluster : array-like, optional
        Group labels, one per moment row. Required when ``vcov='cluster'``.
    hac_bandwidth : int, optional
        Bartlett bandwidth for ``vcov='hac'``; the kernel vanishes at
        ``lag == hac_bandwidth``.
    center : bool, default False
        Centre the moments before forming ``S``. ``False`` matches Stata,
        ``True`` matches R's ``gmm``.

    Returns
    -------
    EconometricResults
        ``diagnostics`` carries ``J_stat`` / ``J_df`` / ``J_p``,
        ``converged``, and ``n_iter`` (0 when the closed form was used).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> z1, z2, u = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    >>> x1 = 0.7 * z1 + 0.5 * z2 + u + rng.normal(size=n)
    >>> y = 1.0 + 2.0 * x1 + u + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'x1': x1, 'z1': z1, 'z2': z2})
    >>>
    >>> def moment_fn(theta, data):
    ...     y, X, Z = data['y'].values, data[['x1']].values, data[['z1', 'z2']].values
    ...     X_full = np.column_stack([np.ones(len(y)), X])
    ...     resid = y - X_full @ theta
    ...     Z_full = np.column_stack([np.ones(len(y)), Z])
    ...     return resid[:, np.newaxis] * Z_full
    >>>
    >>> result = sp.gmm(moment_fn, theta0=np.zeros(2), data=df,
    ...                 param_names=['_cons', 'x1'])
    >>> bool(result is not None)
    True
    >>> result.diagnostics['n_iter']  # affine moments -> closed form
    0
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"gmm: method must be one of {_VALID_METHODS}, got {method!r}."
        )
    if se not in _VALID_SE:
        raise ValueError(f"gmm: se must be one of {_VALID_SE}, got {se!r}.")
    if vcov not in _VALID_VCOV:
        raise ValueError(f"gmm: vcov must be one of {_VALID_VCOV}, got {vcov!r}.")

    theta0 = np.asarray(theta0, dtype=float).ravel()
    k = theta0.size
    G0 = _as_moments(moment_fn, theta0, data)
    n, q = int(G0.shape[0]), int(G0.shape[1])

    if q < k:
        raise ValueError(
            f"gmm: the model is under-identified — {q} moment condition(s) "
            f"for {k} parameter(s). Add moments or drop parameters; this is "
            "an identification failure, not a numerical one."
        )

    if vcov == "cluster":
        if cluster is None:
            raise ValueError("gmm: vcov='cluster' requires a `cluster` array.")
        cluster = np.asarray(cluster).ravel()
        if cluster.size != n:
            raise ValueError(
                f"gmm: `cluster` has {cluster.size} entries but the moment "
                f"function returned {n} rows."
            )

    def G_mat(theta: np.ndarray) -> np.ndarray:
        return _as_moments(moment_fn, theta, data)

    def g_bar(theta: np.ndarray) -> np.ndarray:
        return G_mat(theta).mean(axis=0)

    def S_at(theta: np.ndarray) -> np.ndarray:
        return _omega(
            G_mat(theta),
            vcov=vcov,
            center=center,
            cluster=cluster,
            hac_bandwidth=hac_bandwidth,
        )

    def D_at(theta: np.ndarray) -> np.ndarray:
        if jacobian is not None:
            return np.asarray(jacobian(np.asarray(theta, float), data), dtype=float)
        return _numeric_jacobian(g_bar, theta, q)

    def objective(theta: np.ndarray, W_mat: np.ndarray) -> float:
        gb = g_bar(theta)
        return float(gb @ W_mat @ gb)

    # ---- Affine detection -------------------------------------------------
    # If dgbar/dtheta' does not depend on theta the problem is a linear
    # least-distance one with a closed-form solution; running BFGS on it
    # only introduces optimiser error.
    D0 = D_at(theta0)
    probe = theta0 + 0.5 * (np.abs(theta0) + 1.0)
    D1 = D_at(probe)
    scale = max(1.0, float(np.max(np.abs(D0))))
    is_affine = bool(np.allclose(D0, D1, rtol=1e-7, atol=1e-9 * scale))

    def solve(W_mat: np.ndarray, start: np.ndarray) -> tuple:
        """Return ``(theta_hat, n_iter, converged)`` for weight ``W_mat``."""
        if is_affine:
            g0 = g_bar(start)
            DtW = D0.T @ W_mat
            step = -_safe_inv(DtW @ D0) @ (DtW @ g0)
            return start + step, 0, True
        res = minimize(
            lambda t: objective(t, W_mat),
            start,
            method="BFGS",
            options={"maxiter": maxiter, "gtol": tol},
        )
        return res.x, int(res.nit), bool(res.success)

    n_iter_total = 0
    converged = True

    if method == "cue":

        def cue_objective(theta: np.ndarray) -> float:
            gb = g_bar(theta)
            return float(gb @ _safe_inv(S_at(theta)) @ gb)

        res = minimize(
            cue_objective,
            theta0,
            method="BFGS",
            options={"maxiter": maxiter, "gtol": tol},
        )
        theta_hat = res.x
        n_iter_total, converged = int(res.nit), bool(res.success)
        W_opt = _safe_inv(S_at(theta_hat))
        weight_is_efficient = True
    else:
        W1 = np.eye(q) if W is None else np.asarray(W, dtype=float)
        theta_hat, it, ok = solve(W1, theta0)
        n_iter_total += it
        converged = converged and ok
        W_opt = W1
        weight_is_efficient = W is None and False  # identity is not efficient

        if method in ("twostep", "iterative"):
            W_opt = _safe_inv(S_at(theta_hat))
            theta_hat, it, ok = solve(W_opt, theta_hat)
            n_iter_total += it
            converged = converged and ok
            weight_is_efficient = True

            if method == "iterative":
                for _ in range(maxiter):
                    theta_old = theta_hat.copy()
                    W_opt = _safe_inv(S_at(theta_hat))
                    theta_hat, it, ok = solve(W_opt, theta_hat)
                    n_iter_total += it
                    converged = converged and ok
                    if np.max(np.abs(theta_hat - theta_old)) < tol:
                        break

    # ---- Variance ---------------------------------------------------------
    S_hat = S_at(theta_hat)
    D = D_at(theta_hat)

    # The weight used *in estimation* defines the objective, and therefore
    # Hansen's J. The variance is a different question: at the efficient
    # weight the sandwich and (D'S^-1 D)^-1 coincide only if both are
    # evaluated at the same S, so the variance re-anchors S at the final
    # estimate while J keeps the weight that was actually minimised.
    W_est = W_opt
    W_var = _safe_inv(S_hat) if weight_is_efficient else W_opt

    DtW = D.T @ W_var
    DtWD_inv = _safe_inv(DtW @ D)

    if se == "robust":
        V = DtWD_inv @ DtW @ S_hat @ DtW.T @ DtWD_inv / n
    else:
        if not weight_is_efficient:
            warnings.warn(
                "gmm(se='unadjusted'): the weighting matrix is not efficient "
                "(W != S^-1), so (D'WD)^-1/n is not the variance of the "
                "estimator that was computed and understates it. Use "
                "se='robust' for the sandwich, or a weight-updating method "
                "('twostep', 'iterative', 'cue').",
                UserWarning,
                stacklevel=2,
            )
        V = DtWD_inv / n

    se_hat = np.sqrt(np.abs(np.diag(V)))

    # ---- Over-identification test -----------------------------------------
    J_stat = float(n * objective(theta_hat, W_est))
    J_df = q - k
    J_p = float(1.0 - stats.chi2.cdf(J_stat, J_df)) if J_df > 0 else float("nan")

    if param_names is None:
        param_names = [f"theta_{i}" for i in range(k)]

    return EconometricResults(
        params=pd.Series(theta_hat, index=param_names),
        std_errors=pd.Series(se_hat, index=param_names),
        model_info={
            "model_type": f"GMM ({method})",
            "n_moments": q,
            "n_params": k,
            "overidentified": q > k,
            "vcov": vcov,
            "center": center,
            "affine": is_affine,
            "analytic_jacobian": jacobian is not None,
        },
        data_info={"n_obs": n, "df_resid": n - k},
        diagnostics={
            "J_stat": J_stat,
            "J_df": J_df,
            "J_p": J_p,
            "gmm_objective": float(objective(theta_hat, W_est)),
            "converged": bool(converged),
            "n_iter": int(n_iter_total),
        },
    )
