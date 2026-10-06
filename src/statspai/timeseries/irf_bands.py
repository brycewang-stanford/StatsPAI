"""Sampling uncertainty of VAR impulse responses.

Two routes to a band around the responses of :func:`statspai.irf`:

* asymptotic standard errors by the delta method (the route of Stata's
  ``irf create`` and of Lütkepohl's Proposition 3.6), and
* the residual bootstrap (``irf create, bs``; ``vars::irf(boot = TRUE)``).

Both treat the lag order as known.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..exceptions import MethodIncompatibility

__all__: List[str] = []

#: percentile band; band reflected about the estimate; bias-corrected
BOOT_KINDS = ("efron", "hall", "kilian")
#: above this largest companion root the uncorrected bootstrap warns
PERSISTENT_ROOT = 0.9


def response_array(
    B: np.ndarray,
    sigma: np.ndarray,
    k: int,
    p: int,
    periods: int,
    orthogonal: bool,
    cumulative: bool,
) -> np.ndarray:
    """Responses ``theta[s, i, j]`` of variable ``i`` at horizon ``s`` to
    shock ``j``, from the stacked coefficient matrix ``B`` (rows: lag 1
    variables, lag 2 variables, ..., deterministic terms)."""
    A = [B[lag * k : (lag + 1) * k, :].T for lag in range(p)]
    phi = [np.eye(k)]
    for s in range(1, periods + 1):
        acc = np.zeros((k, k))
        for j in range(min(s, p)):
            acc += phi[s - j - 1] @ A[j]
        phi.append(acc)
    P = np.linalg.cholesky(sigma) if orthogonal else np.eye(k)
    theta = np.stack([f @ P for f in phi])
    return np.cumsum(theta, axis=0) if cumulative else theta


def _duplication(k: int) -> np.ndarray:
    """``D`` with ``vec(S) = D vech(S)`` for symmetric ``S``."""
    cols = [(i, j) for j in range(k) for i in range(j, k)]
    D = np.zeros((k * k, len(cols)))
    for c, (i, j) in enumerate(cols):
        D[j * k + i, c] = 1.0
        D[i * k + j, c] = 1.0
    return D


def _elimination(k: int) -> np.ndarray:
    """``L`` with ``vech(S) = L vec(S)``."""
    cols = [(i, j) for j in range(k) for i in range(j, k)]
    L = np.zeros((len(cols), k * k))
    for c, (i, j) in enumerate(cols):
        L[c, j * k + i] = 1.0
    return L


def _commutation(k: int) -> np.ndarray:
    """``K`` with ``vec(M') = K vec(M)``."""
    Kmat = np.zeros((k * k, k * k))
    for i in range(k):
        for j in range(k):
            Kmat[i * k + j, j * k + i] = 1.0
    return Kmat


def _jacobians(
    B: np.ndarray,
    sigma: np.ndarray,
    xtx_inv: np.ndarray,
    n_obs: int,
    k: int,
    p: int,
    periods: int,
    orthogonal: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """Derivatives of ``vec(response_s)``, ``s = 0..periods``, with respect
    to the lag coefficients and to ``vech(sigma)``, and the covariances of
    those two blocks (which are asymptotically independent).

    ``sigma`` is the residual covariance that orthogonalises the shocks and
    ``xtx_inv`` the inverse cross-product of the regressors; the lag block
    of ``sigma (x) xtx_inv`` is the covariance of the lag coefficients.
    """
    kp = k * p
    A = [B[lag * k : (lag + 1) * k, :].T for lag in range(p)]
    companion = np.zeros((kp, kp))
    companion[:k, :] = np.hstack(A)
    if p > 1:
        companion[k:, :-k] = np.eye(kp - k)
    J = np.zeros((k, kp))
    J[:, :k] = np.eye(k)
    # covariance of alpha = vec([A_1 ... A_p]) (column-major)
    cov_alpha = np.kron(xtx_inv[:kp, :kp], sigma)

    phi = [np.eye(k)]
    for s in range(1, periods + 1):
        acc = np.zeros((k, k))
        for j in range(min(s, p)):
            acc += phi[s - j - 1] @ A[j]
        phi.append(acc)
    powers = [np.eye(kp)]
    for _ in range(periods):
        powers.append(powers[-1] @ companion.T)

    G: List[np.ndarray] = [np.zeros((k * k, k * kp))]
    for i in range(1, periods + 1):
        g = np.zeros((k * k, k * kp))
        for m in range(i):
            g += np.kron(J @ powers[i - 1 - m], phi[m])
        G.append(g)

    n_half = k * (k + 1) // 2
    if not orthogonal:
        zero = [np.zeros((k * k, n_half)) for _ in range(periods + 1)]
        return G, zero, cov_alpha, np.zeros((n_half, n_half))

    P = np.linalg.cholesky(sigma)
    L = _elimination(k)
    D = _duplication(k)
    D_plus = np.linalg.solve(D.T @ D, D.T)
    Hmat = L.T @ np.linalg.inv(
        L @ (np.eye(k * k) + _commutation(k)) @ np.kron(P, np.eye(k)) @ L.T
    )
    cov_sigma = 2.0 * D_plus @ np.kron(sigma, sigma) @ D_plus.T / n_obs
    PI = np.kron(P.T, np.eye(k))
    C = [PI @ g for g in G]
    Cbar = [np.kron(np.eye(k), f) @ Hmat for f in phi]
    return C, Cbar, cov_alpha, cov_sigma


def asymptotic_se(
    B: np.ndarray,
    sigma: np.ndarray,
    xtx_inv: np.ndarray,
    n_obs: int,
    k: int,
    p: int,
    periods: int,
    orthogonal: bool,
    cumulative: bool,
) -> np.ndarray:
    """Delta-method standard errors, same layout as :func:`response_array`."""
    C, Cbar, cov_alpha, cov_sigma = _jacobians(
        B, sigma, xtx_inv, n_obs, k, p, periods, orthogonal
    )
    if cumulative:
        C = list(np.cumsum(np.stack(C), axis=0))
        Cbar = list(np.cumsum(np.stack(Cbar), axis=0))
    out = np.zeros((periods + 1, k, k))
    for i in range(periods + 1):
        v = np.einsum("ij,jk,ik->i", C[i], cov_alpha, C[i]) + np.einsum(
            "ij,jk,ik->i", Cbar[i], cov_sigma, Cbar[i]
        )
        out[i] = np.sqrt(np.clip(v, 0.0, None)).reshape(k, k, order="F")
    return out


def fevd_se(
    B: np.ndarray,
    sigma: np.ndarray,
    xtx_inv: np.ndarray,
    n_obs: int,
    k: int,
    p: int,
    periods: int,
) -> np.ndarray:
    """Delta-method standard errors of the forecast-error variance
    decomposition, ``out[h, i, j]`` for the share of shock ``j`` in the
    ``h``-step forecast-error variance of variable ``i`` (zero at
    ``h = 0``).

    The share is a function of the orthogonalised responses at horizons
    ``0 .. h - 1``; its gradient is chained through their joint
    derivatives with respect to the lag coefficients and the residual
    covariance.
    """
    theta = response_array(B, sigma, k, p, periods, True, False)
    C, Cbar, cov_alpha, cov_sigma = _jacobians(
        B, sigma, xtx_inv, n_obs, k, p, periods, True
    )
    out = np.zeros((periods + 1, k, k))
    for h in range(1, periods + 1):
        sq = theta[:h] ** 2  # (h, i, j)
        share_num = sq.sum(axis=0)  # S[i, j]
        mse = share_num.sum(axis=1)  # M[i]
        for i in range(k):
            for j in range(k):
                ga = np.zeros(cov_alpha.shape[0])
                gs = np.zeros(cov_sigma.shape[0])
                for s in range(h):
                    for col in range(k):
                        # d share / d theta[s, i, col]
                        d = -2.0 * share_num[i, j] * theta[s, i, col]
                        if col == j:
                            d += 2.0 * theta[s, i, j] * mse[i]
                        d /= mse[i] ** 2
                        row = col * k + i  # position in vec(theta_s)
                        ga += d * C[s][row]
                        gs += d * Cbar[s][row]
                v = ga @ cov_alpha @ ga + gs @ cov_sigma @ gs
                out[h, i, j] = np.sqrt(max(float(v), 0.0))
    return out


def _max_root(A_stack: np.ndarray, k: int, p: int) -> float:
    """Largest modulus of the companion roots of the lag coefficients
    (rows: lag 1 variables, lag 2 variables, ...; columns: equations)."""
    kp = k * p
    companion = np.zeros((kp, kp))
    companion[:k, :] = A_stack.T
    if p > 1:
        companion[k:, :-k] = np.eye(kp - k)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _bias_corrected(A_hat: np.ndarray, bias: np.ndarray, k: int, p: int) -> np.ndarray:
    """``A_hat - delta * bias`` with the largest ``delta`` in ``1, 0.99,
    ...`` that leaves the VAR stationary (Kilian's adjustment). A
    non-stationary estimate is returned unchanged."""
    if _max_root(A_hat, k, p) >= 1.0:
        return A_hat
    delta = 1.0
    while delta > 0.0:
        cand = A_hat - delta * bias
        if _max_root(cand, k, p) < 1.0:
            return np.asarray(cand)
        delta -= 0.01
    return A_hat


def bootstrap_fits(
    var_result: Any,
    unbiased: bool,
    reps: int,
    seed: Optional[int],
    boot: str = "efron",
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Coefficients and residual covariance of ``reps`` bootstrap samples.

    Residual bootstrap with the design regenerated recursively: centred
    residuals are drawn with replacement, the series is rebuilt from the
    first ``p`` observations with the estimated coefficients (deterministic
    and exogenous terms as in the sample), and the VAR is re-estimated.

    ``boot='kilian'`` is the bias-corrected bootstrap ("bootstrap after
    bootstrap"): a first round of ``reps`` samples estimates the
    small-sample bias of the lag coefficients; the second round is
    generated from the bias-corrected coefficients, and each of its
    estimates is corrected by the same bias (both corrections shrunk when
    they would make the VAR non-stationary).
    """
    if boot == "kilian":
        k, p = int(var_result._k), int(var_result._lags)
        kp = k * p
        A_hat = np.asarray(var_result._B, float)[:kp, :]
        first = [Bb[:kp, :] for Bb, _ in _draws(var_result, unbiased, reps, seed, None)]
        bias = np.mean(first, axis=0) - A_hat
        A_tilde = _bias_corrected(A_hat, bias, k, p)
        second = None if seed is None else seed + 1
        for Bb, sigma_b in _draws(var_result, unbiased, reps, second, A_tilde):
            Bc = Bb.copy()
            Bc[:kp, :] = _bias_corrected(Bb[:kp, :], bias, k, p)
            yield Bc, sigma_b
        return
    k, p = int(var_result._k), int(var_result._lags)
    root = _max_root(np.asarray(var_result._B, float)[: k * p, :], k, p)
    if root > PERSISTENT_ROOT:
        import warnings

        from ..exceptions import AssumptionWarning

        warnings.warn(
            f"The largest root of the estimated VAR is {root:.3f}. With "
            "roots this close to one the bootstrap replicates are biased "
            "towards less persistence and percentile bands undercover, "
            f"badly in short samples (boot={boot!r}). boot='kilian' "
            "corrects the bias.",
            AssumptionWarning,
            stacklevel=4,
        )
    yield from _draws(var_result, unbiased, reps, seed, None)


def _draws(
    var_result: Any,
    unbiased: bool,
    reps: int,
    seed: Optional[int],
    lag_coefs: Optional[np.ndarray],
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """The residual bootstrap of :func:`bootstrap_fits`; ``lag_coefs``
    replaces the estimated lag coefficients in the data-generating step."""
    B = np.asarray(var_result._B, float)
    X = np.asarray(var_result._X, float)
    levels = np.asarray(var_result._levels, float)
    k, p = int(var_result._k), int(var_result._lags)
    kp = k * p
    T = X.shape[0]
    m = X.shape[1]
    resid = np.asarray(var_result.residuals, float)
    resid = resid - resid.mean(axis=0)
    det = X[:, kp:] @ B[kp:, :] if m > kp else np.zeros((T, k))
    A_stack = B[:kp, :] if lag_coefs is None else lag_coefs
    rng = np.random.default_rng(seed)
    start = levels[len(levels) - T - p : len(levels) - T]
    for _ in range(reps):
        e = resid[rng.integers(0, T, size=T)]
        y = np.empty((T + p, k))
        y[:p] = start
        for t in range(T):
            lagged = y[t : t + p][::-1].ravel()
            y[t + p] = lagged @ A_stack + det[t] + e[t]
        Xb = X.copy()
        for lag in range(1, p + 1):
            Xb[:, (lag - 1) * k : lag * k] = y[p - lag : p - lag + T]
        Yb = y[p:]
        Bb = np.linalg.lstsq(Xb, Yb, rcond=None)[0]
        u = Yb - Xb @ Bb
        yield Bb, u.T @ u / ((T - m) if unbiased else T)


def bootstrap_responses(
    var_result: Any,
    periods: int,
    orthogonal: bool,
    cumulative: bool,
    unbiased: bool,
    reps: int,
    seed: Optional[int],
    boot: str = "efron",
) -> np.ndarray:
    """``reps`` bootstrap replicates of the response array."""
    k, p = int(var_result._k), int(var_result._lags)
    out = np.empty((reps, periods + 1, k, k))
    fits = bootstrap_fits(var_result, unbiased, reps, seed, boot)
    for b, (Bb, sigma_b) in enumerate(fits):
        out[b] = response_array(Bb, sigma_b, k, p, periods, orthogonal, cumulative)
    return out


def bands(
    var_result: Any,
    theta: np.ndarray,
    sigma: np.ndarray,
    periods: int,
    orthogonal: bool,
    cumulative: bool,
    unbiased: bool,
    ci: str,
    alpha: float,
    reps: int,
    seed: Optional[int],
    boot: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Standard errors and the ``1 - alpha`` band of ``theta``."""
    from scipy import stats

    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility("irf: alpha must be in (0, 1).")
    k, p = int(var_result._k), int(var_result._lags)
    if ci == "asymptotic":
        if var_result._XtX_inv is None:
            raise MethodIncompatibility(
                "irf: this VARResult does not carry the regressor moments; "
                "re-fit with sp.var(...)."
            )
        se = asymptotic_se(
            np.asarray(var_result._B, float),
            sigma,
            np.asarray(var_result._XtX_inv, float),
            int(var_result.n_obs),
            k,
            p,
            periods,
            orthogonal,
            cumulative,
        )
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        return se, theta - z * se, theta + z * se, {"ci": ci, "alpha": alpha}
    if ci != "bootstrap":
        raise MethodIncompatibility(
            f"irf: ci={ci!r} is not None, 'asymptotic' or 'bootstrap'.",
            recovery_hint="ci='asymptotic' is the delta method (Stata's "
            "default), ci='bootstrap' the residual bootstrap.",
        )
    if boot not in BOOT_KINDS:
        raise MethodIncompatibility(
            "irf: boot must be 'efron', 'hall' or 'kilian'.",
            recovery_hint="'kilian' is the bias-corrected bootstrap, for "
            "persistent series and short samples.",
        )
    if reps < 20:
        raise MethodIncompatibility(
            f"irf: reps={reps} is too few for percentile bands.",
            recovery_hint="Use at least a few hundred replications.",
        )
    if var_result._X is None or var_result._levels is None:
        raise MethodIncompatibility(
            "irf: this VARResult does not carry its data; re-fit with "
            "sp.var(...) to bootstrap."
        )
    draws = bootstrap_responses(
        var_result, periods, orthogonal, cumulative, unbiased, reps, seed, boot
    )
    lo = np.quantile(draws, alpha / 2.0, axis=0)
    hi = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    if boot == "hall":
        lo, hi = 2.0 * theta - hi, 2.0 * theta - lo
    info = {"ci": ci, "alpha": alpha, "reps": reps, "seed": seed, "boot": boot}
    return draws.std(axis=0, ddof=1), lo, hi, info
