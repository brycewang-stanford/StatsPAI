"""Numerical core of ``sp.blp``: draws, the share inversion, the GMM
objective with its gradient, elasticities and standard errors."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Halton sequence generator for quasi-Monte Carlo integration
# ---------------------------------------------------------------------------


def _halton_sequence(n: int, dim: int, seed: int | None = None) -> np.ndarray:
    """
    Generate a Halton quasi-random sequence mapped to N(0,1) draws.

    Parameters
    ----------
    n : int
        Number of draws.
    dim : int
        Number of dimensions (one per random coefficient).
    seed : int or None
        If provided, shuffle the sequence for randomization.

    Returns
    -------
    np.ndarray
        Shape (n, dim) array of standard-normal draws.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if dim > len(primes):
        raise ValueError(
            f"x_random has {dim} variables; max supported is {len(primes)}."
        )

    def _halton_1d(size: int, base: int) -> np.ndarray:
        seq = np.zeros(size)
        for i in range(size):
            f, r = 1.0, 0.0
            idx = i + 1
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            seq[i] = r
        return seq

    if dim == 0:
        return np.zeros((n, 0))
    rng = np.random.default_rng(seed)
    draws = np.column_stack([_halton_1d(n, primes[d]) for d in range(dim)])

    if seed is not None:
        for d in range(dim):
            draws[:, d] = rng.permutation(draws[:, d])

    # Map uniform to standard normal via inverse CDF
    draws = np.clip(draws, 1e-6, 1 - 1e-6)
    draws = stats.norm.ppf(draws)
    return np.asarray(draws, dtype=float)


def _group_sum(values: np.ndarray, *keys: np.ndarray) -> np.ndarray:
    """Sum of each column of ``values`` within the cells of ``keys``."""
    frame = pd.DataFrame(values)
    by = [pd.Series(k).reset_index(drop=True) for k in keys]
    return np.asarray(frame.groupby(by, sort=False).transform("sum"), dtype=float)


# ---------------------------------------------------------------------------
# Core BLP computation
# ---------------------------------------------------------------------------


def _market_rows(market_ids: np.ndarray) -> list:
    """Row positions of each market, in ``np.unique`` order."""
    return [np.where(market_ids == m)[0] for m in np.unique(market_ids)]


def _choice_probabilities(delta_m: np.ndarray, mu_m: np.ndarray) -> np.ndarray:
    """Consumer-level choice probabilities ``s_jr`` of one market, ``(J_m, R)``."""
    v = delta_m[:, np.newaxis] + mu_m
    v_max = np.maximum(v.max(axis=0), 0.0)
    exp_v = np.exp(v - v_max)
    return np.asarray(exp_v / (exp_v.sum(axis=0) + np.exp(-v_max)), dtype=float)


def _compute_market_shares(
    delta: np.ndarray, mu: np.ndarray, market_ids: np.ndarray
) -> np.ndarray:
    """
    Compute predicted market shares via Monte Carlo integration.

    Parameters
    ----------
    delta : np.ndarray, shape (J,)
        Mean utilities for all product-market observations.
    mu : np.ndarray, shape (J, R)
        Individual-specific utility deviations, where R = n_draws.
    market_ids : np.ndarray, shape (J,)
        Market identifier for each observation.

    Returns
    -------
    np.ndarray, shape (J,)
        Predicted market shares.
    """
    J = len(delta)
    shares = np.zeros(J)

    unique_markets = np.unique(market_ids)
    market_indices = {m: np.where(market_ids == m)[0] for m in unique_markets}

    for m, idx in market_indices.items():
        # Utility: delta_j + mu_jr for each draw r
        # Shape: (J_m, R)
        v = delta[idx, np.newaxis] + mu[idx]

        # Numerically stable softmax with outside option (utility 0)
        v_max = np.maximum(v.max(axis=0), 0.0)
        exp_v = np.exp(v - v_max)
        exp_outside = np.exp(-v_max)
        denom = exp_v.sum(axis=0) + exp_outside  # (R,)

        # Individual choice probabilities, averaged over draws
        s_ir = exp_v / denom  # (J_m, R)
        shares[idx] = s_ir.mean(axis=1)

    return shares


def _contraction_mapping(
    s_obs: np.ndarray,
    delta: np.ndarray,
    mu: np.ndarray,
    market_ids: np.ndarray,
    tol: float,
    maxiter: int,
) -> tuple[np.ndarray, bool]:
    """
    BLP contraction mapping: δ_new = δ_old + ln(s_obs) - ln(s_pred).

    Parameters
    ----------
    s_obs : np.ndarray
        Observed market shares.
    delta : np.ndarray
        Initial mean utilities.
    mu : np.ndarray
        Individual-specific deviations (J, R).
    market_ids : np.ndarray
        Market identifiers.
    tol : float
        Convergence tolerance (sup norm).
    maxiter : int
        Maximum iterations.

    Returns
    -------
    delta : np.ndarray
        Converged mean utilities.
    converged : bool
    """
    log_s_obs = np.log(np.maximum(s_obs, 1e-300))
    out = np.array(delta, dtype=float)
    converged = True
    for idx in _market_rows(market_ids):
        out[idx], ok = _contract_market(log_s_obs[idx], out[idx], mu[idx], tol, maxiter)
        converged = converged and ok
    return out, converged


def _contract_market(
    log_s: np.ndarray, delta: np.ndarray, mu: np.ndarray, tol: float, maxiter: int
) -> tuple[np.ndarray, bool]:
    """One market's fixed point ``delta = delta + ln s_obs - ln s(delta)``.

    Each pass takes two plain contraction steps and then one squared
    extrapolation step (SQUAREM with step length ``-||r|| / ||v||``); a
    market stops as soon as a plain step moves it by less than ``tol``, so
    the returned point satisfies the same criterion as the unaccelerated
    iteration. ``maxiter`` bounds the number of share evaluations.
    """

    def step(d: np.ndarray) -> np.ndarray:
        share = _choice_probabilities(d, mu).mean(axis=1)
        return np.asarray(d + log_s - np.log(np.maximum(share, 1e-300)))

    x = delta
    evals = 0
    while evals < maxiter:
        x1 = step(x)
        r = x1 - x
        if np.max(np.abs(r)) < tol:
            return x1, True
        x2 = step(x1)
        evals += 2
        if np.max(np.abs(x2 - x1)) < tol:
            return x2, True
        v = (x2 - x1) - r
        v_norm = float(np.sqrt(v @ v))
        if v_norm == 0.0:
            x = x2
            continue
        a = -float(np.sqrt(r @ r)) / v_norm
        x = step(x - 2.0 * a * r + a * a * v)
        evals += 1
        if not np.all(np.isfinite(x)):
            x = x2
    return x, False


def _compute_mu(
    X_random: np.ndarray, sigma: np.ndarray, draws: np.ndarray
) -> np.ndarray:
    """
    Compute individual-specific utility deviations.

    Parameters
    ----------
    X_random : np.ndarray, shape (J, K_r)
        Random coefficient characteristics.
    sigma : np.ndarray, shape (K_r,)
        Standard deviations of random coefficients.
    draws : np.ndarray, shape (R, K_r)
        Standard normal draws.

    Returns
    -------
    np.ndarray, shape (J, R)
        mu_{jr} = Σ_k σ_k * x_{jk} * v_{rk}
    """
    # (J, K_r) @ diag(sigma) @ (K_r, R) = (J, R)
    return np.asarray((X_random * sigma) @ draws.T, dtype=float)


def _iv_regression(
    delta: np.ndarray, X: np.ndarray, Z: np.ndarray, W: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    IV/2SLS regression of delta on X using instruments Z with weighting matrix W.

    delta = X @ theta + xi
    theta = (X'Z W Z'X)^{-1} X'Z W Z'delta

    Parameters
    ----------
    delta : np.ndarray, shape (N,)
    X : np.ndarray, shape (N, K)
    Z : np.ndarray, shape (N, L)
    W : np.ndarray, shape (L, L)

    Returns
    -------
    theta : np.ndarray, shape (K,)
    xi : np.ndarray, shape (N,)
    """
    # Standard 2SLS: theta = (X'Pz X)^{-1} X'Pz delta where Pz = Z(Z'Z)^{-1}Z'
    # With GMM weighting: theta = (X'Z W Z'X)^{-1} X'Z W Z'delta
    ZtX = Z.T @ X
    Ztd = Z.T @ delta
    A = ZtX.T @ W @ ZtX
    b = ZtX.T @ W @ Ztd
    theta = np.linalg.solve(A, b)
    xi = delta - X @ theta
    return theta, xi


def _gmm_objective(
    sigma: np.ndarray,
    s_obs: np.ndarray,
    X_linear: np.ndarray,
    X_random: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    draws: np.ndarray,
    market_ids: np.ndarray,
    delta_init: np.ndarray,
    tol_inner: float,
    maxiter_inner: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate the GMM objective for given nonlinear parameters sigma.

    Returns
    -------
    obj : float
        GMM objective value.
    xi : np.ndarray
        Structural residuals.
    delta : np.ndarray
        Converged mean utilities.
    """
    sigma = np.abs(sigma)  # sigma must be non-negative
    mu = _compute_mu(X_random, sigma, draws)
    delta, converged = _contraction_mapping(
        s_obs, delta_init, mu, market_ids, tol_inner, maxiter_inner
    )

    if not converged:
        warnings.warn("BLP contraction mapping did not converge.", stacklevel=2)

    # IV regression to get linear params and structural error
    theta, xi = _iv_regression(delta, X_linear, Z, W)

    # GMM objective: xi'Z W Z'xi
    Zxi = Z.T @ xi
    obj = float(Zxi.T @ W @ Zxi)

    return obj, xi, delta


def _objective_and_gradient(
    sigma: np.ndarray,
    s_obs: np.ndarray,
    X_linear: np.ndarray,
    X_random: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    draws: np.ndarray,
    market_ids: np.ndarray,
    delta_init: np.ndarray,
    tol_inner: float,
    maxiter_inner: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """GMM objective ``xi'Z W Z'xi`` and its gradient in ``sigma``.

    The linear parameters are concentrated out, and their first-order
    condition ``X'Z W Z'xi = 0`` removes their derivative from the chain
    rule, so the gradient is ``2 (d delta / d sigma)' Z W Z'xi``.
    """
    mu = _compute_mu(X_random, sigma, draws)
    delta, converged = _contraction_mapping(
        s_obs, delta_init, mu, market_ids, tol_inner, maxiter_inner
    )
    if not converged:
        warnings.warn("BLP contraction mapping did not converge.", stacklevel=2)
    _, xi = _iv_regression(delta, X_linear, Z, W)
    WZxi = W @ (Z.T @ xi)
    obj = float((Z.T @ xi) @ WZxi)
    D = _delta_jacobian(delta, mu, X_random, draws, market_ids)
    grad = 2.0 * (D.T @ (Z @ WZxi))
    return obj, np.asarray(grad, dtype=float), delta


# ---------------------------------------------------------------------------
# Elasticity computation
# ---------------------------------------------------------------------------


def _share_jacobians(
    delta: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    sigma_price: float,
    draws: np.ndarray,
    market_ids: np.ndarray,
    price_draw_col: Optional[int] = None,
) -> dict:
    """``{market: dS}`` with ``dS[j, k] = ds_j / dp_k``.

    With a random coefficient on price the consumer-level price coefficient
    is ``alpha_r = alpha + sigma_price * nu_r`` (``nu_r`` the price column of
    ``draws``), and

        ds_j/dp_k = (1/R) sum_r alpha_r s_jr (1{j=k} - s_kr).

    Without one, ``alpha_r = alpha`` for every draw.
    """
    R = mu.shape[1]
    if price_draw_col is not None and sigma_price != 0.0:
        alpha_r = alpha + sigma_price * draws[:, price_draw_col]  # (R,)
    else:
        alpha_r = np.full(R, alpha)
    out = {}
    for m, idx in zip(np.unique(market_ids), _market_rows(market_ids)):
        s_ir = _choice_probabilities(delta[idx], mu[idx])  # (J_m, R)
        a_s = s_ir * alpha_r[np.newaxis, :]  # alpha_r * s_jr
        out[m] = np.diag(a_s.mean(axis=1)) - (a_s @ s_ir.T) / R
    return out


def _compute_elasticities(
    delta: np.ndarray,
    mu: np.ndarray,
    prices: np.ndarray,
    alpha: float,
    sigma_price: float,
    draws: np.ndarray,
    market_ids: np.ndarray,
    price_draw_col: Optional[int] = None,
) -> dict:
    """
    Own- and cross-price elasticities ``E_jk = ds_j/dp_k * p_k / s_j``.

    Returns
    -------
    dict
        {market_id: elasticity_matrix} where matrix is (J_m, J_m).
    """
    jac = _share_jacobians(
        delta, mu, alpha, sigma_price, draws, market_ids, price_draw_col
    )
    elasticities = {}
    for m, idx in zip(np.unique(market_ids), _market_rows(market_ids)):
        s_j = _choice_probabilities(delta[idx], mu[idx]).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            E = jac[m] * prices[idx][np.newaxis, :] / s_j[:, np.newaxis]
        E[~np.isfinite(E)] = 0.0
        elasticities[m] = E
    return elasticities


def _nested_logit_elasticities(
    shares: np.ndarray,
    within: np.ndarray,
    prices: np.ndarray,
    alpha: float,
    rho: float,
    market_ids: np.ndarray,
    nests: np.ndarray,
) -> dict:
    """
    Price elasticities of the one-level nested logit (Berry 1994).

    With ``alpha`` the coefficient on price, ``s_{j|g}`` the within-nest
    share and ``rho`` the nesting parameter,

        E_jj = alpha p_j [1/(1-rho) - rho/(1-rho) s_{j|g} - s_j]
        E_jk = -alpha p_k [rho/(1-rho) s_{k|g} + s_k]     (k in the nest of j)
        E_jk = -alpha p_k s_k                             (other nests)

    ``rho = 0`` gives the plain logit.
    """
    out = {}
    r = rho / (1.0 - rho)
    for m in np.unique(market_ids):
        idx = np.where(market_ids == m)[0]
        s, w, p = shares[idx], within[idx], prices[idx]
        same = nests[idx][:, None] == nests[idx][None, :]
        E = -alpha * p[None, :] * (s[None, :] + r * w[None, :] * same)
        E[np.diag_indices(len(idx))] = alpha * p * (1.0 / (1.0 - rho) - r * w - s)
        out[m] = E
    return out


# ---------------------------------------------------------------------------
# Standard-error computation
# ---------------------------------------------------------------------------


def _delta_jacobian(
    delta: np.ndarray,
    mu: np.ndarray,
    X_random: np.ndarray,
    draws: np.ndarray,
    market_ids: np.ndarray,
) -> np.ndarray:
    """Analytic ``d delta / d sigma`` by the implicit function theorem.

    Within market ``m`` the contraction solves ``s(delta, sigma) = s_obs``, so
    ``d delta / d sigma = -(ds/d delta)^{-1} ds/d sigma`` with

        ds_j/d delta_k = (1/R) sum_r s_jr (1{j=k} - s_kr)
        ds_j/d sigma_l = (1/R) sum_r s_jr nu_rl (x_jl - sum_k s_kr x_kl).
    """
    N = len(delta)
    K = X_random.shape[1]
    R = mu.shape[1]
    D = np.zeros((N, K))
    for m in np.unique(market_ids):
        idx = np.where(market_ids == m)[0]
        v = delta[idx, np.newaxis] + mu[idx]
        v_max = np.maximum(v.max(axis=0), 0.0)
        exp_v = np.exp(v - v_max)
        s_ir = exp_v / (exp_v.sum(axis=0) + np.exp(-v_max))  # (J_m, R)
        ds_dd = np.diag(s_ir.mean(axis=1)) - (s_ir @ s_ir.T) / R
        Xm = X_random[idx]  # (J_m, K)
        ds_dsig = np.empty((len(idx), K))
        for c in range(K):
            xbar_r = Xm[:, c] @ s_ir  # (R,) sum_k s_kr x_kc
            ds_dsig[:, c] = (
                s_ir * draws[:, c][np.newaxis, :] * (Xm[:, [c]] - xbar_r[np.newaxis, :])
            ).mean(axis=1)
        D[idx] = -np.linalg.solve(ds_dd, ds_dsig)
    return D


def _compute_standard_errors(
    xi: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    sigma: np.ndarray,
    X_random: np.ndarray,
    delta: np.ndarray,
    mu: np.ndarray,
    s_obs: np.ndarray,
    market_ids: np.ndarray,
    draws: np.ndarray,
    tol_inner: float,
    maxiter_inner: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Heteroskedasticity-robust GMM standard errors for ``(beta, sigma)`` jointly.

    Moments ``g(theta) = Z' xi(theta) / N`` with ``xi = delta(sigma) - X beta``.
    Jacobian ``G = [-Z'X, Z' d delta/d sigma] / N`` (``d delta / d sigma`` from
    the implicit function theorem), ``S = sum_i xi_i^2 z_i z_i' / N`` and

        Var(theta) = (G'WG)^{-1} G'W S W G (G'WG)^{-1} / N,

    the covariance pyblp reports with ``se_type="robust"`` and the weighting
    matrix ``W`` of the final GMM step. The linear block therefore carries
    the estimation error of ``sigma``.

    A standard deviation estimated at its bound of zero is held fixed: the
    utility deviation is odd in the draws, so ``d delta / d sigma`` vanishes
    there up to simulation error and the sandwich would divide by it. Its
    standard error is NaN and the others are those of the model without
    that random coefficient.

    Returns
    -------
    se_linear : np.ndarray
    se_nonlinear : np.ndarray
    """
    N = len(xi)
    K_beta = X.shape[1]
    K_sigma = len(sigma)

    G_beta = -(Z.T @ X) / N
    free = np.asarray(sigma) > 0.0
    if free.any():
        D = _delta_jacobian(delta, mu, X_random, draws, market_ids)
        G = np.column_stack([G_beta, (Z.T @ D[:, free]) / N])
    else:
        G = G_beta

    Zxi = Z * xi[:, np.newaxis]
    S = Zxi.T @ Zxi / N
    bread = G.T @ W @ G
    try:
        bread_inv = np.linalg.inv(bread)
    except np.linalg.LinAlgError:
        bread_inv = np.linalg.pinv(bread)
    V = bread_inv @ (G.T @ W @ S @ W @ G) @ bread_inv / N
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    se_sigma = np.full(K_sigma, np.nan)
    se_sigma[free] = se[K_beta:]
    return se[:K_beta], se_sigma


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------
