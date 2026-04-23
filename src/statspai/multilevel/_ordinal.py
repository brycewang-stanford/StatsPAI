"""
Random-effects ordinal logit (``meologit``).

Model
-----
For ordinal outcome ``y_ij ∈ {1, ..., K}`` (K ≥ 3), latent threshold
formulation:

    P(y_ij ≤ k | u_j) = F(κ_k − x_ij' β − z_ij' u_j),  k = 1, ..., K-1
    P(y_ij = k | u_j) = F(κ_k − η_ij) − F(κ_{k-1} − η_ij)

with κ_0 = -∞, κ_K = +∞, F = standard logistic CDF, and random effects
``u_j ~ N(0, G)``.  No intercept enters β — its identification is
absorbed by the threshold parameters.

Estimation
----------
Laplace approximation (default) and adaptive Gauss-Hermite quadrature
(``nAGQ > 1``, q = 1 only).  Inner Newton uses the Fisher (expected)
information per observation

    W_i(η) = Σ_k [f(κ_{k-1} - η) - f(κ_k - η)]² / P(y_i = k | η),

where ``f(t) = F(t)(1 - F(t))`` is the logistic pdf.  This is positive
semi-definite for any η, unlike the observed information which can lose
definiteness — the same reason GLMs use Fisher scoring.

Threshold parameters are reparameterised as

    κ_1 = δ_1,         κ_k = δ_1 + Σ_{j=2..k} exp(δ_j)   (k ≥ 2)

so the optimiser sees an unconstrained vector and the strict ordering
``κ_1 < κ_2 < ... < κ_{K-1}`` is enforced automatically.

References
----------
McCullagh, P. (1980).  Regression Models for Ordinal Data.
Hartzel, J., Agresti, A., & Caffo, B. (2001).  Multinomial logit RE models.
Hedeker & Gibbons (1996).  MIXOR. [@mccullagh1980regression]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import warnings

import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import minimize

from ._core import (
    _GroupBlock,
    _initial_theta,
    _n_cov_params,
    _prepare_frame,
    _unpack_G,
)
from .glmm import MEGLMResult, _gh_nodes


_LOG_2PI = float(np.log(2.0 * np.pi))
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Threshold parameterisation
# ---------------------------------------------------------------------------


def _unpack_thresholds(delta: np.ndarray) -> np.ndarray:
    """Map unconstrained δ ∈ R^{K-1} to ordered κ_1 < ... < κ_{K-1}."""
    if delta.size == 1:
        return delta.copy()
    kappa = np.empty_like(delta)
    kappa[0] = delta[0]
    kappa[1:] = delta[0] + np.cumsum(np.exp(delta[1:]))
    return kappa


def _initial_delta(y_codes: np.ndarray, K: int) -> np.ndarray:
    """Starting thresholds from marginal cumulative empirical logits."""
    delta = np.zeros(K - 1)
    n = max(len(y_codes), 1)
    cum = 0
    cuts: List[float] = []
    for k in range(1, K):
        cum += int(np.sum(y_codes == k))
        p = (cum + 0.5) / (n + 1.0)
        cuts.append(float(np.log(p / (1.0 - p))))
    delta[0] = cuts[0]
    for j in range(1, K - 1):
        gap = max(cuts[j] - cuts[j - 1], 1e-3)
        delta[j] = float(np.log(gap))
    return delta


# ---------------------------------------------------------------------------
# Per-observation per-group ordinal log-lik primitives
# ---------------------------------------------------------------------------


def _logistic_cdf(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic CDF."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _ordinal_pieces(
    y_codes: np.ndarray,
    eta: np.ndarray,
    kappa: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-observation pieces required for log-likelihood + Newton step.

    Parameters
    ----------
    y_codes
        Observed category in 1..K.
    eta
        Linear predictor without thresholds.
    kappa
        Ordered thresholds κ_1 < ... < κ_{K-1}.

    Returns
    -------
    p_obs : (n,)              probability of the observed category
    score_eta : (n,)          ∂log p / ∂η_i
    fisher_w : (n,)           Fisher information per obs on η
    log_p_obs : (n,)          log p_obs (clipped for numerical safety)
    """
    n = eta.shape[0]
    K = kappa.shape[0] + 1

    # Compute upper / lower CDF arguments for the observed categories.
    # κ_0 = -inf, κ_K = +inf.
    upper_kappa = np.where(y_codes == K, np.inf, kappa[np.minimum(y_codes - 1, K - 2)])
    lower_kappa = np.where(y_codes == 1, -np.inf, kappa[np.maximum(y_codes - 2, 0)])

    a = upper_kappa - eta  # for k < K
    b = lower_kappa - eta  # for k > 1

    F_a = np.where(np.isinf(a) & (a > 0), 1.0, _logistic_cdf(np.where(np.isinf(a), 0.0, a)))
    F_b = np.where(np.isinf(b) & (b < 0), 0.0, _logistic_cdf(np.where(np.isinf(b), 0.0, b)))

    p_obs = np.clip(F_a - F_b, _EPS, 1.0)

    # f(t) = F(t)(1 - F(t)).  Treat ±inf with f → 0.
    f_a = np.where(np.isinf(a), 0.0, F_a * (1.0 - F_a))
    f_b = np.where(np.isinf(b), 0.0, F_b * (1.0 - F_b))

    score_eta = (f_b - f_a) / p_obs

    # Fisher information per obs.  Sum over ALL categories k = 1..K, not
    # only the observed one — that's the expected info under the model.
    # Need full per-obs cumulative CDFs at the K-1 thresholds.
    # Vectorised: F_k = F(κ_k - η_i) shape (n, K-1).
    eta_col = eta[:, None]
    cdf_args = kappa[None, :] - eta_col           # (n, K-1)
    Fk = _logistic_cdf(cdf_args)                  # (n, K-1)
    fk = Fk * (1.0 - Fk)                          # (n, K-1)
    # P_k for k=1..K, and Δf_k = f_{k-1}(t) - f_k(t).
    # Build (n, K) probabilities with sentinel 0/1 at the boundaries.
    F_full = np.concatenate(
        [np.zeros((n, 1)), Fk, np.ones((n, 1))], axis=1
    )                                              # (n, K+1) where col0=0, colK=1
    # P_k = F_full[:, k] - F_full[:, k-1] for k=1..K (1-indexed), so columns 1..K
    Pk_all = np.diff(F_full, axis=1)               # (n, K)
    Pk_all = np.clip(Pk_all, _EPS, 1.0)

    # f columns boundary-padded: f_0 = 0, f_K = 0
    fk_full = np.concatenate(
        [np.zeros((n, 1)), fk, np.zeros((n, 1))], axis=1
    )                                              # (n, K+1)
    df_k = fk_full[:, :-1] - fk_full[:, 1:]        # (n, K) is f_{k-1}-f_k
    fisher_w = np.sum(df_k ** 2 / Pk_all, axis=1)

    log_p_obs = np.log(p_obs)
    return p_obs, score_eta, fisher_w, log_p_obs


def _ordinal_log_lik(
    y_codes: np.ndarray, eta: np.ndarray, kappa: np.ndarray
) -> float:
    _, _, _, log_p = _ordinal_pieces(y_codes, eta, kappa)
    return float(np.sum(log_p))


# ---------------------------------------------------------------------------
# Inner mode-finder for u_j — Newton with Fisher-scoring weight
# ---------------------------------------------------------------------------


def _ordinal_find_mode(
    block: _GroupBlock,
    y_codes: np.ndarray,
    beta: np.ndarray,
    kappa: np.ndarray,
    G: np.ndarray,
    Ginv: np.ndarray,
    offset: np.ndarray,
    u0: np.ndarray,
    max_inner: int = 50,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    u = u0.copy()
    converged = False
    for _ in range(max_inner):
        eta = block.X @ beta + block.Z @ u + offset
        _, score_eta, fisher_w, _ = _ordinal_pieces(y_codes, eta, kappa)
        grad = block.Z.T @ score_eta - Ginv @ u
        H = block.Z.T @ (fisher_w[:, None] * block.Z) + Ginv
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        step_norm = float(np.linalg.norm(step))
        u_norm = float(np.linalg.norm(u)) + 1e-12
        if step_norm > 5.0 * max(u_norm, 1.0):
            step = step * (5.0 * max(u_norm, 1.0) / step_norm)
            step_norm = float(np.linalg.norm(step))
        u_new = u + step
        if step_norm < tol * (1 + u_norm):
            u = u_new
            converged = True
            break
        u = u_new

    eta = block.X @ beta + block.Z @ u + offset
    _, _, fisher_w, _ = _ordinal_pieces(y_codes, eta, kappa)
    H = block.Z.T @ (fisher_w[:, None] * block.Z) + Ginv
    sign, logdet_H = np.linalg.slogdet(H)
    if sign <= 0:
        logdet_H = np.inf
        converged = False
    return u, H, logdet_H, converged


# ---------------------------------------------------------------------------
# Outer NLL (Laplace + AGHQ)
# ---------------------------------------------------------------------------


def _ordinal_nll(
    theta: np.ndarray,
    blocks: List[_GroupBlock],
    y_codes_list: List[np.ndarray],
    offsets_list: List[np.ndarray],
    p_fixed: int,
    q_random: int,
    K: int,
    cov_type: str,
    u_cache: List[np.ndarray],
    nAGQ: int,
    gh_nodes: Optional[np.ndarray],
    gh_log_weights: Optional[np.ndarray],
):
    n_thr = K - 1
    beta = theta[:p_fixed]
    delta = theta[p_fixed : p_fixed + n_thr]
    cov_params = theta[p_fixed + n_thr :]
    kappa = _unpack_thresholds(delta)

    G = _unpack_G(cov_params, q_random, cov_type)
    try:
        sign, logdet_G = np.linalg.slogdet(G)
        if sign <= 0:
            return 1e12
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return 1e12

    use_aghq = nAGQ > 1
    sigma2 = float(G[0, 0]) if use_aghq else None

    nll = 0.0
    for j, block in enumerate(blocks):
        off = offsets_list[j]
        y_c = y_codes_list[j]
        u_hat, H_j, logdet_H, _ = _ordinal_find_mode(
            block, y_c, beta, kappa, G, Ginv, off, u_cache[j]
        )
        u_cache[j] = u_hat

        if use_aghq:
            sigma_hat = 1.0 / np.sqrt(max(float(H_j[0, 0]), _EPS))
            u_grid = float(u_hat[0]) + np.sqrt(2.0) * sigma_hat * gh_nodes
            log_lik_vals = np.empty(gh_nodes.shape[0])
            for k, u_k in enumerate(u_grid):
                eta_k = block.X @ beta + block.Z[:, 0] * u_k + off
                log_lik_vals[k] = _ordinal_log_lik(y_c, eta_k, kappa)
            log_prior = (
                -0.5 * (_LOG_2PI + np.log(max(sigma2, _EPS)))
                - 0.5 * u_grid ** 2 / max(sigma2, _EPS)
            )
            log_terms = (
                log_lik_vals
                + log_prior
                + gh_nodes ** 2
                + gh_log_weights
                + 0.5 * (np.log(2.0) + 2.0 * np.log(sigma_hat))
            )
            ll_j = float(special.logsumexp(log_terms))
        else:
            eta = block.X @ beta + block.Z @ u_hat + off
            ll_data = _ordinal_log_lik(y_c, eta, kappa)
            quad = float(u_hat @ Ginv @ u_hat)
            ll_j = ll_data - 0.5 * logdet_G - 0.5 * quad - 0.5 * logdet_H
        nll -= ll_j
    return nll


# ---------------------------------------------------------------------------
# GLM-only warm-start for β + thresholds
# ---------------------------------------------------------------------------


def _ordinal_glm_init(
    X: np.ndarray, y_codes: np.ndarray, K: int, off: np.ndarray,
    maxiter: int = 30, tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plain ordinal logit (no random effects) for warm start."""
    p = X.shape[1]
    beta = np.zeros(p)
    delta = _initial_delta(y_codes, K)
    n_thr = K - 1
    theta = np.concatenate([beta, delta])

    def _nll(th: np.ndarray) -> float:
        b = th[:p]
        d = th[p:]
        kap = _unpack_thresholds(d)
        eta = X @ b + off
        return -_ordinal_log_lik(y_codes, eta, kap)

    res = minimize(
        _nll, theta, method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
    )
    return res.x[:p], res.x[p:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def meologit(
    data: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    group: str,
    x_random: Optional[Sequence[str]] = None,
    cov_type: str = "unstructured",
    offset: Optional[str] = None,
    nAGQ: int = 1,
    maxiter: int = 300,
    tol: float = 1e-6,
    alpha: float = 0.05,
) -> MEGLMResult:
    """
    Random-effects ordinal logit (Stata ``meologit``, R ``ordinal::clmm``).

    The outcome ``y`` is treated as ordered categorical with K levels.
    If ``y`` is numeric it is coerced to integer codes 1..K based on
    sorted unique values.  Other dtypes (str, pandas categorical) are
    coerced via ``pd.Categorical`` and ordered by appearance.

    No intercept enters β — its role is taken by the K-1 thresholds.
    Returns an :class:`MEGLMResult` with ``family='ordinal'`` and
    ``thresholds`` populated.
    """
    if cov_type not in ("unstructured", "diagonal", "identity"):
        raise ValueError(f"unknown cov_type {cov_type!r}")
    if isinstance(group, (list, tuple)):
        if len(group) != 1:
            raise ValueError(
                "meologit() supports a single grouping variable; "
                "collapse nested levels into one key first."
            )
        group = group[0]
    if not isinstance(group, str):
        raise TypeError("`group` must be a column name string")

    nAGQ = int(nAGQ)
    if nAGQ < 1:
        raise ValueError(f"nAGQ must be >= 1, got {nAGQ}")
    x_fixed = list(x_fixed)
    x_random_cols: List[str] = list(x_random) if x_random is not None else []
    if nAGQ > 1 and len(x_random_cols) > 0:
        raise ValueError(
            "AGHQ (nAGQ > 1) currently supports only random-intercept models "
            "(empty x_random)."
        )

    extra_cols = [offset] if offset else []
    df = _prepare_frame(data, y, x_fixed + extra_cols, [group], x_random_cols)

    # Encode y as 1..K integer codes preserving sort order.
    y_raw = df[y]
    if isinstance(y_raw.dtype, pd.CategoricalDtype):
        levels = list(y_raw.cat.categories)
    else:
        levels = sorted(y_raw.dropna().unique().tolist())
    K = len(levels)
    if K < 3:
        raise ValueError(
            f"meologit() requires at least 3 outcome categories; got {K}."
        )
    code_map = {lvl: k + 1 for k, lvl in enumerate(levels)}
    df = df.copy()
    df[y] = df[y].map(code_map).astype(int)

    # Build per-group blocks.  meologit has NO intercept in β: drop it from X.
    p_fixed = len(x_fixed)              # no intercept
    q_random = 1 + len(x_random_cols)
    n_cov_pars = _n_cov_params(q_random, cov_type)

    blocks: List[_GroupBlock] = []
    y_codes_list: List[np.ndarray] = []
    offsets_list: List[np.ndarray] = []
    fixed_names = list(x_fixed)
    random_names = ["_cons"] + list(x_random_cols)

    positions = np.arange(len(df))
    for key, sub in df.groupby(group, sort=False):
        idx = positions[df.index.get_indexer(sub.index)]
        y_c = sub[y].to_numpy(dtype=int)
        if x_fixed:
            X_j = sub[list(x_fixed)].to_numpy(dtype=float)
        else:
            X_j = np.zeros((len(sub), 0))
        if x_random_cols:
            Z_j = sub[["__intercept__"] + list(x_random_cols)].to_numpy(dtype=float)
        else:
            Z_j = sub[["__intercept__"]].to_numpy(dtype=float)
        blocks.append(_GroupBlock(key=key, y=y_c.astype(float),
                                   X=X_j, Z=Z_j, n=len(sub), row_idx=idx))
        y_codes_list.append(y_c)
        if offset:
            offsets_list.append(sub[offset].to_numpy(dtype=float))
        else:
            offsets_list.append(np.zeros(len(sub)))

    # Warm-start (β, δ) via plain ordinal logit.
    X_all = np.vstack([b.X for b in blocks]) if p_fixed else np.zeros((len(df), 0))
    y_all = np.concatenate(y_codes_list)
    off_all = np.concatenate(offsets_list)
    beta0, delta0 = _ordinal_glm_init(X_all, y_all, K, off_all)

    theta_cov0 = _initial_theta(q_random, cov_type, s2_init=0.3)
    theta0 = np.concatenate([beta0, delta0, theta_cov0])

    u_cache = [np.zeros(q_random) for _ in blocks]
    gh_nodes, gh_log_weights = (None, None)
    if nAGQ > 1:
        nodes, weights = _gh_nodes(nAGQ)
        gh_nodes = nodes
        gh_log_weights = np.log(weights)

    res = minimize(
        _ordinal_nll, theta0,
        args=(
            blocks, y_codes_list, offsets_list,
            p_fixed, q_random, K, cov_type, u_cache,
            nAGQ, gh_nodes, gh_log_weights,
        ),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
    )
    outer_converged = bool(res.success)

    n_thr = K - 1
    beta_hat = res.x[:p_fixed]
    delta_hat = res.x[p_fixed : p_fixed + n_thr]
    cov_hat = res.x[p_fixed + n_thr :]
    kappa_hat = _unpack_thresholds(delta_hat)
    G_hat = _unpack_G(cov_hat, q_random, cov_type)
    Ginv = np.linalg.inv(G_hat)

    # Final pass: BLUPs + observed information for fixed-effect SEs.
    blup_rows: List[Dict[str, float]] = []
    blup_dict: Dict[Any, np.ndarray] = {}
    keys: List[Any] = []
    info = np.zeros((p_fixed, p_fixed)) if p_fixed else np.zeros((0, 0))
    inner_failures = 0
    for j, (block, y_c, off) in enumerate(zip(blocks, y_codes_list, offsets_list)):
        u_hat, H_j, _, inner_ok = _ordinal_find_mode(
            block, y_c, beta_hat, kappa_hat, G_hat, Ginv, off, u_cache[j]
        )
        if not inner_ok:
            inner_failures += 1
        eta = block.X @ beta_hat + block.Z @ u_hat + off
        _, _, fisher_w, _ = _ordinal_pieces(y_c, eta, kappa_hat)
        if p_fixed:
            XtWX = block.X.T @ (fisher_w[:, None] * block.X)
            XtWZ = block.X.T @ (fisher_w[:, None] * block.Z)
            ZtWX = block.Z.T @ (fisher_w[:, None] * block.X)
            info += XtWX - XtWZ @ np.linalg.solve(H_j, ZtWX)

        blup_dict[block.key] = u_hat
        blup_rows.append(dict(zip(random_names, u_hat)))
        keys.append(block.key)

    if p_fixed:
        try:
            cov_beta = np.linalg.inv(info)
            se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
        except np.linalg.LinAlgError:
            cov_beta = np.full((p_fixed, p_fixed), np.nan)
            se_beta = np.full(p_fixed, np.nan)
    else:
        cov_beta = np.zeros((0, 0))
        se_beta = np.zeros(0)

    if inner_failures > 0:
        warnings.warn(
            f"meologit inner Newton failed to converge for "
            f"{inner_failures}/{len(blocks)} clusters.",
            RuntimeWarning,
            stacklevel=2,
        )

    random_effects_df = pd.DataFrame(blup_rows, index=keys)
    random_effects_df.index.name = group

    vc: Dict[str, float] = {}
    for i, name in enumerate(random_names):
        vc[f"var({name})"] = float(G_hat[i, i])
    if cov_type == "unstructured" and q_random >= 2:
        for i in range(q_random):
            for j in range(i):
                denom = np.sqrt(G_hat[i, i] * G_hat[j, j])
                corr = G_hat[i, j] / denom if denom > 0 else np.nan
                vc[f"cov({random_names[j]},{random_names[i]})"] = float(G_hat[i, j])
                vc[f"corr({random_names[j]},{random_names[i]})"] = float(corr)

    threshold_names = [f"cut{k+1}|{levels[k]}/{levels[k+1]}" for k in range(K - 1)]
    method = "laplace" if nAGQ == 1 else f"AGHQ(nAGQ={nAGQ})"

    return MEGLMResult(
        fixed_effects=pd.Series(beta_hat, index=fixed_names) if p_fixed
                        else pd.Series(dtype=float),
        random_effects=random_effects_df,
        variance_components=vc,
        blups=blup_dict,
        n_obs=len(df),
        n_groups=len(blocks),
        log_likelihood=float(-res.fun),
        family="ordinal",
        link="logit",
        _se_fixed=pd.Series(se_beta, index=fixed_names) if p_fixed
                    else pd.Series(dtype=float),
        _cov_fixed=cov_beta,
        _G=G_hat,
        _x_fixed=x_fixed,
        _x_random=x_random_cols,
        _group_col=group,
        _fixed_names=fixed_names,
        _random_names=random_names,
        _y_name=y,
        _converged=outer_converged and inner_failures == 0,
        _method=method,
        _cov_type=cov_type,
        _alpha=alpha,
        _n_cov_params=n_cov_pars,
        _offset_name=offset,
        thresholds=pd.Series(kappa_hat, index=threshold_names),
    )


__all__ = ["meologit"]
