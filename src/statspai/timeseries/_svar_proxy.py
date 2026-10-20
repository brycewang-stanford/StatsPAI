"""Proxy (external-instrument) identification of one structural shock.

A proxy ``m_t`` correlated with the target shock and uncorrelated with the
other shocks identifies the shock's impact column up to scale:
``b ∝ Cov(u_t, m_t)`` for the reduced-form residuals ``u_t``. With the
``target`` variable normalised to a unit impact, ``b_i`` is the two-stage
least squares coefficient of ``u_i`` on ``u_target`` with instrument ``m``
(and a constant), estimated on the periods where the proxy is observed.
Scaled to a one-standard-deviation shock it is
``b / sqrt(b' Sigma^{-1} b)``, ``Sigma`` the residual covariance of the
whole sample; the shock series is ``b' Sigma^{-1} u_t / sqrt(b' Sigma^{-1} b)``.

Inference is a residual-based moving block bootstrap: blocks of
consecutive (residual, proxy) pairs are drawn jointly, centred by block
position, the VAR regenerated recursively from the estimated coefficients,
re-estimated, and the shock re-identified on every replicate. The
residual-based wild bootstrap of Mertens and Ravn (one Rademacher sign per
period on the residual and the proxy) is not used: the sign cancels in
the product ``u_t m_t``, so the covariance that identifies the shock never
varies across replicates and the bands are far too narrow
[@mertens2013dynamic; @jentsch2019dynamic]. On the simulated design of
``tests/reference_parity/test_ts_causal_parity.py`` its nominal 95% bands
covered the true impact response in 17-19% of samples; the block
bootstrap's cover it at about the nominal rate (same test file).

Proxy strength is reported as the heteroskedasticity-robust F statistic of
the proxy in the regression of ``u_target`` on a constant and ``m``; below
10 a warning is raised, because the impact ratio is then weakly identified
and the Wald-type and bootstrap bands both undercover [@olea2021inference].
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import (
    AssumptionWarning,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
)

PROXY_SCALES = ("unit", "std")


def align_proxy(var_result: Any, proxy: Any) -> np.ndarray:
    """The proxy as one value per VAR residual (NaN where unobserved).

    Accepts a sequence of the length of the residuals, or of the length of
    the data the VAR was fitted on (its first ``p`` values are dropped).
    """
    T = int(var_result.n_obs)
    n_levels = len(var_result._levels) if var_result._levels is not None else T
    m = np.asarray(
        proxy.to_numpy(dtype=float) if isinstance(proxy, pd.Series) else proxy,
        dtype=float,
    ).ravel()
    if m.size == T:
        return m
    if m.size == n_levels:
        return m[n_levels - T :]
    raise MethodIncompatibility(
        f"svar: the proxy has {m.size} values; the VAR has {n_levels} "
        f"observations and {T} residuals.",
        recovery_hint="Pass the proxy as a column of the data given to sp.var "
        "(NaN where it is not observed).",
    )


def robust_f(u_target: np.ndarray, m: np.ndarray) -> float:
    """HC0 F (= t^2) of the proxy in ``u_target ~ 1 + m``."""
    X = np.column_stack([np.ones_like(m), m])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ u_target)
    e = u_target - X @ beta
    s = X * e[:, None]
    V = xtx_inv @ (s.T @ s) @ xtx_inv
    return float(beta[1] ** 2 / V[1, 1])


def proxy_impact(
    resid: np.ndarray,
    m: np.ndarray,
    sigma: np.ndarray,
    target: int,
    scale: str,
) -> np.ndarray:
    """Impact column of the shock instrumented by ``m``."""
    obs = np.isfinite(m)
    u = resid[obs]
    mm = m[obs] - m[obs].mean()
    gamma = (u - u.mean(axis=0)).T @ mm
    if abs(gamma[target]) <= 1e-12 * np.sqrt(
        (u[:, target] - u[:, target].mean())
        @ (u[:, target] - u[:, target].mean())
        * (mm @ mm)
    ):
        raise IdentificationFailure(
            "svar: the proxy is uncorrelated with the target variable's "
            "residual, so it identifies no shock.",
            recovery_hint="Use a proxy that moves the target variable on "
            "impact, or a different target=.",
        )
    b = gamma / gamma[target]
    if scale == "std":
        b = b / np.sqrt(b @ np.linalg.solve(sigma, b))
    return b


def default_block_length(T: int) -> int:
    """``ceil(T^(1/3))``, at most ``T``."""
    return int(min(T, max(1, int(np.ceil(T ** (1.0 / 3.0) - 1e-12)))))


def block_draws(
    var_result: Any,
    m: np.ndarray,
    reps: int,
    seed: Optional[int],
    block_length: Optional[int] = None,
) -> Iterator[Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]]:
    """Moving-block bootstrap replicates: lag matrices, sigma, residuals,
    proxy.

    Blocks of ``block_length`` consecutive periods of the residuals and the
    proxy are drawn together (start uniform on ``0..T-l``) and laid end to
    end, cut at ``T``. Each element is centred by the mean over all blocks
    of its position in the block, so the bootstrap residuals have mean zero
    at every date; the proxy enters a regression with a constant, where the
    same centring leaves its covariance with the residuals intact. NaN
    (unobserved proxy) travel with their period.
    """
    B = np.asarray(var_result._B, float)
    X = np.asarray(var_result._X, float)
    levels = np.asarray(var_result._levels, float)
    k, p = int(var_result._k), int(var_result._lags)
    kp = k * p
    T = X.shape[0]
    # validated in proxy_svar
    ell = default_block_length(T) if block_length is None else int(block_length)
    resid = np.asarray(var_result.residuals, float)
    det = X[:, kp:] @ B[kp:, :] if X.shape[1] > kp else np.zeros((T, k))
    A_stack = B[:kp, :]
    start = levels[len(levels) - T - p : len(levels) - T]
    n_start = T - ell + 1
    pos = np.arange(ell)
    # mean of position j over all admissible blocks
    idx_all = np.arange(n_start)[:, None] + pos[None, :]  # (n_start, ell)
    mu_u = resid[idx_all].mean(axis=0)  # (ell, k)
    with np.errstate(invalid="ignore"):
        m_blocks = m[idx_all]
        seen = np.isfinite(m_blocks)
        mu_m = np.where(
            seen.any(axis=0),
            np.where(seen, m_blocks, 0.0).sum(axis=0) / np.maximum(seen.sum(axis=0), 1),
            0.0,
        )
    n_blocks = -(-T // ell)
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        starts = rng.integers(0, n_start, size=n_blocks)
        idx = (starts[:, None] + pos[None, :]).ravel()[:T]
        j = np.tile(pos, n_blocks)[:T]
        e = resid[idx] - mu_u[j]
        m_b = m[idx] - mu_m[j]
        y = np.empty((T + p, k))
        y[:p] = start
        for t in range(T):
            y[t + p] = y[t : t + p][::-1].ravel() @ A_stack + det[t] + e[t]
        Xb = X.copy()
        for lag in range(1, p + 1):
            Xb[:, (lag - 1) * k : lag * k] = y[p - lag : p - lag + T]
        Yb = y[p:]
        Bb = np.linalg.lstsq(Xb, Yb, rcond=None)[0]
        u = Yb - Xb @ Bb
        lag_b = [Bb[jj * k : (jj + 1) * k, :].T for jj in range(p)]
        yield lag_b, u.T @ u / T, u, m_b


def proxy_svar(
    var_result: Any,
    proxy: Any,
    target: Optional[str],
    scale: str,
    alpha: float,
    shock_name: Optional[str] = None,
    block_length: Optional[int] = None,
):
    """Identify the shock; returns an ``SVARResult`` with one shock."""
    from .svar import SVARResult, _lag_matrices

    names = list(var_result.var_names)
    if target is None:
        target = names[0]
    if target not in names:
        raise MethodIncompatibility(
            f"svar: proxy_target={target!r} is not a VAR variable {names}.",
            recovery_hint="Name the variable whose impact response is "
            "normalised to one (usually the policy variable).",
        )
    if scale not in PROXY_SCALES:
        raise MethodIncompatibility(
            f"svar: proxy_scale={scale!r} is not one of {list(PROXY_SCALES)}.",
            recovery_hint="'unit' fixes the target's impact response at 1; "
            "'std' gives a one-standard-deviation shock.",
        )
    m = align_proxy(var_result, proxy)
    obs = np.isfinite(m)
    k = len(names)
    if obs.sum() <= k + 2:
        raise DataInsufficient(
            f"svar: the proxy is observed in {int(obs.sum())} periods.",
            recovery_hint="A proxy needs to overlap the VAR sample in many " "periods.",
        )
    resid = np.asarray(var_result.residuals, float)
    sigma = np.asarray(var_result.sigma_u, float)
    ti = names.index(target)
    b = proxy_impact(resid, m, sigma, ti, scale)
    u_t = resid[obs, ti]
    f_stat = robust_f(u_t, m[obs])
    corr = float(np.corrcoef(u_t, m[obs])[0, 1])
    if f_stat < 10:
        warnings.warn(
            f"svar: the proxy is weak (robust F = {f_stat:.2f} < 10 in the "
            f"regression of the {target} residual on it); the identified "
            "impact ratios are unreliable and their bootstrap bands "
            "undercover.",
            AssumptionWarning,
            stacklevel=3,
        )
    shock_series = resid @ np.linalg.solve(sigma, b)
    shock_series = shock_series / np.sqrt(b @ np.linalg.solve(sigma, b))
    label = shock_name or f"{target}_shock"
    result = SVARResult(
        identification="proxy",
        var_names=names,
        shock_names=[label],
        impact=b.reshape(k, 1),
        lag_matrices=_lag_matrices(var_result),
        n_obs=int(var_result.n_obs),
        alpha=float(alpha),
    )
    result._var_result = var_result
    result._sigma = sigma
    result.proxy_diagnostics = {
        "target": target,
        "scale": scale,
        "n_proxy": int(obs.sum()),
        "robust_f": f_stat,
        "corr_target_residual": corr,
        "weak": bool(f_stat < 10),
    }
    result.shock_series = pd.Series(shock_series, name=label)

    def identify(lag_b, sigma_b, u_b, m_b) -> np.ndarray:
        return proxy_impact(u_b, m_b, sigma_b, ti, scale).reshape(k, 1)

    result._proxy = m
    result._identify_proxy = identify
    T = len(resid)
    if block_length is not None and not (
        isinstance(block_length, (int, np.integer)) and 1 <= int(block_length) <= T
    ):
        raise MethodIncompatibility(
            f"svar: proxy_block_length={block_length!r} must be an integer "
            f"between 1 and the {T} VAR residuals.",
            recovery_hint="Leave it at None for ceil(T ** (1/3)).",
        )
    result._proxy_block_length = None if block_length is None else int(block_length)
    result.proxy_diagnostics["block_length"] = (
        default_block_length(T) if block_length is None else int(block_length)
    )
    return result


def proxy_bootstrap(
    result: Any, path: Any, reps: int, seed: Optional[int]
) -> Tuple[np.ndarray, int]:
    """Stack of ``path(lag_matrices, impact)`` over block-bootstrap draws."""
    draws: List[np.ndarray] = []
    failed = 0
    for lag_b, sigma_b, u_b, m_b in block_draws(
        result._var_result, result._proxy, reps, seed, result._proxy_block_length
    ):
        try:
            impact_b = result._identify_proxy(lag_b, sigma_b, u_b, m_b)
        except (IdentificationFailure, np.linalg.LinAlgError):
            failed += 1
            continue
        draws.append(path(lag_b, impact_b))
    if len(draws) < 20:
        raise IdentificationFailure(
            "svar: the proxy identification failed on almost every bootstrap "
            "sample, so no band can be formed.",
            recovery_hint="The proxy is too weak for this sample.",
        )
    return np.stack(draws), failed


def proxy_fevd(theta: np.ndarray, lag_matrices: List[np.ndarray], sigma: np.ndarray):
    """Share of the ``h``-step forecast-error variance due to the shock.

    The denominator is the full reduced-form forecast-error variance
    ``sum_s Phi_s Sigma Phi_s'``, since the other shocks are not identified.
    """
    from .svar import ma_coefficients

    periods = theta.shape[0] - 1
    phi = ma_coefficients(lag_matrices, periods)
    mse = np.cumsum(np.einsum("sij,jk,slk->sil", phi, sigma, phi), axis=0)
    total = np.diagonal(mse, axis1=1, axis2=2)  # (periods+1, K)
    cum = np.cumsum(theta**2, axis=0)  # (periods+1, K, 1)
    out = np.zeros_like(theta)
    out[1:] = cum[:-1] / total[:-1, :, None]
    return out


def describe(result: Any) -> Dict[str, Any]:
    return dict(getattr(result, "proxy_diagnostics", {}))
