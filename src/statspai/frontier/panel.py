"""
Panel stochastic frontier estimation — :func:`xtfrontier`.

Implements three panel SFA designs:

* ``model='ti'``  — Pitt-Lee (1981) time-invariant: ``u_it = u_i``.
  With ``dist='half-normal'`` u_i ~ N^+(0, sigma_u^2); with
  ``dist='truncated-normal'`` u_i ~ N^+(mu, sigma_u^2).

* ``model='tvd'`` — Battese-Coelli (1992) time-varying decay:
  ``u_it = exp(-eta (t - T_i)) * u_i``  with u_i ~ N^+(mu, sigma_u^2).
  ``eta > 0`` means inefficiency declines over time; ``eta < 0`` rises.
  ``eta = 0`` collapses to Pitt-Lee TI.

* ``model='bc95'`` — Battese-Coelli (1995) inefficiency effects:
  u_it ~ N^+(z_it' delta, sigma_u^2), independent across (i, t).  The
  ``z`` covariates shift the mean of the truncated normal.  This is
  equivalent to the cross-sectional :func:`frontier` call with
  ``dist='truncated-normal'`` and ``emean=z``, but returns a panel-aware
  result with per-unit efficiency aggregation.

Equivalent to Stata's::

    xtfrontier y x, ti
    xtfrontier y x, ti dist(tnormal)
    xtfrontier y x, tvd
    frontier y x, dist(tnormal) emean(z)      (BC95)

and R's ``frontier::sfa`` with panel data, or ``sfaR::sfacross`` /
``sfaR::sfapanel``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from . import _core as _fc
from .sfa import FrontierResult, frontier as _cs_frontier


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def xtfrontier(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id: str,
    time: Optional[str] = None,
    *,
    model: str = "ti",
    dist: str = "half-normal",
    cost: bool = False,
    emean: Optional[List[str]] = None,
    vce: str = "oim",
    cluster: Optional[str] = None,
    maxiter: int = 500,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> FrontierResult:
    """Panel stochastic frontier estimator.

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
    x : list of str
    id : str
        Panel unit identifier.
    time : str, optional
        Time variable (required for ``model='tvd'``).
    model : {'ti', 'tvd', 'bc95'}
        ``'ti'``  Pitt-Lee (1981) time-invariant.
        ``'tvd'`` Battese-Coelli (1992) time-varying decay.
        ``'bc95'`` Battese-Coelli (1995) inefficiency effects model.
    dist : {'half-normal', 'truncated-normal'}
        For ``ti`` and ``tvd``.  BC95 always uses truncated-normal.
    cost : bool, default False
    emean : list of str, optional
        Required for ``model='bc95'``; inefficiency determinants ``z_it``.
    maxiter, tol, alpha : see :func:`frontier`.

    Returns
    -------
    :class:`~statspai.frontier.FrontierResult`
    """
    model = model.lower()
    dist = dist.lower().replace("_", "-")

    if model not in {"ti", "tvd", "bc95"}:
        raise ValueError(f"Unknown panel model: {model!r}.")
    if model == "tvd" and time is None:
        raise ValueError("model='tvd' requires a time variable.")
    if model == "bc95":
        if emean is None:
            raise ValueError("model='bc95' requires emean=[...].")
        # Default BC95 cluster is the panel id (standard for applied papers).
        cl = cluster if cluster is not None else (id if vce != "oim" else None)
        return _fit_bc95(
            data, y, x, id_col=id, time_col=time, emean=emean,
            cost=cost, maxiter=maxiter, tol=tol, alpha=alpha,
            vce=vce, cluster=cl,
        )
    if dist not in {"half-normal", "truncated-normal"}:
        raise ValueError(f"dist={dist!r} not supported for panel model.")

    return _fit_ti_tvd(
        data, y, x,
        id_col=id, time_col=time,
        model=model, dist=dist, cost=cost,
        vce=vce, cluster=cluster,
        maxiter=maxiter, tol=tol, alpha=alpha,
    )


# ---------------------------------------------------------------------------
# BC95 via cross-sectional TN + emean (u_it independent across t)
# ---------------------------------------------------------------------------


def _fit_bc95(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id_col: str,
    time_col: Optional[str],
    emean: List[str],
    cost: bool,
    maxiter: int,
    tol: float,
    alpha: float,
    vce: str = "oim",
    cluster: Optional[str] = None,
) -> FrontierResult:
    """BC95: u_it ~ N^+(z_it' delta, sigma_u^2) independently.

    Estimated identically to cross-sectional truncated-normal with
    ``emean=z``; we then aggregate efficiency scores per panel unit.
    """
    res = _cs_frontier(
        data=data,
        y=y,
        x=x,
        dist="truncated-normal",
        cost=cost,
        emean=emean,
        vce=vce,
        cluster=cluster,
        maxiter=maxiter,
        tol=tol,
        alpha=alpha,
    )
    res.model_info["model_type"] = (
        f"Panel Stochastic Frontier (BC95, {'Cost' if cost else 'Production'})"
    )
    res.model_info["panel_model"] = "bc95"
    # Aggregate unit-level mean efficiency as a convenience.
    idx = res.diagnostics.get("efficiency_index")
    if idx is not None:
        unit_ids = data.loc[idx, id_col].to_numpy()
        te = res.diagnostics["efficiency_bc"]
        unit_te = pd.Series(te, index=unit_ids).groupby(level=0).mean()
        res.diagnostics["efficiency_bc_unit_mean"] = unit_te
    return res


# ---------------------------------------------------------------------------
# Pitt-Lee TI and Battese-Coelli TVD
# ---------------------------------------------------------------------------


def _fit_ti_tvd(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id_col: str,
    time_col: Optional[str],
    model: str,
    dist: str,
    cost: bool,
    maxiter: int,
    tol: float,
    alpha: float,
    vce: str = "oim",
    cluster: Optional[str] = None,
) -> FrontierResult:
    vce = vce.lower()
    if vce not in {"oim", "opg", "robust"}:
        raise ValueError(f"Unknown vce={vce!r}.")
    if cluster is not None and vce == "oim":
        vce = "robust"
    # Default cluster for panel: the panel unit id (groups are units).
    cluster_effective = cluster if cluster is not None else (
        id_col if vce != "oim" else None
    )
    sign = 1 if cost else -1
    has_mu = dist == "truncated-normal"
    has_eta = model == "tvd"

    # ---- Data prep ----
    required = [y] + list(x) + [id_col]
    if time_col is not None:
        required.append(time_col)
    df = data[required].dropna().copy()
    df = df.sort_values(
        [id_col] + ([time_col] if time_col is not None else [])
    ).reset_index(drop=True)

    y_vec, X_mat, beta_names = _fc.build_design(df, y, x, add_constant=True)
    group_idx, time_vec, counts, unique_ids = _fc.group_panel(
        df, id_col=id_col, time_col=time_col
    )
    N = len(unique_ids)
    n = len(df)
    k_beta = X_mat.shape[1]

    # Precompute within-group last period T_i for TVD (relative time = t - T_i).
    # For TVD a_it = exp(-eta*(t - T_i)).  If time unavailable, treat as sequence 1..T_i.
    if time_col is None:
        # Assign within-group rank (0, 1, ..., T_i-1); T_i = counts[i]-1 for last.
        rel_time = np.empty(n, dtype=float)
        for i in range(N):
            mask = group_idx == i
            Ti = int(mask.sum())
            rel_time[mask] = np.arange(Ti) - (Ti - 1)  # ranges (-(T_i-1), 0)
    else:
        # Use actual time minus last observed time per group.
        rel_time = np.empty(n, dtype=float)
        for i in range(N):
            mask = group_idx == i
            t_i = time_vec[mask]
            rel_time[mask] = t_i - t_i.max()

    # ---- Parameter layout: [beta, ln_sigma_v, ln_sigma_u, (mu), (eta)] ----
    k_total = k_beta + 2 + (1 if has_mu else 0) + (1 if has_eta else 0)
    idx_ln_sv = k_beta
    idx_ln_su = k_beta + 1
    idx_mu = k_beta + 2 if has_mu else None
    idx_eta = k_total - 1 if has_eta else None

    param_names = list(beta_names) + ["ln_sigma_v", "ln_sigma_u"]
    if has_mu:
        param_names.append("mu")
    if has_eta:
        param_names.append("eta")

    # ---- LL ----

    def compute_a(eta: float) -> np.ndarray:
        """a_it = exp(-eta * (t - T_i))."""
        if not has_eta:
            return np.ones(n)
        return np.exp(-eta * rel_time)

    def per_group_loglik(theta: np.ndarray) -> np.ndarray:
        """Return the length-N vector of group log-likelihoods."""
        beta = theta[:k_beta]
        sigma_v = float(np.exp(theta[idx_ln_sv]))
        sigma_u = float(np.exp(theta[idx_ln_su]))
        mu_scalar = float(theta[idx_mu]) if has_mu else 0.0
        eta = float(theta[idx_eta]) if has_eta else 0.0
        a_vec = compute_a(eta)
        eps = y_vec - X_mat @ beta
        w_sq = a_vec**2
        C_i = np.bincount(group_idx, weights=w_sq, minlength=N)
        A_i = np.bincount(group_idx, weights=a_vec * eps, minlength=N)
        norm_eps = np.bincount(group_idx, weights=eps**2, minlength=N)
        C_safe = np.where(C_i > 0, C_i, np.nan)
        eps_tilde = A_i / C_safe
        ssw_a = norm_eps - A_i**2 / C_safe
        T_i = counts
        denom = C_i * sigma_u**2 + sigma_v**2
        sigma_star2 = sigma_v**2 * sigma_u**2 / denom
        sigma_star = np.sqrt(sigma_star2)
        mu_star = (sign * sigma_u**2 * A_i + sigma_v**2 * mu_scalar) / denom
        if has_mu:
            term_eps = -C_i * (eps_tilde - sign * mu_scalar) ** 2 / (2.0 * denom)
            log_trunc = _fc._log_phi_cdf(mu_scalar / sigma_u)
        else:
            term_eps = -C_i * eps_tilde**2 / (2.0 * denom)
            log_trunc = -np.log(2.0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ll_group = (
                -T_i / 2.0 * np.log(2.0 * np.pi)
                - T_i * np.log(sigma_v)
                - ssw_a / (2.0 * sigma_v**2)
                + np.log(sigma_star)
                - np.log(sigma_u)
                - log_trunc
                + term_eps
                + _fc._log_phi_cdf(mu_star / sigma_star)
            )
        return ll_group

    def neg_loglik(theta: np.ndarray) -> float:
        if not np.all(np.isfinite(theta)):
            return 1e20
        beta = theta[:k_beta]
        sigma_v = float(np.exp(theta[idx_ln_sv]))
        sigma_u = float(np.exp(theta[idx_ln_su]))
        if sigma_v <= 1e-8 or sigma_u <= 1e-8 or sigma_v > 1e6 or sigma_u > 1e6:
            return 1e20
        mu_scalar = float(theta[idx_mu]) if has_mu else 0.0
        eta = float(theta[idx_eta]) if has_eta else 0.0
        a_vec = compute_a(eta)

        eps = y_vec - X_mat @ beta
        # Aggregates per group (using bincount for speed).
        # C_i = sum a_it^2 ; A_i = sum a_it * eps_it ; ||e_i||^2 = sum eps_it^2
        w_sq = a_vec**2
        C_i = np.bincount(group_idx, weights=w_sq, minlength=N)
        A_i = np.bincount(group_idx, weights=a_vec * eps, minlength=N)
        norm_eps = np.bincount(group_idx, weights=eps**2, minlength=N)

        # SSW_i^{(a)} = ||e_i||^2 - C_i * (A_i/C_i)^2 = ||e_i||^2 - A_i^2/C_i.
        # Protect C_i>0.
        C_safe = np.where(C_i > 0, C_i, np.nan)
        eps_tilde = A_i / C_safe
        ssw_a = norm_eps - A_i**2 / C_safe

        T_i = counts
        # sigma_star^2 = sigma_v^2 sigma_u^2 / (C_i sigma_u^2 + sigma_v^2)
        denom = C_i * sigma_u**2 + sigma_v**2
        sigma_star2 = sigma_v**2 * sigma_u**2 / denom
        sigma_star = np.sqrt(sigma_star2)
        mu_star = (sign * sigma_u**2 * A_i + sigma_v**2 * mu_scalar) / denom

        if has_mu:
            # truncated-normal prior on u_i
            # -log Phi(mu/sigma_u): normalization of truncation
            # Contribution of ε̃ quadratic:  -C_i (ε̃ - sign mu)^2 / (2 denom)
            term_eps = -C_i * (eps_tilde - sign * mu_scalar) ** 2 / (2.0 * denom)
            log_trunc = _fc._log_phi_cdf(mu_scalar / sigma_u)
        else:
            # Half-normal prior (mu=0): -log Phi(0) = log 2 → + log 2 in LL.
            term_eps = -C_i * eps_tilde**2 / (2.0 * denom)
            log_trunc = -np.log(2.0)  # so -log_trunc below adds +log 2

        ll_group = (
            -T_i / 2.0 * np.log(2.0 * np.pi)
            - T_i * np.log(sigma_v)
            - ssw_a / (2.0 * sigma_v**2)
            + np.log(sigma_star)
            - np.log(sigma_u)
            - log_trunc
            + term_eps
            + _fc._log_phi_cdf(mu_star / sigma_star)
        )
        if not np.isfinite(ll_group).all():
            return 1e20
        return -float(ll_group.sum())

    # ---- Starting values ----

    beta0, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    resid0 = y_vec - X_mat @ beta0
    sigma0 = float(max(np.std(resid0), 1e-3))
    ln_sv0 = np.log(sigma0 * 0.5)
    ln_su0 = np.log(sigma0 * 0.5)
    theta0 = np.concatenate([beta0, [ln_sv0, ln_su0]])
    if has_mu:
        theta0 = np.concatenate([theta0, [0.0]])
    if has_eta:
        theta0 = np.concatenate([theta0, [0.0]])

    # Bounds
    bounds = []
    for _ in range(k_beta):
        bounds.append((-1e6, 1e6))
    bounds.append((-12.0, 5.0))   # ln_sigma_v
    bounds.append((-12.0, 5.0))   # ln_sigma_u
    if has_mu:
        bounds.append((-50.0, 50.0))
    if has_eta:
        bounds.append((-2.0, 2.0))  # eta reasonable range

    result = minimize(
        neg_loglik,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
    )
    theta_hat = result.x
    ll_val = -neg_loglik(theta_hat)

    beta_hat = theta_hat[:k_beta]
    sigma_v = float(np.exp(theta_hat[idx_ln_sv]))
    sigma_u = float(np.exp(theta_hat[idx_ln_su]))
    mu_hat = float(theta_hat[idx_mu]) if has_mu else 0.0
    eta_hat = float(theta_hat[idx_eta]) if has_eta else 0.0
    a_vec = compute_a(eta_hat)

    # SE
    H = _fc.numerical_hessian(neg_loglik, theta_hat)
    vcov_oim = _fc.safe_invert_hessian(H)
    if vce == "oim":
        vcov = vcov_oim
    else:
        group_scores = _fc.per_obs_scores(per_group_loglik, theta_hat)
        # per_group_loglik returns shape (N,) so group_scores is (N, k).
        if vce == "opg":
            OPG = group_scores.T @ group_scores
            vcov = _fc.safe_invert_hessian(OPG)
        else:  # robust or cluster
            if (cluster_effective is None or cluster_effective == id_col):
                # Groups already = panel units; score summation is identity.
                vcov = _fc.robust_vcov(H, group_scores, cluster_idx=None)
            else:
                # Re-cluster groups into meta-clusters.
                meta = df.groupby(id_col)[cluster_effective].first()
                meta_idx = pd.Categorical(meta.values).codes.astype(int)
                vcov = _fc.robust_vcov(H, group_scores, cluster_idx=meta_idx)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    # Posterior E[u_i | e_i] using derived formulas
    eps_hat = y_vec - X_mat @ beta_hat
    w_sq = a_vec**2
    C_i = np.bincount(group_idx, weights=w_sq, minlength=N)
    A_i = np.bincount(group_idx, weights=a_vec * eps_hat, minlength=N)
    denom = C_i * sigma_u**2 + sigma_v**2
    sigma_star = np.sqrt(sigma_v**2 * sigma_u**2 / denom)
    mu_star = (sign * sigma_u**2 * A_i + sigma_v**2 * mu_hat) / denom

    E_u_i = _fc._posterior_truncnormal_mean(mu_star, sigma_star)
    TE_bc_i = _fc._battese_coelli_te(mu_star, sigma_star)
    TE_jlms_i = np.clip(np.exp(-E_u_i), 0.0, 1.0)

    # Unit-level and observation-level efficiencies
    # Obs-level: u_it = a_it * u_i, so TE_it = exp(-a_it * u_i) ≈ exp(-a_it * E[u_i|e_i])
    # Using JLMS: TE_jlms_obs = exp(-a_it * E_u_i[group_idx])
    E_u_obs = a_vec * E_u_i[group_idx]
    TE_jlms_obs = np.clip(np.exp(-E_u_obs), 0.0, 1.0)
    # For BC: compute E[exp(-a_it u_i)|e_i] where u_i ~ N+(mu*, sigma*^2).
    # MGF-type formula: E[exp(-c*X)] with X ~ N+(mu, sigma^2) =
    #   exp(-c*mu + 0.5 c^2 sigma^2) * Phi(mu/sigma - c*sigma) / Phi(mu/sigma).
    # Vectorized via group_idx expansion (avoids O(n) Python loop).
    mu_star_obs = mu_star[group_idx]
    sigma_star_obs = sigma_star[group_idx]
    c = a_vec
    log_num = _fc._log_phi_cdf(mu_star_obs / sigma_star_obs - c * sigma_star_obs)
    log_den = _fc._log_phi_cdf(mu_star_obs / sigma_star_obs)
    TE_bc_obs = np.exp(
        -c * mu_star_obs + 0.5 * c**2 * sigma_star_obs**2 + log_num - log_den
    )
    TE_bc_obs = np.clip(TE_bc_obs, 0.0, 1.0)

    params = pd.Series(theta_hat, index=param_names)
    std_errors = pd.Series(se, index=param_names)

    sigma2_total = sigma_v**2 + sigma_u**2
    return FrontierResult(
        params=params,
        std_errors=std_errors,
        model_info={
            "model_type": (
                f"Panel Stochastic Frontier ({model.upper()}, "
                f"{'Cost' if cost else 'Production'})"
            ),
            "method": f"Panel ML ({model}, {dist})",
            "panel_model": model,
            "inefficiency_dist": dist,
            "vce": vce if cluster_effective is None else f"cluster({cluster_effective})",
            "cost": cost,
            "sign": sign,
            "te_method": "bc",
            "sigma_v": sigma_v,
            "sigma_u": sigma_u,
            "lambda": sigma_u / sigma_v if sigma_v > 0 else np.nan,
            "gamma": sigma_u**2 / sigma2_total,
            "mu": mu_hat if has_mu else None,
            "eta": eta_hat if has_eta else None,
            "mean_efficiency_bc": float(np.mean(TE_bc_obs)),
            "mean_efficiency_jlms": float(np.mean(TE_jlms_obs)),
            "mean_unit_efficiency_bc": float(np.mean(TE_bc_i)),
            "converged": bool(result.success),
        },
        data_info={
            "n_obs": n,
            "n_units": N,
            "dep_var": y,
            "regressors": list(x),
            "id_col": id_col,
            "time_col": time_col,
            "df_resid": max(n - k_total, 1),
        },
        diagnostics={
            "log_likelihood": float(ll_val),
            "aic": float(-2.0 * ll_val + 2.0 * k_total),
            "bic": float(-2.0 * ll_val + np.log(n) * k_total),
            "sigma_u": sigma_u,
            "sigma_v": sigma_v,
            "efficiency_bc": TE_bc_obs,            # per observation
            "efficiency_jlms": TE_jlms_obs,
            "inefficiency_jlms": E_u_obs,
            "efficiency_bc_unit": pd.Series(TE_bc_i, index=unique_ids, name="te_bc"),
            "efficiency_jlms_unit": pd.Series(
                np.clip(np.exp(-E_u_i), 0.0, 1.0),
                index=unique_ids, name="te_jlms",
            ),
            "unit_ids": np.asarray(unique_ids),
            "group_idx": group_idx,
            "eps": eps_hat,
            "a_it": a_vec,
            "sigma_u_i": np.full(n, sigma_u),
            "sigma_v_i": np.full(n, sigma_v),
            "mu_i": np.full(n, mu_hat),
            "efficiency_index": df.index.to_numpy(),
            "hessian": H,
            "vcov": vcov,
        },
    )


__all__ = ["xtfrontier"]
