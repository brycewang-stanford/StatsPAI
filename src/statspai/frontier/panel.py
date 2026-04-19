"""
Panel stochastic frontier estimation — :func:`xtfrontier`.

Implements five panel SFA designs:

* ``model='ti'``  — Pitt-Lee (1981) time-invariant: ``u_it = u_i``.
* ``model='tvd'`` — Battese-Coelli (1992) time-varying decay.
* ``model='bc95'`` — Battese-Coelli (1995) inefficiency effects.
* ``model='tfe'`` — Greene (2005) **True Fixed Effects**:
  ``y_it = alpha_i + x_it' beta + v_it + s * u_it``, with ``alpha_i`` as
  firm fixed effects (estimated via dummies) separately from inefficiency
  ``u_it ~ N^+(0, sigma_u^2)``.  Resolves the Pitt-Lee confound where
  unobserved firm heterogeneity was absorbed into ``u_i``.
* ``model='tre'`` — Greene (2005) **True Random Effects**:
  ``alpha_i ~ N(0, sigma_alpha^2)`` integrated out by Gauss-Hermite
  quadrature; ``u_it`` independent inefficiency.  Same identification
  as TFE but efficient when the RE assumption is reasonable.

Equivalent to Stata's::

    xtfrontier y x, ti
    xtfrontier y x, ti dist(tnormal)
    xtfrontier y x, tvd
    frontier y x, dist(tnormal) emean(z)      (BC95)

and R's ``frontier::sfa`` / ``sfaR::sfacross``.  TFE/TRE mirror Greene
(2005) "Reconsidering heterogeneity in panel data estimators of the
stochastic frontier model", *J. Econometrics* 126, 269-303.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp

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
    bias_correct: bool = False,
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
        Time variable (required for ``model='tvd'`` and recommended for
        all panel models; falls back to observation order otherwise).
    model : {'ti', 'tvd', 'bc95', 'tfe', 'tre'}
        ``'ti'``   Pitt-Lee (1981) time-invariant inefficiency.
        ``'tvd'``  Battese-Coelli (1992) time-varying decay.
        ``'bc95'`` Battese-Coelli (1995) inefficiency effects model
        (requires ``emean``).
        ``'tfe'``  Greene (2005) True Fixed Effects: unit dummies +
        cross-sectional composed error.  Recommended for T >= ~10.
        ``'tre'``  Greene (2005) True Random Effects:
        ``alpha_i ~ N(0, sigma_alpha^2)`` integrated out by
        Gauss-Hermite quadrature.
    dist : {'half-normal', 'truncated-normal'}
        For ``ti``, ``tvd``, ``tfe``, ``tre``.  BC95 always uses TN.
    cost : bool, default False
    emean : list of str, optional
        Required for ``model='bc95'``; inefficiency determinants ``z_it``.
    vce : {'oim', 'opg', 'robust', 'cluster'}, default 'oim'
        Variance-covariance estimator.  ``'oim'`` uses the inverse
        observed information.  ``'opg'`` is the outer product of
        gradients (BHHH).  ``'robust'`` is the sandwich
        ``H^-1 (S'S) H^-1``.  Passing ``cluster=`` implies cluster-robust
        SEs (Liang-Zeger 1986).  Note: ``vce='bootstrap'`` is only
        available on the cross-sectional :func:`frontier`; for panel
        models use ``'robust'`` with ``cluster=id`` instead.
    cluster : str, optional
        Column name for cluster-robust SEs.  Defaults to ``id`` whenever
        ``vce != 'oim'`` (the natural grouping for panels).
    bias_correct : bool, default False
        TFE-only.  If True, applies Dhaene-Jochmans (2015) split-panel
        jackknife to reduce the O(1/T) incidental-parameters bias on
        ``beta`` and ``sigma_u``.
    maxiter, tol, alpha : see :func:`frontier`.

    Returns
    -------
    :class:`~statspai.frontier.FrontierResult`
    """
    model = model.lower()
    dist = dist.lower().replace("_", "-")

    if model not in {"ti", "tvd", "bc95", "tfe", "tre"}:
        raise ValueError(f"Unknown panel model: {model!r}.")
    if model == "tvd" and time is None:
        raise ValueError("model='tvd' requires a time variable.")

    if model == "tfe":
        return _fit_tfe(
            data, y, x, id_col=id, time_col=time,
            dist=dist, cost=cost,
            vce=vce, cluster=cluster,
            bias_correct=bias_correct,
            maxiter=maxiter, tol=tol, alpha=alpha,
        )
    if model == "tre":
        return _fit_tre(
            data, y, x, id_col=id, time_col=time,
            dist=dist, cost=cost,
            vce=vce, cluster=cluster,
            maxiter=maxiter, tol=tol, alpha=alpha,
        )
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
    if vce == "bootstrap":
        raise NotImplementedError(
            "vce='bootstrap' is not supported for panel models "
            "('ti'/'tvd'); the group log-likelihood structure makes "
            "row-level bootstrap invalid. Use vce='robust' (or "
            "vce='cluster' with cluster=id) for panel-robust SEs, or "
            "switch to the cross-sectional frontier() if a bootstrap "
            "is essential."
        )
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


# ---------------------------------------------------------------------------
# Greene (2005) True Fixed Effects (TFE)
# ---------------------------------------------------------------------------


def _fit_tfe(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id_col: str,
    time_col: Optional[str],
    dist: str,
    cost: bool,
    vce: str,
    cluster: Optional[str],
    maxiter: int,
    tol: float,
    alpha: float,
    bias_correct: bool = False,
) -> FrontierResult:
    """True Fixed Effects SFA (Greene 2005): firm dummies + composed error.

    y_it = alpha_i + x_it' beta + v_it + sign * u_it,
    with u_it ~ N^+(0, sigma_u^2) independent, v_it ~ N(0, sigma_v^2).
    Firm effects estimated as N-1 dummies (reference firm absorbed in
    constant).  Incidental-parameters concern mild for moderate T.
    """
    required = [y] + list(x) + [id_col]
    if time_col is not None:
        required.append(time_col)
    if cluster is not None and cluster not in required:
        required.append(cluster)
    df = data[required].dropna().copy()
    df = df.sort_values(
        [id_col] + ([time_col] if time_col is not None else [])
    ).reset_index(drop=True)

    # Build firm-dummy columns (drop first for identification, keep cons).
    dummies = pd.get_dummies(df[id_col], prefix=f"_{id_col}", drop_first=True,
                              dtype=float)
    extended_x = list(x) + list(dummies.columns)
    df_ext = pd.concat([df.drop(columns=dummies.columns, errors="ignore"),
                        dummies], axis=1)

    # Delegate to cross-sectional frontier() — it returns a FrontierResult.
    res = _cs_frontier(
        data=df_ext,
        y=y,
        x=extended_x,
        dist=dist,
        cost=cost,
        vce=vce,
        cluster=cluster if cluster is not None else id_col,
        maxiter=maxiter,
        tol=tol,
        alpha=alpha,
    )
    res.model_info["model_type"] = (
        f"Panel Stochastic Frontier (TFE, {'Cost' if cost else 'Production'})"
    )
    res.model_info["method"] = f"Greene 2005 TFE ({dist}, N={df[id_col].nunique()} dummies)"
    res.model_info["panel_model"] = "tfe"
    res.data_info["id_col"] = id_col
    res.data_info["time_col"] = time_col
    res.data_info["n_units"] = df[id_col].nunique()
    # Strip the N-1 firm-dummy param names from the public-facing regressors.
    res.data_info["regressors"] = list(x)

    if not bias_correct:
        return res

    # ------------------------------------------------------------------
    # Dhaene-Jochmans (2015) split-panel jackknife bias correction:
    #   theta_BC = 2 * theta_full - (theta_first_half + theta_second_half) / 2
    # Cuts O(1/T) incidental-parameter bias.  Requires time_col to split.
    # ------------------------------------------------------------------
    if time_col is None:
        raise ValueError("bias_correct=True requires a time variable for split.")
    times = np.sort(df[time_col].unique())
    half = len(times) // 2
    if half < 2:
        raise ValueError(
            "Need at least 4 time periods per unit for split-panel jackknife."
        )
    times_first = set(times[:half])
    times_second = set(times[half:])
    mask1 = df[time_col].isin(times_first)
    mask2 = df[time_col].isin(times_second)

    # Unit-level guard: for unbalanced / short panels, a unit may land in
    # one half with only 1 observation, which makes the firm-dummy TFE
    # likelihood degenerate (the dummy absorbs the single residual
    # perfectly, sigma collapses). Refuse to bias-correct in that case
    # rather than silently emit nonsense.
    t_per_unit_1 = df.loc[mask1.values].groupby(id_col).size()
    t_per_unit_2 = df.loc[mask2.values].groupby(id_col).size()
    min_t1 = int(t_per_unit_1.min()) if len(t_per_unit_1) else 0
    min_t2 = int(t_per_unit_2.min()) if len(t_per_unit_2) else 0
    if min_t1 < 2 or min_t2 < 2:
        import warnings as _warnings
        _warnings.warn(
            f"bias_correct=True: at least one unit has T_i<2 in a "
            f"half-panel (min_T first={min_t1}, second={min_t2}); "
            "skipping Dhaene-Jochmans jackknife to avoid degenerate fit. "
            "Use a longer/more balanced panel for unbiased correction.",
            RuntimeWarning,
            stacklevel=2,
        )
        res.model_info["bias_correct"] = (
            "skipped (degenerate half-panel, min_T<2)"
        )
        return res

    df1 = df_ext[mask1.values].reset_index(drop=True)
    df2 = df_ext[mask2.values].reset_index(drop=True)
    # Refit on each half.
    res1 = _cs_frontier(df1, y=y, x=extended_x, dist=dist, cost=cost,
                        vce="oim", cluster=None,
                        maxiter=maxiter, tol=tol, alpha=alpha)
    res2 = _cs_frontier(df2, y=y, x=extended_x, dist=dist, cost=cost,
                        vce="oim", cluster=None,
                        maxiter=maxiter, tol=tol, alpha=alpha)
    # Guard against degenerate splits: if either half's ln_sigma parameter
    # sits on *either* optimizer bound, BC on that parameter is untrustworthy
    # (documented DJ caveat for very short T). Upper bound catches the
    # sigma-explode case where a unit dummy absorbs all signal.
    corrected = {}
    for name in list(x):
        if name in res1.params.index and name in res2.params.index:
            full = res.params[name]
            avg = 0.5 * (res1.params[name] + res2.params[name])
            corrected[name] = 2.0 * full - avg

    sigma_bound_low = -11.5
    sigma_bound_high = 4.5  # optimizer caps ln_sigma at 5.0
    sigmas_ok = True
    for name in ("ln_sigma_v", "ln_sigma_u"):
        for r in (res1, res2):
            p = r.params[name]
            if p < sigma_bound_low or p > sigma_bound_high:
                sigmas_ok = False
    if sigmas_ok:
        for name in ("ln_sigma_v", "ln_sigma_u"):
            full = res.params[name]
            avg = 0.5 * (res1.params[name] + res2.params[name])
            corrected[name] = 2.0 * full - avg
    # Overwrite in-place (keep SE from full-panel numerical Hessian).
    for k, v in corrected.items():
        res.params[k] = v
    res.model_info["bias_correct"] = (
        "Dhaene-Jochmans 2015 split-panel jackknife"
        + ("" if sigmas_ok else " (sigmas skipped: split on bound)")
    )
    return res


# ---------------------------------------------------------------------------
# Greene (2005) True Random Effects (TRE) with Gauss-Hermite quadrature
# ---------------------------------------------------------------------------


def _fit_tre(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id_col: str,
    time_col: Optional[str],
    dist: str,
    cost: bool,
    vce: str,
    cluster: Optional[str],
    maxiter: int,
    tol: float,
    alpha: float,
    n_quad: int = 24,
) -> FrontierResult:
    """True Random Effects SFA (Greene 2005) via Gauss-Hermite quadrature.

    y_it = alpha_i + x_it' beta + v_it + sign * u_it,
    alpha_i ~ N(0, sigma_alpha^2), v_it ~ N(0, sigma_v^2),
    u_it ~ N^+(0, sigma_u^2)  (half-normal; support for truncated-normal
    omitted here — BC95 is the recommended route when inefficiency has
    covariate-varying mean).

    Integrates alpha_i out of the group likelihood via n_quad-node
    Gauss-Hermite quadrature.
    """
    from scipy import stats as _sst
    if dist not in {"half-normal", "exponential"}:
        raise ValueError(
            "TRE currently supports dist in {'half-normal', 'exponential'}. "
            "For inefficiency determinants use model='bc95'."
        )
    sign = 1 if cost else -1

    required = [y] + list(x) + [id_col]
    if time_col is not None:
        required.append(time_col)
    if cluster is not None and cluster not in required:
        required.append(cluster)
    df = data[required].dropna().copy()
    df = df.sort_values(
        [id_col] + ([time_col] if time_col is not None else [])
    ).reset_index(drop=True)

    y_vec, X_mat, beta_names = _fc.build_design(df, y, x, add_constant=True)
    group_idx, _, counts, unique_ids = _fc.group_panel(df, id_col, time_col)
    N = len(unique_ids)
    n = len(df)
    k_beta = X_mat.shape[1]

    # Gauss-Hermite quadrature nodes/weights (weight exp(-x^2), Sum w = sqrt(pi)).
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)

    # Parameter layout: [beta, ln_sigma_v, ln_sigma_u, ln_sigma_alpha]
    k_total = k_beta + 3
    param_names = list(beta_names) + ["ln_sigma_v", "ln_sigma_u", "ln_sigma_alpha"]

    def per_group_loglik(theta: np.ndarray) -> np.ndarray:
        beta = theta[:k_beta]
        sigma_v = float(np.exp(theta[k_beta]))
        sigma_u = float(np.exp(theta[k_beta + 1]))
        sigma_alpha = float(np.exp(theta[k_beta + 2]))
        eps = y_vec - X_mat @ beta  # (n,)

        alpha_shifts = sigma_alpha * np.sqrt(2.0) * nodes  # (n_quad,)
        # Shift eps by each alpha_k: shape (n_quad, n)
        eps_shifted = eps[None, :] - alpha_shifts[:, None]

        if dist == "half-normal":
            log_f = _fc.loglik_halfnormal(
                eps_shifted, sigma_v, sigma_u, sign
            )
        else:
            log_f = _fc.loglik_exponential(
                eps_shifted, sigma_v, sigma_u, sign
            )
        # Sum log-f within each group, per quadrature node → (n_quad, N).
        log_f_group = np.zeros((n_quad, N))
        for k in range(n_quad):
            log_f_group[k] = np.bincount(
                group_idx, weights=log_f[k], minlength=N
            )
        log_contrib = np.log(weights)[:, None] + log_f_group  # (n_quad, N)
        # scipy logsumexp handles the all-(-inf) corner (returns -inf rather
        # than NaN from exp(-inf - (-inf)) = exp(nan) in a hand-rolled form).
        ll_group = logsumexp(log_contrib, axis=0) - 0.5 * np.log(np.pi)
        return ll_group

    def neg_loglik(theta):
        if not np.all(np.isfinite(theta)):
            return 1e20
        sigma_v = float(np.exp(theta[k_beta]))
        sigma_u = float(np.exp(theta[k_beta + 1]))
        sigma_alpha = float(np.exp(theta[k_beta + 2]))
        if any(s <= 1e-8 or s > 1e6 for s in (sigma_v, sigma_u, sigma_alpha)):
            return 1e20
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ll_group = per_group_loglik(theta)
        if not np.isfinite(ll_group).all():
            return 1e20
        return -float(ll_group.sum())

    # Starting values: OLS for beta, half residual std for sigma_v and sigma_u,
    # between-unit variance for sigma_alpha.
    beta0, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    resid0 = y_vec - X_mat @ beta0
    sigma0 = max(float(np.std(resid0)), 1e-3)
    # Split residual variance between alpha, v, u (rough heuristic).
    ln_sv0 = np.log(sigma0 * 0.4)
    ln_su0 = np.log(sigma0 * 0.4)
    # Between-unit variance estimate.
    group_means = np.bincount(group_idx, weights=resid0, minlength=N) / counts
    between_sd = max(float(np.std(group_means)), 1e-3)
    ln_sa0 = np.log(between_sd)
    theta0 = np.concatenate([beta0, [ln_sv0, ln_su0, ln_sa0]])

    bounds = [(-1e6, 1e6)] * k_beta + [(-12.0, 5.0)] * 3

    result = minimize(
        neg_loglik, theta0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
    )
    theta_hat = result.x
    ll_val = -neg_loglik(theta_hat)

    sigma_v = float(np.exp(theta_hat[k_beta]))
    sigma_u = float(np.exp(theta_hat[k_beta + 1]))
    sigma_alpha = float(np.exp(theta_hat[k_beta + 2]))

    # SE via numerical Hessian, with optional vce='opg'/'robust'
    H = _fc.numerical_hessian(neg_loglik, theta_hat)
    vcov_oim = _fc.safe_invert_hessian(H)
    vce_l = vce.lower()
    if vce_l == "oim":
        vcov = vcov_oim
    else:
        group_scores = _fc.per_obs_scores(per_group_loglik, theta_hat)
        if vce_l == "opg":
            vcov = _fc.safe_invert_hessian(group_scores.T @ group_scores)
        else:
            cluster_effective = cluster if cluster is not None else id_col
            if cluster_effective == id_col:
                vcov = _fc.robust_vcov(H, group_scores, cluster_idx=None)
            else:
                meta = df.groupby(id_col)[cluster_effective].first()
                meta_idx = pd.Categorical(meta.values).codes.astype(int)
                vcov = _fc.robust_vcov(H, group_scores, cluster_idx=meta_idx)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    # TRE efficiency: for simplicity the marginal E[exp(-u_it)] is reported
    # as a single scalar broadcast to all observations. The proper
    # posterior-conditional score E[exp(-u_it) | e_i] (integrating alpha
    # out) is not implemented here — doing so would require a second
    # Gauss-Hermite pass per observation. Users calling
    # ``res.efficiency()`` on a TRE result therefore see *constant*
    # scores; we surface a warning from summary()/efficiency() via the
    # ``panel_model='tre'`` flag in model_info.
    beta_hat = theta_hat[:k_beta]
    eps_hat = y_vec - X_mat @ beta_hat
    E_u_marg = sigma_u * np.sqrt(2.0 / np.pi) if dist == "half-normal" else sigma_u
    TE_jlms_obs = np.full(n, np.exp(-E_u_marg))
    if dist == "half-normal":
        from scipy import stats as _st
        TE_bc_obs = 2.0 * np.exp(sigma_u**2 / 2.0) * _st.norm.cdf(-sigma_u)
    else:
        TE_bc_obs = 1.0 / (1.0 + sigma_u)
    TE_bc_obs = np.full(n, np.clip(TE_bc_obs, 0.0, 1.0))

    params = pd.Series(theta_hat, index=param_names)
    std_errors = pd.Series(se, index=param_names)

    sigma2_v_u = sigma_v**2 + sigma_u**2
    return FrontierResult(
        params=params,
        std_errors=std_errors,
        model_info={
            "model_type": (
                f"Panel Stochastic Frontier (TRE, "
                f"{'Cost' if cost else 'Production'})"
            ),
            "method": f"Greene 2005 TRE ({dist}, n_quad={n_quad})",
            "panel_model": "tre",
            "inefficiency_dist": dist,
            "cost": cost,
            "sign": sign,
            "te_method": "bc",
            "sigma_v": sigma_v,
            "sigma_u": sigma_u,
            "sigma_alpha": sigma_alpha,
            "lambda": sigma_u / sigma_v if sigma_v > 0 else np.nan,
            "gamma": sigma_u**2 / sigma2_v_u,
            "mean_efficiency_bc": float(np.mean(TE_bc_obs)),
            "mean_efficiency_jlms": float(np.mean(TE_jlms_obs)),
            "converged": bool(result.success),
            "vce": vce_l if (cluster is None or cluster == id_col)
                   else f"cluster({cluster})",
            # Tell FrontierResult.efficiency() / summary() that per-obs
            # efficiency is a broadcast marginal, not a posterior score,
            # so callers can be warned rather than silently read a
            # constant vector.
            "efficiency_kind": "tre_marginal",
            "efficiency_note": (
                "TRE reports marginal E[exp(-u)] broadcast to all obs; "
                "posterior E[exp(-u)|e_i] integration not implemented."
            ),
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
            "sigma_alpha": sigma_alpha,
            "efficiency_bc": TE_bc_obs,
            "efficiency_jlms": TE_jlms_obs,
            "efficiency_index": df.index.to_numpy(),
            "sigma_u_i": np.full(n, sigma_u),
            "sigma_v_i": np.full(n, sigma_v),
            "mu_i": np.zeros(n),
            "eps": eps_hat,
            "hessian": H,
            "vcov": vcov,
        },
    )


__all__ = ["xtfrontier"]
