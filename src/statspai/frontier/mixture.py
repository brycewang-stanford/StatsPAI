"""
Mixture stochastic frontier models.

* :func:`zisf` — Zero-Inefficiency SFA (Kumbhakar-Parmeter-Tsionas 2013):
  a mixture of a *fully efficient* regime (u = 0, pure noise) and a
  standard normal / half-normal composed-error regime.  The mixing
  probability can depend on covariates via a logit link.

* :func:`lcsf` — Latent-Class SFA (Orea-Kumbhakar 2004, Greene 2005):
  two technology classes with separate frontiers and composed errors.
  Class probabilities can depend on covariates via a logit link.

Both are fitted by direct maximum likelihood (EM would also work, but
L-BFGS-B on the mixture log-likelihood is simpler and usually faster
for two-component mixtures).

References
----------
Kumbhakar, S.C., Parmeter, C.F. & Tsionas, E.G. (2013).  "A zero
    inefficiency stochastic frontier model."  J. Econometrics 172,
    66-76.
Orea, L. & Kumbhakar, S.C. (2004).  "Efficiency measurement using a
    latent class stochastic frontier model."  Empirical Economics 29,
    169-183.
Greene, W.H. (2005).  "Reconsidering heterogeneity in panel data
    estimators of the stochastic frontier model."  J. Econometrics
    126, 269-303.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.results import EconometricResults
from . import _core as _fc
from .sfa import FrontierResult


# ---------------------------------------------------------------------------
# Zero-Inefficiency SFA
# ---------------------------------------------------------------------------


def zisf(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    *,
    zprob: Optional[List[str]] = None,
    dist: str = "half-normal",
    cost: bool = False,
    maxiter: int = 500,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> FrontierResult:
    """Zero-Inefficiency Stochastic Frontier (Kumbhakar-Parmeter-Tsionas 2013).

    The population is a mixture of two regimes:

    * **Fully efficient** (share ``p_i``): ``y_it = x_it' beta + v_it``.
    * **Inefficient** (share ``1 - p_i``): standard composed-error
      frontier ``y = x'beta + v + sign * u``.

    The mixing probability ``p_i`` is parameterised via a logit link:
    ``p_i = expit(z_i' theta)`` where ``z_i = [1, zprob_vars_i]``.
    If ``zprob=None`` the probability is constant across observations.

    Parameters
    ----------
    zprob : list of str, optional
        Covariates for the mixing probability; a constant is added.
    dist : {'half-normal'}
        Distribution of ``u`` in the inefficient regime.  Currently
        only half-normal is supported (KPT 2013 baseline).

    Returns
    -------
    :class:`~statspai.frontier.FrontierResult`
    """
    if dist not in {"half-normal"}:
        raise ValueError("zisf currently supports dist='half-normal' only.")
    sign = 1 if cost else -1

    required = [y] + list(x) + (list(zprob) if zprob else [])
    df = data[required].dropna().copy()
    n = len(df)
    y_vec, X_mat, beta_names = _fc.build_design(df, y, x, add_constant=True)
    Z_mat, zprob_names = _fc.build_optional_design(
        df, zprob, include_constant=True, prefix="p_"
    )
    if Z_mat is None:
        # Constant-probability: single logit intercept parameter.
        Z_mat = np.ones((n, 1))
        zprob_names = ["p__cons"]

    k_beta = X_mat.shape[1]
    k_theta = Z_mat.shape[1]
    k_total = k_beta + k_theta + 2  # beta + theta + ln_sigma_v + ln_sigma_u

    def per_obs_loglik(params: np.ndarray) -> np.ndarray:
        beta = params[:k_beta]
        theta = params[k_beta:k_beta + k_theta]
        ln_sigma_v = params[k_beta + k_theta]
        ln_sigma_u = params[k_beta + k_theta + 1]
        sigma_v = np.exp(ln_sigma_v)
        sigma_u = np.exp(ln_sigma_u)
        eps = y_vec - X_mat @ beta

        # log p_i and log(1 - p_i)
        xb = Z_mat @ theta
        log_p = -np.logaddexp(0.0, -xb)      # log sigmoid(xb)
        log_1mp = -np.logaddexp(0.0, xb)     # log(1 - sigmoid(xb))

        # Log density under efficient regime: N(0, sigma_v^2)
        log_f_eff = stats.norm.logpdf(eps, loc=0.0, scale=sigma_v)
        # Log density under composed error (HN)
        log_f_ineff = _fc.loglik_halfnormal(
            eps,
            np.full(n, sigma_v),
            np.full(n, sigma_u),
            sign,
        )
        return np.logaddexp(log_p + log_f_eff, log_1mp + log_f_ineff)

    def neg_loglik(params: np.ndarray) -> float:
        if not np.all(np.isfinite(params)):
            return 1e20
        sigma_v = np.exp(params[k_beta + k_theta])
        sigma_u = np.exp(params[k_beta + k_theta + 1])
        if not (1e-8 < sigma_v < 1e6 and 1e-8 < sigma_u < 1e6):
            return 1e20
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ll = per_obs_loglik(params)
        if not np.isfinite(ll).all():
            return 1e20
        return -float(ll.sum())

    # Starting values
    beta0, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    resid0 = y_vec - X_mat @ beta0
    sigma0 = max(float(np.std(resid0)), 1e-3)
    theta0 = np.concatenate([
        beta0,
        np.zeros(k_theta),  # logit intercept 0 -> p = 0.5
        [np.log(sigma0 * 0.5), np.log(sigma0 * 0.5)],
    ])

    bounds = [(-1e6, 1e6)] * k_beta + [(-10.0, 10.0)] * k_theta + [(-12.0, 5.0)] * 2
    result = minimize(neg_loglik, theta0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": maxiter, "ftol": tol, "gtol": tol})
    theta_hat = result.x
    ll_val = -neg_loglik(theta_hat)

    beta_hat = theta_hat[:k_beta]
    theta_p = theta_hat[k_beta:k_beta + k_theta]
    sigma_v = float(np.exp(theta_hat[k_beta + k_theta]))
    sigma_u = float(np.exp(theta_hat[k_beta + k_theta + 1]))
    p_i = 1.0 / (1.0 + np.exp(-(Z_mat @ theta_p)))

    H = _fc.numerical_hessian(neg_loglik, theta_hat)
    vcov = _fc.safe_invert_hessian(H)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    # Posterior efficiency:
    # Given eps, posterior P(efficient | eps) = p_i * f_eff / [p_i*f_eff + (1-p_i)*f_ineff]
    eps_hat = y_vec - X_mat @ beta_hat
    log_p = -np.logaddexp(0.0, -(Z_mat @ theta_p))
    log_1mp = -np.logaddexp(0.0, (Z_mat @ theta_p))
    log_f_eff = stats.norm.logpdf(eps_hat, loc=0.0, scale=sigma_v)
    log_f_ineff = _fc.loglik_halfnormal(
        eps_hat, np.full(n, sigma_v), np.full(n, sigma_u), sign
    )
    log_num = log_p + log_f_eff
    log_den = np.logaddexp(log_num, log_1mp + log_f_ineff)
    p_eff_post = np.exp(log_num - log_den)
    # Conditional TE = P(efficient)*1 + P(inefficient)*TE_HN_BC
    _, TE_HN = _fc.jondrow_halfnormal(
        eps_hat, np.full(n, sigma_v), np.full(n, sigma_u), sign
    )
    TE_zisf = p_eff_post * 1.0 + (1.0 - p_eff_post) * TE_HN
    TE_zisf = np.clip(TE_zisf, 0.0, 1.0)
    E_u = (1.0 - p_eff_post) * _fc.jondrow_halfnormal(
        eps_hat, np.full(n, sigma_v), np.full(n, sigma_u), sign
    )[0]

    param_names = list(beta_names) + list(zprob_names) + ["ln_sigma_v", "ln_sigma_u"]
    params_s = pd.Series(theta_hat, index=param_names)
    std_errors = pd.Series(se, index=param_names)

    return FrontierResult(
        params=params_s,
        std_errors=std_errors,
        model_info={
            "model_type": (
                f"Zero-Inefficiency SFA ({'Cost' if cost else 'Production'})"
            ),
            "method": f"ZISF ({dist})",
            "inefficiency_dist": dist,
            "cost": cost,
            "sign": sign,
            "te_method": "bc_mixture",
            "te_note": (
                "Mixture posterior: p_eff*1 + (1-p_eff)*E[exp(-u)|eps]; "
                "not the vanilla Battese-Coelli scalar"
            ),
            "vce": "oim",
            "has_zprob": zprob is not None,
            "sigma_u_mean": sigma_u,
            "sigma_v_mean": sigma_v,
            "sigma": float(np.sqrt(sigma_u**2 + sigma_v**2)),
            "lambda": sigma_u / sigma_v if sigma_v > 0 else np.nan,
            "gamma": sigma_u**2 / (sigma_u**2 + sigma_v**2),
            "mean_p_efficient": float(np.mean(p_i)),
            "mean_efficiency_bc": float(np.mean(TE_zisf)),
            "mean_efficiency_jlms": float(np.mean(TE_zisf)),
            "converged": bool(result.success),
        },
        data_info={
            "n_obs": n,
            "dep_var": y,
            "regressors": list(x),
            "usigma_cols": None,
            "vsigma_cols": None,
            "emean_cols": None,
            "zprob_cols": list(zprob) if zprob else None,
            "df_resid": max(n - k_total, 1),
        },
        diagnostics={
            "log_likelihood": float(ll_val),
            "aic": float(-2.0 * ll_val + 2.0 * k_total),
            "bic": float(-2.0 * ll_val + np.log(n) * k_total),
            "sigma_u_i": np.full(n, sigma_u),
            "sigma_v_i": np.full(n, sigma_v),
            "mu_i": np.zeros(n),
            "eps": eps_hat,
            "efficiency_bc": TE_zisf,
            "efficiency_jlms": TE_zisf,
            "inefficiency_jlms": E_u,
            "p_efficient_prior": p_i,
            "p_efficient_posterior": p_eff_post,
            "efficiency_index": df.index.to_numpy(),
            "hessian": H,
            "vcov": vcov,
        },
    )


# ---------------------------------------------------------------------------
# Latent-Class SFA (2 classes)
# ---------------------------------------------------------------------------


def lcsf(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    *,
    z_class: Optional[List[str]] = None,
    dist: str = "half-normal",
    cost: bool = False,
    maxiter: int = 500,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> FrontierResult:
    """Two-class Latent-Class SFA (Orea-Kumbhakar 2004; Greene 2005).

    Each observation belongs latently to class 1 or class 2, each with
    its own frontier coefficients ``beta_k`` and variance parameters
    ``sigma_v_k``, ``sigma_u_k``.  Class probability optionally depends
    on ``z_class`` via logit.

    Parameters
    ----------
    z_class : list of str, optional
        Covariates shifting the class-1 logit probability.

    Returns
    -------
    FrontierResult with extended ``params`` block and per-obs posterior
    class probabilities in ``diagnostics['p_class1_posterior']``.
    """
    if dist not in {"half-normal"}:
        raise ValueError("lcsf currently supports dist='half-normal' only.")
    sign = 1 if cost else -1

    required = [y] + list(x) + (list(z_class) if z_class else [])
    df = data[required].dropna().copy()
    n = len(df)
    y_vec, X_mat, beta_names = _fc.build_design(df, y, x, add_constant=True)
    Z_mat, z_class_names = _fc.build_optional_design(
        df, z_class, include_constant=True, prefix="pi_"
    )
    if Z_mat is None:
        Z_mat = np.ones((n, 1))
        z_class_names = ["pi__cons"]

    k_beta = X_mat.shape[1]
    k_theta = Z_mat.shape[1]
    # params: [beta1, ln_sv1, ln_su1, beta2, ln_sv2, ln_su2, theta_class]
    k_total = 2 * (k_beta + 2) + k_theta

    def per_obs_loglik(params: np.ndarray) -> np.ndarray:
        idx = 0
        beta1 = params[idx:idx + k_beta]; idx += k_beta
        ln_sv1 = params[idx]; idx += 1
        ln_su1 = params[idx]; idx += 1
        beta2 = params[idx:idx + k_beta]; idx += k_beta
        ln_sv2 = params[idx]; idx += 1
        ln_su2 = params[idx]; idx += 1
        theta = params[idx:]
        sv1, su1 = np.exp(ln_sv1), np.exp(ln_su1)
        sv2, su2 = np.exp(ln_sv2), np.exp(ln_su2)
        eps1 = y_vec - X_mat @ beta1
        eps2 = y_vec - X_mat @ beta2
        log_pi1 = -np.logaddexp(0.0, -(Z_mat @ theta))
        log_pi2 = -np.logaddexp(0.0, (Z_mat @ theta))
        log_f1 = _fc.loglik_halfnormal(eps1, np.full(n, sv1), np.full(n, su1), sign)
        log_f2 = _fc.loglik_halfnormal(eps2, np.full(n, sv2), np.full(n, su2), sign)
        return np.logaddexp(log_pi1 + log_f1, log_pi2 + log_f2)

    def neg_loglik(params: np.ndarray) -> float:
        if not np.all(np.isfinite(params)):
            return 1e20
        sv1 = np.exp(params[k_beta])
        su1 = np.exp(params[k_beta + 1])
        sv2 = np.exp(params[2 * k_beta + 2])
        su2 = np.exp(params[2 * k_beta + 3])
        if any(not (1e-8 < s < 1e6) for s in (sv1, su1, sv2, su2)):
            return 1e20
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ll = per_obs_loglik(params)
        if not np.isfinite(ll).all():
            return 1e20
        return -float(ll.sum())

    beta0, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    resid0 = y_vec - X_mat @ beta0
    sigma0 = max(float(np.std(resid0)), 1e-3)

    bounds = (
        [(-1e6, 1e6)] * k_beta + [(-12.0, 5.0), (-12.0, 5.0)]
        + [(-1e6, 1e6)] * k_beta + [(-12.0, 5.0), (-12.0, 5.0)]
        + [(-10.0, 10.0)] * k_theta
    )

    # Multi-start to break label symmetry. A single deterministic 0.9/1.1
    # perturbation fails when beta0 ~ 0 (both classes start from the same
    # point). We try several random perturbations on top of a fixed
    # sigma_u asymmetry and take the best-LL result.
    rng = np.random.default_rng(12345)
    n_starts = 4
    sigma_asym = np.array([
        [0.3, 0.6],  # class 1 lower su, class 2 higher
        [0.5, 0.5],
        [0.25, 0.75],
        [0.4, 0.55],
    ])
    best_result = None
    best_fun = np.inf
    for s in range(n_starts):
        jitter1 = rng.normal(0.0, 0.3, size=beta0.shape)
        jitter2 = rng.normal(0.0, 0.3, size=beta0.shape)
        su1_start = np.log(max(sigma0 * sigma_asym[s, 0], 1e-3))
        su2_start = np.log(max(sigma0 * sigma_asym[s, 1], 1e-3))
        theta_start = np.concatenate([
            beta0 + jitter1, [np.log(sigma0 * 0.4), su1_start],
            beta0 + jitter2, [np.log(sigma0 * 0.4), su2_start],
            np.zeros(k_theta),
        ])
        try:
            r = minimize(
                neg_loglik, theta_start, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
            )
        except Exception:
            continue
        if r.fun < best_fun and np.isfinite(r.fun):
            best_fun = r.fun
            best_result = r
    if best_result is None:
        raise RuntimeError("lcsf: all starts failed to converge.")
    result = best_result
    theta_hat = result.x.copy()

    # Canonical labeling: enforce sigma_u_class1 <= sigma_u_class2 so that
    # downstream posterior class probs and param blocks are comparable
    # across datasets / bootstrap draws. Without this, classes are
    # arbitrarily ordered by optimizer path.
    _ln_su1 = theta_hat[k_beta + 1]
    _ln_su2 = theta_hat[2 * k_beta + 3]
    if _ln_su1 > _ln_su2:
        block1 = theta_hat[:k_beta + 2].copy()
        block2 = theta_hat[k_beta + 2:2 * k_beta + 4].copy()
        theta_class = theta_hat[2 * k_beta + 4:].copy()
        # Swap block1 and block2; flip sign of class logit so that
        # the now-class-1 has the same physical probability.
        theta_hat = np.concatenate([block2, block1, -theta_class])
    ll_val = -neg_loglik(theta_hat)

    idx = 0
    beta1_hat = theta_hat[idx:idx + k_beta]; idx += k_beta
    ln_sv1 = theta_hat[idx]; idx += 1
    ln_su1 = theta_hat[idx]; idx += 1
    beta2_hat = theta_hat[idx:idx + k_beta]; idx += k_beta
    ln_sv2 = theta_hat[idx]; idx += 1
    ln_su2 = theta_hat[idx]; idx += 1
    theta_p = theta_hat[idx:]
    sv1, su1 = float(np.exp(ln_sv1)), float(np.exp(ln_su1))
    sv2, su2 = float(np.exp(ln_sv2)), float(np.exp(ln_su2))

    # Posterior class probs.
    eps1 = y_vec - X_mat @ beta1_hat
    eps2 = y_vec - X_mat @ beta2_hat
    log_pi1 = -np.logaddexp(0.0, -(Z_mat @ theta_p))
    log_pi2 = -np.logaddexp(0.0, (Z_mat @ theta_p))
    log_f1 = _fc.loglik_halfnormal(eps1, np.full(n, sv1), np.full(n, su1), sign)
    log_f2 = _fc.loglik_halfnormal(eps2, np.full(n, sv2), np.full(n, su2), sign)
    log_num1 = log_pi1 + log_f1
    log_den = np.logaddexp(log_num1, log_pi2 + log_f2)
    p1_post = np.exp(log_num1 - log_den)

    # Mix efficiency.
    _, TE1 = _fc.jondrow_halfnormal(eps1, np.full(n, sv1), np.full(n, su1), sign)
    _, TE2 = _fc.jondrow_halfnormal(eps2, np.full(n, sv2), np.full(n, su2), sign)
    TE_lcsf = p1_post * TE1 + (1.0 - p1_post) * TE2

    H = _fc.numerical_hessian(neg_loglik, theta_hat)
    vcov = _fc.safe_invert_hessian(H)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    param_names = (
        [f"c1:{b}" for b in beta_names] + ["c1:ln_sigma_v", "c1:ln_sigma_u"]
        + [f"c2:{b}" for b in beta_names] + ["c2:ln_sigma_v", "c2:ln_sigma_u"]
        + list(z_class_names)
    )
    params_s = pd.Series(theta_hat, index=param_names)
    std_errors = pd.Series(se, index=param_names)

    return FrontierResult(
        params=params_s,
        std_errors=std_errors,
        model_info={
            "model_type": (
                f"Latent-Class SFA (2 classes, "
                f"{'Cost' if cost else 'Production'})"
            ),
            "method": f"LCSF ({dist})",
            "inefficiency_dist": dist,
            "cost": cost,
            "sign": sign,
            "te_method": "bc_mixture",
            "te_note": (
                "Mixture posterior: p1*E[exp(-u1)|eps] + "
                "(1-p1)*E[exp(-u2)|eps]; labels canonical by ascending sigma_u"
            ),
            "vce": "oim",
            "sigma_u_mean": (su1 + su2) / 2.0,
            "sigma_v_mean": (sv1 + sv2) / 2.0,
            "mean_efficiency_bc": float(np.mean(TE_lcsf)),
            "mean_efficiency_jlms": float(np.mean(TE_lcsf)),
            "n_classes": 2,
            "class1_sigma_u": su1,
            "class2_sigma_u": su2,
            "mean_p_class1": float(np.mean(p1_post)),
            "converged": bool(result.success),
        },
        data_info={
            "n_obs": n,
            "dep_var": y,
            "regressors": list(x),
            "usigma_cols": None,
            "vsigma_cols": None,
            "emean_cols": None,
            "z_class_cols": list(z_class) if z_class else None,
            "df_resid": max(n - k_total, 1),
        },
        diagnostics={
            "log_likelihood": float(ll_val),
            "aic": float(-2.0 * ll_val + 2.0 * k_total),
            "bic": float(-2.0 * ll_val + np.log(n) * k_total),
            "sigma_u_i": p1_post * su1 + (1 - p1_post) * su2,
            "sigma_v_i": p1_post * sv1 + (1 - p1_post) * sv2,
            "mu_i": np.zeros(n),
            "eps": p1_post * eps1 + (1 - p1_post) * eps2,
            "efficiency_bc": TE_lcsf,
            "efficiency_jlms": TE_lcsf,
            "inefficiency_jlms": -np.log(np.clip(TE_lcsf, 1e-12, 1.0)),
            "p_class1_posterior": p1_post,
            "efficiency_index": df.index.to_numpy(),
            "hessian": H,
            "vcov": vcov,
        },
    )


__all__ = ["zisf", "lcsf"]
