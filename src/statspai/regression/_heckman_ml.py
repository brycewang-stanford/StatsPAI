"""Full-information maximum likelihood for the Heckman selection model.

This is the estimator Stata's ``heckman y x, select(d = z)`` fits by default
(``twostep`` is the option); :func:`statspai.heckman` reaches it with
``method="ml"``. Parameterisation follows Stata: ``theta = (beta, gamma,
atanh(rho), ln(sigma))`` with per-observation log-likelihood

* selected (``d = 1``): ``ln Phi((z'g + rho r) / sqrt(1 - rho^2)) - r^2/2
  - ln sigma - ln sqrt(2 pi)`` with ``r = (y - x'b) / sigma``;
* not selected: ``ln Phi(-z'g)``.

The fit runs BFGS from the two-step estimates, then Newton on complex-step
scores (the shared ML path in ``_optim_helpers``); standard errors are
Stata's ``vce(oim)`` / ``vce(robust)`` / ``vce(cluster)`` through
:func:`statspai.core._vcov.ml_vcov`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from ..core._vcov import ml_vcov
from ..exceptions import MethodIncompatibility
from ._optim_helpers import (
    complex_step_scores,
    inverse_information,
    ml_newton_polish,
    robust_convergence,
    se_from_vcov,
)


def heckman_ml(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    select: str,
    z: List[str],
    *,
    se_kind: str,
    cluster: Optional[str],
    weights: Optional[str],
    start: Dict[str, Any],
) -> Dict[str, Any]:
    """Fit the Heckman model by ML; ``start`` holds two-step estimates.

    Returns a dict with the coefficient table and fit statistics that
    :func:`statspai.regression.heckman.heckman` wraps in its result.
    """
    extra = [c for c in (cluster, weights) if c is not None]
    need_all = list(dict.fromkeys([select] + list(z) + extra))
    df = data.loc[data[need_all].notna().all(axis=1)]
    d_all = df[select].to_numpy(dtype=float)
    if not np.isin(d_all, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"heckman: the selection indicator {select!r} must be 0/1.",
            diagnostics={"select": select},
        )
    # A selected row needs its outcome and regressors; an unselected row
    # contributes only through the selection equation.
    sel_ok = df[[y] + list(x)].notna().all(axis=1).to_numpy()
    keep = (d_all == 0) | sel_ok
    df = df.loc[keep]
    D = df[select].to_numpy(dtype=float)
    sel = D == 1
    n = len(df)
    Z = np.column_stack([np.ones(n)] + [df[v].to_numpy(dtype=float) for v in z])
    X = np.column_stack(
        [np.ones(n)] + [df[v].to_numpy(dtype=float, na_value=0.0) for v in x]
    )
    X[~sel] = 0.0
    Y = np.where(sel, df[y].to_numpy(dtype=float, na_value=0.0), 0.0)
    kx, kz = X.shape[1], Z.shape[1]

    wt = np.ones(n)
    if weights is not None:
        wt = df[weights].to_numpy(dtype=float)
        if not np.all(np.isfinite(wt)) or np.any(wt <= 0):
            raise MethodIncompatibility(
                "heckman: weights must be finite and strictly positive.",
                diagnostics={"weights": weights},
            )
        wt = wt * (n / wt.sum())

    half_log_2pi = 0.5 * np.log(2 * np.pi)

    def obs_loglik(theta: np.ndarray) -> np.ndarray:
        beta, gamma = theta[:kx], theta[kx : kx + kz]
        rho = np.tanh(theta[kx + kz])
        ln_s = theta[kx + kz + 1]
        zg = Z @ gamma
        out = np.zeros(n, dtype=np.result_type(theta, float))
        r = (Y[sel] - X[sel] @ beta) / np.exp(ln_s)
        a = (zg[sel] + rho * r) / np.sqrt(1 - rho * rho)
        out[sel] = special.log_ndtr(a) - 0.5 * r * r - ln_s - half_log_2pi
        out[~sel] = special.log_ndtr(-zg[~sel])
        return wt * out

    def neg_total(theta: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            v = -float(np.sum(obs_loglik(theta)))
        return v if np.isfinite(v) else 1e300

    def neg_grad(theta: np.ndarray) -> np.ndarray:
        grad: np.ndarray = -complex_step_scores(obs_loglik, theta).sum(axis=0)
        return grad

    rho0 = float(np.clip(start["rho"], -0.9, 0.9))
    theta0 = np.concatenate(
        [
            np.asarray(start["beta"], dtype=float),
            np.asarray(start["gamma"], dtype=float),
            [np.arctanh(rho0), np.log(max(float(start["sigma"]), 1e-6))],
        ]
    )
    opt = optimize.minimize(
        neg_total, theta0, jac=neg_grad, method="BFGS", options={"maxiter": 2000}
    )
    converged, _ = robust_convergence(opt)
    theta, scores, H, _ = ml_newton_polish(obs_loglik, opt.x)
    grad_norm = float(np.max(np.abs(scores.sum(axis=0))))
    converged = bool(converged or grad_norm < 1e-6)

    clusters = df[cluster].to_numpy() if se_kind == "cluster" else None
    V = ml_vcov(
        inverse_information(H),
        scores if se_kind != "nonrobust" else None,
        kind=se_kind,
        clusters=clusters,
    )
    se = se_from_vcov(V)

    t_idx, s_idx = kx + kz, kx + kz + 1
    athrho, lnsig = float(theta[t_idx]), float(theta[s_idx])
    rho, sigma = float(np.tanh(athrho)), float(np.exp(lnsig))
    lam = rho * sigma
    V_ts = V[np.ix_([t_idx, s_idx], [t_idx, s_idx])]
    # Delta method (Stata's rho / sigma / lambda rows).
    g_rho = np.array([1 - rho**2, 0.0])
    g_sig = np.array([0.0, sigma])
    g_lam = np.array([sigma * (1 - rho**2), lam])
    se_rho = float(np.sqrt(g_rho @ V_ts @ g_rho))
    se_sig = float(np.sqrt(g_sig @ V_ts @ g_sig))
    se_lam = float(np.sqrt(g_lam @ V_ts @ g_lam))

    names = (
        ["const"]
        + list(x)
        + ["select:const"]
        + [f"select:{v}" for v in z]
        + ["athrho", "lnsigma"]
    )
    coef = np.concatenate([theta, [rho, sigma, lam]])
    ses = np.concatenate([se, [se_rho, se_sig, se_lam]])
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = coef / ses
    detail = pd.DataFrame(
        {
            "variable": names + ["rho", "sigma", "lambda"],
            "coefficient": coef,
            "se": ses,
            "z": zstat,
            "pvalue": 2 * stats.norm.sf(np.abs(zstat)),
        }
    )
    # Wald test of independent equations (rho = 0), on athrho as Stata does
    # under robust / cluster variances.
    wald = float((athrho / se[t_idx]) ** 2) if se[t_idx] > 0 else np.nan
    return {
        "detail": detail,
        "beta": theta[:kx],
        "se_beta": se[:kx],
        "rho": rho,
        "sigma": sigma,
        "lambda": lam,
        "lambda_se": se_lam,
        "loglik": float(np.sum(obs_loglik(theta))),
        "n_total": n,
        "n_selected": int(sel.sum()),
        "converged": converged,
        "gradient_norm": grad_norm,
        "wald_rho0": wald,
        "wald_rho0_p": float(stats.chi2.sf(wald, 1)) if np.isfinite(wald) else np.nan,
        "n_clusters": (
            int(pd.Series(clusters).nunique()) if clusters is not None else None
        ),
        "vcov": V,
    }
