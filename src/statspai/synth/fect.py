"""
Counterfactual estimators for time-series cross-sectional data (fect).

Native port of the estimation core of Liu, Wang and Xu's **fect** R
package (``fect::fect``): a treated-observation-imputation approach in
which the untreated potential outcome of every treated cell is
predicted from a model fitted on the *untreated* cells only, and the
average treatment effect on the treated is the mean of ``Y - Y(0)``
over treated cells. Three outcome models are available:

- ``method="fe"``  -- two-way fixed effects (the imputation / BJS
  estimator when there is no reversal);
- ``method="ife"`` -- interactive fixed effects with ``r`` latent
  factors (Bai 2009; Xu 2017), fitted by the EM algorithm on the
  incomplete untreated panel;
- ``method="mc"``  -- matrix completion with a nuclear-norm penalty
  ``lam`` (Athey et al. 2021), fitted by the same EM scheme with a
  soft-impute M-step.

Every step -- the ``fixest`` two-way initial fit, the E-step that fills
the treated cells with the current fit, the M-step (two-way demeaning,
then either the ``panel_factor`` SVD with fect's ``sqrt(T)`` /
``sqrt(N)`` normalisation or the ``panel_FE`` soft-threshold on
``E/(T*N)``), the relative convergence criterion on the fitted surface
and, for ``ife``, on the interactive component, the relative-period
coding (0 = last untreated period, 1 = first treated period), and the
by-period aggregation -- mirrors ``fect`` 2.4.x so that the parity
module can compare the two implementations on identical bytes.

Standard errors are optional and resampling-based (unit block
bootstrap or unit jackknife), exactly as in ``fect``; they are
stochastic (T3) and are not part of the deterministic parity claim.

References
----------
[@liu2024practical] Liu, Wang and Xu (2024), *American Journal of
Political Science*; [@xu2017generalized]; [@bai2009panel];
[@athey2021matrix].
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient

__all__ = ["fect"]

_FORCE = {"none": 0, "unit": 1, "time": 2, "two-way": 3, "twoway": 3}


# ----------------------------------------------------------------------
# fect primitives (T x N orientation, rows = time, columns = units)
# ----------------------------------------------------------------------


def _y_demean(
    Y: np.ndarray, force: int
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """fect ``Y_demean``: returns (YY, mu_Y, alpha_Y, xi_Y)."""
    T, N = Y.shape
    mu_Y = float(Y.sum() / (N * T))
    alpha_Y = np.zeros(N)
    xi_Y = np.zeros(T)
    if force == 0:
        YY = Y - mu_Y
    elif force == 1:
        alpha_Y = Y.mean(axis=0)
        YY = Y - alpha_Y[None, :]
    elif force == 2:
        xi_Y = Y.mean(axis=1)
        YY = Y - xi_Y[:, None]
    else:
        alpha_Y = Y.mean(axis=0)
        xi_Y = Y.mean(axis=1)
        YY = Y - alpha_Y[None, :] - xi_Y[:, None] + mu_Y
    return YY, mu_Y, alpha_Y, xi_Y


def _fe_add(
    alpha_Y: np.ndarray, xi_Y: np.ndarray, mu_Y: float, T: int, N: int, force: int
):
    """fect ``fe_add``: additive fixed-effect surface and its components."""
    mu = mu_Y
    alpha = np.zeros(N)
    xi = np.zeros(T)
    if force in (1, 3):
        alpha = alpha_Y - mu_Y
    if force in (2, 3):
        xi = xi_Y - mu_Y
    FE_ad = np.full((T, N), mu)
    if force in (1, 3):
        FE_ad = FE_ad + alpha[None, :]
    if force in (2, 3):
        FE_ad = FE_ad + xi[:, None]
    return FE_ad, mu, alpha, xi


def _panel_factor(E: np.ndarray, r: int):
    """fect ``panel_factor``: principal-component factors with fect's normalisation."""
    T, N = E.shape
    if T < N:
        EE = E @ E.T / (N * T)
        U, s, _ = np.linalg.svd(EE)
        factor = U[:, :r] * np.sqrt(float(T))
        lam = E.T @ factor / T
    else:
        EE = E.T @ E / (N * T)
        U, s, _ = np.linalg.svd(EE)
        lam = U[:, :r] * np.sqrt(float(N))
        factor = E @ lam / N
    VNT = np.diag(s[:r])
    return factor, lam, VNT


def _panel_fe_soft(E: np.ndarray, lam: float, hard: int = 0) -> np.ndarray:
    """fect ``panel_FE``: singular-value thresholding of ``E/(T*N)``."""
    T, N = E.shape
    U, s, Vt = np.linalg.svd(E / (T * N), full_matrices=False)
    if hard == 1:
        d = np.where(s > lam, s, 0.0)
    else:
        d = np.where(s > lam, s - lam, 0.0)
    return (U * d) @ Vt * (T * N)


def _ife(
    E: np.ndarray, force: int, mc: int, r: int, hard: int, lam: float
) -> Dict[str, Any]:
    """fect ``ife``: additive FE (+ interactive FE / soft-impute) of a complete matrix."""
    T, N = E.shape
    EE, mu_E, alpha_E, xi_E = _y_demean(E, force)
    FE_add, mu, alpha, xi = _fe_add(alpha_E, xi_E, mu_E, T, N, force)
    out: Dict[str, Any] = {"mu": mu, "alpha": alpha, "xi": xi}
    FE_inter = np.zeros((T, N))
    if r > 0:
        if mc == 0:
            F, L, VNT = _panel_factor(EE, r)
            FE_inter = F @ L.T
            out.update({"factor": F, "lambda": L, "VNT": VNT})
        else:
            FE_inter = _panel_fe_soft(EE, lam, hard)
        out["FE_inter"] = FE_inter
    out["FE"] = FE_add + FE_inter
    return out


def _e_adj(E: np.ndarray, FE: np.ndarray, I: np.ndarray) -> np.ndarray:
    """fect ``E_adj``: keep observed cells, replace unobserved cells by ``FE``."""
    return np.where(I == 1, E, FE)


def _fe_adj(FE: np.ndarray, I: np.ndarray) -> np.ndarray:
    """fect ``FE_adj``: zero the unobserved cells."""
    return np.where(I == 1, FE, 0.0)


def _xx_inv(X: np.ndarray) -> np.ndarray:
    """fect ``XXinv``: inverse of the p x p Gram matrix over all cells."""
    p = X.shape[2]
    G = np.empty((p, p))
    for i in range(p):
        for j in range(p):
            G[i, j] = float(np.sum(X[:, :, i] * X[:, :, j]))
    return np.linalg.inv(G)


def _panel_beta(
    X: np.ndarray, xxinv: np.ndarray, Y: np.ndarray, FE: np.ndarray
) -> np.ndarray:
    """fect ``panel_beta``: (X'X)^{-1} X'(Y - FE) over all cells."""
    p = X.shape[2]
    xy = np.array([float(np.sum(X[:, :, k] * (Y - FE))) for k in range(p)])
    return xxinv @ xy


def _initial_fit(
    Y: np.ndarray, X: Optional[np.ndarray], II: np.ndarray, force: int
) -> Tuple[np.ndarray, np.ndarray]:
    """fect ``initialFit``: OLS with the requested fixed effects on the
    untreated cells, predicted on every cell. Returns (Y0, beta0)."""
    T, N = Y.shape
    p = 0 if X is None else X.shape[2]
    obs = np.where(II.ravel(order="F") == 1)[0]  # column-major like R's c()
    y_all = Y.ravel(order="F")
    unit_id = np.repeat(np.arange(N), T)
    time_id = np.tile(np.arange(T), N)
    cols: List[np.ndarray] = [np.ones(T * N)]
    if force in (1, 3):
        cols.append(unit_id)
    if force in (2, 3):
        cols.append(time_id)
    # Dense dummy design: intercept + unit dummies (drop first) + time
    # dummies (drop first) + covariates. Least squares on the untreated
    # cells reproduces fixest::feols with fixef.rm = "none".
    blocks = [np.ones((T * N, 1))]
    if force in (1, 3):
        Du = np.zeros((T * N, N - 1))
        m = unit_id >= 1
        Du[np.where(m)[0], unit_id[m] - 1] = 1.0
        blocks.append(Du)
    if force in (2, 3):
        Dt = np.zeros((T * N, T - 1))
        m = time_id >= 1
        Dt[np.where(m)[0], time_id[m] - 1] = 1.0
        blocks.append(Dt)
    if p > 0:
        blocks.append(np.column_stack([X[:, :, k].ravel(order="F") for k in range(p)]))
    Z = np.hstack(blocks)
    coef, *_ = np.linalg.lstsq(Z[obs], y_all[obs], rcond=None)
    y0 = Z @ coef
    Y0 = y0.reshape((T, N), order="F")
    beta0 = coef[-p:] if p > 0 else np.zeros(0)
    bad = ~np.isfinite(Y0)
    if bad.any():
        fill = float(np.nanmean(y_all[obs]))
        Y0[bad] = fill if np.isfinite(fill) else 0.0
    return Y0, beta0


def _fe_ad_iter(Y, Y0, I, force, tol, max_iter):
    """fect ``fe_ad_iter`` (r = 0, no covariates)."""
    fit = Y0.copy()
    fit_old = Y0.copy()
    dif = 1.0
    niter = 0
    YY = Y
    T, N = Y.shape
    mu = 0.0
    alpha = np.zeros(N)
    xi = np.zeros(T)
    while dif > tol and niter <= 500:
        YY = _e_adj(Y, fit, I)
        _, mu_Y, alpha_Y, xi_Y = _y_demean(YY, force)
        fit, mu, alpha, xi = _fe_add(alpha_Y, xi_Y, mu_Y, T, N, force)
        dif = np.linalg.norm(fit - fit_old) / np.linalg.norm(fit_old)
        fit_old = fit
        niter += 1
    e = _fe_adj(YY - fit, I)
    return {"mu": mu, "fit": fit, "niter": niter, "e": e, "alpha": alpha, "xi": xi}


def _fe_ad_covar_iter(X, xxinv, Y, Y0, I, beta0, force, tol, max_iter):
    """fect ``fe_ad_covar_iter`` (r = 0, covariates)."""
    p = X.shape[2]
    beta = beta0.copy() if beta0.shape[0] == p else np.zeros(p)
    covar_fit = np.tensordot(X, beta, axes=([2], [0]))
    fit = Y0.copy()
    fit_old = fit.copy()
    FE = fit - covar_fit
    dif = 1.0
    niter = 0
    YY = Y
    inner: Dict[str, Any] = {}
    while dif > tol and niter <= max_iter:
        YY = _e_adj(Y, fit, I)
        beta = _panel_beta(X, xxinv, YY, FE)
        covar_fit = np.tensordot(X, beta, axes=([2], [0]))
        U = _e_adj(YY - covar_fit, FE, I)
        inner = _ife(U, force, 0, 0, 0, 0.0)
        FE = inner["FE"]
        fit = covar_fit + FE
        dif = np.linalg.norm(fit - fit_old) / np.linalg.norm(fit_old)
        fit_old = fit
        niter += 1
    e = _fe_adj(YY - fit, I)
    return {
        "mu": inner["mu"],
        "fit": fit,
        "niter": niter,
        "e": e,
        "beta": beta,
        "alpha": inner["alpha"],
        "xi": inner["xi"],
    }


def _fe_ad_inter_iter(Y, Y0, I, force, mc, r, hard, lam, tol, max_iter):
    """fect ``fe_ad_inter_iter`` (r > 0 or MC, no covariates)."""
    fit = Y0.copy()
    fit_old = fit.copy()
    FE_inter_use = np.zeros_like(Y)
    dif = 1.0
    niter = 0
    YY = Y
    inner: Dict[str, Any] = {}
    while dif > tol and niter <= max_iter:
        YY = _e_adj(Y, fit, I)
        if mc == 0:
            inner = _ife(YY, force, 0, r, 0, 0.0)
        else:
            inner = _ife(YY, force, 1, 1, hard, lam)
        fit = inner["FE"]
        dif = np.linalg.norm(fit - fit_old) / (np.linalg.norm(fit_old) + 1e-10)
        if r > 0 and mc == 0 and "FE_inter" in inner:
            FE_inter_new = inner["FE_inter"]
            norm_inter = np.linalg.norm(FE_inter_use)
            if norm_inter > 1e-10:
                dif_inter = np.linalg.norm(FE_inter_new - FE_inter_use) / norm_inter
                dif = max(dif, dif_inter)
            FE_inter_use = FE_inter_new
        fit_old = fit
        niter += 1
    e = _fe_adj(YY - fit, I)
    out = {
        "mu": inner["mu"],
        "fit": fit,
        "niter": niter,
        "e": e,
        "alpha": inner["alpha"],
        "xi": inner["xi"],
        "validF": int(np.abs(inner.get("FE_inter", np.zeros(1))).sum() >= 1e-10),
    }
    if mc == 0:
        out.update(
            {"factor": inner["factor"], "lambda": inner["lambda"], "VNT": inner["VNT"]}
        )
    return out


def _fe_ad_inter_covar_iter(
    X, xxinv, Y, Y0, I, beta0, force, mc, r, hard, lam, tol, max_iter
):
    """fect ``fe_ad_inter_covar_iter`` (r > 0 or MC, covariates)."""
    p = X.shape[2]
    beta = beta0.copy() if beta0.shape[0] == p else np.zeros(p)
    fit = Y0.copy()
    fit_old = fit.copy()
    covar_fit = np.tensordot(X, beta, axes=([2], [0]))
    FE = fit - covar_fit
    FE_inter_use = np.zeros_like(Y)
    dif = 1.0
    niter = 0
    inner: Dict[str, Any] = {}
    while dif > tol and niter <= max_iter:
        YY = _e_adj(Y, fit, I)
        beta = _panel_beta(X, xxinv, YY, FE)
        covar_fit = np.tensordot(X, beta, axes=([2], [0]))
        U = _e_adj(YY - covar_fit, FE, I)
        inner = _ife(U, force, mc, r, hard, lam)
        FE = inner["FE"]
        fit = covar_fit + FE
        dif = np.linalg.norm(fit - fit_old) / (np.linalg.norm(fit_old) + 1e-10)
        if r > 0 and mc == 0 and "FE_inter" in inner:
            FE_inter_new = inner["FE_inter"]
            norm_inter = np.linalg.norm(FE_inter_use)
            if norm_inter > 1e-10:
                dif_inter = np.linalg.norm(FE_inter_new - FE_inter_use) / norm_inter
                dif = max(dif, dif_inter)
            FE_inter_use = FE_inter_new
        fit_old = fit
        niter += 1
    e = _fe_adj(Y - fit, I)
    out = {
        "mu": inner["mu"],
        "fit": fit,
        "niter": niter,
        "e": e,
        "beta": beta,
        "alpha": inner["alpha"],
        "xi": inner["xi"],
        "validF": int(np.abs(inner.get("FE_inter", np.zeros(1))).sum() >= 1e-10),
    }
    if mc == 0:
        out.update(
            {"factor": inner["factor"], "lambda": inner["lambda"], "VNT": inner["VNT"]}
        )
    return out


def _inter_fe_core(
    Y: np.ndarray,
    Y0: np.ndarray,
    X: Optional[np.ndarray],
    II: np.ndarray,
    beta0: np.ndarray,
    r: int,
    force: int,
    tol: float,
    max_iter: int,
    mc: int = 0,
    lam: float = 0.0,
) -> Dict[str, Any]:
    """fect ``inter_fe_ub`` / ``inter_fe_mc`` on the untreated-cell mask ``II``."""
    T, N = Y.shape
    YY = np.where(II == 1, Y, 0.0)
    p = 0 if X is None else X.shape[2]
    if p > 0:
        keep = [k for k in range(p) if np.abs(X[:, :, k]).sum() >= 1e-5]
        XX = X[:, :, keep] if len(keep) < p else X
        b0 = beta0[keep] if (len(keep) < p and beta0.shape[0] == p) else beta0
    else:
        XX, keep, b0 = None, [], beta0
    p1 = 0 if XX is None else XX.shape[2]
    if p1 == 0:
        if force == 0 and r == 0:
            mu_Y = float(YY.sum() / II.sum())
            YY = _fe_adj(YY - mu_Y, II)
        if r > 0 or mc == 1:
            out = _fe_ad_inter_iter(
                YY, Y0, II, force, mc, r if mc == 0 else 1, 0, lam, tol, max_iter
            )
        elif force == 0:
            mu_Y = float(YY.sum() / II.sum())
            out = {
                "mu": mu_Y,
                "fit": np.full((T, N), mu_Y),
                "e": YY,
                "niter": 0,
                "alpha": np.zeros(N),
                "xi": np.zeros(T),
            }
        else:
            out = _fe_ad_iter(YY, Y0, II, force, tol, max_iter)
        out["beta"] = np.full(p, np.nan) if p > 0 else np.zeros(0)
        out["validX"] = 0
    else:
        xxinv = _xx_inv(XX)
        if r == 0 and mc == 0:
            out = _fe_ad_covar_iter(XX, xxinv, YY, Y0, II, b0, force, tol, max_iter)
        else:
            out = _fe_ad_inter_covar_iter(
                XX,
                xxinv,
                YY,
                Y0,
                II,
                b0,
                force,
                mc,
                r if mc == 0 else 1,
                0,
                lam,
                tol,
                max_iter,
            )
        if len(keep) < p:
            beta_total = np.full(p, np.nan)
            beta_total[keep] = out["beta"]
            out["beta"] = beta_total
        out["validX"] = 1
    # sigma2 / IC as in inter_fe_ub (informational)
    obs = float(II.sum())
    np_ = r * (N + T) - r**2 + p1 + 1
    if force == 1:
        np_ += (N - 1) - r
    elif force == 2:
        np_ += (T - 1) - r
    elif force == 3:
        np_ += (N - 1) + (T - 1) - 2 * r
    U = out["e"]
    out["sigma2"] = float(np.sum(U * U) / max(obs - np_, 1.0))
    out["IC"] = float(np.log(out["sigma2"]) + np_ * np.log(obs) / obs)
    return out


def _get_term(d: np.ndarray, ii: np.ndarray) -> np.ndarray:
    """fect ``get_term(type = "on")``: periods relative to treatment
    onset, 0 = last untreated period, 1 = first treated period; NaN
    when a unit never changes status. Handles reversals like fect."""
    dd = d.astype(float).copy()
    iii = ii.astype(int).copy()
    T_full = len(dd)
    first_pos = int(np.argmax(iii == 1)) if (iii == 1).any() else 0
    if first_pos != 0:
        dd = dd[first_pos:]
        iii = iii[first_pos:]
    T = len(dd)
    if (iii == 0).any() and T > 1:
        for i in range(T - 1):
            if iii[i + 1] == 0:
                dd[i + 1] = dd[i]
    if T == 1:
        term = np.array([np.nan])
    else:
        d1 = dd[:-1]
        d2 = dd[1:]
        if np.all(d1 == d2):
            term = np.full(T, np.nan)
        else:
            change_pos = np.where(d1 != d2)[0] + 2  # 1-based like R
            parts: List[np.ndarray] = []
            if dd[0] == 0:
                for i, cp in enumerate(change_pos, start=1):
                    if i == 1:
                        parts.append(np.arange(2 - cp, 1))
                    elif i % 2 == 0:
                        parts.append(np.arange(1, cp - change_pos[i - 2] + 1))
                    else:
                        parts.append(np.arange(change_pos[i - 2] - cp + 1, 1))
            else:
                for i, cp in enumerate(change_pos, start=1):
                    if i == 1:
                        parts.append(np.full(cp - 1, np.nan))
                    elif i % 2 == 0:
                        parts.append(np.arange(change_pos[i - 2] - cp + 1, 1))
                    else:
                        parts.append(np.arange(1, cp - change_pos[i - 2] + 1))
            last = change_pos[-1]
            if dd[last - 1] == 0:
                parts.append(np.full(T - last + 1, np.nan))
            else:
                parts.append(np.arange(1, T - last + 2))
            term = np.concatenate([np.asarray(p_, dtype=float) for p_ in parts])
    if first_pos != 0:
        term = np.concatenate([np.full(first_pos, np.nan), term])
    return term[:T_full]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def _fect_fit(
    Y: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    X: Optional[np.ndarray],
    method: str,
    r: int,
    lam: Optional[float],
    force: int,
    tol: float,
    max_iter: int,
) -> Dict[str, Any]:
    """One full fect fit on matrices; returns fit surface and effects."""
    T, N = Y.shape
    II = ((I == 1) & (D == 0)).astype(int)
    Y0, beta0 = _initial_fit(Y, X, II, force)
    if method == "fe":
        est = _inter_fe_core(Y, Y0, X, II, beta0, 0, force, tol, max_iter)
    elif method == "ife":
        est = _inter_fe_core(Y, Y0, X, II, beta0, r, force, tol, max_iter)
    else:
        lam_use = float(lam)
        est = _inter_fe_core(
            Y, Y0, X, II, beta0, 1, force, tol, max_iter, mc=1, lam=lam_use
        )
        # fect reports lambda relative to the largest singular value of the
        # initial residual matrix on the untreated cells.
        Y_lambda = np.where(II == 1, np.where(II == 1, Y, 0.0) - Y0, 0.0)
        s = np.linalg.svd(Y_lambda / (T * N), compute_uv=False)
        est["lambda_norm"] = lam_use / float(s.max())
    fit = est["fit"]
    eff = np.where(I == 1, Y - fit, np.nan)
    treated_obs = (D == 1) & (I == 1)
    denom = float(treated_obs.sum())
    att_avg = (
        float(np.nansum(np.where(treated_obs, eff, 0.0)) / denom)
        if denom > 0
        else np.nan
    )
    # Per-unit ATT (treated units only) and their mean.
    tr = np.where((D * I).sum(axis=0) > 0)[0]
    att_unit = []
    for j in tr:
        dj = D[:, j] * I[:, j]
        if dj.sum() > 0:
            att_unit.append(float(np.nansum(eff[:, j] * dj) / dj.sum()))
    att_avg_unit = float(np.mean(att_unit)) if att_unit else np.nan
    # Pre-treatment fit on treated units: RMSE of eff over their untreated cells.
    pre_cells = (II == 1)[:, tr]
    pre_eff = eff[:, tr][pre_cells]
    rmse = float(np.sqrt(np.nanmean(pre_eff**2))) if pre_eff.size else np.nan
    # Relative-time coding and by-period ATT.
    T_on = np.full((T, N), np.nan)
    for j in range(N):
        if I[:, j].sum() > 0:
            T_on[:, j] = _get_term(D[:, j], I[:, j])
    mask = ~np.isnan(eff) & ~np.isnan(T_on)
    periods = np.sort(np.unique(T_on[mask]))
    att_on = np.array([np.mean(eff[mask & (T_on == k)]) for k in periods])
    count_on = np.array([int(np.sum(mask & (T_on == k))) for k in periods])
    return {
        "est": est,
        "fit": fit,
        "eff": eff,
        "att_avg": att_avg,
        "att_avg_unit": att_avg_unit,
        "rmse": rmse,
        "time": periods,
        "att": att_on,
        "count": count_on,
        "T_on": T_on,
        "treated_units": tr,
    }


def fect(
    data: pd.DataFrame,
    y: str,
    treat: str,
    unit: str,
    time: str,
    covariates: Optional[Sequence[str]] = None,
    method: str = "fe",
    r: int = 0,
    lam: Optional[float] = None,
    force: str = "two-way",
    min_t0: Optional[int] = None,
    tol: float = 1e-3,
    max_iter: int = 1000,
    vce: Optional[str] = None,
    n_boot: int = 200,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Counterfactual estimators for TSCS data (fect; Liu, Wang and Xu 2024).

    Imputes the untreated potential outcome of every treated
    unit-period from a model of the *untreated* observations -- two-way
    fixed effects, interactive fixed effects, or matrix completion --
    and averages ``Y - Y(0)`` over the treated cells. Handles staggered
    adoption, multiple treated units, unbalanced panels and treatment
    reversals, with the same conventions as the R package ``fect``.

    Parameters
    ----------
    data : pd.DataFrame
        Long panel.
    y, treat, unit, time : str
        Outcome, binary treatment indicator (1 = treated in that
        period), unit id, and time id columns.
    covariates : sequence of str, optional
        Time-varying covariates entering the outcome model linearly.
    method : {'fe', 'ife', 'mc'}, default 'fe'
        Outcome model: two-way fixed effects, interactive fixed effects
        with ``r`` factors, or nuclear-norm matrix completion with
        penalty ``lam``.
    r : int, default 0
        Number of latent factors for ``method='ife'``.
    lam : float, optional
        Nuclear-norm penalty for ``method='mc'``, on fect's raw scale
        (singular values of ``E/(T*N)`` below ``lam`` are removed; the
        result records ``lambda_norm = lam / max singular value``).
    force : {'none', 'unit', 'time', 'two-way'}, default 'two-way'
        Additive fixed effects.
    min_t0 : int, optional
        Units with fewer untreated periods are dropped. Default 1 for
        ``'fe'`` and 5 otherwise, as in ``fect``.
    tol : float, default 1e-3
        Relative convergence tolerance of the EM fit (fect's default);
        use ``1e-8`` for reference comparisons.
    max_iter : int, default 1000
        Maximum EM iterations.
    vce : {None, 'bootstrap', 'jackknife'}, optional
        Resampling standard errors over units (fect ``vartype``); ``None``
        reports the point estimates only.
    n_boot : int, default 200
        Bootstrap replications.
    seed : int, optional
        Random seed for the bootstrap.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        ``.estimate`` is the ATT averaged over treated observations;
        ``.detail`` is the by-relative-period table (``fect_time`` uses
        fect's coding, 0 = last untreated period and 1 = first treated
        period; ``relative_time = fect_time - 1`` is the StatsPAI
        coding); ``.model_info`` carries ``beta``, ``mu``, ``alpha``,
        ``xi``, factors/loadings, the counterfactual matrix, ``niter``,
        ``att_avg_unit`` and the pre-treatment ``rmse``.

    Examples
    --------
    >>> import statspai as sp
    >>> panel = sp.datasets.mpdta()
    >>> panel["treated"] = ((panel["first_treat"] > 0) & (panel["year"] >= panel["first_treat"])).astype(int)
    >>> res = sp.fect(panel, y="lemp", treat="treated", unit="countyreal", time="year")
    >>> round(float(res.estimate), 4)  # doctest: +SKIP
    -0.0329

    References
    ----------
    [@liu2024practical]; [@xu2017generalized]; [@athey2021matrix].
    """
    method = str(method).lower()
    if method not in {"fe", "ife", "mc"}:
        raise ValueError("method must be 'fe', 'ife', or 'mc'")
    if method == "ife" and int(r) <= 0:
        raise ValueError("method='ife' needs r >= 1 factors")
    if method == "mc" and (lam is None or float(lam) <= 0):
        raise ValueError("method='mc' needs a positive nuclear-norm penalty lam=")
    force_key = str(force).lower()
    if force_key not in _FORCE:
        raise ValueError("force must be 'none', 'unit', 'time', or 'two-way'")
    force_int = _FORCE[force_key]
    if vce not in (None, "bootstrap", "jackknife"):
        raise ValueError("vce must be None, 'bootstrap', or 'jackknife'")
    covs = list(covariates) if covariates else []
    for c in [y, treat, unit, time, *covs]:
        if c not in data.columns:
            raise ValueError(f"column {c!r} not in data")

    df = data[[unit, time, y, treat, *covs]].copy()
    units = np.array(sorted(df[unit].unique(), key=lambda v: (str(type(v)), v)))
    times = np.array(sorted(df[time].unique()))
    N, T = len(units), len(times)
    uidx = pd.Index(units).get_indexer(df[unit])
    tidx = pd.Index(times).get_indexer(df[time])

    Y = np.full((T, N), np.nan)
    D = np.zeros((T, N))
    Y[tidx, uidx] = df[y].to_numpy(dtype=float)
    D[tidx, uidx] = df[treat].to_numpy(dtype=float)
    I = (~np.isnan(Y)).astype(int)
    D = np.where(I == 1, D, 0.0)
    if not set(np.unique(D[I == 1])).issubset({0.0, 1.0}):
        raise ValueError(f"treat={treat!r} must be a 0/1 indicator")
    X: Optional[np.ndarray] = None
    if covs:
        X = np.zeros((T, N, len(covs)))
        for k, c in enumerate(covs):
            X[tidx, uidx, k] = df[c].to_numpy(dtype=float)
        X = np.nan_to_num(X)

    if min_t0 is None:
        min_t0 = 1 if method == "fe" else 5
    II = ((I == 1) & (D == 0)).astype(int)
    T0 = II.sum(axis=0)
    ever_treated = (D * I).sum(axis=0) > 0
    keep = T0 >= min_t0
    if not keep[ever_treated].any():
        raise DataInsufficient(
            "All treated units have fewer untreated periods than min_t0.",
            recovery_hint="Lower min_t0 or use method='fe'.",
            diagnostics={
                "min_t0": int(min_t0),
                "T0_treated": T0[ever_treated].tolist(),
            },
            alternative_functions=["sp.did_imputation", "sp.callaway_santanna"],
        )
    dropped = units[~keep].tolist()
    if dropped:
        warnings.warn(
            f"fect: {len(dropped)} unit(s) with fewer than min_t0={min_t0} "
            "untreated periods were dropped, as in fect.",
            UserWarning,
            stacklevel=2,
        )
        Y, D, I = Y[:, keep], D[:, keep], I[:, keep]
        if X is not None:
            X = X[:, keep, :]
        units = units[keep]
        N = len(units)
    Yz = np.where(I == 1, Y, 0.0)

    main = _fect_fit(
        Yz, D, I, X, method, int(r), lam, force_int, float(tol), int(max_iter)
    )
    att = main["att_avg"]
    est = main["est"]

    # ------------------------------------------------------------------
    # Inference (resampling over units, as in fect)
    # ------------------------------------------------------------------
    se_val: Optional[float] = None
    ci: Optional[Tuple[float, float]] = None
    pvalue: Optional[float] = None
    att_by_period_se = None
    boot_draws: List[float] = []
    tr = main["treated_units"]
    co = np.array([j for j in range(N) if j not in set(tr.tolist())])
    if vce is not None:
        rng = np.random.default_rng(seed)
        reps: List[float] = []
        period_reps: List[Dict[float, float]] = []
        if vce == "bootstrap":
            draws = n_boot
        else:
            draws = N
        for b in range(draws):
            if vce == "bootstrap":
                # fect: resample treated and control units separately, with replacement.
                idx = np.concatenate(
                    [
                        rng.choice(tr, size=len(tr), replace=True),
                        rng.choice(co, size=len(co), replace=True)
                        if len(co)
                        else np.zeros(0, int),
                    ]
                ).astype(int)
            else:
                idx = np.array([j for j in range(N) if j != b])
            Yb, Db, Ib = Yz[:, idx], D[:, idx], I[:, idx]
            Xb = X[:, idx, :] if X is not None else None
            if ((Db * Ib).sum(axis=0) > 0).sum() == 0 or (Db == 0).sum() == 0:
                continue
            try:
                fb = _fect_fit(
                    Yb,
                    Db,
                    Ib,
                    Xb,
                    method,
                    int(r),
                    lam,
                    force_int,
                    float(tol),
                    int(max_iter),
                )
            except (np.linalg.LinAlgError, ValueError, DataInsufficient):
                continue
            if np.isfinite(fb["att_avg"]):
                reps.append(fb["att_avg"])
                period_reps.append(dict(zip(fb["time"].tolist(), fb["att"].tolist())))
        boot_draws = reps
        if len(reps) >= 2:
            reps_arr = np.asarray(reps)
            if vce == "bootstrap":
                se_val = float(np.std(reps_arr, ddof=1))
            else:
                n_j = len(reps_arr)
                se_val = float(
                    np.sqrt((n_j - 1) / n_j * np.sum((reps_arr - reps_arr.mean()) ** 2))
                )
            z = stats.norm.ppf(1 - alpha / 2)
            ci = (att - z * se_val, att + z * se_val)
            pvalue = (
                float(2 * (1 - stats.norm.cdf(abs(att / se_val))))
                if se_val > 0
                else 0.0
            )
            att_by_period_se = []
            for k in main["time"].tolist():
                vals = np.array([p_[k] for p_ in period_reps if k in p_])
                if len(vals) >= 2:
                    if vce == "bootstrap":
                        att_by_period_se.append(float(np.std(vals, ddof=1)))
                    else:
                        n_j = len(vals)
                        att_by_period_se.append(
                            float(
                                np.sqrt(
                                    (n_j - 1) / n_j * np.sum((vals - vals.mean()) ** 2)
                                )
                            )
                        )
                else:
                    att_by_period_se.append(np.nan)
        else:
            warnings.warn(
                "fect: fewer than two successful resamples; standard errors are "
                "not reported.",
                RuntimeWarning,
                stacklevel=2,
            )

    detail = pd.DataFrame(
        {
            "fect_time": main["time"].astype(int),
            "relative_time": main["time"].astype(int) - 1,
            "att": main["att"],
            "count": main["count"],
        }
    )
    if att_by_period_se is not None:
        detail["se"] = att_by_period_se

    model_info: Dict[str, Any] = {
        "estimator": f"fect ({method})",
        "method": method,
        "force": force_key,
        "r": int(r) if method == "ife" else 0,
        "lambda": float(lam) if method == "mc" else None,
        "lambda_norm": est.get("lambda_norm"),
        "tol": float(tol),
        "max_iter": int(max_iter),
        "niter": int(est.get("niter", 0)),
        "converged": bool(est.get("niter", 0) <= max_iter),
        "beta": pd.Series(np.asarray(est.get("beta", np.zeros(0))), index=covs)
        if covs
        else None,
        "mu": float(est["mu"]),
        "alpha": pd.Series(np.asarray(est["alpha"]), index=units),
        "xi": pd.Series(np.asarray(est["xi"]), index=times),
        "factors": est.get("factor"),
        "loadings": est.get("lambda"),
        "sigma2": est.get("sigma2"),
        "IC": est.get("IC"),
        "att_avg_unit": main["att_avg_unit"],
        "pre_treatment_rmse": main["rmse"],
        "counterfactual": pd.DataFrame(main["fit"], index=times, columns=units),
        "effects": pd.DataFrame(main["eff"], index=times, columns=units),
        "relative_period": pd.DataFrame(main["T_on"], index=times, columns=units),
        "n_units": int(N),
        "n_treated_units": int(len(tr)),
        "n_control_units": int(len(co)),
        "n_periods": int(T),
        "dropped_units": dropped,
        "min_t0": int(min_t0),
        "se_method": vce,
        "n_boot_success": len(boot_draws),
        "boot_draws": boot_draws if boot_draws else None,
        "relative_time_note": (
            "fect_time follows fect: 0 is the last untreated period and 1 "
            "the first treated period; relative_time = fect_time - 1 is the "
            "StatsPAI convention (0 = first treated period)."
        ),
    }

    result = CausalResult(
        method=f"fect {method} (Liu, Wang and Xu 2024)",
        estimand="ATT",
        estimate=float(att),
        se=se_val,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=int(I.sum()),
        detail=detail,
        model_info=model_info,
        _citation_key="fect",
    )
    return result
