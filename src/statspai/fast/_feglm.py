"""Binomial GLM (logit / probit) with high-dimensional fixed effects.

The engine behind ``sp.feglm(family='logit' | 'probit')`` on its native
path. Iteratively reweighted least squares on the fixed-effect
residualised working response, as ``fixest::feglm`` and ``sp.fast.fepois``
do: each iteration residualises ``z`` and ``X`` on the fixed effects under
the working weights (the Rust weighted kernel when ``statspai-hdfe`` is
installed), warm-started from the previous fixed-effect component, and
solves the weighted normal equations.

Conventions follow ``fixest::feglm``:

* observations in a fixed-effect group whose outcomes are all 0 or all 1
  are dropped, repeatedly across dimensions (fixest's ``perfect_fit``; for a
  binary outcome every singleton is such a group);
* the variance is Fisher scoring's: bread ``(X~' W X~)^{-1}`` with
  ``W = w dmu^2 / V(mu)``, scores ``w (y - mu) dmu / V(mu) x~``;
* small-sample factors as in ``sp.fast.fepois`` (``ssc='fixest'``):
  ``(n - 1)/(n - K)`` for iid, ``n/(n - K)`` for hetero,
  ``G/(G - 1) (n - 1)/(n - K_c)`` for one-way clusters, with ``K`` counting
  the absorbed effects and ``K_c`` leaving out those nested in the clusters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import special

from ..exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)
from ._validation import require_identified_regressors as _require_identified_regressors

FAMILIES = ("logit", "probit")


@dataclass
class GlmFit:
    """A fitted binomial FE-GLM (internal; ``sp.feglm`` reports it)."""

    coef_names: List[str]
    coef_vec: np.ndarray
    vcov_matrix: np.ndarray
    n_obs: int
    n_kept: int
    n_dropped_separation: int
    deviance: float
    log_likelihood: float
    iterations: int
    converged: bool
    fe_cardinality: List[int]
    fe_kernel: str


def _mean(family: str, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``mu`` and ``dmu/deta``."""
    if family == "logit":
        mu = special.expit(eta)
        return mu, mu * (1.0 - mu)
    mu = special.ndtr(eta)
    return mu, np.exp(-0.5 * eta * eta) / np.sqrt(2.0 * np.pi)


def _deviance(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    m = np.clip(mu, 1e-300, 1.0 - 1e-16)
    ll = special.xlogy(y, m) + special.xlog1py(1.0 - y, -m)
    sat = special.xlogy(y, y) + special.xlog1py(1.0 - y, -y)
    return float(2.0 * np.sum(w * (sat - ll)))


def _drop_perfect_fit(y: np.ndarray, codes: List[np.ndarray]) -> np.ndarray:
    """Rows outside every fixed-effect group whose outcomes are constant."""
    keep = np.ones(y.shape[0], dtype=bool)
    while True:
        changed = False
        for c in codes:
            ck, yk = c[keep], y[keep]
            G = int(c.max()) + 1 if c.size else 0
            n_g = np.bincount(ck, minlength=G)
            s_g = np.bincount(ck, weights=yk, minlength=G)
            bad = (n_g > 0) & ((s_g <= 0.0) | (s_g >= n_g))
            if bad.any():
                drop = bad[ck]
                if drop.any():
                    idx = np.flatnonzero(keep)
                    keep[idx[drop]] = False
                    changed = True
        if not changed:
            return keep


def feglm_fit(
    lhs: str,
    rhs: List[str],
    fe: List[str],
    data: pd.DataFrame,
    *,
    family: str,
    vcov: str = "iid",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    tol: float = 1e-10,
    maxiter: int = 50,
    fe_tol: float = 1e-10,
    fe_maxiter: int = 10_000,
) -> Tuple[GlmFit, Dict[str, Any]]:
    """Fit ``lhs ~ rhs | fe`` (plan already validated by the caller)."""
    if family not in FAMILIES:
        raise MethodIncompatibility(f"feglm: family={family!r}; native: {FAMILIES}")
    n_obs = len(data)
    y_all = data[lhs].to_numpy(dtype=np.float64)
    if ((y_all < 0) | (y_all > 1)).any():
        raise MethodIncompatibility(
            f"feglm(family={family!r}): the outcome {lhs!r} must lie in [0, 1]."
        )
    add_intercept = not fe
    X_all = data[rhs].to_numpy(dtype=np.float64)
    if add_intercept:
        X_all = np.column_stack([np.ones(n_obs), X_all])
        names = ["(Intercept)"] + list(rhs)
    else:
        names = list(rhs)
    w_all = (
        data[weights].to_numpy(dtype=np.float64)
        if weights is not None
        else np.ones(n_obs, dtype=np.float64)
    )

    codes_raw: List[np.ndarray] = []
    for col in fe:
        c, _ = pd.factorize(data[col], sort=False)
        codes_raw.append(c.astype(np.int64))
    keep = _drop_perfect_fit(y_all, codes_raw) if fe else np.ones(n_obs, bool)
    n_kept = int(keep.sum())
    n_sep = n_obs - n_kept
    if n_kept == 0:
        raise DataInsufficient(
            "feglm: every observation is in a fixed-effect group whose "
            "outcomes are all 0 or all 1, so nothing is left to estimate on.",
            recovery_hint="Use coarser fixed effects.",
        )
    if n_sep:
        y, X, w = y_all[keep], X_all[keep], w_all[keep]
        codes = []
        for c in codes_raw:
            d, _ = pd.factorize(c[keep], sort=False)
            codes.append(d.astype(np.int64))
    else:
        y, X, w = y_all, X_all, w_all
        codes = codes_raw
    fe_card = [int(c.max()) + 1 for c in codes]
    counts = [
        np.bincount(c, minlength=G).astype(np.float64) for c, G in zip(codes, fe_card)
    ]
    n, p = X.shape

    rust = _rust_kernels()
    if fe and rust is not None and hasattr(rust, "feglm_irls"):
        res = rust.feglm_irls(
            np.ascontiguousarray(y),
            np.asfortranarray(X),
            [np.ascontiguousarray(c, dtype=np.int64) for c in codes],
            np.asarray(fe_card, dtype=np.int64),
            np.ascontiguousarray(w),
            {
                "maxiter": int(maxiter),
                "tol": float(tol),
                "fe_tol": float(fe_tol),
                "fe_maxiter": int(fe_maxiter),
                "eta_clip": 30.0 if family == "logit" else 8.0,
            },
            family,
        )
        if bool(res["singular"]):
            raise NumericalInstability(
                "feglm: the weighted normal equations are singular: a regressor "
                "is collinear with the fixed effects or with the other regressors.",
                recovery_hint="Drop regressors the fixed effects absorb.",
                diagnostics={"collinear": names},
            )
        if not bool(res["fe_converged"]):
            warnings.warn(
                "feglm: the fixed-effect residualisation did not converge "
                f"within {fe_maxiter} sweeps in the last IRLS iteration; the "
                "estimates may be inaccurate.",
                ConvergenceWarning,
                stacklevel=3,
            )
        beta = np.asarray(res["beta"], dtype=np.float64)
        Xt = np.asarray(res["x_tilde_flat"], dtype=np.float64).reshape(p, n).T
        W = np.asarray(res["w"], dtype=np.float64)
        eta = np.asarray(res["eta"], dtype=np.float64)
        deviance = float(res["deviance"])
        it = int(res["iters"])
        converged = bool(res["converged"])
        kernel = "rust"
    else:
        beta, Xt, W, eta, deviance, it, converged = _irls_python(
            family, y, X, w, fe, codes, counts, names, tol, maxiter, fe_tol, fe_maxiter
        )
        kernel = (
            ("rust" if _rust_loaded() and len(fe) >= 2 else "numpy") if fe else "none"
        )
    return _finish(
        family,
        y,
        X,
        w,
        Xt,
        W,
        eta,
        beta,
        deviance,
        it,
        converged,
        kernel,
        data,
        cluster,
        vcov,
        codes,
        fe_card,
        names,
        n_obs,
        n_kept,
        n_sep,
        keep,
        fe_tol,
        tol,
    )


def _irls_python(
    family: str,
    y: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
    fe: List[str],
    codes: List[np.ndarray],
    counts: List[np.ndarray],
    names: List[str],
    tol: float,
    maxiter: int,
    fe_tol: float,
    fe_maxiter: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """The IRLS loop without the compiled kernel."""
    from .fepois import _weighted_ap_demean

    n, p = X.shape
    # IRLS (Fisher scoring) on the residualised working response.
    mu = (w * y + 0.5) / (w + 1.0)
    eta = special.logit(mu) if family == "logit" else special.ndtri(mu)
    clip = 30.0 if family == "logit" else 8.0
    deviance = np.inf
    converged = False
    z_fe = X_fe = None
    beta = np.zeros(p)
    it = 0
    for it in range(1, maxiter + 1):
        mu, dmu = _mean(family, eta)
        var = np.clip(mu * (1.0 - mu), 1e-300, None)
        W = w * dmu * dmu / var
        z = eta + (y - mu) / np.where(dmu > 0, dmu, 1e-300)
        if fe:
            zt, _, ok_z = _weighted_ap_demean(
                z if z_fe is None else z - z_fe,
                codes,
                counts,
                W,
                max_iter=fe_maxiter,
                tol=fe_tol,
            )
            Xt, _, ok_x = _weighted_ap_demean(
                X if X_fe is None else X - X_fe,
                codes,
                counts,
                W,
                max_iter=fe_maxiter,
                tol=fe_tol,
            )
            if not (ok_z and ok_x):
                warnings.warn(
                    "feglm: the fixed-effect residualisation did not converge "
                    f"within {fe_maxiter} sweeps; the estimates may be inaccurate.",
                    ConvergenceWarning,
                    stacklevel=3,
                )
            z_fe, X_fe = z - zt, X - Xt
        else:
            zt, Xt = z, X
        XtW = Xt.T * W
        try:
            beta_new = np.linalg.solve(XtW @ Xt, XtW @ zt)
        except np.linalg.LinAlgError as exc:
            raise NumericalInstability(
                f"feglm: the weighted normal equations are singular ({exc}).",
                recovery_hint="Drop regressors the fixed effects absorb.",
                diagnostics={"collinear": names},
            ) from exc
        eta_new = np.clip(z - (zt - Xt @ beta_new), -clip, clip)
        dev_new = _deviance(y, _mean(family, eta_new)[0], w)
        halvings = 0
        while dev_new > deviance and np.isfinite(deviance) and halvings < 10:
            eta_new = 0.5 * (eta_new + eta)
            dev_new = _deviance(y, _mean(family, eta_new)[0], w)
            halvings += 1
        rel = abs(dev_new - deviance) / (0.1 + abs(dev_new))
        eta, beta, deviance = eta_new, beta_new, dev_new
        if rel < tol:
            converged = True
            break
    return beta, Xt, W, eta, float(deviance), it, converged


def _finish(
    family: str,
    y: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
    Xt: np.ndarray,
    W: np.ndarray,
    eta: np.ndarray,
    beta: np.ndarray,
    deviance: float,
    it: int,
    converged: bool,
    kernel: str,
    data: pd.DataFrame,
    cluster: Optional[str],
    vcov: str,
    codes: List[np.ndarray],
    fe_card: List[int],
    names: List[str],
    n_obs: int,
    n_kept: int,
    n_sep: int,
    keep: np.ndarray,
    fe_tol: float,
    tol: float,
) -> Tuple[GlmFit, Dict[str, Any]]:
    """Identification check, variance and the fit record."""
    from .feols import _codes_nested, _fixest_fe_dof
    from .inference import _cr1_from_codes

    n, p = X.shape
    _require_identified_regressors(
        X, Xt, names, context="feglm", weights=W, tol_rel=fe_tol
    )

    # Variance at the final iterate.
    mu, dmu = _mean(family, eta)
    var = np.clip(mu * (1.0 - mu), 1e-300, None)
    bread = np.linalg.inv((Xt.T * W) @ Xt)
    fe_dof = _fixest_fe_dof(fe_card)
    df_k = p + fe_dof
    score_w = w * (y - mu) * dmu / var
    n_clusters = 0
    if vcov == "iid":
        V = bread * ((n - 1) / max(n - df_k, 1))
        df_k_c = df_k
    elif vcov == "hc1":
        S = score_w[:, None] * Xt
        V = bread @ (S.T @ S) @ bread * (n / max(n - df_k, 1))
        df_k_c = df_k
    else:
        cl_cols = [cluster] if isinstance(cluster, str) else list(cluster)
        dims = []
        for col in cl_cols:
            raw = data[col].to_numpy()
            cl_codes, levels = pd.factorize(raw[keep] if n_sep else raw, sort=False)
            dims.append((cl_codes, len(levels)))
        if len(dims) == 2:
            # Two-way: M_a + M_b - M_ab with the smallest G (fixest).
            (ca, ga), (cb, gb) = dims
            inter, levels_ab = pd.factorize(ca.astype(np.int64) * gb + cb, sort=False)
            dims.append((inter, len(levels_ab)))
        n_clusters = min(g for _, g in dims[:2])
        if n_clusters < 2:
            raise DataInsufficient(f"feglm: need at least 2 clusters, got {n_clusters}")
        free = [
            G
            for c, G in zip(codes, fe_card)
            if not any(_codes_nested(c, G, cc) for cc, _ in dims)
        ]
        extra = (sum(free) - (len(free) - 1)) if free else (1 if fe_card else 0)
        df_k_c = p + extra
        if len(dims) == 1:
            V = _cr1_from_codes(
                Xt,
                score_w,
                dims[0][0],
                n_clusters,
                weights=np.ones(n),
                bread=bread,
                extra_df=extra,
            )
        else:
            from .inference import _cr1_meat

            meat = np.zeros((p, p))
            for (cc, gc), sign in zip(dims, (1.0, 1.0, -1.0)):
                meat += sign * _cr1_meat(Xt, score_w, cc, gc)
            V = (bread @ meat @ bread) * (
                (n_clusters / (n_clusters - 1.0)) * ((n - 1.0) / max(n - p - extra, 1))
            )
    m = np.clip(mu, 1e-300, 1.0 - 1e-16)
    loglik = float(np.sum(w * (special.xlogy(y, m) + special.xlog1py(1.0 - y, -m))))
    if not converged:
        warnings.warn(
            f"feglm: IRLS stopped after {it} iterations without meeting tol={tol}.",
            ConvergenceWarning,
            stacklevel=3,
        )
    fit = GlmFit(
        coef_names=names,
        coef_vec=beta,
        vcov_matrix=V,
        n_obs=n_obs,
        n_kept=n_kept,
        n_dropped_separation=n_sep,
        deviance=float(deviance),
        log_likelihood=loglik,
        iterations=it,
        converged=converged,
        fe_cardinality=fe_card,
        fe_kernel=kernel,
    )
    arrays = {
        "keep_mask": keep,
        "y": y,
        "mu": mu,
        "eta": eta,
        "n_clusters": n_clusters,
        "df_k": int(df_k),
        "df_k_cluster": int(df_k_c),
    }
    return fit, arrays


def _rust_loaded() -> bool:
    from .._rust_backend import HAS_RUST

    return bool(HAS_RUST)


def _rust_kernels() -> Any:
    from .._rust_backend import rust

    return rust


__all__ = ["FAMILIES", "GlmFit", "feglm_fit"]
