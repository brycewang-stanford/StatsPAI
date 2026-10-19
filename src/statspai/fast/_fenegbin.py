"""Negative binomial (NB2) regression with high-dimensional fixed effects.

The engine behind ``sp.feglm(family='negbin')``. The model is
``y ~ NB(mu, theta)`` with ``log mu = x'b + alpha_1 + alpha_2 + ...`` and
variance ``mu + mu^2 / theta``; ``b``, the effects and ``theta`` are
estimated jointly by maximum likelihood (an unconditional fixed-effects
negative binomial, the model ``fixest::fenegbin`` fits, not Stata's
conditional ``xtnbreg, fe``).

Estimation alternates two steps until the log likelihood stops changing:
one Fisher-scoring (IRLS) step for ``b`` and the effects at the current
``theta``, on the fixed-effect residualised working response, as
``sp.fast.fepois`` does; then Newton steps on ``log theta`` with ``mu``
held fixed. The fixed point is the joint maximum.

Inference uses the observed information of ``(b, theta)`` with the effects
profiled out: with ``W_i = -d2l_i / deta^2`` and
``h_i = -d2l_i / deta dtheta``, the blocks are ``X~' W X~``, ``X~' h`` and
``sum -d2l_i / dtheta^2 - |P v|_W^2`` (``X~`` and ``v~ = v - P v`` the
``W``-weighted residuals of ``X`` and ``v = h / W`` on the effects).
``theta`` is reported as a coefficient ``.theta`` with its standard error.
Robust and clustered variances are sandwiches with scores
``dl/deta x~`` for ``b`` and ``dl/dtheta`` for ``theta``. Small-sample
factors are those of ``sp.fast.fepois``; ``theta`` is not counted in
``K``, except that with every effect nested in the clusters (``K`` would
otherwise be the coefficients plus one) it is, which is what fixest 0.14
reports.
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
from ._validation import require_identified_regressors as _require_identified


@dataclass
class NegbinFit:
    """A fitted FE negative binomial (internal; ``sp.feglm`` reports it)."""

    coef_names: List[str]
    coef_vec: np.ndarray  # b, then theta
    vcov_matrix: np.ndarray
    theta: float
    n_obs: int
    n_kept: int
    n_dropped_separation: int
    log_likelihood: float
    deviance: float
    iterations: int
    converged: bool
    fe_cardinality: List[int]


def _loglik_terms(y: np.ndarray, mu: np.ndarray, th: float) -> np.ndarray:
    return (
        special.gammaln(y + th)
        - special.gammaln(th)
        - special.gammaln(y + 1.0)
        + th * np.log(th)
        + special.xlogy(y, mu)
        - (y + th) * np.log(th + mu)
    )


def _theta_derivs(
    y: np.ndarray, mu: np.ndarray, w: np.ndarray, th: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-observation ``dl/dtheta`` and ``d2l/dtheta2`` (weighted)."""
    g = w * (
        special.digamma(y + th)
        - special.digamma(th)
        + np.log(th)
        + 1.0
        - np.log(th + mu)
        - (y + th) / (th + mu)
    )
    h = w * (
        special.polygamma(1, y + th)
        - special.polygamma(1, th)
        + 1.0 / th
        - 1.0 / (th + mu)
        - (mu - y) / (th + mu) ** 2
    )
    return g, h


def _update_theta(y: np.ndarray, mu: np.ndarray, w: np.ndarray, th: float) -> float:
    """Maximise the likelihood over ``theta`` at fixed ``mu`` (Newton on
    ``log theta``, steps capped, ascent safeguarded)."""
    ll = float(w @ _loglik_terms(y, mu, th))
    for _ in range(100):
        g, h = _theta_derivs(y, mu, w, th)
        gs, hs = float(g.sum()), float(h.sum())
        gl = gs * th  # d l / d log theta
        hl = hs * th * th + gs * th
        step = -gl / hl if hl < 0 else np.sign(gl) * 0.5
        step = float(np.clip(step, -2.0, 2.0))
        for _half in range(30):
            cand = th * np.exp(step)
            ll_c = float(w @ _loglik_terms(y, mu, cand))
            if np.isfinite(ll_c) and ll_c >= ll - 1e-12 * abs(ll):
                break
            step *= 0.5
        if abs(cand - th) <= 1e-12 * th:
            return float(cand)
        th, ll = float(cand), ll_c
        if th > 1e12:
            raise NumericalInstability(
                "feglm(family='negbin'): theta diverges (no overdispersion "
                "left once the fixed effects are in); the Poisson model "
                "is the limit.",
                recovery_hint="Use sp.fepois for these data.",
                diagnostics={"theta": th},
            )
    return th


def _drop_perfect_fit(y: np.ndarray, codes: List[np.ndarray]) -> np.ndarray:
    """Rows outside every fixed-effect group that its effect fits perfectly:
    outcomes all zero (the effect is minus infinity) or a single row,
    repeated across dimensions until none is left (fixest's
    ``fixef.rm = 'perfect_fit'``). The estimates do not change; the
    degrees of freedom and so the standard errors do."""
    keep = np.ones(y.shape[0], dtype=bool)
    while True:
        changed = False
        for c in codes:
            ck = c[keep]
            G = int(c.max()) + 1 if c.size else 0
            s = np.bincount(ck, weights=y[keep], minlength=G)
            n_g = np.bincount(ck, minlength=G)
            bad = (n_g > 0) & ((s <= 0) | (n_g == 1))
            if bad.any():
                drop = bad[ck]
                if drop.any():
                    idx = np.flatnonzero(keep)
                    keep[idx[drop]] = False
                    changed = True
        if not changed:
            return keep


def _demean(
    arr: np.ndarray,
    codes: List[np.ndarray],
    counts: List[np.ndarray],
    weights: np.ndarray,
    *,
    tol: float,
    maxiter: int,
) -> np.ndarray:
    """Weighted residuals on the effects; unconverged columns are finished
    by the sparse direct solve."""
    from ._fe_direct import residualise_codes
    from .fepois import _weighted_ap_demean

    out, _, ok = _weighted_ap_demean(
        arr, codes, counts, weights, max_iter=maxiter, tol=tol
    )
    if not ok:
        out2 = out.reshape(arr.shape[0], -1)
        _, done = residualise_codes(out2, codes, weights)
        if not done:
            raise NumericalInstability(
                "feglm(family='negbin'): the fixed-effect residualisation "
                "did not converge.",
                recovery_hint="Use coarser fixed effects.",
            )
        out = out2.reshape(out.shape)
    return out


def fenegbin_fit(
    lhs: str,
    rhs: List[str],
    fe: List[str],
    data: pd.DataFrame,
    *,
    vcov: str = "iid",
    cluster: Optional[Any] = None,
    weights: Optional[str] = None,
    tol: float = 1e-12,
    maxiter: int = 200,
    fe_tol: float = 1e-12,
    fe_maxiter: int = 10_000,
) -> Tuple[NegbinFit, Dict[str, Any]]:
    """Fit ``lhs ~ rhs | fe`` (plan already validated by the caller)."""
    n_obs = len(data)
    y_all = data[lhs].to_numpy(dtype=np.float64)
    if (y_all < 0).any() or not np.allclose(y_all, np.round(y_all)):
        raise MethodIncompatibility(
            f"feglm(family='negbin'): the outcome {lhs!r} must be a "
            "non-negative count.",
            recovery_hint="Pass a column of non-negative integers.",
        )
    X_all = data[rhs].to_numpy(dtype=np.float64) if rhs else np.empty((n_obs, 0))
    names = list(rhs)
    if not fe:
        X_all = np.column_stack([np.ones(n_obs), X_all])
        names = ["(Intercept)"] + names
    w_all = (
        data[weights].to_numpy(dtype=np.float64)
        if weights is not None
        else np.ones(n_obs)
    )
    codes_raw = [pd.factorize(data[c], sort=False)[0].astype(np.int64) for c in fe]
    keep = _drop_perfect_fit(y_all, codes_raw) if fe else np.ones(n_obs, bool)
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise DataInsufficient(
            "feglm(family='negbin'): every fixed-effect group has only zero "
            "outcomes or one row; nothing is left to estimate on.",
            recovery_hint="Use coarser fixed effects.",
        )
    n_sep = n_obs - n_kept
    y, X, w = y_all[keep], X_all[keep], w_all[keep]
    codes = [pd.factorize(c[keep], sort=False)[0].astype(np.int64) for c in codes_raw]
    fe_card = [int(c.max()) + 1 for c in codes]
    counts = [np.bincount(c, minlength=G).astype(float) for c, G in zip(codes, fe_card)]
    n, p = X.shape
    if p == 0:
        raise MethodIncompatibility(
            "feglm(family='negbin'): no regressors.",
            recovery_hint="Add at least one regressor.",
        )

    def dm(a: np.ndarray, wt: np.ndarray) -> np.ndarray:
        if not fe:
            return a
        return _demean(a, codes, counts, wt, tol=fe_tol, maxiter=fe_maxiter)

    # IRLS for b at theta, alternating with theta given mu.
    ybar = float(np.average(y, weights=w))
    mu = (y + ybar) / 2.0
    eta = np.log(mu)
    v_y = float(np.average((y - ybar) ** 2, weights=w))
    th = ybar * ybar / (v_y - ybar) if v_y > ybar * 1.01 else 10.0
    z_fe = X_fe = None
    beta = np.zeros(p)
    beta_old = np.full(p, np.inf)
    ll_old = -np.inf
    converged = False
    it = 0
    for it in range(1, maxiter + 1):
        W = w * th * mu / (th + mu)
        z = eta + (y - mu) / mu
        zt = dm(z if z_fe is None else z - z_fe, W)
        Xt = dm(X if X_fe is None else X - X_fe, W)
        if fe:
            z_fe, X_fe = z - zt, X - Xt
        XtW = Xt.T * W
        try:
            beta = np.linalg.solve(XtW @ Xt, XtW @ zt)
        except np.linalg.LinAlgError as exc:
            raise NumericalInstability(
                f"feglm(family='negbin'): singular normal equations ({exc}).",
                recovery_hint="Drop regressors the fixed effects absorb.",
                diagnostics={"collinear": names},
            ) from exc
        eta = np.clip(z - (zt - Xt @ beta), -700.0, 700.0)
        mu = np.exp(eta)
        th_old = th
        th = _update_theta(y, mu, w, th)
        ll = float(w @ _loglik_terms(y, mu, th))
        # The likelihood is flat at the top: a change of 1e-12 in it leaves
        # the coefficients ~1e-6 off, so the parameters must settle too.
        settled = np.all(np.abs(beta - beta_old) <= 1e-10 * (1.0 + np.abs(beta)))
        if (
            abs(ll - ll_old) <= tol * (1.0 + abs(ll))
            and settled
            and abs(th - th_old) <= 1e-10 * th
        ):
            converged = True
            break
        ll_old = ll
        beta_old = beta
    _require_identified(X, Xt, names, context="feglm", weights=W, tol_rel=1e-9)

    # Observed information of (b, theta), effects profiled out.
    Wo = w * (y + th) * th * mu / (th + mu) ** 2
    h = -w * (y - mu) * mu / (th + mu) ** 2
    v = h / Wo
    A = dm(np.column_stack([X, v]), Wo)
    Xo, vt = A[:, :p], A[:, p]
    g_th, h_th = _theta_derivs(y, mu, w, th)
    H = np.empty((p + 1, p + 1))
    H[:p, :p] = (Xo.T * Wo) @ Xo
    H[:p, p] = H[p, :p] = Xo.T @ (Wo * vt)
    H[p, p] = -float(h_th.sum()) - float(v @ (Wo * v)) + float(vt @ (Wo * vt))
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded above
        raise NumericalInstability(
            f"feglm(family='negbin'): singular information matrix ({exc}).",
            recovery_hint="Drop regressors the fixed effects absorb.",
        ) from exc

    from .feols import _fixest_cluster_fe_dof_codes, _fixest_fe_dof

    fe_dof = _fixest_fe_dof(fe_card) if fe else 0
    df_k = p + fe_dof
    s_eta = w * th * (y - mu) / (th + mu)
    S = np.column_stack([Xo * s_eta[:, None], g_th])
    n_clusters = 0
    df_k_c = df_k
    if vcov == "iid":
        V = Hinv * ((n - 1.0) / max(n - df_k, 1))
    elif vcov == "hc1":
        V = Hinv @ (S.T @ S) @ Hinv * (n / max(n - df_k, 1))
    else:
        from .inference import _cr1_meat

        cl_cols = [cluster] if isinstance(cluster, str) else list(cluster)
        dims: List[Tuple[np.ndarray, int]] = []
        for col in cl_cols:
            cc, lv = pd.factorize(data[col].to_numpy()[keep], sort=False)
            dims.append((cc.astype(np.int64), len(lv)))
        if len(dims) == 2:
            (ca, ga), (cb, gb) = dims
            inter, lab = pd.factorize(ca * gb + cb, sort=False)
            dims.append((inter.astype(np.int64), len(lab)))
        n_clusters = min(g for _, g in dims[: len(cl_cols)])
        if n_clusters < 2:
            raise DataInsufficient(
                f"feglm(family='negbin'): need at least 2 clusters, got {n_clusters}.",
                recovery_hint="Cluster on a variable with more levels.",
            )
        extra = (
            _fixest_cluster_fe_dof_codes(codes, fe_card, [c for c, _ in dims])
            if fe
            else 0
        )
        all_nested = bool(fe) and all(
            any(_nested(c, G, cc) for cc, _ in dims) for c, G in zip(codes, fe_card)
        )
        df_k_c = p + extra + (1 if all_nested else 0)
        meat = np.zeros((p + 1, p + 1))
        signs = (1.0, 1.0, -1.0) if len(dims) == 3 else (1.0,)
        for (cc, gc), sign in zip(dims, signs):
            meat += sign * _cr1_meat(S, np.ones(n), cc, gc)
        V = (Hinv @ meat @ Hinv) * (
            (n_clusters / (n_clusters - 1.0)) * ((n - 1.0) / max(n - df_k_c, 1))
        )
    if not converged:
        warnings.warn(
            f"feglm(family='negbin'): stopped after {it} iterations without "
            f"meeting tol={tol}.",
            ConvergenceWarning,
            stacklevel=3,
        )
    ll = float(w @ _loglik_terms(y, mu, th))
    dev = float(
        2.0
        * np.sum(
            w
            * (
                special.xlogy(y, y / np.where(mu > 0, mu, 1.0))
                - (y + th) * np.log((y + th) / (mu + th))
            )
        )
    )
    fit = NegbinFit(
        coef_names=names + [".theta"],
        coef_vec=np.concatenate([beta, [th]]),
        vcov_matrix=V,
        theta=float(th),
        n_obs=n_obs,
        n_kept=n_kept,
        n_dropped_separation=n_sep,
        log_likelihood=ll,
        deviance=dev,
        iterations=it,
        converged=converged,
        fe_cardinality=fe_card,
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


def _nested(codes: np.ndarray, G: int, cluster_codes: np.ndarray) -> bool:
    from .feols import _codes_nested

    return _codes_nested(codes, G, cluster_codes)


__all__ = ["NegbinFit", "fenegbin_fit"]
