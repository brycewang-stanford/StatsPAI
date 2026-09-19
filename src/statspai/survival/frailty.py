"""Cox proportional hazards with shared (cluster) frailty.

Extends the standard Cox model to account for unobserved heterogeneity
(frailty) within clusters. The model is:

    h(t | X_ij, z_i) = z_i · h₀(t) · exp(X_ij β)

where z_i is a multiplicative frailty term, assumed iid gamma with mean 1
and variance θ (the parametrisation of R ``coxph(... + frailty(id))`` and
Stata ``stcox, shared(id)``). When θ → 0 the frailties collapse to 1 and
the model reduces to the standard Cox model.

Estimation follows the penalised partial likelihood approach
(Therneau & Grambsch 2000): for fixed θ, (β, log z) maximise the Cox
partial likelihood minus the gamma penalty ``(1/θ) Σ (exp(w_g) - w_g)`` by
full Newton-Raphson, and θ maximises the resulting integrated (marginal)
log-likelihood, which for the gamma frailty is the penalised fit's partial
log-likelihood plus a closed-form correction.

References
----------
Therneau, T.M. & Grambsch, P.M. (2000). *Modeling Survival Data:
  Extending the Cox Model*. Springer.
Duchateau, L. & Janssen, P. (2008). *The Frailty Model*. Springer.
[@therneau2000modeling]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from .._result_serialize import ResultProtocolMixin
from .models import _parse_formula


@dataclass
class FrailtyResult(ResultProtocolMixin):
    beta: np.ndarray
    se: np.ndarray
    var_names: List[str]
    theta: float
    frailties: np.ndarray
    cluster_ids: np.ndarray
    log_likelihood: float
    n: int
    n_events: int
    n_clusters: int
    concordance: float
    log_frailties: Optional[np.ndarray] = None
    loglik_cox: float = float("nan")
    ties: str = "efron"
    theta_fixed: bool = False

    @property
    def lr_theta0(self) -> Tuple[float, float]:
        """Likelihood-ratio test of θ = 0: ``(chibar2, p)``.

        The null sits on the boundary, so the p-value is half a chi2(1)
        tail (Stata's ``chibar2(01)``).
        """
        from scipy import stats

        lr = max(2.0 * (self.log_likelihood - self.loglik_cox), 0.0)
        return lr, float(0.5 * stats.chi2.sf(lr, 1)) if lr > 0 else 1.0

    # Agent-native accessors (consistent with sp.cox / EconometricResults).
    # Cox frailty reports log-hazard-ratio coefficients; the frailty variance
    # parameter stays available via ``.theta``.
    @property
    def params(self) -> pd.Series:
        return pd.Series(np.asarray(self.beta, dtype=float), index=list(self.var_names))

    @property
    def std_errors(self) -> pd.Series:
        return pd.Series(np.asarray(self.se, dtype=float), index=list(self.var_names))

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.std_errors

    @property
    def pvalues(self) -> pd.Series:
        from scipy import stats

        z = (self.params / self.std_errors).to_numpy(dtype=float)
        return pd.Series(
            2.0 * stats.norm.sf(np.abs(z)),
            index=self.params.index,
        )

    def summary(self) -> str:
        lines = [
            "Cox Model with Shared Gamma Frailty",
            "-" * 50,
            f"Observations   : {self.n}",
            f"Events         : {self.n_events}",
            f"Clusters       : {self.n_clusters}",
            f"Theta (frailty variance): {self.theta:.4f}",
            f"Integrated log-lik      : {self.log_likelihood:.4f}",
            f"Concordance    : {self.concordance:.4f}",
            "",
            "Coefficients:",
        ]
        from scipy import stats

        for nm, b, s in zip(self.var_names, self.beta, self.se):
            t = b / s if s > 0 else np.nan
            p = 2 * stats.norm.sf(abs(t))
            lines.append(
                f"  {nm:<15s}  {b: .4f}  (SE {s: .4f}, z {t: .3f}, p {p: .4f})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _cox_logpl_derivs(
    A: np.ndarray,
    eta: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    ties: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Cox log partial likelihood, score and Hessian in the columns of ``A``.

    ``eta = A @ coef``. Ties by Efron's or Breslow's approximation.
    """
    n, q = A.shape
    order = np.argsort(-T, kind="mergesort")
    A, eta, T, E = A[order], eta[order], T[order], E[order]
    r = np.exp(eta - eta.max())  # rescaled; ratios below are invariant
    shift = eta.max()
    ll = 0.0
    score = np.zeros(q)
    hess = np.zeros((q, q))
    # Running risk-set sums as T decreases.
    s0 = 0.0
    s1 = np.zeros(q)
    s2 = np.zeros((q, q))
    i = 0
    while i < n:
        j = i
        while j < n and T[j] == T[i]:
            j += 1
        blk = slice(i, j)
        rb = r[blk]
        Ab = A[blk]
        s0 += rb.sum()
        s1 += rb @ Ab
        s2 += (Ab * rb[:, None]).T @ Ab
        ev = E[blk] == 1
        d = int(ev.sum())
        if d > 0:
            Ad = Ab[ev]
            rd = rb[ev]
            ll += float(eta[blk][ev].sum())
            score += Ad.sum(axis=0)
            d0 = rd.sum()
            d1 = rd @ Ad
            d2 = (Ad * rd[:, None]).T @ Ad
            for ell in range(d):
                c = ell / d if ties == "efron" else 0.0
                den = s0 - c * d0
                xb = (s1 - c * d1) / den
                ll -= np.log(den) + shift
                score -= xb
                hess -= (s2 - c * d2) / den - np.outer(xb, xb)
        i = j
    return ll, score, hess


def _frailty_logpl_derivs(
    X: np.ndarray,
    gidx: np.ndarray,
    n_groups: int,
    coef: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    ties: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Cox log partial likelihood, score and Hessian in (β, w).

    The same quantities as :func:`_cox_logpl_derivs` on the design
    ``[X, Z]`` (``Z`` the cluster indicators), computed from risk-set sums
    by cluster so the cost is O(n p^2 + L G^2) instead of O(n (p + G)^2),
    where L is the number of events.
    """
    n, p = X.shape
    G = n_groups
    beta = coef[:p]
    w = coef[p:]
    eta = X @ beta + w[gidx]
    shift = float(eta.max())
    r = np.exp(eta - shift)
    ev = E == 1
    etimes = np.unique(T[ev])
    K = len(etimes)
    # bucket k(i): subject i is at risk at event times 0..k(i).
    k_of = np.searchsorted(etimes, T, side="right") - 1
    keep = k_of >= 0
    kk = k_of[keep]
    rk = r[keep]
    Xk = X[keep]
    gk = gidx[keep]

    def _rev(a: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.flip(np.cumsum(np.flip(a, axis=0), axis=0), axis=0)
        return out

    s0 = _rev(np.bincount(kk, weights=rk, minlength=K))
    s1 = np.zeros((K, p))
    np.add.at(s1, kk, rk[:, None] * Xk)
    s1 = _rev(s1)
    s2 = np.zeros((K, p, p))
    np.add.at(s2, kk, rk[:, None, None] * Xk[:, :, None] * Xk[:, None, :])
    s2 = _rev(s2)
    sg = np.zeros((K, G))
    np.add.at(sg, (kk, gk), rk)
    sg = _rev(sg)
    sgx = np.zeros((K, G, p))
    np.add.at(sgx, (kk, gk), rk[:, None] * Xk)
    sgx = _rev(sgx)

    # Sums over the events at each time (for Efron's correction).
    ke = np.searchsorted(etimes, T[ev])
    re = r[ev]
    Xe = X[ev]
    ge = gidx[ev]
    d = np.bincount(ke, minlength=K)
    d0 = np.bincount(ke, weights=re, minlength=K)
    d1 = np.zeros((K, p))
    np.add.at(d1, ke, re[:, None] * Xe)
    d2 = np.zeros((K, p, p))
    np.add.at(d2, ke, re[:, None, None] * Xe[:, :, None] * Xe[:, None, :])
    dg = np.zeros((K, G))
    np.add.at(dg, (ke, ge), re)
    dgx = np.zeros((K, G, p))
    np.add.at(dgx, (ke, ge), re[:, None] * Xe)

    # One row per event: time index and Efron fraction ell / d.
    row_k = np.repeat(np.arange(K), d)
    if ties == "efron":
        start = np.repeat(np.cumsum(d) - d, d)
        c = (np.arange(len(row_k)) - start) / d[row_k]
    else:
        c = np.zeros(len(row_k))
    S0 = s0[row_k] - c * d0[row_k]
    S1 = s1[row_k] - c[:, None] * d1[row_k]
    S2 = s2[row_k] - c[:, None, None] * d2[row_k]
    SG = sg[row_k] - c[:, None] * dg[row_k]
    SGX = sgx[row_k] - c[:, None, None] * dgx[row_k]
    inv = 1.0 / S0
    xb = S1 * inv[:, None]  # (L, p)
    gb = SG * inv[:, None]  # (L, G)

    ll = float(eta[ev].sum() - np.sum(np.log(S0) + shift))
    score = np.empty(p + G)
    score[:p] = Xe.sum(axis=0) - xb.sum(axis=0)
    score[p:] = np.bincount(ge, minlength=G) - gb.sum(axis=0)
    hess = np.empty((p + G, p + G))
    hess[:p, :p] = -(np.einsum("l,ljk->jk", inv, S2) - xb.T @ xb)
    hbw = -(np.einsum("l,lgj->jg", inv, SGX) - xb.T @ gb)
    hess[:p, p:] = hbw
    hess[p:, :p] = hbw.T
    hess[p:, p:] = gb.T @ gb
    hess[p:, p:][np.diag_indices(G)] -= gb.sum(axis=0)
    return ll, score, hess


def _gamma_correction(d: np.ndarray, nu: float) -> float:
    """Integrated-likelihood correction for a gamma frailty with 1/θ = nu."""
    return float(
        np.sum(
            d
            + nu * np.log(nu)
            - (nu + d) * np.log(nu + d)
            + gammaln(nu + d)
            - gammaln(nu)
        )
    )


def _fit_fixed_theta(
    X: np.ndarray,
    gidx: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    theta: float,
    start: np.ndarray,
    ties: str,
    maxiter: int,
    tol: float,
) -> Tuple[np.ndarray, float, np.ndarray, bool]:
    """Maximise the gamma-penalised partial likelihood for fixed θ > 0.

    Returns ``(coef, logPL - (1/θ) Σ (exp(w) - w - 1), penalised Hessian,
    converged)``; ``coef`` stacks β and the log-frailties w. The second
    element plus :func:`_gamma_correction` is the integrated log-likelihood
    (Stata's ``e(ll)``, R's ``c.loglik``).
    """
    p = X.shape[1]
    n_groups = len(start) - p
    nu = 1.0 / theta

    def _obj(c: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        ll, g, h = _frailty_logpl_derivs(X, gidx, n_groups, c, T, E, ties)
        w = c[p:]
        ew = np.exp(w)
        pen = nu * np.sum(ew - w)
        g = g.copy()
        g[p:] -= nu * (ew - 1.0)
        h = h.copy()
        h[p:, p:] -= np.diag(nu * ew)
        return ll - pen, ll, g, h

    coef = start.copy()
    ppl, ll, g, h = _obj(coef)
    converged = False
    for _ in range(maxiter):
        step = np.linalg.solve(h, g)
        new = coef - step
        new_ppl, new_ll, new_g, new_h = _obj(new)
        halvings = 0
        while not np.isfinite(new_ppl) or new_ppl < ppl - 1e-12 * abs(ppl):
            step = step / 2.0
            new = coef - step
            new_ppl, new_ll, new_g, new_h = _obj(new)
            halvings += 1
            if halvings > 30:
                break
        coef, ppl, ll, g, h = new, new_ppl, new_ll, new_g, new_h
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    w = coef[p:]
    return coef, ll - nu * float(np.sum(np.exp(w) - w - 1.0)), h, converged


def cox_frailty(
    formula: str,
    data: pd.DataFrame,
    cluster: str,
    alpha: float = 0.05,
    maxiter: int = 50,
    tol: float = 1e-10,
    theta: Optional[float] = None,
    ties: str = "efron",
) -> FrailtyResult:
    """Cox proportional hazards with shared gamma frailty.

    Parameters
    ----------
    formula : str
        ``"duration + event ~ x1 + x2"`` (like R's ``Surv(time, event) ~ x``).
    data : pd.DataFrame
    cluster : str
        Column identifying clusters (e.g. hospital, site).
    alpha : float
        Significance level (kept for API symmetry; intervals are built by
        the result's accessors).
    maxiter, tol : int, float
        Newton-Raphson controls for the penalised fit at each θ.
    theta : float, optional
        Fix the frailty variance instead of estimating it (R
        ``frailty(id, theta=)``). ``0`` fits the ordinary Cox model.
    ties : {"efron", "breslow"}, default "efron"
        Tie handling in the partial likelihood (R's default is Efron,
        Stata's ``stcox`` default is Breslow; identical without ties).

    Returns
    -------
    FrailtyResult
        ``beta`` / ``se`` (log hazard ratios; SEs conditional on θ, from the
        inverse of the full penalised information), ``theta`` (frailty
        variance), ``frailties`` (exp of the log-frailties
        ``log_frailties``), ``log_likelihood`` (the integrated
        log-likelihood maximised over θ) and ``lr_theta0``.

    Notes
    -----
    θ is the maximiser of the integrated likelihood. At a fixed θ the
    coefficients, standard errors and log-frailties are those of Stata
    ``stcox, shared(id)`` and of R ``coxph(... + frailty(id,
    theta=, sparse=FALSE))``. R's default ``sparse=TRUE`` approximates the
    frailty block of the information by its diagonal, which moves the
    standard errors in the fourth digit, and its default ``method="em"``
    stops the θ search on a 1e-5 relative change in the likelihood, so its
    reported θ is not the exact maximiser.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> cluster = np.repeat(np.arange(20), 15)
    >>> n = cluster.size
    >>> frailty = rng.gamma(2.0, 0.5, 20)[cluster]
    >>> x = rng.normal(size=n)
    >>> rate = frailty * np.exp(0.7 * x)
    >>> T = rng.exponential(1.0 / rate)
    >>> C = rng.exponential(2.0, n)
    >>> df = pd.DataFrame({"time": np.minimum(T, C),
    ...                    "event": (T <= C).astype(int),
    ...                    "x": x, "hospital": cluster})
    >>> res = sp.cox_frailty("time + event ~ x", df, cluster="hospital")
    >>> res.var_names
    ['x']
    >>> bool(res.theta >= 0)  # gamma frailty variance
    True

    References
    ----------
    [@therneau2000modeling]
    """
    if ties not in ("efron", "breslow"):
        raise ValueError("ties must be 'efron' or 'breslow'")
    if theta is not None and (not np.isfinite(theta) or theta < 0):
        raise ValueError("theta must be a non-negative finite number")
    lhs_str, covariates = _parse_formula(formula)
    # LHS is "duration + event" — split on "+"
    lhs_parts = [s.strip() for s in lhs_str.split("+")]
    if len(lhs_parts) != 2:
        raise ValueError(
            "formula LHS must be 'duration + event' (e.g. 'T + E ~ x1 + x2')"
        )
    duration_col, event_col = lhs_parts[0], lhs_parts[1]
    df = data[[duration_col, event_col, cluster] + covariates].dropna()
    T = df[duration_col].to_numpy(float)
    E = df[event_col].to_numpy(float).astype(int)
    X = df[covariates].to_numpy(float)
    n, k = X.shape
    n_events = int(E.sum())

    cluster_arr = df[cluster].to_numpy()
    uniq_clusters, cluster_idx = np.unique(cluster_arr, return_inverse=True)
    n_clusters = len(uniq_clusters)
    d_g = np.bincount(cluster_idx, weights=E, minlength=n_clusters)

    # Ordinary Cox fit: the θ = 0 end point and the warm start.
    beta0 = np.zeros(k)
    ll_cox = -np.inf
    for _ in range(maxiter):
        ll_cox, g, h = _cox_logpl_derivs(X, X @ beta0, T, E, ties)
        step = np.linalg.solve(h, g)
        beta0 = beta0 - step
        if np.max(np.abs(step)) < tol:
            break
    ll_cox, _, h_cox = _cox_logpl_derivs(X, X @ beta0, T, E, ties)

    cache: dict = {}
    start = np.concatenate([beta0, np.zeros(n_clusters)])

    def _profile(th: float) -> float:
        nonlocal start
        coef, ll, h, ok = _fit_fixed_theta(
            X, cluster_idx, T, E, th, start, ties, maxiter, tol
        )
        start = coef
        cache[th] = (coef, ll, h, ok)
        return ll + _gamma_correction(d_g, 1.0 / th)

    if theta is None:
        # Maximise the integrated log-likelihood over log θ.
        opt = minimize_scalar(
            lambda s: -_profile(float(np.exp(s))),
            bounds=(np.log(1e-6), np.log(1e3)),
            method="bounded",
            options={"xatol": 1e-10, "maxiter": 500},
        )
        theta_hat = float(np.exp(opt.x))
        theta_fixed = False
        if -opt.fun <= ll_cox + 1e-10:
            theta_hat = 0.0  # boundary: no frailty variation
    else:
        theta_hat = float(theta)
        theta_fixed = True

    if theta_hat > 0:
        c_ll = _profile(theta_hat)
        coef, _, h, ok = cache[theta_hat]
        if not ok:
            import warnings

            warnings.warn(
                "cox_frailty: penalised Newton-Raphson did not converge; "
                "estimates may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
        beta = coef[:k]
        w = coef[k:]
        cov = np.linalg.inv(-h)[:k, :k]
    else:
        c_ll = ll_cox
        beta = beta0
        w = np.zeros(n_clusters)
        cov = np.linalg.inv(-h_cox)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    from .models import _concordance_index

    concordance = _concordance_index(beta, X, T, E)

    _result = FrailtyResult(
        beta=beta,
        se=se,
        var_names=covariates,
        theta=theta_hat,
        frailties=np.exp(w),
        cluster_ids=uniq_clusters,
        log_likelihood=float(c_ll),
        n=n,
        n_events=n_events,
        n_clusters=n_clusters,
        concordance=concordance,
        log_frailties=w,
        loglik_cox=float(ll_cox),
        ties=ties,
        theta_fixed=theta_fixed,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.survival.cox_frailty",
            params={
                "formula": formula,
                "cluster": cluster,
                "alpha": alpha,
                "maxiter": maxiter,
                "tol": tol,
                "theta": theta,
                "ties": ties,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
