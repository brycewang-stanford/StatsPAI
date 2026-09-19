"""GARCH(p,q) volatility models (Bollerslev, 1986).

Maximum-likelihood estimation of conditional variance models:

    r_t = μ + ε_t,   ε_t = σ_t z_t,   z_t ~ N(0,1)
    σ²_t = ω + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j}

The most common specification is GARCH(1,1) where
    σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}

and α + β < 1 for stationarity.

This module provides:
- :func:`garch` — fit GARCH(p,q) by MLE (conditional Gaussian)
- Result with volatility path, standardised residuals, forecast
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .._result_serialize import ResultProtocolMixin
from ..regression._optim_helpers import hessian_cov


@dataclass
class GARCHResult(ResultProtocolMixin):
    """Fitted GARCH(p,q) model returned by :func:`garch`.

    Holds the conditional-variance parameters, the volatility path, and
    standardised residuals, plus :meth:`forecast` for multi-step variance.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> T = 400
    >>> eps = np.zeros(T)
    >>> s2 = np.ones(T)
    >>> omega, a1, b1 = 0.05, 0.1, 0.85
    >>> for t in range(1, T):
    ...     s2[t] = omega + a1 * eps[t - 1] ** 2 + b1 * s2[t - 1]
    ...     eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
    >>> res = sp.garch(eps, p=1, q=1)
    >>> bool(res.persistence < 1.0)   # alpha + beta < 1 => stationary
    True
    >>> res.forecast(horizon=3).shape
    (3,)
    """

    omega: float
    alpha: np.ndarray  # (q,)
    beta: np.ndarray  # (p,)
    mu: float
    sigma2: np.ndarray  # conditional variance path (T,)
    residuals: np.ndarray  # ε_t = r_t - μ
    std_residuals: np.ndarray  # z_t = ε_t / σ_t
    log_likelihood: float
    aic: float
    bic: float
    n: int
    p: int
    q: int
    coef: Optional[np.ndarray] = None  # parameter vector (param_names order)
    se_vec: Optional[np.ndarray] = None  # asymptotic SEs (same order)
    param_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Agent-native accessors (params / std_errors / t / p), so GARCH
    # supports inference like every other estimator. Standard errors come
    # from the inverse observed-information (numerical Hessian) at the MLE.
    # ------------------------------------------------------------------
    @property
    def params(self) -> pd.Series:
        if self.coef is None or self.param_names is None:
            return pd.Series(dtype=float)
        return pd.Series(
            np.asarray(self.coef, float),
            index=list(self.param_names),
        )

    @property
    def std_errors(self) -> pd.Series:
        if self.se_vec is None or self.param_names is None:
            return pd.Series(dtype=float)
        return pd.Series(
            np.asarray(self.se_vec, float),
            index=list(self.param_names),
        )

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.std_errors

    @property
    def pvalues(self) -> pd.Series:
        from scipy import stats

        z = (self.params / self.std_errors).to_numpy(float)
        return pd.Series(2.0 * stats.norm.sf(np.abs(z)), index=self.params.index)

    @property
    def persistence(self) -> float:
        return float(self.alpha.sum() + self.beta.sum())

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Multi-step ahead variance forecast (analytic recursion).

        E[eps^2_{T+h}] = sigma^2_{T+h|T} for h >= 1, so future squared shocks
        are replaced by their forecasts; every ARCH and GARCH lag is used.
        """
        eps2 = list(np.asarray(self.residuals, float) ** 2)
        s2 = list(np.asarray(self.sigma2, float))
        out = np.empty(horizon)
        for h in range(horizon):
            v = self.omega
            for i in range(self.q):
                v += self.alpha[i] * eps2[-1 - i]
            for j in range(self.p):
                v += self.beta[j] * s2[-1 - j]
            out[h] = v
            eps2.append(v)
            s2.append(v)
        return out

    def summary(self) -> str:
        lines = [
            f"GARCH({self.p},{self.q})",
            "-" * 40,
            f"n              : {self.n}",
            f"Log-Lik        : {self.log_likelihood:.4f}",
            f"AIC            : {self.aic:.4f}",
            f"BIC            : {self.bic:.4f}",
            f"Persistence    : {self.persistence:.4f}",
            "",
            "Parameters:",
        ]
        if self.param_names is not None and self.se_vec is not None:
            pr, se = self.params, self.std_errors
            tv, pv = self.tvalues, self.pvalues
            lines.append(
                f"  {'':<10s}{'coef':>11s}{'std err':>11s}" f"{'z':>9s}{'P>|z|':>9s}"
            )
            for nm in self.param_names:
                lines.append(
                    f"  {nm:<10s}{pr[nm]:11.6f}{se[nm]:11.6f}"
                    f"{tv[nm]:9.3f}{pv[nm]:9.4f}"
                )
        else:
            lines.append(f"  mu    = {self.mu: .6f}")
            lines.append(f"  omega = {self.omega: .6f}")
            for i, a in enumerate(self.alpha):
                lines.append(f"  alpha[{i + 1}] = {a: .6f}")
            for j, b in enumerate(self.beta):
                lines.append(f"  beta[{j + 1}] = {b: .6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _garch_filter(
    theta: np.ndarray,
    y: np.ndarray,
    p: int,
    q: int,
    mean: bool,
    presample: str,
    want_grad: bool = False,
):
    """Conditional variances, log-likelihood contributions and scores.

    theta = (mu?, omega, alpha_1..alpha_q, beta_1..beta_p). Pre-sample
    values ``m = mean(eps**2)`` (evaluated at the current mu):

    * ``'stata'``: eps**2 and sigma**2 at t < 0 equal ``m``; the recursion
      runs from the first observation (Stata ``arch``, default ``arch0(xb)``).
    * ``'rugarch'``: sigma**2_t = m for t < max(p, q); the recursion starts
      at t = max(p, q) (``rugarch::ugarchfit`` sGARCH, default recursion
      init over the whole sample).

    Returns (sigma2, eps, ll_t, scores) with ``scores`` the T x K matrix of
    per-observation log-likelihood derivatives (None unless want_grad).
    """
    T = y.shape[0]
    K = theta.shape[0]
    j0 = int(mean)
    mu = float(theta[0]) if mean else 0.0
    omega = float(theta[j0])
    alpha = theta[j0 + 1 : j0 + 1 + q]
    beta = theta[j0 + 1 + q : j0 + 1 + q + p]
    eps = y - mu
    e2 = eps * eps
    m = float(e2.mean())
    dm = np.zeros(K)
    if mean:
        dm[0] = -2.0 * float(eps.mean())
    s2 = np.empty(T)
    ds2 = np.zeros((T, K)) if want_grad else None
    r0 = max(p, q) if presample == "rugarch" else 0
    for t in range(T):
        if t < r0:
            s2[t] = m
            if want_grad:
                ds2[t] = dm
            continue
        v = omega
        g = np.zeros(K) if want_grad else None
        if want_grad:
            g[j0] = 1.0
        for i in range(q):
            s = t - 1 - i
            if s >= 0:
                val = e2[s]
                v += alpha[i] * val
                if want_grad:
                    g[j0 + 1 + i] += val
                    if mean:
                        g[0] += alpha[i] * (-2.0 * eps[s])
            else:
                v += alpha[i] * m
                if want_grad:
                    g[j0 + 1 + i] += m
                    g += alpha[i] * dm
        for j in range(p):
            s = t - 1 - j
            if s >= 0:
                v += beta[j] * s2[s]
                if want_grad:
                    g[j0 + 1 + q + j] += s2[s]
                    g += beta[j] * ds2[s]
            else:
                v += beta[j] * m
                if want_grad:
                    g[j0 + 1 + q + j] += m
                    g += beta[j] * dm
        s2[t] = v
        if want_grad:
            ds2[t] = g
    with np.errstate(divide="ignore", invalid="ignore"):
        ll_t = -0.5 * (np.log(2 * np.pi) + np.log(s2) + e2 / s2)
    scores = None
    if want_grad:
        scores = (-0.5 * (1.0 / s2 - e2 / s2**2))[:, None] * ds2
        if mean:
            scores[:, 0] += eps / s2
    return s2, eps, ll_t, scores


def garch(
    y: object,
    p: int = 1,
    q: int = 1,
    mean: bool = True,
    presample: str = "stata",
    vce: str = "oim",
) -> GARCHResult:
    """Fit GARCH(p,q) by conditional Gaussian MLE.

    Parameters
    ----------
    y : array-like
        Return series (or log-return, etc.).
    p : int, default 1
        Number of GARCH (lagged σ²) terms (Stata ``garch(p)``).
    q : int, default 1
        Number of ARCH (lagged ε²) terms (Stata ``arch(q)``).
    mean : bool, default True
        Estimate a constant mean μ; if False, μ = 0.
    presample : {'stata', 'rugarch'}, default 'stata'
        How the recursion is started; both use ``m = mean(ε²)`` at the
        current μ. ``'stata'``: pre-sample ε² and σ² equal ``m`` and every
        observation's σ² follows the recursion (Stata ``arch``, default
        ``arch0(xb)``). ``'rugarch'``: σ²_t = ``m`` for the first
        ``max(p, q)`` observations (``rugarch::ugarchfit``, sGARCH).
        The log-likelihood sums over all observations in both cases.
    vce : {'oim', 'opg', 'robust'}, default 'oim'
        Covariance of the estimates: inverse observed information (Hessian
        of the log-likelihood, by central differences of the analytic
        score), outer product of the scores (Stata's ``arch`` default), or
        the Bollerslev-Wooldridge sandwich ``H^{-1} (S'S) H^{-1}``.

    Notes
    -----
    Maximised by BFGS on the analytic score, polished by Newton steps on
    the (numerical-Hessian, analytic-gradient) system. ω > 0, α, β >= 0
    and Σα + Σβ < 1 are enforced by an infinite objective outside the
    region.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> T = 400
    >>> eps = np.zeros(T)
    >>> s2 = np.ones(T)
    >>> omega, a1, b1 = 0.05, 0.1, 0.85
    >>> for t in range(1, T):
    ...     s2[t] = omega + a1 * eps[t - 1] ** 2 + b1 * s2[t - 1]
    ...     eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
    >>> res = sp.garch(eps, p=1, q=1)
    >>> isinstance(res, sp.GARCHResult)
    True
    >>> bool(res.persistence < 1.0)   # alpha + beta < 1 => stationary
    True
    >>> res.sigma2.shape
    (400,)
    >>> res.forecast(horizon=3).shape
    (3,)
    >>> bool(np.isfinite(res.aic))
    True
    >>> print(res.summary())  # doctest: +SKIP
    """
    y = np.asarray(y, dtype=float).ravel()
    T = len(y)
    if T < max(p, q) + 10:
        raise ValueError("Time series too short for GARCH estimation.")
    if presample not in ("stata", "rugarch"):
        raise ValueError("presample must be 'stata' or 'rugarch'")
    if vce not in ("oim", "opg", "robust"):
        raise ValueError("vce must be 'oim', 'opg' or 'robust'")
    if not np.all(np.isfinite(y)):
        raise ValueError("garch: y contains NaN or inf")
    y_mean = float(y.mean()) if mean else 0.0
    j0 = int(mean)

    def _feasible(theta: np.ndarray) -> bool:
        omega = theta[j0]
        ab = theta[j0 + 1 :]
        return bool(omega > 0 and np.all(ab >= 0) and ab.sum() < 1.0)

    def neg_ll(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        if not _feasible(theta):
            return 1e15
        s2, _, ll_t, _ = _garch_filter(theta, y, p, q, mean, presample)
        if np.any(s2 <= 0) or not np.all(np.isfinite(ll_t)):
            return 1e15
        return float(-ll_t.sum())

    def neg_grad(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if not _feasible(theta):
            return np.zeros_like(theta)
        _, _, _, sc = _garch_filter(theta, y, p, q, mean, presample, True)
        return -sc.sum(axis=0)

    # Initial guesses
    eps0 = y - y_mean
    var0 = float(np.mean(eps0**2))
    alpha0 = [0.1 / max(q, 1)] * q
    beta0 = [0.8 / max(p, 1)] * p
    omega0 = var0 * max(1.0 - sum(alpha0) - sum(beta0), 0.05)
    x0 = np.asarray(([y_mean] if mean else []) + [omega0] + alpha0 + beta0, float)

    opt = minimize(
        neg_ll,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12},
    )
    theta = np.asarray(opt.x, dtype=float)
    opt2 = minimize(neg_ll, theta, jac=neg_grad, method="BFGS", options={"gtol": 1e-9})
    if np.isfinite(opt2.fun) and opt2.fun <= opt.fun:
        theta = np.asarray(opt2.x, dtype=float)

    def _hess(th: np.ndarray) -> np.ndarray:
        # Hessian of the NEGATIVE log-likelihood by central differences of
        # the analytic score.
        K = th.size
        H = np.empty((K, K))
        for i in range(K):
            h = 1e-5 * max(abs(th[i]), 1e-2)
            e = np.zeros(K)
            e[i] = h
            H[:, i] = (neg_grad(th + e) - neg_grad(th - e)) / (2 * h)
        return (H + H.T) / 2.0

    # Newton polish: drives the analytic gradient to ~machine precision
    for _ in range(20):
        g = neg_grad(theta)
        if np.max(np.abs(g)) < 1e-10:
            break
        try:
            step = np.linalg.solve(_hess(theta), g)
        except np.linalg.LinAlgError:
            break
        cand = theta - step
        if not _feasible(cand) or neg_ll(cand) > neg_ll(theta) + 1e-12:
            break
        theta = cand

    s2, eps, ll_t, scores = _garch_filter(theta, y, p, q, mean, presample, True)
    mu = float(theta[0]) if mean else 0.0
    omega = float(theta[j0])
    alpha = theta[j0 + 1 : j0 + 1 + q]
    beta = theta[j0 + 1 + q : j0 + 1 + q + p]

    param_names = (
        (["mu"] if mean else [])
        + ["omega"]
        + [f"alpha[{i + 1}]" for i in range(q)]
        + [f"beta[{j + 1}]" for j in range(p)]
    )
    H = _hess(theta)
    try:
        H_inv = np.linalg.inv(H)
        if vce == "oim":
            V = H_inv
        elif vce == "opg":
            V = np.linalg.inv(scores.T @ scores)
        else:
            V = H_inv @ (scores.T @ scores) @ H_inv
        se_vec = np.sqrt(np.clip(np.diag(V), 0.0, None))
    except np.linalg.LinAlgError:
        import warnings

        warnings.warn(
            "garch: singular information matrix at the optimum; standard "
            "errors set to NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        se_vec = np.full(len(theta), np.nan)

    ll = float(ll_t.sum())
    k_params = int(mean) + 1 + q + p
    aic = -2 * ll + 2 * k_params
    bic = -2 * ll + k_params * np.log(T)
    std_resid = eps / np.sqrt(s2)

    _result = GARCHResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        mu=mu,
        sigma2=s2,
        residuals=eps,
        std_residuals=std_resid,
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        n=T,
        p=p,
        q=q,
        coef=np.asarray(theta, float),
        se_vec=se_vec,
        param_names=param_names,
    )
    _result.vce = vce
    _result.presample = presample
    _result.gradient_norm = float(np.max(np.abs(scores.sum(axis=0))))
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.timeseries.garch",
            params={"p": p, "q": q, "mean": mean, "presample": presample, "vce": vce},
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def garch_loglik(
    y: object,
    params: object,
    p: int = 1,
    q: int = 1,
    mean: bool = True,
    presample: str = "stata",
) -> float:
    """Gaussian GARCH(p,q) log-likelihood at given parameters.

    ``params`` is ordered as :attr:`GARCHResult.param_names`. Used to check
    that a reference optimum and ours sit on the same objective.
    """
    y = np.asarray(y, dtype=float).ravel()
    th = np.asarray(params, dtype=float)
    _, _, ll_t, _ = _garch_filter(th, y, p, q, mean, presample)
    return float(ll_t.sum())
