"""GARCH(p,q) volatility models (Bollerslev, 1986).

Maximum-likelihood estimation of conditional variance models:

    r_t = μ + ε_t,   ε_t = σ_t z_t,   z_t ~ N(0,1)
    σ²_t = ω + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j}

The most common specification is GARCH(1,1) where
    σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}

and α + β < 1 for stationarity.

The mean may carry autoregressive disturbances (``ar=``), the
standardised innovations may be Student t (``dist='t'``), and the variance
may respond asymmetrically to good and bad news (``model='gjr'`` or
``'egarch'``).

This module provides:
- :func:`garch` — fit GARCH(p,q) by MLE (Gaussian or Student t)
- Result with volatility path, standardised residuals, forecast
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility


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
    ar: Optional[np.ndarray] = None  # AR coefficients of the disturbance
    disturbances: Optional[np.ndarray] = None  # u_t = y_t - mu
    nu: Optional[float] = None  # degrees of freedom when dist == 't'
    dist: str = "normal"
    model: str = "garch"  # 'garch', 'gjr' or 'egarch'
    gamma: Optional[np.ndarray] = None  # asymmetry terms (gjr, egarch)
    theta: Optional[np.ndarray] = None  # egarch: effect of the signed shock
    # coefficient of the variance in the mean (an array, in the order of
    # ``in_mean_lags``, when there is more than one term)
    archm: Any = None
    in_mean: Optional[str] = None  # 'variance', 'sd' or 'log'
    in_mean_lags: tuple = ()

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
        """Sum of the coefficients that carry the variance forward:
        ``sum(alpha) + sum(beta)``, plus half the threshold terms for
        ``'gjr'`` (a symmetric innovation is negative half the time);
        ``sum(beta)`` for ``'egarch'``."""
        if self.model == "egarch":
            return float(self.beta.sum())
        half = 0.0 if self.gamma is None else 0.5 * float(np.sum(self.gamma))
        return float(self.alpha.sum() + self.beta.sum() + half)

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Multi-step ahead variance forecast (analytic recursion).

        E[eps^2_{T+h}] = sigma^2_{T+h|T} for h >= 1, so future squared shocks
        are replaced by their forecasts; every ARCH and GARCH lag is used.
        """
        if self.model == "egarch":
            return self._forecast_egarch(horizon)
        eps = np.asarray(self.residuals, float)
        eps2 = list(eps**2)
        # squared negative shocks; a future one is half the variance for a
        # symmetric innovation
        neg2 = list(np.where(eps < 0, eps**2, 0.0))
        gam = np.zeros(self.q) if self.gamma is None else np.asarray(self.gamma)
        s2 = list(np.asarray(self.sigma2, float))
        out = np.empty(horizon)
        for h in range(horizon):
            v = self.omega
            for i in range(self.q):
                v += self.alpha[i] * eps2[-1 - i] + gam[i] * neg2[-1 - i]
            for j in range(self.p):
                v += self.beta[j] * s2[-1 - j]
            out[h] = v
            eps2.append(v)
            neg2.append(0.5 * v)
            s2.append(v)
        return out

    def _forecast_egarch(self, horizon: int) -> np.ndarray:
        """``exp`` of the forecast of the log variance. Exact one step
        ahead; further out it is the exponential of the expected log
        variance, which is below the expected variance."""
        z = list(np.asarray(self.std_residuals, float))
        ls = list(np.log(np.asarray(self.sigma2, float)))
        th = np.zeros(self.q) if self.theta is None else np.asarray(self.theta)
        ga = np.zeros(self.q) if self.gamma is None else np.asarray(self.gamma)
        kappa = float(np.sqrt(2.0 / np.pi))
        n = len(z)
        out = np.empty(horizon)
        for h in range(horizon):
            v = self.omega
            for i in range(self.q):
                idx = n + h - 1 - i
                if idx < n:  # an observed shock; future ones have mean zero
                    v += th[i] * z[idx] + ga[i] * (abs(z[idx]) - kappa)
            for j in range(self.p):
                v += self.beta[j] * ls[-1 - j]
            ls.append(v)
            out[h] = np.exp(v)
        return out

    def forecast_mean(self, horizon: int = 1) -> np.ndarray:
        """Multi-step forecast of the series: the constant mean plus the
        autoregressive forecast of the disturbance (``ar=``)."""
        rho = np.zeros(0) if self.ar is None else np.asarray(self.ar, float)
        source = self.residuals if self.disturbances is None else self.disturbances
        u = list(np.asarray(source, float))
        out = np.empty(horizon)
        premium = np.zeros(horizon)
        if self.archm is not None:
            # g(sigma2) at the dates the mean needs: observed variances,
            # then their forecasts (exact one step ahead; further out g of
            # the forecast variance stands in for the forecast of g)
            psi = np.atleast_1d(np.asarray(self.archm, float))
            path = np.concatenate(
                [np.asarray(self.sigma2, float), self.forecast(horizon)]
            )
            g = {"sd": np.sqrt, "log": np.log}.get(self.in_mean or "", lambda v: v)
            n = len(self.sigma2)
            for h in range(horizon):
                for coef, lag in zip(psi, self.in_mean_lags or (0,)):
                    premium[h] += coef * float(g(path[n + h - lag]))
        for h in range(horizon):
            nxt = float(sum(rho[k] * u[-1 - k] for k in range(rho.size)))
            u.append(nxt)
            out[h] = self.mu + premium[h] + nxt
        return out

    def value_at_risk(self, alpha: float = 0.01) -> float:
        """One-step-ahead value at risk: the ``alpha`` quantile of the
        forecast distribution of the next observation.

        ``mu_{T+1} + sigma_{T+1} * q_alpha`` with ``q_alpha`` the quantile
        of the standardised innovation (standard normal, or Student t
        scaled to unit variance). A return below it has probability
        ``alpha`` under the fitted model.
        """
        from scipy import stats

        if not 0.0 < alpha < 1.0:
            raise MethodIncompatibility("value_at_risk: alpha must be in (0, 1).")
        if self.dist == "t" and self.nu is not None:
            nu = float(self.nu)
            quantile = float(stats.t.ppf(alpha, nu) * np.sqrt((nu - 2.0) / nu))
        else:
            quantile = float(stats.norm.ppf(alpha))
        sd = float(np.sqrt(self.forecast(1)[0]))
        return float(self.forecast_mean(1)[0] + sd * quantile)

    def summary(self) -> str:
        label = {"garch": "GARCH", "gjr": "GJR-GARCH", "egarch": "EGARCH"}
        head = f"{label.get(self.model, 'GARCH')}({self.p},{self.q})"
        n_ar = 0 if self.ar is None else int(np.size(self.ar))
        if n_ar:
            head = f"AR({n_ar})-" + head
        if self.archm is not None:
            head += " in mean"
        if self.dist == "t":
            head += ", Student t innovations"
        lines = [
            head,
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
) -> Any:
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


def _variance_path(
    omega: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    e2: np.ndarray,
    presample: str,
    gamma: Optional[np.ndarray] = None,
    neg2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Conditional variances by linear filtering; the same recursion and
    pre-sample rules as :func:`_garch_filter`, without the Python loop.

    ``gamma`` and ``neg2`` add ``sum gamma_i neg2_{t-i}`` (threshold terms
    on the squared negative shocks, zero before the sample)."""
    from scipy.signal import lfilter, lfiltic

    T = e2.shape[0]
    q, p = alpha.shape[0], beta.shape[0]
    m = float(e2.mean())
    c = np.full(T, omega)
    if q:
        ext = np.concatenate([np.full(q, m), e2])
        kernel = np.concatenate([[0.0], alpha])
        c = c + np.convolve(ext, kernel)[q : q + T]
        if gamma is not None and neg2 is not None:
            ext_n = np.concatenate([np.zeros(q), neg2])
            kernel_n = np.concatenate([[0.0], gamma])
            c = c + np.convolve(ext_n, kernel_n)[q : q + T]
    r0 = max(p, q) if presample == "rugarch" else 0
    if p == 0:
        s2 = c.copy()
        s2[:r0] = m
        return s2
    a = np.concatenate([[1.0], -beta])
    zi = lfiltic([1.0], a, y=np.full(p, m))
    s2 = np.empty(T)
    s2[:r0] = m
    s2[r0:] = lfilter([1.0], a, c[r0:], zi=zi)[0]
    return s2


def _egarch_loop(
    eps: np.ndarray,
    omega: float,
    th: np.ndarray,
    ga: np.ndarray,
    beta: np.ndarray,
    log_m: float,
    kappa: float,
) -> np.ndarray:
    """Log variances of an EGARCH model. Pre-sample log variances equal
    ``log_m`` and pre-sample shocks contribute nothing."""
    n = eps.shape[0]
    q = th.shape[0]
    p = beta.shape[0]
    ls = np.empty(n)
    z = np.empty(n)
    for t in range(n):
        v = omega
        for i in range(q):
            k = t - 1 - i
            if k >= 0:
                v += th[i] * z[k] + ga[i] * (abs(z[k]) - kappa)
        for j in range(p):
            k = t - 1 - j
            v += beta[j] * (ls[k] if k >= 0 else log_m)
        if v > 700.0:
            v = 700.0
        elif v < -700.0:
            v = -700.0
        ls[t] = v
        z[t] = eps[t] / np.exp(0.5 * v)
    return ls


def _inmean_loop(
    w: np.ndarray,
    rho: np.ndarray,
    psi: np.ndarray,
    psi_lags: np.ndarray,
    transform: int,
    omega: float,
    alpha: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    m: float,
    code: int,
    kappa: float,
) -> tuple:
    """Variances and innovations when the variance enters the mean.

    ``w = y - mu``; the disturbance is ``u_t = w_t - sum_k psi_k
    g(sigma2_{t - lag_k})`` (``transform`` 0: ``g`` the identity, 1: the
    square root, 2: the logarithm; a variance before the sample is ``m``)
    and the innovation ``eps_t = u_t - sum rho_k u_{t-k}``, so the variance
    at ``t`` must be known before the innovation at ``t``: a sequential
    recursion. ``code`` 0: GARCH, 1: threshold terms ``gamma`` on squared
    negative shocks, 2: EGARCH (``alpha`` on the signed standardised
    shock, ``gamma`` on its magnitude). Pre-sample squared shocks and
    variances equal ``m``; pre-sample threshold and EGARCH shock terms are
    zero, pre-sample disturbances are zero.
    """
    n = w.shape[0]
    q = alpha.shape[0]
    g = gamma.shape[0]
    p = beta.shape[0]
    k_ar = rho.shape[0]
    s2 = np.empty(n)
    eps = np.empty(n)
    u = np.empty(n)
    log_m = np.log(m)
    for t in range(n):
        v = omega
        if code == 2:
            for i in range(q):
                k = t - 1 - i
                if k >= 0:
                    z = eps[k] / np.sqrt(s2[k])
                    v += alpha[i] * z + gamma[i] * (abs(z) - kappa)
            for j in range(p):
                k = t - 1 - j
                v += beta[j] * (np.log(s2[k]) if k >= 0 else log_m)
            if v > 700.0:
                v = 700.0
            elif v < -700.0:
                v = -700.0
            v = np.exp(v)
        else:
            for i in range(q):
                k = t - 1 - i
                v += alpha[i] * (eps[k] * eps[k] if k >= 0 else m)
            for i in range(g):
                k = t - 1 - i
                if k >= 0 and eps[k] < 0.0:
                    v += gamma[i] * eps[k] * eps[k]
            for j in range(p):
                k = t - 1 - j
                v += beta[j] * (s2[k] if k >= 0 else m)
        s2[t] = v
        prem = 0.0
        for k in range(psi.shape[0]):
            idx = t - psi_lags[k]
            val = s2[idx] if idx >= 0 else m
            if transform == 1:
                val = np.sqrt(val)
            elif transform == 2:
                val = np.log(val)
            prem += psi[k] * val
        u[t] = w[t] - prem
        e = u[t]
        for k in range(k_ar):
            if t - 1 - k >= 0:
                e -= rho[k] * u[t - 1 - k]
        eps[t] = e
    return s2, eps, u


_COMPILED: dict = {}


def _egarch_kernel() -> Any:
    """:func:`_egarch_loop`, compiled by numba on first use when numba is
    installed (it is imported here so that ``import statspai`` stays
    light)."""
    if "egarch" not in _COMPILED:
        try:
            from numba import njit  # type: ignore[import-untyped]

            _COMPILED["egarch"] = njit(cache=True)(_egarch_loop)
        except ImportError:  # pragma: no cover - numba is a core dependency
            _COMPILED["egarch"] = _egarch_loop
    return _COMPILED["egarch"]


def _inmean_kernel() -> Any:
    """:func:`_inmean_loop`, compiled by numba on first use."""
    if "inmean" not in _COMPILED:
        try:
            from numba import njit  # type: ignore[import-untyped]

            _COMPILED["inmean"] = njit(cache=True)(_inmean_loop)
        except ImportError:  # pragma: no cover - numba is a core dependency
            _COMPILED["inmean"] = _inmean_loop
    return _COMPILED["inmean"]


def _general_filter(
    theta: np.ndarray,
    y: np.ndarray,
    p: int,
    q: int,
    mean: bool,
    n_ar: int,
    dist: str,
    presample: str,
    model: str = "garch",
    n_g: Optional[int] = None,
    in_mean: Any = False,
    m_fixed: Optional[float] = None,
) -> Any:
    """Variances, innovations, disturbances and log-likelihood terms of the
    model with AR(``n_ar``) disturbances and normal or Student t errors.

    theta = (mu?, psi_1..psi_m?, rho_1..rho_k, omega, alpha_1..alpha_q,
    [gamma_1..gamma_g], beta_1..beta_p, nu?); for ``'egarch'`` the
    ``alpha`` block holds the coefficients of the signed shock. The
    disturbance ``u_t = y_t - mu - sum_k psi_k g(sigma2_{t - lag_k})``
    follows ``u_t = sum rho_k u_{t-k} + eps_t`` with pre-sample ``u`` equal
    to 0.
    """
    from scipy.signal import lfilter
    from scipy.special import gammaln

    j = int(mean)
    mu = float(theta[0]) if mean else 0.0
    # in_mean: False, or (transform code, lags of the variance in the mean)
    if in_mean is True:  # the current variance, untransformed
        in_mean = (0, (0,))
    transform, mean_lags = in_mean if in_mean else (0, ())
    n_psi = len(mean_lags)
    psi = np.ascontiguousarray(theta[j : j + n_psi], dtype=float)
    j += n_psi
    rho = theta[j : j + n_ar]
    j += n_ar
    omega = float(theta[j])
    alpha = theta[j + 1 : j + 1 + q]
    g = (q if model != "garch" else 0) if n_g is None else n_g
    gamma = theta[j + 1 + q : j + 1 + q + g]
    beta = theta[j + 1 + q + g : j + 1 + q + g + p]
    if in_mean:
        # The pre-sample value is the mean squared innovation of the model
        # itself, which depends on the variances it starts: a fixed point,
        # reached by iteration (the map is a contraction for the small
        # psi of practice; it stops after 200 rounds regardless).
        w = np.ascontiguousarray(y - mu, dtype=float)
        code = {"garch": 0, "gjr": 1, "egarch": 2}[model]
        args = (
            np.ascontiguousarray(rho, dtype=float),
            psi,
            np.asarray(mean_lags, dtype=np.int64),
            int(transform),
            omega,
            np.ascontiguousarray(alpha, dtype=float),
            np.ascontiguousarray(gamma, dtype=float),
            np.ascontiguousarray(beta, dtype=float),
        )
        kernel = _inmean_kernel()
        kappa = float(np.sqrt(2.0 / np.pi))
        m = float(np.mean(w * w)) if m_fixed is None else float(m_fixed)
        s2 = eps = u = w
        for _ in range(200 if m_fixed is None else 1):
            if not (np.isfinite(m) and m > 0):
                break
            s2, eps, u = kernel(w, *args, m, code, kappa)
            with np.errstate(over="ignore", invalid="ignore"):
                # an explosive trial point; the caller rejects it
                m_new = float(np.mean(eps * eps))
            done = abs(m_new - m) <= 1e-13 * abs(m)
            if m_fixed is None:
                m = m_new
            if done:
                break
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            e2 = eps * eps
            if dist == "t":
                nu = float(theta[-1])
                ll_t = (
                    gammaln((nu + 1.0) / 2.0)
                    - gammaln(nu / 2.0)
                    - 0.5 * np.log(np.pi * (nu - 2.0))
                    - 0.5 * np.log(s2)
                    - 0.5 * (nu + 1.0) * np.log1p(e2 / ((nu - 2.0) * s2))
                )
            else:
                ll_t = -0.5 * (np.log(2 * np.pi) + np.log(s2) + e2 / s2)
        return s2, eps, u, ll_t
    u = y - mu
    eps = lfilter(np.concatenate([[1.0], -rho]), [1.0], u) if n_ar else u
    e2 = eps * eps
    if model == "egarch":
        log_m = float(np.log(e2.mean()))
        ls = _egarch_kernel()(
            np.ascontiguousarray(eps, dtype=float),
            omega,
            np.ascontiguousarray(alpha, dtype=float),
            np.ascontiguousarray(gamma, dtype=float),
            np.ascontiguousarray(beta, dtype=float),
            log_m,
            float(np.sqrt(2.0 / np.pi)),
        )
        s2 = np.exp(ls)
    elif model == "gjr":
        # the threshold term is one more "news" series; before the sample
        # it is zero (Stata's tarch, written on positive shocks, starts
        # from the full pre-sample value, which is the same thing)
        neg2 = np.where(eps < 0, e2, 0.0)
        s2 = _variance_path(omega, alpha, beta, e2, presample, gamma, neg2)
    else:
        s2 = _variance_path(omega, alpha, beta, e2, presample)
    with np.errstate(divide="ignore", invalid="ignore"):
        if dist == "t":
            nu = float(theta[-1])
            ll_t = (
                gammaln((nu + 1.0) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * np.log(np.pi * (nu - 2.0))
                - 0.5 * np.log(s2)
                - 0.5 * (nu + 1.0) * np.log1p(e2 / ((nu - 2.0) * s2))
            )
        else:
            ll_t = -0.5 * (np.log(2 * np.pi) + np.log(s2) + e2 / s2)
    return s2, eps, u, ll_t


def _numeric_scores(fun: Any, theta: np.ndarray) -> np.ndarray:
    """T x K per-observation scores by central differences of ``fun``
    (theta -> T log-likelihood terms)."""
    cols = []
    for i in range(theta.size):
        h = 1e-6 * max(abs(theta[i]), 1e-2)
        e = np.zeros(theta.size)
        e[i] = h
        cols.append((fun(theta + e) - fun(theta - e)) / (2.0 * h))
    return np.column_stack(cols)


def _numeric_hessian(fun: Any, theta: np.ndarray) -> np.ndarray:
    """Hessian of the scalar ``fun`` by central second differences."""
    K = theta.size
    h = np.array([1e-4 * max(abs(t), 1e-2) for t in theta])
    H = np.empty((K, K))
    for i in range(K):
        for k in range(i, K):
            ei = np.zeros(K)
            ek = np.zeros(K)
            ei[i] = h[i]
            ek[k] = h[k]
            H[i, k] = H[k, i] = (
                fun(theta + ei + ek)
                - fun(theta + ei - ek)
                - fun(theta - ei + ek)
                + fun(theta - ei - ek)
            ) / (4.0 * h[i] * h[k])
    return H


def _series_argument(y: object, data: Optional[pd.DataFrame], who: str) -> np.ndarray:
    """``y`` as a float vector; a column of ``data`` when one is given, with
    the missing values at either end trimmed."""
    if data is None:
        if isinstance(y, str):
            raise MethodIncompatibility(
                f"{who}: y={y!r} names a column; pass data= as well.",
                recovery_hint=f"sp.{who}({y!r}, data=df) or sp.{who}(df[{y!r}]).",
            )
        return np.asarray(y, dtype=float).ravel()
    if not isinstance(y, str) or y not in data.columns:
        raise MethodIncompatibility(
            f"{who}: with data=, y must be the name of one of its columns.",
            recovery_hint="Check the column name.",
            diagnostics={"columns": [str(c) for c in data.columns][:20]},
        )
    values = data[y].to_numpy(dtype=float)
    keep = np.flatnonzero(~np.isnan(values))
    if keep.size == 0:
        raise MethodIncompatibility(
            f"{who}: column {y!r} has no non-missing values.",
            recovery_hint="Check the column.",
        )
    return np.asarray(values[keep[0] : keep[-1] + 1], dtype=float)


def garch(
    y: object,
    p: int = 1,
    q: int = 1,
    mean: bool = True,
    presample: str = "stata",
    vce: str = "oim",
    *,
    ar: int = 0,
    dist: str = "normal",
    model: str = "garch",
    threshold: Optional[int] = None,
    in_mean: Any = False,
    in_mean_lags: Any = 0,
    data: Optional[pd.DataFrame] = None,
) -> GARCHResult:
    """Fit GARCH(p,q) by conditional maximum likelihood.

    Parameters
    ----------
    y : array-like or str
        Return series (or log-return, etc.); a column name when ``data`` is
        given.
    data : pandas.DataFrame, optional
        Frame holding the column ``y``, in time order. Leading and trailing
        missing values (the first row of a differenced series) are dropped.
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
    ar : int, default 0
        Autoregressive terms of the mean equation, written as AR
        disturbances: ``y_t = mu + u_t``, ``u_t = sum_k rho_k u_{t-k} +
        eps_t`` (Stata ``arch y, ar(1/k)``; ``rugarch`` ``armaOrder =
        c(k, 0)``). Pre-sample disturbances are zero, so every observation
        enters the likelihood.
    dist : {'normal', 't'}, default 'normal'
        Distribution of the standardised innovation ``z_t``. ``'t'`` is
        Student t scaled to unit variance, with the degrees of freedom
        ``nu > 2`` estimated (Stata ``distribution(t)``, ``rugarch``
        ``'std'``). Stata reports ``ln(nu - 2)``; ``nu`` itself is the
        parameter here.
    model : {'garch', 'gjr', 'egarch'}, default 'garch'
        The variance equation.

        ``'gjr'`` (threshold GARCH of Glosten, Jagannathan and Runkle):
        ``sigma2_t = omega + sum_i (alpha_i + gamma_i 1[eps_{t-i} < 0])
        eps2_{t-i} + sum_j beta_j sigma2_{t-j}``. ``gamma > 0`` is the
        leverage effect: bad news raises the variance more than good news.
        Stata's ``arch, arch() tarch() garch()`` writes the threshold term
        on positive shocks, so ``gamma = -tarch`` and ``alpha = arch +
        tarch``; the likelihood is the same.

        ``'egarch'`` (Nelson's exponential GARCH): ``ln sigma2_t = omega +
        sum_i [theta_i z_{t-i} + gamma_i (|z_{t-i}| - sqrt(2 / pi))] +
        sum_j beta_j ln sigma2_{t-j}`` with ``z = eps / sigma``.
        ``theta < 0`` is the leverage effect. Stata's ``earch()`` is
        ``theta`` and its ``earch_a`` is ``gamma``; ``q`` counts the shock
        lags and ``p`` the lagged log variances.
    threshold : int, optional
        ``model='gjr'``: number of threshold terms, ``1 <= threshold <=
        q`` (default ``q``, one per ARCH lag; Stata ``tarch(1/k)``).
    in_mean : bool or {'variance', 'sd', 'log'}, default False
        Let the conditional variance enter the mean: ``y_t = mu + archm
        g(sigma2_t) + u_t`` (ARCH in mean; Stata's ``archm``). ``True`` or
        ``'variance'``: ``g`` is the identity; ``'sd'``: the conditional
        standard deviation (Stata ``archmexp(sqrt(X))``); ``'log'``: the
        log variance (``archmexp(ln(X))``). A positive ``archm`` is a risk
        premium. Works with every ``model``, with ``ar=`` and with
        ``dist='t'``.
    in_mean_lags : int or sequence of int, default 0
        Which variances enter the mean: ``0`` the current one, ``[0, 1]``
        the current one and its first lag (Stata ``archm archmlags(1)``),
        ``[1]`` the first lag alone (``archmlags(1)`` without ``archm``).
        The coefficients are named ``archm`` and ``archm[L1]``, ...

    Notes
    -----
    The likelihood surface of a model with more than one lag has ridges
    and boundary solutions, so the search runs a bounded quasi-Newton from
    several starting values (different splits of the persistence between
    the ARCH and GARCH terms and across the lags) besides the simplex of
    earlier releases, and keeps the best. The Gaussian constant-mean model
    is then polished by Newton steps on the analytic score. ω > 0,
    α, β >= 0 and Σα + Σβ < 1 are enforced by an infinite objective
    outside the region. An estimate that ends on the boundary (a
    coefficient at zero) raises a ``RuntimeWarning``: its standard error is
    not meaningful and the lower-order model fits as well.

    With ``ar > 0``, ``dist='t'`` or an asymmetric ``model`` the scores
    and the Hessian are numerical (central differences), so standard errors
    agree with Stata to about four or five significant digits rather than
    to rounding.

    Asymmetric models start the recursion as Stata does: pre-sample squared
    shocks and variances at the mean squared residual, no pre-sample
    threshold or EGARCH shock term. EGARCH centres ``|z|`` at
    ``sqrt(2 / pi)``, its mean under normality, under Student t innovations
    too (as Stata does); the difference is absorbed by ``omega``.
    With ``in_mean=True`` the estimator is Stata's: the pre-sample
    variance equals the mean squared innovation at the estimates, and is a
    constant while the likelihood is climbed. (Letting it move with the
    parameters during the climb gives another point, within a hundredth
    of a standard error in the cases examined; the mean and ``archm`` are
    nearly collinear, so the likelihood is flat between the two.)

    A variance dated before the sample is that pre-sample value, and a
    transformed term uses its transform, ``g(m)``. Stata uses the
    untransformed ``m`` there, so with ``in_mean='sd'`` or ``'log'``
    *and* lags the first ``max(in_mean_lags)`` observations enter
    differently and the two likelihoods differ in the second decimal;
    without lags, or without a transform, they agree.

    ``'gjr'`` requires ``alpha >= 0``, ``alpha + gamma >= 0`` and
    ``sum(alpha) + sum(gamma) / 2 + sum(beta) < 1``; ``'egarch'`` only
    ``|sum(beta)| < 1``. The multi-step :meth:`GARCHResult.forecast` of an
    EGARCH model is the exponential of the forecast log variance, which is
    below the expected variance beyond one step. Stata does not impose the
    sign restrictions of ``'gjr'``; where its estimate has ``arch + tarch <
    0`` (the variance falling after a positive shock), ``sp.garch`` stops
    at ``alpha = 0`` and warns.

    Releases through 1.38.0 used a single simplex search. For GARCH(p,q)
    with ``p >= 2`` it could stop with a lagged-variance coefficient at
    zero and a log-likelihood below that of the nested GARCH(1,q).

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

    An AR(1) mean with Student t innovations, and the 1% value at risk of
    the next observation:

    >>> t_fit = sp.garch(eps, ar=1, dist="t")
    >>> list(t_fit.params.index)
    ['mu', 'ar[1]', 'omega', 'alpha[1]', 'beta[1]', 'nu']
    >>> bool(t_fit.value_at_risk(0.01) < 0)
    True

    A threshold model, in which a negative shock may move the variance by
    more than a positive one:

    >>> gjr = sp.garch(eps, model="gjr")
    >>> list(gjr.params.index)
    ['mu', 'omega', 'alpha[1]', 'gamma[1]', 'beta[1]']
    >>> print(res.summary())  # doctest: +SKIP

    References
    ----------
    [@engle1982autoregressive],
    [@bollerslev1986generalized],
    [@bollerslev1987conditionally],
    [@glosten1993relation],
    [@nelson1991conditional]
    """
    y = _series_argument(y, data, "garch")
    T = len(y)
    if p < 0 or q < 0 or (p == 0 and q == 0):
        raise MethodIncompatibility(
            "garch: p and q are non-negative lag counts, not both zero.",
            recovery_hint="GARCH(1,1) is p=1, q=1; ARCH(1) is p=0, q=1.",
        )
    if q == 0:
        raise MethodIncompatibility(
            "garch: with no ARCH term (q=0) the lagged-variance coefficients "
            "are not identified: the variance never responds to the data, so "
            "every beta gives the same constant variance.",
            recovery_hint="Use q >= 1; ARCH(1) is p=0, q=1 (note that p "
            "counts the lagged variances, q the lagged squared errors).",
        )
    if T < max(p, q) + 10:
        raise ValueError("Time series too short for GARCH estimation.")
    if presample not in ("stata", "rugarch"):
        raise MethodIncompatibility("presample must be 'stata' or 'rugarch'")
    if vce not in ("oim", "opg", "robust"):
        raise MethodIncompatibility("vce must be 'oim', 'opg' or 'robust'")
    if not np.all(np.isfinite(y)):
        raise MethodIncompatibility(
            "garch: y contains NaN or inf",
            recovery_hint="Drop or impute non-finite values of y.",
        )
    if dist not in ("normal", "t"):
        raise MethodIncompatibility(
            f"garch: dist={dist!r} is not 'normal' or 't'.",
            recovery_hint="dist='t' fits Student t innovations.",
        )
    n_ar = int(ar)
    if n_ar < 0 or n_ar != ar:
        raise MethodIncompatibility("garch: ar is a non-negative lag count.")
    if model not in ("garch", "gjr", "egarch"):
        raise MethodIncompatibility(
            f"garch: model={model!r} is not 'garch', 'gjr' or 'egarch'.",
            recovery_hint="'gjr' is the threshold model (Stata tarch), "
            "'egarch' Nelson's exponential model.",
        )
    if model != "garch" and presample != "stata":
        raise MethodIncompatibility(
            f"garch: presample={presample!r} is defined for model='garch' " "only.",
            recovery_hint="Leave presample at its default for 'gjr' and 'egarch'.",
        )
    kinds = {True: "variance", "variance": "variance", "sd": "sd", "log": "log"}
    if in_mean is not False and in_mean is not None and in_mean not in kinds:
        raise MethodIncompatibility(
            f"garch: in_mean={in_mean!r} is not True, 'variance', 'sd' or 'log'.",
            recovery_hint="'sd' puts the conditional standard deviation in "
            "the mean, 'log' the log variance.",
        )
    mean_kind = kinds[in_mean] if in_mean else None
    lag_list = (
        [int(in_mean_lags)]
        if isinstance(in_mean_lags, (int, np.integer))
        else [int(v) for v in in_mean_lags]
    )
    if mean_kind is None:
        lag_list = []
    elif (
        not lag_list
        or min(lag_list) < 0
        or len(set(lag_list)) != len(lag_list)
        or sorted(lag_list) != lag_list
    ):
        raise MethodIncompatibility(
            f"garch: in_mean_lags={in_mean_lags!r} must be distinct "
            "non-negative lags in increasing order.",
            recovery_hint="0 is the current variance; [0, 1] adds its first lag.",
        )
    code = {"variance": 0, "sd": 1, "log": 2}.get(mean_kind or "", 0)
    # what the filter needs: False, or (transform code, lags)
    in_mean = (code, tuple(lag_list)) if mean_kind else False
    n_psi = len(lag_list)
    if in_mean and presample != "stata":
        raise MethodIncompatibility(
            "garch: in_mean= is defined for presample='stata' only."
        )
    n_g = q if model != "garch" else 0  # asymmetry terms
    if threshold is not None:
        if model != "gjr":
            raise MethodIncompatibility(
                "garch: threshold= sets the number of threshold terms of "
                "model='gjr'.",
                recovery_hint="Pass model='gjr', or drop threshold=.",
            )
        if int(threshold) != threshold or not 1 <= threshold <= q:
            raise MethodIncompatibility(
                f"garch: threshold={threshold!r} must be between 1 and q={q}.",
                recovery_hint="A threshold term needs the ARCH term of the "
                "same lag.",
            )
        n_g = int(threshold)
    y_mean = float(y.mean()) if mean else 0.0
    j0 = int(mean) + n_psi  # position of the first AR coefficient
    jv = j0 + n_ar  # position of omega
    n_var = 1 + q + n_g + p
    K = jv + n_var + int(dist == "t")

    # in_mean: the pre-sample value held fixed during a round of the
    # optimisation (None: recomputed as the model's own fixed point)
    held: dict = {"m": None}

    def _filter(theta: np.ndarray) -> Any:
        return _general_filter(
            theta, y, p, q, mean, n_ar, dist, presample, model, n_g, in_mean,
            held["m"],
        )  # fmt: skip

    def _terms(theta: np.ndarray) -> np.ndarray:
        return np.asarray(_filter(theta)[3])

    def _feasible(theta: np.ndarray) -> bool:
        omega = theta[jv]
        a = theta[jv + 1 : jv + 1 + q]
        g = theta[jv + 1 + q : jv + 1 + q + n_g]
        b = theta[jv + 1 + q + n_g : jv + n_var]
        if dist == "t" and not theta[-1] > 2.0:
            return False
        if model == "egarch":
            # the log variance needs no sign restriction, only stationarity
            return bool(abs(b.sum()) < 1.0)
        if model == "gjr":
            # variance after a positive shock (alpha) and after a negative
            # one (alpha + gamma) both non-negative
            ok = np.all(a >= 0) and np.all(a[:n_g] + g >= 0) and np.all(b >= 0)
            return bool(omega > 0 and ok and a.sum() + 0.5 * g.sum() + b.sum() < 1.0)
        return bool(
            omega > 0 and np.all(a >= 0) and np.all(b >= 0) and a.sum() + b.sum() < 1.0
        )

    def _fast_neg_ll(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        if not _feasible(theta):
            return 1e15
        ll_t = _terms(theta)
        if not np.all(np.isfinite(ll_t)):
            return 1e15
        return float(-ll_t.sum())

    # -- search: quasi-Newton inside the bounds from several starts. A
    # single simplex from the equal-split start stops on the boundary of
    # higher-order models (a lagged-variance coefficient at zero) with a
    # likelihood below that of the nested lower-order model.
    eps0 = y - y_mean
    var0 = float(np.mean(eps0**2))
    rho0 = np.zeros(n_ar)
    if n_ar:
        lagged = np.column_stack(
            [np.concatenate([np.zeros(k), eps0[:-k]]) for k in range(1, n_ar + 1)]
        )
        rho0 = np.linalg.lstsq(lagged, eps0, rcond=None)[0]

    def _spread(total: float, n: int, shape: str) -> List[float]:
        if n == 0:
            return []
        w: np.ndarray
        if shape == "equal":
            w = np.ones(n)
        elif shape == "first":
            w = np.array([1.0] + [0.05] * (n - 1))
        else:  # geometric decay
            w = 0.5 ** np.arange(n)
        return list(total * w / w.sum())

    def _start(a_tot: float, b_tot: float, shape: str) -> np.ndarray:
        a = _spread(a_tot, q, shape)
        b = _spread(b_tot, p, shape)
        omega = var0 * max(1.0 - sum(a) - sum(b), 0.05)
        head = ([y_mean] if mean else []) + [0.0] * n_psi + list(rho0)
        tail = [8.0] if dist == "t" else []
        if model == "egarch":
            # no sign effect, a magnitude effect of twice the ARCH share,
            # and the unconditional log variance
            omega = float(np.log(var0)) * (1.0 - sum(b))
            mid = [0.0] * q + [2.0 * v for v in a]
        elif model == "gjr":
            # the same persistence, all of the news effect on bad news
            mid = [0.2 * v for v in a] + [1.6 * v for v in a[:n_g]]
        else:
            mid = a
        return np.asarray(head + [omega] + mid + b + tail, float)

    wide = [(None, None)]
    if model == "egarch":
        var_bounds = wide * (1 + 2 * q) + [(-0.9999, 0.9999)] * p
    elif model == "gjr":
        var_bounds = (
            [(1e-12 * max(var0, 1e-300), None)]
            + [(0.0, 0.9999)] * q
            + [(-0.9999, 1.9999)] * n_g
            + [(0.0, 0.9999)] * p
        )
    else:
        var_bounds = [(1e-12 * max(var0, 1e-300), None)] + [(0.0, 0.9999)] * (q + p)
    bounds = wide * jv + var_bounds + ([(2.01, 1e4)] if dist == "t" else [])
    grid = [(0.1, 0.8), (0.05, 0.9), (0.2, 0.6), (0.3, 0.3)]
    shapes = ["equal"] if max(p, q) <= 1 else ["equal", "first", "decay"]
    best_theta = _start(0.1, 0.8, "equal")
    best_fun = _fast_neg_ll(best_theta)
    for a_tot, b_tot in grid:
        for shape in shapes:
            x0 = _start(a_tot, b_tot, shape)
            res = minimize(
                _fast_neg_ll,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-8},
            )
            if np.isfinite(res.fun) and res.fun < best_fun - 1e-9:
                best_fun, best_theta = float(res.fun), np.asarray(res.x, float)

    general = bool(n_ar or dist == "t" or model != "garch" or n_psi)
    scores: np.ndarray
    if not general:

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
            return np.asarray(-sc.sum(axis=0))

        x0 = _start(0.1, 0.8, "equal")
        opt = minimize(
            _fast_neg_ll,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12},
        )
        theta = np.asarray(opt.x, dtype=float)
        opt2 = minimize(
            neg_ll, theta, jac=neg_grad, method="BFGS", options={"gtol": 1e-9}
        )
        if np.isfinite(opt2.fun) and opt2.fun <= opt.fun:
            theta = np.asarray(opt2.x, dtype=float)
        # the multi-start optimum replaces it only when it is better
        if best_fun < neg_ll(theta) - 1e-7:
            theta = best_theta

        def _hess(th: np.ndarray) -> np.ndarray:
            # Hessian of the NEGATIVE log-likelihood by central differences
            # of the analytic score.
            k = th.size
            H = np.empty((k, k))
            for i in range(k):
                h = 1e-5 * max(abs(th[i]), 1e-2)
                e = np.zeros(k)
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
        u = eps
        H = _hess(theta)
    else:
        theta = best_theta

        def _grad_norm(th: np.ndarray) -> float:
            return float(np.max(np.abs(_numeric_scores(_terms, th).sum(axis=0))))

        def _polish(th: np.ndarray) -> np.ndarray:
            for _ in range(3):
                # unconstrained polish, kept only when it improves the fit
                pol = minimize(_fast_neg_ll, th, method="BFGS", options={"gtol": 1e-7})
                if np.isfinite(pol.fun) and pol.fun <= _fast_neg_ll(th):
                    th = np.asarray(pol.x, float)
                if _grad_norm(th) < 1e-4:
                    break
                simplex = minimize(
                    _fast_neg_ll,
                    th,
                    method="Nelder-Mead",
                    options={"maxiter": 1000 * K, "xatol": 1e-9, "fatol": 1e-11},
                )
                if float(simplex.fun) < _fast_neg_ll(th):
                    th = np.asarray(simplex.x, float)
            return th

        theta = _polish(theta)
        if n_psi:
            # Stata's estimator: the pre-sample value is the mean squared
            # innovation at the estimates, but it is a constant while the
            # likelihood is climbed. Alternate until the two agree.
            for _ in range(50):
                held["m"] = None
                m_now = float(np.mean(_filter(theta)[1] ** 2))
                held["m"] = m_now
                theta = _polish(theta)
                m_next = float(np.mean(_filter(theta)[1] ** 2))
                if abs(m_next - m_now) <= 1e-11 * abs(m_now):
                    break
        s2, eps, u, ll_t = _filter(theta)
        scores = _numeric_scores(_terms, theta)
        H = _numeric_hessian(_fast_neg_ll, theta)

    mu = float(theta[0]) if mean else 0.0
    psi_all = np.asarray(theta[int(mean) : int(mean) + n_psi], float)
    psi_hat: Any = None
    if n_psi == 1:
        psi_hat = float(psi_all[0])
    elif n_psi > 1:
        psi_hat = psi_all
    psi_names = ["archm" if lag == 0 else f"archm[L{lag}]" for lag in lag_list]
    rho = np.asarray(theta[j0:jv], float)
    omega = float(theta[jv])
    first = np.asarray(theta[jv + 1 : jv + 1 + q], float)
    gamma_hat = np.asarray(theta[jv + 1 + q : jv + 1 + q + n_g], float)
    beta = theta[jv + 1 + q + n_g : jv + n_var]
    nu = float(theta[-1]) if dist == "t" else None
    # 'egarch' has no alpha: its first block multiplies the signed shock
    alpha = np.zeros(0) if model == "egarch" else first
    first_name = "theta" if model == "egarch" else "alpha"

    param_names = (
        (["mu"] if mean else [])
        + psi_names
        + [f"ar[{k + 1}]" for k in range(n_ar)]
        + ["omega"]
        + [f"{first_name}[{i + 1}]" for i in range(q)]
        + [f"gamma[{i + 1}]" for i in range(n_g)]
        + [f"beta[{j + 1}]" for j in range(p)]
        + (["nu"] if dist == "t" else [])
    )
    on_boundary = [
        nm
        for nm, val in zip(param_names[jv + 1 : jv + n_var], theta[jv + 1 :])
        if model != "egarch" and not nm.startswith("gamma") and val <= 1e-8
    ]
    if on_boundary:
        import warnings

        warnings.warn(
            "garch: the estimate of "
            + ", ".join(on_boundary)
            + " is on the boundary of the parameter space (zero); the "
            "standard errors and tests are not valid there. A lower-order "
            "model fits the data as well.",
            RuntimeWarning,
            stacklevel=2,
        )
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
    k_params = len(theta)
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
        ar=rho,
        nu=nu,
        dist=dist,
        model=model,
        gamma=gamma_hat if n_g else None,
        theta=first if model == "egarch" else None,
        archm=psi_hat,
        in_mean=mean_kind,
        in_mean_lags=tuple(lag_list),
    )
    _result.vce = vce
    _result.presample = presample
    _result.disturbances = np.asarray(u, float)
    _result.gradient_norm = float(np.max(np.abs(scores.sum(axis=0))))
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.timeseries.garch",
            params={
                "p": p,
                "q": q,
                "mean": mean,
                "presample": presample,
                "vce": vce,
                "ar": n_ar,
                "dist": dist,
                "model": model,
                "threshold": threshold,
                "in_mean": mean_kind,
                "in_mean_lags": lag_list,
            },
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
