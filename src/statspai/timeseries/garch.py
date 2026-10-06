"""GARCH(p,q) volatility models (Bollerslev, 1986).

Maximum-likelihood estimation of conditional variance models:

    r_t = μ + ε_t,   ε_t = σ_t z_t,   z_t ~ N(0,1)
    σ²_t = ω + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j}

The most common specification is GARCH(1,1) where
    σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}

and α + β < 1 for stationarity.

The mean may carry autoregressive disturbances (``ar=``) and the
standardised innovations may be Student t (``dist='t'``).

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

    def forecast_mean(self, horizon: int = 1) -> np.ndarray:
        """Multi-step forecast of the series: the constant mean plus the
        autoregressive forecast of the disturbance (``ar=``)."""
        rho = np.zeros(0) if self.ar is None else np.asarray(self.ar, float)
        source = self.residuals if self.disturbances is None else self.disturbances
        u = list(np.asarray(source, float))
        out = np.empty(horizon)
        for h in range(horizon):
            nxt = float(sum(rho[k] * u[-1 - k] for k in range(rho.size)))
            u.append(nxt)
            out[h] = self.mu + nxt
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
        head = f"GARCH({self.p},{self.q})"
        n_ar = 0 if self.ar is None else int(np.size(self.ar))
        if n_ar:
            head = f"AR({n_ar})-" + head
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
) -> np.ndarray:
    """Conditional variances by linear filtering; the same recursion and
    pre-sample rules as :func:`_garch_filter`, without the Python loop."""
    from scipy.signal import lfilter, lfiltic

    T = e2.shape[0]
    q, p = alpha.shape[0], beta.shape[0]
    m = float(e2.mean())
    c = np.full(T, omega)
    if q:
        ext = np.concatenate([np.full(q, m), e2])
        kernel = np.concatenate([[0.0], alpha])
        c = c + np.convolve(ext, kernel)[q : q + T]
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


def _general_filter(
    theta: np.ndarray,
    y: np.ndarray,
    p: int,
    q: int,
    mean: bool,
    n_ar: int,
    dist: str,
    presample: str,
) -> Any:
    """Variances, innovations, disturbances and log-likelihood terms of the
    model with AR(``n_ar``) disturbances and normal or Student t errors.

    theta = (mu?, rho_1..rho_k, omega, alpha_1..alpha_q, beta_1..beta_p,
    nu?). The disturbance ``u_t = y_t - mu`` follows
    ``u_t = sum rho_k u_{t-k} + eps_t`` with pre-sample ``u`` equal to 0.
    """
    from scipy.signal import lfilter
    from scipy.special import gammaln

    j = int(mean)
    mu = float(theta[0]) if mean else 0.0
    rho = theta[j : j + n_ar]
    j += n_ar
    omega = float(theta[j])
    alpha = theta[j + 1 : j + 1 + q]
    beta = theta[j + 1 + q : j + 1 + q + p]
    u = y - mu
    eps = lfilter(np.concatenate([[1.0], -rho]), [1.0], u) if n_ar else u
    e2 = eps * eps
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

    With ``ar > 0`` or ``dist='t'`` the scores and the Hessian are
    numerical (central differences), so standard errors agree with Stata
    to about five significant digits rather than to rounding.

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
    >>> print(res.summary())  # doctest: +SKIP

    References
    ----------
    [@engle1982autoregressive],
    [@bollerslev1986generalized],
    [@bollerslev1987conditionally]
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
    y_mean = float(y.mean()) if mean else 0.0
    j0 = int(mean)
    jv = j0 + n_ar  # position of omega
    n_var = 1 + q + p
    K = jv + n_var + int(dist == "t")

    def _terms(theta: np.ndarray) -> np.ndarray:
        return np.asarray(
            _general_filter(theta, y, p, q, mean, n_ar, dist, presample)[3]
        )

    def _feasible(theta: np.ndarray) -> bool:
        omega = theta[jv]
        ab = theta[jv + 1 : jv + n_var]
        if dist == "t" and not theta[-1] > 2.0:
            return False
        return bool(omega > 0 and np.all(ab >= 0) and ab.sum() < 1.0)

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
        head = ([y_mean] if mean else []) + list(rho0)
        tail = [8.0] if dist == "t" else []
        return np.asarray(head + [omega] + a + b + tail, float)

    bounds = (
        [(None, None)] * jv
        + [(1e-12 * max(var0, 1e-300), None)]
        + [(0.0, 0.9999)] * (q + p)
        + ([(2.01, 1e4)] if dist == "t" else [])
    )
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

    general = bool(n_ar or dist == "t")
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

        for _ in range(3):
            # unconstrained polish, kept only when it improves the fit
            pol = minimize(_fast_neg_ll, theta, method="BFGS", options={"gtol": 1e-7})
            if np.isfinite(pol.fun) and pol.fun <= _fast_neg_ll(theta):
                theta = np.asarray(pol.x, float)
            if _grad_norm(theta) < 1e-4:
                break
            simplex = minimize(
                _fast_neg_ll,
                theta,
                method="Nelder-Mead",
                options={"maxiter": 1000 * K, "xatol": 1e-9, "fatol": 1e-11},
            )
            if float(simplex.fun) < _fast_neg_ll(theta):
                theta = np.asarray(simplex.x, float)
        s2, eps, u, ll_t = _general_filter(theta, y, p, q, mean, n_ar, dist, presample)
        scores = _numeric_scores(_terms, theta)
        H = _numeric_hessian(_fast_neg_ll, theta)

    mu = float(theta[0]) if mean else 0.0
    rho = np.asarray(theta[j0:jv], float)
    omega = float(theta[jv])
    alpha = theta[jv + 1 : jv + 1 + q]
    beta = theta[jv + 1 + q : jv + 1 + q + p]
    nu = float(theta[-1]) if dist == "t" else None

    param_names = (
        (["mu"] if mean else [])
        + [f"ar[{k + 1}]" for k in range(n_ar)]
        + ["omega"]
        + [f"alpha[{i + 1}]" for i in range(q)]
        + [f"beta[{j + 1}]" for j in range(p)]
        + (["nu"] if dist == "t" else [])
    )
    on_boundary = [
        nm
        for nm, val in zip(param_names[jv + 1 : jv + n_var], theta[jv + 1 :])
        if val <= 1e-8
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
