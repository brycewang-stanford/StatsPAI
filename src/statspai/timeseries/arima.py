"""ARIMA(p,d,q) / SARIMAX wrapper with automatic order selection.

Wraps ``statsmodels.tsa.statespace.SARIMAX`` to provide a StatsPAI-style
interface: formula-like specification, ``.summary()``, ``.forecast()``,
``.plot()``, and automatic (p,d,q) selection via AICc grid search.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> import statspai as sp
>>> rng = np.random.default_rng(0)
>>> e = rng.normal(size=150)
>>> dy = np.zeros(150)
>>> for t in range(1, 150):
...     dy[t] = 0.3 + 0.5 * dy[t - 1] + e[t] + 0.3 * e[t - 1]
>>> df = pd.DataFrame({"gdp": 100 + np.cumsum(dy)})
>>> result = sp.arima(df["gdp"], order=(1, 1, 1))
>>> fc = result.forecast(horizon=12)
>>> fc.shape, list(fc.columns)
((12, 3), ['forecast', 'lower', 'upper'])
>>> ax = result.plot()
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from .ts_tools import _kpss_level as _kpss_stat  # noqa: F401  (re-exported)
from .ts_tools import _ndiffs as _kpss_ndiffs
from .ts_tools import _nsdiffs


@dataclass
class ARIMAResult(ResultProtocolMixin):
    """Fitted ARIMA(p,d,q) / SARIMAX model returned by :func:`statspai.arima`.

    Carries the estimated ``params`` / ``se`` (indexed by parameter name),
    information criteria (``aic`` / ``bic`` / ``aicc``), the log-likelihood,
    residuals and fitted values, plus inference accessors
    (:attr:`tvalues`, :attr:`pvalues`, :meth:`conf_int`) and a
    ``.summary()`` / ``.forecast()`` interface.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.6 * y[t - 1] + rng.normal(0, 1)
    >>> res = sp.arima(y, order=(1, 0, 0))
    >>> type(res).__name__
    'ARIMAResult'
    >>> bool(np.isfinite(res.aic))
    True
    >>> list(res.params.index)
    ['const', 'ar.L1', 'sigma2']
    """

    order: Tuple[int, int, int]
    seasonal_order: Optional[Tuple[int, int, int, int]]
    params: pd.Series
    se: pd.Series  # asymptotic standard errors (param index)
    aic: float
    bic: float
    aicc: float
    log_likelihood: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    n: int
    _model: Any  # statsmodels result (opaque)
    exog_names: Tuple[str, ...] = ()
    candidates: Optional[pd.DataFrame] = None
    _index: Optional[pd.Index] = None
    #: deterministic regressor the fit carries in the level equation:
    #: ``"const"`` (a column of ones), ``"drift"`` (the time index) or None
    _det: Optional[str] = None
    #: Box-Cox parameter the series was modelled under, and whether point
    #: forecasts are bias-adjusted on the way back
    boxcox: Optional[float] = None
    biasadj: bool = False

    @property
    def period(self) -> int:
        """Seasonal period (1 for a non-seasonal model)."""
        return int(self.seasonal_order[3]) if self.seasonal_order else 1

    @property
    def n_arma_params(self) -> int:
        """Number of estimated ARMA coefficients, ``p + q + P + Q``: the
        degrees of freedom a portmanteau test of the residuals gives up."""
        k = int(self.order[0]) + int(self.order[2])
        if self.seasonal_order:
            k += int(self.seasonal_order[0]) + int(self.seasonal_order[2])
        return k

    @property
    def sigma2(self) -> float:
        """Maximum likelihood estimate of the innovation variance."""
        return float(self.params["sigma2"])

    # --- inference accessors -------------------------------------------------
    @property
    def std_errors(self) -> pd.Series:
        """Alias for :attr:`se` (regression-style naming)."""
        return self.se

    @property
    def tvalues(self) -> pd.Series:
        """z-statistics ``params / se`` (SARIMAX uses a normal reference)."""
        return self.params / self.se

    @property
    def pvalues(self) -> pd.Series:
        """Two-sided p-values from the normal reference distribution."""
        from scipy import stats

        z = (self.params / self.se).to_numpy()
        return pd.Series(
            2.0 * stats.norm.sf(np.abs(z)),
            index=self.params.index,
        )

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """Confidence intervals for each parameter.

        Parameters
        ----------
        alpha : float, default 0.05
            ``1 - alpha`` is the coverage (0.05 → 95% CI).

        Returns
        -------
        pd.DataFrame
            Indexed by parameter name with ``lower`` / ``upper`` columns.
        """
        from scipy import stats

        z = stats.norm.ppf(1.0 - alpha / 2.0)
        lower = self.params - z * self.se
        upper = self.params + z * self.se
        return pd.DataFrame(
            {"lower": lower, "upper": upper},
            index=self.params.index,
        )

    def forecast(
        self,
        horizon: int = 10,
        alpha: float = 0.05,
        *,
        level: Optional[Any] = None,
        exog: Optional[Any] = None,
        dof_adjust: bool = False,
    ) -> pd.DataFrame:
        """Forecasts with prediction intervals.

        Parameters
        ----------
        horizon : int, default 10
            Number of periods ahead.
        alpha : float, default 0.05
            One minus the coverage of the interval, used when ``level``
            is not given; the columns are then ``forecast``, ``lower``,
            ``upper``.
        level : float or sequence of float, optional
            Coverage in percent, e.g. ``(80, 95)``; the columns are then
            ``forecast``, ``lower_80``, ``upper_80``, ``lower_95``,
            ``upper_95``, the layout of :func:`statspai.ets`.
        exog : array-like or pd.DataFrame, optional
            Values of the regressors over the forecast periods, one row
            per period, columns in the order of the fit (a DataFrame is
            matched by column name). Required when the model has
            regressors: the forecast is conditional on them.
        dof_adjust : bool, default False
            Scale the innovation variance by ``n / (n - k)``, ``n`` the
            number of observations after differencing and ``k`` the
            number of estimated coefficients, as R's ``forecast::Arima``
            and ``fable::ARIMA`` do. The default uses the maximum
            likelihood variance, as ``stats::arima`` and Stata do.

        Returns
        -------
        pd.DataFrame
            Indexed by the forecast periods when the series had a date
            or period index.

        Notes
        -----
        The intervals treat the coefficients, and the future regressors,
        as known.
        """
        h = int(horizon)
        if h < 1:
            raise MethodIncompatibility(
                f"forecast: horizon must be at least 1, got {horizon!r}.",
                recovery_hint="Pass a positive number of periods.",
            )
        x_future = None
        k_exog = len(self.exog_names)
        if k_exog:
            if exog is None:
                raise MethodIncompatibility(
                    "forecast: the model has regressors "
                    f"({', '.join(self.exog_names)}); their values over the "
                    "forecast periods are needed.",
                    recovery_hint=(
                        "Pass exog= with one row per forecast period, e.g. a "
                        "scenario or the regressors' own forecasts."
                    ),
                    diagnostics={"exog_names": list(self.exog_names)},
                )
            if isinstance(exog, pd.DataFrame):
                missing = [c for c in self.exog_names if c not in exog.columns]
                if missing and exog.shape[1] != k_exog:
                    raise MethodIncompatibility(
                        f"forecast: exog lacks the regressors {missing}.",
                        recovery_hint="Give the columns the fit used.",
                    )
                x_future = (
                    exog.to_numpy(dtype=float)
                    if missing
                    else exog[list(self.exog_names)].to_numpy(dtype=float)
                )
            else:
                x_future = np.asarray(exog, dtype=float)
                if x_future.ndim == 1:
                    x_future = x_future.reshape(-1, k_exog)
            if x_future.shape != (h, k_exog):
                raise MethodIncompatibility(
                    f"forecast: exog has shape {x_future.shape}; "
                    f"({h}, {k_exog}) is needed.",
                    recovery_hint="One row per forecast period, one column per "
                    "regressor.",
                )
            if not np.isfinite(x_future).all():
                raise MethodIncompatibility(
                    "forecast: exog has missing values.",
                    recovery_hint="Every future regressor value must be given.",
                )
        elif exog is not None:
            raise MethodIncompatibility(
                "forecast: exog was given but the model has no regressors.",
                recovery_hint="Drop exog=.",
            )
        if self._det is not None:
            det = (
                np.ones(h)
                if self._det == "const"
                else np.arange(self.n + 1, self.n + h + 1, dtype=float)
            )
            x_future = (
                det.reshape(-1, 1)
                if x_future is None
                else np.column_stack([det, x_future])
            )
        fc = self._model.get_forecast(steps=h, exog=x_future)
        pred = np.asarray(fc.predicted_mean, dtype=float).ravel()
        sd = np.asarray(fc.se_mean, dtype=float).ravel()
        if dof_adjust:
            n_star = self.n - int(self.order[1])
            if self.seasonal_order:
                n_star -= int(self.seasonal_order[1]) * int(self.seasonal_order[3])
            k = len(self.params) - 1
            if n_star - k > 0:
                sd = sd * np.sqrt(n_star / (n_star - k))
        from ._forecast_common import (
            back_transform_frame,
            forecast_frame,
            future_index,
            normalise_levels,
        )

        idx = future_index(self._index, self.n, h)
        if level is not None:
            out = forecast_frame(pred, normalise_levels(level), sd=sd, index=idx)
            if self.boxcox is not None:
                out = back_transform_frame(out, self.boxcox, self.biasadj)
            return out
        from scipy import stats

        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        out = pd.DataFrame(
            {"forecast": pred, "lower": pred - z * sd, "upper": pred + z * sd}
        )
        if self.boxcox is not None:
            out.attrs["level"] = 100.0 * (1.0 - alpha)
            out = back_transform_frame(out, self.boxcox, self.biasadj)
        if not isinstance(idx, pd.RangeIndex):
            out.index = idx
        return out

    def plot(
        self,
        horizon: int = 20,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
    ) -> Any:
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        T = np.arange(self.n)
        ax.plot(
            T,
            self.fitted_values,
            color="C0",
            linewidth=0.8,
            label="fitted",
        )
        ax.plot(
            T,
            self.fitted_values + self.residuals,
            ".",
            color="grey",
            markersize=2,
            alpha=0.5,
            label="observed",
        )
        if self.exog_names:
            ax.legend()
            return ax
        fc = self.forecast(horizon, alpha).reset_index(drop=True)
        T_fc = np.arange(self.n, self.n + horizon)
        ax.plot(T_fc, fc["forecast"], "-", color="C3", label="forecast")
        ax.fill_between(T_fc, fc["lower"], fc["upper"], color="C3", alpha=0.2)
        ax.legend()
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        return ax

    def summary(self) -> str:
        lines = [
            f"ARIMA{self.order}"
            + (f" x {self.seasonal_order}" if self.seasonal_order else "")
            + (" with regressors" if self.exog_names else ""),
            "-" * 40,
            f"n          : {self.n}",
            f"AIC        : {self.aic:.2f}",
            f"BIC        : {self.bic:.2f}",
            f"AICc       : {self.aicc:.2f}",
            f"Log-Lik    : {self.log_likelihood:.2f}",
            "",
            (
                f"  {'':<15s}  {'coef':>10s}  {'std err':>10s}"
                f"  {'z':>8s}  {'P>|z|':>8s}"
            ),
        ]
        pvals = self.pvalues
        for nm, val in self.params.items():
            s = float(self.se.get(nm, np.nan))
            z = val / s if s and np.isfinite(s) else np.nan
            p = float(pvals.get(nm, np.nan))
            lines.append(
                f"  {nm:<15s}  {val:>10.4f}  {s:>10.4f}" f"  {z:>8.3f}  {p:>8.3f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _invert_ma(theta: np.ndarray) -> np.ndarray:
    """Coefficients of the invertible moving-average polynomial with the
    same autocovariances: roots inside the unit circle are reflected."""
    q = theta.shape[0]
    if q == 0:
        return theta
    roots = np.roots(np.r_[theta[::-1], 1.0])
    if not roots.size or np.all(np.abs(roots) > 1.0):
        return theta
    inside = np.abs(roots) < 1.0
    roots = np.where(inside, 1.0 / np.conj(roots), roots)
    poly = np.array([1.0 + 0.0j])
    for r in roots:
        poly = np.convolve(poly, np.array([1.0, -1.0 / r]))
    return np.real(poly[1:])


def _css_start(
    w: np.ndarray,
    Xd: Optional[np.ndarray],
    p: int,
    q: int,
    P: int,
    Q: int,
    s: int,
) -> Optional[np.ndarray]:
    """Conditional-sum-of-squares estimates of a (seasonal) ARMA model
    with regressors for the differenced series ``w``: a second starting
    point for the likelihood search, as in R's ``arima(method="CSS-ML")``.

    Returns the vector ``[beta, ar, ma, seasonal ar, seasonal ma]``, or
    ``None`` when the estimates are not a stationary model.
    """
    from scipy import optimize
    from scipy.signal import lfilter, lfiltic

    n = w.shape[0]
    k_x = 0 if Xd is None else Xd.shape[1]
    n_cond = p + P * s
    if p + q + P + Q == 0 or n - n_cond < p + q + P + Q + k_x + 2:
        return None
    beta0 = np.linalg.lstsq(Xd, w, rcond=None)[0] if Xd is not None else np.empty(0)

    def polys(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ar = np.r_[1.0, -v[:p]]
        ma = np.r_[1.0, v[p : p + q]]
        if P:
            sar = np.zeros(P * s + 1)
            sar[0] = 1.0
            sar[s::s] = -v[p + q : p + q + P]
            ar = np.convolve(ar, sar)
        if Q:
            sma = np.zeros(Q * s + 1)
            sma[0] = 1.0
            sma[s::s] = v[p + q + P :]
            ma = np.convolve(ma, sma)
        return ar, ma

    def objective(z: np.ndarray) -> float:
        u = w - Xd @ z[:k_x] if Xd is not None else w
        ar, ma = polys(z[k_x:])
        with np.errstate(all="ignore"):
            if n_cond:
                zi = lfiltic(ar, ma, y=np.zeros(len(ma) - 1), x=u[n_cond - 1 :: -1])
                e = lfilter(ar, ma, u[n_cond:], zi=zi)[0]
            else:
                e = lfilter(ar, ma, u)
            sse = float(e @ e)
        if not np.isfinite(sse) or sse <= 0.0:
            return 1.0e10
        return 0.5 * float(np.log(sse / (n - n_cond)))

    z0 = np.r_[beta0, np.zeros(p + q + P + Q)]
    try:
        res = optimize.minimize(objective, z0, method="BFGS", options={"maxiter": 300})
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None
    z = np.asarray(res.x, dtype=float)
    if not np.isfinite(z).all() or objective(z) >= 1.0e10:
        return None
    v = z[k_x:]
    phi, theta = v[:p], v[p : p + q]
    Phi, Theta = v[p + q : p + q + P], v[p + q + P :]
    for coefs in (phi, Phi):
        if coefs.size:
            roots = np.roots(np.r_[-coefs[::-1], 1.0])
            if roots.size and np.min(np.abs(roots)) <= 1.0 + 1e-6:
                return None
    theta, Theta = _invert_ma(theta), _invert_ma(Theta)
    for coefs in (theta, Theta):
        if coefs.size:
            roots = np.roots(np.r_[coefs[::-1], 1.0])
            if roots.size and np.min(np.abs(roots)) <= 1.0 + 1e-6:
                return None
    return np.asarray(np.r_[z[:k_x], phi, theta, Phi, Theta], dtype=float)


def _exact_diffuse(model: Any) -> int:
    """Start the integration states from an exact diffuse prior.

    statsmodels starts them from a normal prior of variance 1e6, the
    "approximate diffuse" initialisation. That prior is diffuse only
    against an innovation variance far below 1e6: on a series measured
    in the thousands it is informative, and the filtered states, the
    fitted values and the forecasts then depend on the unit of
    measurement. The exact initialisation of Durbin and Koopman has no
    such constant. The ``d + s D`` observations that identify the
    integration states are left out of the likelihood, which is then the
    likelihood of the differenced series (R ``stats::arima``, Stata
    ``arima``).
    """
    k_diff = int(model._k_states_diff)
    if k_diff > 0:
        from statsmodels.tsa.statespace.initialization import Initialization

        init = Initialization(model.k_states)
        init.set((0, k_diff), "diffuse")
        if model.k_states > k_diff:
            init.set((k_diff, model.k_states), "stationary")
        model.ssm.initialize(init)
        model._manual_initialization = True
        model.ssm.loglikelihood_burn = k_diff
    return k_diff


def _near_unit_root(res: Any, tol: float = 1.01) -> bool:
    """Whether a fitted model has an autoregressive or moving-average root
    of modulus below ``tol``: nearly non-stationary or nearly
    non-invertible, which the automatic search does not return.

    Hyndman and Khandakar (2008) state the bound as 1.001; R's
    ``auto.arima`` (forecast 9.0.2) rejects a model whose smallest root has
    modulus 1.0058, consistent with a bound of 1.01, which is used here."""
    for attr in ("arroots", "maroots"):
        roots = getattr(res, attr, None)
        if roots is None:
            continue
        roots = np.asarray(roots)
        if roots.size and float(np.min(np.abs(roots))) < tol:
            return True
    return False


def arima(
    y: Any,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    exog: Optional[Any] = None,
    auto: bool = False,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
    method: str = "statespace",
    trend: Optional[str] = None,
    *,
    period: Optional[int] = None,
    stepwise: bool = True,
    max_P: int = 2,
    max_Q: int = 2,
    max_D: int = 1,
    boxcox: Any = None,
    biasadj: bool = False,
    data: Optional[pd.DataFrame] = None,
) -> ARIMAResult:
    """Fit ARIMA(p,d,q) or SARIMAX.

    Parameters
    ----------
    y : array-like, pd.Series or str
        The series; a column name when ``data`` is given.
    order : (p, d, q)
    seasonal_order : (P, D, Q, s), optional
    exog : array-like, pd.DataFrame or list of str, optional
        Regressors: the model is a regression with ARIMA errors
        (dynamic regression). Column names when ``data`` is given.
        Forecasting then needs their future values,
        ``result.forecast(h, exog=...)``.
    data : pandas.DataFrame, optional
        Frame holding ``y`` (and ``exog``), in time order. Rows before the
        first and after the last non-missing ``y`` are dropped, so a
        differenced column can be passed as it is.
    auto : bool, default False
        If True, select the model and ignore ``order``, by the algorithm
        of Hyndman and Khandakar (2008):

        1. With ``period=``, the number of seasonal differences ``D``
           (at most ``max_D``) is one when the strength of seasonality
           of an STL decomposition exceeds 0.64
           (:func:`statspai.nsdiffs`).
        2. ``d`` is the number of differences after which a KPSS test no
           longer rejects level stationarity at 5%, at most ``max_d``
           (:func:`statspai.ndiffs`), applied to the seasonally
           differenced series.
        3. ``(p, q)``, with ``period=`` also ``(P, Q)``, and whether to
           include a constant (a mean when ``d + D = 0``, a drift when
           ``d + D = 1``) minimise AICc.

        With regressors, the differencing rules are applied to the
        residuals of the regression of ``y`` on them.
    period : int, optional
        Seasonal period for ``auto=True`` (4 quarterly, 12 monthly): the
        seasonal orders are then chosen too. Not needed when
        ``seasonal_order`` is given, which keeps the seasonal part fixed
        and searches ``(p, q)`` only.
    stepwise : bool, default True
        Search the neighbours of the best model so far, starting from
        four standard models, until none is better. ``False`` fits every
        model with ``p + q + P + Q <= 5``: slower, and not trapped by
        the path of the stepwise search.
    max_p, max_q, max_d, max_P, max_Q, max_D : int
        Bounds of the search.
    boxcox : float or "auto", optional
        Model a Box-Cox transform of the series (0 is the logarithm;
        ``"auto"`` uses :func:`statspai.boxcox_lambda`). Coefficients,
        residuals and the likelihood are on the transformed scale;
        ``fitted_values`` and forecasts are back-transformed.
    biasadj : bool, default False
        The back-transformed point forecast is the median of the forecast
        distribution; ``True`` returns its mean instead.
    method : {'statespace', 'css_ml', 'innovations_mle'}, default 'statespace'
        How the exact Gaussian likelihood is maximised. Both conventions
        maximise the same likelihood, of every observation, with the ARMA
        part started from its stationary distribution; they agree with
        Stata's ``arima``, R's ``stats::arima(method="ML")`` and
        statsmodels' ``ARIMA`` up to optimiser tolerance. ``'statespace'``
        runs a quasi-Newton search on the Kalman-filter likelihood and
        handles seasonal terms and regressors. ``'innovations_mle'``
        (alias ``'css_ml'``) uses the innovations algorithm, which is
        tighter for pure ARMA models and is what the cross-language parity
        rows run.

        Releases through 1.38.0 started ``'statespace'`` from a diffuse prior
        and left the first ``max(p, q + 1)`` observations out of the
        likelihood. The estimates were not the exact MLE, and ``auto=True``
        compared models scored on different numbers of observations.

        With differencing, the likelihood is that of the differenced
        series: the ``d + s D`` observations that pin down the integration
        states are left out, and those states start from an exact diffuse
        prior when the fitted values and forecasts are computed. Releases
        through 1.38.0 used statsmodels' default, a normal prior of
        variance 1e6. On a series whose innovation variance is not far
        below 1e6 that prior is informative: the estimates of a
        differenced model with AR or MA terms changed with the unit of
        measurement, and the reported log-likelihood was too low. ``AIC``,
        ``BIC`` and ``AICc`` count the estimated parameters only and use
        the number of observations after differencing, as R and Stata do.

    trend : {None, 'c', 'n'}, optional
        Deterministic term. ``'c'`` estimates a constant: the mean of the
        series when it is not differenced (``const``), the mean of the
        differenced series -- a drift in the level -- when ``d + D = 1``
        (``drift``). ``'n'`` estimates neither. The default follows R's
        ``stats::arima`` and statsmodels' ``ARIMA``: a constant without
        differencing, none with it; with ``auto=True`` the default lets
        AICc decide. Stata's ``arima`` always includes the constant; pass
        ``trend='c'`` to reproduce it on a differenced series. A constant
        with ``d + D > 1`` is a polynomial trend and is refused.

    Returns
    -------
    ARIMAResult
        Exposes ``params`` and the matching standard errors ``se`` (alias
        ``std_errors``), plus ``tvalues``, ``pvalues``, and
        ``conf_int(alpha)`` for inference, alongside ``aic`` / ``bic`` /
        ``aicc`` / ``log_likelihood`` and ``forecast`` / ``plot``.
        ``candidates`` lists the models an automatic search compared.

    Notes
    -----
    ``aicc`` uses the number of observations left after differencing,
    ``n - d - D s``, as R's ``forecast::Arima`` does. Releases through
    1.38.0 subtracted ``d`` only, which made the AICc of a seasonally
    differenced model slightly too small.

    The automatic search fits every candidate by exact maximum
    likelihood. R's ``auto.arima`` approximates the likelihood during
    the search on long or seasonal series unless called with
    ``approximation=FALSE``, and can therefore settle on another model
    there.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> e = rng.normal(size=n)
    >>> y = np.zeros(n)
    >>> for t in range(2, n):
    ...     y[t] = 0.5 * y[t - 1] - 0.2 * y[t - 2] + e[t]
    >>> gdp = pd.Series(y, name="gdp")
    >>> res = sp.arima(gdp, order=(2, 0, 0))
    >>> bool((res.se > 0).all())  # standard errors indexed by parameter name
    True
    >>> res.conf_int().shape  # 95% CIs: const, two AR terms, sigma2
    (4, 2)

    A regression with ARIMA errors, forecast under a scenario for the
    regressor:

    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1.0 + 0.8 * x + y, "x": x})
    >>> dyn = sp.arima("y", order=(2, 0, 0), exog=["x"], data=df)
    >>> "x" in dyn.params.index
    True
    >>> dyn.forecast(4, level=(80, 95), exog=np.zeros((4, 1))).shape
    (4, 5)

    References
    ----------
    hyndman2008automatic, hyndman2026fpppy
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA as SMARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as e:
        raise ImportError(
            "statsmodels is required for arima(). "
            "Install with `pip install statsmodels`."
        ) from e

    index: Optional[pd.Index] = None
    exog_names: Tuple[str, ...] = ()
    if data is not None:
        if not isinstance(y, str) or y not in data.columns:
            raise MethodIncompatibility(
                "arima: with data=, y must be the name of one of its columns.",
                recovery_hint="Check the column name.",
            )
        col = data[y].to_numpy(dtype=float)
        keep = np.flatnonzero(~np.isnan(col))
        if keep.size == 0:
            raise MethodIncompatibility(
                f"arima: column {y!r} has no non-missing values.",
                recovery_hint="Check the column.",
            )
        rows = slice(keep[0], keep[-1] + 1)
        if exog is not None:
            names = [exog] if isinstance(exog, str) else list(exog)
            absent = [c for c in names if c not in data.columns]
            if absent:
                raise MethodIncompatibility(
                    f"arima: exog columns {absent} are not in data.",
                    recovery_hint="Check the column names.",
                )
            exog_names = tuple(str(c) for c in names)
            exog = data[names].to_numpy(dtype=float)[rows]
        index = data.index[rows]
        y = col[rows]
    elif isinstance(y, str):
        raise MethodIncompatibility(
            f"arima: y={y!r} names a column; pass data= as well.",
            recovery_hint=f"sp.arima({y!r}, data=df) or sp.arima(df[{y!r}]).",
        )
    elif isinstance(y, pd.Series):
        index = y.index
    if exog is not None and not exog_names:
        if isinstance(exog, pd.DataFrame):
            exog_names = tuple(str(c) for c in exog.columns)
            exog = exog.to_numpy(dtype=float)
        elif isinstance(exog, pd.Series):
            exog_names = (str(exog.name) if exog.name is not None else "x1",)
            exog = exog.to_numpy(dtype=float).reshape(-1, 1)
        else:
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            exog_names = tuple(f"x{i + 1}" for i in range(exog.shape[1]))
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    from ._forecast_common import boxcox_forward, boxcox_inverse, resolve_boxcox

    _lam = resolve_boxcox(
        y,
        boxcox,
        int(period or (seasonal_order[3] if seasonal_order else 1)),
        fn="arima",
    )
    if _lam is not None:
        y = boxcox_forward(y, _lam)
    if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1:
        index = None
    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] != n:
            raise MethodIncompatibility(
                f"arima: exog has {exog.shape[0]} rows, the series {n}.",
                recovery_hint="Align the regressors with the series.",
            )
        if not np.isfinite(exog).all():
            raise MethodIncompatibility(
                "arima: exog has missing values.",
                recovery_hint=(
                    "Drop the rows where a regressor is missing (lagged "
                    "regressors lose their first rows)."
                ),
            )
    method_norm = method.lower().replace("-", "_")
    if method_norm in {"kalman", "exact", "mle"}:
        method_norm = "statespace"
    elif method_norm == "css_ml":
        method_norm = "innovations_mle"
    elif method_norm in {"css", "conditional", "conditional_mle"}:
        raise MethodIncompatibility(
            f"arima: method={method!r} asks for conditional sum of squares, "
            "which is not implemented; every method here is exact maximum "
            "likelihood.",
            recovery_hint="Use method='statespace' or 'innovations_mle'.",
            diagnostics={"method": method},
        )
    if method_norm not in {"statespace", "innovations_mle"}:
        raise ValueError("method must be 'statespace', 'css_ml', or 'innovations_mle'")

    if trend is not None and trend not in ("c", "n"):
        raise MethodIncompatibility(
            f"arima: trend must be None, 'c' or 'n', got {trend!r}.",
            recovery_hint="'c' estimates a constant (a drift once differenced).",
        )
    if period is not None:
        if int(period) != period or int(period) < 1:
            raise MethodIncompatibility(
                f"arima: period must be a positive integer, got {period!r}.",
                recovery_hint="4 for quarterly data, 12 for monthly.",
            )
        period = int(period)
        if seasonal_order is not None and int(seasonal_order[3]) != period:
            raise MethodIncompatibility(
                f"arima: period={period} contradicts seasonal_order="
                f"{tuple(seasonal_order)}.",
                recovery_hint="Give one of the two.",
            )
        if not auto and seasonal_order is None and period > 1:
            raise MethodIncompatibility(
                "arima: period= selects seasonal orders and needs auto=True.",
                recovery_hint=(
                    "Pass auto=True, or seasonal_order=(P, D, Q, period) for a "
                    "fixed model."
                ),
            )

    def _trend_for(
        order_: Tuple[int, int, int],
        seas_: Optional[Tuple[int, int, int, int]],
        constant: Optional[bool] = None,
    ) -> Tuple[str, Optional[str]]:
        """statsmodels' trend code and the name of the parameter it adds."""
        n_diff = int(order_[1]) + (int(seas_[1]) if seas_ else 0)
        if constant is None:
            want = trend if trend is not None else ("c" if n_diff == 0 else "n")
        else:
            want = "c" if constant else "n"
        if want == "n":
            return "n", None
        if n_diff == 0:
            return "c", "const"
        if n_diff == 1:
            # a constant in the differenced series is a linear trend in the
            # level, which is how statsmodels parameterises it
            return "t", "drift"
        raise MethodIncompatibility(
            "arima: a constant with d + D > 1 is a polynomial trend in the "
            "level; it is not estimated.",
            recovery_hint="Use trend='n', or difference the series once less.",
        )

    class _Fit:
        """One fitted model: the estimates and what is derived from them."""

        def __init__(
            self,
            res: Any,
            llf: float,
            k: int,
            n_eff: int,
            det: Optional[str],
            converged: bool,
            bse: Optional[np.ndarray] = None,
            probe: Any = None,
        ) -> None:
            self.res = res  # statsmodels results in levels (forecasts)
            self.bse = bse  # standard errors, when not those of ``res``
            # object carrying ``arroots`` / ``maroots`` of the estimates
            self.probe = probe if probe is not None else res
            self.llf = float(llf)
            self.k = int(k)  # coefficients and the innovation variance
            self.n_eff = int(n_eff)
            self.det = det
            self.converged = converged

        @property
        def aic(self) -> float:
            return -2.0 * self.llf + 2.0 * self.k

        @property
        def bic(self) -> float:
            return -2.0 * self.llf + self.k * float(np.log(self.n_eff))

        @property
        def aicc(self) -> float:
            return self.aic + 2.0 * self.k * (self.k + 1) / max(
                self.n_eff - self.k - 1, 1
            )

    def _fit(
        order_: Tuple[int, int, int],
        seas_: Optional[Tuple[int, int, int, int]],
        constant: Optional[bool] = None,
        effort: int = 2,
        nested: bool = True,
    ) -> "_Fit":
        """``effort`` 2: every optimiser path (a fit the user asked for, and
        the leading candidates of a search); 1: the simplex check from the
        quasi-Newton answer; 0: that check only when the quasi-Newton
        answer looks suspect (not converged, or a root near the unit
        circle), which is what the candidates of a search get."""
        sm_trend, det = _trend_for(order_, seas_, constant)
        seas_sm = tuple(seas_) if seas_ else (0, 0, 0, 0)
        n_diff_obs = int(order_[1]) + int(seas_sm[1]) * int(seas_sm[3])
        n_eff = n - n_diff_obs
        if method_norm != "statespace":

            def _build_innov(series: np.ndarray) -> Any:
                model = SMARIMA(
                    series,
                    order=order_,
                    seasonal_order=seas_sm,
                    exog=exog,
                    trend=sm_trend,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                _exact_diffuse(model)
                return model

            # The likelihood does not depend on the unit of measurement,
            # but the search does: on a series in the thousands it stops
            # short of the maximum in the third digit of the coefficients,
            # and a mean far from zero is found slowly. A badly scaled
            # series is therefore searched on a standardised copy; the
            # estimates are mapped back and the reported fit is evaluated
            # on the series as given.
            x_ = y
            for _ in range(int(order_[1])):
                x_ = np.diff(x_)
            for _ in range(int(seas_sm[1])):
                x_ = x_[int(seas_sm[3]) :] - x_[: -int(seas_sm[3])]
            sd_i = float(np.std(x_)) if len(x_) > 1 else 1.0
            if not np.isfinite(sd_i) or sd_i <= 0:
                sd_i = 1.0
            scale_i = 1.0 if 0.1 <= sd_i <= 10.0 else sd_i
            mean_i = float(np.mean(y))
            loc_i = mean_i if sm_trend == "c" and abs(mean_i) > 10.0 * sd_i else 0.0
            with _warnings.catch_warnings():
                # the burn is deliberate, see _exact_diffuse
                _warnings.filterwarnings(
                    "ignore", message="Care should be used when applying"
                )
                if scale_i == 1.0 and loc_i == 0.0:
                    r0 = _build_innov(y).fit(method="innovations_mle")
                else:
                    work = _build_innov((y - loc_i) / scale_i)
                    par_i = np.array(
                        work.fit(method="innovations_mle").params, dtype=float
                    )
                    # constant or drift first, then the regressors
                    k_reg_i = int(work.k_exog) + int(getattr(work, "k_trend", 0))
                    par_i[:k_reg_i] *= scale_i
                    if sm_trend == "c":
                        par_i[0] += loc_i
                    par_i[-1] *= scale_i * scale_i  # innovation variance
                    r0 = _build_innov(y).smooth(par_i, cov_type="opg")
            return _Fit(r0, r0.llf, len(np.asarray(r0.params)), n_eff, None, True)
        # The deterministic term is a regressor of the level equation: a
        # column of ones (the mean), or the time index, whose difference is
        # a constant (the drift).
        cols: list[Any] = []
        if det == "const":
            cols.append(np.ones(n))
        elif det == "drift":
            cols.append(np.arange(1, n + 1, dtype=float))
        if exog is not None:
            cols.append(exog)
        X = np.column_stack(cols) if cols else None
        # Estimation is on the differenced series: its exact Gaussian
        # likelihood, with the ARMA part started from its stationary
        # distribution, is the likelihood R's arima and Stata's arima
        # maximise for a differenced model. (Carrying the differences as
        # states with an approximately diffuse start gives nearly the same
        # estimates at several times the cost.)
        # The likelihood does not depend on the unit of measurement, but
        # the search does: on a series in the thousands the regression
        # coefficients (mean, drift, regressors) are found slowly and the
        # search can stop short. A badly scaled series is searched on a
        # rescaled copy and the estimates are mapped back.
        w_ = y
        for _ in range(int(order_[1])):
            w_ = np.diff(w_)
        for _ in range(int(seas_sm[1])):
            w_ = w_[int(seas_sm[3]) :] - w_[: -int(seas_sm[3])]
        sd_ = float(np.std(w_)) if len(w_) > 1 else 1.0
        if not np.isfinite(sd_) or sd_ <= 0:
            sd_ = 1.0
        unit = 1.0 if 0.1 <= sd_ <= 10.0 else sd_
        est = SARIMAX(
            y / unit,
            exog=X,
            order=order_,
            seasonal_order=seas_sm,
            simple_differencing=True,
            concentrate_scale=True,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )

        # The likelihood of the differenced series is evaluated by the
        # innovations algorithm (``_arma_core``): the same number the
        # Kalman filter of ``est`` returns, about fifteen times faster.
        # ``est`` supplies the starting values and the map between the
        # unconstrained search space and stationary, invertible
        # coefficients.
        from types import SimpleNamespace

        from scipy import optimize as _opt

        from . import _arma_core as _ac

        w_d = np.ascontiguousarray(np.asarray(est.endog)[:, 0], dtype=float)
        X_d = None if X is None else np.asarray(est.exog, dtype=float)
        k_x = 0 if X_d is None else X_d.shape[1]
        orders = (
            int(order_[0]),
            int(order_[2]),
            int(seas_sm[0]),
            int(seas_sm[2]),
            int(seas_sm[3]),
        )
        n_w = w_d.shape[0]

        def evaluate(par: np.ndarray) -> Tuple[float, float]:
            u = w_d if X_d is None else np.ascontiguousarray(w_d - X_d @ par[:k_x])
            phi, th = _ac.expand(np.ascontiguousarray(par[k_x:]), *orders)
            try:
                ll, sig = _ac.arma_loglike(u, phi, th)
            except np.linalg.LinAlgError:
                return float("-inf"), float("nan")
            return float(ll), float(sig)

        def negative(u: np.ndarray) -> float:
            try:
                par = np.asarray(est.transform_params(u), dtype=float)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                return 1.0e10
            if not np.isfinite(par).all():
                return 1.0e10
            ll = evaluate(par)[0]
            return -ll / n_w if np.isfinite(ll) else 1.0e10

        def summarise(par: np.ndarray, converged_: bool) -> Any:
            ll, sig = evaluate(par)
            phi, th = _ac.expand(np.ascontiguousarray(par[k_x:]), *orders)
            ar_r = np.roots(np.r_[-phi[::-1], 1.0]) if phi.size else np.empty(0)
            ma_r = np.roots(np.r_[th[::-1], 1.0]) if th.size else np.empty(0)
            return SimpleNamespace(
                params=par,
                names=[nm for nm in est.param_names if nm != "sigma2"],
                llf=ll,
                scale=sig,
                arroots=ar_r,
                maroots=ma_r,
                mle_retvals={"converged": bool(converged_)},
            )

        def run(opt: str, maxiter: int, start: Any = None) -> Any:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                start_c = est.start_params if start is None else start
                u0 = np.asarray(
                    est.untransform_params(np.asarray(start_c, dtype=float)),
                    dtype=float,
                )
                if not np.isfinite(u0).all():
                    raise FloatingPointError("starting values on the boundary")
                if opt == "lbfgs":
                    out = _opt.minimize(
                        negative,
                        u0,
                        method="L-BFGS-B",
                        options={
                            "maxiter": maxiter,
                            "maxcor": 12,
                            "gtol": 1e-8,
                            "ftol": 1e2 * np.finfo(float).eps,
                            "eps": 1e-8,
                        },
                    )
                else:
                    out = _opt.minimize(
                        negative,
                        u0,
                        method="Nelder-Mead",
                        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4},
                    )
                par = np.asarray(est.transform_params(out.x), dtype=float)
            fit_ = summarise(par, bool(out.success))
            if not np.isfinite(fit_.llf):
                raise FloatingPointError("the likelihood is not finite at the optimum")
            return fit_

        n_par = int(order_[0]) + int(order_[2]) + int(seas_sm[0]) + int(seas_sm[2])
        if est.k_params == 0:
            # a random walk without drift: nothing to optimise
            best = summarise(np.empty(0), True)
        else:
            try:
                best = run("lbfgs", 500 if effort >= 2 else 200)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                from ..exceptions import NumericalInstability

                raise NumericalInstability(
                    f"arima: the likelihood of ARIMA{tuple(order_)}"
                    + (f"x{tuple(seas_sm)}" if seas_ else "")
                    + f" could not be maximised ({exc}).",
                    recovery_hint=(
                        "Check the series for constant stretches or extreme "
                        "values, or fit a lower order."
                    ),
                ) from exc
            # a second start, from the conditional-sum-of-squares estimates
            css = _css_start(w_d, X_d, *orders)
            if css is not None and css.shape[0] == est.k_params:
                try:
                    alt = run("lbfgs", 500 if effort >= 2 else 200, css)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    alt = None
                if alt is not None and np.isfinite(alt.llf):
                    if alt.llf > best.llf + 1e-7:
                        best = alt
            # a third start for a final fit: every ARMA coefficient at zero
            # (where R's arima(method="ML") begins). Over-parameterised
            # mixed models have maxima the other two starts do not reach.
            if effort >= 2 and n_par > 1:
                flat = np.array(est.start_params, dtype=float)
                arma = [
                    i
                    for i, nm in enumerate(est.param_names)
                    if nm.startswith(("ar.", "ma."))
                ]
                if arma and np.any(flat[arma] != 0.0):
                    flat[arma] = 0.0
                    try:
                        alt = run("lbfgs", 500, flat)
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                        alt = None
                    if alt is not None and np.isfinite(alt.llf):
                        if alt.llf > best.llf + 1e-7:
                            best = alt
            # Further starts for a final fit: the estimates of the models
            # with one AR or one MA term fewer, the dropped coefficient at
            # zero. A model then never fits worse than the ones it nests,
            # which the other starts do not guarantee (a moving-average
            # root on the unit circle traps them).
            p_, q_ = int(order_[0]), int(order_[2])
            if effort >= 2 and nested and p_ + q_ >= 2:
                from ..exceptions import NumericalInstability as _Unstable

                names = list(est.param_names)
                for child in ((p_ - 1, order_[1], q_), (p_, order_[1], q_ - 1)):
                    if min(child[0], child[2]) < 0:
                        continue
                    try:
                        sub = _fit(child, seas_, constant, effort=1, nested=False)
                        inner = sub.probe
                        known = dict(zip(inner.names, inner.params))
                        known["sigma2"] = float(inner.scale)
                        guess = np.array([known.get(nm, 0.0) for nm in names], float)
                        alt = run("lbfgs", 500, guess)
                    except (
                        AttributeError,
                        ValueError,
                        FloatingPointError,
                        np.linalg.LinAlgError,
                        _Unstable,
                    ):
                        # a starting point that could not be built or used;
                        # the other starts stand
                        continue
                    if np.isfinite(alt.llf) and alt.llf > best.llf + 1e-7:
                        best = alt
        # The quasi-Newton search alone stops at an inferior local maximum
        # on a sizeable share of mixed and seasonal models (near-cancelling
        # roots, a moving-average root on the unit circle). A simplex
        # search from its answer, and for a final fit a second one from the
        # default starting values, each followed by a quasi-Newton polish,
        # recover the maximum that R's and Stata's optimisers report.
        k_all = len(np.asarray(best.params))
        thorough = effort >= 2
        check = n_par > 0
        if check and effort == 0:
            flagged = (getattr(best, "mle_retvals", None) or {}).get(
                "converged"
            ) is False
            check = flagged or _near_unit_root(best, 1.05)
        if check:
            plans = [(best.params, (200 if thorough else 60) * k_all)]
            if thorough:
                plans.append((None, 200 * k_all))
            for st, budget in plans:
                try:
                    cand = run("nm", budget, st)
                    if cand.llf > best.llf + 1e-7 or st is None:
                        polished = run("lbfgs", 200, cand.params)
                        cand = polished if polished.llf >= cand.llf else cand
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    # an optimiser path failing is not fatal: the
                    # quasi-Newton answer stands
                    continue
                if np.isfinite(cand.llf) and cand.llf > best.llf + 1e-7:
                    best = cand
        converged = (getattr(best, "mle_retvals", None) or {}).get(
            "converged"
        ) is not False
        if effort < 2:
            # a candidate of a search: its criterion and roots are enough
            llf_c = float(best.llf) - n_eff * float(np.log(unit))
            return _Fit(None, llf_c, k_all + 1, n_eff, det, converged, None, best)
        # the same parameters in the level model, for fitted values,
        # standard errors and forecasts
        full = SARIMAX(
            y,
            exog=X,
            order=order_,
            seasonal_order=seas_sm,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        _exact_diffuse(full)
        coefs = np.asarray(best.params, dtype=float).copy()
        k_reg = 0 if X is None else X.shape[1]
        theta_s = np.append(coefs, float(best.scale))
        # back to the units of y: mean, drift and regressors scale with the
        # unit, the innovation variance with its square
        to_y = np.ones(theta_s.shape[0])
        to_y[:k_reg] = unit
        to_y[-1] = unit * unit
        theta = theta_s * to_y
        llf = float(best.llf) - n_eff * float(np.log(unit))
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            if unit == 1.0:
                res_full = full.smooth(theta, cov_type="opg")
                bse = None
            else:
                # the outer-product covariance is computed by numerical
                # differentiation, which is only reliable on the rescaled
                # series; its standard errors are mapped back
                full_s = SARIMAX(
                    y / unit,
                    exog=X,
                    order=order_,
                    seasonal_order=seas_sm,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                _exact_diffuse(full_s)
                bse = (
                    np.asarray(full_s.smooth(theta_s, cov_type="opg").bse, dtype=float)
                    * to_y
                )
                res_full = full.smooth(theta, cov_type="none")
        return _Fit(res_full, llf, len(theta), n_eff, det, converged, bse, best)

    constant: Optional[bool] = None
    candidates: Optional[pd.DataFrame] = None
    final_res: Any = None
    if auto:
        from ._auto_arima import search as _search

        # Differencing is settled first, by tests on the (regression
        # residuals of the) series: likelihoods of differently differenced
        # series describe different data and cannot be ranked by AICc.
        base = y
        if exog is not None:
            X1 = np.column_stack([np.ones(n), exog])
            base = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
        search_seasonal = seasonal_order is None and period is not None and period > 1
        if search_seasonal:
            assert period is not None
            s_len = period
            D = _nsdiffs(base, s_len, 0.64, int(max_D)) if n >= 2 * s_len + 1 else 0
            if D:
                base = base[s_len:] - base[:-s_len]
        elif seasonal_order is not None:
            s_len = int(seasonal_order[3])
            D = int(seasonal_order[1])
            for _ in range(D):
                base = base[s_len:] - base[:-s_len]
        else:
            s_len, D = 0, 0
        d = _kpss_ndiffs(base, max_d)
        allow_const = trend is None and (d + D) <= 1
        fixed_const: Optional[bool] = None if trend is None else trend == "c"
        if fixed_const and d + D > 1:
            _trend_for((0, d, 0), (0, D, 0, s_len) if s_len else None, True)
        n_failed = 0
        search_effort = 0
        fixed_seasonal: Optional[Tuple[int, int, int, int]] = None
        if seasonal_order:
            so_ = [int(v) for v in seasonal_order]
            fixed_seasonal = (so_[0], so_[1], so_[2], so_[3])

        def _score(p: int, q: int, P: int, Q: int, c: bool) -> float:
            nonlocal n_failed
            seas_c: Optional[Tuple[int, int, int, int]]
            if search_seasonal:
                seas_c = (P, D, Q, s_len)
            else:
                seas_c = fixed_seasonal
            want = c if fixed_const is None else fixed_const
            try:
                res_c = _fit((p, d, q), seas_c, want, effort=search_effort)
            except Exception:
                n_failed += 1
                return float(np.inf)
            if _near_unit_root(res_c.probe):
                return float(np.inf)
            val = res_c.aicc
            return val if np.isfinite(val) else float(np.inf)

        best, tried = _search(
            _score,
            seasonal=search_seasonal,
            allow_constant=allow_const,
            max_p=int(max_p),
            max_q=int(max_q),
            max_P=int(max_P),
            max_Q=int(max_Q),
            stepwise=bool(stepwise),
        )
        if best is None:
            raise MethodIncompatibility(
                "arima(auto=True): no candidate order could be fitted.",
                recovery_hint="Check the series for constants or missing values.",
                diagnostics={"n_failed": n_failed},
            )
        # The search fits each candidate once; the leaders are refitted
        # thoroughly and ranked again before one is returned.
        leaders = [k for k, v in tried if np.isfinite(v)][:3]
        best_val = np.inf
        for key in leaders:
            kp, kq, kP, kQ, kc = key
            seas_k: Optional[Tuple[int, int, int, int]]
            if search_seasonal:
                seas_k = (kP, D, kQ, s_len)
            else:
                seas_k = fixed_seasonal
            want_k = kc if fixed_const is None else fixed_const
            try:
                res_k = _fit((kp, d, kq), seas_k, want_k)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
            if _near_unit_root(res_k.probe):
                continue
            val_k = res_k.aicc
            if val_k < best_val:
                best, best_val, final_res = key, val_k, res_k
        p_, q_, P_, Q_, c_ = best
        order = (p_, d, q_)
        if search_seasonal:
            seasonal_order = (P_, D, Q_, s_len)
        constant = c_ if fixed_const is None else fixed_const
        rows_c = []
        for (pp, qq, PP, QQ, cc), val in tried:
            lab = f"ARIMA({pp},{d},{qq})"
            if search_seasonal:
                lab += f"({PP},{D},{QQ})[{s_len}]"
            elif seasonal_order:
                so = seasonal_order
                lab += f"({so[0]},{so[1]},{so[2]})[{so[3]}]"
            if cc:
                lab += " with drift" if d + D == 1 else " with mean"
            rows_c.append({"model": lab, "aicc": val})
        candidates = pd.DataFrame(rows_c)

    seas_final = tuple(seasonal_order) if seasonal_order else None
    if auto and final_res is not None:
        fit_ = final_res
    else:
        fit_ = _fit(order, seas_final, constant)  # type: ignore[arg-type]
    res = fit_.res
    if not fit_.converged:
        from ..exceptions import ConvergenceWarning

        _warnings.warn(
            f"arima: the optimiser did not report convergence for ARIMA{tuple(order)}"
            + (f"x{seas_final}" if seas_final else "")
            + "; the estimates may not be the maximum likelihood ones. A "
            "simpler model, or fewer differences, usually fits cleanly.",
            ConvergenceWarning,
            stacklevel=2,
        )

    _param_index = list(res.param_names) if hasattr(res, "param_names") else None
    _, _trend_name = _trend_for(order, seas_final, constant)  # type: ignore[arg-type]
    if _param_index is not None and method_norm == "statespace":
        # the regressors lead the parameter vector as x1, x2, ...: the
        # deterministic term first, then the user's regressors
        lead = ([_trend_name] if _trend_name is not None else []) + list(exog_names)
        if len(lead) == 1 and _param_index and not _param_index[0].startswith("x"):
            _param_index[0] = lead[0]
        else:
            for i, nm in enumerate(lead):
                _param_index[i] = nm
    elif _param_index is not None:
        if _trend_name is not None:
            # statsmodels puts the deterministic term first ("const" / "x1")
            _param_index[0] = _trend_name
        offset = 1 if _trend_name is not None else 0
        for i, nm in enumerate(exog_names):
            if offset + i < len(_param_index):
                _param_index[offset + i] = nm
    _params = pd.Series(np.asarray(res.params, dtype=float), index=_param_index)
    # statsmodels computes the asymptotic SEs (sqrt of the diagonal of the
    # covariance of the MLE) but we never surfaced them before; expose them.
    _bse = fit_.bse if fit_.bse is not None else getattr(res, "bse", None)
    if _bse is not None:
        _se = pd.Series(np.asarray(_bse, dtype=float), index=_param_index)
    else:  # pragma: no cover - defensive; SARIMAX always populates bse
        _se = pd.Series(np.full(len(_params), np.nan), index=_param_index)

    _result = ARIMAResult(
        order=(int(order[0]), int(order[1]), int(order[2])),
        seasonal_order=(
            (
                int(seas_final[0]),
                int(seas_final[1]),
                int(seas_final[2]),
                int(seas_final[3]),
            )
            if seas_final
            else None
        ),
        params=_params,
        se=_se,
        aic=float(fit_.aic),
        bic=float(fit_.bic),
        aicc=float(fit_.aicc),
        log_likelihood=float(fit_.llf),
        residuals=np.asarray(res.resid),
        fitted_values=(
            np.asarray(res.fittedvalues)
            if _lam is None
            else boxcox_inverse(np.asarray(res.fittedvalues, dtype=float), _lam)
        ),
        n=n,
        _model=res,
        exog_names=exog_names,
        candidates=candidates,
        _index=index,
        _det=fit_.det,
        boxcox=_lam,
        biasadj=bool(biasadj),
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.timeseries.arima",
            params={
                "order": list(order),
                "seasonal_order": (list(seasonal_order) if seasonal_order else None),
                "auto": auto,
                "max_p": max_p,
                "max_q": max_q,
                "max_d": max_d,
                "method": method,
                "trend": trend,
                "period": period,
                "stepwise": stepwise,
            },
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
