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

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility


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

    def forecast(self, horizon: int = 10, alpha: float = 0.05) -> pd.DataFrame:
        fc = self._model.get_forecast(steps=horizon)
        pred = np.asarray(fc.predicted_mean).ravel()
        ci = fc.conf_int(alpha=alpha)
        ci = np.asarray(ci)
        return pd.DataFrame(
            {
                "forecast": pred,
                "lower": ci[:, 0] if ci.ndim == 2 else ci,
                "upper": ci[:, 1] if ci.ndim == 2 else ci,
            }
        )

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
        fc = self.forecast(horizon, alpha)
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
            + (f" x {self.seasonal_order}" if self.seasonal_order else ""),
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


#: 5% critical value of the KPSS level-stationarity statistic.
_KPSS_CRIT_5 = 0.463


def _kpss_stat(x: np.ndarray) -> float:
    """KPSS level-stationarity statistic, Bartlett kernel with
    ``trunc(3 * sqrt(n) / 13)`` lags (``nan`` when it is not defined)."""
    n = len(x)
    e = x - x.mean()
    partial = np.cumsum(e)
    lags = int(3 * np.sqrt(n) / 13)
    lrv = float(e @ e) / n
    for h in range(1, lags + 1):
        lrv += 2.0 * (1.0 - h / (lags + 1.0)) * float(e[h:] @ e[:-h]) / n
    if lrv <= 0:
        return float("nan")
    return float(partial @ partial) / (n * n * lrv)


def _kpss_ndiffs(y: np.ndarray, max_d: int) -> int:
    """Number of differences after which KPSS no longer rejects level
    stationarity at 5%, capped at ``max_d``.

    The lag truncation is the choice of R's
    ``forecast::ndiffs(test="kpss")``.
    """
    x = np.asarray(y, dtype=float)
    d = 0
    while d < max_d and len(x) >= 8 and np.ptp(x) > 0:
        stat = _kpss_stat(x)
        if not stat > _KPSS_CRIT_5:
            break
        x = np.diff(x)
        d += 1
    return d


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
    data: Optional[pd.DataFrame] = None,
) -> ARIMAResult:
    """Fit ARIMA(p,d,q) or SARIMAX.

    Parameters
    ----------
    y : array-like, pd.Series or str
        The series; a column name when ``data`` is given.
    order : (p, d, q)
    seasonal_order : (P, D, Q, s), optional
    exog : array-like or list of str, optional
        Exogenous regressors (ARIMAX); column names when ``data`` is given.
    data : pandas.DataFrame, optional
        Frame holding ``y`` (and ``exog``), in time order. Rows before the
        first and after the last non-missing ``y`` are dropped, so a
        differenced column can be passed as it is.
    auto : bool, default False
        If True, select the order and ignore ``order``. ``d`` is chosen
        first, as the number of differences after which a KPSS test no
        longer rejects level stationarity at 5% (at most ``max_d``); then
        ``(p, q)`` minimise AICc over ``0..max_p`` by ``0..max_q``,
        ``(0, d, 0)`` included. Unless ``trend`` is given the constant is
        part of that search when ``d + D <= 1``: a mean without
        differencing, a drift with one difference. A candidate with an AR
        or MA root within 1% of the unit circle is dropped: it is a model
        differenced once too often or too seldom, fitted on the boundary.
        This is the exhaustive search of R's ``forecast::auto.arima(
        stepwise=FALSE, approximation=FALSE)``, without its cap on
        ``p + q``.
    max_p, max_q, max_d : int
        Bounds for the auto search.
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

        With differencing, the integration states start from an exact
        diffuse prior and the ``d + s D`` observations that pin them down
        are left out of the likelihood, which is then the likelihood of the
        differenced series. Releases through 1.38.0 used statsmodels'
        default, a normal prior of variance 1e6. On a series whose
        innovation variance is not far below 1e6 that prior is informative:
        the estimates of a differenced model with AR or MA terms changed
        with the unit of measurement, and the reported log-likelihood was
        too low. ``AIC``, ``BIC`` and ``AICc`` count the estimated
        parameters only and use the number of observations after
        differencing, as R and Stata do.

    trend : {None, 'c', 'n'}, optional
        Deterministic term. ``'c'`` estimates a constant: the mean of the
        series when it is not differenced (``const``), the mean of the
        differenced series -- a drift in the level -- when ``d + D = 1``
        (``drift``). ``'n'`` estimates neither. The default follows R's
        ``stats::arima`` and statsmodels' ``ARIMA``: a constant without
        differencing, none with it. Stata's ``arima`` always includes the
        constant; pass ``trend='c'`` to reproduce it on a differenced
        series. A constant with ``d + D > 1`` is a polynomial trend and is
        refused.

    Returns
    -------
    ARIMAResult
        Exposes ``params`` and the matching standard errors ``se`` (alias
        ``std_errors``), plus ``tvalues``, ``pvalues``, and
        ``conf_int(alpha)`` for inference, alongside ``aic`` / ``bic`` /
        ``aicc`` / ``log_likelihood`` and ``forecast`` / ``plot``.

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
    >>> res.conf_int().shape  # 95% CIs, one row per param
    (4, 2)
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA as SMARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as e:
        raise ImportError(
            "statsmodels is required for arima(). "
            "Install with `pip install statsmodels`."
        ) from e

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
            exog = data[names].to_numpy(dtype=float)[rows]
        y = col[rows]
    elif isinstance(y, str):
        raise MethodIncompatibility(
            f"arima: y={y!r} names a column; pass data= as well.",
            recovery_hint=f"sp.arima({y!r}, data=df) or sp.arima(df[{y!r}]).",
        )
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
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
    seasonal_d = int(seasonal_order[1]) if seasonal_order else 0

    # ``auto=True`` settles the deterministic term as part of the search and
    # records its choice here; an explicit ``trend=`` is never overridden.
    chosen_trend: Optional[str] = trend

    def _trend_for(order_: Tuple[int, int, int]) -> Tuple[str, Optional[str]]:
        """statsmodels' trend code and the name of the parameter it adds."""
        n_diff = int(order_[1]) + seasonal_d
        want = (
            chosen_trend if chosen_trend is not None else ("c" if n_diff == 0 else "n")
        )
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

    def _exact_diffuse(model: Any) -> int:
        """Start the integration states from an exact diffuse prior.

        statsmodels starts them from a normal prior of variance 1e6, the
        "approximate diffuse" initialisation. That prior is diffuse only
        against an innovation variance far below 1e6: on a series measured
        in the thousands it is informative, the likelihood depends on the
        unit of measurement, and so do the estimates. The exact
        initialisation of Durbin and Koopman has no such constant. The
        ``d + s D`` observations that identify the integration states are
        left out of the likelihood, which is then the likelihood of the
        differenced series (R ``stats::arima``, Stata ``arima``).
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

    seasonal = tuple(seasonal_order) if seasonal_order else (0, 0, 0, 0)

    def _build(series: np.ndarray, order_: Tuple[int, int, int], sm_trend: str) -> Any:
        # SARIMAX for the model without a deterministic term, ARIMA (which
        # folds the constant into the regression) otherwise.
        if method_norm == "statespace" and sm_trend == "n":
            model = SARIMAX(
                series,
                order=order_,
                seasonal_order=seasonal,
                exog=exog,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
        else:
            model = SMARIMA(
                series,
                order=order_,
                seasonal_order=seasonal,
                exog=exog,
                trend=sm_trend,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
        _exact_diffuse(model)
        return model

    def _fit(
        order_: Tuple[int, int, int],
        maxiter: Optional[int] = None,
    ) -> Any:
        sm_trend, _ = _trend_for(order_)

        def _run(model: Any) -> Any:
            with warnings.catch_warnings():
                # the burn is deliberate, see _exact_diffuse
                warnings.filterwarnings(
                    "ignore", message="Care should be used when applying"
                )
                if method_norm != "statespace":
                    return model.fit(method="innovations_mle")
                opts: dict[str, Any] = {"disp": False}
                if maxiter is not None:
                    opts["maxiter"] = maxiter
                if isinstance(model, SMARIMA):
                    return model.fit(method_kwargs=opts)
                return model.fit(**opts)

        # The likelihood does not depend on the unit of measurement, but the
        # quasi-Newton search does: on a series in the thousands it stops
        # short of the maximum in the third digit of the coefficients, and a
        # mean far from zero is found slowly. A badly scaled series is
        # therefore searched on a standardised copy; the estimates are mapped
        # back and the reported fit is evaluated on the series as given. A
        # series already of order one is fitted as it is.
        x = y
        for _ in range(int(order_[1])):
            x = np.diff(x)
        for _ in range(int(seasonal[1])):
            x = x[int(seasonal[3]) :] - x[: -int(seasonal[3])]
        sd = float(np.std(x)) if len(x) > 1 else 1.0
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        scale = 1.0 if 0.1 <= sd <= 10.0 else sd
        mean = float(np.mean(y))
        loc = mean if sm_trend == "c" and abs(mean) > 10.0 * sd else 0.0
        if scale == 1.0 and loc == 0.0:
            return _run(_build(y, order_, sm_trend))

        work = _build((y - loc) / scale, order_, sm_trend)
        params = np.array(_run(work).params, dtype=float)
        # constant or drift first, then the regressors
        k_reg = int(work.k_exog) + int(getattr(work, "k_trend", 0))
        params[:k_reg] *= scale
        if sm_trend == "c":
            params[0] += loc
        params[-1] *= scale * scale  # innovation variance
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Care should be used when applying"
            )
            return _build(y, order_, sm_trend).smooth(params, cov_type="opg")

    def _ic(res: Any, order_: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """AIC, BIC and AICc from the log-likelihood and the parameter count.

        statsmodels adds the number of diffuse observations to the parameter
        count; R and Stata do not, and neither does this.
        """
        k = len(np.asarray(res.params))
        n_used = n - int(res.model._k_states_diff)
        aic_ = -2.0 * float(res.llf) + 2.0 * k
        bic_ = -2.0 * float(res.llf) + k * float(np.log(max(n_used, 1)))
        aicc_ = aic_ + 2.0 * k * (k + 1) / max(n_used - k - 1, 1)
        return aic_, bic_, aicc_

    if auto:
        # The order of differencing is settled first, by KPSS tests on the
        # successively differenced series (the rule of R's
        # forecast::auto.arima). Likelihoods of differently differenced
        # series describe different data and cannot be ranked by AICc.
        d = _kpss_ndiffs(y, max_d)
        # The constant is part of the search unless the caller fixed it: a
        # mean when the series is not differenced, a drift when it is
        # differenced once (forecast::auto.arima's allowmean / allowdrift).
        # Without the drift candidate a trending series can only be fitted
        # by differencing it again or by an AR root at one.
        if trend is not None:
            trend_candidates = [trend]
        elif d + seasonal_d <= 1:
            trend_candidates = ["c", "n"]
        else:
            trend_candidates = ["n"]
        best_aicc = np.inf
        best_order = (0, d, 0)
        best_trend = trend_candidates[0]
        n_failed = 0
        n_boundary = 0
        for cand in trend_candidates:
            chosen_trend = cand
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    # (0, d, 0) is a candidate: white noise, or a random walk.
                    try:
                        res = _fit((p, d, q), maxiter=50)
                    except Exception:
                        n_failed += 1
                        continue
                    # A root of the AR or MA polynomial within 1% of the
                    # unit circle marks a model that is differenced once
                    # too often or not often enough; its likelihood is
                    # maximised on the boundary and its AICc is not
                    # comparable. forecast::auto.arima drops such fits.
                    roots = np.concatenate(
                        [np.atleast_1d(res.arroots), np.atleast_1d(res.maroots)]
                    )
                    if roots.size and float(np.min(np.abs(roots))) < 1.01:
                        n_boundary += 1
                        continue
                    aicc = _ic(res, (p, d, q))[2]
                    if aicc < best_aicc:
                        best_aicc = aicc
                        best_order = (p, d, q)
                        best_trend = cand
        chosen_trend = best_trend
        if not np.isfinite(best_aicc):
            raise MethodIncompatibility(
                "arima(auto=True): no candidate order could be fitted.",
                recovery_hint="Check the series for constants or missing values.",
                diagnostics={"n_failed": n_failed, "n_boundary": n_boundary},
            )
        order = best_order

    res = _fit(order)

    aic, bic, aicc = _ic(res, order)

    _param_index = list(res.param_names) if hasattr(res, "param_names") else None
    _, _trend_name = _trend_for(order)
    if _param_index is not None and _trend_name is not None:
        # statsmodels puts the deterministic term first ("const" / "x1")
        _param_index[0] = _trend_name
    _params = pd.Series(res.params, index=_param_index)
    # statsmodels computes the asymptotic SEs (sqrt of the diagonal of the
    # covariance of the MLE) but we never surfaced them before; expose them.
    _bse = getattr(res, "bse", None)
    if _bse is not None:
        _se = pd.Series(np.asarray(_bse, dtype=float), index=_param_index)
    else:  # pragma: no cover - defensive; SARIMAX always populates bse
        _se = pd.Series(np.full(len(_params), np.nan), index=_param_index)

    _result = ARIMAResult(
        order=order,
        seasonal_order=seasonal_order,
        params=_params,
        se=_se,
        aic=float(aic),
        bic=float(bic),
        aicc=float(aicc),
        log_likelihood=float(res.llf),
        residuals=np.asarray(res.resid),
        fitted_values=np.asarray(res.fittedvalues),
        n=n,
        _model=res,
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
            },
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
