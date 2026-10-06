"""The four benchmark forecasting methods: mean, naive, seasonal naive
and drift.

Every forecasting exercise needs them as the yardstick: a method that
does not beat the naive forecast out of sample is not worth its
complexity (Hyndman and Athanasopoulos, *Forecasting: Principles and
Practice*, chapter 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._forecast_common import (
    Levels,
    back_transform_frame,
    boxcox_forward,
    boxcox_inverse,
    check_period,
    forecast_frame,
    future_index,
    normalise_levels,
    path_quantiles,
    read_series,
    resolve_boxcox,
)

_ALIASES = {
    "mean": "mean",
    "average": "mean",
    "meanf": "mean",
    "historic_average": "mean",
    "naive": "naive",
    "rw": "naive",
    "random_walk": "naive",
    "snaive": "snaive",
    "seasonal_naive": "snaive",
    "drift": "drift",
    "rwd": "drift",
    "rwf": "drift",
    "random_walk_drift": "drift",
}


@dataclass
class SimpleForecastResult(ResultProtocolMixin):
    """A fitted benchmark method returned by :func:`statspai.simple_forecast`.

    Attributes
    ----------
    method : {"mean", "naive", "snaive", "drift"}
    period : int
        Seasonal period of the seasonal naive method (1 otherwise).
    fitted_values, residuals : np.ndarray
        One-step fitted values and residuals; ``nan`` for the first
        observations a method cannot fit.
    sigma : float
        Residual standard deviation, ``sqrt(SSE / (T - K - M))`` with
        ``K`` estimated parameters and ``M`` missing residuals.
    drift : float or None
        Average change per period (drift method).

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> y = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 16.0])
    >>> fit = sp.simple_forecast(y, "drift")
    >>> round(fit.drift, 4)
    0.8571
    >>> fit.forecast(2, level=95).round(2)["forecast"].tolist()
    [16.86, 17.71]
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("hyndman2026fpppy",)

    method: str
    period: int
    fitted_values: np.ndarray
    residuals: np.ndarray
    sigma: float
    n: int
    drift: Optional[float] = None
    mean: Optional[float] = None
    boxcox: Optional[float] = None
    biasadj: bool = False
    _y: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    _index: Optional[pd.Index] = field(default=None, repr=False)
    _name: str = "y"

    @property
    def n_arma_params(self) -> int:
        return 0

    def _back(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.boxcox is None:
            return frame
        return back_transform_frame(frame, self.boxcox, self.biasadj)

    def _point(self, h: int) -> np.ndarray:
        y = self._y
        steps = np.arange(1, h + 1, dtype=float)
        if self.method == "mean":
            return np.full(h, float(self.mean or 0.0))
        if self.method == "naive":
            return np.full(h, y[-1])
        if self.method == "drift":
            return np.asarray(y[-1] + steps * float(self.drift or 0.0), dtype=float)
        m = self.period
        return np.asarray(y[-m:][(np.arange(h)) % m], dtype=float)

    def _sd(self, h: int) -> np.ndarray:
        steps = np.arange(1, h + 1, dtype=float)
        T = self.n
        if self.method == "mean":
            return np.full(h, self.sigma * np.sqrt(1.0 + 1.0 / T))
        if self.method == "naive":
            return np.asarray(self.sigma * np.sqrt(steps), dtype=float)
        if self.method == "drift":
            return np.asarray(
                self.sigma * np.sqrt(steps * (1.0 + steps / (T - 1.0))), dtype=float
            )
        k = np.floor((steps - 1.0) / self.period)
        return np.asarray(self.sigma * np.sqrt(k + 1.0), dtype=float)

    def _paths(self, h: int, innov: np.ndarray) -> np.ndarray:
        """Future paths given innovations (paths by horizon)."""
        y = self._y
        if self.method == "mean":
            return np.asarray(float(self.mean or 0.0) + innov, dtype=float)
        if self.method == "naive":
            return np.asarray(y[-1] + np.cumsum(innov, axis=1), dtype=float)
        if self.method == "drift":
            steps = np.arange(1, h + 1, dtype=float)
            drift = float(self.drift or 0.0)
            return np.asarray(
                y[-1] + steps * drift + np.cumsum(innov, axis=1), dtype=float
            )
        m = self.period
        out = np.empty_like(innov)
        base = y[-m:]
        for k in range(h):
            prev = base[k % m] if k < m else out[:, k - m]
            out[:, k] = prev + innov[:, k]
        return out

    def forecast(
        self,
        horizon: int = 10,
        level: Levels = (80, 95),
        *,
        bootstrap: bool = False,
        n_paths: int = 5000,
        seed: Optional[int] = 0,
    ) -> pd.DataFrame:
        """Point forecasts and prediction intervals.

        Parameters
        ----------
        horizon : int, default 10
        level : float or sequence of float, default (80, 95)
            Coverage in percent.
        bootstrap : bool, default False
            Take the intervals from future paths built by resampling the
            residuals instead of assuming them normal.
        n_paths : int, default 5000
        seed : int or None, default 0

        Returns
        -------
        pd.DataFrame
            ``forecast``, ``lower_<level>``, ``upper_<level>``.
        """
        h = int(horizon)
        if h < 1:
            raise MethodIncompatibility(
                f"forecast: horizon must be at least 1, got {horizon!r}.",
                recovery_hint="Pass a positive number of periods.",
            )
        levels = normalise_levels(level)
        idx = future_index(self._index, self.n, h)
        mean = self._point(h)
        if not bootstrap:
            return self._back(forecast_frame(mean, levels, sd=self._sd(h), index=idx))
        res = self.residuals[np.isfinite(self.residuals)]
        res = res - res.mean()
        rng = np.random.default_rng(seed)
        innov = rng.choice(res, size=(int(n_paths), h), replace=True)
        paths = self._paths(h, innov)
        return self._back(
            forecast_frame(
                mean, levels, bounds=path_quantiles(paths, levels), index=idx
            )
        )

    def simulate(
        self,
        horizon: int = 10,
        n_paths: int = 5,
        *,
        bootstrap: bool = True,
        seed: Optional[int] = 0,
    ) -> pd.DataFrame:
        """Future sample paths, one per column."""
        h = int(horizon)
        rng = np.random.default_rng(seed)
        if bootstrap:
            res = self.residuals[np.isfinite(self.residuals)]
            res = res - res.mean()
            innov = rng.choice(res, size=(int(n_paths), h), replace=True)
        else:
            innov = rng.normal(0.0, self.sigma, size=(int(n_paths), h))
        paths = self._paths(h, innov)
        if self.boxcox is not None:
            paths = boxcox_inverse(paths, self.boxcox)
        return pd.DataFrame(
            paths.T,
            index=future_index(self._index, self.n, h),
            columns=[f"path_{i}" for i in range(int(n_paths))],
        )

    def summary(self) -> str:
        label = {
            "mean": "Mean method",
            "naive": "Naive method",
            "snaive": f"Seasonal naive method (period {self.period})",
            "drift": "Drift method",
        }[self.method]
        lines = [label, "-" * 40, f"n          : {self.n}"]
        if self.mean is not None:
            lines.append(f"mean       : {self.mean:.6g}")
        if self.drift is not None:
            lines.append(f"drift      : {self.drift:.6g}")
        lines.append(f"sigma      : {self.sigma:.6g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def simple_forecast(
    y: Any,
    method: str = "naive",
    *,
    period: int = 1,
    boxcox: Any = None,
    biasadj: bool = False,
    data: Optional[pd.DataFrame] = None,
) -> SimpleForecastResult:
    """Benchmark forecasting methods: mean, naive, seasonal naive, drift.

    Parameters
    ----------
    y : array-like, pd.Series or str
        The series, in time order; a column name when ``data`` is given.
    method : {"naive", "snaive", "drift", "mean"}, default "naive"
        ``"mean"``: every forecast is the historical average.
        ``"naive"``: the last observation (a random walk).
        ``"snaive"``: the last observation of the same season.
        ``"drift"``: the last observation plus the average historical
        change, a line through the first and the last observation.
    period : int, default 1
        Seasonal period; required for ``"snaive"``.
    boxcox : float or "auto", optional
        Apply the method to a Box-Cox transform of the series (0 is the
        logarithm; ``"auto"`` uses :func:`statspai.boxcox_lambda`) and
        back-transform the forecasts and intervals. ``fitted_values`` are
        on the scale of the data, ``residuals`` on the transformed scale.
    biasadj : bool, default False
        The back-transformed point forecast is the median of the forecast
        distribution; ``True`` returns its mean instead.
    data : pd.DataFrame, optional

    Returns
    -------
    SimpleForecastResult
        ``fitted_values``, ``residuals``, ``sigma`` and
        ``forecast(horizon, level, bootstrap=...)``.

    Notes
    -----
    Intervals are ``forecast +/- z * sigma_h`` with, for ``T``
    observations and seasonal period ``m``,

    ==========  =====================================
    mean        ``sigma * sqrt(1 + 1/T)``
    naive       ``sigma * sqrt(h)``
    snaive      ``sigma * sqrt(floor((h-1)/m) + 1)``
    drift       ``sigma * sqrt(h * (1 + h/(T-1)))``
    ==========  =====================================

    and ``sigma`` the residual standard deviation with the degrees of
    freedom of the method. They coincide with R's ``forecast::naive``,
    ``snaive`` and ``rwf(drift=TRUE)``. ``forecast::meanf`` uses a
    Student-t quantile where this function, like the formula of the
    book and ``fable::MEAN``, uses the normal one.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> y = np.tile([10.0, 20.0, 30.0, 40.0], 5) + np.arange(20)
    >>> fit = sp.simple_forecast(y, "snaive", period=4)
    >>> fit.forecast(4, level=80)["forecast"].tolist()
    [26.0, 37.0, 48.0, 59.0]

    References
    ----------
    hyndman2026fpppy
    """
    values, index, name = read_series(y, data, fn="simple_forecast")
    lam = resolve_boxcox(values, boxcox, int(period), fn="simple_forecast")
    if lam is not None:
        values = boxcox_forward(values, lam)
    key = _ALIASES.get(str(method).lower().replace("-", "_").replace(" ", "_"))
    if key is None:
        raise MethodIncompatibility(
            f"simple_forecast: method={method!r} is not a benchmark method.",
            recovery_hint="Use 'naive', 'snaive', 'drift' or 'mean'.",
        )
    m = check_period(period, fn="simple_forecast")
    n = values.shape[0]
    fitted = np.full(n, np.nan)
    drift: Optional[float] = None
    mean: Optional[float] = None
    if key == "mean":
        if n < 2:
            raise DataInsufficient(
                "simple_forecast: the mean method needs at least 2 observations.",
                recovery_hint="Provide a longer series.",
            )
        mean = float(values.mean())
        fitted[:] = mean
        dof = n - 1
    elif key == "naive":
        if n < 2:
            raise DataInsufficient(
                "simple_forecast: the naive method needs at least 2 observations.",
                recovery_hint="Provide a longer series.",
            )
        fitted[1:] = values[:-1]
        dof = n - 1
    elif key == "drift":
        if n < 3:
            raise DataInsufficient(
                "simple_forecast: the drift method needs at least 3 observations.",
                recovery_hint="Provide a longer series.",
            )
        drift = float((values[-1] - values[0]) / (n - 1))
        fitted[1:] = values[:-1] + drift
        dof = n - 2
    else:
        if m < 2:
            raise MethodIncompatibility(
                "simple_forecast: the seasonal naive method needs period >= 2.",
                recovery_hint="Pass period= (4 quarterly, 12 monthly, 7 daily).",
            )
        if n <= m:
            raise DataInsufficient(
                f"simple_forecast: the seasonal naive method needs more than "
                f"one cycle ({m} periods); the series has {n} observations.",
                recovery_hint="Provide a longer series or use method='naive'.",
            )
        fitted[m:] = values[:-m]
        dof = n - m
    resid = values - fitted
    sse = float(np.nansum(resid**2))
    sigma = float(np.sqrt(sse / dof)) if dof > 0 else float("nan")
    return SimpleForecastResult(
        method=key,
        period=m if key == "snaive" else 1,
        fitted_values=fitted if lam is None else boxcox_inverse(fitted, lam),
        boxcox=lam,
        biasadj=bool(biasadj),
        residuals=resid,
        sigma=sigma,
        n=n,
        drift=drift,
        mean=mean,
        _y=values,
        _index=index,
        _name=name,
    )
