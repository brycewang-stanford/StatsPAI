"""Bootstrapped series and bagged forecasts.

A series is decomposed (Box-Cox transform, then STL or a loess trend),
its remainder is resampled in blocks so that its autocorrelation
survives, and the pieces are put back together. Forecasting each
bootstrapped series and averaging the forecasts ("bagging") usually beats
forecasting the original series alone (Bergmeir, Hyndman and Benitez,
2016).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._forecast_common import (
    boxcox_forward,
    boxcox_inverse,
    check_period,
    future_index,
    read_series,
)


def _loess_trend(x: np.ndarray, q: int = 6) -> np.ndarray:
    """Local linear fit through the ``q`` nearest points with tricube
    weights, evaluated at every observation."""
    n = x.shape[0]
    t = np.arange(n, dtype=float)
    out = np.empty(n)
    q = min(max(q, 2), n)
    for i in range(n):
        d = np.abs(t - t[i])
        idx = np.argpartition(d, q - 1)[:q]
        h = d[idx].max()
        w = (1.0 - (d[idx] / h) ** 3) ** 3 if h > 0 else np.ones(q)
        w = np.maximum(w, 0.0)
        X = np.column_stack([np.ones(q), t[idx] - t[i]])
        WX = X * w[:, None]
        try:
            beta = np.linalg.solve(X.T @ WX, WX.T @ x[idx])
            out[i] = beta[0]
        except np.linalg.LinAlgError:
            out[i] = float(np.average(x[idx], weights=w + 1e-12))
    return out


def _moving_blocks(r: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = r.shape[0]
    k = n // block + 2
    starts = rng.integers(0, n - block + 1, size=k)
    long = np.concatenate([r[s : s + block] for s in starts])
    offset = int(rng.integers(0, block))
    return np.asarray(long[offset : offset + n], dtype=float)


def bootstrap_series(
    y: Any,
    n_boot: int = 100,
    *,
    period: int = 1,
    block_size: Optional[int] = None,
    boxcox: Any = "auto",
    seed: Optional[int] = 0,
    data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Bootstrapped versions of a series that keep its trend, seasonal
    pattern and the autocorrelation of what is left.

    Parameters
    ----------
    y : array-like, pd.Series or str
        The series, in time order; a column name when ``data`` is given.
    n_boot : int, default 100
        Number of series returned. The first is the original.
    period : int, default 1
        Seasonal period. With ``period > 1`` the series is decomposed by
        STL with a periodic seasonal component; otherwise a loess trend
        is removed.
    block_size : int, optional
        Length of the resampled blocks of the remainder. Default twice
        the period, or ``min(8, n // 2)`` for non-seasonal data.
    boxcox : float, "auto" or None, default "auto"
        Box-Cox parameter applied before decomposing, so that the
        components add up. ``"auto"`` is Guerrero's choice restricted to
        ``[0, 1]``; ``None`` decomposes the series as it is, which is
        also what happens when the series is not strictly positive.
    seed : int or None, default 0
    data : pd.DataFrame, optional

    Returns
    -------
    pd.DataFrame
        One column per series, ``boot_0`` (the original) to
        ``boot_<n_boot - 1>``, on the index of the series.

    Notes
    -----
    The procedure is that of Bergmeir, Hyndman and Benitez (2016) and of
    R's ``forecast::bld.mbb.bootstrap``. The draws are random, so the
    series are not those R returns for the same seed.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(80)
    >>> y = (50 + t) * (1 + 0.2 * np.sin(2 * np.pi * t / 4)) + rng.normal(0, 2, 80)
    >>> boot = sp.bootstrap_series(y, 10, period=4)
    >>> boot.shape
    (80, 10)
    >>> bool(np.allclose(boot["boot_0"], y))
    True

    References
    ----------
    bergmeir2016bagging
    """
    values, index, _ = read_series(y, data, fn="bootstrap_series")
    m = check_period(period, fn="bootstrap_series")
    n = values.shape[0]
    if int(n_boot) < 1:
        raise MethodIncompatibility(
            "bootstrap_series: n_boot must be at least 1.",
            recovery_hint="Pass a positive number of series.",
        )
    if n < max(2 * m + 1, 8):
        raise DataInsufficient(
            f"bootstrap_series: {n} observations are too few.",
            recovery_hint="At least two seasonal cycles (or 8 points) are needed.",
        )
    lam: Optional[float] = None
    if boxcox is not None and boxcox is not False and values.min() > 0:
        if isinstance(boxcox, str):
            if boxcox.lower() != "auto":
                raise MethodIncompatibility(
                    f"bootstrap_series: boxcox={boxcox!r} is not a number, "
                    "'auto' or None.",
                    recovery_hint="Use boxcox='auto'.",
                )
            from .ts_tools import boxcox_lambda

            lam = float(boxcox_lambda(values, m, lower=0.0, upper=1.0))
        else:
            lam = float(boxcox)
    x = values if lam is None else boxcox_forward(values, lam)
    if m > 1:
        from .stl import stl

        dec = stl(x, m, seasonal="periodic")
        smooth = dec.trend + dec.seasonal
        rem = dec.remainder
    else:
        smooth = _loess_trend(x, 6)
        rem = x - smooth
    block = (
        int(block_size)
        if block_size is not None
        else (2 * m if m > 1 else min(8, n // 2))
    )
    if block < 1 or block > n:
        raise MethodIncompatibility(
            f"bootstrap_series: block_size={block} must be between 1 and {n}.",
            recovery_hint="Leave block_size=None for the default.",
        )
    rng = np.random.default_rng(seed)
    cols = {"boot_0": values.copy()}
    for i in range(1, int(n_boot)):
        z = smooth + _moving_blocks(rem, block, rng)
        cols[f"boot_{i}"] = z if lam is None else boxcox_inverse(z, lam)
    out = pd.DataFrame(cols)
    if index is not None:
        out.index = index
    return out


@dataclass
class BaggedForecastResult(ResultProtocolMixin):
    """Bagged forecasts returned by :func:`statspai.bagged_forecast`.

    Attributes
    ----------
    forecast : pd.DataFrame
        ``forecast`` (the average over the bootstrapped series) with
        ``ensemble_min`` and ``ensemble_max``, the range of the members.
        The range describes how much the forecast depends on the sample;
        it is not a prediction interval.
    members : pd.DataFrame
        The forecast from each bootstrapped series, one column each.
    forecaster : str
    n_boot : int

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> y = 100 + np.cumsum(rng.normal(0.5, 1.0, 60))
    >>> bag = sp.bagged_forecast(y, "drift", horizon=4, n_boot=20)
    >>> bag.forecast.shape, bag.members.shape
    ((4, 3), (4, 20))
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("bergmeir2016bagging",)

    forecast: pd.DataFrame
    members: pd.DataFrame = field(repr=False)
    forecaster: str = "ets"
    n_boot: int = 100

    def summary(self) -> str:
        table = str(self.forecast.to_string(float_format=lambda v: f"{v:.4f}"))
        return (
            f"Bagged forecast: {self.forecaster}, {self.n_boot} bootstrapped "
            f"series\n{'-' * 46}\n" + table
        )

    def __repr__(self) -> str:
        return self.summary()


def bagged_forecast(
    y: Any,
    forecaster: Union[str, Callable[..., Any]] = "ets",
    *,
    horizon: int = 10,
    n_boot: int = 100,
    period: int = 1,
    block_size: Optional[int] = None,
    boxcox: Any = "auto",
    seed: Optional[int] = 0,
    data: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> BaggedForecastResult:
    """Forecast every bootstrapped version of a series and average.

    Parameters
    ----------
    y : array-like, pd.Series or str
    forecaster : str or callable, default "ets"
        ``"ets"``, ``"arima"``, ``"naive"``, ``"snaive"``, ``"drift"``,
        ``"mean"`` (extra keyword arguments are passed on), or a function
        ``f(series, horizon)`` as in :func:`statspai.tscv`.
    horizon : int, default 10
    n_boot : int, default 100
        Number of series forecast, the original included.
    period, block_size, boxcox, seed
        As in :func:`statspai.bootstrap_series`.
    data : pd.DataFrame, optional

    Returns
    -------
    BaggedForecastResult
        ``forecast`` (average, minimum and maximum over the members) and
        ``members``.

    Notes
    -----
    With ``forecaster="ets"`` and the model chosen separately for each
    series this is the bagged ETS of Bergmeir, Hyndman and Benitez (2016),
    R's ``forecast::baggedETS``. Bagging averages over model choice and
    parameter estimates, the two sources of error a single fit ignores.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(72)
    >>> y = (40 + 0.5 * t) * (1 + 0.15 * np.sin(2 * np.pi * t / 4))
    >>> y = y + rng.normal(0, 1.5, 72)
    >>> bag = sp.bagged_forecast(y, "snaive", horizon=4, n_boot=10, period=4)
    >>> list(bag.forecast.columns)
    ['forecast', 'ensemble_min', 'ensemble_max']

    References
    ----------
    bergmeir2016bagging, hyndman2026fpppy
    """
    from .forecast_accuracy import _builtin, _extract

    values, index, name = read_series(y, data, fn="bagged_forecast")
    m = check_period(period, fn="bagged_forecast")
    h = int(horizon)
    if h < 1:
        raise MethodIncompatibility(
            "bagged_forecast: horizon must be at least 1.",
            recovery_hint="Pass a positive number of periods.",
        )
    boot = bootstrap_series(
        values, n_boot, period=m, block_size=block_size, boxcox=boxcox, seed=seed
    )
    if isinstance(forecaster, str):
        label = forecaster
        run = _builtin(forecaster, m, kwargs)
        as_series = False
    elif callable(forecaster):
        label = getattr(forecaster, "__name__", "forecaster")
        run = forecaster
        as_series = True
        if kwargs:
            raise MethodIncompatibility(
                f"bagged_forecast: unexpected arguments {sorted(kwargs)} with a "
                "callable forecaster.",
                recovery_hint="Bind them in the function itself.",
            )
    else:
        raise MethodIncompatibility(
            "bagged_forecast: forecaster must be a method name or a function.",
            recovery_hint="E.g. forecaster='ets'.",
        )
    members = np.empty((h, boot.shape[1]))
    for j, col in enumerate(boot.columns):
        series = boot[col].to_numpy(dtype=float)
        arg: Any = pd.Series(series, index=index, name=name) if as_series else series
        members[:, j] = _extract(run(arg, h), h)
    idx = future_index(index, values.shape[0], h)
    frame = pd.DataFrame(
        {
            "forecast": members.mean(axis=1),
            "ensemble_min": members.min(axis=1),
            "ensemble_max": members.max(axis=1),
        },
        index=idx,
    )
    return BaggedForecastResult(
        forecast=frame,
        members=pd.DataFrame(members, index=idx, columns=list(boot.columns)),
        forecaster=label,
        n_boot=int(boot.shape[1]),
    )
