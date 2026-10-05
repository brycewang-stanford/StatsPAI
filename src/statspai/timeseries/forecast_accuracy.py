"""Forecast evaluation: accuracy measures for point, interval and
distributional forecasts, and time series cross-validation on a rolling
forecast origin.

The scaled errors follow Hyndman and Koehler (2006); the interval score
is Winkler's (1972) and the quantile score and CRPS follow Gneiting and
Raftery (2007).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._forecast_common import check_period, level_label, read_series


def _as_float(a: Any) -> np.ndarray:
    if isinstance(a, (pd.Series, pd.DataFrame)):
        return np.asarray(a.to_numpy(dtype=float)).ravel()
    return np.asarray(a, dtype=float).ravel()


def _acf1(e: np.ndarray) -> float:
    e = e[np.isfinite(e)]
    if e.size < 2:
        return float("nan")
    d = e - e.mean()
    den = float(d @ d)
    return float(d[1:] @ d[:-1] / den) if den > 0 else float("nan")


def _scale(train: Optional[np.ndarray], m: int, power: int) -> float:
    """In-sample mean absolute (or squared) error of the (seasonal) naive
    forecast: the denominator of the scaled errors."""
    if train is None:
        return float("nan")
    tr = train[np.isfinite(train)]
    if tr.size <= m:
        return float("nan")
    d = np.abs(tr[m:] - tr[:-m]) ** power
    return float(d.mean())


def _one(
    actual: np.ndarray,
    fc: Union[pd.DataFrame, np.ndarray],
    train: Optional[np.ndarray],
    m: int,
    crps: bool,
) -> Dict[str, float]:
    if isinstance(fc, pd.DataFrame):
        if "forecast" not in fc.columns:
            raise MethodIncompatibility(
                "forecast_accuracy: a forecast table needs a 'forecast' column.",
                recovery_hint="Pass the frame returned by .forecast(), or an array.",
            )
        point = fc["forecast"].to_numpy(dtype=float)
        frame: Optional[pd.DataFrame] = fc
    else:
        point = _as_float(fc)
        frame = None
    if point.shape[0] != actual.shape[0]:
        raise MethodIncompatibility(
            f"forecast_accuracy: {point.shape[0]} forecasts for "
            f"{actual.shape[0]} actual values.",
            recovery_hint="Align the forecasts with the hold-out sample.",
        )
    e = actual - point
    ok = np.isfinite(e)
    if not ok.any():
        raise DataInsufficient(
            "forecast_accuracy: no pair of actual and forecast is finite.",
            recovery_hint="Check the alignment of the two inputs.",
        )
    ee = e[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        pe = 100.0 * e / actual
    pe = pe[ok & np.isfinite(pe)]
    out: Dict[str, float] = {
        "ME": float(ee.mean()),
        "RMSE": float(np.sqrt(np.mean(ee**2))),
        "MAE": float(np.mean(np.abs(ee))),
        "MPE": float(pe.mean()) if pe.size else float("nan"),
        "MAPE": float(np.mean(np.abs(pe))) if pe.size else float("nan"),
    }
    q1 = _scale(train, m, 1)
    q2 = _scale(train, m, 2)
    out["MASE"] = out["MAE"] / q1 if q1 > 0 else float("nan")
    out["RMSSE"] = out["RMSE"] / np.sqrt(q2) if q2 > 0 else float("nan")
    out["ACF1"] = _acf1(e)
    if frame is None:
        return out
    widest: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    for col in frame.columns:
        if not col.startswith("lower_"):
            continue
        lab = col[len("lower_") :]
        up = f"upper_{lab}"
        if up not in frame.columns:
            continue
        pct = float(lab)
        a = 1.0 - pct / 100.0
        lo = frame[col].to_numpy(dtype=float)
        hi = frame[up].to_numpy(dtype=float)
        k = ok & np.isfinite(lo) & np.isfinite(hi)
        y, lo_k, hi_k = actual[k], lo[k], hi[k]
        w = (hi_k - lo_k) + (2.0 / a) * (
            np.maximum(lo_k - y, 0.0) + np.maximum(y - hi_k, 0.0)
        )
        out[f"winkler_{level_label(pct)}"] = float(w.mean())
        out[f"coverage_{level_label(pct)}"] = float(np.mean((y >= lo_k) & (y <= hi_k)))
        if widest is None or pct > widest[0]:
            widest = (pct, lo, hi)
    if crps:
        if widest is None:
            raise MethodIncompatibility(
                "forecast_accuracy: crps=True needs prediction intervals.",
                recovery_hint="Pass the frame returned by .forecast(level=...).",
            )
        pct, lo, hi = widest
        z = float(stats.norm.ppf(0.5 + pct / 200.0))
        sd = (hi - lo) / (2.0 * z)
        k = ok & np.isfinite(sd) & (sd > 0)
        zz = (actual[k] - point[k]) / sd[k]
        val = sd[k] * (
            zz * (2.0 * stats.norm.cdf(zz) - 1.0)
            + 2.0 * stats.norm.pdf(zz)
            - 1.0 / np.sqrt(np.pi)
        )
        out["CRPS"] = float(val.mean())
    return out


def forecast_accuracy(
    actual: Any,
    forecast: Any,
    *,
    train: Any = None,
    period: int = 1,
    crps: bool = False,
) -> pd.DataFrame:
    """Accuracy measures of one or several forecasts against the outcomes.

    Parameters
    ----------
    actual : array-like
        Observed values of the forecast periods.
    forecast : array-like, pd.DataFrame or dict
        Point forecasts; or the table returned by a ``.forecast()`` method
        (a ``forecast`` column and ``lower_<level>`` / ``upper_<level>``
        columns), which adds interval measures; or a dict of either,
        keyed by the name of the method, to compare several.
    train : array-like, optional
        The series the forecasts were fitted on. Needed for the scaled
        errors ``MASE`` and ``RMSSE``, which divide by the in-sample
        error of the naive forecast (the seasonal naive one when
        ``period > 1``).
    period : int, default 1
        Seasonal period used for that scaling.
    crps : bool, default False
        Add the continuous ranked probability score of the normal
        distribution implied by the point forecast and the widest
        interval. Exact for the analytic intervals of ``sp.ets``,
        ``sp.arima`` and ``sp.simple_forecast``; an approximation for
        simulated or bootstrapped ones.

    Returns
    -------
    pd.DataFrame
        One row per forecast with

        - ``ME``, ``RMSE``, ``MAE``: mean, root mean squared and mean
          absolute error, in the units of the data;
        - ``MPE``, ``MAPE``: mean and mean absolute percentage error;
        - ``MASE``, ``RMSSE``: scaled errors (below one beats the naive
          forecast in sample);
        - ``ACF1``: first autocorrelation of the errors;
        - ``winkler_<level>``, ``coverage_<level>``: the interval score
          (width plus ``2/alpha`` times the miss) and the share of
          outcomes inside the interval;
        - ``CRPS`` when asked for.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> y = np.tile([10.0, 20.0, 30.0, 40.0], 6) + np.arange(24)
    >>> train, test = y[:20], y[20:]
    >>> fcs = {
    ...     "naive": sp.simple_forecast(train, "naive").forecast(4),
    ...     "snaive": sp.simple_forecast(train, "snaive", period=4).forecast(4),
    ... }
    >>> acc = sp.forecast_accuracy(test, fcs, train=train, period=4)
    >>> acc["RMSE"].round(2).to_dict()
    {'naive': 17.54, 'snaive': 4.0}

    References
    ----------
    hyndman2006another, gneiting2007strictly, winkler1972decision
    """
    y = _as_float(actual)
    m = check_period(period, fn="forecast_accuracy")
    tr = None if train is None else _as_float(train)
    if isinstance(forecast, Mapping):
        items = [(str(k), v) for k, v in forecast.items()]
    else:
        items = [("forecast", forecast)]
    rows = {name: _one(y, fc, tr, m, crps) for name, fc in items}
    return pd.DataFrame.from_dict(rows, orient="index")


# ----------------------------------------------------------------------
# rolling-origin cross-validation
# ----------------------------------------------------------------------
Forecaster = Union[str, Callable[..., Any]]


@dataclass
class TSCVResult(ResultProtocolMixin):
    """Rolling-origin forecast errors returned by :func:`statspai.tscv`.

    Attributes
    ----------
    errors : pd.DataFrame
        ``actual - forecast``; one row per forecast origin (labelled by
        the last observation of its training window), one column per
        horizon ``h=1..H``. ``nan`` where the outcome lies beyond the
        sample or the forecaster failed.
    forecasts, actuals : pd.DataFrame
        Laid out like ``errors``.
    failures : list of (origin, message)
        Origins where the forecaster raised.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = 100 + np.cumsum(rng.normal(0.2, 1.0, 60))
    >>> cv = sp.tscv(y, "drift", horizon=2, initial=20)
    >>> cv.errors.shape
    (40, 2)
    >>> list(cv.accuracy().columns)
    ['RMSE', 'MAE', 'MAPE', 'n']
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("hyndman2026fpppy",)

    errors: pd.DataFrame
    forecasts: pd.DataFrame
    actuals: pd.DataFrame
    horizon: int
    initial: int
    step: int
    window: Optional[int]
    forecaster: str
    failures: List[Tuple[Any, str]] = field(default_factory=list)

    def accuracy(self) -> pd.DataFrame:
        """RMSE, MAE and MAPE by forecast horizon, over the origins."""
        rows = {}
        for col in self.errors.columns:
            e = self.errors[col].to_numpy(dtype=float)
            a = self.actuals[col].to_numpy(dtype=float)
            k = np.isfinite(e)
            if not k.any():
                rows[col] = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "n": 0}
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                pe = 100.0 * np.abs(e[k] / a[k])
            rows[col] = {
                "RMSE": float(np.sqrt(np.mean(e[k] ** 2))),
                "MAE": float(np.mean(np.abs(e[k]))),
                "MAPE": float(np.mean(pe[np.isfinite(pe)])),
                "n": int(k.sum()),
            }
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = "horizon"
        return out

    def summary(self) -> str:
        acc = self.accuracy()
        lines = [
            f"Time series cross-validation: {self.forecaster}",
            "-" * 46,
            f"origins    : {len(self.errors)}"
            + (f"  ({len(self.failures)} failed)" if self.failures else ""),
            f"first train: {self.initial} observations"
            + (f", rolling window {self.window}" if self.window else ", expanding"),
            "",
            acc.to_string(float_format=lambda v: f"{v:.4f}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _builtin(name: str, period: int, kwargs: Dict[str, Any]) -> Callable[..., Any]:
    key = name.lower()
    if key in ("ets",):
        from ._ets import ets

        def run(train: np.ndarray, h: int) -> np.ndarray:
            fit = ets(train, period=period, **kwargs)
            return np.asarray(fit.forecast(h, level=95)["forecast"], dtype=float)

        return run
    if key in ("arima", "auto_arima"):
        from .arima import arima

        def run(train: np.ndarray, h: int) -> np.ndarray:
            kw = dict(kwargs)
            if "order" not in kw:
                kw.setdefault("auto", True)
            if period > 1 and "seasonal_order" not in kw and kw.get("auto"):
                kw.setdefault("period", period)
            fit = arima(train, **kw)
            return np.asarray(fit.forecast(h)["forecast"], dtype=float)

        return run
    from .simple_forecast import _ALIASES, simple_forecast

    if key.replace("-", "_") in _ALIASES:

        def run(train: np.ndarray, h: int) -> np.ndarray:
            fit = simple_forecast(train, key, period=period)
            return np.asarray(fit.forecast(h, level=95)["forecast"], dtype=float)

        return run
    raise MethodIncompatibility(
        f"tscv: forecaster={name!r} is not a built-in method.",
        recovery_hint=(
            "Use 'naive', 'snaive', 'drift', 'mean', 'ets', 'arima', or a "
            "function f(train, horizon) returning the forecasts."
        ),
    )


def _extract(out: Any, h: int) -> np.ndarray:
    if hasattr(out, "forecast") and callable(out.forecast):
        out = out.forecast(h)
    if isinstance(out, pd.DataFrame):
        col = "forecast" if "forecast" in out.columns else out.columns[0]
        out = out[col]
    arr = np.asarray(out, dtype=float).ravel()
    if arr.shape[0] < h:
        raise MethodIncompatibility(
            f"the forecaster returned {arr.shape[0]} values, {h} needed",
            recovery_hint="Return one forecast per step of the horizon.",
        )
    return arr[:h]


def tscv(
    y: Any,
    forecaster: Forecaster = "naive",
    *,
    horizon: int = 1,
    initial: Optional[int] = None,
    step: int = 1,
    window: Optional[int] = None,
    period: int = 1,
    data: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> TSCVResult:
    """Time series cross-validation on a rolling forecast origin.

    For each origin the forecaster sees only the observations up to it
    and forecasts the next ``horizon`` periods; the errors against what
    then happened estimate the out-of-sample accuracy at each horizon.

    Parameters
    ----------
    y : array-like, pd.Series or str
        The series, in time order; a column name when ``data`` is given.
    forecaster : str or callable, default "naive"
        A built-in method -- ``"naive"``, ``"snaive"``, ``"drift"``,
        ``"mean"``, ``"ets"``, ``"arima"`` (extra keyword arguments are
        passed to it) -- or a function ``f(train, horizon)``. The
        function receives the training window as a ``pd.Series`` (with
        the index of ``y``) and returns the ``horizon`` forecasts, a
        table with a ``forecast`` column, or a fitted model with a
        ``.forecast(horizon)`` method.
    horizon : int, default 1
        Longest forecast horizon.
    initial : int, optional
        Number of observations in the first training window. The default
        is the larger of 10 and two seasonal periods plus one, the least
        the built-in methods can be fitted on; choose it so that the
        first window is long enough for the model you evaluate.
    step : int, default 1
        Number of periods between successive origins.
    window : int, optional
        Length of a rolling training window. ``None`` (default) keeps
        every past observation, an expanding window.
    period : int, default 1
        Seasonal period handed to the built-in methods.
    data : pd.DataFrame, optional

    Returns
    -------
    TSCVResult
        ``errors`` (origins by horizons), ``forecasts``, ``actuals``,
        ``accuracy()`` by horizon and ``failures``.

    Notes
    -----
    The error matrix has the layout of R's ``forecast::tsCV``: the row of
    an origin holds ``y[origin + h] - forecast``. Origins where the
    forecaster raises are reported in ``failures`` and warned about; they
    are not dropped silently.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> y = 50 + np.cumsum(rng.normal(0.3, 1.0, 80))
    >>> cv = sp.tscv(y, "drift", horizon=4, initial=30)
    >>> cv.accuracy()["RMSE"].is_monotonic_increasing
    True
    >>> ses = sp.tscv(y, lambda train, h: sp.ets(train, "ANN"), horizon=1, initial=30)
    >>> ses.errors.shape
    (50, 1)

    References
    ----------
    hyndman2026fpppy
    """
    values, index, name = read_series(y, data, fn="tscv")
    m = check_period(period, fn="tscv")
    h = int(horizon)
    st = int(step)
    if h < 1 or st < 1:
        raise MethodIncompatibility(
            "tscv: horizon and step must be at least 1.",
            recovery_hint="Pass positive integers.",
        )
    n = values.shape[0]
    first = int(initial) if initial is not None else max(10, 2 * m + 1)
    if window is not None:
        if int(window) < 2:
            raise MethodIncompatibility(
                "tscv: window must be at least 2.",
                recovery_hint="Use window=None for an expanding window.",
            )
        first = max(first, int(window))
    if first < 1 or first >= n:
        raise DataInsufficient(
            f"tscv: the first training window ({first} observations) leaves "
            f"nothing to forecast in a series of {n}.",
            recovery_hint="Lower initial= (or window=) or use a longer series.",
        )
    if isinstance(forecaster, str):
        label = forecaster
        fn_builtin: Optional[Callable[..., Any]] = _builtin(forecaster, m, kwargs)
        fn_user: Optional[Callable[..., Any]] = None
    elif callable(forecaster):
        label = getattr(forecaster, "__name__", "forecaster")
        fn_builtin, fn_user = None, forecaster
        if kwargs:
            raise MethodIncompatibility(
                f"tscv: unexpected arguments {sorted(kwargs)} with a callable "
                "forecaster.",
                recovery_hint="Bind them in the function itself.",
            )
    else:
        raise MethodIncompatibility(
            "tscv: forecaster must be a method name or a function.",
            recovery_hint="E.g. forecaster='ets' or lambda train, h: ...",
        )
    origins = list(range(first, n, st))
    fc = np.full((len(origins), h), np.nan)
    act = np.full((len(origins), h), np.nan)
    failures: List[Tuple[Any, str]] = []
    labels = []
    for r, end in enumerate(origins):
        start = 0 if window is None else end - int(window)
        lab = (end - 1) if index is None else index[end - 1]
        labels.append(lab)
        k = min(h, n - end)
        act[r, :k] = values[end : end + k]
        try:
            if fn_builtin is not None:
                out = fn_builtin(values[start:end], h)
            else:
                assert fn_user is not None
                tr = pd.Series(
                    values[start:end],
                    index=None if index is None else index[start:end],
                    name=name,
                )
                out = fn_user(tr, h)
            fc[r] = _extract(out, h)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append((lab, f"{type(exc).__name__}: {exc}"))
    if failures:
        if len(failures) == len(origins):
            raise DataInsufficient(
                f"tscv: the forecaster failed at every origin; first error: "
                f"{failures[0][1]}",
                recovery_hint="Raise initial= or check the forecaster.",
                diagnostics={"n_origins": len(origins)},
            )
        warnings.warn(
            f"tscv: the forecaster failed at {len(failures)} of {len(origins)} "
            f"origins (first: {failures[0][1]}); their errors are missing. "
            "See .failures.",
            UserWarning,
            stacklevel=2,
        )
    cols = [f"h={j}" for j in range(1, h + 1)]
    idx = pd.Index(labels, name="origin")
    return TSCVResult(
        errors=pd.DataFrame(act - fc, index=idx, columns=cols),
        forecasts=pd.DataFrame(fc, index=idx, columns=cols),
        actuals=pd.DataFrame(act, index=idx, columns=cols),
        horizon=h,
        initial=first,
        step=st,
        window=None if window is None else int(window),
        forecaster=label,
        failures=failures,
    )
