"""Shared pieces of the forecasting functions: reading a series, building
the index of the forecast periods, laying out prediction intervals and the
classical decomposition used for starting values.

Kept in one place so that ``sp.ets``, ``sp.simple_forecast``, ``sp.stl``,
``sp.arima`` and ``sp.tscv`` agree on what a series, a level and a
forecast table are.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility

Levels = Union[float, Sequence[float]]


def read_series(
    y: Any,
    data: Optional[pd.DataFrame],
    *,
    fn: str,
    allow_missing: bool = False,
) -> Tuple[np.ndarray, Optional[pd.Index], str]:
    """Return ``(values, index, name)`` of a series given as an array, a
    ``pd.Series`` or a column name of ``data``.

    Leading and trailing missing values are trimmed (a differenced or
    lagged column can be passed as it is); a missing value inside the
    series is refused unless ``allow_missing``.
    """
    index: Optional[pd.Index] = None
    name = "y"
    if data is not None:
        if not isinstance(y, str) or y not in data.columns:
            raise MethodIncompatibility(
                f"{fn}: with data=, y must be the name of one of its columns.",
                recovery_hint="Check the column name.",
            )
        name = y
        index = data.index
        values = data[y].to_numpy(dtype=float)
    elif isinstance(y, str):
        raise MethodIncompatibility(
            f"{fn}: y={y!r} names a column; pass data= as well.",
            recovery_hint=f"sp.{fn}({y!r}, data=df) or sp.{fn}(df[{y!r}]).",
        )
    elif isinstance(y, pd.Series):
        name = str(y.name) if y.name is not None else "y"
        index = y.index
        values = y.to_numpy(dtype=float)
    elif isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise MethodIncompatibility(
                f"{fn}: y is a DataFrame with {y.shape[1]} columns; one "
                "series is expected.",
                recovery_hint="Pass one column, or its name with data=.",
            )
        name = str(y.columns[0])
        index = y.index
        values = y.iloc[:, 0].to_numpy(dtype=float)
    else:
        values = np.asarray(y, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    ok = np.flatnonzero(np.isfinite(values))
    if ok.size == 0:
        raise DataInsufficient(
            f"{fn}: the series has no finite values.",
            recovery_hint="Check the column.",
        )
    lo, hi = int(ok[0]), int(ok[-1]) + 1
    values = values[lo:hi]
    if index is not None:
        index = index[lo:hi]
    if not allow_missing and not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise MethodIncompatibility(
            f"{fn}: the series has {n_bad} missing value(s) between its "
            "first and last observation.",
            recovery_hint=(
                "Interpolate them first (e.g. Series.interpolate()), or "
                "model the longest complete stretch."
            ),
            diagnostics={"n_missing_inside": n_bad},
        )
    if isinstance(index, pd.RangeIndex) and index.step == 1 and index.start == 0:
        index = None
    return values, index, name


def normalise_levels(level: Levels) -> Tuple[float, ...]:
    """Coverage levels in percent, ascending. ``0.95`` and ``95`` are the
    same request."""
    raw: Iterable[float]
    if np.isscalar(level):
        raw = [float(level)]  # type: ignore[arg-type]
    else:
        raw = [float(v) for v in level]  # type: ignore[union-attr]
    out = []
    for v in raw:
        pct = v * 100.0 if 0.0 < v < 1.0 else v
        if not 0.0 < pct < 100.0:
            raise MethodIncompatibility(
                f"level={v!r} is not a coverage between 0 and 100 percent.",
                recovery_hint="Use e.g. level=(80, 95).",
            )
        out.append(pct)
    return tuple(sorted(set(out)))


def level_label(pct: float) -> str:
    """``95.0`` -> ``'95'``, ``99.5`` -> ``'99.5'``."""
    return f"{pct:g}"


def future_index(index: Optional[pd.Index], n: int, horizon: int) -> pd.Index:
    """Index of the ``horizon`` periods after a series of length ``n``.

    A date or period index with a regular frequency is continued; anything
    else gets the positions ``n, n + 1, ...``.
    """
    if index is not None and len(index) >= 3:
        try:
            if isinstance(index, pd.PeriodIndex):
                return pd.period_range(index[-1] + 1, periods=horizon, freq=index.freq)
            if isinstance(index, pd.DatetimeIndex):
                freq = index.freq or pd.infer_freq(index)
                if freq is not None:
                    full = pd.date_range(index[-1], periods=horizon + 1, freq=freq)
                    return full[1:]
            elif pd.api.types.is_integer_dtype(index):
                steps = np.diff(np.asarray(index, dtype=np.int64))
                if steps.size and (steps == steps[0]).all() and steps[0] > 0:
                    start = int(index[-1]) + int(steps[0])
                    return pd.Index(start + int(steps[0]) * np.arange(horizon))
        except (ValueError, TypeError):
            pass
    return pd.RangeIndex(n, n + horizon)


def forecast_frame(
    mean: np.ndarray,
    levels: Tuple[float, ...],
    *,
    sd: Optional[np.ndarray] = None,
    bounds: Optional[Dict[float, Tuple[np.ndarray, np.ndarray]]] = None,
    index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Forecast table: ``forecast`` and, per level, ``lower_<L>`` and
    ``upper_<L>``.

    Normal intervals ``mean +/- z * sd`` when ``sd`` is given; otherwise
    the bounds supplied per level (simulated or bootstrapped quantiles).
    """
    cols: Dict[str, np.ndarray] = {"forecast": np.asarray(mean, dtype=float)}
    for pct in levels:
        lab = level_label(pct)
        if bounds is not None and pct in bounds:
            lo, hi = bounds[pct]
        elif sd is not None:
            z = float(stats.norm.ppf(0.5 + pct / 200.0))
            lo = cols["forecast"] - z * sd
            hi = cols["forecast"] + z * sd
        else:  # pragma: no cover - callers always give one of the two
            raise MethodIncompatibility(
                "forecast_frame needs sd or bounds",
                recovery_hint="Internal: pass a standard deviation or bounds.",
            )
        cols[f"lower_{lab}"] = np.asarray(lo, dtype=float)
        cols[f"upper_{lab}"] = np.asarray(hi, dtype=float)
    out = pd.DataFrame(cols)
    if index is not None:
        out.index = index
    return out


def path_quantiles(
    paths: np.ndarray, levels: Tuple[float, ...]
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """Per-level lower and upper quantiles of simulated paths (paths by
    horizon), with the median-unbiased quantile definition (R type 8)."""
    out: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for pct in levels:
        a = 0.5 - pct / 200.0
        lo = np.nanquantile(paths, a, axis=0, method="median_unbiased")
        hi = np.nanquantile(paths, 1.0 - a, axis=0, method="median_unbiased")
        out[pct] = (lo, hi)
    return out


def check_period(period: Any, *, fn: str) -> int:
    """A seasonal period as a positive integer."""
    try:
        m = int(period)
    except (TypeError, ValueError):
        m = -1
    if m < 1 or m != period:
        raise MethodIncompatibility(
            f"{fn}: period must be a positive integer, got {period!r}.",
            recovery_hint="4 for quarterly data, 12 for monthly, 7 for daily.",
        )
    return m


def centred_moving_average(x: np.ndarray, m: int) -> np.ndarray:
    """Centred moving average of order ``m`` (a ``2 x m`` average when
    ``m`` is even); ``nan`` where the window does not fit."""
    n = x.shape[0]
    out = np.full(n, np.nan)
    if m % 2 == 1:
        w = np.full(m, 1.0 / m)
    else:
        w = np.full(m + 1, 1.0 / m)
        w[0] = w[-1] = 0.5 / m
    k = w.shape[0]
    half = k // 2
    if n >= k:
        out[half : n - half] = np.convolve(x, w[::-1], mode="valid")
    return out


def classical_decomposition(
    x: np.ndarray, m: int, multiplicative: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classical decomposition into ``(trend, seasonal, remainder)``.

    The trend is a centred moving average of order ``m``; the seasonal
    figure is the average detrended value at each position of the cycle,
    centred to sum to zero (to average one when multiplicative).
    """
    n = x.shape[0]
    trend = centred_moving_average(x, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        detr = x / trend if multiplicative else x - trend
    figure = np.empty(m)
    for j in range(m):
        figure[j] = np.nanmean(detr[j::m])
    if multiplicative:
        figure = figure / figure.mean()
    else:
        figure = figure - figure.mean()
    seasonal = np.tile(figure, n // m + 1)[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        rem = x / trend / seasonal if multiplicative else x - trend - seasonal
    return trend, seasonal, rem


# ----------------------------------------------------------------------
# Box-Cox transformation of the series a forecaster models
# ----------------------------------------------------------------------
def boxcox_forward(y: np.ndarray, lam: float) -> np.ndarray:
    """``log(y)`` when ``lam`` is zero, ``(y^lam - 1) / lam`` otherwise."""
    if lam == 0.0:
        return np.asarray(np.log(y), dtype=float)
    return np.asarray((np.power(y, lam) - 1.0) / lam, dtype=float)


def boxcox_inverse(
    x: np.ndarray, lam: float, fvar: Optional[np.ndarray] = None
) -> np.ndarray:
    """Back-transform; with ``fvar`` (the forecast variance on the
    transformed scale) return the bias-adjusted mean instead of the median.

    The adjustment is the second-order one of Hyndman and Athanasopoulos:
    ``exp(x) (1 + fvar / 2)`` for the logarithm and
    ``(lam x + 1)^(1/lam) (1 + fvar (1 - lam) / (2 (lam x + 1)^2))``
    otherwise.
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore", over="ignore"):
        if lam == 0.0:
            out = np.exp(x)
            if fvar is not None:
                out = out * (1.0 + 0.5 * fvar)
            return np.asarray(out, dtype=float)
        base = lam * x + 1.0
        base = np.where(base > 0.0, base, np.nan)
        out = np.power(base, 1.0 / lam)
        if fvar is not None:
            out = out * (1.0 + 0.5 * fvar * (1.0 - lam) / base**2)
    return np.asarray(out, dtype=float)


def resolve_boxcox(
    values: np.ndarray, boxcox: Any, period: int, *, fn: str
) -> Optional[float]:
    """The Box-Cox parameter a forecaster was asked to use: ``None``, a
    number, or ``"auto"`` for Guerrero's choice."""
    if boxcox is None or boxcox is False:
        return None
    if isinstance(boxcox, str):
        if boxcox.lower() != "auto":
            raise MethodIncompatibility(
                f"{fn}: boxcox={boxcox!r} is not a number or 'auto'.",
                recovery_hint="Pass the parameter (0 is the logarithm) or 'auto'.",
            )
        from .ts_tools import boxcox_lambda

        lam = float(boxcox_lambda(values, period))
    else:
        lam = float(boxcox)
    if not np.isfinite(lam):
        raise MethodIncompatibility(
            f"{fn}: boxcox={boxcox!r} is not finite.",
            recovery_hint="Pass a number such as 0 or 0.5.",
        )
    if np.nanmin(values) <= 0:
        raise MethodIncompatibility(
            f"{fn}: a Box-Cox transformation needs a strictly positive "
            f"series; the minimum is {np.nanmin(values):g}.",
            recovery_hint="Drop boxcox=, or shift the series.",
        )
    return lam


def back_transform_frame(
    frame: pd.DataFrame, lam: float, biasadj: bool
) -> pd.DataFrame:
    """Forecast table on the transformed scale -> table on the scale of
    the data. Bounds are back-transformed as they are (quantiles commute
    with a monotone map); the point forecast is the median, or with
    ``biasadj`` the mean, using the variance implied by the widest
    interval."""
    out = frame.copy()
    fvar: Optional[np.ndarray] = None
    if biasadj:
        best = None
        for col in frame.columns:
            if col.startswith("lower"):
                lab = col[len("lower") :].lstrip("_")
                up = "upper" + col[len("lower") :]
                if up in frame.columns:
                    pct = float(lab) if lab else float(frame.attrs.get("level", 95.0))
                    if best is None or pct > best[0]:
                        best = (pct, col, up)
        if best is None:
            raise MethodIncompatibility(
                "biasadj=True needs prediction intervals to take the forecast "
                "variance from.",
                recovery_hint="Ask for at least one level.",
            )
        pct, lo_c, up_c = best
        z = float(stats.norm.ppf(0.5 + pct / 200.0))
        half = frame[up_c].to_numpy(dtype=float) - frame[lo_c].to_numpy(dtype=float)
        fvar = (half / (2.0 * z)) ** 2
    for col in frame.columns:
        x = frame[col].to_numpy(dtype=float)
        out[col] = boxcox_inverse(x, lam, fvar if col == "forecast" else None)
    return out
