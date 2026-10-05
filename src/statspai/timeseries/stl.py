"""Time series decomposition: STL (seasonal-trend decomposition by
loess), its extension to several seasonal periods, and the classical
moving-average decomposition.

The loess recursion is the one of Cleveland, Cleveland, McRae and
Terpenning (1990), run by ``statsmodels.tsa.seasonal.STL``; this module
sets its windows, degrees, iteration counts and interpolation jumps the
way R's ``stats::stl`` and ``forecast::mstl`` do, adds the periodic
seasonal option, the multiple-seasonality loop of Bandara, Hyndman and
Bergmeir, the strength-of-trend and strength-of-seasonality measures
and forecasting from the decomposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._forecast_common import (
    Levels,
    check_period,
    classical_decomposition,
    future_index,
    normalise_levels,
    read_series,
)


def _next_odd(x: float) -> int:
    k = int(math.ceil(x))
    return k if k % 2 == 1 else k + 1


@dataclass
class DecompositionResult(ResultProtocolMixin):
    """Decomposition of a series returned by :func:`statspai.stl` and
    :func:`statspai.classical_decompose`.

    Attributes
    ----------
    observed, trend, remainder : np.ndarray
    seasonal : np.ndarray
        The seasonal component (the sum of them with several periods;
        the separate ones are in ``seasonal_components``).
    periods : tuple of int
    multiplicative : bool
        Whether ``observed = trend * seasonal * remainder`` rather than
        their sum (classical decomposition only).
    method : str

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> t = np.arange(120)
    >>> y = 0.1 * t + 3 * np.sin(2 * np.pi * t / 12) + np.cos(t)
    >>> dec = sp.stl(y, period=12)
    >>> dec.strength["seasonal"] > 0.9
    True
    >>> list(dec.to_frame().columns)
    ['observed', 'trend', 'seasonal', 'remainder', 'seasonally_adjusted']
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("cleveland1990stl",)

    observed: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    remainder: np.ndarray
    periods: Tuple[int, ...]
    method: str
    multiplicative: bool = False
    seasonal_components: Dict[int, np.ndarray] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    _index: Optional[pd.Index] = field(default=None, repr=False)
    _name: str = "y"

    @property
    def seasonally_adjusted(self) -> np.ndarray:
        """The series with the seasonal component removed."""
        if self.multiplicative:
            return np.asarray(self.observed / self.seasonal, dtype=float)
        return np.asarray(self.observed - self.seasonal, dtype=float)

    @property
    def strength(self) -> Dict[str, float]:
        """Strength of trend and of seasonality, each between 0 and 1:
        ``max(0, 1 - Var(R) / Var(T + R))`` and
        ``max(0, 1 - Var(R) / Var(S + R))`` (Wang, Smith and Hyndman,
        2006). With several seasonal periods there is one
        ``seasonal_<period>`` entry per period as well."""
        ok = np.isfinite(self.trend) & np.isfinite(self.remainder)
        if self.multiplicative:
            r = np.log(self.remainder[ok])
            t = np.log(self.trend[ok])
            s = np.log(self.seasonal[ok])
            comps = {m: np.log(v[ok]) for m, v in self.seasonal_components.items()}
        else:
            r, t, s = self.remainder[ok], self.trend[ok], self.seasonal[ok]
            comps = {m: v[ok] for m, v in self.seasonal_components.items()}
        vr = float(np.var(r, ddof=1))

        def share(x: np.ndarray) -> float:
            den = float(np.var(x + r, ddof=1))
            return float(max(0.0, min(1.0, 1.0 - vr / den))) if den > 0 else 0.0

        out = {"trend": share(t), "seasonal": share(s)}
        if len(comps) > 1:
            for m, v in comps.items():
                out[f"seasonal_{m}"] = share(v)
        return out

    def to_frame(self) -> pd.DataFrame:
        """Components as columns, on the index of the series."""
        cols: Dict[str, np.ndarray] = {"observed": self.observed, "trend": self.trend}
        if len(self.seasonal_components) > 1:
            for m, v in self.seasonal_components.items():
                cols[f"seasonal_{m}"] = v
        else:
            cols["seasonal"] = self.seasonal
        cols["remainder"] = self.remainder
        cols["seasonally_adjusted"] = self.seasonally_adjusted
        out = pd.DataFrame(cols)
        if self._index is not None:
            out.index = self._index
        return out

    def forecast(
        self,
        horizon: int = 10,
        level: Levels = (80, 95),
        *,
        method: str = "ets",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Forecast by modelling the seasonally adjusted series and adding
        back a seasonal naive forecast of the seasonal component.

        Parameters
        ----------
        horizon : int, default 10
        level : float or sequence of float, default (80, 95)
        method : {"ets", "naive", "drift", "arima"}, default "ets"
            Model of the seasonally adjusted series: a non-seasonal ETS
            model chosen by AICc, the naive or drift method, or a
            non-seasonal ARIMA model chosen automatically. Further
            keyword arguments go to that model.

        Notes
        -----
        The intervals are those of the seasonally adjusted series shifted
        by the seasonal forecast: the uncertainty of the seasonal
        component is ignored, as in R's ``forecast::stlf``.
        """
        if self.multiplicative:
            raise MethodIncompatibility(
                "forecast: not available for a multiplicative decomposition.",
                recovery_hint="Decompose the logarithm of the series with sp.stl.",
            )
        h = int(horizon)
        levels = normalise_levels(level)
        sa = self.seasonally_adjusted
        key = method.lower()
        if key == "ets":
            from ._ets import ets

            kw = dict(kwargs)
            kw.setdefault("model", "ZZN")
            fc = ets(sa, **kw).forecast(h, levels)
        elif key in ("naive", "drift"):
            from .simple_forecast import simple_forecast

            fc = simple_forecast(sa, key).forecast(h, levels)
        elif key == "arima":
            from .arima import arima

            kw = dict(kwargs)
            if "order" not in kw:
                kw.setdefault("auto", True)
            fc = arima(sa, **kw).forecast(h, level=levels)
        else:
            raise MethodIncompatibility(
                f"forecast: method={method!r} is not available.",
                recovery_hint="Use 'ets', 'naive', 'drift' or 'arima'.",
            )
        n = self.observed.shape[0]
        seas = np.zeros(h)
        for m, comp in self.seasonal_components.items():
            seas += comp[-m:][np.arange(h) % m]
        out = fc.copy()
        for col in out.columns:
            out[col] = out[col].to_numpy(dtype=float) + seas
        out.index = future_index(self._index, n, h)
        return out

    def summary(self) -> str:
        st = self.strength
        per = ", ".join(str(m) for m in self.periods)
        lines = [
            f"{self.method} decomposition (period {per})",
            "-" * 46,
            f"n                       : {self.observed.shape[0]}",
            f"strength of trend       : {st['trend']:.4f}",
            f"strength of seasonality : {st['seasonal']:.4f}",
        ]
        for k, v in st.items():
            if k.startswith("seasonal_"):
                lines.append(f"  period {k[9:]:<15s} : {v:.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def plot(self, figsize: Tuple[float, float] = (9, 7)) -> Any:
        """Stacked panels: the series and each component."""
        import matplotlib.pyplot as plt

        frame = self.to_frame().drop(columns="seasonally_adjusted")
        x = frame.index
        if isinstance(x, pd.PeriodIndex):
            x = x.to_timestamp()
        fig, axes = plt.subplots(frame.shape[1], 1, figsize=figsize, sharex=True)
        for ax, col in zip(np.atleast_1d(axes), frame.columns):
            ax.plot(x, frame[col].to_numpy(), color="black", linewidth=0.9)
            ax.set_ylabel(col)
        return fig


def _stl_once(
    x: np.ndarray,
    m: int,
    seasonal: Union[int, str],
    trend: Optional[int],
    low_pass: Optional[int],
    seasonal_deg: int,
    trend_deg: int,
    low_pass_deg: int,
    robust: bool,
    inner_iter: Optional[int],
    outer_iter: Optional[int],
    jumps: Tuple[Optional[int], Optional[int], Optional[int]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """One STL pass with R's rules for everything left unset."""
    from statsmodels.tsa.seasonal import STL

    n = x.shape[0]
    periodic = isinstance(seasonal, str)
    if periodic:
        if str(seasonal).lower() not in ("periodic", "per"):
            raise MethodIncompatibility(
                f"stl: seasonal={seasonal!r} is not a window.",
                recovery_hint="Pass an odd integer of at least 7, or 'periodic'.",
            )
        s_win = 10 * n + 1
        s_deg = 0
    else:
        s_win = int(seasonal)
        s_deg = int(seasonal_deg)
        if s_win < 3 or s_win % 2 == 0:
            raise MethodIncompatibility(
                f"stl: seasonal={seasonal!r} must be an odd integer of at least 3.",
                recovery_hint="E.g. seasonal=11, or seasonal='periodic'.",
            )
    t_win = (
        int(trend) if trend is not None else _next_odd(1.5 * m / (1.0 - 1.5 / s_win))
    )
    l_win = int(low_pass) if low_pass is not None else _next_odd(m)
    if l_win <= m:
        l_win = _next_odd(m + 1)
    if t_win % 2 == 0 or t_win < 3:
        raise MethodIncompatibility(
            f"stl: trend={trend!r} must be an odd integer of at least 3.",
            recovery_hint="Leave trend=None for the default window.",
        )
    s_jump = jumps[0] if jumps[0] is not None else int(math.ceil(s_win / 10))
    t_jump = jumps[1] if jumps[1] is not None else int(math.ceil(t_win / 10))
    l_jump = jumps[2] if jumps[2] is not None else int(math.ceil(l_win / 10))
    inner = inner_iter if inner_iter is not None else (1 if robust else 2)
    outer = outer_iter if outer_iter is not None else (15 if robust else 0)
    fit = STL(
        x,
        period=m,
        seasonal=s_win,
        trend=t_win,
        low_pass=l_win,
        seasonal_deg=s_deg,
        trend_deg=int(trend_deg),
        low_pass_deg=int(low_pass_deg),
        robust=bool(robust),
        seasonal_jump=int(s_jump),
        trend_jump=int(t_jump),
        low_pass_jump=int(l_jump),
    ).fit(inner_iter=int(inner), outer_iter=int(outer))
    seas = np.asarray(fit.seasonal, dtype=float)
    tr = np.asarray(fit.trend, dtype=float)
    if periodic:
        pos = np.arange(n) % m
        means = np.array([seas[pos == j].mean() for j in range(m)])
        seas = means[pos]
    settings = {
        "seasonal": "periodic" if periodic else s_win,
        "trend": t_win,
        "low_pass": l_win,
        "seasonal_deg": s_deg,
        "trend_deg": int(trend_deg),
        "low_pass_deg": int(low_pass_deg),
        "robust": bool(robust),
        "inner_iter": int(inner),
        "outer_iter": int(outer),
        "jumps": (int(s_jump), int(t_jump), int(l_jump)),
    }
    return seas, tr, settings


def stl(
    y: Any,
    period: Union[int, Sequence[int]],
    *,
    seasonal: Union[int, str, Sequence[int], None] = None,
    trend: Optional[int] = None,
    low_pass: Optional[int] = None,
    seasonal_deg: int = 0,
    trend_deg: int = 1,
    low_pass_deg: int = 1,
    robust: bool = False,
    inner_iter: Optional[int] = None,
    outer_iter: Optional[int] = None,
    seasonal_jump: Optional[int] = None,
    trend_jump: Optional[int] = None,
    low_pass_jump: Optional[int] = None,
    iterate: int = 2,
    data: Optional[pd.DataFrame] = None,
) -> DecompositionResult:
    """STL decomposition of a series into trend, seasonal and remainder.

    Parameters
    ----------
    y : array-like, pd.Series or str
        The series, in time order; a column name when ``data`` is given.
    period : int or sequence of int
        Seasonal period (12 for monthly data with a yearly pattern).
        Several periods, e.g. ``[48, 336]`` for half-hourly data with a
        daily and a weekly pattern, give an MSTL decomposition with one
        seasonal component each.
    seasonal : int, "periodic" or sequence of int, optional
        Seasonal smoothing window: an odd number of cycles, larger means
        a seasonal pattern that changes more slowly; ``"periodic"``
        forces an identical pattern in every cycle. Default 11 for one
        period and ``11, 15, 19, ...`` for several.
    trend : int, optional
        Trend smoothing window (odd). Default: the smallest odd integer
        of at least ``1.5 * period / (1 - 1.5 / seasonal)``.
    low_pass : int, optional
        Low-pass filter window; default the smallest odd integer above
        ``period``.
    seasonal_deg, trend_deg, low_pass_deg : {0, 1}
        Degree of the local polynomials; defaults 0, 1, 1.
    robust : bool, default False
        Down-weight outliers through robustness iterations, so that they
        end up in the remainder instead of distorting trend and season.
    inner_iter, outer_iter : int, optional
        Loess passes and robustness iterations. Defaults 2 and 0, or 1
        and 15 when ``robust``.
    seasonal_jump, trend_jump, low_pass_jump : int, optional
        Evaluate each loess smoother at every ``jump``-th point and
        interpolate in between. Default ``ceil(window / 10)``; pass 1
        for the exact smoother.
    iterate : int, default 2
        Passes over the seasonal components with several periods.
    data : pd.DataFrame, optional

    Returns
    -------
    DecompositionResult
        ``trend``, ``seasonal``, ``remainder``, ``seasonally_adjusted``,
        ``strength`` (of trend and seasonality), ``to_frame()``,
        ``forecast(horizon, method=...)`` and ``plot()``.

    Notes
    -----
    The defaults are those of R: ``stats::stl`` for everything but the
    seasonal window, which that function leaves to the user and which is
    set to 11 as in ``forecast::mstl`` and ``feasts::STL``. With the same
    arguments the components equal R's to rounding error.

    ``statsmodels.tsa.seasonal.STL`` runs the same algorithm with other
    defaults; ``sp.stl(y, period, seasonal=7, seasonal_deg=1,
    inner_iter=5, seasonal_jump=1, trend_jump=1, low_pass_jump=1)``
    reproduces ``STL(y, period=period).fit()``.

    STL is additive. For a series whose seasonal swings grow with its
    level, decompose its logarithm (or a Box-Cox transform, see
    :func:`statspai.boxcox_lambda`).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> idx = pd.period_range("2010-01", periods=144, freq="M")
    >>> t = np.arange(144)
    >>> rng = np.random.default_rng(0)
    >>> y = pd.Series(
    ...     50 + 0.3 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 144),
    ...     index=idx,
    ... )
    >>> dec = sp.stl(y, period=12, robust=True)
    >>> bool(np.allclose(dec.trend + dec.seasonal + dec.remainder, y))
    True
    >>> dec.forecast(12, method="naive").shape
    (12, 5)

    References
    ----------
    cleveland1990stl, bandara2025mstl, wang2006characteristic,
    hyndman2026fpppy
    """
    values, index, name = read_series(y, data, fn="stl")
    if np.isscalar(period):
        periods: List[int] = [check_period(period, fn="stl")]
    else:
        several: Sequence[int] = period  # type: ignore[assignment]
        periods = sorted({check_period(p, fn="stl") for p in several})
    n = values.shape[0]
    for m in periods:
        if m < 2:
            raise MethodIncompatibility(
                "stl: the seasonal period must be at least 2.",
                recovery_hint="Non-seasonal data have no seasonal component to remove.",
            )
        if n < 2 * m:
            raise DataInsufficient(
                f"stl: {n} observations are fewer than two cycles of period {m}.",
                recovery_hint="At least two full cycles are needed.",
                diagnostics={"n": n, "period": m},
            )
    if seasonal is None:
        windows: List[Union[int, str]] = [11 + 4 * i for i in range(len(periods))]
    elif isinstance(seasonal, (str, int, np.integer)):
        windows = [seasonal] * len(periods)  # type: ignore[list-item]
    else:
        windows = list(seasonal)
        if len(windows) != len(periods):
            raise MethodIncompatibility(
                f"stl: {len(windows)} seasonal windows for {len(periods)} periods.",
                recovery_hint="Give one window per period, or a single one.",
            )
    jumps = (seasonal_jump, trend_jump, low_pass_jump)
    comps: Dict[int, np.ndarray] = {}
    settings: Dict[str, Any] = {}
    tr = np.zeros(n)
    if len(periods) == 1:
        seas, tr, settings = _stl_once(
            values,
            periods[0],
            windows[0],
            trend,
            low_pass,
            seasonal_deg,
            trend_deg,
            low_pass_deg,
            robust,
            inner_iter,
            outer_iter,
            jumps,
        )
        comps[periods[0]] = seas
        method = "STL"
    else:
        deseas = values.copy()
        for m in periods:
            comps[m] = np.zeros(n)
        for _ in range(max(int(iterate), 1)):
            for m, win in zip(periods, windows):
                deseas = deseas + comps[m]
                seas, tr, settings = _stl_once(
                    deseas,
                    m,
                    win,
                    trend,
                    low_pass,
                    seasonal_deg,
                    trend_deg,
                    low_pass_deg,
                    robust,
                    inner_iter,
                    outer_iter,
                    jumps,
                )
                comps[m] = seas
                deseas = deseas - seas
        settings = dict(settings, seasonal=list(windows), iterate=int(iterate))
        method = "MSTL"
    total = np.sum(list(comps.values()), axis=0)
    return DecompositionResult(
        observed=values,
        trend=tr,
        seasonal=total,
        remainder=values - tr - total,
        periods=tuple(periods),
        method=method,
        multiplicative=False,
        seasonal_components=comps,
        settings=settings,
        _index=index,
        _name=name,
    )


def classical_decompose(
    y: Any,
    period: int,
    *,
    model: str = "additive",
    data: Optional[pd.DataFrame] = None,
) -> DecompositionResult:
    """Classical decomposition by moving averages.

    The trend is a centred moving average of one full cycle (a
    ``2 x m`` average when the period is even), the seasonal figure the
    average detrended value at each position of the cycle. Kept as the
    textbook baseline; :func:`statspai.stl` is the better tool: it has
    no missing ends, lets the seasonal pattern change and can be made
    robust to outliers.

    Parameters
    ----------
    y : array-like, pd.Series or str
    period : int
    model : {"additive", "multiplicative"}, default "additive"
        ``observed = trend + seasonal + remainder`` or their product.
    data : pd.DataFrame, optional

    Returns
    -------
    DecompositionResult
        ``trend`` and ``remainder`` are ``nan`` for the first and last
        half cycle, where the moving average is not defined.

    Notes
    -----
    Equal to R's ``stats::decompose`` and to
    ``statsmodels.tsa.seasonal.seasonal_decompose`` with its default
    two-sided filter.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> y = np.tile([8.0, 12.0, 10.0, 14.0], 6) * (1 + 0.02 * np.arange(24))
    >>> dec = sp.classical_decompose(y, period=4, model="multiplicative")
    >>> round(float(dec.seasonal[:4].mean()), 6)
    1.0

    References
    ----------
    hyndman2026fpppy
    """
    values, index, name = read_series(y, data, fn="classical_decompose")
    m = check_period(period, fn="classical_decompose")
    key = model.lower()
    if key not in ("additive", "multiplicative"):
        raise MethodIncompatibility(
            f"classical_decompose: model={model!r} is not 'additive' or "
            "'multiplicative'.",
            recovery_hint="Use model='additive'.",
        )
    n = values.shape[0]
    if m < 2 or n < 2 * m:
        raise DataInsufficient(
            f"classical_decompose: {n} observations are fewer than two cycles "
            f"of period {m}.",
            recovery_hint="At least two full cycles and period >= 2 are needed.",
        )
    mult = key == "multiplicative"
    if mult and values.min() <= 0:
        raise MethodIncompatibility(
            "classical_decompose: a multiplicative decomposition needs a "
            "strictly positive series.",
            recovery_hint="Use model='additive'.",
        )
    tr, seas, rem = classical_decomposition(values, m, multiplicative=mult)
    return DecompositionResult(
        observed=values,
        trend=tr,
        seasonal=seas,
        remainder=rem,
        periods=(m,),
        method="Classical " + key,
        multiplicative=mult,
        seasonal_components={m: seas},
        settings={"model": key},
        _index=index,
        _name=name,
    )
