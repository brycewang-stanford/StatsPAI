"""Trend-cycle decomposition of one series by a linear filter:
Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald, Butterworth and
Hamilton's regression filter.

Every filter writes the series as ``trend + cycle``. The cycle is what the
filter extracts; the trend is the series minus the cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg, special

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["tsfilter", "hp_smoothing", "FilterResult"]

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float]]

_METHODS = ("hp", "bk", "cf", "bw", "hamilton")
_FREQUENCIES = {"annual": 1.0, "quarterly": 4.0, "monthly": 12.0}


def hp_smoothing(observations_per_year: Union[float, str]) -> float:
    """Hodrick-Prescott smoothing parameter by the Ravn-Uhlig rule.

    Parameters
    ----------
    observations_per_year : float or {'annual', 'quarterly', 'monthly'}
        Sampling frequency of the series.

    Returns
    -------
    float
        ``1600 * (observations_per_year / 4) ** 4``: 6.25 for annual, 1600
        for quarterly and 129600 for monthly data.

    Raises
    ------
    MethodIncompatibility
        An unknown name or a non-positive number.

    Examples
    --------
    >>> from statspai.timeseries.tsfilter import hp_smoothing
    >>> hp_smoothing("annual"), hp_smoothing(12)
    (6.25, 129600.0)

    ``sp.tsfilter(..., smooth="annual")`` applies the same rule.
    """
    if isinstance(observations_per_year, str):
        if observations_per_year not in _FREQUENCIES:
            raise MethodIncompatibility(
                f"tsfilter: {observations_per_year!r} is not one of "
                f"{sorted(_FREQUENCIES)}.",
                recovery_hint="Pass the number of observations per year.",
            )
        observations_per_year = _FREQUENCIES[observations_per_year]
    if not observations_per_year > 0:
        raise MethodIncompatibility(
            f"tsfilter: {observations_per_year!r} observations per year.",
            recovery_hint="Pass a positive number, e.g. 4 for quarterly data.",
        )
    return float(1600.0 * (float(observations_per_year) / 4.0) ** 4)


def _read(data: ArrayLike, y: Optional[str]) -> Tuple[np.ndarray, pd.Index, str]:
    if isinstance(data, pd.DataFrame):
        if y is None or y not in data.columns:
            raise MethodIncompatibility(
                f"sp.tsfilter: y={y!r} is not a column of the data.",
                recovery_hint="Pass the name of the series, e.g. "
                "sp.tsfilter(df, 'gdp').",
            )
        return data[y].to_numpy(dtype=float, na_value=np.nan), data.index, str(y)
    if isinstance(data, pd.Series):
        name = "y" if data.name is None else str(data.name)
        return data.to_numpy(dtype=float, na_value=np.nan), data.index, name
    x = np.asarray(data, dtype=float).ravel()
    return x, pd.RangeIndex(x.size), "y"


def _ideal_weights(low: float, high: float, count: int) -> np.ndarray:
    """Weights ``b_0..b_count`` of the ideal band-pass filter that keeps
    cycles lasting between ``low`` and ``high`` observations."""
    a, b = 2.0 * np.pi / high, 2.0 * np.pi / low
    j = np.arange(1, count + 1)
    return np.concatenate(
        ([(b - a) / np.pi], (np.sin(j * b) - np.sin(j * a)) / (np.pi * j))
    )


def _bk_weights(low: float, high: float, K: int, stationary: bool) -> np.ndarray:
    b = _ideal_weights(low, high, K)
    if not stationary:
        b = b - (b[0] + 2.0 * b[1:].sum()) / (2 * K + 1)
    return b


def _difference_filter(
    x: np.ndarray, order: int, smooth: float, butterworth: bool
) -> np.ndarray:
    """Cycle ``smooth * Q (W + smooth * Q'Q)^{-1} Q' x``.

    ``Q'`` takes ``order``-th differences. ``W`` is the identity for the
    Hodrick-Prescott filter and the matching band matrix of ``order``-fold
    two-term sums for the Butterworth filter. Both are band matrices, so
    the system is solved in O(n).
    """
    n = x.size
    k = np.arange(order + 1)
    diff = (-1.0) ** k * special.comb(order, k)
    high = np.convolve(diff, diff[::-1])[order:]
    if butterworth:
        add = special.comb(order, k)
        low = np.convolve(add, add[::-1])[order:]
    else:
        low = np.zeros(order + 1)
        low[0] = 1.0
    band = np.empty((order + 1, n - order))
    for j in range(order + 1):
        band[order - j, :] = low[j] + smooth * high[j]
    dx = np.convolve(x, diff[::-1], mode="valid")
    z = linalg.solveh_banded(band, dx)
    return np.asarray(smooth * np.convolve(z, diff), dtype=float)


def _hp_one_sided(x: np.ndarray, smooth: float) -> np.ndarray:
    """Cycle of the one-sided Hodrick-Prescott filter.

    The two-sided trend is the smoothed state of ``y = trend + e``,
    ``second difference of trend = u`` with ``var(u) / var(e) = 1 /
    smooth`` and a diffuse start, so its last point on the data up to
    ``t`` is the filtered state. The first two observations identify the
    two diffuse components exactly: state ``(trend_2, trend_1)`` has mean
    ``(y_2, y_1)`` and identity covariance, from which the Kalman
    recursion runs without any large-number approximation.
    """
    n = x.size
    trend = x.copy()
    a = np.array([x[1], x[0]])
    P = np.eye(2)
    T = np.array([[2.0, -1.0], [1.0, 0.0]])
    q = 1.0 / smooth
    for t in range(2, n):
        a = T @ a
        P = T @ P @ T.T
        P[0, 0] += q
        gain = P[:, 0] / (P[0, 0] + 1.0)
        a = a + gain * (x[t] - a[0])
        P = P - np.outer(gain, P[0, :])
        P = 0.5 * (P + P.T)
        trend[t] = a[0]
    return np.asarray(x - trend, dtype=float)


def _cf_sma_weights(low: float, high: float, q: int, stationary: bool) -> np.ndarray:
    """Weights ``b_0..b_q`` of the fixed-length symmetric
    Christiano-Fitzgerald filter: ideal up to ``q - 1``; the outermost is
    the ideal one for a stationary series and otherwise the value that
    makes the ``2 q + 1`` weights sum to zero."""
    b = _ideal_weights(low, high, q)
    if not stationary:
        b[q] = -(0.5 * b[0] + b[1:q].sum())
    return b


def _cf_cycle(
    x: np.ndarray, low: float, high: float, drift: bool, stationary: bool = False
) -> np.ndarray:
    n = x.size
    if drift:
        x = x - np.arange(n) * (x[-1] - x[0]) / (n - 1)
    b = _ideal_weights(low, high, n - 1)
    idx = np.arange(n)
    w = b[np.abs(idx[:, None] - idx[None, :])]
    if stationary:
        # the ideal filter cut off at the ends of the sample
        return np.asarray(w @ x, dtype=float)
    w[: n - 1, n - 1] = 0.0
    w[1:, 0] = 0.0
    # the weights of the two end observations stand for all the unobserved
    # ones beyond them, which a random walk predicts by the end value
    partial = np.concatenate(([0.0], np.cumsum(b[1:])))
    w[:, n - 1] += -(0.5 * b[0] + partial[np.maximum(n - 2 - idx, 0)])
    w[:, 0] += -(0.5 * b[0] + partial[np.maximum(idx - 1, 0)])
    return np.asarray(w @ x, dtype=float)


@dataclass
class FilterResult(ResultProtocolMixin):
    """Trend and cycle returned by :func:`statspai.tsfilter`.

    Attributes
    ----------
    observed, trend, cycle : pandas.Series
        On the index of the input. ``trend`` and ``cycle`` are NaN where
        the filter is not defined (see :func:`statspai.tsfilter`).
    method : str
    params : dict
        The settings that define the filter. For ``'hamilton'`` also the
        regression coefficients, constant first.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> y = np.cumsum(np.random.default_rng(0).normal(0.5, 1, 120))
    >>> res = sp.tsfilter(y, method="hp")
    >>> bool(np.allclose(res.trend + res.cycle, res.observed))
    True
    """

    observed: pd.Series
    trend: pd.Series
    cycle: pd.Series
    method: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Observed series, trend and cycle as columns."""
        return pd.DataFrame(
            {"observed": self.observed, "trend": self.trend, "cycle": self.cycle}
        )

    def gain(self, omega: Union[Sequence[float], np.ndarray, None] = None) -> pd.Series:
        """Gain of the cycle filter: by how much it scales a wave of
        frequency ``omega``.

        Parameters
        ----------
        omega : array-like, optional
            Frequencies in radians per observation, between 0 and pi.
            Default: 201 equally spaced points on that interval.

        Returns
        -------
        pandas.Series
            Gain, indexed by ``omega``. The squared gain multiplies the
            spectral density of the input.

        Raises
        ------
        MethodIncompatibility
            For ``method='cf'`` without ``sma_order``, whose weights
            differ at every date, so that no single gain describes it,
            and for the one-sided Hodrick-Prescott filter.

        Notes
        -----
        ``'hp'``: ``4 s (1 - cos w)**2 / (1 + 4 s (1 - cos w)**2)`` with
        ``s = smooth``. ``'bw'``: ``s (1 - cos w)**m / ((1 + cos w)**m +
        s (1 - cos w)**m)`` with ``m = order`` and
        ``s = tan(pi / high)**(-2 m)``. Both describe the filter away
        from the ends of the sample. ``'bk'`` and ``'cf'`` with
        ``sma_order``: ``|b_0 + 2 sum_j b_j cos(j w)|`` from the weights
        used, exact at every date where the cycle is defined.
        ``'hamilton'``:
        ``|1 - sum_j beta_j exp(-i w (h + j))|`` from the estimated
        coefficients.
        """
        w = (
            np.linspace(0.0, np.pi, 201)
            if omega is None
            else np.atleast_1d(np.asarray(omega, dtype=float))
        )
        p = self.params
        if self.method == "hp" and p.get("one_sided"):
            raise MethodIncompatibility(
                "FilterResult.gain: the one-sided Hodrick-Prescott filter "
                "shifts the phase and its weights change over the sample.",
                recovery_hint="Use one_sided=False for the gain of the "
                "two-sided filter.",
            )
        if self.method == "hp":
            a = 4.0 * p["smooth"] * (1.0 - np.cos(w)) ** 2
            g = a / (1.0 + a)
        elif self.method == "bw":
            m = p["order"]
            up = p["smooth"] * (1.0 - np.cos(w)) ** m
            g = up / ((1.0 + np.cos(w)) ** m + up)
        elif self.method == "bk" or "weights" in p:
            b = np.asarray(p["weights"])
            j = np.arange(1, b.size)
            g = np.abs(b[0] + 2.0 * np.cos(np.outer(w, j)) @ b[1:])
        elif self.method == "hamilton":
            beta = np.asarray(p["coef"])[1:]
            lags = p["h"] + np.arange(beta.size)
            g = np.abs(1.0 - np.exp(-1j * np.outer(w, lags)) @ beta)
        else:
            raise MethodIncompatibility(
                "FilterResult.gain: the Christiano-Fitzgerald filter uses "
                "different weights at every date and has no single gain.",
                recovery_hint="Its target is the ideal band-pass filter: gain "
                "1 for 2*pi/high <= omega <= 2*pi/low, 0 elsewhere. With "
                "sma_order= the weights are fixed and the gain is defined.",
            )
        return pd.Series(np.asarray(g, dtype=float), index=pd.Index(w, name="omega"))

    def summary(self) -> str:
        """Plain-text description of the decomposition."""
        names = {
            "hp": "Hodrick-Prescott filter",
            "bk": "Baxter-King band-pass filter",
            "cf": "Christiano-Fitzgerald band-pass filter",
            "bw": "Butterworth high-pass filter",
            "hamilton": "Hamilton regression filter",
        }
        c = self.cycle.to_numpy(dtype=float)
        c = c[~np.isnan(c)]
        shown = {k: v for k, v in self.params.items() if k not in ("weights", "coef")}
        lines = [
            names[self.method],
            "  " + ", ".join(f"{k}={v}" for k, v in shown.items()),
            f"  observations            {int(self.observed.notna().sum())}",
            f"  cycle defined at        {c.size}",
            f"  cycle std. deviation    {float(np.std(c, ddof=1)):.6g}",
        ]
        if c.size > 2:
            d = c - c.mean()
            lines.append(
                f"  cycle autocorrelation   {float(d[1:] @ d[:-1] / (d @ d)):.4f}"
            )
        return "\n".join(lines)

    def plot(self, axes: Any = None) -> Any:
        """Series with its trend (top) and the cycle (bottom)."""
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(2, 1, sharex=True)
        x = self.observed.index
        axes[0].plot(x, self.observed.to_numpy(), color="0.5", lw=1.0, label="series")
        axes[0].plot(x, self.trend.to_numpy(), color="C0", lw=1.4, label="trend")
        axes[0].legend(frameon=False)
        axes[1].plot(x, self.cycle.to_numpy(), color="C0", lw=1.2)
        axes[1].axhline(0.0, color="0.4", lw=0.6)
        axes[1].set_ylabel("cycle")
        return axes


def tsfilter(
    data: ArrayLike,
    y: Optional[str] = None,
    *,
    method: str = "hp",
    smooth: Union[float, str] = 1600.0,
    low: float = 6.0,
    high: float = 32.0,
    K: int = 12,
    drift: bool = True,
    stationary: bool = False,
    order: int = 2,
    h: int = 8,
    p: int = 4,
    sma_order: Optional[int] = None,
    one_sided: bool = False,
) -> FilterResult:
    """Split a series into trend and cycle with a linear filter.

    The defaults of every filter are those for quarterly data.

    Parameters
    ----------
    data : DataFrame, Series or array
        The series, in time order. With a DataFrame, ``y`` names the column.
    y : str, optional
        Column of ``data``.
    method : {'hp', 'bk', 'cf', 'bw', 'hamilton'}, default 'hp'
        Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald, Butterworth
        (high-pass) or Hamilton's regression filter.
    smooth : float or {'annual', 'quarterly', 'monthly'}, default 1600
        Hodrick-Prescott penalty on changes in the slope of the trend. A
        name applies the Ravn-Uhlig rule (6.25, 1600, 129600).
    low, high : float, default 6 and 32
        Shortest and longest cycle kept by the band-pass filters, in
        observations (``'bk'``, ``'cf'``). ``'bw'`` uses ``high`` only: it
        removes cycles longer than ``high``.
    K : int, default 12
        Leads and lags of the Baxter-King moving average.
    drift : bool, default True
        ``'cf'``: remove the line through the first and the last
        observation before filtering, as Christiano and Fitzgerald
        recommend for a random walk with drift. Stata's ``tsfilter cf``
        does this only with its ``drift`` option.
    stationary : bool, default False
        ``'bk'``: use the truncated ideal weights as they are. By default
        a constant is subtracted so that they sum to zero, which removes a
        unit root; without it the "cycle" keeps part of the level.
        ``'cf'``: use the ideal weights at every observation of the
        sample, the best filter for a serially uncorrelated series. By
        default the weights of the first and the last observation (with
        ``sma_order``: of the two outermost ones) are replaced so that
        they sum to zero. The same caveat applies: these weights do not
        sum to zero, so subtract the mean of the series first, and set
        ``drift=False`` unless a line is to be removed as well.
    order : int, default 2
        Order of the Butterworth filter; higher is closer to a sharp
        cut-off.
    h, p : int, default 8 and 4
        Hamilton: horizon and number of lags. The cycle at ``t`` is the
        residual of a least-squares regression of ``y[t]`` on a constant
        and ``y[t-h], ..., y[t-h-p+1]``.
    sma_order : int, optional
        ``'cf'``: use the fixed-length symmetric Christiano-Fitzgerald
        filter, a moving average of ``2 * sma_order + 1`` terms with the
        same weights at every date. It must be below ``(T - 1) / 2``.
        ``drift`` still removes the line through the first and the last
        observation of the whole series.
    one_sided : bool, default False
        ``'hp'``: the trend at ``t`` is the end point of the
        Hodrick-Prescott trend of the observations up to ``t``, so that
        no later observation enters and no value is ever revised.

    Returns
    -------
    FilterResult
        ``.trend`` and ``.cycle`` on the index of the input,
        ``.gain(omega)``, ``.summary()``, ``.plot()``, ``.to_frame()``.

    Raises
    ------
    MethodIncompatibility
        Unknown method, parameters out of range, a gap in the series.
    DataInsufficient
        A series too short for the filter.

    Notes
    -----
    Missing values at the start or the end are set aside and come back as
    NaN. In addition the cycle is undefined, by construction and not for
    lack of data, at the first and last ``K`` observations for ``'bk'``
    (a two-sided moving average of ``2 K + 1`` terms), the first and last
    ``sma_order`` for ``'cf'`` with ``sma_order``, and at the first
    ``h + p - 1`` for ``'hamilton'``.

    ``'hp'`` minimises ``sum (y - trend)**2 + smooth * sum
    (second difference of trend)**2`` in the sample at hand. ``'bw'`` is
    the finite-sample high-pass Butterworth filter with cut-off
    ``2 pi / high``. ``'cf'`` is the random-walk filter with weights that
    change with the distance to either end of the sample, so that every
    date gets a value; estimates near the ends are revised as data arrive.
    The same is true of ``'hp'`` and ``'bw'``.

    With ``one_sided=True`` the Hodrick-Prescott trend is computed by the
    Kalman filter of the model whose smoother is the two-sided filter
    (the series is the trend plus noise, the second difference of the
    trend is noise with ``1 / smooth`` times the variance), started from
    the exact diffuse distribution. With one or two observations the
    penalty is empty and the trend is the data, so the first two cycle
    values are zero. The one-sided cycle lags the two-sided one and has a
    different gain; the two are not interchangeable.

    ``'hp'``, ``'bk'``, ``'cf'`` and ``'bw'`` reproduce Stata's
    ``tsfilter`` (``drift=False`` for its default ``cf``; ``sma_order``
    is its ``smaorder()``); statsmodels agrees on the cycles, and reports
    as the trend of ``cffilter`` the series less the drift line less the
    cycle. The exception is ``'cf'`` with ``stationary=True``: the Stata
    manual sets every weight to the ideal one, which is what is computed
    here, but Stata 18 puts the ideal weight of the next smaller lag on
    the first and the last observation of the sample (with
    ``smaorder(q)``: ``b_{q-1}`` in place of ``b_q`` on the two outermost
    terms) and keeps the sum-to-zero weights at the two end dates, so its
    numbers differ from these.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(160)
    >>> y = 0.5 * t + 3 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.3, 160)
    >>> res = sp.tsfilter(y, method="bk", low=6, high=32, K=12)
    >>> int(res.cycle.isna().sum())
    24
    >>> bool(res.cycle.std() > 1.5)
    True
    >>> hp = sp.tsfilter(y, method="hp", smooth="quarterly")
    >>> round(float(hp.gain([2 * np.pi / 20]).iloc[0]), 3)
    0.939

    References
    ----------
    [@hodrick1997postwar],
    [@ravn2002adjusting],
    [@baxter1999measuring],
    [@christiano2003band],
    [@pollock2000trend],
    [@hamilton2018why]
    """
    if method not in _METHODS:
        raise MethodIncompatibility(
            f"sp.tsfilter: method={method!r} is not one of {_METHODS}.",
            recovery_hint="Use 'hp', 'bk', 'cf', 'bw' or 'hamilton'.",
        )
    raw, index, name = _read(data, y)
    seen = np.flatnonzero(~np.isnan(raw))
    if seen.size == 0:
        raise DataInsufficient(
            "sp.tsfilter: the series has no observed value.",
            recovery_hint="Check the column.",
        )
    first, last = int(seen[0]), int(seen[-1]) + 1
    x = raw[first:last]
    if not np.isfinite(x).all():
        raise MethodIncompatibility(
            "sp.tsfilter: the series has missing or infinite values between "
            "observed ones; the filters need consecutive periods.",
            recovery_hint="Fill the gap or filter the longest complete stretch.",
        )
    n = x.size
    params: Dict[str, Any]
    if one_sided and method != "hp":
        raise MethodIncompatibility(
            f"sp.tsfilter: one_sided=True is defined for method='hp', not "
            f"{method!r}.",
            recovery_hint="method='hamilton' is one-sided by construction.",
        )
    if sma_order is not None and method != "cf":
        raise MethodIncompatibility(
            f"sp.tsfilter: sma_order= belongs to method='cf', not {method!r}.",
            recovery_hint="The Baxter-King moving average takes K=.",
        )
    if method in ("bk", "cf", "bw"):
        if not high > 2.0 or (method != "bw" and not 2.0 <= low < high):
            raise MethodIncompatibility(
                f"sp.tsfilter: low={low}, high={high} is not a band of cycle "
                "lengths with 2 <= low < high.",
                recovery_hint="Lengths are in observations, e.g. low=6, high=32 "
                "for quarterly business cycles.",
            )

    def short(need: int) -> DataInsufficient:
        return DataInsufficient(
            f"sp.tsfilter: method={method!r} needs at least {need} "
            f"observations; the series has {n}.",
            recovery_hint="Use a longer series or a shorter filter.",
        )

    if method == "hp":
        lam = hp_smoothing(smooth) if isinstance(smooth, str) else float(smooth)
        if not lam > 0 or not np.isfinite(lam):
            raise MethodIncompatibility(
                f"sp.tsfilter: smooth={smooth!r} is not a positive number.",
                recovery_hint="Use 1600 for quarterly data.",
            )
        if n < 4:
            raise short(4)
        if one_sided:
            cycle = _hp_one_sided(x, lam)
        else:
            cycle = _difference_filter(x, 2, lam, butterworth=False)
        params = {"smooth": lam, "one_sided": bool(one_sided)}
    elif method == "bw":
        if int(order) != order or order < 1:
            raise MethodIncompatibility(
                f"sp.tsfilter: order={order!r} is not a positive integer.",
                recovery_hint="Use order=2.",
            )
        m = int(order)
        if n < 2 * m + 2:
            raise short(2 * m + 2)
        lam = float(np.tan(np.pi / high) ** (-2 * m))
        cycle = _difference_filter(x, m, lam, butterworth=True)
        params = {"high": float(high), "order": m, "smooth": lam}
    elif method == "bk":
        if int(K) != K or K < 1:
            raise MethodIncompatibility(
                f"sp.tsfilter: K={K!r} is not a positive integer.",
                recovery_hint="Use K=12 for quarterly data.",
            )
        K = int(K)
        if n < 2 * K + 1:
            raise short(2 * K + 1)
        b = _bk_weights(low, high, K, bool(stationary))
        cycle = np.full(n, np.nan)
        cycle[K : n - K] = np.convolve(x, np.concatenate((b[:0:-1], b)), "valid")
        params = {
            "low": float(low),
            "high": float(high),
            "K": K,
            "stationary": bool(stationary),
            "weights": b,
        }
    elif method == "cf":
        if n < 3:
            raise short(3)
        params = {
            "low": float(low),
            "high": float(high),
            "drift": bool(drift),
            "stationary": bool(stationary),
        }
        if sma_order is None:
            cycle = _cf_cycle(x, low, high, bool(drift), bool(stationary))
        else:
            if int(sma_order) != sma_order or sma_order < 1:
                raise MethodIncompatibility(
                    f"sp.tsfilter: sma_order={sma_order!r} is not a positive "
                    "integer.",
                    recovery_hint="Use sma_order=12 for quarterly data, or "
                    "leave it out for the full-sample filter.",
                )
            q = int(sma_order)
            if n <= 2 * q + 1:
                raise short(2 * q + 2)
            z = x - np.arange(n) * (x[-1] - x[0]) / (n - 1) if drift else x
            b = _cf_sma_weights(low, high, q, bool(stationary))
            cycle = np.full(n, np.nan)
            cycle[q : n - q] = np.convolve(z, np.concatenate((b[:0:-1], b)), "valid")
            params.update({"sma_order": q, "weights": b})
    else:
        if int(h) != h or int(p) != p or h < 1 or p < 1:
            raise MethodIncompatibility(
                f"sp.tsfilter: h={h!r}, p={p!r} are not positive integers.",
                recovery_hint="Use h=8, p=4 for quarterly data.",
            )
        h, p = int(h), int(p)
        start = h + p - 1
        if n - start < p + 2:
            raise short(start + p + 2)
        design = np.column_stack(
            [np.ones(n - start)] + [x[start - h - j : n - h - j] for j in range(p)]
        )
        coef, _, rank, _ = np.linalg.lstsq(design, x[start:], rcond=None)
        if rank < p + 1:
            raise DataInsufficient(
                "sp.tsfilter: the lags of the series are collinear; the "
                "Hamilton regression is not identified.",
                recovery_hint="Use fewer lags (p=) or a longer series.",
            )
        cycle = np.full(n, np.nan)
        cycle[start:] = x[start:] - design @ coef
        params = {"h": h, "p": p, "coef": coef}

    full_cycle = np.full(raw.size, np.nan)
    full_cycle[first:last] = cycle
    return FilterResult(
        observed=pd.Series(raw, index=index, name=name),
        trend=pd.Series(raw - full_cycle, index=index, name="trend"),
        cycle=pd.Series(full_cycle, index=index, name="cycle"),
        method=method,
        params=params,
    )
