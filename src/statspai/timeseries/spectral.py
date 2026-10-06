"""Spectral density of one series: periodogram, kernel-smoothed periodogram
and autoregressive spectrum, and Bartlett's cumulative-periodogram test for
white noise.

Two normalisations are in use and differ by ``2 * pi``. ``scale='cycles'``
is a density per unit of frequency measured in cycles per observation: its
integral over ``(-1/2, 1/2]`` is the variance (R ``spec.pgram``,
``spec.ar``). ``scale='radians'`` is a density per radian, whose integral
over ``(-pi, pi]`` is the variance: the textbook
``I(w) = |sum_t x_t exp(-i w t)|**2 / (2 pi T)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import special, stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = [
    "periodogram",
    "cumulative_periodogram_test",
    "SpectrumResult",
    "CumulativePeriodogramResult",
]

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float]]


def _series(data: ArrayLike, y: Optional[str], who: str) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        if y is None or y not in data.columns:
            raise MethodIncompatibility(
                f"sp.{who}: y={y!r} is not a column of the data.",
                recovery_hint=f"Pass the name of the series, e.g. sp.{who}(df, 'x').",
            )
        x = data[y].to_numpy(dtype=float, na_value=np.nan)
    else:
        x = np.asarray(data, dtype=float).ravel()
    observed = np.flatnonzero(~np.isnan(x))
    if observed.size == 0:
        raise DataInsufficient(
            f"sp.{who}: the series has no observed value.",
            recovery_hint="Check the column.",
        )
    # missing values at either end only shorten the sample (growth rates
    # and differences start late); one in the middle is a gap
    x = x[observed[0] : observed[-1] + 1]
    if not np.isfinite(x).all():
        raise MethodIncompatibility(
            f"sp.{who}: the series has missing or infinite values between "
            "observed ones; a spectrum needs consecutive periods.",
            recovery_hint="Fill the gap or analyse the longest complete stretch.",
        )
    return np.asarray(x, dtype=float)


def _next_fast_length(n: int) -> int:
    """Smallest integer >= n whose only prime factors are 2, 3 and 5."""
    m = n
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def _daniell_kernel(spans: Sequence[int]) -> np.ndarray:
    """Weights at offsets ``-m..m`` of (convolved) modified Daniell kernels.

    A modified Daniell kernel of half-width ``m`` puts ``1 / (2m)`` on the
    offsets inside ``(-m, m)`` and half of that on ``-m`` and ``m``. A span
    ``s`` means ``m = s // 2``.
    """
    kernel = np.array([1.0])
    for s in spans:
        m = int(s) // 2
        if m == 0:
            continue
        w = np.full(2 * m + 1, 1.0 / (2 * m))
        w[0] = w[-1] = 1.0 / (4 * m)
        kernel = np.convolve(kernel, w)
    return kernel


def _circular_smooth(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    m = (kernel.size - 1) // 2
    out = np.zeros_like(values)
    for j in range(-m, m + 1):
        out += kernel[j + m] * np.roll(values, -j)
    return out


def _yule_walker(
    x: np.ndarray, max_order: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Levinson-Durbin on the divisor-``n`` autocovariances.

    Returns the coefficient table (row ``k`` holds the AR(``k``) fit), the
    innovation variance of each order and the autocovariances.
    """
    n = x.size
    xc = x - x.mean()
    acov = np.array([xc[: n - k] @ xc[k:] / n for k in range(max_order + 1)])
    phi = np.zeros((max_order + 1, max_order + 1))
    v = np.zeros(max_order + 1)
    v[0] = acov[0]
    for k in range(1, max_order + 1):
        prev = phi[k - 1, 1:k]
        kappa = (acov[k] - prev @ acov[k - 1 : 0 : -1]) / v[k - 1]
        phi[k, 1:k] = prev - kappa * prev[::-1]
        phi[k, k] = kappa
        v[k] = v[k - 1] * (1.0 - kappa**2)
    return phi, v, acov


@dataclass
class SpectrumResult(ResultProtocolMixin):
    """Estimated spectral density returned by :func:`statspai.periodogram`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per frequency: ``freq`` (cycles per observation), ``omega``
        (radians per observation, ``2 pi freq``), ``cycle_length``
        (observations per cycle, ``1 / freq``), ``spectrum`` and, for the
        periodogram estimates, ``lower`` and ``upper``.
    method : {'raw', 'smoothed', 'ar'}
    scale : {'cycles', 'radians'}
    df : float or None
        Equivalent degrees of freedom of each ordinate (``None`` for the
        autoregressive spectrum).
    bandwidth : float or None
        Equivalent bandwidth of the smoothing kernel, in cycles per
        observation (``None`` for the autoregressive spectrum).
    n : int
        Observations used.
    order : int or None
        Autoregressive order (``method='ar'``).
    ar_coef : np.ndarray or None
        Yule-Walker coefficients (``method='ar'``).
    var_pred : float or None
        Innovation variance (``method='ar'``).
    settings : dict

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> res = sp.periodogram(rng.normal(size=120), method="smoothed", spans=5)
    >>> list(res.table.columns)
    ['freq', 'omega', 'cycle_length', 'spectrum', 'lower', 'upper']
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("brockwell1991time",)

    table: pd.DataFrame
    method: str
    scale: str
    df: Optional[float]
    bandwidth: Optional[float]
    n: int
    order: Optional[int] = None
    ar_coef: Optional[np.ndarray] = None
    var_pred: Optional[float] = None
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def peak(self) -> Dict[str, float]:
        """Frequency at which the estimate is largest."""
        row = self.table.loc[self.table["spectrum"].idxmax()]
        return {k: float(row[k]) for k in ("freq", "cycle_length", "spectrum")}

    def summary(self) -> str:
        """Plain-text description of the estimate."""
        names = {
            "raw": "Periodogram",
            "smoothed": "Smoothed periodogram (modified Daniell)",
            "ar": f"AR({self.order}) spectrum (Yule-Walker)",
        }
        lines = [
            names[self.method],
            f"  observations            {self.n}",
            f"  frequencies             {len(self.table)}",
            f"  density per             "
            f"{'cycle' if self.scale == 'cycles' else 'radian'}",
        ]
        if self.df is not None and self.bandwidth is not None:
            lines.append(f"  equivalent df           {self.df:.4f}")
            lines.append(f"  bandwidth (cycles)      {self.bandwidth:.6g}")
        if self.var_pred is not None:
            lines.append(f"  innovation variance     {self.var_pred:.6g}")
        peak = self.peak
        lines.append(
            f"  peak at frequency       {peak['freq']:.5g} "
            f"(cycle of {peak['cycle_length']:.4g} observations)"
        )
        return "\n".join(lines)

    def plot(self, ax: Any = None, log: bool = True, omega: bool = False) -> Any:
        """Plot the estimate against frequency.

        Parameters
        ----------
        ax : matplotlib Axes, optional
        log : bool, default True
            Logarithmic vertical axis, on which the confidence band has
            constant width.
        omega : bool, default False
            Put radians rather than cycles per observation on the
            horizontal axis.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        tab = self.table
        if log:
            tab = tab[tab["spectrum"] > 0]
        f = tab["omega" if omega else "freq"]
        ax.plot(f, tab["spectrum"], color="C0", lw=1.2)
        if "lower" in tab.columns:
            ax.fill_between(f, tab["lower"], tab["upper"], color="C0", alpha=0.15)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("radians" if omega else "cycles per observation")
        ax.set_ylabel("spectral density")
        return ax


def periodogram(
    data: ArrayLike,
    y: Optional[str] = None,
    *,
    method: str = "raw",
    spans: Union[int, Sequence[int], None] = None,
    taper: float = 0.1,
    detrend: str = "linear",
    pad: float = 0.0,
    fast: bool = False,
    order: Optional[int] = None,
    max_order: Optional[int] = None,
    n_freq: int = 500,
    scale: str = "cycles",
    alpha: float = 0.05,
) -> SpectrumResult:
    """Estimate the spectral density of a series.

    Parameters
    ----------
    data : DataFrame, Series or array
        The series, in time order. With a DataFrame, ``y`` names the column.
    y : str, optional
        Column of ``data``.
    method : {'raw', 'smoothed', 'ar'}, default 'raw'
        ``'raw'`` is the periodogram at the Fourier frequencies ``j / n``,
        ``j = 1, ..., floor(n / 2)``. ``'smoothed'`` averages neighbouring
        periodogram ordinates with the modified Daniell kernel(s) given by
        ``spans``. ``'ar'`` fits an autoregression by Yule-Walker and
        returns its spectral density on ``n_freq`` equally spaced
        frequencies from 0 to 1/2.
    spans : int or sequence of int, optional
        Widths of the modified Daniell smoothers, applied one after the
        other (``method='smoothed'``). A span ``s`` uses ``s // 2``
        ordinates on each side, the outermost with half weight.
    taper : float, default 0.1
        Share of the sample at each end that is tapered by a split cosine
        bell before the transform, between 0 and 0.5. Ignored by ``'ar'``.
    detrend : {'linear', 'mean', 'none'}, default 'linear'
        What is removed before the transform: a least-squares line, the
        mean, or nothing. ``'ar'`` always removes the mean only.
    pad : float, default 0
        Zeros appended before the transform, as a share of the sample
        length; ``pad=1`` doubles the number of frequencies.
    fast : bool, default False
        Pad with zeros to the next length whose prime factors are 2, 3
        and 5 (the default of R ``spec.pgram``). The frequencies are then
        ``j / n_padded`` and no longer the Fourier frequencies of the
        sample.
    order : int, optional
        Autoregressive order (``method='ar'``). Default: the order with
        the smallest AIC among 0, ..., ``max_order``.
    max_order : int, optional
        Largest order considered. Default
        ``min(n - 1, floor(10 * log10(n)))``.
    n_freq : int, default 500
        Number of frequencies for ``method='ar'``.
    scale : {'cycles', 'radians'}, default 'cycles'
        ``'cycles'``: density per cycle per observation, integrating to
        the variance over ``(-1/2, 1/2]``. ``'radians'``: density per
        radian, the former divided by ``2 pi``.
    alpha : float, default 0.05
        ``lower`` and ``upper`` are a pointwise ``1 - alpha`` interval.

    Returns
    -------
    SpectrumResult
        ``.table`` has one row per frequency; ``.df`` and ``.bandwidth``
        describe the smoother; ``.summary()`` and ``.plot()``.

    Raises
    ------
    MethodIncompatibility
        Unknown option values, ``spans`` missing or given with the wrong
        method, a gap in the series.
    DataInsufficient
        Fewer than four observations, a constant series, or an
        autoregressive order the sample cannot support.

    Notes
    -----
    With ``taper=0`` and ``scale='radians'`` the raw estimate at
    ``w_j = 2 pi j / n`` is the textbook periodogram
    ``I(w_j) = |sum_t x_t exp(-i w_j t)|**2 / (2 pi n)``; neither the mean
    nor a constant changes it at these frequencies, so ``detrend='mean'``
    and ``detrend='none'`` agree. ``scale='cycles'`` returns
    ``|sum_t x_t exp(-2 pi i f t)|**2 / n``.

    The defaults ``taper=0.1`` and ``detrend='linear'`` are those of R
    ``spec.pgram``; ``fast=True`` completes the match when ``n`` has a
    prime factor above 5. The estimate is divided by the mean square of
    the taper, ``1 - 5 taper / 4``.

    Each ordinate is treated as ``spectrum * chi2(df) / df`` with
    ``df = 2 n / (n_padded * c * sum_j K_j**2)``, where ``K`` is the kernel
    and ``c = (1 - 93 taper / 64) / (1 - 5 taper / 4)**2`` allows for the
    taper; the interval is ``spectrum * df / chi2`` at the two quantiles.
    The raw periodogram has ``df = 2`` and is not consistent: the interval
    does not shrink as the sample grows. ``bandwidth`` is the standard
    deviation of the kernel, with a continuity correction of 1/12, in
    cycles per observation.

    For ``'ar'`` the density is ``s2 / |1 - sum_k a_k exp(-2 pi i f k)|**2``
    with ``s2`` the Yule-Walker innovation variance times
    ``n / (n - order - 1)``, and the order minimises
    ``n log(s2_k) + 2 k`` (R ``ar`` and ``spec.ar``). No interval is
    reported: the chi-squared approximation does not apply.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> t = np.arange(200)
    >>> rng = np.random.default_rng(1)
    >>> x = np.cos(2 * np.pi * t / 20) + 0.5 * rng.normal(size=200)
    >>> res = sp.periodogram(x)
    >>> float(res.peak["cycle_length"])
    20.0
    >>> smooth = sp.periodogram(x, method="smoothed", spans=(3, 5))
    >>> round(smooth.df, 2)
    9.36
    >>> sp.periodogram(x, method="ar", order=2).order
    2

    References
    ----------
    [@brockwell1991time],
    [@neusser2016time]
    """
    who = "periodogram"
    if method not in ("raw", "smoothed", "ar"):
        raise MethodIncompatibility(
            f"sp.periodogram: method={method!r} is not 'raw', 'smoothed' or 'ar'.",
            recovery_hint="Choose one of the three estimators.",
        )
    if scale not in ("cycles", "radians"):
        raise MethodIncompatibility(
            f"sp.periodogram: scale={scale!r} is not 'cycles' or 'radians'.",
            recovery_hint="Use 'cycles' (R) or 'radians' (textbook).",
        )
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(
            f"sp.periodogram: alpha={alpha} is not between 0 and 1.",
            recovery_hint="Pass e.g. alpha=0.05.",
        )
    x = _series(data, y, who)
    n = x.size
    if n < 4:
        raise DataInsufficient(
            f"sp.periodogram: {n} observations.",
            recovery_hint="A spectrum needs at least four observations.",
        )
    if np.ptp(x) == 0.0:
        raise DataInsufficient(
            "sp.periodogram: the series is constant.",
            recovery_hint="There is no variance to decompose.",
        )
    divide = 2.0 * np.pi if scale == "radians" else 1.0
    if method == "ar":
        if spans is not None:
            raise MethodIncompatibility(
                "sp.periodogram: spans= does not apply to method='ar'.",
                recovery_hint="Drop spans, or use method='smoothed'.",
            )
        return _ar_spectrum(x, order, max_order, int(n_freq), scale, divide)

    if method == "raw" and spans is not None:
        raise MethodIncompatibility(
            "sp.periodogram: spans= was given with method='raw'.",
            recovery_hint="Use method='smoothed' to smooth the periodogram.",
        )
    if method == "smoothed" and spans is None:
        raise MethodIncompatibility(
            "sp.periodogram: method='smoothed' needs spans=.",
            recovery_hint="Pass odd widths, e.g. spans=(3, 5); a wider span "
            "lowers the variance and blurs peaks.",
        )
    span_list: Tuple[int, ...] = ()
    if spans is not None:
        raw_spans = [spans] if np.isscalar(spans) else list(spans)  # type: ignore
        if not raw_spans or any(
            int(s) != s or int(s) < 1 for s in raw_spans  # type: ignore[arg-type]
        ):
            raise MethodIncompatibility(
                f"sp.periodogram: spans={spans!r} must be positive integers.",
                recovery_hint="Pass e.g. spans=5 or spans=(3, 5).",
            )
        span_list = tuple(int(s) for s in raw_spans)  # type: ignore[arg-type]
    if not 0.0 <= taper <= 0.5:
        raise MethodIncompatibility(
            f"sp.periodogram: taper={taper} is not between 0 and 0.5.",
            recovery_hint="taper is the share tapered at each end.",
        )
    if detrend not in ("linear", "mean", "none"):
        raise MethodIncompatibility(
            f"sp.periodogram: detrend={detrend!r} is not 'linear', 'mean' or 'none'.",
            recovery_hint="Choose what to remove before the transform.",
        )
    if pad < 0:
        raise MethodIncompatibility(
            f"sp.periodogram: pad={pad} is negative.",
            recovery_hint="pad is the share of zeros appended.",
        )

    if detrend == "linear":
        t = np.arange(n) - (n - 1) / 2.0
        xw = x - x.mean() - t * (t @ x) / (t @ t)
    elif detrend == "mean":
        xw = x - x.mean()
    else:
        xw = x.copy()
    m = int(np.floor(n * taper))
    if m > 0:
        bell = 0.5 * (1.0 - np.cos(np.pi * (2 * np.arange(1, m + 1) - 1) / (2 * m)))
        xw[:m] *= bell
        xw[n - m :] *= bell[::-1]
    u2 = 1.0 - (5.0 / 8.0) * taper * 2.0
    u4 = 1.0 - (93.0 / 128.0) * taper * 2.0
    n_pad = n + int(round(n * pad))
    if fast:
        n_pad = _next_fast_length(n_pad)
    pg = np.abs(np.fft.fft(xw, n_pad)) ** 2 / n
    # the ordinate at frequency zero carries no information once a mean or
    # a line was removed; smoothing borrows its two neighbours instead
    pg[0] = 0.5 * (pg[1] + pg[-1])
    kernel = _daniell_kernel(span_list)
    if kernel.size > n_pad:
        raise DataInsufficient(
            f"sp.periodogram: spans={span_list} cover {kernel.size} ordinates "
            f"but there are only {n_pad}.",
            recovery_hint="Use narrower spans.",
        )
    if kernel.size > 1:
        pg = _circular_smooth(pg, kernel)
    n_spec = n_pad // 2
    spec = pg[1 : n_spec + 1] / u2 / divide
    freq = np.arange(1, n_spec + 1) / n_pad
    offsets = np.arange(kernel.size) - (kernel.size - 1) // 2
    df = 2.0 / float(kernel @ kernel) / (u4 / u2**2) * n / n_pad
    bandwidth = float(np.sqrt(np.sum((1.0 / 12.0 + offsets**2) * kernel))) / n_pad
    lo_q, hi_q = stats.chi2.ppf([alpha / 2.0, 1.0 - alpha / 2.0], df)
    table = pd.DataFrame(
        {
            "freq": freq,
            "omega": 2.0 * np.pi * freq,
            "cycle_length": 1.0 / freq,
            "spectrum": spec,
            "lower": spec * df / hi_q,
            "upper": spec * df / lo_q,
        }
    )
    return SpectrumResult(
        table=table,
        method=method,
        scale=scale,
        df=float(df),
        bandwidth=bandwidth,
        n=n,
        settings={
            "spans": span_list,
            "taper": float(taper),
            "detrend": detrend,
            "pad": float(pad),
            "fast": bool(fast),
            "n_padded": n_pad,
            "alpha": float(alpha),
        },
    )


def _ar_spectrum(
    x: np.ndarray,
    order: Optional[int],
    max_order: Optional[int],
    n_freq: int,
    scale: str,
    divide: float,
) -> SpectrumResult:
    n = x.size
    if n_freq < 2:
        raise MethodIncompatibility(
            f"sp.periodogram: n_freq={n_freq} is below 2.",
            recovery_hint="Use the default of 500.",
        )
    if order is not None:
        if int(order) != order or order < 0:
            raise MethodIncompatibility(
                f"sp.periodogram: order={order!r} is not a non-negative integer.",
                recovery_hint="Pass e.g. order=2, or leave it to AIC.",
            )
        top = int(order)
    else:
        top = (
            min(n - 1, int(np.floor(10 * np.log10(n))))
            if max_order is None
            else int(max_order)
        )
        if top < 0:
            raise MethodIncompatibility(
                f"sp.periodogram: max_order={max_order} is negative.",
                recovery_hint="Pass a non-negative integer.",
            )
    if top >= n - 1:
        raise DataInsufficient(
            f"sp.periodogram: an AR({top}) cannot be fitted to {n} observations.",
            recovery_hint="Lower order / max_order or use a longer series.",
        )
    phi, v, _ = _yule_walker(x, top)
    if order is None:
        aic = n * np.log(v) + 2.0 * np.arange(top + 1)
        p = int(np.argmin(aic))
    else:
        aic = None
        p = top
    coef = phi[p, 1 : p + 1].copy()
    var_pred = float(v[p] * n / (n - (p + 1)))
    freq = np.linspace(0.0, 0.5, n_freq)
    if p > 0:
        lags = np.arange(1, p + 1)
        transfer = 1.0 - np.exp(-2j * np.pi * np.outer(freq, lags)) @ coef
        spec = var_pred / np.abs(transfer) ** 2
    else:
        spec = np.full(n_freq, var_pred)
    with np.errstate(divide="ignore"):
        cycle = 1.0 / freq
    table = pd.DataFrame(
        {
            "freq": freq,
            "omega": 2.0 * np.pi * freq,
            "cycle_length": cycle,
            "spectrum": spec / divide,
        }
    )
    settings: Dict[str, Any] = {"max_order": top, "selected_by_aic": order is None}
    if aic is not None:
        settings["aic"] = aic - aic.min()
    return SpectrumResult(
        table=table,
        method="ar",
        scale=scale,
        df=None,
        bandwidth=None,
        n=n,
        order=p,
        ar_coef=coef,
        var_pred=var_pred,
        settings=settings,
    )


@dataclass
class CumulativePeriodogramResult(ResultProtocolMixin):
    """Bartlett's test returned by
    :func:`statspai.cumulative_periodogram_test`.

    Attributes
    ----------
    statistic : float
        ``sqrt(q)`` times the largest distance between the cumulative
        periodogram and the straight line of white noise.
    pvalue : float
    n : int
    table : pandas.DataFrame
        ``freq`` (cycles per observation), ``cumulative`` (share of the
        variance at frequencies up to ``freq``) and ``expected`` (the same
        under white noise, ``2 * freq``).
    alpha : float
    band : float
        Half-width of the ``1 - alpha`` band around ``expected``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(3)
    >>> res = sp.cumulative_periodogram_test(rng.normal(size=200))
    >>> bool(res.pvalue > 0.05)
    True
    """

    statistic: float
    pvalue: float
    n: int
    table: pd.DataFrame
    alpha: float
    band: float

    @property
    def reject(self) -> bool:
        """Whether white noise is rejected at level ``alpha``."""
        return bool(self.pvalue < self.alpha)

    def summary(self) -> str:
        """Plain-text report of the test."""
        return "\n".join(
            [
                "Cumulative periodogram white-noise test (Bartlett)",
                f"  observations            {self.n}",
                f"  Bartlett's B statistic  {self.statistic:.4f}",
                f"  Prob > B                {self.pvalue:.4f}",
            ]
        )

    def plot(self, ax: Any = None) -> Any:
        """Cumulative periodogram with the white-noise band."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        tab = self.table
        ax.plot(tab["freq"], tab["cumulative"], color="C0", lw=1.2)
        ax.plot(tab["freq"], tab["expected"], color="0.4", lw=0.8)
        for sign in (-1.0, 1.0):
            ax.plot(
                tab["freq"],
                tab["expected"] + sign * self.band,
                color="0.4",
                lw=0.8,
                ls="--",
            )
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("cycles per observation")
        ax.set_ylabel("cumulative periodogram")
        return ax


def cumulative_periodogram_test(
    data: ArrayLike, y: Optional[str] = None, *, alpha: float = 0.05
) -> CumulativePeriodogramResult:
    """Bartlett's cumulative-periodogram test that a series is white noise.

    Parameters
    ----------
    data : DataFrame, Series or array
        The series, in time order. With a DataFrame, ``y`` names the column.
    y : str, optional
        Column of ``data``.
    alpha : float, default 0.05
        Level of the band reported in ``.band`` and drawn by ``.plot()``.

    Returns
    -------
    CumulativePeriodogramResult
        ``.statistic``, ``.pvalue``, ``.table``, ``.summary()``,
        ``.plot()``.

    Raises
    ------
    DataInsufficient
        Fewer than four observations or a constant series.
    MethodIncompatibility
        A gap in the series, or ``alpha`` outside (0, 1).

    Notes
    -----
    The periodogram of the demeaned series is taken at the frequencies
    ``k / n``, ``k = 0, ..., q - 1`` with ``q = floor(n / 2) + 1``, and
    accumulated into ``F_k``, the share of the total up to ``k / n``. Under
    white noise ``F_k`` is close to ``2 k / n``. The statistic is
    ``B = sqrt(q) * max_k |F_k - 2 k / n|`` and is referred to the
    Kolmogorov distribution, ``P(B > b) = 2 sum_j (-1)**(j-1)
    exp(-2 j**2 b**2)``. This is the statistic of Stata ``wntestb``; R
    ``cpgram`` draws the same curve. The p-value sums the series to
    machine precision; Stata stops once a term falls below about 1e-8, so
    its ``r(p)`` agrees to that accuracy only.

    A series that was estimated (residuals of a fitted model) is closer
    to the line than white noise would be, and the test is then
    conservative.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(3)
    >>> e = rng.normal(size=200)
    >>> x = np.cumsum(e) * 0.2 + e
    >>> sp.cumulative_periodogram_test(x).reject
    True
    """
    who = "cumulative_periodogram_test"
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(
            f"sp.{who}: alpha={alpha} is not between 0 and 1.",
            recovery_hint="Pass e.g. alpha=0.05.",
        )
    x = _series(data, y, who)
    n = x.size
    if n < 4:
        raise DataInsufficient(
            f"sp.{who}: {n} observations.",
            recovery_hint="The test needs at least four observations.",
        )
    if np.ptp(x) == 0.0:
        raise DataInsufficient(
            f"sp.{who}: the series is constant.",
            recovery_hint="There is no variance to decompose.",
        )
    q = n // 2 + 1
    pg = np.abs(np.fft.fft(x - x.mean())[:q]) ** 2
    pg[0] = 0.0
    cumulative = np.cumsum(pg) / pg.sum()
    freq = np.arange(q) / n
    expected = 2.0 * freq
    statistic = float(np.sqrt(q) * np.max(np.abs(cumulative - expected)))
    table = pd.DataFrame({"freq": freq, "cumulative": cumulative, "expected": expected})
    return CumulativePeriodogramResult(
        statistic=statistic,
        pvalue=float(special.kolmogorov(statistic)),
        n=n,
        table=table,
        alpha=float(alpha),
        band=float(special.kolmogi(alpha) / np.sqrt(q)),
    )
