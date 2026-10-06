"""Summary features of a time series: autocorrelation, decomposition
strength, stability and shift measures.

One row of numbers per series turns a large collection of series into a
table that can be plotted, clustered or screened for unusual members
(Hyndman and Athanasopoulos, chapter 4). The definitions are those of the
R package ``tsfeatures`` (Hyndman, Wang and Laptev's feature set), on a
series scaled to mean zero and unit variance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._forecast_common import check_period, read_series


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    d = x - x.mean()
    den = float(d @ d)
    k = min(nlags, x.shape[0] - 1)
    out = np.full(nlags, np.nan)
    for j in range(1, k + 1):
        out[j - 1] = float(d[j:] @ d[:-j]) / den
    return out


def _pacf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Partial autocorrelations by the Durbin-Levinson recursion on the
    sample autocorrelations (R ``pacf``)."""
    r = np.r_[1.0, _acf(x, nlags)]
    out = np.full(nlags, np.nan)
    phi_prev: np.ndarray = np.zeros(0)
    for k in range(1, nlags + 1):
        if not np.isfinite(r[k]):
            break
        if k == 1:
            a = r[1]
            phi: np.ndarray = np.array([a])
        else:
            num = r[k] - float(phi_prev @ r[k - 1 : 0 : -1])
            den = 1.0 - float(phi_prev @ r[1:k])
            a = num / den
            phi = np.r_[phi_prev - a * phi_prev[::-1], a]
        out[k - 1] = a
        phi_prev = phi
    return out


def _tiles(x: np.ndarray, width: int) -> np.ndarray:
    k = x.shape[0] // width
    return x[: k * width].reshape(k, width)


def _roll(x: np.ndarray, width: int, fn: Any) -> np.ndarray:
    n = x.shape[0] - width + 1
    return np.array([fn(x[i : i + width]) for i in range(n)])


def _one(values: np.ndarray, m: int, scale: bool) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    n = x.shape[0]
    if scale:
        sd = float(np.std(x, ddof=1))
        if not sd > 0:
            raise DataInsufficient(
                "ts_features: the series is constant.",
                recovery_hint="A constant series has no features to compute.",
            )
        x = (x - x.mean()) / sd
    out: Dict[str, float] = {}
    # ---- autocorrelation
    d1, d2 = np.diff(x), np.diff(x, 2)
    a, a1, a2 = _acf(x, 10), _acf(d1, 10), _acf(d2, 10)
    out["x_acf1"] = float(a[0])
    out["x_acf10"] = float(np.nansum(a**2))
    out["diff1_acf1"] = float(a1[0])
    out["diff1_acf10"] = float(np.nansum(a1**2))
    out["diff2_acf1"] = float(a2[0])
    out["diff2_acf10"] = float(np.nansum(a2**2))
    if m > 1 and n > m + 1:
        out["seas_acf1"] = float(_acf(x, m)[m - 1])
    lags = max(m, 5)
    pc = _pacf(x, lags)
    out["x_pacf5"] = float(np.sum(pc[:5] ** 2))
    out["diff1x_pacf5"] = float(np.sum(_pacf(d1, 5) ** 2))
    out["diff2x_pacf5"] = float(np.sum(_pacf(d2, 5) ** 2))
    if m > 1 and n > m + 1:
        out["seas_pacf"] = float(pc[m - 1])
    # ---- decomposition
    if m > 1 and n >= 2 * m:
        from .stl import stl

        dec = stl(x, m)
        rem, tr, se = dec.remainder, dec.trend, dec.seasonal
        st = dec.strength
        out["trend"] = st["trend"]
        out["seasonal_strength"] = st["seasonal"]
        # spike: variance of the remainder's variance with one
        # observation left out in turn (deviations from the full mean, as
        # in tsfeatures)
        dev2 = (rem - rem.mean()) ** 2
        loo_var = (float(dev2.sum()) - dev2) / (n - 2)
        out["spike"] = float(np.var(loo_var, ddof=1))
        # linearity and curvature: orthogonal quadratic regression of the trend
        t = np.arange(1, n + 1, dtype=float)
        q, _ = np.linalg.qr(np.column_stack([np.ones(n), t, t**2]))
        q1, q2 = q[:, 1], q[:, 2]
        if q1[-1] < q1[0]:
            q1 = -q1
        # R's poly() makes the quadratic term positive at both ends
        if q2[0] < 0:
            q2 = -q2
        out["linearity"] = float(q1 @ tr)
        out["curvature"] = float(q2 @ tr)
        e = _acf(rem, 10)
        out["e_acf1"] = float(e[0])
        out["e_acf10"] = float(np.nansum(e**2))
        figure = np.array([se[j::m].mean() for j in range(m)])
        out["peak"] = float(np.argmax(figure) + 1)
        out["trough"] = float(np.argmin(figure) + 1)
    # ---- stability of the mean and of the variance over tiles
    width = m if m > 1 else 10
    if n >= 2 * width:
        tiles = _tiles(x, width)
        out["lumpiness"] = float(np.var(tiles.var(axis=1, ddof=1), ddof=1))
        out["stability"] = float(np.var(tiles.mean(axis=1), ddof=1))
    med = np.median(x)
    above = x > med
    out["crossing_points"] = float(np.sum(above[1:] != above[:-1]))
    # longest run inside one of ten equal-width bins
    lo, hi = float(x.min()), float(x.max())
    edges = np.linspace(lo - (hi - lo) / 1000.0, hi + (hi - lo) / 1000.0, 11)
    bins = np.clip(np.searchsorted(edges, x, side="left"), 1, 10)
    run = best = 1
    for i in range(1, n):
        run = run + 1 if bins[i] == bins[i - 1] else 1
        best = max(best, run)
    out["flat_spots"] = float(best)
    # ---- shifts between consecutive windows
    if n >= 2 * width + 1:
        means = _roll(x, width, np.mean)
        dm = np.abs(means[width:] - means[:-width])
        out["max_level_shift"] = float(dm.max())
        out["time_level_shift"] = float(np.argmax(dm) + width)
        var = _roll(x, width, lambda v: float(np.var(v, ddof=1)))
        dv = np.abs(var[width:] - var[:-width])
        out["max_var_shift"] = float(dv.max())
        out["time_var_shift"] = float(np.argmax(dv) + width)
    # ---- ARCH effect: R^2 of the squared series on twelve of its lags
    if n > 30:
        z = (x - x.mean()) ** 2
        L = 12
        Y = z[L:]
        X = np.column_stack(
            [np.ones(n - L)] + [z[L - j : n - j] for j in range(1, L + 1)]
        )
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        res = Y - X @ beta
        tss = float(np.sum((Y - Y.mean()) ** 2))
        out["arch_lm"] = float(1.0 - (res @ res) / tss) if tss > 0 else float("nan")
    return out


def ts_features(
    y: Any,
    period: int = 1,
    *,
    scale: bool = True,
    data: Optional[pd.DataFrame] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Features of one series, or of every column of a table of series.

    Parameters
    ----------
    y : array-like, pd.Series, pd.DataFrame or str
        One series; a column name when ``data`` is given; or a wide
        DataFrame with one series per column (missing values at the ends
        of a column are trimmed).
    period : int, default 1
        Seasonal period. The seasonal and decomposition features need
        ``period > 1``.
    scale : bool, default True
        Standardise each series to mean zero and unit variance first, so
        that the features describe its shape, not its units.
    data : pd.DataFrame, optional

    Returns
    -------
    pd.Series or pd.DataFrame
        One value per feature; one row per series for a table.

        - ``x_acf1``, ``x_acf10``: first autocorrelation and sum of the
          first ten squared; ``diff1_*``, ``diff2_*`` for the series
          differenced once and twice; ``seas_acf1`` at the seasonal lag.
        - ``x_pacf5``, ``diff1x_pacf5``, ``diff2x_pacf5``: sum of the
          first five squared partial autocorrelations; ``seas_pacf``.
        - ``trend``, ``seasonal_strength``: strength of trend and of
          seasonality from an STL decomposition; ``spike``, ``linearity``,
          ``curvature``, ``e_acf1``, ``e_acf10``, ``peak``, ``trough``.
        - ``lumpiness``, ``stability``: variance of the variances and of
          the means of consecutive blocks.
        - ``crossing_points``, ``flat_spots``.
        - ``max_level_shift``, ``max_var_shift`` and their times.
        - ``arch_lm``: R-squared of the squared series on 12 of its lags.

    Notes
    -----
    The values equal those of R's ``tsfeatures::tsfeatures`` for the
    features listed. Its spectral entropy, Hurst exponent, nonlinearity
    and GARCH features are not computed, nor the decomposition features
    of a non-seasonal series, which R takes from a different smoother.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(120)
    >>> wide = pd.DataFrame({
    ...     "seasonal": 5 * np.sin(2 * np.pi * t / 12) + rng.normal(size=120),
    ...     "noise": rng.normal(size=120),
    ... })
    >>> feats = sp.ts_features(wide, period=12)
    >>> bool(feats.loc["seasonal", "seasonal_strength"] > 0.9)
    True
    >>> bool(feats.loc["noise", "seasonal_strength"] < 0.5)
    True

    References
    ----------
    wang2006characteristic, hyndman2026fpppy
    """
    m = check_period(period, fn="ts_features")
    if isinstance(y, pd.DataFrame) and data is None and y.shape[1] != 1:
        rows = {}
        for col in y.columns:
            values, _, _ = read_series(y[col], None, fn="ts_features")
            rows[col] = _one(values, m, scale)
        return pd.DataFrame.from_dict(rows, orient="index")
    values, _, name = read_series(y, data, fn="ts_features")
    if values.shape[0] < 12:
        raise DataInsufficient(
            f"ts_features: {values.shape[0]} observations are too few.",
            recovery_hint="At least 12 observations are needed.",
        )
    if scale not in (True, False):
        raise MethodIncompatibility(
            "ts_features: scale must be True or False.",
            recovery_hint="Use scale=True.",
        )
    return pd.Series(_one(values, m, bool(scale)), name=name)
