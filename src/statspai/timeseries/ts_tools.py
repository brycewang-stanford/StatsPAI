"""Small tools of a forecasting workflow: the portmanteau test for
residual autocorrelation, the number of (seasonal) differences a series
needs, the Box-Cox parameter that stabilises its variance and Fourier
terms for seasonal patterns.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._forecast_common import check_period, read_series

#: critical values of the KPSS level-stationarity statistic
#: (Kwiatkowski, Phillips, Schmidt and Shin, 1992, Table 1)
_KPSS_CRIT = {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739}


# ----------------------------------------------------------------------
# portmanteau test
# ----------------------------------------------------------------------
def ljungbox(
    x: Any,
    lags: Union[int, Sequence[int], None] = None,
    *,
    model_df: Optional[int] = None,
    method: str = "ljung-box",
    period: int = 1,
    data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Ljung-Box (or Box-Pierce) portmanteau test of no autocorrelation.

    Parameters
    ----------
    x : array-like, pd.Series, str or fitted model
        The series, usually residuals; a column name when ``data`` is
        given. A fitted ``sp.arima`` / ``sp.ets`` /
        ``sp.simple_forecast`` result can be passed directly: its
        innovation residuals are tested.
    lags : int or sequence of int, optional
        Number of autocorrelations in the statistic. An integer gives
        one test at that lag, a sequence one test per entry. Default:
        10 for non-seasonal data and twice the period for seasonal
        data, capped at a fifth of the sample.
    model_df : int, optional
        Number of ARMA parameters estimated in the model the residuals
        come from; the reference distribution is chi-squared with
        ``lags - model_df`` degrees of freedom. Default 0, or
        ``p + q + P + Q`` when ``x`` is a fitted ARIMA model.
    method : {"ljung-box", "box-pierce"}, default "ljung-box"
    period : int, default 1
        Seasonal period, used only for the default number of lags.
    data : pd.DataFrame, optional

    Returns
    -------
    pd.DataFrame
        Indexed by ``lag``: ``statistic``, ``df``, ``p_value``.

    Notes
    -----
    ``Q = n (n + 2) sum_{k=1}^{L} r_k^2 / (n - k)`` (Ljung and Box, 1978)
    or ``Q = n sum r_k^2`` (Box and Pierce, 1970). Equal to R's
    ``Box.test(x, lag, type, fitdf)``, Stata's ``wntestq`` and
    statsmodels' ``acorr_ljungbox(x, lags, model_df=)``.

    A small p-value says the residuals are autocorrelated: the model
    leaves forecastable structure behind, and its prediction intervals
    are not to be trusted.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> e = rng.normal(size=200)
    >>> out = sp.ljungbox(e, lags=10)
    >>> out.index.tolist(), list(out.columns)
    ([10], ['statistic', 'df', 'p_value'])
    >>> walk = np.cumsum(e)
    >>> bool(sp.ljungbox(walk, lags=10)["p_value"].iloc[0] < 0.001)
    True

    References
    ----------
    ljung1978measure, box1970distribution
    """
    key = method.lower().replace("_", "-").replace(" ", "-")
    if key in ("ljung-box", "ljung", "lb", "ljungbox"):
        ljung = True
    elif key in ("box-pierce", "bp", "boxpierce"):
        ljung = False
    else:
        raise MethodIncompatibility(
            f"ljungbox: method={method!r} is not 'ljung-box' or 'box-pierce'.",
            recovery_hint="Use method='ljung-box'.",
        )
    if data is None and not isinstance(x, (str, pd.Series, np.ndarray, list, tuple)):
        res = getattr(x, "residuals", None)
        if res is None:
            raise MethodIncompatibility(
                "ljungbox: x is neither a series nor a fitted model with " "residuals.",
                recovery_hint="Pass the residuals as an array.",
            )
        if model_df is None:
            model_df = int(getattr(x, "n_arma_params", 0) or 0)
        if period == 1:
            period = int(getattr(x, "period", 1) or 1)
        x = np.asarray(res, dtype=float)
        x = x[np.isfinite(x)]
    values, _, _ = read_series(x, data, fn="ljungbox")
    n = values.shape[0]
    m = check_period(period, fn="ljungbox")
    k = int(model_df or 0)
    if k < 0:
        raise MethodIncompatibility(
            "ljungbox: model_df cannot be negative.",
            recovery_hint="Pass the number of estimated ARMA parameters.",
        )
    if lags is None:
        default = 2 * m if m > 1 else 10
        lag_list = [max(min(default, n // 5), k + 1)]
    elif np.isscalar(lags):
        lag_list = [int(lags)]  # type: ignore[arg-type]
    else:
        lag_list = [int(v) for v in lags]  # type: ignore[union-attr]
    if not lag_list or min(lag_list) < 1:
        raise MethodIncompatibility(
            "ljungbox: lags must be positive.",
            recovery_hint="E.g. lags=10.",
        )
    top = max(lag_list)
    if top >= n:
        raise DataInsufficient(
            f"ljungbox: {top} lags need more than {n} observations.",
            recovery_hint="Lower lags=.",
        )
    d = values - values.mean()
    den = float(d @ d)
    if den <= 0:
        raise DataInsufficient(
            "ljungbox: the series is constant.",
            recovery_hint="There is no autocorrelation to test.",
        )
    r = np.array([float(d[j:] @ d[:-j]) / den for j in range(1, top + 1)])
    if ljung:
        terms = n * (n + 2.0) * r**2 / (n - np.arange(1, top + 1))
    else:
        terms = n * r**2
    cum = np.cumsum(terms)
    rows = []
    for L in lag_list:
        df = L - k
        q = float(cum[L - 1])
        p = float(stats.chi2.sf(q, df)) if df > 0 else float("nan")
        rows.append({"lag": L, "statistic": q, "df": df, "p_value": p})
    out = pd.DataFrame(rows).set_index("lag")
    out.attrs["method"] = "Ljung-Box" if ljung else "Box-Pierce"
    out.attrs["model_df"] = k
    return out


# ----------------------------------------------------------------------
# number of differences
# ----------------------------------------------------------------------
def _kpss_level(x: np.ndarray) -> float:
    """KPSS level-stationarity statistic with a Bartlett kernel of
    ``trunc(3 sqrt(n) / 13)`` lags (``nan`` when it is not defined)."""
    n = x.shape[0]
    e = x - x.mean()
    partial = np.cumsum(e)
    lags = int(3 * np.sqrt(n) / 13)
    lrv = float(e @ e) / n
    for h in range(1, lags + 1):
        lrv += 2.0 * (1.0 - h / (lags + 1.0)) * float(e[h:] @ e[:-h]) / n
    if lrv <= 0:
        return float("nan")
    return float(float(partial @ partial) / (n * n * lrv))


def ndiffs(
    y: Any,
    *,
    alpha: float = 0.05,
    max_d: int = 2,
    data: Optional[pd.DataFrame] = None,
) -> int:
    """Number of first differences that make a series stationary.

    The series is differenced until a KPSS test no longer rejects level
    stationarity at ``alpha``, at most ``max_d`` times.

    Parameters
    ----------
    y : array-like, pd.Series or str
    alpha : {0.10, 0.05, 0.025, 0.01}, default 0.05
        Level of each KPSS test (the levels its critical values are
        tabulated for).
    max_d : int, default 2
    data : pd.DataFrame, optional

    Returns
    -------
    int

    Notes
    -----
    The rule of Hyndman and Khandakar (2008) and of R's
    ``forecast::ndiffs(test="kpss")``, including its lag truncation
    ``trunc(3 sqrt(n) / 13)``. :func:`statspai.arima` with ``auto=True``
    uses it to fix ``d`` before comparing models by AICc, because
    likelihoods of differently differenced series are not comparable.
    For a seasonal series take the seasonal differences first, see
    :func:`statspai.nsdiffs`.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> e = rng.normal(size=300)
    >>> sp.ndiffs(e), sp.ndiffs(np.cumsum(e))
    (0, 1)

    References
    ----------
    hyndman2008automatic, kwiatkowski1992testing
    """
    values, _, _ = read_series(y, data, fn="ndiffs")
    crit = None
    for a, c in _KPSS_CRIT.items():
        if abs(float(alpha) - a) < 1e-12:
            crit = c
    if crit is None:
        raise MethodIncompatibility(
            f"ndiffs: alpha={alpha!r} has no tabulated KPSS critical value.",
            recovery_hint="Use 0.10, 0.05, 0.025 or 0.01.",
        )
    if int(max_d) < 0:
        raise MethodIncompatibility(
            "ndiffs: max_d cannot be negative.", recovery_hint="Use max_d=2."
        )
    return _ndiffs(values, int(max_d), crit)


def _ndiffs(x: np.ndarray, max_d: int, crit: float = 0.463) -> int:
    x = np.asarray(x, dtype=float)
    d = 0
    while d < max_d and len(x) >= 8 and np.ptp(x) > 0:
        stat = _kpss_level(x)
        if not stat > crit:
            break
        x = np.diff(x)
        d += 1
    return d


def _seasonal_strength(x: np.ndarray, m: int) -> float:
    from .stl import stl

    return float(stl(x, m).strength["seasonal"])


def nsdiffs(
    y: Any,
    period: int,
    *,
    threshold: float = 0.64,
    max_D: int = 1,
    data: Optional[pd.DataFrame] = None,
) -> int:
    """Number of seasonal differences a series needs.

    A seasonal difference is taken while the strength of seasonality of
    an STL decomposition exceeds ``threshold``.

    Parameters
    ----------
    y : array-like, pd.Series or str
    period : int
        Seasonal period.
    threshold : float, default 0.64
        Strength of seasonality above which a seasonal difference is
        taken; 0.64 minimised forecast errors on the M3 competition
        data.
    max_D : int, default 1
    data : pd.DataFrame, optional

    Returns
    -------
    int

    Notes
    -----
    The strength is ``max(0, 1 - Var(R) / Var(S + R))`` from
    :func:`statspai.stl` with its defaults, the measure of Wang, Smith
    and Hyndman (2006); the rule is that of R's
    ``forecast::nsdiffs(test="seas")``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> t = np.arange(120)
    >>> rng = np.random.default_rng(0)
    >>> y = 10 * np.sin(2 * np.pi * t / 12) + rng.normal(size=120)
    >>> sp.nsdiffs(y, 12), sp.nsdiffs(rng.normal(size=120), 12)
    (1, 0)

    References
    ----------
    wang2006characteristic, hyndman2008automatic
    """
    values, _, _ = read_series(y, data, fn="nsdiffs")
    m = check_period(period, fn="nsdiffs")
    if m < 2:
        raise MethodIncompatibility(
            "nsdiffs: non-seasonal data (period=1) have no seasonal difference.",
            recovery_hint="Pass the seasonal period, or use sp.ndiffs.",
        )
    return _nsdiffs(values, m, float(threshold), int(max_D))


def _nsdiffs(x: np.ndarray, m: int, threshold: float = 0.64, max_D: int = 1) -> int:
    x = np.asarray(x, dtype=float)
    D = 0
    while D < max_D and len(x) >= 2 * m + 1 and np.ptp(x) > 0:
        if not _seasonal_strength(x, m) > threshold:
            break
        x = x[m:] - x[:-m]
        D += 1
    return D


# ----------------------------------------------------------------------
# Box-Cox parameter
# ----------------------------------------------------------------------
def boxcox_lambda(
    y: Any,
    period: int = 1,
    *,
    lower: float = -1.0,
    upper: float = 2.0,
    data: Optional[pd.DataFrame] = None,
) -> float:
    """Box-Cox parameter that makes the variation of a series the same
    across its level, by Guerrero's method.

    The series is cut into consecutive cycles; ``lambda`` minimises the
    coefficient of variation, across cycles, of
    ``sd / mean^(1 - lambda)``.

    Parameters
    ----------
    y : array-like, pd.Series or str
        A strictly positive series.
    period : int, default 1
        Seasonal period; cycles of length 2 are used for non-seasonal
        data.
    lower, upper : float, default -1 and 2
        Search interval.
    data : pd.DataFrame, optional

    Returns
    -------
    float
        Apply it with ``scipy.special.boxcox(y, lam)`` and invert with
        ``scipy.special.inv_boxcox``. ``lambda = 0`` is the logarithm,
        ``1`` no transformation.

    Notes
    -----
    The criterion is that of R's ``forecast::BoxCox.lambda`` and
    ``fabletools::guerrero``. It is minimised to full precision, where
    R's ``optimize`` stops at a tolerance of about ``1e-4``; expect
    agreement to four decimals.

    A back-transformed point forecast is the median of the forecast
    distribution, not its mean. For a logarithm the mean is
    ``exp(mu) * (1 + sigma_h^2 / 2)`` to second order, with ``sigma_h``
    the forecast standard deviation on the log scale.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> t = np.arange(96)
    >>> y = np.exp(0.03 * t) * (1 + 0.2 * np.sin(2 * np.pi * t / 4))
    >>> abs(sp.boxcox_lambda(y, period=4)) < 0.05      # the logarithm
    True

    References
    ----------
    guerrero1993time
    """
    values, _, _ = read_series(y, data, fn="boxcox_lambda")
    m = max(2, check_period(period, fn="boxcox_lambda"))
    if values.min() <= 0:
        raise MethodIncompatibility(
            "boxcox_lambda: Guerrero's method needs a strictly positive series.",
            recovery_hint="Shift the series or model it untransformed.",
        )
    n = values.shape[0]
    n_cycles = n // m
    if n_cycles < 2:
        raise DataInsufficient(
            f"boxcox_lambda: {n} observations give fewer than two cycles of "
            f"length {m}.",
            recovery_hint="Provide a longer series.",
        )
    mat = values[n - n_cycles * m :].reshape(n_cycles, m)
    mean = mat.mean(axis=1)
    sd = mat.std(axis=1, ddof=1)

    def cv(lam: float) -> float:
        rat = sd / mean ** (1.0 - lam)
        return float(np.std(rat, ddof=1) / np.mean(rat))

    res = optimize.minimize_scalar(
        cv,
        bounds=(float(lower), float(upper)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(res.x)


# ----------------------------------------------------------------------
# Fourier terms
# ----------------------------------------------------------------------
def fourier_terms(
    n: Any,
    period: float,
    K: int,
    *,
    start: int = 1,
) -> pd.DataFrame:
    """Sine and cosine regressors for a seasonal pattern.

    ``K`` pairs ``sin(2 pi k t / period)``, ``cos(2 pi k t / period)``,
    ``k = 1..K``, describe a smooth seasonal pattern with ``2K``
    coefficients instead of ``period - 1`` dummies. They handle long
    and non-integer periods (52.18 weeks in a year) and several
    seasonal patterns at once, as regressors of :func:`statspai.arima`
    (dynamic harmonic regression) or of :func:`statspai.regress`.

    Parameters
    ----------
    n : int, pd.Series, pd.DataFrame or pd.Index
        Number of rows; or an object whose length and index are used.
    period : float
        Length of the seasonal cycle, in observations.
    K : int
        Number of harmonics, at most ``period / 2``.
    start : int, default 1
        Time of the first row. The regressors of the ``h`` periods after
        a sample of length ``T`` are ``fourier_terms(h, period, K,
        start=T + 1)``.

    Returns
    -------
    pd.DataFrame
        Columns ``sin1_<period>``, ``cos1_<period>``, ... A sine that is
        identically zero (``2k = period``) is left out.

    Notes
    -----
    The values equal R's ``forecast::fourier(x, K)`` and, for the future
    rows, ``fourier(x, K, h)``. Choose ``K`` by AICc: every extra pair
    costs two parameters.

    Examples
    --------
    >>> import statspai as sp
    >>> X = sp.fourier_terms(8, period=4, K=2)
    >>> list(X.columns)
    ['sin1_4', 'cos1_4', 'cos2_4']
    >>> sp.fourier_terms(2, period=4, K=1, start=9).round(6).values.tolist()
    [[1.0, 0.0], [0.0, -1.0]]

    References
    ----------
    hyndman2026fpppy
    """
    index: Optional[pd.Index] = None
    if isinstance(n, (pd.Series, pd.DataFrame)):
        index = n.index
        rows = len(n)
    elif isinstance(n, pd.Index):
        index = n
        rows = len(n)
    else:
        rows = int(n)
    if rows < 1:
        raise MethodIncompatibility(
            "fourier_terms: at least one row is needed.",
            recovery_hint="Pass the number of periods.",
        )
    per = float(period)
    if not per > 1:
        raise MethodIncompatibility(
            f"fourier_terms: period={period!r} must be above 1.",
            recovery_hint="12 for monthly data, 52.18 for weekly data.",
        )
    kk = int(K)
    if kk < 1 or 2 * kk > per:
        raise MethodIncompatibility(
            f"fourier_terms: K={K!r} must be between 1 and period/2 = {per / 2:g}.",
            recovery_hint="More harmonics than half the period are collinear.",
        )
    t = np.arange(int(start), int(start) + rows, dtype=float)
    lab = f"{per:g}"
    cols = {}
    for k in range(1, kk + 1):
        if abs(2 * k - per) > 1e-9:
            cols[f"sin{k}_{lab}"] = np.sin(2.0 * np.pi * k * t / per)
        cols[f"cos{k}_{lab}"] = np.cos(2.0 * np.pi * k * t / per)
    out = pd.DataFrame(cols)
    if index is not None:
        out.index = index
    return out


def seasonal_dummies(
    n: Any,
    period: int,
    *,
    start: int = 1,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Seasonal indicator regressors for a regression with a seasonal
    pattern.

    Parameters
    ----------
    n : int, pd.Series, pd.DataFrame or pd.Index
        Number of rows; or an object whose length and index are used.
    period : int
        Seasonal period.
    start : int, default 1
        Time of the first row; row ``t`` is in season
        ``(t - 1) mod period + 1``. The rows of the ``h`` periods after a
        sample of length ``T`` are ``seasonal_dummies(h, period,
        start=T + 1)``.
    drop_first : bool, default True
        Leave out the indicator of season 1, the reference season of a
        regression with an intercept (the "dummy variable trap").

    Returns
    -------
    pd.DataFrame
        Columns ``season_2`` ... ``season_<period>`` of zeros and ones.

    Notes
    -----
    R's ``forecast::seasonaldummy`` and the ``season`` term of ``tslm``
    build the same columns (with the last season as the reference in
    ``seasonaldummy``). With a long period prefer
    :func:`statspai.fourier_terms`.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.seasonal_dummies(5, 4).values.tolist()
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]

    References
    ----------
    hyndman2026fpppy
    """
    index: Optional[pd.Index] = None
    if isinstance(n, (pd.Series, pd.DataFrame)):
        index = n.index
        rows = len(n)
    elif isinstance(n, pd.Index):
        index = n
        rows = len(n)
    else:
        rows = int(n)
    m = check_period(period, fn="seasonal_dummies")
    if m < 2 or rows < 1:
        raise MethodIncompatibility(
            "seasonal_dummies: period must be at least 2 and n at least 1.",
            recovery_hint="E.g. sp.seasonal_dummies(len(y), 4).",
        )
    season = (np.arange(int(start), int(start) + rows) - 1) % m + 1
    first = 2 if drop_first else 1
    out = pd.DataFrame(
        {f"season_{k}": (season == k).astype(int) for k in range(first, m + 1)}
    )
    if index is not None:
        out.index = index
    return out
