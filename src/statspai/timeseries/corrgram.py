"""Correlogram of one series: autocorrelations, partial autocorrelations and
the portmanteau (Ljung-Box) Q statistic at every lag."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["corrgram"]


def _series(
    data: Union[pd.DataFrame, pd.Series, np.ndarray], y: Optional[str]
) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        if y is None or y not in data.columns:
            raise MethodIncompatibility(
                f"sp.corrgram: y={y!r} is not a column of the data.",
                recovery_hint="Pass the name of the series, e.g. "
                "sp.corrgram(df, 'resid').",
            )
        x = data[y].to_numpy(dtype=float, na_value=np.nan)
    else:
        x = np.asarray(data, dtype=float).ravel()
    observed = np.flatnonzero(~np.isnan(x))
    if observed.size == 0:
        raise DataInsufficient(
            "sp.corrgram: the series has no observed value.",
            recovery_hint="Check the column.",
        )
    # missing values at either end only shorten the sample (a lag or a
    # residual is missing in the first periods); one in the middle is a gap
    x = x[observed[0] : observed[-1] + 1]
    if np.isnan(x).any():
        raise MethodIncompatibility(
            "sp.corrgram: the series has missing values between observed "
            "ones; autocorrelations need consecutive periods.",
            recovery_hint="Fill or drop the gap, or analyse the longest "
            "complete stretch.",
        )
    return x


def corrgram(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Optional[str] = None,
    *,
    lags: Optional[int] = None,
    pac: str = "regression",
) -> pd.DataFrame:
    """Autocorrelations, partial autocorrelations and Q statistics.

    Parameters
    ----------
    data : DataFrame, Series or array
        The series, in time order. With a DataFrame, ``y`` names the column.
    y : str, optional
        Column of ``data``.
    lags : int, optional
        Number of lags. Default ``min(floor(n / 2) - 2, 40)``, as Stata's
        ``corrgram``.
    pac : {'regression', 'yw'}, default 'regression'
        How the partial autocorrelation at lag ``k`` is computed:
        ``'regression'`` is the coefficient on the ``k``-th lag in a
        regression of the series on its first ``k`` lags and a constant
        (Stata's default); ``'yw'`` solves the Yule-Walker equations from
        the autocorrelations (Stata's ``yw`` option, R's ``pacf``).

    Returns
    -------
    pandas.DataFrame
        One row per lag with columns ``AC``, ``PAC``, ``Q`` and ``Prob>Q``.
        ``Q`` is the Ljung-Box statistic for no autocorrelation up to that
        lag, chi-squared with as many degrees of freedom as lags; the last
        row is Stata's ``wntestq`` at that lag. ``attrs['n']`` is the
        number of observations.

    Notes
    -----
    ``AC[k]`` divides the lag-``k`` autocovariance by the variance, both
    computed about the full-sample mean with divisor ``n``. Applied to
    regression residuals, the Q statistic's chi-squared reference
    distribution ignores that the residuals were estimated; after a fit use
    ``sp.estat(result, 'bgodfrey')``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> e = rng.normal(size=300)
    >>> x = np.zeros(300)
    >>> for t in range(1, 300):
    ...     x[t] = 0.7 * x[t - 1] + e[t]
    >>> table = sp.corrgram(pd.DataFrame({"x": x}), "x", lags=5)
    >>> list(table.columns)
    ['AC', 'PAC', 'Q', 'Prob>Q']
    >>> bool(table.loc[1, "AC"] > 0.5 and abs(table.loc[3, "PAC"]) < 0.2)
    True

    References
    ----------
    ljung1978measure
    """
    x = _series(data, y)
    n = x.size
    if lags is None:
        lags = min(n // 2 - 2, 40)
    if lags < 1 or lags >= n - 1:
        raise DataInsufficient(
            f"sp.corrgram: lags={lags} with {n} observations.",
            recovery_hint="Use a longer series or fewer lags.",
        )
    if pac not in ("regression", "yw"):
        raise MethodIncompatibility(
            f"sp.corrgram: pac={pac!r} is not 'regression' or 'yw'.",
            recovery_hint="Use pac='regression' (the default).",
        )
    d = x - x.mean()
    denom = float(d @ d)
    if denom <= 0:
        raise DataInsufficient(
            "sp.corrgram: the series is constant.",
            recovery_hint="Autocorrelations need variation.",
        )
    ac = np.array([float(d[k:] @ d[:-k]) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.cumsum(ac**2 / (n - np.arange(1, lags + 1)))
    prob = stats.chi2.sf(q, np.arange(1, lags + 1))

    part = np.full(lags, np.nan)
    if pac == "regression":
        for k in range(1, lags + 1):
            if n - k <= k + 1:
                break
            design = np.column_stack(
                [np.ones(n - k)] + [x[k - j : n - j] for j in range(1, k + 1)]
            )
            part[k - 1] = np.linalg.lstsq(design, x[k:], rcond=None)[0][-1]
    else:
        rho = np.concatenate([[1.0], ac])
        for k in range(1, lags + 1):
            toeplitz = rho[np.abs(np.subtract.outer(np.arange(k), np.arange(k)))]
            part[k - 1] = np.linalg.solve(toeplitz, rho[1 : k + 1])[-1]

    table = pd.DataFrame(
        {"AC": ac, "PAC": part, "Q": q, "Prob>Q": prob},
        index=pd.RangeIndex(1, lags + 1, name="lag"),
    )
    table.attrs["n"] = int(n)
    table.attrs["variable"] = y
    return table
