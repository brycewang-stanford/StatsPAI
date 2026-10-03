"""Unit-root tests for a single time series: augmented Dickey-Fuller and DF-GLS.

Both test ``H0: the series has a unit root`` against stationarity (around a
constant or a linear trend) with the t statistic on ``y[t-1]`` in

    dy[t] = rho * y[t-1] + sum_j b_j * dy[t-j] + deterministics + e[t]

* **ADF** runs that regression on the series itself. P-values follow
  MacKinnon (1994) and critical values MacKinnon's (2010) response surface,
  the tables already used by :func:`statspai.engle_granger` and
  :func:`statspai.panel_unitroot`.
* **DF-GLS** (Elliott, Rothenberg and Stock 1996) first removes the constant
  or trend by GLS under a local alternative, then runs the regression with
  no deterministic terms. It has more power than ADF when the root is close
  to one. Its critical values come from a response surface in the sample
  size and the lag order, fitted to simulated null distributions
  (``scripts/simulate_dfgls_critical_values.py``): the asymptotic values
  are too lenient in the samples macro data come in (with a constant the 5%
  point is -1.95 asymptotically and about -2.27 at 50 observations).

The panel counterparts are in :mod:`statspai.panel.unit_root`.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility
from ..panel.unit_root import _adf_unit, mackinnon1994_pvalue
from ._critvals import DFGLS_MIN_T, dfgls_cv, mackinnon_cv

__all__ = ["unitroot", "UnitRootResult"]

_LEVELS = (1, 5, 10)
_CBAR = {"c": -7.0, "ct": -13.5}
_TREND_LABEL = {"n": "none", "c": "constant", "ct": "constant and trend"}


class UnitRootResult(ResultProtocolMixin):
    """Outcome of :func:`unitroot`.

    Attributes
    ----------
    test : str
        ``"ADF"`` or ``"DF-GLS"``.
    statistic : float
        The t statistic on the lagged level.
    pvalue : float or None
        MacKinnon (1994) approximate p-value for ADF. ``None`` for DF-GLS,
        whose finite-sample null distribution is only tabulated at the 1%,
        5% and 10% points.
    critical_values : dict
        ``{"1%": ..., "5%": ..., "10%": ...}`` for this sample size (and,
        for DF-GLS, this lag order).
    reject : bool
        Whether the unit root is rejected at ``alpha``.
    lags : int
        Lagged differences in the test regression.
    lag_selection : str
        ``"fixed"`` or the information criterion that chose ``lags``.
    n_obs : int
        Observations in the test regression.
    rho, se : float
        Coefficient on the lagged level and its standard error.
    trend : str
        ``"n"``, ``"c"`` or ``"ct"``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> walk = pd.DataFrame({"y": np.cumsum(rng.normal(size=200))})
    >>> res = sp.unitroot(walk, "y", lags=1)
    >>> type(res).__name__
    'UnitRootResult'
    >>> sorted(res.critical_values)
    ['1%', '10%', '5%']
    >>> res.lags
    1
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "dickey1979distribution",
        "elliott1996efficient",
    )

    def __init__(
        self,
        *,
        test: str,
        statistic: float,
        pvalue: Optional[float],
        critical_values: Dict[str, float],
        reject: bool,
        alpha: float,
        lags: int,
        lag_selection: str,
        n_obs: int,
        rho: float,
        se: float,
        trend: str,
        ic_table: Optional[pd.DataFrame] = None,
    ) -> None:
        self.test = test
        self.method = f"{test} unit-root test"
        self.statistic = statistic
        self.pvalue = pvalue
        self.critical_values = critical_values
        self.reject = reject
        self.alpha = alpha
        self.lags = lags
        self.lag_selection = lag_selection
        self.n_obs = n_obs
        self.rho = rho
        self.se = se
        self.trend = trend
        self.ic_table = ic_table

    def summary(self) -> str:
        head = f"{self.test} test for a unit root"
        cv = "   ".join(f"{k}: {v:.3f}" for k, v in self.critical_values.items())
        lines = [
            head,
            "=" * len(head),
            "H0: the series has a unit root",
            f"Deterministic terms : {_TREND_LABEL[self.trend]}",
            f"Lagged differences  : {self.lags} ({self.lag_selection})",
            f"Observations        : {self.n_obs}",
            "",
            f"Test statistic      : {self.statistic:.4f}",
            f"Critical values     : {cv}",
        ]
        if self.pvalue is not None:
            lines.append(f"MacKinnon p-value   : {self.pvalue:.4f}")
        verdict = "rejected" if self.reject else "not rejected"
        lines.append(f"Unit root {verdict} at the {100 * self.alpha:g}% level.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"UnitRootResult({self.test}: statistic={self.statistic:.4f}, "
            f"lags={self.lags}, reject={self.reject})"
        )


def _series(
    data: Union[pd.DataFrame, pd.Series, np.ndarray], y: Optional[str], time: Any
) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        if y is None:
            raise MethodIncompatibility(
                "unitroot: pass y=<column> when data is a DataFrame.",
                recovery_hint="sp.unitroot(df, 'gdp')",
            )
        if y not in data.columns:
            raise MethodIncompatibility(
                f"unitroot: column {y!r} is not in data.",
                recovery_hint="Check the column name.",
                diagnostics={"columns": [str(c) for c in data.columns][:20]},
            )
        frame = data if time is None else data.sort_values(time, kind="stable")
        values = frame[y].to_numpy(dtype=float)
    else:
        values = np.asarray(data, dtype=float).ravel()
    keep = np.flatnonzero(~np.isnan(values))
    if keep.size == 0:
        raise DataInsufficient(
            "unitroot: the series has no non-missing values.",
            recovery_hint="Check the column.",
        )
    values = values[keep[0] : keep[-1] + 1]
    if np.isnan(values).any():
        raise MethodIncompatibility(
            "unitroot: the series has missing values between its first and "
            "last observation; lags are not defined across a gap.",
            recovery_hint="Fill or drop the gap deliberately, or test the "
            "longest complete stretch.",
            diagnostics={"n_missing": int(np.isnan(values).sum())},
        )
    return np.asarray(values, dtype=float)


def _gls_detrend(y: np.ndarray, trend: str) -> np.ndarray:
    """Remove the constant / trend by GLS under the local alternative."""
    T = y.size
    a = 1.0 + _CBAR[trend] / T
    z: np.ndarray = np.ones((T, 1))
    if trend == "ct":
        z = np.column_stack([np.ones(T), np.arange(1.0, T + 1.0)])
    yq = np.concatenate([y[:1], y[1:] - a * y[:-1]])
    zq = np.vstack([z[:1], z[1:] - a * z[:-1]])
    delta = np.linalg.lstsq(zq, yq, rcond=None)[0]
    return np.asarray(y - z @ delta)


def _fit(series: np.ndarray, lags: int, trend: str) -> Dict[str, Any]:
    try:
        return _adf_unit(series, lags, trend, dfcor=True)
    except np.linalg.LinAlgError as exc:
        raise DataInsufficient(
            f"unitroot: {series.size} observations are too few for a test "
            f"regression with {lags} lagged differences ({exc}).",
            recovery_hint="Use fewer lags or a longer series.",
            diagnostics={"n_obs": int(series.size), "lags": int(lags)},
        ) from exc


def _select_lags(
    series: np.ndarray, trend: str, max_lags: int, criterion: str
) -> Tuple[int, pd.DataFrame]:
    """Pick the lag order by AIC / BIC on the common sample of ``max_lags``."""
    rows = []
    for p in range(max_lags + 1):
        # drop the first (max_lags - p) points so every p uses the same rows
        fit = _fit(series[max_lags - p :], p, trend)
        n, k = fit["n"], fit["K"]
        base = np.log(fit["rss"] / n)
        rows.append(
            {
                "lags": p,
                "aic": base + 2.0 * k / n,
                "bic": base + np.log(n) * k / n,
                "n_obs": n,
            }
        )
    table = pd.DataFrame(rows).set_index("lags")
    return int(table[criterion].idxmin()), table


def unitroot(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Optional[str] = None,
    *,
    test: str = "adf",
    trend: str = "c",
    lags: Union[int, str] = "aic",
    max_lags: Optional[int] = None,
    time: Optional[str] = None,
    alpha: float = 0.05,
) -> UnitRootResult:
    """Augmented Dickey-Fuller or DF-GLS test for a unit root in one series.

    Equivalent to Stata's ``dfuller`` / ``dfgls``, R's ``urca::ur.df`` /
    ``urca::ur.ers`` and ``statsmodels.tsa.stattools.adfuller``.

    Parameters
    ----------
    data : pandas.DataFrame, pandas.Series or array
        The series, in time order (or pass ``time=``). Leading and trailing
        missing values are trimmed; a gap inside the series is an error.
    y : str, optional
        Column to test when ``data`` is a DataFrame.
    test : {"adf", "dfgls"}, default "adf"
        ``"dfgls"`` is the Elliott-Rothenberg-Stock test, more powerful when
        the largest root is near one.
    trend : {"c", "ct", "n"}, default "c"
        Deterministic terms under the alternative: a constant, a constant
        and a linear trend, or nothing (``"n"``, ADF only). Use ``"ct"``
        for a series that visibly trends.
    lags : int or {"aic", "bic"}, default "aic"
        Lagged differences in the test regression. An information criterion
        searches ``0 .. max_lags`` on the common sample those regressions
        share, then the test is run on every observation the chosen order
        allows. Stata's ``dfuller`` default is ``lags=0``.
    max_lags : int, optional
        Upper end of the search. Default: Schwert's (1989) rule
        ``floor(12 * (T / 100) ** 0.25)``.
    time : str, optional
        Column to sort a DataFrame by first.
    alpha : float, default 0.05
        Level for ``result.reject``. DF-GLS supports 0.01, 0.05 and 0.10.

    Returns
    -------
    UnitRootResult

    Raises
    ------
    MethodIncompatibility
        Unknown ``test`` / ``trend``, ``trend="n"`` with DF-GLS, a gap in the
        series, or a DF-GLS level other than 1%, 5%, 10%.
    DataInsufficient
        Too few observations for the requested lag order.

    Notes
    -----
    Rejecting means the data are unlikely under a unit root. Not rejecting
    is not evidence of a unit root: these tests have little power against a
    root of, say, 0.95 in a few decades of quarterly data.

    The ADF critical values are MacKinnon's response surface, not the
    interpolated Fuller table Stata prints, so they differ in the second
    decimal; the statistic is identical.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> e = rng.normal(size=300)
    >>> df = pd.DataFrame({"walk": np.cumsum(e)})
    >>> df["ar"] = 0.0
    >>> for t in range(1, 300):
    ...     df.loc[t, "ar"] = 0.5 * df.loc[t - 1, "ar"] + e[t]
    >>> bool(sp.unitroot(df, "walk").reject)
    False
    >>> bool(sp.unitroot(df, "ar").reject)
    True
    >>> gls = sp.unitroot(df, "ar", test="dfgls", lags=1)
    >>> gls.test
    'DF-GLS'

    References
    ----------
    [@dickey1979distribution],
    [@said1984testing],
    [@elliott1996efficient],
    [@mackinnon1994approximate],
    [@schwert1989tests]
    """
    test_key = str(test).lower().replace("-", "").replace("_", "")
    if test_key not in ("adf", "dfgls"):
        raise MethodIncompatibility(
            f"unitroot: unknown test {test!r}.",
            recovery_hint="Use test='adf' or test='dfgls'.",
        )
    if trend not in ("n", "c", "ct"):
        raise MethodIncompatibility(
            f"unitroot: unknown trend {trend!r}.",
            recovery_hint="Use 'c' (constant), 'ct' (constant and trend) or "
            "'n' (none).",
        )
    if test_key == "dfgls" and trend == "n":
        raise MethodIncompatibility(
            "unitroot: DF-GLS removes a constant or a trend by GLS; there is "
            "no trend='n' version.",
            recovery_hint="Use test='adf', trend='n', or trend='c'.",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"unitroot: alpha must be in (0, 1), got {alpha!r}.",
            recovery_hint="Use 0.01, 0.05 or 0.10.",
        )
    level = int(round(100 * alpha))
    tabulated = abs(100 * alpha - level) < 1e-9 and level in _LEVELS
    if test_key == "dfgls" and not tabulated:
        raise MethodIncompatibility(
            "unitroot: DF-GLS critical values are tabulated at the 1%, 5% "
            f"and 10% levels only, not alpha={alpha!r}.",
            recovery_hint="Use alpha=0.01, 0.05 or 0.10.",
        )

    series = _series(data, y, time)
    T = series.size
    if test_key == "dfgls":
        series = _gls_detrend(series, trend)
        reg_trend = "n"
    else:
        reg_trend = trend

    if isinstance(lags, str):
        criterion = lags.lower()
        if criterion not in ("aic", "bic"):
            raise MethodIncompatibility(
                f"unitroot: unknown lag rule {lags!r}.",
                recovery_hint="Pass an integer, 'aic' or 'bic'.",
            )
        if max_lags is None:
            max_lags = int(np.floor(12.0 * (T / 100.0) ** 0.25))
        # keep the widest regression estimable
        max_lags = max(0, min(int(max_lags), (T - 6) // 3))
        chosen, ic_table = _select_lags(series, reg_trend, max_lags, criterion)
        selection: str = criterion.upper()
    else:
        chosen, ic_table, selection = int(lags), None, "fixed"
        if chosen < 0:
            raise MethodIncompatibility(
                f"unitroot: lags must be non-negative, got {lags!r}.",
                recovery_hint="Pass lags=0 for the plain Dickey-Fuller test.",
            )

    fit = _fit(series, chosen, reg_trend)
    stat, n = float(fit["t"]), int(fit["n"])
    if test_key == "adf":
        case = {"n": "nc", "c": "c", "ct": "ct"}[trend]
        cvs = {f"{lv}%": float(mackinnon_cv(case, 1, lv, n)) for lv in _LEVELS}
        adf_p = float(mackinnon1994_pvalue(stat, trend))
        pvalue: Optional[float] = adf_p
        reject = bool(stat < cvs[f"{level}%"]) if tabulated else bool(adf_p < alpha)
        name = "ADF"
    else:
        if T < DFGLS_MIN_T or chosen > 12:
            warnings.warn(
                f"unitroot: DF-GLS critical values were fitted for series of "
                f"at least {DFGLS_MIN_T} observations and at most 12 lags; "
                f"at T={T}, lags={chosen} they are extrapolated.",
                AssumptionWarning,
                stacklevel=2,
            )
        cvs = {f"{lv}%": float(dfgls_cv(trend, lv, T, chosen)) for lv in _LEVELS}
        pvalue = None
        reject = bool(stat < cvs[f"{level}%"])
        name = "DF-GLS"

    return UnitRootResult(
        test=name,
        statistic=stat,
        pvalue=pvalue,
        critical_values=cvs,
        reject=reject,
        alpha=float(alpha),
        lags=chosen,
        lag_selection=selection,
        n_obs=n,
        rho=float(fit["rho"]),
        se=float(fit["se"]),
        trend=trend,
        ic_table=ic_table,
    )
