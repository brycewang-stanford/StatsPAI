"""Zivot-Andrews unit-root test with one break at an unknown date.

A unit-root test that ignores a break in the level or the slope of the
trend mistakes the break for persistence and keeps the unit root. Perron's
remedy takes the break date as known; here it is chosen by the data, as the
date that is least favourable to the unit root, and the critical values
account for the search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["zivot_andrews", "ZivotAndrewsResult"]

# Asymptotic critical values of the minimum t statistic (1%, 5%, 10%),
# Zivot and Andrews (1992), Tables 2-4.
_CRITICAL = {
    "intercept": (-5.34, -4.80, -4.58),
    "trend": (-4.93, -4.42, -4.11),
    "both": (-5.57, -5.08, -4.82),
}


@dataclass
class ZivotAndrewsResult(ResultProtocolMixin):
    """Result of :func:`zivot_andrews`.

    Attributes
    ----------
    statistic : float
        Minimum over the candidate break dates of the t statistic of the
        lagged level.
    break_index : int
        Number of observations in the first regime: the break dummies
        switch on at observation ``break_index + 1`` (1-based).
    break_label : object
        Index label (or ``time`` value) of the last observation before the
        break.
    critical_values : dict
        Asymptotic 1%, 5% and 10% values.
    lags : int
        Lagged differences in the regression at the chosen break.
    model : str
    path : pandas.Series
        The t statistic at every candidate break, indexed by
        ``break_index``.
    coefficients : pandas.DataFrame
        The regression at the chosen break.
    n_obs : int
        Observations in the regression at the chosen break.
    lag_method : str
        ``'fixed'``, or the criterion with the scope of the choice, e.g.
        ``'bic, once'`` or ``'ttest, break'``.
    lags_path : pandas.Series or None
        With ``lag_selection='break'``, the order chosen at every
        candidate break, indexed like ``path``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(0.1 + rng.normal(size=150))   # random walk with drift
    >>> res = sp.zivot_andrews(y, model="both")
    >>> sorted(res.critical_values)
    ['1%', '10%', '5%']
    >>> len(res.path) == 150 - 2 * 22 + 1
    True
    """

    statistic: float
    break_index: int
    break_label: Any
    critical_values: Dict[str, float]
    lags: int
    model: str
    path: pd.Series
    coefficients: pd.DataFrame
    n_obs: int
    trim: float
    lag_method: str = "fixed"
    lags_path: Optional[pd.Series] = None

    @property
    def reject(self) -> Dict[str, bool]:
        """Whether the unit root is rejected at each tabulated level."""
        return {k: bool(self.statistic < v) for k, v in self.critical_values.items()}

    def summary(self) -> str:
        what = {
            "intercept": "break in the level",
            "trend": "break in the slope of the trend",
            "both": "break in the level and the slope",
        }[self.model]
        cv = self.critical_values
        verdict = "rejected" if self.reject["5%"] else "not rejected"
        lines = [
            "Zivot-Andrews test for a unit root with one break",
            "=================================================",
            "H0: unit root with drift, no break",
            f"H1: trend stationary with a {what}",
            f"Lagged differences  : {self.lags} ({self.lag_method})",
            f"Observations        : {self.n_obs}",
            f"Trimming            : {self.trim:g}",
            "",
            f"Minimum t statistic : {self.statistic:.4f}",
            f"Break after         : {self.break_label} "
            f"(observation {self.break_index})",
            f"Critical values     : 1%: {cv['1%']:.2f}   5%: {cv['5%']:.2f}"
            f"   10%: {cv['10%']:.2f}",
            f"Unit root {verdict} at the 5% level.",
        ]
        return "\n".join(lines)

    def plot(self, ax: Any = None) -> Any:
        """The t statistic over the candidate break dates."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.path.index, self.path.to_numpy(), lw=1.2)
        ax.axhline(self.critical_values["5%"], color="gray", ls="--", lw=0.8)
        ax.axvline(self.break_index, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("Observations in the first regime")
        ax.set_ylabel("t statistic of the lagged level")
        return ax

    def __repr__(self) -> str:
        return (
            f"ZivotAndrewsResult(statistic={self.statistic:.4f}, "
            f"break_index={self.break_index}, model={self.model!r})"
        )


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Coefficients, standard errors and residual sum of squares of an OLS
    fit; columns that are numerically zero (a break dummy with no
    observations) are dropped and reported as NaN. A rank-deficient fit
    returns NaN throughout."""
    keep = np.flatnonzero(np.abs(X).sum(axis=0) > 0)
    Xk = X[:, keep]
    beta, _, rank, _ = np.linalg.lstsq(Xk, y, rcond=None)
    coef = np.full(X.shape[1], np.nan)
    se = np.full(X.shape[1], np.nan)
    if rank < Xk.shape[1] or Xk.shape[0] <= Xk.shape[1]:
        return coef, se, float("nan")
    resid = y - Xk @ beta
    rss = float(resid @ resid)
    s2 = rss / (Xk.shape[0] - Xk.shape[1])
    cov = s2 * np.linalg.inv(Xk.T @ Xk)
    coef[keep] = beta
    se[keep] = np.sqrt(np.diag(cov))
    return coef, se, rss


def _ols_t(X: np.ndarray, y: np.ndarray, col: int) -> tuple[np.ndarray, np.ndarray]:
    coef, se, _ = _ols(X, y)
    return coef, se


def _bad(msg: str, hint: str) -> MethodIncompatibility:
    return MethodIncompatibility(f"zivot_andrews: {msg}", recovery_hint=hint)


def zivot_andrews(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Optional[str] = None,
    *,
    model: str = "intercept",
    lags: Union[int, str] = 0,
    max_lags: Optional[int] = None,
    trim: float = 0.15,
    time: Optional[str] = None,
    lag_selection: str = "once",
    lag_alpha: float = 0.10,
    lag_rule: str = "standard",
    trim_rule: str = "floor",
) -> ZivotAndrewsResult:
    """Zivot-Andrews unit-root test allowing one break at an unknown date.

    For every candidate break date ``TB`` the regression

    ``y_t = mu + rho y_{t-1} + beta t + theta DU_t + gamma DT_t
    + sum_j c_j Delta y_{t-j} + e_t``

    is fitted, with ``DU_t = 1(t > TB)`` and ``DT_t = (t - TB) 1(t > TB)``,
    and the statistic is the smallest t ratio of ``rho - 1``.

    Parameters
    ----------
    data : DataFrame, Series or array
        The series, in time order; a frame needs ``y``.
    y : str, optional
        Column of ``data``.
    model : {'intercept', 'trend', 'both'}, default 'intercept'
        Which break is allowed: in the level (``DU`` only; model A of the
        paper), in the slope (``DT`` only; model B) or in both (model C).
    lags : int or {'aic', 'bic', 'ttest'}, default 0
        Lagged differences. ``'aic'`` / ``'bic'`` minimise the criterion
        over the orders ``0..max_lags``. ``'ttest'`` goes from general to
        specific: starting at ``max_lags``, the last lag is dropped until
        it is significant at ``lag_alpha``. All orders are compared on the
        same observations, those available with ``max_lags`` lags; the
        test regression then uses every observation the chosen order
        allows.
    max_lags : int, optional
        Largest order searched; default ``floor(12 (T / 100)^{1/4})``
        (``floor(T^{1/4})`` with ``lag_rule='zandrews'``).
    trim : float, default 0.15
        Share of the sample excluded at each end from the candidate break
        dates: they run from ``floor(trim * T)`` to ``T - floor(trim *
        T)`` observations in the first regime. ``trim=0`` searches every
        date, as ``urca::ur.za`` does; the regressions at the very ends
        are poorly determined and the asymptotic theory excludes them.
    time : str, optional
        Column whose value labels the break date.
    lag_selection : {'once', 'break'}, default 'once'
        Where a data-driven order is chosen. ``'once'``: in the
        regression without break terms, and used at every candidate date.
        ``'break'``: afresh at every candidate date, in the regression
        that includes the break terms of that date, as in the original
        procedure of Zivot and Andrews; ``lags`` of the result is the
        order at the chosen break and ``lags_path`` has all of them.
    lag_alpha : float, default 0.10
        Significance level of ``lags='ttest'``. The last lag is kept when
        the absolute value of its t ratio reaches the two-sided standard
        normal critical value (1.645 at 0.10).
    lag_rule : {'standard', 'zandrews'}, default 'standard'
        ``'zandrews'`` computes the selection the way the Stata command
        ``zandrews`` does; see Notes. It needs ``lag_selection='once'``.
    trim_rule : {'floor', 'zandrews'}, default 'floor'
        ``'zandrews'``: with ``m = floor(trim * T + 0.49)`` and ``k``
        lags, the candidates run from ``m + k`` to ``T - m - 1``
        observations in the first regime (``k`` is ``max_lags`` when
        ``lag_selection='break'``).

    Returns
    -------
    ZivotAndrewsResult
        ``statistic``, ``break_index``, ``break_label``,
        ``critical_values``, ``lags``, ``path`` (the statistic at every
        candidate) and ``summary()``.

    Raises
    ------
    MethodIncompatibility
        Unknown ``model`` or rule, ``trim`` outside ``[0, 0.5)``, a
        negative lag, ``lag_rule='zandrews'`` with
        ``lag_selection='break'``.
    DataInsufficient
        Too few observations for the lags and the trimming.

    Notes
    -----
    The null is a unit root with drift and *no* break. A rejection says
    the series is better described as stationary around a trend that
    breaks once; it does not date the break with any stated precision, and
    under the null the chosen date is not consistent for anything. The
    critical values are asymptotic and assume the break is not at the
    ends of the sample; they are also those of a fixed lag order.

    ``lag_rule='standard'`` minimises ``N log(RSS / N) + c K`` with ``N``
    the observations of the selection regression, ``K`` its regressors
    and ``c = 2`` (AIC) or ``log N`` (BIC); the smallest order wins a tie.

    The Stata command ``zandrews`` (Baum) chooses the order once, in the
    regression without break terms. Its criteria are ``log(RSS / N) +
    c K / T`` with ``c = 2`` or ``log T`` and ``T`` the length of the
    series. Its t test keeps the last lag when the *one-sided* tail
    probability of ``|t|`` in a t distribution with ``k + 2`` degrees of
    freedom (the number of slopes of the selection regression, not its
    residual degrees of freedom) is at most ``level()``. It searches the
    orders up to ``floor(T^{1/4})`` and ignores ``maxlags()`` unless
    ``lagmethod(input)`` is given. To reproduce ``zandrews y, break(b)
    lagmethod(AIC|BIC|TTest) trim(f) level(a)`` call
    ``sp.zivot_andrews(y, model=b, lags='aic'|'bic'|'ttest', trim=f,
    lag_alpha=a, lag_rule='zandrews', trim_rule='zandrews')``, leaving
    ``max_lags`` unset, and for ``lagmethod(input) maxlags(k)`` pass
    ``lags=k, trim_rule='zandrews'``. ``zandrews`` reports the first
    observation of the new regime: ``r(tminobs) = break_index + 1``.
    Given the lag order and the candidate dates, ``zandrews``,
    ``urca::ur.za`` and this function fit the same regressions.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(200)
    >>> y = 0.05 * t + 3.0 * (t >= 120) + rng.normal(scale=0.5, size=200)
    >>> res = sp.zivot_andrews(y, model="intercept")
    >>> bool(res.reject["5%"])
    True
    >>> 115 <= res.break_index <= 125
    True

    References
    ----------
    [@zivot1992further]
    """
    if model not in _CRITICAL:
        raise MethodIncompatibility(
            f"zivot_andrews: model={model!r} is not 'intercept', 'trend' or 'both'.",
            recovery_hint="'intercept' allows a level shift, 'trend' a change "
            "of slope, 'both' the two together.",
        )
    if not 0.0 <= trim < 0.5:
        raise MethodIncompatibility("zivot_andrews: trim must be in [0, 0.5).")
    if lag_selection not in ("once", "break"):
        raise _bad(
            f"lag_selection={lag_selection!r} is not 'once' or 'break'.",
            "'once' chooses the order without break terms, 'break' at "
            "every candidate date.",
        )
    if lag_rule not in ("standard", "zandrews"):
        raise _bad(
            f"lag_rule={lag_rule!r} is not 'standard' or 'zandrews'.",
            "Use 'zandrews' only to reproduce the Stata command.",
        )
    if trim_rule not in ("floor", "zandrews"):
        raise _bad(
            f"trim_rule={trim_rule!r} is not 'floor' or 'zandrews'.",
            "Use 'zandrews' only to reproduce the Stata command.",
        )
    if not 0.0 < lag_alpha < 1.0:
        raise _bad(f"lag_alpha={lag_alpha!r} is not in (0, 1).", "Use 0.10.")
    labels: Any
    if isinstance(data, pd.DataFrame):
        if y is None or y not in data.columns:
            raise MethodIncompatibility(
                "zivot_andrews: with a DataFrame, y must name one of its columns.",
                recovery_hint="sp.zivot_andrews(df, 'gdp').",
            )
        series = data[y]
        labels = data[time].to_numpy() if time is not None else data.index.to_numpy()
    elif isinstance(data, pd.Series):
        series, labels = data, data.index.to_numpy()
    else:
        series = pd.Series(np.asarray(data, dtype=float).ravel())
        labels = series.index.to_numpy()
    x = series.to_numpy(dtype=float)
    ok = ~np.isnan(x)
    if not ok.all():
        first, last = np.flatnonzero(ok)[[0, -1]]
        x, labels = x[first : last + 1], labels[first : last + 1]
        if np.isnan(x).any():
            raise MethodIncompatibility(
                "zivot_andrews: the series has missing values inside the sample.",
                recovery_hint="Fill or drop the gaps; the test needs a "
                "regularly spaced series.",
            )
    n = x.size
    n_break = 2 if model == "both" else 1

    def design(k: int, drop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Regressors without the break terms, for t = drop + 1 .. n."""
        t = np.arange(drop + 1, n + 1, dtype=float)
        dx = np.diff(x)
        cols = [np.ones(n - drop), x[drop - 1 : n - 1], t]
        cols += [dx[drop - 1 - j : n - 1 - j] for j in range(1, k + 1)]
        return np.column_stack(cols), x[drop:], t

    def with_break(X: np.ndarray, t: np.ndarray, tb: int) -> np.ndarray:
        du = (t > tb).astype(float)
        dt = np.where(t > tb, t - tb, 0.0)
        extra = {"intercept": [du], "trend": [dt], "both": [du, dt]}[model]
        return np.column_stack([X] + extra)

    zandrews = lag_rule == "zandrews"
    top = 0

    def choose(tb: Optional[int]) -> int:
        """Order by the criterion, all orders on the sample of ``top``."""
        fits = {}
        for k in range(top + 1):
            X, yy, t = design(k, top + 1)
            if tb is not None:
                X = with_break(X, t, tb)
            coef, se, rss = _ols(X, yy)
            last = abs(coef[2 + k] / se[2 + k]) if k and se[2 + k] > 0 else np.nan
            fits[k] = (rss, yy.size, X.shape[1], last)
        if lags == "ttest":
            from scipy import stats

            for k in range(top, 0, -1):
                last = fits[k][3]
                if not np.isfinite(last):
                    continue
                if zandrews:
                    keep = stats.t.sf(last, k + 2) <= lag_alpha
                else:
                    keep = last >= stats.norm.isf(lag_alpha / 2.0)
                if keep:
                    return k
            return 0
        best, k_sel = np.inf, 0
        # zandrews compares order 0 first, then max_lags down to 1
        order = [0] + list(range(top, 0, -1)) if zandrews else list(range(top + 1))
        for k in order:
            rss, m, n_reg, _ = fits[k]
            if not rss > 0:
                continue
            if zandrews:
                pen = 2.0 if lags == "aic" else np.log(n)
                crit = np.log(rss / m) + pen * n_reg / n
            else:
                pen = 2.0 if lags == "aic" else np.log(m)
                crit = m * np.log(rss / m) + pen * n_reg
            if crit < best:
                best, k_sel = crit, k
        return k_sel

    data_driven = isinstance(lags, str)
    k_fixed: Optional[int] = None
    if isinstance(lags, str):
        if lags not in ("aic", "bic", "ttest"):
            raise MethodIncompatibility(
                f"zivot_andrews: lags={lags!r} is not an integer, 'aic', "
                "'bic' or 'ttest'."
            )
        if zandrews and lag_selection == "break":
            raise _bad(
                "lag_rule='zandrews' chooses the order once.",
                "Use lag_selection='once', or lag_rule='standard'.",
            )
        if max_lags is not None and (max_lags < 0 or int(max_lags) != max_lags):
            raise _bad(
                f"max_lags={max_lags!r} is not a non-negative integer.",
                "Use e.g. max_lags=8.",
            )
        if zandrews:
            top = int(n**0.25) if max_lags is None else int(max_lags)
        else:
            top = (
                int(np.floor(12 * (n / 100.0) ** 0.25))
                if max_lags is None
                else int(max_lags)
            )
            top = max(0, min(top, n // 4))
        if n - top - 1 <= top + 3 + n_break + 2:
            raise DataInsufficient(
                f"zivot_andrews: {n} observations are too few to compare "
                f"lag orders up to {top}.",
                recovery_hint="Lower max_lags or use a longer series.",
            )
        if lag_selection == "once":
            k_fixed = choose(None)
        lag_method = f"{lags}, {lag_selection}"
    else:
        k_fixed = int(lags)
        if k_fixed < 0 or k_fixed != lags:
            raise MethodIncompatibility("zivot_andrews: lags must be >= 0.")
        lag_method = "fixed"

    k_edge = top if k_fixed is None else k_fixed
    if trim_rule == "zandrews":
        cut = int(np.floor(trim * n + 0.49))
        lo, hi = max(cut + k_edge, 1), n - cut - 1
    else:
        lo = max(int(np.floor(trim * n)), 1) if trim > 0 else 1
        hi = n - lo if trim > 0 else n - 1
    if n - k_edge - 1 <= 3 + k_edge + n_break + 2 or hi <= lo:
        raise DataInsufficient(
            f"zivot_andrews: {n} observations are too few for {k_edge} lags and "
            f"trim={trim:g}.",
            recovery_hint="Use fewer lags or a longer series.",
        )
    stats_path = {}
    fits = {}
    base: Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for tb in range(lo, hi + 1):
        k = choose(tb) if k_fixed is None else k_fixed
        if k not in base:
            base[k] = design(k, k + 1)
        X0, yy, t = base[k]
        coef, se, _ = _ols(with_break(X0, t, tb), yy)
        stats_path[tb] = (coef[1] - 1.0) / se[1] if se[1] > 0 else np.nan
        fits[tb] = (coef, se, k, yy.size)
    path = pd.Series(stats_path, name="t")
    path.index.name = "break_index"
    if path.notna().sum() == 0:
        raise DataInsufficient("zivot_andrews: no break date gives a regression.")
    tb_min = int(path.idxmin())
    coef, se, k, n_used = fits[tb_min]
    names = ["_cons", "L.y", "trend"] + [f"L{j}D.y" for j in range(1, k + 1)]
    names += {"intercept": ["du"], "trend": ["dt"], "both": ["du", "dt"]}[model]
    table = pd.DataFrame({"coef": coef, "se": se}, index=names)
    table["t"] = table["coef"] / table["se"]
    cv = dict(zip(("1%", "5%", "10%"), _CRITICAL[model]))
    lags_path = None
    if data_driven and lag_selection == "break":
        lags_path = pd.Series({tb: f[2] for tb, f in fits.items()}, name="lags")
        lags_path.index.name = "break_index"
    return ZivotAndrewsResult(
        statistic=float(path.loc[tb_min]),
        break_index=tb_min,
        break_label=labels[tb_min - 1],
        critical_values=cv,
        lags=k,
        model=model,
        path=path,
        coefficients=table,
        n_obs=int(n_used),
        trim=float(trim),
        lag_method=lag_method,
        lags_path=lags_path,
    )
