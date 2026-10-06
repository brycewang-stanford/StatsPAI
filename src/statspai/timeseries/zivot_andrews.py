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
class ZivotAndrewsResult:
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
        Lagged differences in the regression.
    model : str
    path : pandas.Series
        The t statistic at every candidate break, indexed by
        ``break_index``.
    coefficients : pandas.DataFrame
        The regression at the chosen break.
    n_obs : int

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
            f"Lagged differences  : {self.lags}",
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


def _ols_t(X: np.ndarray, y: np.ndarray, col: int) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and standard errors of an OLS fit; columns that are
    numerically zero (a break dummy with no observations) are dropped and
    reported as NaN."""
    keep = np.flatnonzero(np.abs(X).sum(axis=0) > 0)
    Xk = X[:, keep]
    beta, _, rank, _ = np.linalg.lstsq(Xk, y, rcond=None)
    coef = np.full(X.shape[1], np.nan)
    se = np.full(X.shape[1], np.nan)
    if rank < Xk.shape[1] or Xk.shape[0] <= Xk.shape[1]:
        return coef, se
    resid = y - Xk @ beta
    s2 = float(resid @ resid) / (Xk.shape[0] - Xk.shape[1])
    cov = s2 * np.linalg.inv(Xk.T @ Xk)
    coef[keep] = beta
    se[keep] = np.sqrt(np.diag(cov))
    return coef, se


def zivot_andrews(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Optional[str] = None,
    *,
    model: str = "intercept",
    lags: Union[int, str] = 0,
    max_lags: Optional[int] = None,
    trim: float = 0.15,
    time: Optional[str] = None,
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
    lags : int or {'aic', 'bic'}, default 0
        Lagged differences. ``'aic'`` / ``'bic'`` choose the order once,
        from the regression without a break on the sample common to all
        orders up to ``max_lags``.
    max_lags : int, optional
        Largest order searched; default ``floor(12 (T / 100)^{1/4})``.
    trim : float, default 0.15
        Share of the sample excluded at each end from the candidate break
        dates: they run from ``floor(trim * T)`` to ``T - floor(trim *
        T)`` observations in the first regime. ``trim=0`` searches every
        date, as ``urca::ur.za`` does; the regressions at the very ends
        are poorly determined and the asymptotic theory excludes them.
    time : str, optional
        Column whose value labels the break date.

    Returns
    -------
    ZivotAndrewsResult
        ``statistic``, ``break_index``, ``break_label``,
        ``critical_values``, ``path`` (the statistic at every candidate)
        and ``summary()``.

    Raises
    ------
    MethodIncompatibility
        Unknown ``model``, ``trim`` outside ``[0, 0.5)``, a negative lag.
    DataInsufficient
        Too few observations for the lags and the trimming.

    Notes
    -----
    The null is a unit root with drift and *no* break. A rejection says
    the series is better described as stationary around a trend that
    breaks once; it does not date the break with any stated precision, and
    under the null the chosen date is not consistent for anything. The
    critical values are asymptotic and assume the break is not at the
    ends of the sample.

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

    def design(k: int, drop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Regressors without the break terms, for t = drop + 1 .. n."""
        t = np.arange(drop + 1, n + 1, dtype=float)
        dx = np.diff(x)
        cols = [np.ones(n - drop), x[drop - 1 : n - 1], t]
        cols += [dx[drop - 1 - j : n - 1 - j] for j in range(1, k + 1)]
        return np.column_stack(cols), x[drop:], t

    if isinstance(lags, str):
        if lags not in ("aic", "bic"):
            raise MethodIncompatibility(
                f"zivot_andrews: lags={lags!r} is not an integer, 'aic' or 'bic'."
            )
        top = int(np.floor(12 * (n / 100.0) ** 0.25)) if max_lags is None else max_lags
        top = max(0, min(int(top), n // 4))
        best, k_sel = np.inf, 0
        for k in range(top + 1):
            X, yy, _ = design(k, top + 1)
            resid = yy - X @ np.linalg.lstsq(X, yy, rcond=None)[0]
            m = yy.size
            pen = 2.0 if lags == "aic" else np.log(m)
            crit = m * np.log(float(resid @ resid) / m) + pen * X.shape[1]
            if crit < best:
                best, k_sel = crit, k
        k = k_sel
    else:
        k = int(lags)
        if k < 0 or k != lags:
            raise MethodIncompatibility("zivot_andrews: lags must be >= 0.")

    X0, yy, t = design(k, k + 1)
    n_reg = X0.shape[1] + (2 if model == "both" else 1)
    lo = max(int(np.floor(trim * n)), 1) if trim > 0 else 1
    hi = n - lo if trim > 0 else n - 1
    if yy.size <= n_reg + 2 or hi <= lo:
        raise DataInsufficient(
            f"zivot_andrews: {n} observations are too few for {k} lags and "
            f"trim={trim:g}.",
            recovery_hint="Use fewer lags or a longer series.",
        )
    stats_path = {}
    fits = {}
    for tb in range(lo, hi + 1):
        du = (t > tb).astype(float)
        dt = np.where(t > tb, t - tb, 0.0)
        extra = {"intercept": [du], "trend": [dt], "both": [du, dt]}[model]
        X = np.column_stack([X0] + extra)
        coef, se = _ols_t(X, yy, 1)
        stats_path[tb] = (coef[1] - 1.0) / se[1] if se[1] > 0 else np.nan
        fits[tb] = (coef, se)
    path = pd.Series(stats_path, name="t")
    path.index.name = "break_index"
    if path.notna().sum() == 0:
        raise DataInsufficient("zivot_andrews: no break date gives a regression.")
    tb_min = int(path.idxmin())
    coef, se = fits[tb_min]
    names = ["_cons", "L.y", "trend"] + [f"L{j}D.y" for j in range(1, k + 1)]
    names += {"intercept": ["du"], "trend": ["dt"], "both": ["du", "dt"]}[model]
    table = pd.DataFrame({"coef": coef, "se": se}, index=names)
    table["t"] = table["coef"] / table["se"]
    cv = dict(zip(("1%", "5%", "10%"), _CRITICAL[model]))
    return ZivotAndrewsResult(
        statistic=float(path.loc[tb_min]),
        break_index=tb_min,
        break_label=labels[tb_min - 1],
        critical_values=cv,
        lags=k,
        model=model,
        path=path,
        coefficients=table,
        n_obs=int(yy.size),
        trim=float(trim),
    )
