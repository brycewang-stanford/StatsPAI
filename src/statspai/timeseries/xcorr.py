"""Cross-correlogram of two series, raw or after prewhitening.

The cross-correlations of two autocorrelated series are hard to read: their
sampling variance depends on both autocorrelation functions, and unrelated
series can show large values. Filtering each series to white noise first
(Haugh 1976) restores the ``N(0, 1/T)`` reference, and with it a
portmanteau test of no relation at any lead or lag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["xcorr", "CrossCorrelogram"]

_Input = Union[str, pd.Series, np.ndarray, list]


@dataclass
class CrossCorrelogram(ResultProtocolMixin):
    """Cross-correlogram returned by :func:`statspai.xcorr`.

    Attributes
    ----------
    table : pandas.DataFrame
        Indexed by ``lag`` (``-lags..lags``) with columns ``xcorr``,
        ``lower``, ``upper`` and ``outside``.
    haugh : dict
        Portmanteau test of no cross-correlation at any lag in the table
        (``prewhiten='ar'`` only; empty otherwise): ``statistic``,
        ``pvalue``, ``statistic_adj``, ``pvalue_adj``, ``df``.
    n_obs : int
        Observations the correlations are computed from.
    prewhiten : str or None
    ar_orders : dict
        Order of the AR filter of each series.
    ar_coefs : dict
        Its coefficients.
    residuals : pandas.DataFrame
        The two series that were cross-correlated, about their means.
    alpha : float
    names : tuple of str

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> y = rng.normal(size=200)
    >>> x = np.r_[0.0, 0.0, y[:-2]] + 0.3 * rng.normal(size=200)
    >>> cc = sp.xcorr(x, y, lags=4)
    >>> int(cc.table["xcorr"].idxmax())    # x follows y by two periods
    2
    """

    table: pd.DataFrame
    haugh: Dict[str, Any]
    n_obs: int
    prewhiten: Optional[str]
    ar_orders: Dict[str, int]
    ar_coefs: Dict[str, np.ndarray]
    residuals: pd.DataFrame
    alpha: float
    names: Tuple[str, str] = ("x", "y")
    ar_method: str = field(default="ols")

    def summary(self) -> str:
        """Plain-text table of the cross-correlations and the test."""
        xn, yn = self.names
        head = f"Cross-correlogram of {xn} and {yn}"
        if self.prewhiten == "ar":
            head += (
                f" (each prewhitened: AR({self.ar_orders['x']}), "
                f"AR({self.ar_orders['y']}), {self.ar_method})"
            )
        elif self.prewhiten == "x":
            head += f" (both filtered with the AR({self.ar_orders['x']}) of {xn})"
        band = float(self.table["upper"].iloc[0])
        lines = [
            head,
            f"  rho(h) = corr({xn}[t+h], {yn}[t]); h > 0: {yn} leads {xn}",
            f"  observations {self.n_obs}, {100 * (1 - self.alpha):g}% band "
            f"+/-{band:.4f}",
            "",
            f"  {'lag':>4} {'xcorr':>9}",
        ]
        for lag, row in self.table.iterrows():
            flag = " *" if row["outside"] else ""
            lines.append(f"  {int(lag):>4} {row['xcorr']:>9.4f}{flag}")
        if self.haugh:
            h = self.haugh
            lines += [
                "",
                f"  Haugh test of no cross-correlation, lags -{h['lags']}.."
                f"{h['lags']}:",
                f"    S  = {h['statistic']:.4f}, p = {h['pvalue']:.4f} "
                f"(chi2({h['df']}))",
                f"    S* = {h['statistic_adj']:.4f}, p = {h['pvalue_adj']:.4f} "
                "(small-sample version)",
            ]
        return "\n".join(lines)

    def plot(self, ax: Any = None) -> Any:
        """Stem plot of the cross-correlations with the band."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3.5))
        lag = self.table.index.to_numpy()
        ax.vlines(lag, 0.0, self.table["xcorr"].to_numpy(), color="C0", lw=2)
        ax.axhline(0.0, color="black", lw=0.8)
        band = float(self.table["upper"].iloc[0])
        ax.axhline(band, color="C3", ls="--", lw=0.8)
        ax.axhline(-band, color="C3", ls="--", lw=0.8)
        xn, yn = self.names
        ax.set_xlabel(f"h  (h > 0: {yn} leads {xn})")
        ax.set_ylabel(f"corr({xn}[t+h], {yn}[t])")
        return ax


def _pair(
    x: _Input, y: _Input, data: Optional[pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, str]]:
    """The two series on their common sample without missing values."""
    cols = []
    names = []
    for arg, default in ((x, "x"), (y, "y")):
        if isinstance(arg, str):
            if data is None or arg not in data.columns:
                raise MethodIncompatibility(
                    f"sp.xcorr: {arg!r} is not a column of data.",
                    recovery_hint="Pass data= with that column, or pass "
                    "the series itself.",
                )
            cols.append(data[arg].to_numpy(dtype=float, na_value=np.nan))
            names.append(arg)
        else:
            cols.append(np.asarray(arg, dtype=float).ravel())
            name = getattr(arg, "name", None)
            names.append(str(name) if name is not None else default)
    if names[0] == names[1]:
        names = ["x", "y"]
    a, b = cols
    if a.size != b.size:
        raise MethodIncompatibility(
            f"sp.xcorr: the series have {a.size} and {b.size} observations.",
            recovery_hint="Align the two series on the same periods.",
        )
    seen = np.flatnonzero(~(np.isnan(a) | np.isnan(b)))
    if seen.size == 0:
        raise DataInsufficient(
            "sp.xcorr: no period has both series observed.",
            recovery_hint="Check the columns.",
        )
    a = a[seen[0] : seen[-1] + 1]
    b = b[seen[0] : seen[-1] + 1]
    if np.isnan(a).any() or np.isnan(b).any():
        raise MethodIncompatibility(
            "sp.xcorr: a series has missing values between observed ones; "
            "cross-correlations need consecutive periods.",
            recovery_hint="Fill the gap or use the longest complete stretch.",
        )
    return a, b, (names[0], names[1])


def _ols_ar(z: np.ndarray, p: int, start: int) -> Tuple[np.ndarray, float]:
    """AR(p) with intercept by least squares on rows ``start..``: phi, RSS."""
    n = z.size
    design = np.column_stack(
        [np.ones(n - start)] + [z[start - j : n - j] for j in range(1, p + 1)]
    )
    coef, *_ = np.linalg.lstsq(design, z[start:], rcond=None)
    resid = z[start:] - design @ coef
    return coef[1:], float(resid @ resid)


def _yw_path(z: np.ndarray, pmax: int) -> Tuple[list, np.ndarray]:
    """Levinson-Durbin: coefficients and innovation variance of each order."""
    n = z.size
    d = z - z.mean()
    r = np.array([float(d[: n - k] @ d[k:]) / n for k in range(pmax + 1)])
    phis: list = [np.zeros(0)]
    var = np.empty(pmax + 1)
    var[0] = r[0]
    phi = np.zeros(0)
    for k in range(1, pmax + 1):
        kappa = (r[k] - float(phi @ r[k - 1 : 0 : -1][: k - 1])) / var[k - 1]
        phi = np.r_[phi - kappa * phi[::-1], kappa]
        var[k] = var[k - 1] * (1.0 - kappa**2)
        phis.append(phi.copy())
    return phis, var


def _fit_ar(
    z: np.ndarray, order: Optional[int], pmax: int, method: str, ic: str
) -> np.ndarray:
    """AR coefficients of one series; the order is chosen when not given."""
    n = z.size
    if method == "yw":
        phis, var = _yw_path(z, pmax if order is None else order)
        if order is not None:
            return np.asarray(phis[order])
        pen = 2.0 if ic == "aic" else float(np.log(n))
        crit = n * np.log(var) + pen * np.arange(pmax + 1)
        return np.asarray(phis[int(np.argmin(crit))])
    if order is None:
        m = n - pmax
        pen = 2.0 if ic == "aic" else float(np.log(m))
        crit = [
            m * np.log(_ols_ar(z, p, pmax)[1] / m) + pen * (p + 1)
            for p in range(pmax + 1)
        ]
        order = int(np.argmin(crit))
    return _ols_ar(z, order, order)[0]


def _filter(z: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """``z[t] - sum phi[j] z[t-j]`` for the rows where every lag exists."""
    p = phi.size
    out = z[p:].copy()
    for j in range(1, p + 1):
        out -= phi[j - 1] * z[p - j : z.size - j]
    return np.asarray(out)


def xcorr(
    x: _Input,
    y: _Input,
    *,
    data: Optional[pd.DataFrame] = None,
    lags: Optional[int] = None,
    prewhiten: Optional[str] = None,
    ar_order: Union[None, int, Tuple[int, int]] = None,
    max_order: Optional[int] = None,
    ar_method: str = "ols",
    ic: str = "aic",
    alpha: float = 0.05,
) -> CrossCorrelogram:
    """Cross-correlations of two series at leads and lags.

    Computes ``rho(h) = corr(x[t+h], y[t])`` for ``h = -lags..lags``. A
    large value at ``h > 0`` says that ``y`` moves first and ``x`` follows
    ``h`` periods later (``y`` leads ``x``); at ``h < 0``, ``x`` leads.

    Parameters
    ----------
    x, y : str, Series or array
        The two series in time order; column names when ``data`` is given.
        Periods missing in either series at the two ends of the sample are
        dropped; a gap inside is an error.
    data : pandas.DataFrame, optional
    lags : int, optional
        Largest lead and lag. Default ``min(floor(n / 2) - 2, 20)``.
    prewhiten : {None, 'ar', 'x'}, default None
        ``None`` cross-correlates the series as they are. ``'ar'`` fits an
        autoregression to each series separately and cross-correlates the
        two residual series (Haugh 1976). ``'x'`` fits an autoregression to
        ``x`` and applies that one filter to both series (Box-Jenkins
        transfer-function identification).
    ar_order : int or (int, int), optional
        Order of the autoregressions, one number for both series or one
        for each. Default: chosen by ``ic`` among ``0..max_order``.
    max_order : int, optional
        Largest order considered. Default ``min(n - 1, floor(10 log10 n))``.
    ar_method : {'ols', 'yw'}, default 'ols'
        ``'ols'``: regression of the series on a constant and its lags.
        The order is chosen on the common sample that starts after
        ``max_order`` periods and the chosen order is then refitted on all
        rows it can use. ``'yw'``: Yule-Walker equations from the sample
        autocovariances, as R ``ar()`` does by default.
    ic : {'aic', 'bic'}, default 'aic'
    alpha : float, default 0.05
        The band is ``+/- z(1 - alpha/2) / sqrt(n)``.

    Returns
    -------
    CrossCorrelogram
        ``table`` (lag, ``xcorr``, ``lower``, ``upper``, ``outside``),
        ``haugh``, ``residuals``, ``ar_orders``, ``n_obs``, ``summary()``
        and ``plot()``.

    Raises
    ------
    MethodIncompatibility
        Unknown option, series of different length, or a gap in the data.
    DataInsufficient
        Too few observations for the lags or the AR order, or a constant
        series.

    Notes
    -----
    Sign convention. ``rho(h)`` uses the full-sample means and divisor
    ``n``: ``sum_t (x[t+h] - xbar)(y[t] - ybar) / (n s_x s_y)``. This is
    the lag-``h`` value of R ``ccf(x, y)``. Stata ``xcorr x y`` reports
    ``corr(x[t], y[t+h])`` at lag ``h``, which is ``rho(-h)`` here: its
    table is this one read from the bottom up, or equally
    ``sp.xcorr(y, x)``.

    The band is the reference for two series that are unrelated *and* of
    which at least one is white noise. For raw autocorrelated series it is
    too narrow. With ``prewhiten='ar'`` both inputs are approximately
    white noise, ``sqrt(n) rho(h)`` are asymptotically independent
    ``N(0, 1)`` across ``h`` when the series are unrelated, and

    * ``S = n * sum_{h=-M}^{M} rho(h)^2`` and
    * ``S* = n^2 * sum_{h=-M}^{M} rho(h)^2 / (n - |h|)``

    are chi-squared with ``2M + 1`` degrees of freedom, ``M = lags``.
    With ``prewhiten='x'`` the single band is valid but the values at
    different lags are correlated, so no portmanteau statistic is given.

    Yule-Walker residuals of an AR(p) match R
    ``ar(x, order.max=p, aic=FALSE)$resid`` and the order chosen by
    ``ic='aic'`` matches ``ar(x)``. Least-squares residuals of a fixed
    order match ``ar(x, method='ols', order.max=p, aic=FALSE)``; the order
    choice differs from ``ar.ols``, which compares orders fitted on
    different samples.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(3)
    >>> e = rng.normal(size=(300, 2))
    >>> y = np.zeros(300)
    >>> for t in range(1, 300):
    ...     y[t] = 0.7 * y[t - 1] + e[t, 0]
    >>> x = np.r_[0.0, y[:-1]] + e[:, 1]          # y leads x by one period
    >>> cc = sp.xcorr(x, y, lags=6, prewhiten="ar")
    >>> int(cc.table["xcorr"].abs().idxmax())
    1
    >>> bool(cc.haugh["pvalue"] < 0.01)
    True

    References
    ----------
    [@haugh1976checking],
    [@brockwell1991time],
    [@neusser2016time]
    """
    if prewhiten not in (None, "ar", "x"):
        raise MethodIncompatibility(
            f"sp.xcorr: prewhiten={prewhiten!r} is not None, 'ar' or 'x'.",
            recovery_hint="Use prewhiten='ar' to filter each series with "
            "its own autoregression.",
        )
    method = {"ols": "ols", "yw": "yw", "yule-walker": "yw"}.get(str(ar_method))
    if method is None:
        raise MethodIncompatibility(
            f"sp.xcorr: ar_method={ar_method!r} is not 'ols' or 'yw'.",
            recovery_hint="Use ar_method='ols'.",
        )
    if ic not in ("aic", "bic"):
        raise MethodIncompatibility(
            f"sp.xcorr: ic={ic!r} is not 'aic' or 'bic'.",
            recovery_hint="Use ic='aic' or ic='bic'.",
        )
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(
            f"sp.xcorr: alpha={alpha} is not between 0 and 1.",
            recovery_hint="Use alpha=0.05 for a 95% band.",
        )
    a, b, names = _pair(x, y, data)
    orders: Dict[str, int] = {}
    coefs: Dict[str, np.ndarray] = {}
    if prewhiten is not None:
        n_raw = a.size
        if ar_order is None:
            fixed: Tuple[Optional[int], Optional[int]] = (None, None)
        elif isinstance(ar_order, (int, np.integer)):
            fixed = (int(ar_order), int(ar_order))
        else:
            fixed = (int(ar_order[0]), int(ar_order[1]))
        if any(p is not None and p < 0 for p in fixed):
            raise MethodIncompatibility(
                f"sp.xcorr: ar_order={ar_order!r} has a negative order.",
                recovery_hint="Orders are non-negative integers.",
            )
        pmax = (
            min(n_raw - 1, int(np.floor(10 * np.log10(n_raw))))
            if max_order is None
            else int(max_order)
        )
        need = max([pmax if p is None else p for p in fixed])
        if n_raw < 2 * need + 10:
            raise DataInsufficient(
                f"sp.xcorr: {n_raw} observations are too few for an "
                f"autoregression of order {need}.",
                recovery_hint="Lower ar_order or max_order.",
            )
        phi_x = _fit_ar(a, fixed[0], pmax, method, ic)
        phi_y = phi_x if prewhiten == "x" else _fit_ar(b, fixed[1], pmax, method, ic)
        orders = {"x": int(phi_x.size), "y": int(phi_y.size)}
        coefs = {"x": phi_x, "y": phi_y}
        # the AR intercept or mean only shifts a residual series, and the
        # cross-correlation is computed about the mean
        ra = _filter(a - a.mean(), phi_x)
        rb = _filter(b - b.mean(), phi_y)
        common = min(ra.size, rb.size)
        a, b = ra[ra.size - common :], rb[rb.size - common :]
    n = a.size
    if lags is None:
        lags = min(n // 2 - 2, 20)
    lags = int(lags)
    if lags < 0 or lags >= n - 1:
        raise DataInsufficient(
            f"sp.xcorr: lags={lags} with {n} observations.",
            recovery_hint="Use a longer series or fewer lags.",
        )
    da, db = a - a.mean(), b - b.mean()
    scale = float(np.sqrt((da @ da) * (db @ db)))
    if scale <= 0:
        raise DataInsufficient(
            "sp.xcorr: a series is constant.",
            recovery_hint="Cross-correlations need variation in both series.",
        )
    hs = np.arange(-lags, lags + 1)
    rho = np.array(
        [
            float(da[h:] @ db[: n - h]) if h >= 0 else float(da[: n + h] @ db[-h:])
            for h in hs
        ]
    )
    rho = rho / scale
    band = float(stats.norm.ppf(1.0 - alpha / 2.0) / np.sqrt(n))
    table = pd.DataFrame(
        {
            "xcorr": rho,
            "lower": -band,
            "upper": band,
            "outside": np.abs(rho) > band,
        },
        index=pd.Index(hs, name="lag"),
    )
    haugh: Dict[str, Any] = {}
    if prewhiten == "ar":
        dof = 2 * lags + 1
        stat = float(n * (rho**2).sum())
        stat_adj = float(n**2 * (rho**2 / (n - np.abs(hs))).sum())
        haugh = {
            "statistic": stat,
            "pvalue": float(stats.chi2.sf(stat, dof)),
            "statistic_adj": stat_adj,
            "pvalue_adj": float(stats.chi2.sf(stat_adj, dof)),
            "df": dof,
            "lags": lags,
        }
    return CrossCorrelogram(
        table=table,
        haugh=haugh,
        n_obs=int(n),
        prewhiten=prewhiten,
        ar_orders=orders,
        ar_coefs=coefs,
        residuals=pd.DataFrame({names[0]: da, names[1]: db}),
        alpha=float(alpha),
        names=names,
        ar_method=method,
    )
