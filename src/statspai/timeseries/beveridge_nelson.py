"""Beveridge-Nelson decomposition of an integrated series.

An I(1) series is split into a permanent component, a random walk with
drift, and a transitory component that is stationary with mean zero. The
permanent component at ``t`` is the level the series is forecast to reach
in the long run once the deterministic growth is removed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
    StatsPAIError,
)

__all__ = ["beveridge_nelson", "BeveridgeNelsonResult"]


@dataclass
class BeveridgeNelsonResult(ResultProtocolMixin):
    """Decomposition returned by :func:`statspai.beveridge_nelson`.

    Attributes
    ----------
    trend, cycle : pandas.Series
        Permanent and transitory components, aligned with the input;
        ``trend + cycle`` is the series. With ``method == 'ols'`` the
        first ``order`` values are missing, with ``'mle'`` the first one.
    drift : float
        Mean of the first difference, ``intercept / (1 - sum(ar_coefs))``.
    order : int
        Autoregressive order of the model for the first difference.
    ma_order : int
        Moving-average order; zero for an autoregression.
    intercept : float
        Constant of the autoregressive form, ``drift * (1 - sum(ar_coefs))``.
    ar_coefs, ma_coefs : numpy.ndarray
        ``phi_1..phi_p`` and ``theta_1..theta_q`` of
        ``phi(L) (dy - drift) = theta(L) e`` with
        ``theta(L) = 1 + theta_1 L + ...``.
    sigma2 : float
        Innovation variance: ``RSS / (N - order - 1)`` under ``'ols'``, the
        maximum-likelihood estimate under ``'mle'``.
    long_run_multiplier : float
        ``psi(1) = theta(1) / phi(1)``: the long-run response of the level
        to a unit innovation. The trend moves by
        ``drift + psi(1) * innovation`` each period.
    variance_ratio : float
        Variance of the change in the trend over the variance of the
        change in the series implied by the fitted model,
        ``psi(1)^2 / sum(psi_j^2)``.
    residuals : pandas.Series
        Innovations of the model, aligned with the input: least-squares
        residuals under ``'ols'``, one-step prediction errors of the
        exact likelihood under ``'mle'``.
    selection : pandas.DataFrame or None
        Information criteria by order, when the order was chosen.
    n_obs : int
        Number of first differences the model was fitted to.
    method : str
        ``'ols'`` (autoregression by least squares) or ``'mle'`` (ARMA by
        exact Gaussian maximum likelihood).
    loglik : float or None
        Log-likelihood under ``'mle'``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(0.5 + rng.normal(size=200))    # random walk with drift
    >>> bn = sp.beveridge_nelson(y, order=0)
    >>> float(abs(bn.cycle).max())
    0.0
    """

    trend: pd.Series
    cycle: pd.Series
    drift: float
    order: int
    intercept: float
    ar_coefs: np.ndarray
    sigma2: float
    long_run_multiplier: float
    variance_ratio: float
    residuals: pd.Series
    n_obs: int
    selection: Optional[pd.DataFrame] = field(default=None, repr=False)
    ic: Optional[str] = None
    ma_order: int = 0
    ma_coefs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    method: str = "ols"
    loglik: Optional[float] = None

    def summary(self) -> str:
        """Plain-text report of the fitted model and the decomposition."""
        how = f"chosen by {self.ic.upper()}" if self.ic else "fixed"
        model = f"AR({self.order})"
        if self.method == "mle":
            model = f"ARMA({self.order}, {self.ma_order}) by exact ML"
        lines = [
            "Beveridge-Nelson decomposition",
            f"  model for the first difference  {model}, {how}",
            f"  observations (differences)      {self.n_obs}",
            f"  intercept                       {self.intercept:.6g}",
        ]
        for j, phi in enumerate(self.ar_coefs, start=1):
            lines.append(f"  phi[{j}]{'':<26}{phi:.6g}")
        for j, theta in enumerate(self.ma_coefs, start=1):
            lines.append(f"  theta[{j}]{'':<24}{theta:.6g}")
        lines += [
            f"  innovation variance             {self.sigma2:.6g}",
            f"  drift                           {self.drift:.6g}",
            f"  long-run multiplier psi(1)      {self.long_run_multiplier:.4f}",
            f"  variance ratio (trend / series) {self.variance_ratio:.4f}",
            f"  sd of the cycle                 {float(self.cycle.std()):.6g}",
        ]
        return "\n".join(lines)

    def plot(self, ax: Any = None) -> Any:
        """The series with its trend (top) and the cycle (bottom)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        top, bottom = ax
        level = self.trend + self.cycle
        top.plot(level.index, level.to_numpy(), color="C0", label="series")
        top.plot(self.trend.index, self.trend.to_numpy(), color="C3", label="trend")
        top.legend(frameon=False)
        bottom.plot(self.cycle.index, self.cycle.to_numpy(), color="C0")
        bottom.axhline(0.0, color="black", lw=0.8)
        bottom.set_ylabel("cycle")
        return ax


def _level(
    y: Union[str, pd.Series, np.ndarray, list], data: Optional[pd.DataFrame]
) -> Tuple[np.ndarray, pd.Index, str]:
    if isinstance(y, str):
        if data is None or y not in data.columns:
            raise MethodIncompatibility(
                f"sp.beveridge_nelson: {y!r} is not a column of data.",
                recovery_hint="Pass data= with that column, or the series.",
            )
        series = data[y]
        name = y
    elif isinstance(y, pd.Series):
        series = y
        name = str(y.name) if y.name is not None else "y"
    else:
        series = pd.Series(np.asarray(y, dtype=float).ravel())
        name = "y"
    values = series.to_numpy(dtype=float, na_value=np.nan)
    if values.size and not np.isfinite(values).all():
        raise MethodIncompatibility(
            "sp.beveridge_nelson: the series has missing or infinite values.",
            recovery_hint="Pass a complete stretch of consecutive periods.",
        )
    return values, series.index, name


def _ols_ar(dy: np.ndarray, p: int, start: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """AR(p) with intercept on rows ``start..``: intercept, phi, residuals."""
    n = dy.size
    design = np.column_stack(
        [np.ones(n - start)] + [dy[start - j : n - j] for j in range(1, p + 1)]
    )
    coef, *_ = np.linalg.lstsq(design, dy[start:], rcond=None)
    return float(coef[0]), coef[1:], dy[start:] - design @ coef


def _companion(phi: np.ndarray) -> np.ndarray:
    p = phi.size
    mat = np.zeros((p, p))
    mat[0] = phi
    if p > 1:
        mat[1:, :-1] = np.eye(p - 1)
    return mat


def _arma_system(phi: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transition matrix and shock loading of an ARMA(p, q) in levels form.

    The state is ``(x[t], ..., x[t-p+1], e[t], ..., e[t-q+1])`` with
    ``x`` the demeaned difference (one lag is kept when ``p = 0``).
    """
    p, q = phi.size, theta.size
    k = max(p, 1)
    F = np.zeros((k + q, k + q))
    F[0, :p] = phi
    F[0, k:] = theta
    for i in range(1, k):
        F[i, i - 1] = 1.0
    for i in range(1, q):
        F[k + i, k + i - 1] = 1.0
    g = np.zeros(k + q)
    g[0] = 1.0
    if q:
        g[k] = 1.0
    return F, g


def _fit_arma(dy: np.ndarray, p: int, q: int) -> Dict[str, Any]:
    """ARMA(p, q) with a mean by exact ML, through :func:`statspai.arima`."""
    from .arima import arima

    fit = arima(dy, order=(p, 0, q), trend="c")
    par = fit.params
    return {
        "mean": float(par["const"]),
        "phi": np.array([par[f"ar.L{j}"] for j in range(1, p + 1)], dtype=float),
        "theta": np.array([par[f"ma.L{j}"] for j in range(1, q + 1)], dtype=float),
        "sigma2": float(fit.sigma2),
        "loglik": float(fit.log_likelihood),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
    }


def _arma_cycle(
    dy: np.ndarray, est: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Cycle, prediction errors, ``psi(1)`` and the variance ratio."""
    from ._statespace_core import stationary_cov
    from .statespace import kalman_filter

    phi, theta = est["phi"], est["theta"]
    F, g = _arma_system(phi, theta)
    radius = float(np.max(np.abs(np.linalg.eigvals(F))))
    if radius >= 1.0 - 1e-8:
        raise MethodIncompatibility(
            "sp.beveridge_nelson: the autoregressive part fitted to the "
            f"first difference is not stationary (largest root {radius:.4f}).",
            recovery_hint="The series may be I(2): difference it once "
            "more, or use another order.",
            diagnostics={"ar_coefs": phi.tolist()},
        )
    k = F.shape[0]
    Q = est["sigma2"] * np.outer(g, g)
    G = np.zeros((1, k))
    G[0, 0] = 1.0
    # the filtered state is E_t of the lags and innovations the forecasts
    # need; the first element is the datum itself
    out = kalman_filter(
        dy - est["mean"], F=F, G=G, Q=Q, R=0.0, init="stationary", smooth=False
    )
    load = -(F @ np.linalg.inv(np.eye(k) - F))[0]
    cycle = out.filtered_state @ load + 0.0
    multiplier = (1.0 + float(theta.sum())) / (1.0 - float(phi.sum()))
    gamma0 = float(stationary_cov(F, np.outer(g, g))[0, 0])
    return cycle, out.innovations[:, 0], multiplier, multiplier**2 / gamma0


def _orders(
    order: Any, max_order: Any, arma: bool
) -> Tuple[Optional[int], Optional[int], bool]:
    """``(p, q, by maximum likelihood?)`` from the ``order`` argument."""
    if order is None:
        return None, None, bool(arma)
    if isinstance(order, (tuple, list, np.ndarray)):
        if len(order) != 2:
            raise MethodIncompatibility(
                f"sp.beveridge_nelson: order={order!r} is not (p, q).",
                recovery_hint="Pass order=(p, q), or an integer for an AR(p).",
            )
        p, q = int(order[0]), int(order[1])
        ml = True
    else:
        p, q, ml = int(order), 0, bool(arma)
    if p < 0 or q < 0:
        raise MethodIncompatibility(
            f"sp.beveridge_nelson: order={order!r} is negative.",
            recovery_hint="Use order=0 for a random walk with drift.",
        )
    return p, q, ml


def _search(
    dy: np.ndarray, max_order: Any, ic: str
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Every ARMA(p, q) on the grid by exact ML; the best by ``ic``."""
    n = dy.size
    if max_order is None:
        pmax = qmax = min(3, n // 10)
    elif isinstance(max_order, (tuple, list, np.ndarray)):
        pmax, qmax = int(max_order[0]), int(max_order[1])
    else:
        pmax = qmax = int(max_order)
    if pmax < 0 or qmax < 0 or n < 2 * (pmax + qmax) + 8:
        raise DataInsufficient(
            f"sp.beveridge_nelson: max_order=({pmax}, {qmax}) with {n} " "differences.",
            recovery_hint="Lower max_order or use a longer series.",
        )
    rows: List[Dict[str, Any]] = []
    fits: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for p in range(pmax + 1):
        for q in range(qmax + 1):
            row: Dict[str, Any] = {"p": p, "q": q}
            try:
                est = _fit_arma(dy, p, q)
            except (ValueError, np.linalg.LinAlgError, StatsPAIError) as exc:
                row.update(loglik=np.nan, aic=np.inf, bic=np.inf)
                row["note"] = f"{type(exc).__name__}: {exc}"
            else:
                fits[(p, q)] = est
                row.update(loglik=est["loglik"], aic=est["aic"], bic=est["bic"])
                row["note"] = ""
            rows.append(row)
    table = pd.DataFrame(rows).set_index(["p", "q"])
    if not fits:
        raise MethodIncompatibility(
            "sp.beveridge_nelson: no ARMA model on the grid could be fitted.",
            recovery_hint="See the notes; try a fixed order=(p, q).",
            diagnostics={"notes": table["note"].tolist()},
        )
    flagged = []
    for (p, q), est in fits.items():
        nested = [fits[k]["loglik"] for k in ((p - 1, q), (p, q - 1)) if k in fits]
        if nested and est["loglik"] < max(nested) - 1e-6:
            flagged.append((p, q))
            table.loc[(p, q), "note"] = (
                "log-likelihood below a nested model: a local maximum"
            )
    if flagged:
        warnings.warn(
            f"sp.beveridge_nelson: the ARMA fits {flagged} have a lower "
            "likelihood than a model nested in them, so the optimiser "
            "stopped at a local maximum there; see .selection.",
            ConvergenceWarning,
            stacklevel=4,
        )
    best = table[ic].idxmin()
    return {**fits[best], "p": int(best[0]), "q": int(best[1])}, table


def beveridge_nelson(
    y: Union[str, pd.Series, np.ndarray, list],
    *,
    data: Optional[pd.DataFrame] = None,
    order: Optional[Union[int, Tuple[int, int]]] = None,
    max_order: Optional[Union[int, Tuple[int, int]]] = None,
    ic: str = "bic",
    arma: bool = False,
) -> BeveridgeNelsonResult:
    """Beveridge-Nelson trend and cycle of an I(1) series.

    Fits an autoregression, or an ARMA model, with a constant to the first
    difference and defines the trend as the long-horizon forecast of the
    level net of deterministic growth,

    ``trend[t] = lim_h E_t(y[t+h]) - h * drift
    = y[t] + sum_{j>=1} E_t(dy[t+j] - drift)``,

    and the cycle as ``y[t] - trend[t]``.

    Parameters
    ----------
    y : str, Series or array
        The level of the series in time order (for example the log of
        GDP); a column name when ``data`` is given.
    data : pandas.DataFrame, optional
    order : int or (int, int), optional
        An integer ``p`` fits an AR(``p``) to the first difference by
        least squares; ``0`` is a random walk with drift, whose cycle is
        zero. A pair ``(p, q)`` fits an ARMA(``p``, ``q``) by exact
        Gaussian maximum likelihood (:func:`statspai.arima`); ``(p, 0)``
        is the autoregression by maximum likelihood. Default: chosen by
        ``ic``.
    max_order : int or (int, int), optional
        Largest order considered when ``order`` is not given. For the
        autoregression, default ``min(8, (T - 1) // 4)``. With
        ``arma=True`` the search is over every ``(p, q)`` with
        ``0 <= p <= P`` and ``0 <= q <= Q``; an integer sets ``P = Q``,
        default ``min(3, (T - 1) // 10)``.
    ic : {'bic', 'aic'}, default 'bic'
        Criterion for the order. Autoregression:
        ``N log(RSS / N) + penalty * (p + 1)`` with every order fitted on
        the same ``N`` observations (the differences after the first
        ``max_order``); the chosen order is refitted on all the
        differences it can use. ARMA: the criterion of
        :func:`statspai.arima`, from the exact likelihood of all the
        differences.
    arma : bool, default False
        ``True`` switches to maximum likelihood: an integer ``order`` is
        read as ``(order, 0)``, and with no ``order`` the ARMA grid is
        searched. A pair passed as ``order`` implies it.

    Returns
    -------
    BeveridgeNelsonResult
        ``trend``, ``cycle``, ``drift``, ``long_run_multiplier``,
        ``variance_ratio``, the fitted model (``order``, ``ma_order``,
        ``intercept``, ``ar_coefs``, ``ma_coefs``, ``sigma2``,
        ``residuals``, ``method``, ``loglik``), ``selection``,
        ``summary()`` and ``plot()``.

    Raises
    ------
    MethodIncompatibility
        Missing values, an unknown criterion, or a fitted autoregressive
        part that is not stationary (the first difference is then not I(0)
        and the decomposition is not defined).
    DataInsufficient
        Too few observations for the order.

    Notes
    -----
    With ``z[t]`` the vector of the last ``p`` demeaned differences and
    ``F`` the companion matrix of the autoregression, forecasts are
    ``E_t(dy[t+j] - drift) = e1' F^j z[t]``, so

    ``cycle[t] = -e1' F (I - F)^-1 z[t]``.

    For an AR(1) this is ``-phi / (1 - phi) * (dy[t] - drift)``. The trend
    is a random walk: its change is ``drift + psi(1) * e[t]`` with
    ``psi(1) = 1 / (1 - sum(phi))``, so trend and cycle innovations are
    perfectly correlated (negatively when ``psi(1) > 1``).

    ARMA models. With ``phi(L) x[t] = theta(L) e[t]`` for the demeaned
    difference ``x``, the state ``s[t] = (x[t], ..., x[t-p+1], e[t], ...,
    e[t-q+1])`` follows ``s[t] = F s[t-1] + g e[t]`` and

    ``cycle[t] = -e1' F (I - F)^-1 E_t(s[t])``,

    with ``psi(1) = theta(1) / phi(1)``. The innovations in the state are
    not observed: ``E_t(s[t])`` is the filtered state of the Kalman filter
    started from the stationary distribution, the expectation given the
    differences up to ``t`` and nothing else. The cycle is therefore the
    exact long-horizon forecast of the fitted model at every date, and it
    is defined from the first difference on. For an ARIMA(0,1,1) it is
    ``-theta E_t(e[t])``, which tends to ``-theta`` times the one-step
    prediction error as the filter settles. ``residuals`` are those
    prediction errors (as in :func:`statspai.arima`; R's ``arima``
    reports them divided by the square root of their relative variance,
    which differs at the first dates).

    An autoregression by maximum likelihood, ``order=(p, 0)``, is not the
    integer ``order=p``: the coefficients differ (least squares conditions
    on the first ``p`` differences, the likelihood includes them), the
    drift is the estimated mean rather than ``intercept / phi(1)``,
    ``sigma2`` has no degrees-of-freedom correction, and the first ``p``
    dates get a cycle from backcast lags. The integer form is unchanged
    and remains the default.

    The result depends on the model: low orders give a small, noisy
    cycle, a moving-average root near one makes ``psi(1)`` small and puts
    nearly all of the variation in the cycle, and the decomposition of a
    seasonally unadjusted series puts the seasonal pattern in the cycle.
    The ARMA likelihood can have several local maxima; a fit on the grid
    whose likelihood is below that of a model nested in it is flagged in
    ``selection`` and a ``ConvergenceWarning`` is issued.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> e = rng.normal(size=400)
    >>> dy = np.zeros(400)
    >>> for t in range(1, 400):
    ...     dy[t] = 0.2 + 0.5 * dy[t - 1] + e[t]
    >>> bn = sp.beveridge_nelson(np.cumsum(dy), order=1)
    >>> bool(1.5 < bn.long_run_multiplier < 2.6)     # truth: 1 / (1 - 0.5)
    True
    >>> bool(np.allclose((bn.trend + bn.cycle).dropna(), np.cumsum(dy)[1:]))
    True

    An ARMA(1,1) for the difference, by exact maximum likelihood:

    >>> arma = sp.beveridge_nelson(np.cumsum(dy), order=(1, 1))
    >>> arma.method, arma.order, arma.ma_order
    ('mle', 1, 1)

    References
    ----------
    [@beveridge1981new],
    [@neusser2016time]
    """
    if ic not in ("aic", "bic"):
        raise MethodIncompatibility(
            f"sp.beveridge_nelson: ic={ic!r} is not 'aic' or 'bic'.",
            recovery_hint="Use ic='bic' or ic='aic'.",
        )
    level, index, name = _level(y, data)
    if level.size < 8:
        raise DataInsufficient(
            f"sp.beveridge_nelson: {level.size} observations are too few.",
            recovery_hint="Use a longer series.",
        )
    dy = np.diff(level)
    n = dy.size
    selection: Optional[pd.DataFrame] = None
    chosen_by: Optional[str] = None
    p_ar, q_ma, by_ml = _orders(order, max_order, arma)
    if by_ml:
        return _by_ml(level, index, name, dy, p_ar, q_ma, max_order, ic)
    if order is None:
        if isinstance(max_order, (tuple, list, np.ndarray)):
            raise MethodIncompatibility(
                "sp.beveridge_nelson: max_order=(P, Q) is the grid of the "
                "ARMA search.",
                recovery_hint="Add arma=True, or pass an integer max_order.",
            )
        pmax = min(8, n // 4) if max_order is None else int(max_order)
        if pmax < 0 or n - pmax < pmax + 3:
            raise DataInsufficient(
                f"sp.beveridge_nelson: max_order={pmax} with {n} differences.",
                recovery_hint="Lower max_order or use a longer series.",
            )
        m = n - pmax
        rows = []
        for p in range(pmax + 1):
            rss = float(np.sum(_ols_ar(dy, p, pmax)[2] ** 2))
            if rss <= 0:
                raise DataInsufficient(
                    "sp.beveridge_nelson: the first difference is fitted "
                    "without error; there is nothing to decompose.",
                    recovery_hint="Check that the series is not deterministic.",
                )
            base = m * np.log(rss / m)
            rows.append(
                {
                    "order": p,
                    "aic": base + 2.0 * (p + 1),
                    "bic": base + np.log(m) * (p + 1),
                }
            )
        selection = pd.DataFrame(rows).set_index("order")
        order = int(selection[ic].idxmin())
        chosen_by = ic
    else:
        order = int(p_ar if p_ar is not None else 0)
        if n - order < order + 3:
            raise DataInsufficient(
                f"sp.beveridge_nelson: order={order} with {n} differences.",
                recovery_hint="Lower the order or use a longer series.",
            )
    intercept, phi, resid = _ols_ar(dy, order, order)
    dof = n - order - (order + 1)
    sigma2 = float(resid @ resid) / dof
    phi_sum = float(phi.sum())
    cycle = np.full(level.size, np.nan)
    if order == 0:
        drift = intercept
        multiplier = 1.0
        ratio = 1.0
        cycle[:] = 0.0
    else:
        comp = _companion(phi)
        radius = float(np.max(np.abs(np.linalg.eigvals(comp))))
        if radius >= 1.0 - 1e-8:
            raise MethodIncompatibility(
                "sp.beveridge_nelson: the autoregression fitted to the "
                f"first difference is not stationary (largest root "
                f"{radius:.4f}).",
                recovery_hint="The series may be I(2): difference it once "
                "more, or use another order.",
                diagnostics={"ar_coefs": phi.tolist()},
            )
        drift = intercept / (1.0 - phi_sum)
        multiplier = 1.0 / (1.0 - phi_sum)
        eye = np.eye(order)
        load = -(comp @ np.linalg.inv(eye - comp))[0]
        dev = dy - drift
        # z[t] = (dev[t], dev[t-1], ..., dev[t-p+1]); dev[i] is the change
        # that ends at level index i + 1
        for i in range(order - 1, n):
            cycle[i + 1] = float(load @ dev[i - order + 1 : i + 1][::-1])
        # variance of the difference implied by the model, per unit sigma2
        unit = np.zeros(order * order)
        unit[0] = 1.0
        gamma = np.linalg.solve(np.eye(order * order) - np.kron(comp, comp), unit)[0]
        ratio = multiplier**2 / float(gamma)
    full_resid = np.full(level.size, np.nan)
    full_resid[order + 1 :] = resid
    return BeveridgeNelsonResult(
        trend=pd.Series(level - cycle, index=index, name=f"{name}_trend"),
        cycle=pd.Series(cycle, index=index, name=f"{name}_cycle"),
        drift=float(drift),
        order=order,
        intercept=intercept,
        ar_coefs=np.asarray(phi, dtype=float),
        sigma2=sigma2,
        long_run_multiplier=float(multiplier),
        variance_ratio=float(ratio),
        residuals=pd.Series(full_resid, index=index, name=f"{name}_resid"),
        n_obs=int(n - order),
        selection=selection,
        ic=chosen_by,
    )


def _by_ml(
    level: np.ndarray,
    index: pd.Index,
    name: str,
    dy: np.ndarray,
    p: Optional[int],
    q: Optional[int],
    max_order: Any,
    ic: str,
) -> BeveridgeNelsonResult:
    """The decomposition from an ARMA(p, q) fitted by exact ML."""
    n = dy.size
    selection: Optional[pd.DataFrame] = None
    chosen_by: Optional[str] = None
    if p is None or q is None:
        est, selection = _search(dy, max_order, ic)
        p, q = int(est["p"]), int(est["q"])
        chosen_by = ic
    else:
        if n < 2 * (p + q) + 8:
            raise DataInsufficient(
                f"sp.beveridge_nelson: order=({p}, {q}) with {n} differences.",
                recovery_hint="Lower the order or use a longer series.",
            )
        est = _fit_arma(dy, p, q)
    part, innov, multiplier, ratio = _arma_cycle(dy, est)
    cycle = np.full(level.size, np.nan)
    cycle[1:] = part
    resid = np.full(level.size, np.nan)
    resid[1:] = innov
    phi = est["phi"]
    return BeveridgeNelsonResult(
        trend=pd.Series(level - cycle, index=index, name=f"{name}_trend"),
        cycle=pd.Series(cycle, index=index, name=f"{name}_cycle"),
        drift=float(est["mean"]),
        order=int(p),
        intercept=float(est["mean"] * (1.0 - phi.sum())),
        ar_coefs=phi,
        sigma2=float(est["sigma2"]),
        long_run_multiplier=float(multiplier),
        variance_ratio=float(ratio),
        residuals=pd.Series(resid, index=index, name=f"{name}_resid"),
        n_obs=int(n),
        selection=selection,
        ic=chosen_by,
        ma_order=int(q),
        ma_coefs=est["theta"],
        method="mle",
        loglik=float(est["loglik"]),
    )
