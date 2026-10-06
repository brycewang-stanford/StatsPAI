"""Beveridge-Nelson decomposition of an integrated series.

An I(1) series is split into a permanent component, a random walk with
drift, and a transitory component that is stationary with mean zero. The
permanent component at ``t`` is the level the series is forecast to reach
in the long run once the deterministic growth is removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["beveridge_nelson", "BeveridgeNelsonResult"]


@dataclass
class BeveridgeNelsonResult(ResultProtocolMixin):
    """Decomposition returned by :func:`statspai.beveridge_nelson`.

    Attributes
    ----------
    trend, cycle : pandas.Series
        Permanent and transitory components, aligned with the input;
        ``trend + cycle`` is the series. The first ``order`` values are
        missing.
    drift : float
        Mean of the first difference, ``intercept / (1 - sum(ar_coefs))``.
    order : int
        Order of the autoregression fitted to the first difference.
    intercept : float
    ar_coefs : numpy.ndarray
    sigma2 : float
        Innovation variance, ``RSS / (N - order - 1)``.
    long_run_multiplier : float
        ``psi(1) = 1 / (1 - sum(ar_coefs))``: the long-run response of the
        level to a unit innovation. The trend moves by
        ``drift + psi(1) * innovation`` each period.
    variance_ratio : float
        Variance of the change in the trend over the variance of the
        change in the series implied by the fitted model,
        ``psi(1)^2 / sum(psi_j^2)``.
    residuals : pandas.Series
        Innovations of the autoregression, aligned with the input.
    selection : pandas.DataFrame or None
        Information criteria by order, when the order was chosen.
    n_obs : int
        Number of first differences the autoregression was fitted to.

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

    def summary(self) -> str:
        """Plain-text report of the fitted model and the decomposition."""
        how = f"chosen by {self.ic.upper()}" if self.ic else "fixed"
        lines = [
            "Beveridge-Nelson decomposition",
            f"  model for the first difference  AR({self.order}), {how}",
            f"  observations (differences)      {self.n_obs}",
            f"  intercept                       {self.intercept:.6g}",
        ]
        for j, phi in enumerate(self.ar_coefs, start=1):
            lines.append(f"  phi[{j}]{'':<26}{phi:.6g}")
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


def beveridge_nelson(
    y: Union[str, pd.Series, np.ndarray, list],
    *,
    data: Optional[pd.DataFrame] = None,
    order: Optional[int] = None,
    max_order: Optional[int] = None,
    ic: str = "bic",
) -> BeveridgeNelsonResult:
    """Beveridge-Nelson trend and cycle of an I(1) series.

    Fits an autoregression with a constant to the first difference and
    defines the trend as the long-horizon forecast of the level net of
    deterministic growth,

    ``trend[t] = lim_h E_t(y[t+h]) - h * drift
    = y[t] + sum_{j>=1} E_t(dy[t+j] - drift)``,

    and the cycle as ``y[t] - trend[t]``.

    Parameters
    ----------
    y : str, Series or array
        The level of the series in time order (for example the log of
        GDP); a column name when ``data`` is given.
    data : pandas.DataFrame, optional
    order : int, optional
        Order ``p`` of the autoregression for the first difference.
        ``0`` is a random walk with drift, whose cycle is zero. Default:
        chosen by ``ic`` among ``0..max_order``.
    max_order : int, optional
        Largest order considered. Default ``min(8, (T - 1) // 4)``.
    ic : {'bic', 'aic'}, default 'bic'
        Criterion for the order, ``N log(RSS / N) + penalty * (p + 1)``
        with every order fitted on the same ``N`` observations (the
        differences after the first ``max_order``). The chosen order is
        refitted on all the differences it can use.

    Returns
    -------
    BeveridgeNelsonResult
        ``trend``, ``cycle``, ``drift``, ``long_run_multiplier``,
        ``variance_ratio``, the autoregression (``order``, ``intercept``,
        ``ar_coefs``, ``sigma2``, ``residuals``), ``selection``,
        ``summary()`` and ``plot()``.

    Raises
    ------
    MethodIncompatibility
        Missing values, an unknown criterion, or a fitted autoregression
        that is not stationary (the first difference is then not I(0) and
        the decomposition is not defined).
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

    Only autoregressions are fitted; a moving-average part in the first
    difference has to be approximated by a longer autoregression. The
    result depends on the order: low orders give a small, noisy cycle,
    and the decomposition of a seasonally unadjusted series puts the
    seasonal pattern in the cycle.

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
    if order is None:
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
        order = int(order)
        if order < 0:
            raise MethodIncompatibility(
                f"sp.beveridge_nelson: order={order} is negative.",
                recovery_hint="Use order=0 for a random walk with drift.",
            )
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
