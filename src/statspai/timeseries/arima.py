"""ARIMA(p,d,q) / SARIMAX wrapper with automatic order selection.

Wraps ``statsmodels.tsa.statespace.SARIMAX`` to provide a StatsPAI-style
interface: formula-like specification, ``.summary()``, ``.forecast()``,
``.plot()``, and automatic (p,d,q) selection via AICc grid search.

Examples
--------
>>> import statspai as sp
>>> result = sp.arima(df["gdp"], order=(1, 1, 1))
>>> fc = result.forecast(horizon=12)
>>> result.plot()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin


@dataclass
class ARIMAResult(ResultProtocolMixin):
    """Fitted ARIMA(p,d,q) / SARIMAX model returned by :func:`statspai.arima`.

    Carries the estimated ``params`` / ``se`` (indexed by parameter name),
    information criteria (``aic`` / ``bic`` / ``aicc``), the log-likelihood,
    residuals and fitted values, plus inference accessors
    (:attr:`tvalues`, :attr:`pvalues`, :meth:`conf_int`) and a
    ``.summary()`` / ``.forecast()`` interface.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.6 * y[t - 1] + rng.normal(0, 1)
    >>> res = sp.arima(y, order=(1, 0, 0))
    >>> type(res).__name__
    'ARIMAResult'
    >>> bool(np.isfinite(res.aic))
    True
    >>> len(res.params)
    2
    """

    order: Tuple[int, int, int]
    seasonal_order: Optional[Tuple[int, int, int, int]]
    params: pd.Series
    se: pd.Series  # asymptotic standard errors (param index)
    aic: float
    bic: float
    aicc: float
    log_likelihood: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    n: int
    _model: Any  # statsmodels result (opaque)

    # --- inference accessors -------------------------------------------------
    @property
    def std_errors(self) -> pd.Series:
        """Alias for :attr:`se` (regression-style naming)."""
        return self.se

    @property
    def tvalues(self) -> pd.Series:
        """z-statistics ``params / se`` (SARIMAX uses a normal reference)."""
        return self.params / self.se

    @property
    def pvalues(self) -> pd.Series:
        """Two-sided p-values from the normal reference distribution."""
        from scipy import stats

        z = (self.params / self.se).to_numpy()
        return pd.Series(
            2.0 * stats.norm.sf(np.abs(z)),
            index=self.params.index,
        )

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """Confidence intervals for each parameter.

        Parameters
        ----------
        alpha : float, default 0.05
            ``1 - alpha`` is the coverage (0.05 → 95% CI).

        Returns
        -------
        pd.DataFrame
            Indexed by parameter name with ``lower`` / ``upper`` columns.
        """
        from scipy import stats

        z = stats.norm.ppf(1.0 - alpha / 2.0)
        lower = self.params - z * self.se
        upper = self.params + z * self.se
        return pd.DataFrame(
            {"lower": lower, "upper": upper},
            index=self.params.index,
        )

    def forecast(self, horizon: int = 10, alpha: float = 0.05) -> pd.DataFrame:
        fc = self._model.get_forecast(steps=horizon)
        pred = np.asarray(fc.predicted_mean).ravel()
        ci = fc.conf_int(alpha=alpha)
        ci = np.asarray(ci)
        return pd.DataFrame(
            {
                "forecast": pred,
                "lower": ci[:, 0] if ci.ndim == 2 else ci,
                "upper": ci[:, 1] if ci.ndim == 2 else ci,
            }
        )

    def plot(
        self,
        horizon: int = 20,
        alpha: float = 0.05,
        ax: Optional[Any] = None,
    ) -> Any:
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        T = np.arange(self.n)
        ax.plot(
            T,
            self.fitted_values,
            color="C0",
            linewidth=0.8,
            label="fitted",
        )
        ax.plot(
            T,
            self.fitted_values + self.residuals,
            ".",
            color="grey",
            markersize=2,
            alpha=0.5,
            label="observed",
        )
        fc = self.forecast(horizon, alpha)
        T_fc = np.arange(self.n, self.n + horizon)
        ax.plot(T_fc, fc["forecast"], "-", color="C3", label="forecast")
        ax.fill_between(T_fc, fc["lower"], fc["upper"], color="C3", alpha=0.2)
        ax.legend()
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        return ax

    def summary(self) -> str:
        lines = [
            f"ARIMA{self.order}"
            + (f" x {self.seasonal_order}" if self.seasonal_order else ""),
            "-" * 40,
            f"n          : {self.n}",
            f"AIC        : {self.aic:.2f}",
            f"BIC        : {self.bic:.2f}",
            f"AICc       : {self.aicc:.2f}",
            f"Log-Lik    : {self.log_likelihood:.2f}",
            "",
            (
                f"  {'':<15s}  {'coef':>10s}  {'std err':>10s}"
                f"  {'z':>8s}  {'P>|z|':>8s}"
            ),
        ]
        pvals = self.pvalues
        for nm, val in self.params.items():
            s = float(self.se.get(nm, np.nan))
            z = val / s if s and np.isfinite(s) else np.nan
            p = float(pvals.get(nm, np.nan))
            lines.append(
                f"  {nm:<15s}  {val:>10.4f}  {s:>10.4f}" f"  {z:>8.3f}  {p:>8.3f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def arima(
    y: Any,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    exog: Optional[Any] = None,
    auto: bool = False,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
    method: str = "statespace",
) -> ARIMAResult:
    """Fit ARIMA(p,d,q) or SARIMAX.

    Parameters
    ----------
    y : array-like or pd.Series
    order : (p, d, q)
    seasonal_order : (P, D, Q, s), optional
    exog : array-like, optional
        Exogenous regressors (ARIMAX).
    auto : bool, default False
        If True, select (p, d, q) by AICc grid search (ignores ``order``).
    max_p, max_q, max_d : int
        Bounds for the auto search.
    method : {'statespace', 'css_ml', 'innovations_mle'}, default 'statespace'
        Estimation convention. ``'statespace'`` keeps the default exact
        Kalman/SARIMAX likelihood. ``'css_ml'`` is retained as a compatibility
        alias for ``'innovations_mle'``. The innovations-MLE path uses
        statsmodels' stationary/invertible exact-MLE parameterization, matching
        ``stats::arima(method='ML')`` and tightly converged Stata
        ``arima`` coefficient conventions for pure ARMA models.

    Returns
    -------
    ARIMAResult
        Exposes ``params`` and the matching standard errors ``se`` (alias
        ``std_errors``), plus ``tvalues``, ``pvalues``, and
        ``conf_int(alpha)`` for inference, alongside ``aic`` / ``bic`` /
        ``aicc`` / ``log_likelihood`` and ``forecast`` / ``plot``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> e = rng.normal(size=n)
    >>> y = np.zeros(n)
    >>> for t in range(2, n):
    ...     y[t] = 0.5 * y[t - 1] - 0.2 * y[t - 2] + e[t]
    >>> gdp = pd.Series(y, name="gdp")
    >>> res = sp.arima(gdp, order=(2, 0, 0))
    >>> bool((res.se > 0).all())  # standard errors indexed by parameter name
    True
    >>> res.conf_int().shape  # 95% CIs, one row per param
    (3, 2)
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.arima.model import ARIMA as SMARIMA
    except ImportError as e:
        raise ImportError(
            "statsmodels is required for arima(). "
            "Install with `pip install statsmodels`."
        ) from e

    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    method_norm = method.lower().replace("-", "_")
    if method_norm in {"kalman", "exact", "mle"}:
        method_norm = "statespace"
    elif method_norm in {"css", "css_ml", "conditional", "conditional_mle"}:
        method_norm = "innovations_mle"
    if method_norm not in {"statespace", "innovations_mle"}:
        raise ValueError("method must be 'statespace', 'css_ml', or 'innovations_mle'")

    def _fit(
        order_: Tuple[int, int, int],
        maxiter: Optional[int] = None,
    ) -> Any:
        if method_norm == "statespace":
            model = SARIMAX(
                y,
                order=order_,
                seasonal_order=seasonal_order or (0, 0, 0, 0),
                exog=exog,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            kwargs: dict[str, Any] = {"disp": False}
            if maxiter is not None:
                kwargs["maxiter"] = maxiter
            return model.fit(**kwargs)

        model = SMARIMA(
            y,
            order=order_,
            seasonal_order=seasonal_order or (0, 0, 0, 0),
            exog=exog,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        return model.fit(method="innovations_mle")

    if auto:
        best_aicc = np.inf
        best_order = order
        for d in range(max_d + 1):
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        res = _fit((p, d, q), maxiter=50)
                        k = len(np.asarray(res.params))
                        aicc = res.aic + 2 * k * (k + 1) / max(n - k - 1, 1)
                        if aicc < best_aicc:
                            best_aicc = aicc
                            best_order = (p, d, q)
                    except Exception:
                        continue
        order = best_order

    res = _fit(order)

    k = len(np.asarray(res.params))
    aicc = res.aic + 2 * k * (k + 1) / max(n - k - 1, 1)

    _param_index = res.param_names if hasattr(res, "param_names") else None
    _params = pd.Series(res.params, index=_param_index)
    # statsmodels computes the asymptotic SEs (sqrt of the diagonal of the
    # covariance of the MLE) but we never surfaced them before; expose them.
    _bse = getattr(res, "bse", None)
    if _bse is not None:
        _se = pd.Series(np.asarray(_bse, dtype=float), index=_param_index)
    else:  # pragma: no cover - defensive; SARIMAX always populates bse
        _se = pd.Series(np.full(len(_params), np.nan), index=_param_index)

    _result = ARIMAResult(
        order=order,
        seasonal_order=seasonal_order,
        params=_params,
        se=_se,
        aic=float(res.aic),
        bic=float(res.bic),
        aicc=float(aicc),
        log_likelihood=float(res.llf),
        residuals=np.asarray(res.resid),
        fitted_values=np.asarray(res.fittedvalues),
        n=n,
        _model=res,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.timeseries.arima",
            params={
                "order": list(order),
                "seasonal_order": (list(seasonal_order) if seasonal_order else None),
                "auto": auto,
                "max_p": max_p,
                "max_q": max_q,
                "max_d": max_d,
                "method": method,
            },
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
