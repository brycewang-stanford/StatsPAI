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
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ARIMAResult:
    order: Tuple[int, int, int]
    seasonal_order: Optional[Tuple[int, int, int, int]]
    params: pd.Series
    aic: float
    bic: float
    aicc: float
    log_likelihood: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    n: int
    _model: object                   # statsmodels result (opaque)

    def forecast(self, horizon: int = 10, alpha: float = 0.05) -> pd.DataFrame:
        fc = self._model.get_forecast(steps=horizon)
        pred = np.asarray(fc.predicted_mean).ravel()
        ci = fc.conf_int(alpha=alpha)
        ci = np.asarray(ci)
        return pd.DataFrame({
            "forecast": pred,
            "lower": ci[:, 0] if ci.ndim == 2 else ci,
            "upper": ci[:, 1] if ci.ndim == 2 else ci,
        })

    def plot(self, horizon: int = 20, alpha: float = 0.05, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        T = np.arange(self.n)
        ax.plot(T, self.fitted_values, color="C0", linewidth=0.8, label="fitted")
        ax.plot(T, self.fitted_values + self.residuals, ".", color="grey",
                markersize=2, alpha=0.5, label="observed")
        fc = self.forecast(horizon, alpha)
        T_fc = np.arange(self.n, self.n + horizon)
        ax.plot(T_fc, fc["forecast"], "-", color="C3", label="forecast")
        ax.fill_between(T_fc, fc["lower"], fc["upper"], color="C3", alpha=0.2)
        ax.legend(); ax.set_xlabel("t"); ax.set_ylabel("y")
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
            "Parameters:",
        ]
        for nm, val in self.params.items():
            lines.append(f"  {nm:<15s}  {val: .4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def arima(
    y,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    exog=None,
    auto: bool = False,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
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
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as e:
        raise ImportError(
            "statsmodels is required for arima(). "
            "Install with `pip install statsmodels`."
        ) from e

    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    if auto:
        best_aicc = np.inf
        best_order = order
        for d in range(max_d + 1):
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        m = SARIMAX(y, order=(p, d, q),
                                    seasonal_order=seasonal_order or (0, 0, 0, 0),
                                    exog=exog, enforce_stationarity=False,
                                    enforce_invertibility=False)
                        res = m.fit(disp=False, maxiter=50)
                        k = p + q + 1 + d
                        aicc = res.aic + 2 * k * (k + 1) / max(n - k - 1, 1)
                        if aicc < best_aicc:
                            best_aicc = aicc
                            best_order = (p, d, q)
                    except Exception:
                        continue
        order = best_order

    model = SARIMAX(y, order=order,
                    seasonal_order=seasonal_order or (0, 0, 0, 0),
                    exog=exog, enforce_stationarity=False,
                    enforce_invertibility=False)
    res = model.fit(disp=False)

    k = sum(order) + 1
    aicc = res.aic + 2 * k * (k + 1) / max(n - k - 1, 1)

    _result = ARIMAResult(
        order=order,
        seasonal_order=seasonal_order,
        params=pd.Series(res.params, index=res.param_names) if hasattr(res, "param_names") else pd.Series(res.params),
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
                "seasonal_order": list(seasonal_order) if seasonal_order else None,
                "auto": auto,
                "max_p": max_p, "max_q": max_q, "max_d": max_d,
            },
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
