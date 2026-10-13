"""Phillips-Perron and KPSS statistics behind :func:`statspai.unitroot`.

Both correct for serial correlation with a Newey-West (Bartlett) long-run
variance instead of the lagged differences of the ADF regression.

* **Phillips-Perron** runs the Dickey-Fuller regression with no lagged
  differences and adjusts ``n (rho - 1)`` and the t statistic; the adjusted
  statistics have the Dickey-Fuller distributions under the unit root.
* **KPSS** reverses the hypotheses: the null is stationarity around a
  constant or a trend, and the statistic is the scaled sum of squared
  partial sums of the residuals from that regression.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..exceptions import DataInsufficient

#: Kwiatkowski, Phillips, Schmidt and Shin (1992), Table 1: upper-tail
#: critical values of the level- and trend-stationarity statistics.
KPSS_CV: Dict[str, Dict[str, float]] = {
    "c": {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739},
    "ct": {"10%": 0.119, "5%": 0.146, "2.5%": 0.176, "1%": 0.216},
}


def _long_run_variance(u: np.ndarray, lags: int) -> float:
    """Newey-West long-run variance with divisor ``n``."""
    n = u.size
    lrv = float(u @ u) / n
    for j in range(1, lags + 1):
        lrv += 2.0 * (1.0 - j / (lags + 1.0)) * float(u[j:] @ u[:-j]) / n
    return lrv


def _deterministics(n: int, trend: str) -> np.ndarray:
    cols: List[np.ndarray] = []
    if trend in ("c", "ct"):
        cols.append(np.ones(n))
    if trend == "ct":
        cols.append(np.arange(1.0, n + 1.0))
    return np.column_stack(cols) if cols else np.empty((n, 0))


def _checked_long_run_variance(
    u: np.ndarray, lags: int, scale: float, what: str
) -> float:
    """Long-run variance of the residuals ``u``, refusing the two cases in
    which the statistic divides by zero: residuals that vanish to rounding
    (relative to ``scale``, the size of the series), and a Bartlett sum
    that cancels (relative to the residual variance)."""
    if float(np.max(np.abs(u))) <= 1e-10 * scale:
        raise DataInsufficient(
            f"unitroot: the {what} regression fits the series exactly; with "
            "no residual variation the statistic is not defined.",
            recovery_hint="The series is a deterministic function of the "
            "terms in the regression; there is nothing to test.",
        )
    lrv = _long_run_variance(u, lags)
    gamma0 = float(u @ u) / u.size
    if lrv <= 1e-10 * gamma0:
        raise DataInsufficient(
            f"unitroot: the long-run variance of the {what} residuals is "
            f"zero to rounding with {lags} Newey-West lags; the statistic "
            "is not defined.",
            recovery_hint="Use a different number of lags.",
            diagnostics={"lags": int(lags), "residual_variance": gamma0},
        )
    return lrv


def pp_statistics(y: np.ndarray, lags: int, trend: str) -> Dict[str, Any]:
    """Phillips-Perron ``Z(rho)`` and ``Z(t)``, with the residual variance
    on ``n - k`` degrees of freedom and autocovariances on ``n`` (the
    formulas documented for Stata's ``pperron``)."""
    dep, lag = y[1:], y[:-1]
    n = dep.size
    X = np.column_stack([lag, _deterministics(n, trend)])
    k = X.shape[1]
    # rank of the design with its columns scaled to unit length, so that
    # the verdict does not depend on the units of the series
    norms = np.linalg.norm(X, axis=0)
    if n <= k or np.any(norms == 0) or np.linalg.matrix_rank(X / norms) < k:
        raise DataInsufficient(
            "unitroot: the Phillips-Perron regression is singular: the "
            "lagged series is an exact linear function of the deterministic "
            "terms (or there are too few observations).",
            recovery_hint="An exact constant or linear trend has no "
            "stochastic part to test.",
            diagnostics={"n_obs": int(n), "n_regressors": int(k)},
        )
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ dep)
    u = dep - X @ beta
    s2 = float(u @ u) / (n - k)
    se = float(np.sqrt(s2 * xtx_inv[0, 0]))
    rho = float(beta[0])
    gamma0 = float(u @ u) / n
    lam2 = _checked_long_run_variance(
        u, lags, float(np.max(np.abs(y))), "Phillips-Perron"
    )
    t_rho = (rho - 1.0) / se
    z_rho = n * (rho - 1.0) - 0.5 * (n**2 * se**2 / s2) * (lam2 - gamma0)
    z_t = np.sqrt(gamma0 / lam2) * t_rho - 0.5 * (lam2 - gamma0) * n * se / np.sqrt(
        lam2 * s2
    )
    return {
        "z_t": float(z_t),
        "z_rho": float(z_rho),
        "rho": rho,
        "se": se,
        "n": int(n),
    }


def kpss_statistic(y: np.ndarray, lags: int, trend: str) -> Dict[str, Any]:
    """KPSS statistic: ``sum(S_t^2) / (T^2 * lrv)`` with ``S`` the partial
    sums of the residuals from the regression on the deterministic terms."""
    T = y.size
    Z = _deterministics(T, trend)
    e = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    partial = np.cumsum(e)
    lrv = _checked_long_run_variance(e, lags, float(np.max(np.abs(y))), "KPSS")
    return {"stat": float(partial @ partial) / (T**2 * lrv), "n": int(T)}
