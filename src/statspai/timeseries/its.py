"""
Interrupted Time Series (ITS) analysis — Wagner, Soumerai, Zhang &
Ross-Degnan (2002); Bernal, Cummins & Gasparrini (2017).

For a policy / shock occurring at known time :math:`T^*`, we fit the
segmented regression

.. math::

   Y_t = \\beta_0 + \\beta_1 t + \\beta_2 D_t + \\beta_3 (t - T^*) D_t
         + \\text{seasonality}(t) + \\varepsilon_t,

where :math:`D_t = \\mathbf 1\\{t \\ge T^*\\}`. The identified quantities
are:

* :math:`\\beta_2` — **level change** at the intervention.
* :math:`\\beta_3` — **slope change** (difference-in-slopes).

This module also:

* reports Newey-West HAC standard errors for the two coefficients,
* supports optional Fourier seasonal terms of user-specified period,
* can plot the counterfactual trend extrapolated from the pre-period.

References
----------
Wagner, A. K., Soumerai, S. B., Zhang, F., & Ross-Degnan, D. (2002).
"Segmented regression analysis of interrupted time series studies in
medication use research." *Journal of Clinical Pharmacy and
Therapeutics*, 27(4), 299-309. [@wagner2002segmented]

Bernal, J. L., Cummins, S., & Gasparrini, A. (2017).
"Interrupted time series regression for the evaluation of public
health interventions: a tutorial." *International Journal of
Epidemiology*, 46(1), 348-355. [@lopezbernal2016interrupted]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ITSResult:
    level_change: float
    slope_change: float
    se_level: float
    se_slope: float
    ci_level: tuple
    ci_slope: tuple
    pvalue_level: float
    pvalue_slope: float
    coefficients: pd.DataFrame
    n_obs: int
    intervention_time: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        return (
            "Interrupted Time Series (segmented regression)\n"
            "----------------------------------------------\n"
            f"  intervention time : t = {self.intervention_time}\n"
            f"  n                 : {self.n_obs}\n"
            f"  level change      : {self.level_change:+.4f}  (SE={self.se_level:.4f}, "
            f"p={self.pvalue_level:.4f})\n"
            f"  slope change      : {self.slope_change:+.4f}  (SE={self.se_slope:.4f}, "
            f"p={self.pvalue_slope:.4f})\n\n"
            f"Coefficients:\n{self.coefficients.to_string(index=False)}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ITSResult(level={self.level_change:+.4f}, "
            f"slope={self.slope_change:+.4f})"
        )


def _newey_west_vcov(X: np.ndarray, resid: np.ndarray, L: int) -> np.ndarray:
    n = X.shape[0]
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = np.zeros((X.shape[1], X.shape[1]))
    for lag in range(L + 1):
        w = 1 - lag / (L + 1.0)
        if lag == 0:
            sub = X.T @ np.diag(resid ** 2) @ X
        else:
            xe_t = (X[lag:].T * resid[lag:]) @ ((X[:-lag].T * resid[:-lag]).T)
            sub = w * (xe_t + xe_t.T)
        S += sub
    return XtX_inv @ S @ XtX_inv


def its(
    data: pd.DataFrame,
    y: str,
    time: Optional[str] = None,
    intervention: Optional[int] = None,
    seasonality_period: Optional[int] = None,
    seasonality_harmonics: int = 2,
    hac_lag: int = 4,
    alpha: float = 0.05,
) -> ITSResult:
    """
    Segmented regression for interrupted time series.

    Parameters
    ----------
    data : pd.DataFrame
        Observed series, sorted in time.
    y : str
        Outcome column.
    time : str, optional
        Time column. If None, uses row index 0..n-1.
    intervention : int
        Time index (integer row position) at which the intervention
        begins. Required.
    seasonality_period : int, optional
        Period P of Fourier seasonal terms (e.g. 12 for monthly data
        with annual cycle). If None, no seasonality is added.
    seasonality_harmonics : int, default 2
    hac_lag : int, default 4
        Newey-West truncation lag.
    alpha : float, default 0.05

    Returns
    -------
    ITSResult
    """
    if intervention is None:
        raise ValueError("intervention (time index) is required")
    df = data.reset_index(drop=True)
    n = len(df)
    if time is None:
        t = np.arange(n, dtype=float)
    else:
        t = df[time].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)

    D = (np.arange(n) >= intervention).astype(float)
    # time since intervention (for slope change term)
    t_post = np.where(np.arange(n) >= intervention,
                       np.arange(n) - intervention, 0).astype(float)

    # Build design: [1, t, D, t_post]
    parts = [np.ones(n), t, D, t_post]
    coef_names = ["(Intercept)", "time", "level_change", "slope_change"]

    # Fourier seasonality
    if seasonality_period is not None and seasonality_period > 1:
        for k in range(1, seasonality_harmonics + 1):
            parts.append(np.sin(2 * np.pi * k * t / seasonality_period))
            parts.append(np.cos(2 * np.pi * k * t / seasonality_period))
            coef_names += [f"sin_{k}", f"cos_{k}"]

    X = np.column_stack(parts)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ beta

    vcov = _newey_west_vcov(X, resid, hac_lag)
    se = np.sqrt(np.diag(vcov))

    idx_level = coef_names.index("level_change")
    idx_slope = coef_names.index("slope_change")

    crit = float(stats.norm.ppf(1 - alpha / 2))

    level = float(beta[idx_level])
    slope = float(beta[idx_slope])
    se_level = float(se[idx_level])
    se_slope = float(se[idx_slope])
    ci_level = (level - crit * se_level, level + crit * se_level)
    ci_slope = (slope - crit * se_slope, slope + crit * se_slope)
    p_level = float(2 * stats.norm.sf(abs(level / max(se_level, 1e-12))))
    p_slope = float(2 * stats.norm.sf(abs(slope / max(se_slope, 1e-12))))

    coef_df = pd.DataFrame({
        "variable": coef_names,
        "coef": beta,
        "se": se,
        "z": beta / np.maximum(se, 1e-12),
    })

    return ITSResult(
        level_change=level,
        slope_change=slope,
        se_level=se_level,
        se_slope=se_slope,
        ci_level=ci_level,
        ci_slope=ci_slope,
        pvalue_level=p_level,
        pvalue_slope=p_slope,
        coefficients=coef_df,
        n_obs=n,
        intervention_time=int(intervention),
    )


__all__ = ["its", "ITSResult"]
