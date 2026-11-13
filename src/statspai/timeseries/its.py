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
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import require_columns
from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility


def _time_ordered(
    data: pd.DataFrame, time: Optional[str], function: str
) -> pd.DataFrame:
    """Rows of ``data`` in time order, for a single interrupted series.

    The segmented regression reads ``intervention`` as a row position of the
    time-ordered series, so an unsorted frame used to put the break in the
    wrong place. Rows are sorted on ``time`` with a stable sort; a time
    point that appears twice cannot belong to one series and is refused.
    Without ``time`` the row order is the time order.
    """
    df = data.reset_index(drop=True)
    if time is None:
        return df
    tcol = df[time]
    if tcol.isna().any():
        raise DataInsufficient(
            f"{function}: the time column {time!r} has missing values.",
            recovery_hint="Every row needs its period; drop or date those rows.",
            diagnostics={"time": time, "n_missing_time": int(tcol.isna().sum())},
        )
    dup = tcol.duplicated(keep=False) & tcol.notna()
    if dup.any():
        repeated = pd.unique(tcol[dup])[:5].tolist()
        raise MethodIncompatibility(
            f"{function}: the time column {time!r} repeats time points "
            f"(e.g. {repeated}); an interrupted time series has one "
            "observation per period.",
            recovery_hint=(
                "Aggregate to one row per period first, or fit one series at "
                "a time (for several units use a panel estimator such as "
                "sp.did or sp.synth)."
            ),
            diagnostics={"time": time, "n_duplicated_rows": int(dup.sum())},
        )
    order = np.argsort(tcol.to_numpy(), kind="stable")
    if (order == np.arange(len(order))).all():
        return df
    return df.iloc[order].reset_index(drop=True)


@dataclass
class ITSResult(ResultProtocolMixin):
    """Result container for :func:`its`.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(60)
    >>> y = 1 + 0.1 * t + (t >= 30) * 3.0 + rng.normal(scale=0.5, size=60)
    >>> df = pd.DataFrame({"y": y, "t": t})
    >>> res = sp.its(df, y="y", time="t", intervention=30)
    >>> type(res).__name__
    'ITSResult'
    >>> float(res.level_change) > 0
    True
    """

    _citation_keys = ("wagner2002segmented", "lopezbernal2016interrupted")

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
    #: value of the ``time`` column at the first post-intervention row
    intervention_label: Any = None
    #: ``'time_unit'``: what one unit of the trend and of the slope change
    #: is; ``'period_step'``: the spacing of the time column
    model_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-safe dict of every field (agent-native serialization)."""
        from .._result_serialize import result_to_dict

        return result_to_dict(self)

    def _intervention_text(self) -> str:
        row = f"row {self.intervention_time}"
        if self.intervention_label is None:
            return row
        return f"{self.intervention_label} ({row})"

    def summary(self) -> str:  # pragma: no cover
        unit = self.model_info.get("time_unit")
        note = f"\nTrend and slope change are per {unit}." if unit else ""
        return (
            "Interrupted Time Series (segmented regression)\n"
            "----------------------------------------------\n"
            f"  intervention      : {self._intervention_text()}\n"
            f"  n                 : {self.n_obs}\n"
            f"  level change      : {self.level_change:+.4f}  "
            f"(SE={self.se_level:.4f}, "
            f"p={self.pvalue_level:.4f})\n"
            f"  slope change      : {self.slope_change:+.4f}  "
            f"(SE={self.se_slope:.4f}, "
            f"p={self.pvalue_slope:.4f})\n\n"
            f"Coefficients:\n{self.coefficients.to_string(index=False)}"
            f"{note}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ITSResult(level={self.level_change:+.4f}, "
            f"slope={self.slope_change:+.4f})"
        )


def _clock(df: pd.DataFrame, time: Optional[str], who: str) -> tuple:
    """The regressor a trend is fitted on, the time labels and the unit.

    A numeric time column is used as it is (the trend is per unit of that
    column). Dates, periods and labels are not numbers a trend can be
    fitted on (a datetime is nanoseconds since 1970), so there the clock
    is the row position and the column only labels the rows.
    """
    n = len(df)
    if time is None:
        return np.arange(n, dtype=float), None, "period (row)"
    tcol = df[time]
    labels = tcol.to_numpy()
    if pd.api.types.is_numeric_dtype(tcol) and not pd.api.types.is_bool_dtype(tcol):
        return tcol.to_numpy(dtype=float), labels, f"unit of {time!r}"
    if tcol.isna().any():
        raise DataInsufficient(
            f"{who}: the time column {time!r} has missing values.",
            recovery_hint="Every row needs its period.",
            diagnostics={"n_missing_time": int(tcol.isna().sum())},
        )
    return np.arange(n, dtype=float), labels, "period (row)"


def _plain(value: Any) -> Any:
    """A time label as a plain Python value (for printing and JSON)."""
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value if isinstance(value, (int, float, str)) else str(value)


def _newey_west_vcov(X: np.ndarray, resid: np.ndarray, L: int) -> np.ndarray:
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = np.zeros((X.shape[1], X.shape[1]))
    for lag in range(L + 1):
        w = 1 - lag / (L + 1.0)
        if lag == 0:
            sub = (X * (resid**2)[:, None]).T @ X
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
    hac_small_sample: bool = False,
) -> ITSResult:
    """
    Segmented regression for interrupted time series.

    Parameters
    ----------
    data : pd.DataFrame
        Observed series, one row per period. Rows are put in time order
        (stable sort on ``time``) before fitting, so the row order of
        ``data`` does not matter when ``time`` is given; a repeated time
        point raises ``MethodIncompatibility``.
    y : str
        Outcome column.
    time : str, optional
        Time column. If None, the rows are taken to be in time order and
        the row index 0..n-1 is the time. A numeric column is the clock
        of the trend and of the slope change, so both are per unit of
        that column (per year for ``1949.0, 1949.083, ...``). A date,
        period or other non-numeric column only orders and labels the
        rows: the clock is then the row position, one unit per period.
    intervention : int
        Time index (integer row position in the time-ordered series) at
        which the intervention begins. Required.
    seasonality_period : int, optional
        Period P of Fourier seasonal terms, in observations (e.g. 12 for
        monthly data with annual cycle), whatever the unit of ``time``.
        If None, no seasonality is added.
    seasonality_harmonics : int, default 2
    hac_lag : int, default 4
        Newey-West truncation lag (Bartlett weights ``1 - j/(L+1)``).
    alpha : float, default 0.05
    hac_small_sample : bool, default False
        False: the plain Newey-West sandwich with normal inference
        (``sandwich::NeweyWest(fit, lag = L, prewhite = FALSE,
        adjust = FALSE)``). True: scale it by ``n / (n - k)`` and use
        ``t(n - k)`` for p-values and intervals -- Stata ``newey, lag(L)``
        and hence ``itsa, single ... lag(L)`` (Linden, SSC).

    Returns
    -------
    ITSResult

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(60)
    >>> y = 2 + 0.05 * t + (t >= 30) * 2.0 + rng.normal(scale=0.4, size=60)
    >>> df = pd.DataFrame({"y": y, "month": t})
    >>> res = sp.its(df, y="y", time="month", intervention=30)
    >>> round(float(res.level_change), 1) > 0
    True
    >>> res.n_obs
    60
    """
    if intervention is None:
        raise ValueError("intervention (time index) is required")
    # Column / type check up front (clear error instead of a bare KeyError).
    # NB: do *not* dropna here — ``intervention`` is an integer row position
    # into the time-ordered series, so removing rows would shift it; instead
    # we fail loudly below if the series is non-finite.
    require_columns(data, [y] + ([time] if time is not None else []), function="its")
    df = _time_ordered(data, time, "sp.its")
    n = len(df)
    n_params = 4 + (
        2 * seasonality_harmonics
        if seasonality_period is not None and seasonality_period > 1
        else 0
    )
    if n <= n_params:
        raise DataInsufficient(
            f"its: {n} observation(s) but the segmented regression estimates "
            f"{n_params} parameter(s) — need strictly more.",
            recovery_hint="Provide a longer series or drop seasonality " "harmonics.",
            diagnostics={"n_obs": int(n), "n_params": int(n_params)},
        )
    if not (0 < intervention < n):
        raise DataInsufficient(
            f"its: intervention index {intervention} leaves an empty pre- or "
            f"post-period for a series of length {n}; it must satisfy "
            "0 < intervention < n.",
            recovery_hint="Pass the integer row position where the "
            "intervention begins, with observations on both sides.",
            diagnostics={"intervention": int(intervention), "n_obs": int(n)},
        )
    t, labels, time_unit = _clock(df, time, "its")
    Y = df[y].to_numpy(dtype=float)
    if not (np.isfinite(Y).all() and np.isfinite(t).all()):
        raise DataInsufficient(
            "its: outcome/time column contains NaN or inf — segmented "
            "regression cannot fit a series with missing values.",
            recovery_hint="Impute or drop missing periods before calling "
            "sp.its (note the intervention index refers to row position).",
            diagnostics={"n_nonfinite_y": int((~np.isfinite(Y)).sum())},
        )

    D = (np.arange(n) >= intervention).astype(float)
    # time since intervention (for slope change term), on the clock of the
    # trend: with a time column that does not advance by one per row the
    # row count would put the two slopes in different units
    t_post = np.where(np.arange(n) >= intervention, t - t[intervention], 0.0)
    # one period on that clock: the Fourier terms count observations
    steps = np.diff(t)
    step = float(np.median(steps)) if steps.size else 1.0
    if not step > 0:  # pragma: no cover - duplicates are refused above
        step = 1.0

    # Build design: [1, t, D, t_post]
    parts = [np.ones(n), t, D, t_post]
    coef_names = ["(Intercept)", "time", "level_change", "slope_change"]

    # Fourier seasonality
    if seasonality_period is not None and seasonality_period > 1:
        for k in range(1, seasonality_harmonics + 1):
            parts.append(np.sin(2 * np.pi * k * (t / step) / seasonality_period))
            parts.append(np.cos(2 * np.pi * k * (t / step) / seasonality_period))
            coef_names += [f"sin_{k}", f"cos_{k}"]

    X = np.column_stack(parts)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ beta

    vcov = _newey_west_vcov(X, resid, hac_lag)
    n_obs, k_par = X.shape
    if hac_small_sample:
        vcov = vcov * n_obs / (n_obs - k_par)
    se = np.sqrt(np.diag(vcov))

    idx_level = coef_names.index("level_change")
    idx_slope = coef_names.index("slope_change")

    if hac_small_sample:
        dist = stats.t(df=n_obs - k_par)
    else:
        dist = stats.norm()
    crit = float(dist.ppf(1 - alpha / 2))

    level = float(beta[idx_level])
    slope = float(beta[idx_slope])
    se_level = float(se[idx_level])
    se_slope = float(se[idx_slope])
    ci_level = (level - crit * se_level, level + crit * se_level)
    ci_slope = (slope - crit * se_slope, slope + crit * se_slope)
    p_level = float(2 * dist.sf(abs(level / max(se_level, 1e-12))))
    p_slope = float(2 * dist.sf(abs(slope / max(se_slope, 1e-12))))

    coef_df = pd.DataFrame(
        {
            "variable": coef_names,
            "coef": beta,
            "se": se,
            "z": beta / np.maximum(se, 1e-12),
        }
    )

    # Fitted series and the no-intervention counterfactual: the counterfactual
    # is the fitted model with the level-change and slope-change contributions
    # removed (the pre-period trend and any seasonality continue unchanged).
    # This backs the documented counterfactual plot and the shared
    # ``sp.counterfactual_data`` / ``sp.counterfactual_plot`` contract.
    fitted = X @ beta
    counterfactual = fitted - beta[idx_level] * D - beta[idx_slope] * t_post
    detail = {
        "time": t if labels is None else labels,
        "observed": Y,
        "fitted": fitted,
        "counterfactual": counterfactual,
        "post": D.astype(bool),
    }

    _result = ITSResult(
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
        detail=detail,
        intervention_label=(
            None if labels is None else _plain(labels[int(intervention)])
        ),
        model_info={"time_unit": time_unit, "period_step": step},
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.timeseries.its",
            params={
                "y": y,
                "time": time,
                "intervention": intervention,
                "seasonality_period": seasonality_period,
                "seasonality_harmonics": seasonality_harmonics,
                "hac_lag": hac_lag,
                "alpha": alpha,
                "hac_small_sample": hac_small_sample,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


__all__ = ["its", "ITSResult"]
