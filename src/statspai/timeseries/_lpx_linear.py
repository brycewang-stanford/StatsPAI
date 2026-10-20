"""Single-series local projections with an explicit design.

At horizon ``h`` the projection is

    y_{t+h} [- y_{t-1}] = a_h + b_h s_t + c_h' w_t
                          + sum_{v in V} sum_{j in J} d_{h,v,j} v_{t-j}
                          + trend terms + e_{t+h},

with ``s`` the shock, ``w`` contemporaneous controls, ``V`` the variables
in ``endog_order`` and ``J`` the lag orders. ``b_h`` is the response
[@jorda2005estimation]. With ``instruments=`` the shock is endogenous and
the projection is fitted by two-stage least squares, every other regressor
instrumenting itself (LP-IV).

Conventions follow ``lpirfs`` (Newey-West lag ``h + 1`` at our horizon
``h``, no small-sample factor on the sandwich, ``n - k`` divisor for the
homoskedastic variance, a sample per horizon) unless ``dfk=`` or
``common_sample=True`` ask for Stata ``lpirf``'s.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import AssumptionWarning
from ..exceptions import warn as _warn
from . import _lpx_common as C

#: first-stage F below which an instrument is flagged weak (Staiger-Stock)
WEAK_F = 10.0


def shifted(x: np.ndarray, k: int) -> np.ndarray:
    """``x`` moved by ``k`` rows: ``out[t] = x[t - k]`` (NaN outside)."""
    out = np.full(len(x), np.nan)
    if k > 0:
        out[k:] = x[:-k] if k < len(x) else out[k:]
    elif k < 0:
        out[:k] = x[-k:] if -k < len(x) else out[:k]
    else:
        out[:] = x
    return out


def lead_outcome(y: np.ndarray, h: int, cumulative: bool) -> np.ndarray:
    """``y_{t+h}``, or ``y_{t+h} - y_{t-1}`` for a cumulative response."""
    lead = shifted(y, -h)
    return lead - shifted(y, 1) if cumulative else lead


def regressor_block(
    cols: Dict[str, np.ndarray],
    *,
    shock: str,
    controls: Sequence[str],
    lag_variables: Sequence[str],
    lag_orders: Sequence[int],
    trend: int,
) -> Tuple[List[np.ndarray], List[str]]:
    """Columns of the projection other than the constant, shock first."""
    n = len(cols[shock])
    X = [cols[shock]]
    names = [shock]
    for c in controls:
        X.append(cols[c])
        names.append(c)
    for v in lag_variables:
        for j in lag_orders:
            X.append(shifted(cols[v], j))
            names.append(f"L{j}.{v}")
    for j, col in enumerate(C.trend_columns(n, trend)):
        X.append(col)
        names.append("trend" if j == 0 else "trend^2")
    return X, names


def fit_linear(
    data: pd.DataFrame,
    *,
    outcome: str,
    shock: str,
    controls: Sequence[str],
    horizons: int,
    nw_lags: Optional[int],
    alpha: float,
    cumulative: bool,
    lag_orders: Sequence[int],
    lag_variables: Sequence[str],
    instruments: Sequence[str],
    vce: str,
    dfk: Optional[bool],
    trend: int,
    common_sample: bool,
) -> Any:
    """Explicit-design projection of one series; see the module docstring."""
    from .local_projections import LocalProjectionsResult

    frame = data.reset_index(drop=True)
    cols = C.as_float_columns(
        frame, [outcome, shock, *controls, *lag_variables, *instruments]
    )
    y = cols[outcome]
    rhs, rhs_names = regressor_block(
        cols,
        shock=shock,
        controls=controls,
        lag_variables=lag_variables,
        lag_orders=lag_orders,
        trend=trend,
    )
    n_rows = len(frame)
    const = np.ones(n_rows)
    X_all = np.column_stack([const, *rhs])
    names = ["_cons", *rhs_names]
    W_all = None
    excluded: List[int] = []
    if instruments:
        W_all = np.column_stack([const, *[cols[z] for z in instruments], *rhs[1:]])
        excluded = list(range(1, 1 + len(instruments)))
        w_names = ["_cons", *instruments, *rhs_names[1:]]
    dropped: set = set()

    lhs = [lead_outcome(y, h, cumulative) for h in range(horizons + 1)]
    finite_rhs = np.isfinite(X_all).all(axis=1)
    if W_all is not None:
        finite_rhs &= np.isfinite(W_all).all(axis=1)
    masks = [finite_rhs & np.isfinite(v) for v in lhs]
    if common_sample:
        joint = np.logical_and.reduce(masks)
        masks = [joint] * len(masks)

    irf = np.empty(horizons + 1)
    se = np.empty(horizons + 1)
    n_used = np.empty(horizons + 1, dtype=int)
    f_stats: List[float] = []
    for h in range(horizons + 1):
        m = masks[h]
        X = X_all[m]
        k = X.shape[1]
        C.check_dof(int(m.sum()), k, h)
        C.require_full_rank(X, names, h)
        hac_lag = (h + 1) if nw_lags is None else int(nw_lags)
        if W_all is not None:
            W, kept, lost = C.prune_instruments(W_all[m], w_names, excluded, h)
            dropped.update(lost)
            fit = C.fit_2sls(X, lhs[h][m], W)
            f_stats.append(
                C.first_stage_f(X[:, 1], W, kept, vce, dfk=dfk, hac_lag=hac_lag)
            )
        else:
            fit = C.fit_ols(X, lhs[h][m])
        C.check_exact_fit(fit, lhs[h][m], h)
        divisor, factor = C.variance_settings(dfk, fit.n, k)
        V = C.covariance(
            fit, vce, resid_dof=divisor, hac_lag=hac_lag, sandwich_factor=factor
        )
        irf[h] = fit.coef[1]
        se[h] = np.sqrt(max(V[1, 1], 0.0))
        n_used[h] = fit.n

    diagnostics: Dict[str, Any] = {
        "design": "iv" if instruments else "ols",
        "vce": vce,
        "dfk": dfk,
        "nw_lags": (
            ("h + 1" if nw_lags is None else int(nw_lags)) if vce == "hac" else None
        ),
        "lag_variables": list(lag_variables) if lag_orders else [],
        "lags": list(lag_orders),
        "controls": list(controls),
        "trend": trend,
        "common_sample": bool(common_sample),
    }
    if instruments:
        weak = [h for h, f in enumerate(f_stats) if not f >= WEAK_F]
        diagnostics.update(
            instruments=list(instruments),
            first_stage_f=f_stats,
            weak_instrument_horizons=weak,
            dropped_instruments=sorted(dropped),
        )
        if dropped:
            _warn(
                AssumptionWarning,
                f"{C.FN}: instrument(s) {sorted(dropped)} add nothing to the "
                "included regressors (a control, a lag or a combination of "
                "them) and were dropped.",
                stacklevel=4,
                recovery_hint="Drop them from instruments=.",
                diagnostics={"dropped_instruments": sorted(dropped)},
            )
        if weak:
            _warn(
                AssumptionWarning,
                f"{C.FN}: first-stage F below {WEAK_F:g} at horizon(s) "
                f"{weak} (min {min(f_stats):.2f}); two-stage least squares "
                "is biased toward OLS and its bands are unreliable.",
                stacklevel=4,
                recovery_hint="Use a stronger instrument, or report the "
                "reduced form (the instrument as the shock).",
                diagnostics={"first_stage_f": f_stats, "weak_horizons": weak},
            )
    z = stats.norm.ppf(1 - alpha / 2)
    return LocalProjectionsResult(
        horizons=np.arange(horizons + 1),
        irf=irf,
        se=se,
        ci_lower=irf - z * se,
        ci_upper=irf + z * se,
        alpha=alpha,
        shock_name=shock,
        outcome_name=outcome,
        n_obs_per_horizon=n_used,
        method="iv" if instruments else "linear",
        diagnostics=diagnostics,
    )
