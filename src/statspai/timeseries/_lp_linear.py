"""Linear local projections with an explicit lag design, OLS or IV.

At horizon ``h`` the projection is

    y_{t+h} = a_h + b_h s_t + sum_{k in lags} c_{h,k}' w_{t-k}
              + d_h' x_t + trend_t + e_{t+h}

with ``w`` the variables of ``endog_order`` and ``x`` the contemporaneous
``controls``. With ``instruments`` the shock ``s_t`` is endogenous and the
projection is fitted by two-stage least squares, every other regressor
being its own instrument. ``b_h`` is the impulse response.

This is ``lpirfs::lp_lin_iv`` (``use_twosls = TRUE`` for the IV form,
``FALSE`` otherwise) with ``endog_data = endog_order``,
``lags_endog_lin = lags``, ``contemp_data = controls``; the horizon ``h``
here is ``lpirfs``'s ``h + 1`` and the default Newey-West lag is the same,
``h + 1``. With ``common_sample=True`` and ``vce='ols', dfk=False`` it is
the dynamic multiplier of Stata's ``lpirf <outcome>, exog(<shock>)``
(``dfk=True`` for its ``dfk`` option, which scales every variance type by
``n / (n - k)``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core._collinear import independent_columns
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._lp_core import lag_columns, lead, ols_fit, resolve_lags, tsls_fit


def trend_columns(n: int, trend: int) -> Dict[str, np.ndarray]:
    """``lpirfs`` trend codes: 0 none, 1 linear, 2 linear and quadratic."""
    if trend not in (0, 1, 2):
        raise MethodIncompatibility(
            f"sp.local_projections: trend={trend!r} is not 0, 1 or 2.",
            recovery_hint="0 = no trend, 1 = linear trend, 2 = linear and "
            "quadratic trend.",
        )
    t = np.arange(1, n + 1, dtype=float)
    out: Dict[str, np.ndarray] = {}
    if trend >= 1:
        out["trend"] = t
    if trend == 2:
        out["trend_sq"] = t**2
    return out


def lp_linear(
    data: pd.DataFrame,
    outcome: str,
    shock: str,
    controls: Sequence[str],
    horizons: int,
    lags: Any,
    endog_order: Optional[List[str]],
    instruments: Optional[List[str]],
    trend: int,
    cumulative: bool,
    nw_lags: Optional[int],
    vce: str,
    dfk: Optional[bool],
    alpha: float,
    common_sample: bool,
):
    """Fit the projections; returns a ``LocalProjectionsResult``."""
    from .local_projections import LocalProjectionsResult

    instruments = list(instruments or [])
    lag_vars = list(dict.fromkeys(endog_order or [outcome, shock]))
    check_columns(
        data,
        {"endog_order": lag_vars, "instruments": instruments},
        function="sp.local_projections",
    )
    if shock in instruments:
        raise MethodIncompatibility(
            "sp.local_projections: the shock is listed among its own " "instruments.",
            recovery_hint="instruments= are the excluded instruments for the "
            "endogenous shock, e.g. shock='g', instruments=['news'].",
        )
    if nw_lags is not None and nw_lags < 0:
        raise MethodIncompatibility(
            f"nw_lags must be a non-negative integer, got {nw_lags!r}.",
            recovery_hint="Leave it unset for the lpirfs rule h + 1.",
        )
    df = data.reset_index(drop=True)
    n = len(df)
    y = df[outcome].to_numpy(dtype=float)
    s = df[shock].to_numpy(dtype=float)
    exog: Dict[str, np.ndarray] = {}
    exog.update(lag_columns(df, lag_vars, resolve_lags(lags)))
    exog.update(trend_columns(n, trend))
    for c in controls:
        exog[str(c)] = df[c].to_numpy(dtype=float)
    exog_names = list(exog)
    E = np.column_stack(list(exog.values())) if exog else np.empty((n, 0))
    Zx = (
        np.column_stack([df[z].to_numpy(dtype=float) for z in instruments])
        if instruments
        else np.empty((n, 0))
    )
    y_lag1 = np.concatenate([[np.nan], y[:-1]])

    lhs_by_h = []
    valid_by_h = []
    base_ok = np.isfinite(s) & np.all(np.isfinite(E), axis=1)
    base_ok &= np.all(np.isfinite(Zx), axis=1)
    for h in range(horizons + 1):
        lhs = lead(y, h) - y_lag1 if cumulative else lead(y, h)
        lhs_by_h.append(lhs)
        valid_by_h.append(base_ok & np.isfinite(lhs))
    if common_sample:
        common = np.logical_and.reduce(valid_by_h)
        valid_by_h = [common] * (horizons + 1)

    names = ["_cons", shock] + exog_names
    irf = np.full(horizons + 1, np.nan)
    se = np.full(horizons + 1, np.nan)
    n_used = np.zeros(horizons + 1, dtype=int)
    fs_f = np.full(horizons + 1, np.nan)
    for h in range(horizons + 1):
        rows = valid_by_h[h]
        X = np.column_stack([np.ones(n), s, E])[rows]
        k = X.shape[1] + (Zx.shape[1] - 1 if instruments else 0)
        if X.shape[0] <= k + 1:
            raise DataInsufficient(
                f"sp.local_projections: {X.shape[0]} usable observations at "
                f"horizon {h} for {k} coefficients.",
                recovery_hint="Use fewer horizons, lags or controls.",
                diagnostics={"horizon": h, "n": int(X.shape[0])},
            )
        _, dependent = independent_columns(X, names)
        if dependent:
            raise MethodIncompatibility(
                f"sp.local_projections: at horizon {h} the regressors are "
                "perfectly collinear ("
                + "; ".join(f"{o['variable']}: {o['reason']}" for o in dependent)
                + "), so the coefficient on the shock is not identified.",
                recovery_hint="Drop the redundant control or lag variable.",
                diagnostics={"horizon": h, "omitted": dependent},
            )
        hac = h + 1 if nw_lags is None else int(nw_lags)
        lhs = lhs_by_h[h][rows]
        if instruments:
            Z = np.column_stack([np.ones(n), Zx, E])[rows]
            fit = tsls_fit(lhs, X, Z, vce, hac, dfk, endog_col=1)
            fs_f[h] = fit.first_stage_f
        else:
            fit = ols_fit(lhs, X, vce, hac, dfk)
        irf[h] = fit.beta[1]
        se[h] = float(np.sqrt(max(fit.vcov[1, 1], 0.0)))
        n_used[h] = fit.nobs

    z_crit = stats.norm.ppf(1 - alpha / 2)
    diagnostics: Dict[str, Any] = {
        "design": "iv" if instruments else "ols",
        "vce": vce,
        "nw_lags": "h + 1" if nw_lags is None else int(nw_lags),
        "lag_variables": lag_vars,
        "lags": resolve_lags(lags),
        "controls": list(controls),
        "trend": trend,
        "common_sample": common_sample,
    }
    if instruments:
        diagnostics["instruments"] = instruments
        diagnostics["first_stage_f"] = fs_f.tolist()
        weak = [h for h in range(horizons + 1) if fs_f[h] < 10]
        if weak:
            import warnings

            from ..exceptions import AssumptionWarning

            warnings.warn(
                "sp.local_projections: first-stage F below 10 at horizons "
                f"{weak} (min {np.nanmin(fs_f):.2f}); the IV responses and "
                "their Wald bands are unreliable under weak instruments.",
                AssumptionWarning,
                stacklevel=3,
            )
            diagnostics["weak_instrument_horizons"] = weak
    return LocalProjectionsResult(
        horizons=np.arange(horizons + 1),
        irf=irf,
        se=se,
        ci_lower=irf - z_crit * se,
        ci_upper=irf + z_crit * se,
        alpha=alpha,
        shock_name=shock,
        outcome_name=outcome,
        n_obs_per_horizon=n_used,
        method="iv" if instruments else "linear",
        diagnostics=diagnostics,
    )
