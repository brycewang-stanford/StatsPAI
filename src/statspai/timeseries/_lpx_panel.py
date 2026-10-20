"""Panel local projections.

For unit ``i`` and period ``t`` the horizon-``h`` projection is

    y_{i,t+h} [- y_{i,t-1}] = b_h s_{it} + c_h' w_{it}
                              + sum_v sum_j d_{h,v,j} v_{i,t-j}
                              + effects + e_{i,t+h},

with ``s`` the shock (its first difference with ``diff_shock=True``),
``w`` contemporaneous controls and the lags of the ``endog_order``
variables. Leads, lags and differences are taken within a unit by period,
so a unit that skips a period loses the rows that would straddle the gap.
The effects are swept out by the within transformation (unit, period or
both) or replaced by a common intercept (``panel_effect='none'``). With
``instruments=`` the shock is instrumented by two-stage least squares on
the transformed data.

Standard errors: clustered by unit with no small-sample factor (the
default), Driscoll-Kraay with Bartlett weights over periods (default lag
``floor(T^(1/4))``, ``T`` the number of periods in the horizon's sample),
or homoskedastic with the absorbed effects counted in the degrees of
freedom.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..exceptions import AssumptionWarning, DataInsufficient
from ..exceptions import warn as _warn
from . import _lpx_common as C
from ._lpx_linear import WEAK_F

PANEL_VCE = ("cluster", "driscoll_kraay", "ols")
EFFECTS = ("individual", "time", "twoways", "none")


def period_index(values: pd.Series, what: str) -> np.ndarray:
    """Integer periods: the values themselves when they are whole numbers,
    otherwise the rank of each value among the distinct sorted values."""
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        arr = values.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise C.refuse(
                f"the {what} column has missing values.",
                "Drop the rows without a period first.",
            )
        if np.all(arr == np.round(arr)):
            return np.asarray(arr, dtype=np.int64)
    if values.isna().any():
        raise C.refuse(
            f"the {what} column has missing values.",
            "Drop the rows without a period first.",
        )
    codes, _ = pd.factorize(values, sort=True)
    return np.asarray(codes, dtype=np.int64)


class PanelFrame:
    """Rows of a panel with within-unit lookups by period."""

    def __init__(self, data: pd.DataFrame, id: str, time: str, who: str) -> None:
        self.units, _ = pd.factorize(data[id])
        self.periods = period_index(data[time], f"{who} time")
        key = pd.MultiIndex.from_arrays([self.units, self.periods])
        if key.has_duplicates:
            raise C.refuse(
                f"({id}, {time}) does not identify the rows: some unit has "
                "the same period twice.",
                "Aggregate or drop the duplicate rows first.",
            )
        self._key = key

    def at(self, values: np.ndarray, offset: int) -> np.ndarray:
        """``values`` of the same unit at period ``t + offset`` (NaN if absent)."""
        if offset == 0:
            return values
        series = pd.Series(values, index=self._key)
        target = pd.MultiIndex.from_arrays([self.units, self.periods + offset])
        return np.asarray(series.reindex(target).to_numpy(dtype=float))


def _group_demean(M: np.ndarray, codes: np.ndarray) -> np.ndarray:
    _, inv = np.unique(codes, return_inverse=True)
    sums = np.zeros((inv.max() + 1, M.shape[1]))
    np.add.at(sums, inv, M)
    counts = np.bincount(inv).astype(float)
    return np.asarray(M - (sums / counts[:, None])[inv], dtype=float)


def _components(units: np.ndarray, periods: np.ndarray) -> int:
    """Connected components of the bipartite unit-period graph."""
    _, u = np.unique(units, return_inverse=True)
    _, t = np.unique(periods, return_inverse=True)
    nu, nt = u.max() + 1, t.max() + 1
    graph = coo_matrix((np.ones(len(u)), (u, nu + t)), shape=(nu + nt, nu + nt))
    return int(connected_components(graph, directed=False)[0])


def sweep_effects(
    M: np.ndarray, units: np.ndarray, periods: np.ndarray, effect: str
) -> Tuple[np.ndarray, int]:
    """Within transformation of the columns of ``M`` and the number of
    absorbed parameters."""
    n_units = len(np.unique(units))
    n_periods = len(np.unique(periods))
    if effect == "individual":
        return _group_demean(M, units), n_units
    if effect == "time":
        return _group_demean(M, periods), n_periods
    absorbed = n_units + n_periods - _components(units, periods)
    if len(M) == n_units * n_periods:  # balanced: closed form
        Mu = _group_demean(M, units)
        return _group_demean(Mu, periods), absorbed
    # unbalanced: partial out unit means, then the unit-demeaned period
    # dummies (Frisch-Waugh-Lovell; exact, unlike alternating projections)
    Mu = _group_demean(M, units)
    _, t = np.unique(periods, return_inverse=True)
    D = np.zeros((len(M), t.max() + 1))
    D[np.arange(len(M)), t] = 1.0
    Du = _group_demean(D[:, 1:], units)
    coef, *_ = np.linalg.lstsq(Du, Mu, rcond=None)
    return Mu - Du @ coef, absorbed


def fit_panel(
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
    id: str,
    time: str,
    panel_effect: str,
    diff_shock: bool,
) -> Any:
    """Panel projection; see the module docstring."""
    from .local_projections import LocalProjectionsResult

    frame = data.reset_index(drop=True)
    pf = PanelFrame(frame, id, time, C.FN)
    cols = C.as_float_columns(
        frame, [outcome, shock, *controls, *lag_variables, *instruments]
    )

    def treated(name: str) -> np.ndarray:
        v = cols[name]
        return v - pf.at(v, -1) if diff_shock else v

    rhs = [treated(shock), *[cols[c] for c in controls]]
    rhs += [pf.at(cols[v], -j) for v in lag_variables for j in lag_orders]
    names = [shock, *controls] + [
        f"L{j}.{v}" for v in lag_variables for j in lag_orders
    ]
    excl = [treated(z) for z in instruments]
    if panel_effect == "none":
        rhs = [np.ones(len(frame))] + rhs
        names = ["_cons"] + names
    i_shock = names.index(shock)
    X_all = np.column_stack(rhs)
    W_all = None
    if instruments:
        W_all = np.column_stack(
            [col for j, col in enumerate(rhs) if j != i_shock] + excl
        )
        w_names = [nm for j, nm in enumerate(names) if j != i_shock] + list(instruments)
    y = cols[outcome]
    base = pf.at(y, -1) if cumulative else 0.0
    finite = np.isfinite(X_all).all(axis=1)
    if W_all is not None:
        finite &= np.isfinite(W_all).all(axis=1)

    irf = np.empty(horizons + 1)
    se = np.empty(horizons + 1)
    n_used = np.empty(horizons + 1, dtype=int)
    units_used: List[int] = []
    f_stats: List[float] = []
    dropped: set = set()
    dk_lags: List[int] = []
    for h in range(horizons + 1):
        lhs = pf.at(y, h) - base
        m = finite & np.isfinite(lhs)
        units, periods = pf.units[m], pf.periods[m]
        n = int(m.sum())
        n_units = len(np.unique(units))
        if n == 0:
            raise DataInsufficient(
                f"{C.FN}: no complete rows at horizon {h}.",
                recovery_hint="Reduce horizons= or the lags.",
                diagnostics={"horizon": h},
            )
        if vce == "cluster" and n_units < 2:
            raise DataInsufficient(
                f"{C.FN}: horizon {h} has a single unit, and a one-cluster "
                "variance is degenerate (zero-width bands).",
                recovery_hint="Use more units, or vce='driscoll_kraay' / "
                "vce='ols' for a single unit.",
                diagnostics={"horizon": h, "n_units": n_units},
            )
        block = [X_all[m], lhs[m][:, None]]
        if W_all is not None:
            block.append(W_all[m])
        stacked = np.column_stack(block)
        if panel_effect == "none":
            swept, absorbed = stacked, 0
        else:
            swept, absorbed = sweep_effects(stacked, units, periods, panel_effect)
        kx = X_all.shape[1]
        X, yy = swept[:, :kx], swept[:, kx]
        C.check_dof(n, kx, h, absorbed)
        C.require_full_rank(X, names, h)
        lag = 0
        if vce == "driscoll_kraay":
            n_per = len(np.unique(periods))
            lag = int(np.floor(n_per**0.25)) if nw_lags is None else int(nw_lags)
            dk_lags.append(lag)
        cov_kw = dict(hac_lag=lag, groups=units, periods=periods)
        if W_all is not None:
            q = len(instruments)
            W, kept, lost = C.prune_instruments(
                swept[:, kx + 1 :],
                w_names,
                list(range(len(w_names) - q, len(w_names))),
                h,
            )
            dropped.update(lost)
            fit = C.fit_2sls(X, yy, W)
            f_stats.append(
                C.first_stage_f(
                    X[:, i_shock], W, kept, vce, absorbed=absorbed, **cov_kw
                )
            )
        else:
            fit = C.fit_ols(X, yy)
        C.check_exact_fit(fit, yy, h)
        V = C.covariance(fit, vce, resid_dof=n - kx - absorbed, **cov_kw)
        irf[h] = fit.coef[i_shock]
        se[h] = np.sqrt(max(V[i_shock, i_shock], 0.0))
        n_used[h] = n
        units_used.append(n_units)

    diagnostics: Dict[str, Any] = {
        "panel_effect": panel_effect,
        "vce": vce,
        "diff_shock": bool(diff_shock),
        "cumulative": bool(cumulative),
        "lag_variables": list(lag_variables) if lag_orders else [],
        "lags": list(lag_orders),
        "controls": list(controls),
        "units_per_horizon": units_used,
        "instruments": list(instruments),
    }
    if vce == "driscoll_kraay":
        diagnostics["dk_lags"] = dk_lags
    if instruments:
        weak = [h for h, f in enumerate(f_stats) if not f >= WEAK_F]
        diagnostics.update(
            first_stage_f=f_stats,
            weak_instrument_horizons=weak,
            dropped_instruments=sorted(dropped),
        )
        if dropped:
            _warn(
                AssumptionWarning,
                f"{C.FN}: instrument(s) {sorted(dropped)} add nothing to the "
                "included regressors and were dropped.",
                stacklevel=4,
                recovery_hint="Drop them from instruments=.",
                diagnostics={"dropped_instruments": sorted(dropped)},
            )
        if weak:
            _warn(
                AssumptionWarning,
                f"{C.FN}: first-stage F below {WEAK_F:g} at horizon(s) "
                f"{weak} (min {min(f_stats):.2f}); the panel two-stage "
                "least squares response is unreliable.",
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
        method="panel_iv" if instruments else "panel",
        diagnostics=diagnostics,
    )
