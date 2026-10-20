"""State-dependent local projections.

A regime weight ``F_t`` in [0, 1] splits the projection in two. With an
observed shock ``s`` the horizon-``h`` regression is

    y_{t+h} = a_h + (1 - F_{t-1}) [b0_h s_t + lags]
                  + F_{t-1} [b1_h s_t + lags] + c_h' w_t + trend + e,

so ``b1_h`` is the response in the regime ``F = 1`` and ``b0_h`` in
``F = 0`` (one intercept; controls and trend are common to both regimes,
as in ``lpirfs::lp_nl_iv``). Their difference is tested with the joint
covariance of the two coefficients.

With ``identification='lpirfs_cholesky'`` the shock is the unit Cholesky
shock of a linear VAR(p) in ``endog_order`` (``lpirfs::lp_nl``): the
impact response is that shock vector ``d`` in both regimes, and the
response at ``h >= 1`` is ``B_h d``, where ``B_h`` holds the coefficients
of the regime's first lag in the projection of ``Y_{t+h-1}`` on the
regime-weighted lags ``1..p``. Its band is ``lpirfs``' mapping of the
coefficient bands through ``d``, ``|sum_j se_j d_j|``, which is not a
delta-method standard error.

The weight is ``F = exp(-gamma z) / (1 + exp(-gamma z))`` of the state
``z`` (near 1 when ``z`` is low), of its standardised Hodrick-Prescott
cycle [@hodrick1997postwar] when ``hp_lambda`` is given, or the state
itself when ``state_transform='none'``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solveh_banded

from .._result_serialize import ResultProtocolMixin
from . import _lpx_common as C
from ._lpx_linear import lead_outcome, shifted

# --------------------------------------------------------------------- #
#  Result
# --------------------------------------------------------------------- #


@dataclass
class StateDependentLPResult(ResultProtocolMixin):
    """Regime-specific impulse responses from a state-dependent projection.

    ``*_state1`` is the regime ``F = 1`` and ``*_state0`` the regime
    ``F = 0``; ``diff = irf_state1 - irf_state0`` with its standard error
    and two-sided p-value (missing for the Cholesky projection, whose
    bands are not standard errors).
    """

    horizons: np.ndarray
    irf_state1: np.ndarray
    se_state1: np.ndarray
    irf_state0: np.ndarray
    se_state0: np.ndarray
    diff: np.ndarray
    se_diff: np.ndarray
    p_diff: np.ndarray
    alpha: float
    shock_name: str
    outcome_name: str
    n_obs_per_horizon: np.ndarray
    weights: np.ndarray
    method: str = "state_direct"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def _z(self) -> float:
        return float(stats.norm.ppf(1 - self.alpha / 2))

    def to_frame(self) -> pd.DataFrame:
        """One row per regime and horizon, regime ``state1`` first."""
        z = self._z()
        blocks = []
        for regime, irf, se in (
            ("state1", self.irf_state1, self.se_state1),
            ("state0", self.irf_state0, self.se_state0),
        ):
            blocks.append(
                pd.DataFrame(
                    {
                        "horizon": self.horizons,
                        "regime": regime,
                        "irf": irf,
                        "se": se,
                        "ci_lower": irf - z * se,
                        "ci_upper": irf + z * se,
                        "n": self.n_obs_per_horizon,
                    }
                )
            )
        return pd.concat(blocks, ignore_index=True)

    def difference_frame(self) -> pd.DataFrame:
        """Response in ``F = 1`` minus response in ``F = 0``, by horizon."""
        return pd.DataFrame(
            {
                "horizon": self.horizons,
                "diff": self.diff,
                "se": self.se_diff,
                "p_value": self.p_diff,
            }
        )

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        z = self._z()
        for color, label, irf, se in (
            ("C3", "F = 1", self.irf_state1, self.se_state1),
            ("C0", "F = 0", self.irf_state0, self.se_state0),
        ):
            ax.plot(self.horizons, irf, "-o", color=color, label=label)
            ax.fill_between(
                self.horizons, irf - z * se, irf + z * se, color=color, alpha=0.2
            )
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_xlabel("Horizon h")
        ax.set_ylabel(f"Response of {self.outcome_name} to {self.shock_name}")
        ax.legend()
        return ax

    def summary(self) -> str:
        table = pd.DataFrame(
            {
                "horizon": self.horizons,
                "irf_F1": self.irf_state1,
                "se_F1": self.se_state1,
                "irf_F0": self.irf_state0,
                "se_F0": self.se_state0,
                "diff": self.diff,
                "p_diff": self.p_diff,
                "n": self.n_obs_per_horizon,
            }
        )
        note = (
            "Bands of the Cholesky projection map the coefficient bands "
            "through the shock vector (lpirfs); they are not standard errors."
            if self.method == "state_cholesky"
            else "p_diff tests equal responses in the two regimes."
        )
        lines = [
            f"State-dependent local projections: {self.outcome_name} on "
            f"{self.shock_name}",
            "-" * 64,
            table.to_string(index=False, float_format=lambda x: f"{x: .4f}"),
            f"Regimes: F = 1 and F = 0 of the weight F_t. {note}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------- #
#  Regime weight
# --------------------------------------------------------------------- #


def hp_cycle(x: np.ndarray, lam: float) -> np.ndarray:
    """Cycle ``x - tau`` of the Hodrick-Prescott filter.

    ``tau`` solves ``(I + lam D'D) tau = x`` with ``D`` the second-difference
    operator; the pentadiagonal system is solved in banded form.
    """
    n = len(x)
    if n < 3:
        raise C.refuse(
            "the Hodrick-Prescott filter needs at least three periods.",
            "Use a longer state series or drop hp_lambda=.",
        )
    main = np.full(n, 6.0)
    main[[0, -1]] = 1.0
    main[[1, -2]] = 5.0
    off1 = np.full(n - 1, -4.0)
    off1[[0, -1]] = -2.0
    bands = np.zeros((3, n))
    bands[2] = 1.0 + lam * main
    bands[1, 1:] = lam * off1
    bands[0, 2:] = lam
    return np.asarray(x - solveh_banded(bands, x), dtype=float)


def regime_weight(
    z: np.ndarray,
    *,
    transform: str,
    gamma: Optional[float],
    hp_lambda: Optional[float],
    lag_state: bool,
    state: str,
) -> np.ndarray:
    """``F_t`` (or ``F_{t-1}`` with ``lag_state``) for every row."""
    C.check_choice(
        "state_transform",
        transform,
        ("logistic", "none"),
        "'logistic' for a smooth transition of the state, 'none' when the "
        "state is already a 0/1 indicator or a weight in [0, 1].",
    )
    if transform == "logistic":
        if gamma is None or not np.isfinite(gamma) or gamma <= 0:
            raise C.refuse(
                f"gamma={gamma!r}: the logistic transition needs gamma > 0.",
                "Pass gamma=, e.g. gamma=3 for a standardised state.",
            )
        if hp_lambda is not None:
            if not hp_lambda > 0:
                raise C.refuse(
                    f"hp_lambda={hp_lambda!r} is not positive.",
                    "Pass the smoothing parameter, e.g. 1600 for quarterly data.",
                )
            if not np.isfinite(z).all():
                raise C.refuse(
                    f"the state {state!r} has missing values; the "
                    "Hodrick-Prescott filter needs a complete series.",
                    "Fill or drop the missing periods, or drop hp_lambda=.",
                )
            cycle = hp_cycle(z, float(hp_lambda))
            z = (cycle - cycle.mean()) / cycle.std(ddof=1)
        F = np.exp(-gamma * z) / (1.0 + np.exp(-gamma * z))
    else:
        if gamma is not None or hp_lambda is not None:
            raise C.refuse(
                "gamma= and hp_lambda= shape the logistic transition; with "
                "state_transform='none' the state is used as it is.",
                "Drop them, or use state_transform='logistic'.",
            )
        obs = z[np.isfinite(z)]
        if obs.size and (obs.min() < 0 or obs.max() > 1):
            raise C.refuse(
                f"with state_transform='none' the state {state!r} is the "
                f"regime weight and must lie in [0, 1] (it spans "
                f"[{obs.min():.4g}, {obs.max():.4g}]).",
                "Pass a 0/1 indicator or a weight in [0, 1], or use "
                "state_transform='logistic' with gamma=.",
            )
        F = z.astype(float)
    return shifted(F, 1) if lag_state else F


def _finish(
    *,
    horizons: int,
    b1: np.ndarray,
    s1: np.ndarray,
    b0: np.ndarray,
    s0: np.ndarray,
    sd: np.ndarray,
    n_used: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    shock: str,
    outcome: str,
    method: str,
    diagnostics: Dict[str, Any],
) -> StateDependentLPResult:
    diff = b1 - b0
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2.0 * stats.norm.sf(np.abs(diff / sd))
    return StateDependentLPResult(
        horizons=np.arange(horizons + 1),
        irf_state1=b1,
        se_state1=s1,
        irf_state0=b0,
        se_state0=s0,
        diff=diff,
        se_diff=sd,
        p_diff=p,
        alpha=alpha,
        shock_name=shock,
        outcome_name=outcome,
        n_obs_per_horizon=n_used,
        weights=weights,
        method=method,
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------- #
#  Observed shock
# --------------------------------------------------------------------- #


def fit_state_direct(
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
    vce: str,
    dfk: Optional[bool],
    trend: int,
    common_sample: bool,
    state: str,
    weight_kw: Dict[str, Any],
) -> StateDependentLPResult:
    """State-dependent projection on an observed shock."""
    frame = data.reset_index(drop=True)
    cols = C.as_float_columns(frame, [outcome, shock, *controls, *lag_variables, state])
    F = regime_weight(cols[state], state=state, **weight_kw)
    n_rows = len(frame)
    shared = [cols[shock]] + [
        shifted(cols[v], j) for v in lag_variables for j in lag_orders
    ]
    shared_names = [shock] + [f"L{j}.{v}" for v in lag_variables for j in lag_orders]
    X_all = np.column_stack(
        [np.ones(n_rows)]
        + [col * (1.0 - F) for col in shared]
        + [col * F for col in shared]
        + [cols[c] for c in controls]
        + C.trend_columns(n_rows, trend)
    )
    names = (
        ["_cons"]
        + [f"{nm}*(1-F)" for nm in shared_names]
        + [f"{nm}*F" for nm in shared_names]
        + list(controls)
        + ["trend", "trend^2"][:trend]
    )
    i0, i1 = 1, 1 + len(shared)
    y = cols[outcome]
    lhs = [lead_outcome(y, h, cumulative) for h in range(horizons + 1)]
    finite = np.isfinite(X_all).all(axis=1)
    masks = [finite & np.isfinite(v) for v in lhs]
    if common_sample:
        joint = np.logical_and.reduce(masks)
        masks = [joint] * len(masks)

    out = np.full((5, horizons + 1), np.nan)
    n_used = np.empty(horizons + 1, dtype=int)
    for h in range(horizons + 1):
        m = masks[h]
        X = X_all[m]
        C.check_dof(int(m.sum()), X.shape[1], h)
        C.require_full_rank(X, names, h)
        fit = C.fit_ols(X, lhs[h][m])
        C.check_exact_fit(fit, lhs[h][m], h)
        divisor, factor = C.variance_settings(dfk, fit.n, fit.k)
        V = C.covariance(
            fit,
            vce,
            resid_dof=divisor,
            hac_lag=(h + 1) if nw_lags is None else int(nw_lags),
            sandwich_factor=factor,
        )
        var_diff = V[i1, i1] + V[i0, i0] - 2.0 * V[i0, i1]
        out[:, h] = (
            fit.coef[i1],
            np.sqrt(max(V[i1, i1], 0.0)),
            fit.coef[i0],
            np.sqrt(max(V[i0, i0], 0.0)),
            np.sqrt(max(var_diff, 0.0)),
        )
        n_used[h] = fit.n

    diagnostics = {
        "state": state,
        "state_transform": weight_kw["transform"],
        "gamma": weight_kw["gamma"],
        "hp_lambda": weight_kw["hp_lambda"],
        "lag_state": weight_kw["lag_state"],
        "lag_variables": list(lag_variables) if lag_orders else [],
        "lags": list(lag_orders),
        "controls": list(controls),
        "trend": trend,
        "vce": vce,
        "dfk": dfk,
        "nw_lags": (
            ("h + 1" if nw_lags is None else int(nw_lags)) if vce == "hac" else None
        ),
        "common_sample": bool(common_sample),
        "mean_weight": float(np.nanmean(F)),
    }
    return _finish(
        horizons=horizons,
        b1=out[0],
        s1=out[1],
        b0=out[2],
        s0=out[3],
        sd=out[4],
        n_used=n_used,
        weights=F,
        alpha=alpha,
        shock=shock,
        outcome=outcome,
        method="state_direct",
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------- #
#  Cholesky shock
# --------------------------------------------------------------------- #


def fit_state_cholesky(
    data: pd.DataFrame,
    *,
    outcome: str,
    shock: str,
    order: Sequence[str],
    horizons: int,
    nw_lags: Optional[int],
    alpha: float,
    p: int,
    vce: str,
    dfk: Optional[bool],
    trend: int,
    state: str,
    weight_kw: Dict[str, Any],
) -> StateDependentLPResult:
    """State-dependent projection on the unit Cholesky shock of ``shock``."""
    frame = data.reset_index(drop=True)
    cols = C.as_float_columns(frame, [*order, state])
    Y = np.column_stack([cols[v] for v in order])
    if not np.isfinite(Y).all():
        raise C.refuse(
            "the Cholesky projection needs complete endog_order data (the "
            "VAR that identifies the shock runs on consecutive periods).",
            "Drop or fill the missing periods first.",
        )
    F = regime_weight(cols[state], state=state, **weight_kw)
    n_rows, K = Y.shape
    resp, imp = list(order).index(outcome), list(order).index(shock)
    lag_block = np.column_stack(
        [shifted(Y[:, k], j) for j in range(1, p + 1) for k in range(K)]
    )
    trends = C.trend_columns(n_rows, trend)

    # unit Cholesky shock from the linear VAR(p)
    X_var = np.column_stack([np.ones(n_rows), lag_block] + trends)
    keep = np.isfinite(X_var).all(axis=1)
    C.check_dof(int(keep.sum()), X_var.shape[1], 0)
    resid = np.column_stack(
        [C.fit_ols(X_var[keep], Y[keep, k]).resid for k in range(K)]
    )
    try:
        chol = np.linalg.cholesky(resid.T @ resid)
    except np.linalg.LinAlgError:
        raise C.refuse(
            "the VAR residual covariance is singular, so the Cholesky shock "
            "is not identified.",
            "Remove a variable of endog_order that is a linear function of "
            "the others.",
        ) from None
    d = chol[:, imp] / chol[imp, imp]

    X_all = np.column_stack(
        [np.ones(n_rows), lag_block * (1.0 - F)[:, None], lag_block * F[:, None]]
        + trends
    )
    lag_names = [f"L{j}.{v}" for j in range(1, p + 1) for v in order]
    names = (
        ["_cons"]
        + [f"{nm}*(1-F)" for nm in lag_names]
        + [f"{nm}*F" for nm in lag_names]
        + ["trend", "trend^2"][:trend]
    )
    first0, first1 = 1, 1 + K * p  # first-lag block of each regime
    out = np.full((4, horizons + 1), np.nan)
    out[:, 0] = (d[resp], 0.0, d[resp], 0.0)
    n_used = np.empty(horizons + 1, dtype=int)
    n_used[0] = int(keep.sum())
    finite = np.isfinite(X_all).all(axis=1)
    for h in range(1, horizons + 1):
        lhs = shifted(Y[:, resp], -(h - 1))
        m = finite & np.isfinite(lhs)
        X = X_all[m]
        C.check_dof(int(m.sum()), X.shape[1], h)
        C.require_full_rank(X, names, h)
        fit = C.fit_ols(X, lhs[m])
        C.check_exact_fit(fit, lhs[m], h)
        divisor, factor = C.variance_settings(dfk, fit.n, fit.k)
        V = C.covariance(
            fit,
            vce,
            resid_dof=divisor,
            hac_lag=h if nw_lags is None else int(nw_lags),
            sandwich_factor=factor,
        )
        se = np.sqrt(np.maximum(np.diag(V), 0.0))
        b1 = fit.coef[first1 : first1 + K]
        b0 = fit.coef[first0 : first0 + K]
        out[:, h] = (
            b1 @ d,
            abs(se[first1 : first1 + K] @ d),
            b0 @ d,
            abs(se[first0 : first0 + K] @ d),
        )
        n_used[h] = fit.n

    diagnostics = {
        "state": state,
        "state_transform": weight_kw["transform"],
        "gamma": weight_kw["gamma"],
        "hp_lambda": weight_kw["hp_lambda"],
        "lag_state": weight_kw["lag_state"],
        "endog_order": list(order),
        "lags": list(range(1, p + 1)),
        "trend": trend,
        "impact_vector": [float(v) for v in d],
        "vce": vce,
        "dfk": dfk,
        "nw_lags": ("h" if nw_lags is None else int(nw_lags)) if vce == "hac" else None,
        "se_note": "lpirfs mapped bands |sum_j se_j d_j| through the shock "
        "vector d, not delta-method standard errors; no regime-difference test.",
        "mean_weight": float(np.nanmean(F)),
    }
    nan = np.full(horizons + 1, np.nan)
    return _finish(
        horizons=horizons,
        b1=out[0],
        s1=out[1],
        b0=out[2],
        s0=out[3],
        sd=nan,
        n_used=n_used,
        weights=F,
        alpha=alpha,
        shock=shock,
        outcome=outcome,
        method="state_cholesky",
        diagnostics=diagnostics,
    )
