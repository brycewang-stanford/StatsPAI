"""State-dependent (regime-switching) local projections.

Every regressor except the constant, the trend and the contemporaneous
controls is interacted with the regime weight ``F_t`` and with ``1 - F_t``,

    y_{t+h} = a_h + b1_h F_t s_t + b0_h (1 - F_t) s_t
              + sum_k [c1_{h,k}' F_t + c0_{h,k}' (1 - F_t)] w_{t-k}
              + d_h' x_t + e_{t+h},

so ``b1_h`` is the response in the regime ``F = 1`` and ``b0_h`` the one in
``F = 0``. ``F`` comes from :func:`._lp_core.regime_weight`.

Two shock definitions:

* observed shock (``identification='direct'``): ``lpirfs::lp_nl_iv``;
* Cholesky shock (``identification='lpirfs_cholesky'``):
  ``lpirfs::lp_nl`` with ``shock_type = 1``: the unit structural shock is
  identified once, from the residual covariance of the linear VAR with the
  same lags, and the regime coefficients on the first lag of the system are
  mapped through it.

``lpirfs`` names the regime with weight ``1 - F`` "s1" and the one with
weight ``F`` "s2"; here they are ``state0`` and ``state1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._lp_core import (
    bartlett_meat,
    lag_columns,
    lead,
    ols_fit,
    regime_weight,
    resolve_lags,
)
from ._lp_linear import trend_columns


@dataclass
class StateDependentLPResult(ResultProtocolMixin):
    """Regime-specific impulse responses from a state-dependent projection.

    ``irf_state1`` is the response when the regime weight ``F`` is 1,
    ``irf_state0`` when it is 0. ``diff = irf_state1 - irf_state0`` with its
    standard error and Wald p-value from the joint covariance of the two
    coefficients in the same horizon regression (observed shock only; NaN
    for the Cholesky shock, whose bands are the ``lpirfs`` mapped bands).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> T = 300
    >>> s = rng.normal(size=T)
    >>> d = (rng.uniform(size=T) < 0.5).astype(float)
    >>> d_lag = np.r_[0.0, d[:-1]]
    >>> y = np.where(d_lag == 1, 1.0, 0.2) * s + rng.normal(0, 0.3, T)
    >>> df = pd.DataFrame({"y": y, "s": s, "d": d})
    >>> r = sp.local_projections(df, outcome="y", shock="s", horizons=2,
    ...                          state="d", state_transform="none", lags=0)
    >>> round(float(r.irf_state1[0]), 1), round(float(r.irf_state0[0]), 1)
    (1.0, 0.2)
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

    def to_frame(self) -> pd.DataFrame:
        z = stats.norm.ppf(1 - self.alpha / 2)
        parts = []
        for name, b, s in (
            ("state1", self.irf_state1, self.se_state1),
            ("state0", self.irf_state0, self.se_state0),
        ):
            parts.append(
                pd.DataFrame(
                    {
                        "horizon": self.horizons,
                        "regime": name,
                        "irf": b,
                        "se": s,
                        "ci_lower": b - z * s,
                        "ci_upper": b + z * s,
                        "n": self.n_obs_per_horizon,
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)

    def difference_frame(self) -> pd.DataFrame:
        """Per-horizon test that the two regime responses are equal."""
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
        z = stats.norm.ppf(1 - self.alpha / 2)
        for name, b, s, color in (
            ("F = 1", self.irf_state1, self.se_state1, "C3"),
            ("F = 0", self.irf_state0, self.se_state0, "C0"),
        ):
            ax.plot(self.horizons, b, "-o", color=color, label=name)
            ax.fill_between(self.horizons, b - z * s, b + z * s, alpha=0.2, color=color)
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_xlabel("Horizon h")
        ax.set_ylabel(f"Response of {self.outcome_name} to {self.shock_name}")
        ax.legend()
        return ax

    def summary(self) -> str:
        wide = pd.DataFrame(
            {
                "horizon": self.horizons,
                "irf_F1": self.irf_state1,
                "se_F1": self.se_state1,
                "irf_F0": self.irf_state0,
                "se_F0": self.se_state0,
                "p_equal": self.p_diff,
                "n": self.n_obs_per_horizon,
            }
        )
        lines = [
            "State-dependent local projections — "
            f"{self.outcome_name} on {self.shock_name}",
            "-" * 64,
            wide.to_string(index=False, float_format=lambda x: f"{x: .4f}"),
            "F = 1 and F = 0 are the two regimes of the weight F_t; p_equal "
            "tests equal responses.",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _interact(block: Dict[str, np.ndarray], F: np.ndarray):
    names, cols = [], []
    for tag, w in (("s0", 1.0 - F), ("s1", F)):
        for k, v in block.items():
            names.append(f"{k}_{tag}")
            cols.append(v * w)
    return names, cols


def lp_state(
    data: pd.DataFrame,
    outcome: str,
    shock: str,
    controls: Sequence[str],
    horizons: int,
    lags: Any,
    endog_order: Optional[List[str]],
    state: str,
    state_transform: str,
    gamma: Optional[float],
    hp_lambda: Optional[float],
    lag_state: bool,
    trend: int,
    cumulative: bool,
    nw_lags: Optional[int],
    vce: str,
    alpha: float,
    identification: str,
) -> StateDependentLPResult:
    check_columns(data, {"state": state}, function="sp.local_projections")
    df = data.reset_index(drop=True)
    n = len(df)
    F = regime_weight(
        df[state].to_numpy(dtype=float), state_transform, gamma, hp_lambda, lag_state
    )
    lag_set = resolve_lags(1 if lags is None else lags)
    if identification == "cholesky":
        return _state_cholesky(
            df,
            outcome,
            shock,
            controls,
            horizons,
            lag_set,
            endog_order,
            F,
            trend,
            nw_lags,
            alpha,
            cumulative,
        )
    lag_vars = list(dict.fromkeys(endog_order or [outcome, shock]))
    check_columns(df, {"endog_order": lag_vars}, function="sp.local_projections")
    y = df[outcome].to_numpy(dtype=float)
    s = df[shock].to_numpy(dtype=float)
    lag_block = lag_columns(df, lag_vars, lag_set)
    inter_names, inter_cols = _interact({shock: s, **lag_block}, F)
    plain: Dict[str, np.ndarray] = dict(trend_columns(n, trend))
    for c in controls:
        plain[str(c)] = df[c].to_numpy(dtype=float)
    X_all = np.column_stack([np.ones(n)] + inter_cols + list(plain.values()))
    names = ["_cons"] + inter_names + list(plain)
    i0 = names.index(f"{shock}_s0")
    i1 = names.index(f"{shock}_s1")
    ok = np.all(np.isfinite(X_all), axis=1)
    y_lag1 = np.concatenate([[np.nan], y[:-1]])

    out = {
        k: np.full(horizons + 1, np.nan) for k in ("b1", "s1", "b0", "s0", "d", "sd")
    }
    n_used = np.zeros(horizons + 1, dtype=int)
    for h in range(horizons + 1):
        lhs = lead(y, h) - y_lag1 if cumulative else lead(y, h)
        rows = ok & np.isfinite(lhs)
        X = X_all[rows]
        if X.shape[0] <= X.shape[1] + 1:
            raise DataInsufficient(
                f"sp.local_projections: {X.shape[0]} usable observations at "
                f"horizon {h} for {X.shape[1]} state-dependent coefficients.",
                recovery_hint="Use fewer horizons or lags, or a state with "
                "more observations in each regime.",
                diagnostics={"horizon": h, "n": int(X.shape[0])},
            )
        if np.linalg.matrix_rank(X) < X.shape[1]:
            raise MethodIncompatibility(
                f"sp.local_projections: the regime-interacted regressors are "
                f"collinear at horizon {h}: one regime has too few "
                "observations to identify its coefficients.",
                recovery_hint="Check that the state varies over the sample; "
                "with an indicator, both values must occur.",
                diagnostics={"horizon": h, "regime_share": float(np.nanmean(F))},
            )
        fit = ols_fit(lhs[rows], X, vce, h + 1 if nw_lags is None else nw_lags)
        V = fit.vcov
        out["b1"][h], out["b0"][h] = fit.beta[i1], fit.beta[i0]
        out["s1"][h] = np.sqrt(max(V[i1, i1], 0.0))
        out["s0"][h] = np.sqrt(max(V[i0, i0], 0.0))
        out["d"][h] = fit.beta[i1] - fit.beta[i0]
        out["sd"][h] = np.sqrt(max(V[i1, i1] + V[i0, i0] - 2 * V[i0, i1], 0.0))
        n_used[h] = fit.nobs
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * stats.norm.sf(np.abs(out["d"] / out["sd"]))
    return StateDependentLPResult(
        horizons=np.arange(horizons + 1),
        irf_state1=out["b1"],
        se_state1=out["s1"],
        irf_state0=out["b0"],
        se_state0=out["s0"],
        diff=out["d"],
        se_diff=out["sd"],
        p_diff=p,
        alpha=alpha,
        shock_name=shock,
        outcome_name=outcome,
        n_obs_per_horizon=n_used,
        weights=F,
        method="state_direct",
        diagnostics={
            "state": state,
            "state_transform": state_transform,
            "gamma": gamma,
            "hp_lambda": hp_lambda,
            "lag_state": lag_state,
            "lag_variables": lag_vars,
            "lags": lag_set,
            "vce": vce,
            "nw_lags": "h + 1" if nw_lags is None else int(nw_lags),
            "mean_weight": float(np.nanmean(F)),
        },
    )


def _state_cholesky(
    df,
    outcome,
    shock,
    controls,
    horizons,
    lag_set,
    endog_order,
    F,
    trend,
    nw_lags,
    alpha,
    cumulative,
) -> StateDependentLPResult:
    """``lpirfs::lp_nl(shock_type = 1)`` with the same lags in both parts."""
    if cumulative:
        raise MethodIncompatibility(
            "sp.local_projections: cumulative=True is not available for the "
            "Cholesky state-dependent projection (lpirfs::lp_nl has none).",
            recovery_hint="Use identification='direct' with an observed shock.",
        )
    if lag_set != list(range(1, len(lag_set) + 1)) or not lag_set:
        raise MethodIncompatibility(
            "sp.local_projections: the Cholesky projection needs lags 1..p.",
            recovery_hint="Pass lags=p with p >= 1.",
        )
    order = list(endog_order) if endog_order is not None else [outcome, shock]
    if outcome not in order or shock not in order or len(set(order)) != len(order):
        raise MethodIncompatibility(
            "sp.local_projections: endog_order must list outcome and shock "
            "once each.",
            recovery_hint="e.g. endog_order=['gdp', 'rate'] (Cholesky order).",
        )
    check_columns(df, {"endog_order": order}, function="sp.local_projections")
    n = len(df)
    Y = df[order].to_numpy(dtype=float)
    if np.isnan(Y).any():
        raise MethodIncompatibility(
            "sp.local_projections: the Cholesky projection needs complete "
            "endog_order data.",
            recovery_hint="Drop or fill the missing periods first.",
        )
    k = len(order)
    lag_block = lag_columns(df, order, lag_set)
    plain: Dict[str, np.ndarray] = dict(trend_columns(n, trend))
    for c in controls:
        plain[str(c)] = df[c].to_numpy(dtype=float)
    # Linear VAR(p) for the shock: residual covariance -> unit Cholesky.
    X_lin = np.column_stack(
        [np.ones(n)] + list(lag_block.values()) + list(plain.values())
    )
    ok_lin = np.all(np.isfinite(X_lin), axis=1)
    B, *_ = np.linalg.lstsq(X_lin[ok_lin], Y[ok_lin], rcond=None)
    resid = Y[ok_lin] - X_lin[ok_lin] @ B
    chol = np.linalg.cholesky(np.cov(resid, rowvar=False, ddof=1))
    d = chol / np.diag(chol)[None, :]
    shock_vec = d[:, order.index(shock)]
    r = order.index(outcome)

    inter_names, inter_cols = _interact(lag_block, F)
    X_all = np.column_stack([np.ones(n)] + inter_cols + list(plain.values()))
    names = ["_cons"] + inter_names + list(plain)
    first = [f"{v}_L{lag_set[0]}" for v in order]
    i0 = [names.index(f"{c}_s0") for c in first]
    i1 = [names.index(f"{c}_s1") for c in first]
    ok = np.all(np.isfinite(X_all), axis=1)

    b1 = np.full(horizons + 1, np.nan)
    b0 = np.full(horizons + 1, np.nan)
    s1 = np.zeros(horizons + 1)
    s0 = np.zeros(horizons + 1)
    n_used = np.zeros(horizons + 1, dtype=int)
    b1[0] = b0[0] = shock_vec[r]
    n_used[0] = int(ok.sum())
    for h in range(1, horizons + 1):
        coef1, coef0 = np.empty((k, k)), np.empty((k, k))
        half1, half0 = np.empty((k, k)), np.empty((k, k))
        for j in range(k):
            lhs = lead(Y[:, j], h - 1)
            rows = ok & np.isfinite(lhs)
            X = X_all[rows]
            lag_nw = h if nw_lags is None else int(nw_lags)
            xtx_inv = np.linalg.inv(X.T @ X)
            beta = xtx_inv @ (X.T @ lhs[rows])
            e = lhs[rows] - X @ beta
            V = xtx_inv @ bartlett_meat(X * e[:, None], lag_nw) @ xtx_inv
            sd = np.sqrt(np.maximum(np.diag(V), 0.0))
            coef1[j], coef0[j] = beta[i1], beta[i0]
            half1[j], half0[j] = sd[i1], sd[i0]
            if j == r:
                n_used[h] = int(rows.sum())
        b1[h] = (coef1 @ shock_vec)[r]
        b0[h] = (coef0 @ shock_vec)[r]
        # lpirfs maps b -/+ se through the shock vector; half the distance
        # between the two mapped limits is sum_j se_j w_j (sign kept by
        # lpirfs, so take its absolute value as the band half-width).
        s1[h] = abs(float((half1 @ shock_vec)[r]))
        s0[h] = abs(float((half0 @ shock_vec)[r]))
    nan = np.full(horizons + 1, np.nan)
    return StateDependentLPResult(
        horizons=np.arange(horizons + 1),
        irf_state1=b1,
        se_state1=s1,
        irf_state0=b0,
        se_state0=s0,
        diff=b1 - b0,
        se_diff=nan,
        p_diff=nan.copy(),
        alpha=alpha,
        shock_name=shock,
        outcome_name=outcome,
        n_obs_per_horizon=n_used,
        weights=F,
        method="state_cholesky",
        diagnostics={
            "endog_order": order,
            "lags": lag_set,
            "impact_vector": shock_vec.tolist(),
            "nw_lags": "h" if nw_lags is None else int(nw_lags),
            "se_note": "lpirfs mapped bands: |sum_j se_j d_j|, not a delta-"
            "method standard error; no regime-difference test.",
            "mean_weight": float(np.nanmean(F)),
        },
    )
