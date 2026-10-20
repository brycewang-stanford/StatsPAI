"""Panel local projections (``lpirfs::lp_lin_panel``).

At horizon ``h`` the projection for unit ``i`` is

    y_{i,t+h} = a_i [+ l_t] + b_h s_{it} + sum_k c_{h,k}' w_{i,t-k}
                + d_h' x_{it} + e_{i,t+h},

with ``s`` replaced by its first difference when ``diff_shock=True`` and
``y_{i,t+h}`` by ``y_{i,t+h} - y_{i,t-1}`` when ``cumulative=True``. Leads,
lags and differences are taken within unit on the ``time`` order. The
fixed effects are swept out by the within transformation on each
horizon's own sample (``plm(model = "within")``); ``panel_effect='none'``
is pooled OLS with a constant.

Variance (``vce``):

* ``'ols'``: ``s^2 (X'X)^{-1}`` with the within degrees of freedom
  ``n - (#effects) - k``, what ``plm``'s ``summary`` reports and so the
  ``lpirfs`` default (``robust_cov = NULL``);
* ``'cluster'``: clustered by unit, ``plm::vcovHC(method = "arellano",
  type = "HC0", cluster = "group")``, no small-sample factor;
* ``'driscoll_kraay'``: ``plm::vcovSCC(type = "HC0")``, Bartlett weights
  over time periods with ``nw_lags`` lags (default ``floor(T^(1/4))``,
  ``T`` the number of periods in the horizon sample, ``plm``'s rule).

With ``instruments`` the shock is endogenous and every horizon is fitted
by within two-stage least squares (``iv_reg = TRUE``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._lp_core import resolve_lags

_EFFECTS = ("individual", "time", "twoways", "none")
_PANEL_VCE = ("cluster", "ols", "driscoll_kraay")


def _demean(M: np.ndarray, groups: List[np.ndarray], tol: float = 1e-13) -> np.ndarray:
    """Sweep out one or two sets of group means (alternating projections)."""
    out = M.copy()
    if not groups:
        return out
    for _ in range(10_000):
        prev = out.copy()
        for g in groups:
            counts = np.bincount(g)
            for j in range(out.shape[1]):
                means = np.bincount(g, weights=out[:, j]) / counts
                out[:, j] -= means[g]
        if len(groups) == 1:
            break
        if np.max(np.abs(out - prev)) <= tol * max(1.0, np.max(np.abs(out))):
            break
    return out


def _codes(values: np.ndarray) -> np.ndarray:
    return pd.factorize(values, sort=True)[0]


def lp_panel(
    data: pd.DataFrame,
    outcome: str,
    shock: str,
    controls: Sequence[str],
    horizons: int,
    id: str,
    time: str,
    lags: Any,
    endog_order: Optional[List[str]],
    instruments: Optional[List[str]],
    panel_effect: str,
    diff_shock: bool,
    cumulative: bool,
    vce: Optional[str],
    nw_lags: Optional[int],
    alpha: float,
):
    from .local_projections import LocalProjectionsResult

    instruments = list(instruments or [])
    if panel_effect not in _EFFECTS:
        raise MethodIncompatibility(
            f"sp.local_projections: panel_effect={panel_effect!r} is not one "
            f"of {list(_EFFECTS)}.",
            recovery_hint="'individual' = unit fixed effects (plm within), "
            "'twoways' adds period effects, 'none' is pooled OLS.",
        )
    vce = "cluster" if vce is None else str(vce).lower()
    if vce in ("scc", "dk"):
        vce = "driscoll_kraay"
    if vce not in _PANEL_VCE:
        raise MethodIncompatibility(
            f"sp.local_projections: vce={vce!r} is not available for panel "
            f"projections ({list(_PANEL_VCE)}).",
            recovery_hint="Use vce='cluster' (by unit), 'driscoll_kraay' or " "'ols'.",
        )
    lag_set = resolve_lags(lags)
    lag_vars = list(dict.fromkeys(endog_order or [outcome, shock])) if lag_set else []
    check_columns(
        data,
        {
            "id": id,
            "time": time,
            "endog_order": lag_vars,
            "instruments": instruments,
        },
        function="sp.local_projections",
    )
    df = data.sort_values([id, time], kind="mergesort").reset_index(drop=True)
    if df.duplicated([id, time]).any():
        raise MethodIncompatibility(
            "sp.local_projections: repeated (id, time) pairs in the panel.",
            recovery_hint="Each unit must have one row per period.",
        )
    unit = df[id].to_numpy()
    n_units = int(pd.unique(unit).size)
    if vce == "cluster" and n_units < 2:
        # With one cluster the unit-clustered meat is the outer product of
        # a score sum that the normal equations set to zero: SE ~ 1e-16
        # and a zero-width interval (2026-10-20 probe).
        raise DataInsufficient(
            "sp.local_projections: vce='cluster' clusters by unit and the "
            f"panel has {n_units} unit; a one-cluster variance is zero by "
            "construction.",
            recovery_hint="Use vce='driscoll_kraay' or 'ols', or drop id= / "
            "time= and run the single-unit time-series projection "
            "(vce='hac').",
        )
    tcode = _codes(df[time].to_numpy())
    # Position of each row's period among the unit's own periods is not what
    # a lead means: a lead is the row whose time code is t + h in the same
    # unit. Index the panel by (unit, period code) to look leads up.
    ucode = _codes(unit)
    n_t = int(tcode.max()) + 1
    grid_index = ucode * n_t + tcode

    def shifted(col: np.ndarray, k: int) -> np.ndarray:
        """``col`` at period ``t + k`` of the same unit (NaN if absent)."""
        full = np.full(ucode.max() * n_t + n_t, np.nan)
        full[grid_index] = col
        target = tcode + k
        out = np.full(len(col), np.nan)
        inside = (target >= 0) & (target < n_t)
        out[inside] = full[ucode[inside] * n_t + target[inside]]
        return out

    y = df[outcome].to_numpy(dtype=float)
    s = df[shock].to_numpy(dtype=float)
    if diff_shock:
        s = s - shifted(s, -1)
    reg: Dict[str, np.ndarray] = {("d" + shock) if diff_shock else shock: s}
    for lag in lag_set:
        for v in lag_vars:
            reg[f"{v}_L{lag}"] = shifted(df[v].to_numpy(dtype=float), -lag)
    for c in controls:
        reg[str(c)] = df[c].to_numpy(dtype=float)
    inst: Dict[str, np.ndarray] = {}
    for z in instruments:
        zc = df[z].to_numpy(dtype=float)
        inst[("d" + z) if diff_shock else z] = (
            zc - shifted(zc, -1) if diff_shock else zc
        )
    R = np.column_stack(list(reg.values()))
    Zx = np.column_stack(list(inst.values())) if inst else np.empty((len(df), 0))
    y_lag1 = shifted(y, -1)

    irf = np.full(horizons + 1, np.nan)
    se = np.full(horizons + 1, np.nan)
    n_used = np.zeros(horizons + 1, dtype=int)
    groups_used = np.zeros(horizons + 1, dtype=int)
    for h in range(horizons + 1):
        lhs = shifted(y, h)
        if cumulative:
            lhs = lhs - y_lag1
        rows = np.isfinite(lhs) & np.all(np.isfinite(R), axis=1)
        rows &= np.all(np.isfinite(Zx), axis=1)
        yy, XX, ZZ = lhs[rows], R[rows], Zx[rows]
        u = _codes(ucode[rows])
        tt = _codes(tcode[rows])
        if panel_effect == "individual":
            groups, n_fe = [u], int(u.max()) + 1
        elif panel_effect == "time":
            groups, n_fe = [tt], int(tt.max()) + 1
        elif panel_effect == "twoways":
            groups, n_fe = [u, tt], int(u.max()) + int(tt.max()) + 1
        else:
            groups, n_fe = [], 0
        stacked = np.column_stack([yy, XX, ZZ])
        if groups:
            stacked = _demean(stacked, groups)
        else:
            XX_c = np.column_stack([np.ones(len(yy)), XX])
            stacked = np.column_stack([yy, XX_c, ZZ])
        yd = stacked[:, 0]
        kx = XX.shape[1] + (0 if groups else 1)
        Xd = stacked[:, 1 : 1 + kx]
        Zd = stacked[:, 1 + kx :]
        pos = 0 if groups else 1
        n = len(yd)
        if n - n_fe - kx <= 0:
            raise DataInsufficient(
                f"sp.local_projections: {n} observations at horizon {h} "
                f"leave no degrees of freedom after {n_fe} fixed effects.",
                recovery_hint="Use fewer horizons or lags.",
                diagnostics={"horizon": h, "n": n},
            )
        if instruments:
            W = np.column_stack([Zd, np.delete(Xd, pos, axis=1)])
            P = W @ np.linalg.solve(W.T @ W, W.T @ Xd)
        else:
            P = Xd
        if np.linalg.matrix_rank(P) < P.shape[1]:
            raise MethodIncompatibility(
                f"sp.local_projections: the shock is not identified at "
                f"horizon {h} (collinear with the fixed effects or controls).",
                recovery_hint="A shock that is common to all units is "
                "absorbed by period effects; use panel_effect='individual'.",
                diagnostics={"horizon": h},
            )
        bread = np.linalg.inv(P.T @ P)
        beta = bread @ (P.T @ yd)
        e = yd - Xd @ beta
        if vce == "ols":
            V = float(e @ e) / (n - n_fe - kx) * bread
        elif vce == "cluster":
            sc = P * e[:, None]
            G = np.zeros((int(u.max()) + 1, P.shape[1]))
            np.add.at(G, u, sc)
            V = bread @ (G.T @ G) @ bread
        else:
            sc = P * e[:, None]
            G = np.zeros((int(tt.max()) + 1, P.shape[1]))
            np.add.at(G, tt, sc)
            L = (
                int(np.floor((int(tt.max()) + 1) ** 0.25))
                if nw_lags is None
                else int(nw_lags)
            )
            meat = G.T @ G
            for j in range(1, L + 1):
                w = 1.0 - j / (L + 1.0)
                gj = G[j:].T @ G[:-j]
                meat += w * (gj + gj.T)
            V = bread @ meat @ bread
        irf[h] = beta[pos]
        se[h] = float(np.sqrt(max(V[pos, pos], 0.0)))
        n_used[h] = n
        groups_used[h] = int(u.max()) + 1
    z_crit = stats.norm.ppf(1 - alpha / 2)
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
        method="panel_iv" if instruments else "panel",
        diagnostics={
            "panel_effect": panel_effect,
            "vce": vce,
            "diff_shock": diff_shock,
            "cumulative": cumulative,
            "lag_variables": lag_vars,
            "lags": lag_set,
            "units_per_horizon": groups_used.tolist(),
            "instruments": instruments,
        },
    )
