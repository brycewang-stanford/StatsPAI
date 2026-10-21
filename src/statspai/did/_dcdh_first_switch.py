"""First-time-switcher estimators behind ``sp.did_multiplegt(robust_dynamic=True)``.

These are the DID_l estimators of de Chaisemartin and D'Haultfoeuille
[@dechaisemartin2024difference] as Stata ``did_multiplegt_old, robust_dynamic``
computes them. A group's baseline is its treatment in the first period it is
observed, and its first switch is the first period its treatment differs
from that baseline. DID_l compares the outcome change from the last period
before the first switch to l periods after it, between the groups that first
switched then (each with the sign of its first change) and the groups with
the same baseline that have not switched yet l periods after it. The
estimate is not divided by the size of the treatment change.

Every convention below was established by running Stata 18 on the same CSV
bytes (``tests/reference_parity/_fixtures/_generate_dcdh_dynamic_Stata.do``):

* placebo l is the long difference ``Y_{F-1-l} - Y_{F-1}`` over the
  switchers and controls of DID_{l-1}: controls have not switched by
  ``F+l-1``, and a switcher whose DID_{l-1} lies beyond the panel has no
  placebo l;
* with cells of several rows, a cell weighs its row count at the end of the
  effect's window (``F+l`` for DID_l, ``F+l-1`` for placebo l);
* ``controls=`` nets out ``dX @ theta_d``, where ``theta_d`` comes from a
  regression of the first difference of the outcome on the first
  differences of the controls and period effects, over the (group, period)
  cells of baseline ``d`` that have not switched by that period (weighted
  like the cells).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def period_fe_slopes(
    dx: np.ndarray, period: Any, dy: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Weighted least-squares slopes of ``dy`` on ``dx`` with period effects.

    The one regression behind every ``controls=`` adjustment of
    ``sp.did_multiplegt``; only the sample and the differences change.

    A control that is constant within periods in the sample (for instance a
    baseline with a single stayer cell), or collinear with the controls
    before it, gets a slope of 0, as R ``DIDmultiplegt`` does (``lm`` reports
    it as aliased; checked on the fixture, where a minimum-norm solution is
    4e-3 off). Stata's regressions omit collinear regressors too.
    """
    period = pd.Series(np.asarray(period))
    k = dx.shape[1]
    dummies = pd.get_dummies(period).to_numpy(dtype=float)
    sw = np.sqrt(w)
    X = np.hstack([dx, dummies]) * sw[:, None]
    # residual of each control on the period effects and the controls kept
    # so far, relative to its own size: ~0 means not identified
    keep: List[int] = []
    base = dummies * sw[:, None]
    for j in range(k):
        col = dx[:, j] * sw
        Z = np.hstack([base, X[:, keep]]) if keep else base
        resid = col - Z @ np.linalg.lstsq(Z, col, rcond=None)[0]
        scale = max(float(np.linalg.norm(col)), 1e-300)
        if float(np.linalg.norm(resid)) > 1e-10 * scale:
            keep.append(j)
    beta = np.zeros(k)
    if keep:
        cols = keep + list(range(k, X.shape[1]))
        beta[keep] = np.linalg.lstsq(X[:, cols], dy * sw, rcond=None)[0][: len(keep)]
    return beta


def _baseline_and_first_switch(dpiv: pd.DataFrame) -> pd.DataFrame:
    """Baseline treatment, first-switch period and its sign, per group.

    ``F`` is ``inf`` for a group that never leaves its baseline.
    """
    periods = list(dpiv.columns)
    first_col = dpiv.notna().to_numpy().argmax(axis=1)
    values = dpiv.to_numpy(dtype=float)
    base = values[np.arange(len(values)), first_col]
    first = np.full(len(values), np.inf)
    sign = np.zeros(len(values))
    changed = dpiv.notna().to_numpy() & (values != base[:, None])
    for i in np.flatnonzero(changed.any(axis=1)):
        j = int(changed[i].argmax())
        first[i] = j  # column index of the first switch
        # The sign is that of the summed deviation from the baseline over
        # every observed period from the first switch on, not the sign of
        # the first change; a sum of zero counts as a decrease (Stata, black
        # box: a group going 1 -> 2 and later below 1 enters negatively).
        sign[i] = 1.0 if np.nansum(values[i, j:] - base[i]) > 0 else -1.0
    out = pd.DataFrame({"base": base, "F": first, "S": sign}, index=dpiv.index)
    out.attrs["periods"] = periods
    return out


def _not_yet_switched_slopes(
    ypiv: pd.DataFrame,
    xpivs: List[pd.DataFrame],
    wpiv: Optional[pd.DataFrame],
    info: pd.DataFrame,
) -> Dict[float, np.ndarray]:
    """Slopes of dY on dX with period effects, per baseline, over the cells
    that have not switched yet (Stata ``did_multiplegt_old, robust_dynamic
    controls()``)."""
    periods = list(ypiv.columns)
    rows = []
    for k in range(1, len(periods)):
        t, tp = periods[k], periods[k - 1]
        keep = info["F"].to_numpy() > k
        dy = (ypiv[t] - ypiv[tp]).to_numpy(dtype=float)
        dx = np.column_stack([(xp[t] - xp[tp]).to_numpy(dtype=float) for xp in xpivs])
        w = wpiv[t].to_numpy(dtype=float) if wpiv is not None else np.ones(len(dy))
        ok = keep & np.isfinite(dy) & np.isfinite(dx).all(axis=1) & np.isfinite(w)
        for i in np.flatnonzero(ok):
            rows.append((info["base"].iat[i], k, dy[i], dx[i], w[i]))
    slopes: Dict[float, np.ndarray] = {}
    if not rows:
        return slopes
    frame = pd.DataFrame(rows, columns=["base", "k", "dy", "dx", "w"])
    for base, sub in frame.groupby("base", sort=True):
        slopes[float(base)] = period_fe_slopes(
            np.vstack(sub["dx"].to_numpy()),
            sub["k"],
            sub["dy"].to_numpy(dtype=float),
            sub["w"].to_numpy(dtype=float),
        )
    return slopes


def first_switch_effect(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    kind: str,
    lag: int,
    w: Optional[str] = None,
) -> Dict[str, Any]:
    """DID_l (``kind='dynamic'``, ``lag=l >= 0``) or the long-difference
    placebo l (``kind='placebo'``, ``lag=l >= 1``) of Stata
    ``did_multiplegt_old, robust_dynamic``.

    Returns the estimate (NaN when no switcher has a control), the number of
    switchers (rows, i.e. summed cell weights) and one record per
    (baseline, first-switch period) cell.
    """
    dpiv = df.pivot_table(index=group, columns=time, values=treatment, aggfunc="first")
    periods = list(dpiv.columns)
    ypiv = df.pivot_table(
        index=group, columns=time, values=y, aggfunc="first", dropna=False
    ).reindex(index=dpiv.index, columns=periods)
    wpiv = (
        df.pivot_table(
            index=group, columns=time, values=w, aggfunc="first", dropna=False
        ).reindex(index=dpiv.index, columns=periods)
        if w
        else None
    )
    xpivs = [
        df.pivot_table(
            index=group, columns=time, values=c, aggfunc="first", dropna=False
        ).reindex(index=dpiv.index, columns=periods)
        for c in (controls or [])
    ]
    info = _baseline_and_first_switch(dpiv)
    slopes = _not_yet_switched_slopes(ypiv, xpivs, wpiv, info) if controls else {}

    obs = ypiv.notna().to_numpy()
    F = info["F"].to_numpy()
    base_all = info["base"].to_numpy()
    S_all = info["S"].to_numpy()
    num = 0.0
    den = 0.0
    cells: List[Dict[str, Any]] = []
    for i in range(1, len(periods)):
        if kind == "dynamic":
            end = i + lag  # last period of the long difference and weight
            if end >= len(periods):
                continue
            a, b = periods[end], periods[i - 1]
        else:
            end = i + lag - 1  # DID_{l-1}'s window: controls and weights
            if i - 1 - lag < 0 or end >= len(periods):
                continue
            a, b = periods[i - 1 - lag], periods[i - 1]
        t_w = periods[end]
        ya = ypiv[a].to_numpy(dtype=float)
        yb = ypiv[b].to_numpy(dtype=float)
        diff = ya - yb
        # A group enters only if it is observed in every period of the
        # effect's window, from F-1 to its end (for placebo l: DID_{l-1}'s
        # window), not only at the two ends of the difference. Gaps inside
        # a placebo's own pre-period window are allowed (Stata, black box).
        observed = np.isfinite(diff) & obs[:, i - 1 : end + 1].all(axis=1)
        if xpivs:
            dx = np.column_stack(
                [(xp[a] - xp[b]).to_numpy(dtype=float) for xp in xpivs]
            )
            observed &= np.isfinite(dx).all(axis=1)
        wt = wpiv[t_w].to_numpy(dtype=float) if wpiv is not None else None
        if wt is not None:
            observed &= np.isfinite(wt)
        for base in np.unique(base_all[F == i]):
            sw = np.flatnonzero((F == i) & (base_all == base) & observed)
            ct = np.flatnonzero((F > end) & (base_all == base) & observed)
            if len(sw) == 0 or len(ct) == 0:
                continue
            d_sw, d_ct = diff[sw], diff[ct]
            if xpivs:
                theta = slopes.get(float(base))
                if theta is None:
                    continue
                d_sw = d_sw - dx[sw] @ theta
                d_ct = d_ct - dx[ct] @ theta
            w_sw = wt[sw] if wt is not None else np.ones(len(sw))
            w_ct = wt[ct] if wt is not None else np.ones(len(ct))
            ctrl_mean = float(np.sum(w_ct * d_ct) / np.sum(w_ct))
            contrib = float(np.sum(w_sw * S_all[sw] * (d_sw - ctrl_mean)))
            n_sw = float(np.sum(w_sw))
            num += contrib
            den += n_sw
            cells.append(
                {
                    "time": periods[i],
                    "baseline_treatment": float(base),
                    "n_switchers": n_sw,
                    "n_controls": float(np.sum(w_ct)),
                    "did_gt": contrib / n_sw,
                }
            )
    if den == 0:
        return {"estimate": float("nan"), "n_cells": 0, "n_switchers": 0, "cells": []}
    return {
        "estimate": num / den,
        "n_cells": len(cells),
        "n_switchers": int(round(den)),
        "cells": cells,
    }
