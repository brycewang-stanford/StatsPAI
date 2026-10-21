"""Pair-by-pair estimators of ``sp.did_multiplegt`` (the default path).

DID_M of de Chaisemartin and D'Haultfoeuille (2020) and its placebos and
dynamic effects as R ``DIDmultiplegt`` 0.1.4 extends it: every consecutive
period pair (t-1, t) compares the groups that switch with the groups of the
same baseline treatment that stay. The first-time-switcher estimators of
``robust_dynamic=True`` are in ``_dcdh_first_switch.py``.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._dcdh_cells import _stayer_slopes, _wmean
from ._dcdh_first_switch import period_fe_slopes


def _estimate_did_m(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    w: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Core DID_M estimator: weighted average of cell-level DID estimates.

    ``w`` names a column of (group, period) cell sizes (data collapsed by
    :func:`_collapse_to_cells`); each cell then counts that many times.

    For each period t, identify switchers and stayers, compute DID_gt.

    With ``controls`` the first difference of the outcome is replaced by
    ``dY - dX @ theta_d``, where ``theta_d`` comes from a regression of dY
    on dX and period effects over the stayer cells whose treatment is
    ``d`` in both periods (Stata ``did_multiplegt_old``, one regression per
    baseline value, weighted like the cells).
    """
    periods = sorted(df[time].unique())
    cell_estimates = []
    total_switchers = 0
    slopes = (
        _stayer_slopes(df, y, group, time, treatment, controls, w) if controls else {}
    )

    for idx in range(1, len(periods)):
        t_prev, t_curr = periods[idx - 1], periods[idx]

        df_prev = df[df[time] == t_prev][
            [group, y, treatment] + (controls or [])
        ].copy()
        df_curr = df[df[time] == t_curr][
            [group, y, treatment] + (controls or []) + ([w] if w else [])
        ].copy()

        df_prev = df_prev.rename(
            columns={y: f"{y}_prev", treatment: f"{treatment}_prev"},
        )
        if controls:
            df_prev = df_prev.rename(columns={c: f"{c}_prev" for c in controls})

        merged = df_curr.merge(df_prev, on=group, how="inner")
        if merged.empty:
            continue

        merged["_switched"] = merged[treatment] != merged[f"{treatment}_prev"]
        merged["_dy"] = merged[y] - merged[f"{y}_prev"]

        # de Chaisemartin--D'Haultfoeuille (2020): the cell-level DID must
        # condition on the BASELINE treatment d_{t-1}. Switchers are compared
        # only to stayers that shared the same period-(t-1) treatment value, and
        # switch-OFF cells (baseline 1 -> 0) enter with a flipped sign so the
        # estimand is the effect of *gaining* treatment. Pooling across baselines
        # (i) contaminates the control trend with already-treated stayers and
        # (ii) mixes switch-on / switch-off effects under a single sign -- both
        # bias DID_M (see Paper-DiD-JAE/replication/did_multiplegt/).
        for base in sorted(merged[f"{treatment}_prev"].unique()):
            grp = merged[merged[f"{treatment}_prev"] == base]
            switchers = grp[grp["_switched"]]
            stayers = grp[~grp["_switched"]]

            n_switch = len(switchers)
            if n_switch == 0 or len(stayers) == 0:
                continue
            w_sw = switchers[w].to_numpy(dtype=float) if w else None
            w_st = stayers[w].to_numpy(dtype=float) if w else None

            dy_switch = switchers["_dy"].values.copy()
            dy_stay = stayers["_dy"].values.copy()

            # Net out the controls with the stayer-sample slopes of this
            # baseline value (Stata did_multiplegt_old's controls()).
            if controls:
                theta = slopes.get(float(base))
                if theta is not None:
                    dX_switch = np.column_stack(
                        [
                            switchers[c].to_numpy(dtype=float)
                            - switchers[f"{c}_prev"].to_numpy(dtype=float)
                            for c in controls
                        ]
                    )
                    dX_stay = np.column_stack(
                        [
                            stayers[c].to_numpy(dtype=float)
                            - stayers[f"{c}_prev"].to_numpy(dtype=float)
                            for c in controls
                        ]
                    )
                    dy_switch = dy_switch - dX_switch @ theta
                    dy_stay = dy_stay - dX_stay @ theta

            # Each switcher enters with the sign of its own treatment change
            # and the cell is weighted by the total absolute change, so that
            # DID_M is the effect per unit of treatment (dCDH 2020, the
            # non-binary case). With a binary treatment the change is +1 from
            # baseline 0 and -1 from baseline 1, and this is the switcher-
            # count-weighted average of signed cell DIDs it has always been.
            dd_switch = switchers[treatment].to_numpy(dtype=float) - float(base)
            sgn = np.sign(dd_switch)
            if w_sw is None:
                abs_change = float(np.abs(dd_switch).sum())
                numerator = float(np.sum(sgn * (dy_switch - np.mean(dy_stay))))
            else:
                abs_change = float(np.sum(w_sw * np.abs(dd_switch)))
                numerator = float(
                    np.sum(w_sw * sgn * (dy_switch - _wmean(dy_stay, w_st)))
                )
                n_switch = int(round(w_sw.sum()))
            did_gt = numerator / abs_change
            if np.all(sgn > 0):
                direction = "on"
            elif np.all(sgn < 0):
                direction = "off"
            else:
                direction = "mixed"

            cell_estimates.append(
                {
                    "time": t_curr,
                    "time_prev": t_prev,
                    "baseline_treatment": float(base),
                    "n_switchers": n_switch,
                    "n_stayers": (
                        len(stayers) if w_st is None else int(round(w_st.sum()))
                    ),
                    "did_gt": did_gt,
                    "mean_dy_switch": _wmean(dy_switch, w_sw),
                    "mean_dy_stay": _wmean(dy_stay, w_st),
                    "direction": direction,
                    "treatment_change": abs_change,
                }
            )
            total_switchers += n_switch

    if total_switchers == 0:
        return {
            # Not estimable: missing, not zero (a bootstrap replicate without
            # a switching cell is then left out of the standard error).
            "did_m": float("nan"),
            "n_switching_cells": 0,
            "n_switchers": 0,
            "cell_estimates": [],
        }

    # Weighted average, weights = total absolute treatment change of the cell
    total_change = sum(ce["treatment_change"] for ce in cell_estimates)
    did_m = sum(
        ce["did_gt"] * ce["treatment_change"] / total_change for ce in cell_estimates
    )

    return {
        "did_m": float(did_m),
        "n_switching_cells": len(cell_estimates),
        "n_switchers": total_switchers,
        "cell_estimates": cell_estimates,
    }


def _control_diffs(
    xpivs: List[pd.DataFrame], idx: Any, t_a: Any, t_b: Any
) -> np.ndarray:
    """``X_{t_a} - X_{t_b}`` for the groups ``idx``, one column per control."""
    return np.column_stack(
        [(xp.loc[idx, t_a] - xp.loc[idx, t_b]).to_numpy(dtype=float) for xp in xpivs]
    )


def _refit_slopes(
    stayer_sets: List[Tuple[Any, float, Any, Any, Any]],
    ypiv: pd.DataFrame,
    xpivs: List[pd.DataFrame],
    wpiv: Optional[pd.DataFrame],
) -> Dict[float, np.ndarray]:
    """Slopes refitted on one placebo's or dynamic effect's own stayers.

    ``stayer_sets`` holds ``(t, baseline, stayers, t_a, t_b)`` for every
    switching period ``t``: the regression is of ``Y_{t_a} - Y_{t_b}`` on
    ``X_{t_a} - X_{t_b}`` and period effects over all of them, per baseline,
    each cell weighted by its size at ``t`` (R ``DIDmultiplegt`` 0.1.4, black
    box: it runs the DID_M machinery, controls included, on the lagged or
    long differences).
    """
    by_base: Dict[float, List[Tuple[np.ndarray, np.ndarray, np.ndarray, Any]]] = {}
    for t_c, base, st, t_a, t_b in stayer_sets:
        if len(st) == 0:
            continue
        dy = (ypiv.loc[st, t_a] - ypiv.loc[st, t_b]).to_numpy(dtype=float)
        dx = _control_diffs(xpivs, st, t_a, t_b)
        w = (
            wpiv.loc[st, t_c].to_numpy(dtype=float)
            if wpiv is not None
            else np.ones(len(st))
        )
        by_base.setdefault(float(base), []).append((dy, dx, w, t_c))
    slopes: Dict[float, np.ndarray] = {}
    for base, parts in by_base.items():
        period = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(parts)])
        slopes[base] = period_fe_slopes(
            np.vstack([p[1] for p in parts]),
            period,
            np.concatenate([p[0] for p in parts]),
            np.concatenate([p[2] for p in parts]),
        )
    return slopes


def _controls_complete(xpivs: List[pd.DataFrame], idx: Any, t_a: Any, t_b: Any) -> Any:
    """The groups of ``idx`` whose controls are observed at both dates."""
    keep = np.ones(len(idx), dtype=bool)
    for xp in xpivs:
        keep &= (xp.loc[idx, t_a].notna() & xp.loc[idx, t_b].notna()).to_numpy()
    return idx[keep]


def _estimate_placebo(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    lag: int,
    placebo_sign: str = "stata",
    w: Optional[str] = None,
    placebo_controls: str = "refit",
    main_slopes: Optional[Dict[float, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Placebo test at lag *l* (de Chaisemartin--D'Haultfoeuille 2020).

    For switchers at time t, the first-difference placebo compares the
    pre-treatment outcome change over (t-l-1, t-l) for switchers vs stayers,
    CONDITIONING on the baseline treatment d_{t-1} (switchers are compared only
    to same-baseline stayers; switch-off cells enter with a flipped sign).

    ⚠️ **Fixed**: every unit in the cell must have held its treatment fixed
    over the ``l`` periods BEFORE the switch (``placebo_cond`` in the
    reference). A unit that was already moving is not a clean pre-trend
    observation. This is invisible on absorbing panels — which is why it
    survived — and matters as soon as treatment can switch back.

    ⚠️ **Not fixed, because there is nothing to fix**: the SIGN is a genuine
    disagreement between dCDH's own two implementations. On ``did::mpdta``
    both report ``|placebo_1| = 0.024269`` and their three effects agree to
    six decimals, but Stata's ``did_multiplegt_old`` reports ``+0.024269``
    and R's ``DIDmultiplegt`` 0.1.4 reports ``-0.024269``. ``placebo_sign``
    selects between them; the default preserves the Stata convention this
    function has always used. See MIGRATION.md.

    With ``controls`` the placebo differences become ``dY - dX @ theta_d``.
    ``placebo_controls="refit"`` (R ``DIDmultiplegt`` 0.1.4) fits
    ``theta_d`` on this placebo's own stayers and lagged differences;
    ``"main"`` (Stata ``did_multiplegt_old``) reuses ``main_slopes``, the
    slopes of the main DID_M estimate.
    """
    periods = sorted(df[time].unique())
    dpiv = df.pivot_table(index=group, columns=time, values=treatment, aggfunc="first")
    ypiv = df.pivot_table(index=group, columns=time, values=y, aggfunc="first")
    wpiv = (
        df.pivot_table(index=group, columns=time, values=w, aggfunc="first")
        if w
        else None
    )
    xpivs = [
        df.pivot_table(index=group, columns=time, values=c, aggfunc="first")
        for c in (controls or [])
    ]

    estimates: List[float] = []
    weights: List[float] = []
    cells: List[Tuple[Any, float, Any, Any, Any, Any]] = []

    for i in range(1, len(periods)):
        t_prev, t_curr = periods[i - 1], periods[i]  # switching step (t-1 -> t)
        e_idx = i - lag  # placebo "end" period index
        b_idx = e_idx - 1  # placebo "start" period index
        if b_idx < 0:
            continue
        t_end, t_base = periods[e_idx], periods[b_idx]

        common = dpiv.index[dpiv[t_prev].notna() & dpiv[t_curr].notna()]
        for base in sorted(set(dpiv.loc[common, t_prev].dropna().unique())):
            in_base = common[dpiv.loc[common, t_prev] == base]
            sw = in_base[dpiv.loc[in_base, t_curr] != base]  # switchers from baseline
            st = in_base[dpiv.loc[in_base, t_curr] == base]  # stayers at baseline
            pre_window = periods[b_idx:i]  # t_base .. t_prev
            if lag > 0 and len(pre_window) > 1:
                # Treatment must be flat over the pre-window for both arms.
                sw = sw[
                    (dpiv.loc[sw, pre_window].eq(dpiv.loc[sw, t_prev], axis=0)).all(
                        axis=1
                    )
                ]
                st = st[
                    (dpiv.loc[st, pre_window].eq(dpiv.loc[st, t_prev], axis=0)).all(
                        axis=1
                    )
                ]
            sw = sw[ypiv.loc[sw, t_end].notna() & ypiv.loc[sw, t_base].notna()]
            st = st[ypiv.loc[st, t_end].notna() & ypiv.loc[st, t_base].notna()]
            if xpivs:
                sw = _controls_complete(xpivs, sw, t_end, t_base)
                st = _controls_complete(xpivs, st, t_end, t_base)
            cells.append((t_curr, base, sw, st, t_end, t_base))

    theta: Dict[float, np.ndarray] = {}
    if xpivs:
        theta = (
            dict(main_slopes or {})
            if placebo_controls == "main"
            else _refit_slopes(
                [(c[0], c[1], c[3], c[4], c[5]) for c in cells], ypiv, xpivs, wpiv
            )
        )

    for t_curr, base, sw, st, t_end, t_base in cells:
        if len(sw) == 0 or len(st) == 0:
            continue
        dd = dpiv.loc[sw, t_curr].to_numpy(dtype=float) - float(base)
        pdiff = (ypiv.loc[sw, t_end] - ypiv.loc[sw, t_base]).to_numpy(dtype=float)
        st_diff = ypiv.loc[st, t_end] - ypiv.loc[st, t_base]
        if xpivs:
            th = theta.get(float(base))
            if th is None:
                continue
            pdiff = pdiff - _control_diffs(xpivs, sw, t_end, t_base) @ th
            st_diff = st_diff - _control_diffs(xpivs, st, t_end, t_base) @ th
        if wpiv is None:
            pt = float(st_diff.mean())
            abs_change = float(np.abs(dd).sum())
            signed = float(np.sum(np.sign(dd) * (pdiff - pt))) / abs_change
        else:
            # cells weigh their size in the switching period (Stata and R)
            w_sw = wpiv.loc[sw, t_curr].to_numpy(dtype=float)
            w_st = wpiv.loc[st, t_curr].to_numpy(dtype=float)
            pt = _wmean(st_diff.to_numpy(dtype=float), w_st)
            abs_change = float(np.sum(w_sw * np.abs(dd)))
            signed = float(np.sum(w_sw * np.sign(dd) * (pdiff - pt))) / abs_change
        # ⚠️ dCDH's own Stata and R implementations disagree here. On
        # did::mpdta both return |placebo_1| = 0.024269, and the three
        # effects agree to six decimals, but Stata reports +0.024269 and
        # DIDmultiplegt 0.1.4 reports -0.024269. Neither is "the" answer,
        # so the convention is a parameter and the default preserves what
        # this function has always reported.
        flip = -1.0 if placebo_sign == "stata" else 1.0
        estimates.append(flip * signed)
        weights.append(abs_change)

    if not estimates:
        # No switching cell with a comparison cell: not estimable. Zero
        # would read as an estimated null effect.
        return {"estimate": float("nan"), "n_cells": 0}

    total_w = sum(weights)
    weighted_est = sum(e * w / total_w for e, w in zip(estimates, weights))

    return {"estimate": float(weighted_est), "n_cells": len(estimates)}


def _estimate_dynamic(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    horizon: int,
    w: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Dynamic effect at horizon *l*, pair by pair (R ``DIDmultiplegt`` 0.1.4).

    For switchers at time t (relative to baseline d_{t-1}), the long-difference
    Y_{t+l} - Y_{t-1} is compared to that of ROBUST stayers: units that share the
    baseline treatment AND keep it unchanged over the whole window [t, t+l] (so
    the control trend is not contaminated by units treated during the horizon).
    Switch-off cells enter with a flipped sign.

    ⚠️ Switchers must also hold their NEW treatment through the window. A unit
    that switches at t and switches again at t+1 has no well-defined "effect of
    having switched l periods ago", and the reference implementation excludes
    it (``dynamic_cond``). Including such units is what this used to do; see
    MIGRATION.md.

    Cells of several rows weigh their size at the switching period ``t``, and
    ``controls`` are netted out with slopes refitted on the long differences
    of these robust stayers (both R, black box). The first-time-switcher
    estimator of Stata ``did_multiplegt_old, robust_dynamic`` is
    :func:`._dcdh_first_switch.first_switch_effect`.
    """
    periods = sorted(df[time].unique())
    dpiv = df.pivot_table(index=group, columns=time, values=treatment, aggfunc="first")
    ypiv = df.pivot_table(index=group, columns=time, values=y, aggfunc="first")
    wpiv = (
        df.pivot_table(index=group, columns=time, values=w, aggfunc="first")
        if w
        else None
    )
    xpivs = [
        df.pivot_table(index=group, columns=time, values=c, aggfunc="first")
        for c in (controls or [])
    ]

    estimates: List[float] = []
    weights: List[float] = []
    cells: List[Tuple[Any, float, Any, Any, Any, Any]] = []

    for i in range(1, len(periods)):
        t_prev, t_curr = periods[i - 1], periods[i]
        f_idx = i + horizon
        if f_idx >= len(periods):
            continue
        t_future = periods[f_idx]
        window = periods[i : f_idx + 1]  # t_curr .. t_future

        common = dpiv.index[dpiv[t_prev].notna() & dpiv[t_curr].notna()]
        for base in sorted(set(dpiv.loc[common, t_prev].dropna().unique())):
            in_base = common[dpiv.loc[common, t_prev] == base]
            sw = in_base[dpiv.loc[in_base, t_curr] != base]
            if horizon > 0 and len(sw) > 0:
                # Switchers must stay at the level they switched TO for the
                # whole horizon; a second switch inside the window makes the
                # horizon-l effect undefined.
                held = (dpiv.loc[sw, window].eq(dpiv.loc[sw, t_curr], axis=0)).all(
                    axis=1
                )
                sw = sw[held]
            # robust stayers: keep the baseline treatment over the whole window
            stable = (dpiv.loc[in_base, window] == base).all(axis=1)
            st = in_base[stable]
            sw = sw[ypiv.loc[sw, t_future].notna() & ypiv.loc[sw, t_prev].notna()]
            st = st[ypiv.loc[st, t_future].notna() & ypiv.loc[st, t_prev].notna()]
            if xpivs:
                sw = _controls_complete(xpivs, sw, t_future, t_prev)
                st = _controls_complete(xpivs, st, t_future, t_prev)
            cells.append((t_curr, base, sw, st, t_future, t_prev))

    theta: Dict[float, np.ndarray] = {}
    if xpivs:
        theta = _refit_slopes(
            [(c[0], c[1], c[3], c[4], c[5]) for c in cells], ypiv, xpivs, wpiv
        )

    for t_curr, base, sw, st, t_future, t_prev in cells:
        if len(sw) == 0 or len(st) == 0:
            continue
        dd = dpiv.loc[sw, t_curr].to_numpy(dtype=float) - float(base)
        ldiff = (ypiv.loc[sw, t_future] - ypiv.loc[sw, t_prev]).to_numpy(dtype=float)
        st_diff = ypiv.loc[st, t_future] - ypiv.loc[st, t_prev]
        if xpivs:
            th = theta.get(float(base))
            if th is None:
                continue
            ldiff = ldiff - _control_diffs(xpivs, sw, t_future, t_prev) @ th
            st_diff = st_diff - _control_diffs(xpivs, st, t_future, t_prev) @ th
        if wpiv is None:
            ldt = float(st_diff.mean())
            abs_change = float(np.abs(dd).sum())
            estimates.append(float(np.sum(np.sign(dd) * (ldiff - ldt))) / abs_change)
        else:
            # cells weigh their size in the switching period t (R, black
            # box; Stata's first-time-switcher DID_l weighs t + l instead)
            w_sw = wpiv.loc[sw, t_curr].to_numpy(dtype=float)
            w_st = wpiv.loc[st, t_curr].to_numpy(dtype=float)
            ldt = _wmean(st_diff.to_numpy(dtype=float), w_st)
            abs_change = float(np.sum(w_sw * np.abs(dd)))
            estimates.append(
                float(np.sum(w_sw * np.sign(dd) * (ldiff - ldt))) / abs_change
            )
        weights.append(abs_change)

    if not estimates:
        # No switching cell with a comparison cell: not estimable. Zero
        # would read as an estimated null effect.
        return {"estimate": float("nan"), "n_cells": 0}

    total_w = sum(weights)
    weighted_est = sum(e * w / total_w for e, w in zip(estimates, weights))

    return {"estimate": float(weighted_est), "n_cells": len(estimates)}
