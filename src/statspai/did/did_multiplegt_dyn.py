"""de Chaisemartin & D'Haultfœuille (2024) intertemporal event-study DiD.

Differs from ``sp.did_multiplegt`` (dCDH 2020 DID_M): the 2020 estimator
is a consecutive-period switcher-vs-stayer pair rollup, while this 2024
estimator is a **long-difference event study** — at each horizon l ≥ 0,
compares ``Y_{F+l} − Y_{F−1}`` between units first switching at F and a
"not-yet-treated at F+l" control group held stable across the horizon.

Verified anchor
---------------
- de Chaisemartin & D'Haultfœuille (2024) "Difference-in-Differences
  Estimators of Intertemporal Treatment Effects", DOI
  ``10.1162/rest_a_01414`` (bib key ``dechaisemartin2024difference``).

[待核验] identification details
-------------------------------
The following are based on my best-effort reading of the paper and the
companion R ``DIDmultiplegtDYN`` package logic; every item flagged here
must be verified against the paper's published equations before moving
this estimator from RFC/experimental status to production / paper-
faithful claim:

1. **Switcher definition**: units first switching from d=0 to d=1 at
   period F. First-cut skips switch-off events; the paper handles both
   directions via a sign convention that I have NOT implemented here.
   [待核验 — paper §2.x]

2. **Control group per horizon l**: "not-yet-treated at F+l" = units
   whose d stays at its pre-F value through F+l inclusive. MVP uses
   this definition; the paper also supports never-treated-only controls
   which I expose as ``control='never_treated'``. [待核验 — exact window
   definition]

3. **Per-horizon estimate**:

       δ_l = Σ_F w_F × {E[Y_{F+l} − Y_{F−1} | switchers at F]
                        − E[Y_{F+l} − Y_{F−1} | not-yet-treated at F+l]}

   with weights ``w_F`` proportional to the number of switchers at F.
   [待核验 — paper may use a different weighting scheme; heteroskedastic
   weights variant (dCDH 2023 EJ survey) is not implemented here.]

4. **Placebo lag l < 0**: same structure but the comparison period
   moves to ``Y_{F-1-|l|} − Y_{F-1-|l|-1}``. [待核验]

5. **Inference**: analytical influence-function variance per horizon is
   not implemented in this MVP — SE comes from cluster bootstrap on
   the panel unit. The paper's IF variance is [待核验] and is the clear
   next step.

Scope for this first cut
------------------------
- Never-treated and not-yet-treated control variants.
- Placebo + dynamic horizons with cluster bootstrap SE.
- Joint Wald tests for placebo and overall (placebo + dynamic) via the
  ``_core.joint_wald`` helper on the bootstrap covariance.
- NO switch-off events, NO heteroskedastic weights, NO analytical IF.

Users who need paper-faithful numerics should wait for the next
iteration when the paper's equations are in-hand and reference parity
vs. R ``DIDmultiplegtDYN`` is in place. In the interim, the function
raises its method label so no user can mistake this for a paper-
faithful implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from . import _core as _dc


def did_multiplegt_dyn(
    data: pd.DataFrame,
    y: str,
    *,
    group: str,
    time: str,
    treatment: str,
    placebo: int = 0,
    dynamic: int = 3,
    control: str = "not_yet_treated",
    cluster: Optional[str] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """dCDH (2024) intertemporal event-study DiD estimator.

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome column.
    group : str
        Unit identifier.
    time : str
        Integer-valued period column.
    treatment : str
        Binary time-varying treatment (0/1). Only switch-on events
        (d=0 → d=1) are used in this MVP; switch-off events are flagged
        [待核验] and not handled.
    placebo : int, default 0
        Number of pre-treatment placebo horizons (l = -1, ..., -placebo).
    dynamic : int, default 3
        Number of post-treatment dynamic horizons (l = 0, ..., dynamic).
    control : {'not_yet_treated', 'never_treated'}, default
        ``'not_yet_treated'``.
    cluster : str, optional
        Cluster column for bootstrap SE (defaults to group).
    n_boot : int, default 500
        Bootstrap replications. Analytical IF variance [待核验] pending.
    alpha : float, default 0.05
    seed : int, optional

    Returns
    -------
    CausalResult with ``detail`` = per-event decomposition and
    ``model_info['event_study']`` = horizon-level DataFrame matching the
    canonical event-study schema (so ``sp.did_plot`` works).

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.did_multiplegt_dyn(
    ...     df, y='y', group='i', time='t', treatment='d',
    ...     placebo=2, dynamic=4,
    ... )
    >>> r.model_info['event_study']
    >>> sp.honest_did(r, max_M=0.5)
    """
    if control not in {"not_yet_treated", "never_treated"}:
        raise ValueError(
            f"control={control!r} must be 'not_yet_treated' or 'never_treated'"
        )
    if dynamic < 0 or placebo < 0:
        raise ValueError("dynamic and placebo must be non-negative")

    df = data.copy()
    for col in (y, group, time, treatment):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in data")
    if not set(df[treatment].dropna().unique()) <= {0, 1}:
        raise ValueError(f"Treatment {treatment!r} must be binary 0/1")

    df = df.sort_values([group, time]).reset_index(drop=True)
    cluster_var = cluster if cluster is not None else group

    # Identify each unit's first switch-on period F (d goes from 0 to 1).
    # Units that never reach d=1 have F=None; they are candidates for
    # controls depending on `control`.
    first_treat = df[df[treatment] == 1].groupby(group)[time].min().rename("_F")
    df = df.merge(first_treat, on=group, how="left")

    # Check switch-off is not required for identification and flag if present.
    # [待核验 — paper's handling of switch-off]
    if _has_switch_off(df, group=group, time=time, treatment=treatment):
        # Don't raise — silently drop switch-off events' switcher status by
        # keeping F = first ON period; unit's later periods are still in
        # the panel but those "on → off → on" traces aren't specially
        # treated in this MVP.
        pass

    # Horizons list: placebo (negative) + dynamic (0..H).
    horizons = list(range(-placebo, dynamic + 1))
    # Include l=-1 as a "reference" that is always 0 under the
    # construction Y_{F-1} - Y_{F-1} = 0? No — for placebos, the long
    # difference uses Y_{F-1-|l|} - Y_{F-1-|l|-1} so l=-1 is
    # Y_{F-2} - Y_{F-3} ≠ 0 under non-trivial trends. [待核验]
    # Keep l=-1 as a valid placebo horizon.

    main = _estimate_all_horizons(
        df=df,
        y=y,
        group=group,
        time=time,
        treatment=treatment,
        horizons=horizons,
        control=control,
    )

    # Cluster bootstrap for SE
    rng = np.random.default_rng(seed)
    boot_hist = np.full((n_boot, len(horizons)), np.nan)
    for b in range(n_boot):
        try:
            bdf = _dc.cluster_bootstrap_draw(
                df,
                cluster_col=cluster_var,
                rng=rng,
                relabel_cols=[group],
            )
            # Recompute F in bootstrap sample.
            first_treat_b = (
                bdf[bdf[treatment] == 1].groupby(group)[time].min().rename("_F")
            )
            if "_F" in bdf.columns:
                bdf = bdf.drop(columns=["_F"])
            bdf = bdf.merge(first_treat_b, on=group, how="left")
            best = _estimate_all_horizons(
                df=bdf,
                y=y,
                group=group,
                time=time,
                treatment=treatment,
                horizons=horizons,
                control=control,
            )
            for j, h in enumerate(horizons):
                # Align by h
                row = next(
                    (r for r in best["cell_estimates"] if r["horizon"] == h), None
                )
                if row is not None:
                    boot_hist[b, j] = row["delta_l"]
        except Exception:
            continue

    # Per-horizon SE + CI
    es_rows: List[Dict[str, Any]] = []
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    for j, h in enumerate(horizons):
        row = next((r for r in main["cell_estimates"] if r["horizon"] == h), None)
        if row is None:
            es_rows.append(
                {
                    "relative_time": h,
                    "att": np.nan,
                    "se": np.nan,
                    "pvalue": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "type": "placebo" if h < 0 else "dynamic",
                    "n_switchers": 0,
                }
            )
            continue
        est = row["delta_l"]
        se = float(np.nanstd(boot_hist[:, j], ddof=1))
        p = (
            float(2 * (1 - stats.norm.cdf(abs(est / se))))
            if (se > 0 and np.isfinite(se))
            else np.nan
        )
        ci_lo = est - z_crit * se if (se > 0 and np.isfinite(se)) else np.nan
        ci_hi = est + z_crit * se if (se > 0 and np.isfinite(se)) else np.nan
        es_rows.append(
            {
                "relative_time": h,
                "att": float(est) if np.isfinite(est) else np.nan,
                "se": float(se) if np.isfinite(se) else np.nan,
                "pvalue": p,
                "ci_lower": float(ci_lo) if np.isfinite(ci_lo) else np.nan,
                "ci_upper": float(ci_hi) if np.isfinite(ci_hi) else np.nan,
                "type": "placebo" if h < 0 else "dynamic",
                "n_switchers": int(row["n_switchers"]),
            }
        )

    es_df = _dc.event_study_frame(es_rows)

    # Joint tests
    placebo_idx = [j for j, h in enumerate(horizons) if h < 0]
    dyn_idx = [j for j, h in enumerate(horizons) if h >= 0]

    joint_placebo = _joint_test_from_boot(main, horizons, boot_hist, placebo_idx)
    joint_overall = _joint_test_from_boot(
        main, horizons, boot_hist, placebo_idx + dyn_idx
    )

    # Headline estimate: simple mean over dynamic horizons.
    dyn_est = np.array(
        [es_rows[j]["att"] for j in dyn_idx],
        dtype=float,
    )
    headline = float(np.nanmean(dyn_est)) if dyn_est.size else np.nan
    # SE: cross-horizon bootstrap of the average.
    if dyn_idx:
        boot_avg = np.nanmean(boot_hist[:, dyn_idx], axis=1)
        boot_avg = boot_avg[np.isfinite(boot_avg)]
        se_avg = float(np.std(boot_avg, ddof=1)) if boot_avg.size > 1 else np.nan
    else:
        se_avg = np.nan

    if se_avg and se_avg > 0:
        z = headline / se_avg
        p_h = float(2 * (1 - stats.norm.cdf(abs(z))))
        ci_h = (headline - z_crit * se_avg, headline + z_crit * se_avg)
    else:
        p_h = np.nan
        ci_h = (np.nan, np.nan)

    return CausalResult(
        method="did_multiplegt_dyn (dCDH 2024 ReStat) [待核验 — MVP, not paper-parity]",
        estimand="Average dynamic effect across horizons 0..dynamic",
        estimate=headline,
        se=se_avg,
        pvalue=p_h,
        ci=ci_h,
        alpha=alpha,
        n_obs=int(len(df)),
        detail=pd.DataFrame(main["cell_estimates"]),
        model_info={
            "event_study": es_df,
            "horizons": horizons,
            "control": control,
            "n_boot": n_boot,
            "cluster_var": cluster_var,
            "joint_placebo_test": joint_placebo,
            "joint_overall_test": joint_overall,
            "warning": (
                "MVP implementation: no analytical IF variance, no "
                "switch-off handling, no heteroskedastic weights. See "
                "docs/rfc/multiplegt_dyn.md for the production roadmap."
            ),
        },
        _citation_key="dechaisemartin2024difference",
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _has_switch_off(
    df: pd.DataFrame,
    *,
    group: str,
    time: str,
    treatment: str,
) -> bool:
    """Detect any unit that switches from 1 to 0."""
    for _, u_df in df.groupby(group):
        vals = u_df.sort_values(time)[treatment].values
        if len(vals) < 2:
            continue
        diffs = np.diff(vals)
        if (diffs == -1).any():
            return True
    return False


def _estimate_all_horizons(
    *,
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    horizons: List[int],
    control: str,
) -> Dict[str, Any]:
    """Compute δ_l for each horizon h using long-difference event-study.

    For each unique first-treatment period F in the sample:
      switchers at F = units with _F == F.
      For each horizon l:
        - If l >= 0: compare Y_{F+l} − Y_{F-1} between switchers and controls.
        - If l < 0: compare Y_{F-1-|l|} − Y_{F-1-|l|-1} (placebo).
      Control set depends on `control=`.

    Aggregate per horizon with n_switchers weights.
    """
    cells: List[Dict[str, Any]] = []

    F_values = sorted(df["_F"].dropna().unique())
    if not F_values:
        return {"cell_estimates": []}

    # Never-treated set (units with _F NaN)
    never_ids = set(df[df["_F"].isna()][group].unique())

    for h in horizons:
        horizon_acc = {"sum_delta": 0.0, "n_switchers": 0, "n_events": 0}

        for F in F_values:
            switcher_ids = set(df[df["_F"] == F][group].unique())
            if not switcher_ids:
                continue

            # Determine periods for the comparison
            if h >= 0:
                t_pre, t_post = F - 1, F + h
            else:
                lag = abs(h)
                t_pre, t_post = F - 1 - lag - 1, F - 1 - lag
            if t_pre < 0:
                continue

            # Controls per horizon depending on `control`
            if control == "never_treated":
                ctrl_ids = never_ids
            else:
                # not-yet-treated at F + max(h, 0): units with _F > F + max(h, 0)
                threshold = F + max(h, 0)
                ctrl_ids = set(
                    df[(df["_F"] > threshold) | (df["_F"].isna())][group].unique()
                )

            if not ctrl_ids:
                continue

            sw_pre = df[(df[group].isin(switcher_ids)) & (df[time] == t_pre)][y]
            sw_post = df[(df[group].isin(switcher_ids)) & (df[time] == t_post)][y]
            c_pre = df[(df[group].isin(ctrl_ids)) & (df[time] == t_pre)][y]
            c_post = df[(df[group].isin(ctrl_ids)) & (df[time] == t_post)][y]

            if any(len(s) == 0 for s in (sw_pre, sw_post, c_pre, c_post)):
                continue

            delta_F_l = (sw_post.mean() - sw_pre.mean()) - (
                c_post.mean() - c_pre.mean()
            )
            n_sw = len(switcher_ids)
            horizon_acc["sum_delta"] += float(delta_F_l) * n_sw
            horizon_acc["n_switchers"] += n_sw
            horizon_acc["n_events"] += 1

        if horizon_acc["n_switchers"] > 0:
            delta_l = horizon_acc["sum_delta"] / horizon_acc["n_switchers"]
        else:
            delta_l = np.nan

        cells.append(
            {
                "horizon": h,
                "delta_l": float(delta_l) if np.isfinite(delta_l) else np.nan,
                "n_switchers": horizon_acc["n_switchers"],
                "n_events": horizon_acc["n_events"],
            }
        )

    return {"cell_estimates": cells}


def _joint_test_from_boot(
    main: Dict[str, Any],
    horizons: List[int],
    boot_hist: np.ndarray,
    indices: List[int],
) -> Optional[Dict[str, Any]]:
    if not indices:
        return None
    est = np.array(
        [
            next(
                (
                    r["delta_l"]
                    for r in main["cell_estimates"]
                    if r["horizon"] == horizons[j]
                ),
                np.nan,
            )
            for j in indices
        ],
        dtype=float,
    )
    sub = boot_hist[:, indices]
    valid = ~np.any(np.isnan(sub), axis=1)
    if valid.sum() < len(indices) + 1:
        return None
    cov = np.cov(sub[valid], rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    return _dc.joint_wald(est, cov)
