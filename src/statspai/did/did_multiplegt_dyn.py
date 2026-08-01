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

Identification details
----------------------
Items 2-4 below are now pinned against the authors' R package
``DIDmultiplegtDYN`` 2.3.4 (Track A module ``78_multiplegt_dyn``,
``tests/reference_parity/test_multiplegt_dyn_parity.py``): on a
binary absorbing design the effect and placebo estimates agree to
machine precision, which settles the window and weighting conventions
that used to carry ``[待核验]`` markers. Items 1 and 5 remain open and
are what keeps this estimator off a paper-faithful claim.

1. **Switcher definition**: units first switching from d=0 to d=1 at
   period F. First-cut skips switch-off events; the paper handles both
   directions via a sign convention that is NOT implemented here.
   [待核验 — paper §2.x]. The parity above therefore only covers
   absorbing treatment.

2. **Control group per horizon l**: "not-yet-treated at F+l" = units
   whose d stays at its pre-F value through F+l inclusive, which is
   what reproduces the R package's per-horizon samples (matching
   switcher and observation counts, not just point estimates). The
   never-treated-only variant is exposed as ``control='never_treated'``
   and is not separately pinned.

3. **Per-horizon estimate**:

       δ_l = Σ_F w_F × {E[Y_{F+l} − Y_{F−1} | switchers at F]
                        − E[Y_{F+l} − Y_{F−1} | not-yet-treated at F+l]}

   with weights ``w_F`` proportional to the number of switchers at F.
   Confirmed by parity. The heteroskedastic-weights variant (dCDH 2023
   EJ survey) is still not implemented.

4. **Placebo lag l < 0**: the effect window reflected about F-1, i.e.
   ``Y_{F-1-|l|} − Y_{F-1}``, reported with the reverse sign so it sits
   on the same event-study scale as the effects. Confirmed by parity
   against ``Placebo_|l|``.

   .. versionchanged:: 1.21.0
      ⚠️ This used to be ``Y_{F-1-|l|} − Y_{F-1-|l|-1}`` -- a
      one-period difference sliding backwards rather than a mirrored
      long difference. That is a different quantity, it silently used
      fewer cohorts at each lag, and it did not match the reference.
      Every placebo number changes. See MIGRATION.md.

5. **Inference**: analytical influence-function variance per horizon is
   not implemented in this MVP — SE comes from cluster bootstrap on
   the panel unit. The paper's IF variance is [待核验] and is the clear
   next step. Standard errors are consequently NOT pinned against the
   R package, which reports analytical ones.

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

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core._bootstrap import bootstrap_se as _bootstrap_se
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
    aggregation: str = "simple",
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
        Bootstrap replications. Analytical IF variance [待核验] pending,
        so standard errors are not comparable to DIDmultiplegtDYN's.
    alpha : float, default 0.05
    seed : int, optional
    aggregation : {"simple", "switchers"}, default "simple"
        How the dynamic horizons are combined into the headline
        ``estimate``. ``"simple"`` gives each horizon equal weight;
        ``"switchers"`` weights horizon ``l`` by the number of switchers
        contributing to it, which reproduces ``DIDmultiplegtDYN``'s
        ``Av_tot_eff``. The two differ whenever later horizons rest on
        fewer cohorts, which is the normal case in staggered designs.
        The default is left on ``"simple"`` because changing it would
        move the number existing callers get back;
        ``model_info["aggregation"]`` records which was used.

    Returns
    -------
    CausalResult with ``detail`` = per-event decomposition and
    ``model_info['event_study']`` = horizon-level DataFrame matching the
    canonical event-study schema (so ``sp.did_plot`` works).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(20):
    ...     g = int(rng.choice([5, 8, 0]))  # cohort; 0 = never treated
    ...     for t in range(1, 13):
    ...         d = 1 if (g != 0 and t >= g) else 0
    ...         rows.append({'i': i, 't': t, 'd': d,
    ...                      'y': i + 0.1 * t + 1.0 * d + rng.normal(0, 0.4)})
    >>> df = pd.DataFrame(rows)
    >>> r = sp.did_multiplegt_dyn(
    ...     df, y='y', group='i', time='t', treatment='d',
    ...     placebo=2, dynamic=4, n_boot=50, seed=0,
    ... )
    >>> es = r.model_info['event_study']  # horizon-level event study
    >>> bool(len(es) > 0)
    True
    >>> sens = sp.honest_did(r, m_grid=[0.5])  # Rambachan-Roth sensitivity
    """
    if control not in {"not_yet_treated", "never_treated"}:
        raise ValueError(
            f"control={control!r} must be 'not_yet_treated' or 'never_treated'"
        )
    if dynamic < 0 or placebo < 0:
        raise ValueError("dynamic and placebo must be non-negative")
    if aggregation not in {"simple", "switchers"}:
        raise ValueError(
            f"aggregation must be 'simple' or 'switchers', got {aggregation!r}"
        )

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
    # l = -1 is a genuine placebo, not a mechanical zero: it is
    # Y_{F-2} - Y_{F-1} differenced against the controls, which is only
    # zero in expectation under parallel trends. It maps to the R
    # package's Placebo_1.

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
            continue  # replicate stays NaN; bootstrap_se tracks the failure

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
        se = _bootstrap_se(boot_hist[:, j], label=f"did.multiplegt_dyn[h={h}]")
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

    # Headline estimate over the dynamic horizons. "simple" gives each
    # horizon equal weight; "switchers" weights by the switchers behind
    # each one, which is DIDmultiplegtDYN's Av_tot_eff.
    dyn_est = np.array(
        [es_rows[j]["att"] for j in dyn_idx],
        dtype=float,
    )
    if not dyn_est.size:
        headline = np.nan
    elif aggregation == "switchers":
        w = np.array(
            [float(main["cell_estimates"][j]["n_switchers"]) for j in dyn_idx],
            dtype=float,
        )
        ok = np.isfinite(dyn_est) & (w > 0)
        headline = (
            float(np.sum(w[ok] * dyn_est[ok]) / np.sum(w[ok])) if ok.any() else np.nan
        )
    else:
        headline = float(np.nanmean(dyn_est))
    # SE: cross-horizon bootstrap of the average. A replicate contributes
    # only when at least one dynamic horizon was estimated; a fully-failed
    # draw stays NaN so bootstrap_se can surface the failure rate.
    if dyn_idx:
        with warnings.catch_warnings():
            # nanmean of an all-NaN replicate row is an intended NaN.
            warnings.simplefilter("ignore", RuntimeWarning)
            if aggregation == "switchers":
                wb = np.array(
                    [float(main["cell_estimates"][j]["n_switchers"]) for j in dyn_idx],
                    dtype=float,
                )
                sub = boot_hist[:, dyn_idx]
                mask = np.isfinite(sub)
                denom = (mask * wb).sum(axis=1)
                num = np.nansum(np.where(mask, sub, 0.0) * wb, axis=1)
                boot_avg = np.where(
                    denom > 0, num / np.where(denom > 0, denom, 1.0), np.nan
                )
            else:
                boot_avg = np.nanmean(boot_hist[:, dyn_idx], axis=1)
        se_avg = _bootstrap_se(boot_avg, label="did.multiplegt_dyn.headline")
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
        estimand=(
            "Average dynamic effect across horizons 0..dynamic "
            f"({aggregation}-weighted)"
        ),
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
            "aggregation": aggregation,
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
        - If l < 0: compare Y_{F-1-|l|} − Y_{F-1} (placebo), i.e. the
          mirror image of the l = |l| - 1 effect, run backwards over the
          pre-period. Matches DIDmultiplegtDYN's Placebo_|l|.
      Control set depends on `control=`.

    Aggregate per horizon with n_switchers weights.
    """
    cells: List[Dict[str, Any]] = []

    F_values = sorted(df["_F"].dropna().unique())
    if not F_values:
        return {"cell_estimates": []}

    # Never-treated set (units with _F NaN)
    never_ids = set(df[df["_F"].isna()][group].unique())
    # Earliest period actually observed -- a horizon whose base period
    # falls before it has no data and is skipped, exactly as the R
    # package drops the cohorts that cannot support a given placebo.
    t_min = float(df[time].min())

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
                # dCDH placebo l: the effect window reflected about F-1,
                # Y_{F-1-l} -> Y_{F-1}. Reported with the reverse sign so
                # it sits on the same event-study scale as the effects
                # (this is DIDmultiplegtDYN's convention).
                t_pre, t_post = F - 1 - abs(h), F - 1
            if t_pre < t_min:
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
            if h < 0:
                delta_F_l = -delta_F_l
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
