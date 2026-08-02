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

1. **Switcher definition**: each unit's FIRST treatment change, in either
   direction, at period F. Switch-off events are switchers too -- that is
   the design's whole point -- and they are handled the way the reference
   does: their controls must share the switcher's BASELINE treatment level
   (a unit going 1 -> 0 belongs against units that were at 1 and stayed),
   and the difference is divided by the change in treatment so both
   directions measure the same effect per unit of treatment.

   Pinned on a panel where treatment switches both ways: effects, placebo
   and every switcher count match the reference exactly. Earlier releases
   dropped switch-off events silently, which changed the estimand on any
   non-absorbing panel.

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

5. **Inference**: ``se_method='analytic'`` builds each horizon's influence
   function directly -- every horizon is a switcher-weighted sum of
   two-sample mean differences, so its influence function is the matching
   sum of within-group deviations, and the horizons are combined as
   functions rather than by adding variances (they share control units).

   That is a legitimate variance estimator and it agrees with this
   module's own cluster bootstrap, but it is **not the paper's formula**
   and is **not pinned** against ``DIDmultiplegtDYN``: on the parity
   fixture it runs about 1% below the package's reported standard errors
   at every horizon (worst 1.0%), and 1.7% below on the aggregate. The
   remaining gap is [待核验] -- dCDH's own variance derivation, which
   this does not reproduce. The default stays ``'bootstrap'``.

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
    se_method: str = "bootstrap",
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
    se_method : {"bootstrap", "analytic"}, default "bootstrap"
        ``"bootstrap"`` resamples clusters; ``"analytic"`` uses the
        influence functions and needs no draws, which makes it roughly a
        hundred times faster.

        The analytic variance is NOT pinned against ``DIDmultiplegtDYN``.
        It agrees with this module's own bootstrap and sits about 1% below
        the package's reported standard errors on the parity fixture; the
        residual is dCDH's own variance derivation, which is not
        reproduced here. The default stays on the bootstrap for that
        reason, and ``model_info["se_method"]`` records the choice.

        ``joint_placebo_test`` and ``joint_overall_test`` still come from
        the bootstrap, so they are ``None`` when ``n_boot=0``.
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
    if se_method not in {"bootstrap", "analytic"}:
        raise ValueError(
            f"se_method must be 'bootstrap' or 'analytic', got {se_method!r}"
        )

    df = data.copy()
    for col in (y, group, time, treatment):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in data")
    if not set(df[treatment].dropna().unique()) <= {0, 1}:
        raise ValueError(f"Treatment {treatment!r} must be binary 0/1")

    df = df.sort_values([group, time]).reset_index(drop=True)
    cluster_var = cluster if cluster is not None else group

    # Identify each unit's FIRST switch, in either direction, and which way
    # it went. A unit that turns treatment off is as much a switcher as one
    # that turns it on -- that is the whole point of the dCDH design, and
    # dropping those events (as this used to) silently changes the estimand.
    first_switch, switch_dir, baseline = _first_switch(
        df, group=group, time=time, treatment=treatment
    )
    df = df.merge(first_switch, on=group, how="left")
    df = df.merge(switch_dir, on=group, how="left")
    df = df.merge(baseline, on=group, how="left")

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
        if se_method == "analytic":
            se = row["_se_analytic"]
        else:
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
    if se_method == "analytic" and dyn_idx:
        # Combine the horizons' influence functions with the same weights the
        # headline uses, then square once -- the horizons share control units,
        # so adding their variances would understate the spread.
        rows_by_h = {r["horizon"]: r for r in main["cell_estimates"]}
        psis, wts = [], []
        for k, j in enumerate(dyn_idx):
            r_h = rows_by_h.get(horizons[j])
            if r_h is None or not np.isfinite(dyn_est[k]):
                continue
            psis.append(r_h["_influence"])
            wts.append(float(r_h["n_switchers"]) if aggregation == "switchers" else 1.0)
        if psis:
            wv = np.asarray(wts, dtype=float)
            wv = wv / wv.sum()
            psi_head = np.sum([w * p for w, p in zip(wv, psis)], axis=0)
            se_avg = float(np.sqrt(np.mean(psi_head**2) / len(psi_head)))

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
            "se_method": se_method,
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


def _first_switch(
    df: pd.DataFrame,
    *,
    group: str,
    time: str,
    treatment: str,
):
    """Each unit's first treatment change: when, which way, and from what.

    Returns three frames keyed on ``group``: ``_F`` (the period of the first
    change), ``_dir`` (+1 for a switch on, -1 for a switch off) and ``_base``
    (the treatment level held just before it). Units that never change get
    no row and are therefore eligible as controls.

    The baseline matters as much as the direction. A unit turning treatment
    OFF has to be compared with units that were also ON and stayed ON; using
    the never-treated as its control group compares two different
    counterfactuals and does not give the reference's answer.
    """
    F: Dict[Any, Any] = {}
    direction: Dict[Any, int] = {}
    base: Dict[Any, float] = {}
    for uid, u_df in df.sort_values([group, time]).groupby(group, sort=False):
        vals = u_df[treatment].to_numpy()
        times = u_df[time].to_numpy()
        if len(vals) < 2:
            continue
        changed = np.nonzero(np.diff(vals) != 0)[0]
        if changed.size == 0:
            continue
        k = int(changed[0]) + 1
        F[uid] = times[k]
        direction[uid] = 1 if vals[k] > vals[k - 1] else -1
        base[uid] = float(vals[k - 1])
    idx = pd.Index(list(F), name=group)
    return (
        pd.Series(list(F.values()), index=idx, name="_F").reset_index(),
        pd.Series(list(direction.values()), index=idx, name="_dir").reset_index(),
        pd.Series(list(base.values()), index=idx, name="_base").reset_index(),
    )


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

    # Unit-level index for the influence functions. Each horizon's estimate
    # is a weighted sum of two-sample mean differences, so its influence
    # function is the corresponding sum of within-group deviations -- and
    # summing rather than adding variances is what carries the fact that a
    # control unit can serve several events.
    all_units = pd.Index(sorted(df[group].unique()))
    n_panel = len(all_units)
    unit_pos = pd.Series(np.arange(n_panel), index=all_units)

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
        psi = np.zeros(n_panel, dtype=float)

        for F in F_values:
            for direction in (1, -1):
                _cell = _one_event(
                    df=df,
                    y=y,
                    group=group,
                    time=time,
                    treatment=treatment,
                    F=F,
                    h=h,
                    direction=direction,
                    control=control,
                    never_ids=never_ids,
                    t_min=t_min,
                    n_panel=n_panel,
                    unit_pos=unit_pos,
                )
                if _cell is None:
                    continue
                horizon_acc["sum_delta"] += _cell["delta"] * _cell["n_sw"]
                horizon_acc["n_switchers"] += _cell["n_sw"]
                horizon_acc["n_events"] += 1
                psi += _cell["psi"] * _cell["n_sw"]

        if horizon_acc["n_switchers"] > 0:
            delta_l = horizon_acc["sum_delta"] / horizon_acc["n_switchers"]
            psi = psi / horizon_acc["n_switchers"]
            se_analytic = float(np.sqrt(np.mean(psi**2) / n_panel))
        else:
            delta_l = np.nan
            se_analytic = np.nan

        cells.append(
            {
                "horizon": h,
                "delta_l": float(delta_l) if np.isfinite(delta_l) else np.nan,
                "n_switchers": horizon_acc["n_switchers"],
                "n_events": horizon_acc["n_events"],
                "_influence": psi,
                "_se_analytic": se_analytic,
            }
        )

    return {"cell_estimates": cells}


def _one_event(
    *,
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    F: Any,
    h: int,
    direction: int,
    control: str,
    never_ids: set,
    t_min: float,
    n_panel: int,
    unit_pos: pd.Series,
) -> Optional[Dict[str, Any]]:
    """One (switch period, direction) event at horizon ``h``.

    Three things distinguish a switch-off event from a switch-on one, and
    all three matter:

    * Its controls must share its BASELINE treatment level, not merely be
      untreated. A unit going 1 -> 0 belongs against units that were at 1
      and stayed there; comparing it with the never-treated is a different
      counterfactual and gives a different number.
    * The difference is divided by the change in treatment, so a switch-off
      contributes with the opposite sign and both directions measure the
      same "effect per unit of treatment".
    * It is otherwise an ordinary two-sample comparison, so the influence
      function has the same shape.
    """
    switchers = df[(df["_F"] == F) & (df["_dir"] == direction)]
    if switchers.empty:
        return None
    switcher_ids = set(switchers[group].unique())
    base_level = float(switchers["_base"].iloc[0])

    if h >= 0:
        t_pre, t_post = F - 1, F + h
    else:
        t_pre, t_post = F - 1 - abs(h), F - 1
    if t_pre < t_min:
        return None

    # Controls: never-switchers plus units that have not switched by the
    # comparison period, restricted to the switcher's baseline level.
    if control == "never_treated":
        candidates = never_ids
    else:
        threshold = F + max(h, 0)
        candidates = set(df[(df["_F"] > threshold) | (df["_F"].isna())][group].unique())
    pre_rows = df[df[time] == t_pre]
    same_base = set(pre_rows[pre_rows[treatment] == base_level][group].unique())
    ctrl_ids = (candidates & same_base) - switcher_ids
    if not ctrl_ids:
        return None

    sw_dy = _unit_change(df, group, time, y, switcher_ids, t_pre, t_post)
    c_dy = _unit_change(df, group, time, y, ctrl_ids, t_pre, t_post)
    if sw_dy is None or c_dy is None or len(sw_dy) == 0 or len(c_dy) == 0:
        return None

    sign = -1.0 if h < 0 else 1.0
    delta = sign * (sw_dy.mean() - c_dy.mean()) / direction

    psi = np.zeros(n_panel, dtype=float)
    psi[unit_pos.reindex(sw_dy.index).to_numpy()] = (
        sw_dy.to_numpy() - sw_dy.mean()
    ) * (n_panel / len(sw_dy))
    psi[unit_pos.reindex(c_dy.index).to_numpy()] -= (c_dy.to_numpy() - c_dy.mean()) * (
        n_panel / len(c_dy)
    )
    psi = psi * sign / direction

    return {"delta": float(delta), "n_sw": len(sw_dy), "psi": psi}


def _unit_change(df, group, time, y, ids, t_pre, t_post):
    """Per-unit outcome change between two periods, indexed by unit.

    Only units observed at BOTH ends contribute -- a unit missing either
    period has no change and cannot enter the difference.
    """
    sub = df[df[group].isin(ids) & df[time].isin([t_pre, t_post])]
    pre = sub[sub[time] == t_pre].set_index(group)[y]
    post = sub[sub[time] == t_post].set_index(group)[y]
    idx = pre.index.intersection(post.index)
    if len(idx) == 0:
        return None
    return post.loc[idx] - pre.loc[idx]


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
