"""Heterogeneity-robust triple differences (DDD) for staggered adoption.

Applies a Callaway-Sant'Anna-style group-time decomposition to the DDD
estimator so that staggered adoption with heterogeneous treatment
effects does not produce negative weights (the classical issue with
TWFE DDD, analogous to Goodman-Bacon 2021's concern for TWFE DID).

Identification outline (canonical 2x2x2 extended to staggered)
--------------------------------------------------------------
Let G ∈ {0, g_1, g_2, ...} index first-treatment cohort, T the period,
and B ∈ {0, 1} the within-treatment-group subgroup (B=1 affected,
B=0 placebo / unaffected). For each pair (g, t) with t ≥ g:

    DID_B(g, t)        = {E[Y_{it} | G=g,  B=1] − E[Y_{ig-1} | G=g,  B=1]}
                         − {E[Y_{it} | G=0, B=1] − E[Y_{ig-1} | G=0, B=1]}

    DID_placebo(g, t)  = {E[Y_{it} | G=g,  B=0] − E[Y_{ig-1} | G=g,  B=0]}
                         − {E[Y_{it} | G=0, B=0] − E[Y_{ig-1} | G=0, B=0]}

    DDD(g, t)          = DID_B(g, t) − DID_placebo(g, t)

Aggregated across (g, t) with CS-style simple weights (share of treated
units in cohort × post-treatment period), this yields the overall DDD
ATT estimate. Under the Olden-Møen (2022) interpretation, the placebo
arm DID_placebo(g, t) is the tested quantity — it should be zero under
the DDD parallel-trends relaxation.

References (verified anchors and [待核验] markers)
-------------------------------------------------
- Olden & Møen (2022). *The Econometrics Journal*, DOI 10.1093/ectj/utac010.
  Verified via paper.bib `olden2022triple`.
- Strezhnev (2023) proposes a CS-style decomposition for DDD; bib key
  [待核验 — not yet added to paper.bib, confirm exact citation before
  moving this reference out of RFC/guide to a published docstring].
- Callaway & Sant'Anna (2021) `callaway2021difference` for the
  group-time aggregation template.

Scope & caveats
---------------
- First cut supports **never-treated controls only**. Not-yet-treated
  controls per-(g, t) is a straightforward extension left for a
  follow-up once parity tests exist.
- Inference is cluster bootstrap at the unit level (n_boot draws),
  matching the pattern used in `sp.did_multiplegt`. Analytical
  influence-function variance is [待核验 — depends on locking the
  Strezhnev (2023) formulas].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from . import _core as _dc


def ddd_heterogeneous(
    data: pd.DataFrame,
    y: str,
    *,
    unit: str,
    time: str,
    cohort: str,
    subgroup: str,
    never_value: Any = 0,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """Heterogeneity-robust DDD estimator for staggered adoption panels.

    Parameters
    ----------
    data : DataFrame
        Long-format panel.
    y : str
        Outcome column.
    unit : str
        Unit identifier (panel id).
    time : str
        Time period column (int-valued).
    cohort : str
        First-treatment period column. ``never_value`` (default 0)
        marks never-treated units.
    subgroup : str
        Binary column: 1 = affected subgroup (where effect is expected),
        0 = unaffected / placebo subgroup (where effect should be zero
        under DDD parallel trends).
    never_value : any, default 0
        Value in ``cohort`` that marks never-treated units.
    n_boot : int, default 500
        Cluster-bootstrap replications for SE.
    alpha : float, default 0.05
    seed : int, optional

    Returns
    -------
    CausalResult
        ``estimate`` is the aggregate DDD ATT; ``detail`` carries per
        (g, t) decomposition; ``model_info['placebo_joint_test']`` is a
        joint Wald on the unaffected-subgroup DIDs.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.ddd_heterogeneous(df, y='earnings', unit='i', time='year',
    ...                           cohort='first_treat', subgroup='affected')
    >>> r.summary()
    """
    df = data.copy()
    for col in (y, unit, time, cohort, subgroup):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in data")

    if not set(df[subgroup].dropna().unique()) <= {0, 1}:
        raise ValueError(f"subgroup column {subgroup!r} must be binary 0/1")

    rng = np.random.default_rng(seed)

    # Identify cohorts and control set.
    cohort_vals = sorted(df[cohort].dropna().unique())
    treated_cohorts = [g for g in cohort_vals if g != never_value]
    if not treated_cohorts:
        raise ValueError("No treated cohorts found (all units are never-treated).")
    if never_value not in df[cohort].values:
        raise ValueError(
            f"No never-treated units (cohort == {never_value!r}) found. "
            "The not-yet-treated control variant is on the roadmap."
        )

    # Compute per-(g, t) DDD decomposition
    def _estimate(work_df: pd.DataFrame) -> Dict[str, Any]:
        return _compute_ddd_gt(
            df=work_df,
            y=y,
            unit=unit,
            time=time,
            cohort=cohort,
            subgroup=subgroup,
            treated_cohorts=treated_cohorts,
            never_value=never_value,
        )

    main = _estimate(df)

    # Cluster bootstrap for SE
    boot_overall = np.full(n_boot, np.nan)
    boot_placebo_gt = np.full((n_boot, len(main["cell_estimates"])), np.nan)

    for b in range(n_boot):
        bdf = _dc.cluster_bootstrap_draw(
            df,
            cluster_col=unit,
            rng=rng,
            relabel_cols=[unit],
        )
        try:
            best = _estimate(bdf)
        except Exception:
            continue
        boot_overall[b] = best["ddd_overall"]
        plac_vals = [r["did_placebo"] for r in best["cell_estimates"]]
        # Align by (g, t) — assume same order if #cells matches.
        if len(plac_vals) == boot_placebo_gt.shape[1]:
            boot_placebo_gt[b, :] = plac_vals

    se_overall = float(np.nanstd(boot_overall, ddof=1))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    est = float(main["ddd_overall"])
    if se_overall > 0 and np.isfinite(se_overall):
        z = est / se_overall
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
        ci = (est - z_crit * se_overall, est + z_crit * se_overall)
    else:
        p = np.nan
        ci = (np.nan, np.nan)

    # Joint placebo test via bootstrap covariance
    plac_est = np.array(
        [r["did_placebo"] for r in main["cell_estimates"]],
        dtype=float,
    )
    valid_rows = ~np.any(np.isnan(boot_placebo_gt), axis=1)
    if valid_rows.sum() >= boot_placebo_gt.shape[1] + 1:
        cov = np.cov(boot_placebo_gt[valid_rows], rowvar=False, ddof=1)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        placebo_joint = _dc.joint_wald(plac_est, cov)
    else:
        placebo_joint = None

    detail_df = pd.DataFrame(main["cell_estimates"])

    return CausalResult(
        method="DDD — heterogeneity-robust (Olden-Møen 2022 / Strezhnev 2023 [待核验])",
        estimand="ATT_DDD aggregated across cohort × time",
        estimate=est,
        se=se_overall,
        pvalue=p,
        ci=ci,
        alpha=alpha,
        n_obs=int(len(df)),
        detail=detail_df,
        model_info={
            "n_cohorts": len(treated_cohorts),
            "n_cells": len(main["cell_estimates"]),
            "placebo_joint_test": placebo_joint,
            "n_boot": n_boot,
            "cluster_var": unit,
        },
    )


def _compute_ddd_gt(
    *,
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    cohort: str,
    subgroup: str,
    treated_cohorts: List[Any],
    never_value: Any,
) -> Dict[str, Any]:
    """Compute the DDD decomposition per (g, t) cell + aggregated.

    For each cohort g and each post-treatment period t ≥ g:
    - Compute DID among affected subgroup (subgroup == 1)
    - Compute DID among unaffected subgroup (subgroup == 0) — placebo
    - DDD(g, t) = DID_affected − DID_unaffected
    Aggregate DDD(g, t) via simple average weighted by cell size.
    """
    never_df = df[df[cohort] == never_value]

    cells: List[Dict[str, Any]] = []

    for g in treated_cohorts:
        cohort_df = df[df[cohort] == g]
        if cohort_df.empty:
            continue
        times = sorted(df[time].unique())
        post_times = [t for t in times if t >= g]
        for t in post_times:
            pre_period = g - 1
            if pre_period not in times:
                continue

            did_b1 = _group_time_did(
                cohort_df,
                never_df,
                y=y,
                time=time,
                subgroup=subgroup,
                sub_val=1,
                t_pre=pre_period,
                t_post=t,
            )
            did_b0 = _group_time_did(
                cohort_df,
                never_df,
                y=y,
                time=time,
                subgroup=subgroup,
                sub_val=0,
                t_pre=pre_period,
                t_post=t,
            )
            if did_b1 is None or did_b0 is None:
                continue
            n_treated_affected = len(
                cohort_df[(cohort_df[time] == t) & (cohort_df[subgroup] == 1)]
            )
            if n_treated_affected == 0:
                continue

            cells.append(
                {
                    "cohort": g,
                    "time": t,
                    "did_affected": float(did_b1),
                    "did_placebo": float(did_b0),
                    "ddd": float(did_b1 - did_b0),
                    "n_treated_affected": int(n_treated_affected),
                }
            )

    if not cells:
        return {
            "ddd_overall": np.nan,
            "cell_estimates": [],
        }

    # Simple CS-style aggregation: weight each DDD(g, t) by the share of
    # treated-affected units contributing to it.
    weights = np.array([c["n_treated_affected"] for c in cells], dtype=float)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.full(len(cells), 1.0 / len(cells))
    ddd_vals = np.array([c["ddd"] for c in cells], dtype=float)
    overall = float(np.nansum(weights * ddd_vals))

    return {"ddd_overall": overall, "cell_estimates": cells}


def _group_time_did(
    cohort_df: pd.DataFrame,
    never_df: pd.DataFrame,
    *,
    y: str,
    time: str,
    subgroup: str,
    sub_val: int,
    t_pre: Any,
    t_post: Any,
) -> Optional[float]:
    """2x2 DID within a subgroup slice."""
    c_pre = cohort_df[(cohort_df[time] == t_pre) & (cohort_df[subgroup] == sub_val)][y]
    c_post = cohort_df[(cohort_df[time] == t_post) & (cohort_df[subgroup] == sub_val)][
        y
    ]
    n_pre = never_df[(never_df[time] == t_pre) & (never_df[subgroup] == sub_val)][y]
    n_post = never_df[(never_df[time] == t_post) & (never_df[subgroup] == sub_val)][y]
    if len(c_pre) == 0 or len(c_post) == 0 or len(n_pre) == 0 or len(n_post) == 0:
        return None
    return float((c_post.mean() - c_pre.mean()) - (n_post.mean() - n_pre.mean()))
