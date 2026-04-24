"""Time-varying covariate DiD (Caetano-Callaway-Payne-Rodrigues 2022).

Motivation
----------
The canonical "controlled DiD" regression with contemporaneous
time-varying covariates X_{i,t} suffers a bad-controls problem when
treatment affects the covariates. Caetano, Callaway, Payne & Rodrigues
(2022) [待核验 — paper exact title + DOI to be confirmed and added to
paper.bib before citing in docstring / guide] argue the right fix is
to freeze covariates at their pre-treatment value X_{i, g-1} and fit a
DR-DiD-style ATT(g, t) estimator with the frozen covariate.

This implementation provides the cohort-by-period ATT computation with
baseline-frozen covariates, aggregated via a Callaway-Sant'Anna-style
simple weighting.

Scope & caveats
---------------
- First cut does outcome-regression (OR) adjustment only; a doubly
  robust (DR) variant with propensity score on frozen covariates is an
  easy extension and on the roadmap.
- Never-treated units are required as controls in this MVP; the
  not-yet-treated variant is a follow-up.
- Paper-faithful identification statements carry [待核验] until we lock
  the exact paper version.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from . import _core as _dc


def did_timevarying_covariates(
    data: pd.DataFrame,
    y: str,
    *,
    unit: str,
    time: str,
    cohort: str,
    covariates: List[str],
    never_value: Any = 0,
    baseline_offset: int = -1,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """DiD with time-varying covariates frozen at baseline.

    Parameters
    ----------
    data : DataFrame
        Long-format panel.
    y : str
        Outcome.
    unit : str
        Unit identifier.
    time : str
        Integer-valued period column.
    cohort : str
        First-treatment period column; ``never_value`` marks never-treated.
    covariates : list of str
        Time-varying covariates. Values at ``g + baseline_offset``
        (default: ``g - 1``, i.e. one period before first treatment) are
        frozen and used as the controls for all periods of cohort g.
    never_value : any, default 0
        Value in ``cohort`` that marks never-treated units.
    baseline_offset : int, default -1
        Offset relative to first-treatment period for freezing covariates.
        -1 = last pre-treatment period.
    n_boot : int, default 500
        Cluster-bootstrap replications for SE.
    alpha : float, default 0.05
    seed : int, optional

    Returns
    -------
    CausalResult
        Aggregate ATT across (g, t); ``detail`` carries per-(g, t) ATTs
        and the covariate-adjustment baseline.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.did_timevarying_covariates(
    ...     df, y='earnings', unit='i', time='year', cohort='g',
    ...     covariates=['age', 'wage_prev'],
    ... )
    """
    df = data.copy()
    for col in [y, unit, time, cohort] + list(covariates):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in data")

    rng = np.random.default_rng(seed)
    cohort_vals = sorted(df[cohort].dropna().unique())
    treated_cohorts = [g for g in cohort_vals if g != never_value]
    if not treated_cohorts:
        raise ValueError("No treated cohorts found")
    if never_value not in df[cohort].values:
        raise ValueError(
            f"No never-treated units (cohort == {never_value!r}); "
            "not-yet-treated variant is on the roadmap."
        )

    # Build baseline covariates per unit (using first-treatment g + baseline_offset)
    df_with_base = _attach_baseline_covariates(
        df,
        unit=unit,
        time=time,
        cohort=cohort,
        covariates=covariates,
        never_value=never_value,
        baseline_offset=baseline_offset,
    )

    main = _compute_att_gt(
        df_with_base,
        y=y,
        unit=unit,
        time=time,
        cohort=cohort,
        covariates=covariates,
        treated_cohorts=treated_cohorts,
        never_value=never_value,
    )

    # Cluster bootstrap
    boot_overall = np.full(n_boot, np.nan)
    for b in range(n_boot):
        try:
            bdf = _dc.cluster_bootstrap_draw(
                df_with_base,
                cluster_col=unit,
                rng=rng,
                relabel_cols=[unit],
            )
            best = _compute_att_gt(
                bdf,
                y=y,
                unit=unit,
                time=time,
                cohort=cohort,
                covariates=covariates,
                treated_cohorts=treated_cohorts,
                never_value=never_value,
            )
            boot_overall[b] = best["att_overall"]
        except Exception:
            continue

    se = float(np.nanstd(boot_overall, ddof=1))
    est = float(main["att_overall"])
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    if se > 0 and np.isfinite(se):
        z = est / se
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
        ci = (est - z_crit * se, est + z_crit * se)
    else:
        p = np.nan
        ci = (np.nan, np.nan)

    detail_df = pd.DataFrame(main["cell_estimates"])

    return CausalResult(
        method="DiD with time-varying covariates (Caetano et al. 2022 [待核验])",
        estimand="ATT aggregated across cohort × time",
        estimate=est,
        se=se,
        pvalue=p,
        ci=ci,
        alpha=alpha,
        n_obs=int(len(df)),
        detail=detail_df,
        model_info={
            "covariates": list(covariates),
            "baseline_offset": baseline_offset,
            "n_cells": len(main["cell_estimates"]),
            "n_boot": n_boot,
            "cluster_var": unit,
        },
    )


def _attach_baseline_covariates(
    df: pd.DataFrame,
    *,
    unit: str,
    time: str,
    cohort: str,
    covariates: List[str],
    never_value: Any,
    baseline_offset: int,
) -> pd.DataFrame:
    """Append columns ``<cov>_base`` carrying the value of each covariate
    at the unit's baseline period (first-treatment + offset).

    For never-treated units, uses the median observed period as baseline.
    """
    out = df.copy()

    baselines: List[Dict[str, Any]] = []

    for uid, u_df in df.groupby(unit):
        g_val = u_df[cohort].iloc[0]
        if g_val == never_value:
            # Use median period for never-treated — neutral baseline.
            base_t = int(np.nanmedian(u_df[time].values))
        else:
            base_t = int(g_val + baseline_offset)

        base_rows = u_df[u_df[time] == base_t]
        if base_rows.empty:
            # Fallback: first observed period.
            base_rows = u_df.iloc[[0]]
        base_vals = {c: base_rows[c].iloc[0] for c in covariates}
        base_vals[unit] = uid
        baselines.append(base_vals)

    bl_df = pd.DataFrame(baselines)
    bl_df = bl_df.rename(columns={c: f"{c}_base" for c in covariates})
    out = out.merge(bl_df, on=unit, how="left")
    return out


def _compute_att_gt(
    df: pd.DataFrame,
    *,
    y: str,
    unit: str,
    time: str,
    cohort: str,
    covariates: List[str],
    treated_cohorts: List[Any],
    never_value: Any,
) -> Dict[str, Any]:
    """Compute ATT(g, t) using outcome regression on frozen covariates.

    For each (g, t) with t >= g:
      - Restrict to cohort g + never-treated
      - Form ΔY = Y_{i, t} - Y_{i, g-1}
      - Regress ΔY on treatment dummy + baseline covariates
      - β on treatment dummy is ATT(g, t)
    Aggregate with unit-count weights.
    """
    base_cols = [f"{c}_base" for c in covariates]
    cells: List[Dict[str, Any]] = []
    times = sorted(df[time].unique())

    for g in treated_cohorts:
        pre_t = g - 1
        if pre_t not in times:
            continue
        post_times = [t for t in times if t >= g]

        pop_mask = (df[cohort] == g) | (df[cohort] == never_value)
        pop = df[pop_mask].copy()
        # Merge in ΔY using base period outcome per unit
        pre_df = pop[pop[time] == pre_t][[unit, y]].rename(columns={y: "_y_pre"})
        for t in post_times:
            post_df = pop[pop[time] == t][[unit, y, cohort] + base_cols]
            merged = post_df.merge(pre_df, on=unit, how="inner")
            if merged.empty:
                continue
            merged["_dy"] = merged[y] - merged["_y_pre"]
            merged["_treated"] = (merged[cohort] == g).astype(float)

            n_treated = int(merged["_treated"].sum())
            n_control = int((merged["_treated"] == 0).sum())
            if n_treated < 2 or n_control < 2:
                continue

            # OLS of _dy on _treated + base covariates
            X_cols = ["_treated"] + base_cols
            X = merged[X_cols].values.astype(float)
            ones = np.ones((X.shape[0], 1))
            X = np.hstack([X, ones])
            dy = merged["_dy"].values.astype(float)
            valid = np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
            X, dy = X[valid], dy[valid]
            n, k = X.shape
            if n <= k:
                continue
            beta = np.linalg.pinv(X.T @ X) @ (X.T @ dy)
            att_gt = float(beta[0])
            cells.append(
                {
                    "cohort": g,
                    "time": t,
                    "att_gt": att_gt,
                    "n_treated": n_treated,
                    "n_control": n_control,
                }
            )

    if not cells:
        return {"att_overall": np.nan, "cell_estimates": []}

    weights = np.array([c["n_treated"] for c in cells], dtype=float)
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.full(len(cells), 1.0 / len(cells))
    att_vals = np.array([c["att_gt"] for c in cells], dtype=float)
    overall = float(np.nansum(weights * att_vals))
    return {"att_overall": overall, "cell_estimates": cells}
