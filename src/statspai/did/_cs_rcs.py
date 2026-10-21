"""Callaway-Sant'Anna on repeated cross-sections and unbalanced panels.

Clean-room implementation (2026-10-20). Written from Callaway and
Sant'Anna (2021) [@callaway2021difference], Sant'Anna and Zhao (2020)
[@santanna2020doubly] and the public documentation of R ``did`` (the
``att_gt`` help page: ``base_period``, ``anticipation``, ``panel``,
``allow_unbalanced_panel``, ``clustervars``). R ``did`` (GPL-3) was used
only as a black box; its source was not read. Specification and evidence:
``docs/dev/cleanroom/cs_rcs.md``.

Two pieces live here:

* :func:`gt_cells`, the list of ATT(g, t) cells and their base periods,
  shared with the balanced-panel path;
* :func:`cs_repeated_cross_sections`, the estimator when rows are not a
  balanced panel: every row is an observation of its own, each ATT(g, t)
  is a two-period repeated-cross-section DiD (:mod:`statspai.did._rcs`) on
  the rows of cohort ``g`` and of the comparison group in periods ``t``
  and the base period, and the cell influence functions are stacked into
  an ``n x K`` matrix that the aggregation, the pre-test and the
  multiplier bootstrap use. On an unbalanced panel (``unit_col`` given)
  the row-level influence functions are summed within unit, so that
  inference treats the unit as the sampling unit.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["cs_repeated_cross_sections", "gt_cells"]


def _last_period_before(periods: Sequence[Any], bound: Any) -> Optional[Any]:
    """Largest period strictly smaller than ``bound`` (None if there is none)."""
    earlier = [p for p in periods if p < bound]
    return earlier[-1] if earlier else None


def gt_cells(
    cohorts: Sequence[Any],
    time_periods: Sequence[Any],
    base_period: str,
    anticipation: int = 0,
) -> List[Tuple[Any, Any, Any]]:
    """The ATT(g, t) cells to estimate, as ``(g, t, base)`` triples.

    For cohort ``g`` the last period that is unaffected by treatment, even
    under ``anticipation`` periods of anticipation, is the latest observed
    period ``b_g < g - anticipation`` (Callaway and Sant'Anna 2021,
    Assumption 3; the ``att_gt`` help: the universal base period is
    ``g - anticipation - 1``). A cohort without such a period has no
    identified ATT(g, t) and is skipped with a warning.

    * ``'universal'``: every cell is compared with ``b_g``; one cell for
      every observed period except ``b_g`` itself.
    * ``'varying'``: post-treatment cells (``t >= g``) are compared with
      ``b_g``; a pre-treatment cell ``t < g`` is compared with the period
      just before ``t`` (the help: "changes from period t-1 to period t"),
      so the first observed period has no cell.

    Cells are returned sorted by cohort, then by period.
    """
    periods = sorted(time_periods)
    cells: List[Tuple[Any, Any, Any]] = []
    for g in sorted(cohorts):
        base_g = _last_period_before(periods, g - anticipation)
        if base_g is None:
            warnings.warn(
                f"callaway_santanna: cohort g={g} has no pre-treatment period "
                f"once anticipation={anticipation} is accounted for (no "
                f"observed period t with t + anticipation < g). None of its "
                "ATT(g, t) is identified, so the cohort is left out (as R "
                "`did` does). Its units still serve as comparisons where "
                "the control group allows.",
                UserWarning,
                stacklevel=3,
            )
            continue
        for t in periods:
            if base_period == "universal":
                if t == base_g:
                    continue
                cells.append((g, t, base_g))
            elif t >= g:
                cells.append((g, t, base_g))
            else:
                prev = _last_period_before(periods, t)
                if prev is not None:
                    cells.append((g, t, prev))
    return cells


def _check_weights(
    df: pd.DataFrame, w_raw: np.ndarray, weights_col: str, unit_col: Optional[str]
) -> None:
    """Sampling weights must be non-negative, not all zero, and (on an
    unbalanced panel) constant within unit."""
    if np.any(w_raw < 0) or not np.all(np.isfinite(w_raw)):
        raise MethodIncompatibility(
            f"callaway_santanna: weights column {weights_col!r} must be "
            "non-negative and finite.",
            recovery_hint="Repair or drop the negative / infinite weights.",
            diagnostics={"weights": weights_col, "n_negative": int(np.sum(w_raw < 0))},
        )
    if not np.sum(w_raw) > 0:
        raise DataInsufficient(
            f"callaway_santanna: weights column {weights_col!r} sums to zero "
            "on the estimation sample.",
            recovery_hint="Pass weights with positive mass.",
            diagnostics={"weights": weights_col},
        )
    if unit_col is not None:
        varying = df.groupby(unit_col)[weights_col].nunique(dropna=False) > 1
        if varying.any():
            raise MethodIncompatibility(
                f"callaway_santanna: weights column {weights_col!r} varies "
                f"within unit ({int(varying.sum())} units). The unbalanced-"
                "panel estimator sums a unit's rows into one influence "
                "contribution, so the unit needs a single weight.",
                recovery_hint=(
                    "Use a time-invariant weight (e.g. the base-period value), "
                    "or pass panel=False to treat the rows as repeated cross "
                    "sections with row-level weights."
                ),
                diagnostics={
                    "weights": weights_col,
                    "n_units_varying": int(varying.sum()),
                },
            )


def _check_cluster_cuts_cells(
    df: pd.DataFrame, clustervar: str, g_arr: np.ndarray, t_arr: np.ndarray
) -> None:
    """Refuse a row-level cluster made of whole cohort x period cells.

    Within a cohort x period cell every treated (and every comparison) row
    enters an ATT(g, t) influence function as a deviation from that cell's
    mean, so those deviations sum to zero over the cell. If every cluster
    is a union of whole cells, the cluster sums lose all of that variation:
    the clustered variance is exactly zero against never-treated
    comparisons and badly understated against not-yet-treated ones.
    """
    cells = pd.DataFrame({"g": g_arr, "t": t_arr, "c": df[clustervar].to_numpy()})
    per_cell = cells.groupby(["g", "t"])["c"].nunique(dropna=False)
    if bool((per_cell <= 1).all()):
        raise MethodIncompatibility(
            f"callaway_santanna: cluster variable {clustervar!r} is constant "
            "within every cohort x period cell (e.g. the period or the "
            "cohort itself). Each ATT(g, t) influence function sums to zero "
            "within such a cell, so clustering on it removes the sampling "
            "variation instead of allowing for correlation.",
            recovery_hint=(
                "Cluster on a variable that cuts across cells (a region or "
                "a sampling unit), or drop clustervars=."
            ),
            diagnostics={"clustervar": clustervar, "n_cells": int(len(per_cell))},
        )


def _estimator_label(estimator: str, x: Optional[List[str]], unbalanced: bool) -> str:
    if estimator == "reg" and not x:
        label = "REG (RCS, cell-mean)"
    else:
        label = f"{estimator.upper()} (RCS, Sant'Anna-Zhao)"
    return label + (" — unbalanced panel" if unbalanced else "")


def _cohort_sizes(cohorts_of: np.ndarray, cohorts: List[Any], w: Optional[np.ndarray]):
    """Head count (or weight mass) of each estimated cohort."""
    s = pd.Series(np.ones(len(cohorts_of)) if w is None else w)
    sizes = s.groupby(cohorts_of).sum()
    return sizes.reindex(cohorts).astype(float)


def cs_repeated_cross_sections(
    data: pd.DataFrame,
    y: str,
    g: str,
    t: str,
    base_period: str,
    anticipation: int,
    alpha: float,
    x: Optional[List[str]] = None,
    estimator: str = "reg",
    control_group: str = "nevertreated",
    notyet_cutoff: str = "period",
    pretest: str = "joint",
    pretest_periods: Optional[int] = None,
    unit_col: Optional[str] = None,
    weights_col: Optional[str] = None,
    clustervar: Optional[str] = None,
    bstrap: bool = False,
    biters: int = 1000,
    cband: bool = False,
    boot_weight_type: str = "rademacher",
    random_state: Optional[int] = None,
    exclude_cohorts: Sequence[Any] = (),
    variance_convention: str = "delta",
    pretest_pvalue: str = "hotelling_f",
    analytic_aggregates: bool = False,
) -> CausalResult:
    """Callaway-Sant'Anna ATT(g, t) when the rows are not a balanced panel.

    See the module docstring and ``docs/dev/cleanroom/cs_rcs.md``.
    ``unit_col`` switches on the unbalanced-panel reading (influence
    functions summed within unit); without it every row is its own unit.
    ``analytic_aggregates`` keeps the simple ATT and the event study on
    analytic standard errors under ``bstrap=True`` (``parity='stata'``).
    The remaining arguments are those of
    :func:`statspai.did.callaway_santanna`.
    """
    import importlib

    # The package re-exports the function under the module's name, so the
    # module itself is looked up explicitly.
    _cs = importlib.import_module(__package__ + ".callaway_santanna")
    from ._core import multiplier_bootstrap

    unbalanced = unit_col is not None
    keep_cols = [y, g, t, *(x or [])]
    if weights_col is not None:
        keep_cols.append(weights_col)
    df = data.dropna(subset=keep_cols).reset_index(drop=True)
    if df.empty:
        raise DataInsufficient(
            "callaway_santanna: No observations after dropping NaNs in the "
            "outcome, period, cohort, covariate or weight columns.",
            recovery_hint="Check the columns for missing values.",
            diagnostics={"n_rows_in": int(len(data))},
        )

    y_arr = df[y].to_numpy(dtype=float)
    t_arr = df[t].to_numpy()
    g_raw = pd.to_numeric(df[g], errors="coerce").fillna(0)
    g_arr = g_raw.replace([np.inf, -np.inf], 0).to_numpy()
    x_mat = df[list(x)].to_numpy(dtype=float) if x else None
    w_arr: Optional[np.ndarray] = None
    if weights_col is not None:
        w_raw = df[weights_col].to_numpy(dtype=float)
        _check_weights(df, w_raw, weights_col, unit_col)
        w_arr = w_raw / w_raw.mean()
    n_obs = len(df)
    if clustervar is not None and not unbalanced:
        _check_cluster_cuts_cells(df, clustervar, g_arr, t_arr)

    time_periods = sorted(pd.unique(t_arr))
    max_time = time_periods[-1]
    cohorts = sorted(
        c
        for c in pd.unique(g_arr)
        if 0 < c <= max_time and not any(c == e for e in exclude_cohorts)
    )

    unit_codes: Optional[np.ndarray] = None
    unit_ids: np.ndarray
    if unbalanced:
        unit_codes, unit_ids = pd.factorize(df[unit_col], sort=True)
        unit_ids = np.asarray(unit_ids)
        n_units = len(unit_ids)
        if n_units < 2:
            raise DataInsufficient(
                "callaway_santanna: the unbalanced-panel estimator needs at "
                f"least 2 units, found {n_units}.",
                recovery_hint="Pass a panel with several units.",
                diagnostics={"n_units": n_units},
            )
    else:
        unit_ids = np.arange(n_obs)
        n_units = n_obs

    if not cohorts:
        raise DataInsufficient(
            "No treatment cohorts found. Check group variable encoding.",
            recovery_hint=(
                "Encode first treatment periods in `g`, using 0 for "
                "never-treated units."
            ),
            diagnostics={"function": "callaway_santanna"},
        )
    cells = gt_cells(cohorts, time_periods, base_period, anticipation)
    if not cells:
        raise DataInsufficient(
            "No valid (group, time) pairs to estimate.",
            recovery_hint=(
                "Check treatment timing, base_period, anticipation, and "
                "available periods."
            ),
            diagnostics={
                "function": "callaway_santanna",
                "cohorts": cohorts,
                "time_periods": time_periods,
            },
        )

    def to_units(inf_rows: np.ndarray) -> np.ndarray:
        # Row-level influence functions live on the n_obs scale
        # (theta_hat - theta ~ mean over rows). The same sum over units is
        # mean over units of (n_units / n_obs) * (sum of the unit's rows).
        if unit_codes is None:
            return inf_rows
        folded = np.zeros(n_units)
        np.add.at(folded, unit_codes, inf_rows)
        return folded * (n_units / n_obs)

    cell_mean = estimator == "reg" and not x
    z_crit = stats.norm.ppf(1 - alpha / 2)
    rows: List[Dict[str, Any]] = []
    inf_cols: List[np.ndarray] = []
    for g_val, t_val, b_val in cells:
        if cell_mean:
            att, se, inf_rows = _cs._estimate_single_att_rcs(
                y_arr,
                g_arr,
                t_arr,
                g_val,
                t_val,
                b_val,
                n_obs,
                w_arr=w_arr,
                control_group=control_group,
                notyet_cutoff=notyet_cutoff,
                anticipation=anticipation,
            )
        else:
            att, se, inf_rows = _cs._estimate_single_att_rcs_sz(
                y_arr,
                g_arr,
                t_arr,
                x_mat,
                g_val=g_val,
                t_val=t_val,
                base_val=b_val,
                estimator=estimator,
                control_group=control_group,
                n_obs=n_obs,
                w_arr=w_arr,
                anticipation=anticipation,
                notyet_cutoff=notyet_cutoff,
                variance_convention=variance_convention,
            )
        inf = to_units(inf_rows)
        if unbalanced and np.isfinite(se):
            se = float(np.sqrt(np.mean(inf**2) / n_units))
        rows.append(
            {
                "group": g_val,
                "time": t_val,
                "att": att,
                "se": se,
                "ci_lower": att - z_crit * se,
                "ci_upper": att + z_crit * se,
                "pvalue": (2 * stats.norm.sf(abs(att / se)) if se > 0 else 1.0),
                "relative_time": t_val - g_val,
                "n_treated_obs": int(
                    np.sum((g_arr == g_val) & ((t_arr == t_val) | (t_arr == b_val)))
                ),
            }
        )
        inf_cols.append(inf)

    detail = pd.DataFrame(rows)
    detail, inf_cols, dropped_cells = _cs._drop_unidentified_cells(detail, inf_cols)
    inf_matrix = np.column_stack(inf_cols) if inf_cols else None

    # Per-unit quantities for the aggregation weights and the clusters.
    if unbalanced:
        assert unit_codes is not None
        first_row = np.unique(unit_codes, return_index=True)[1]
        unit_cohorts = g_arr[first_row]
        unit_w = None if w_arr is None else w_arr[first_row]
    else:
        unit_cohorts = g_arr
        unit_w = w_arr
    if unit_w is not None:
        unit_w = unit_w / unit_w.mean()

    cluster_ids: Optional[np.ndarray] = None
    if clustervar is not None:
        cl = df[clustervar].to_numpy()
        cluster_ids = cl[first_row] if unbalanced else cl

    analytic_cluster = cluster_ids if not bstrap else None
    if analytic_cluster is not None and inf_matrix is not None:
        detail = _cs._cluster_detail_se(
            detail, inf_matrix, n_units, analytic_cluster, alpha
        )

    boot_cfg: Optional[Dict[str, Any]] = None
    crit_val_uniform: Optional[float] = None
    if bstrap and inf_matrix is not None:
        boot_cfg = {
            "n_boot": biters,
            "random_state": random_state,
            "weight_type": boot_weight_type,
            "cluster_ids": cluster_ids,
        }
        se_boot, crit = multiplier_bootstrap(
            inf_matrix,
            n_units,
            alpha,
            biters,
            random_state,
            weight_type=boot_weight_type,
            cluster_ids=cluster_ids,
        )
        ok = np.isfinite(detail["se"].to_numpy(dtype=float))
        se_new = np.where(ok, se_boot, np.inf)
        att_vals = detail["att"].to_numpy(dtype=float)
        positive = np.isfinite(se_new) & (se_new > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(positive, att_vals / se_new, 0.0)
        detail["se"] = se_new
        detail["pvalue"] = np.where(positive, 2 * stats.norm.sf(np.abs(z)), 1.0)
        detail["ci_lower"] = att_vals - z_crit * se_new
        detail["ci_upper"] = att_vals + z_crit * se_new
        if cband:
            crit_val_uniform = float(crit)
            detail["cband_lower"] = att_vals - crit * se_new
            detail["cband_upper"] = att_vals + crit * se_new

    agg_boot_cfg = None if analytic_aggregates else boot_cfg
    cohort_sizes = _cohort_sizes(unit_cohorts, cohorts, unit_w)
    post = (detail["relative_time"] >= 0).to_numpy()
    post_inf = inf_matrix[:, post] if inf_matrix is not None else None
    agg_est, agg_se, agg_pval, agg_ci = _cs._aggregate_simple(
        detail[post],
        post_inf,
        cohort_sizes,
        n_units,
        alpha,
        boot_cfg=agg_boot_cfg,
        unit_cohorts=unit_cohorts,
        unit_weights=unit_w,
        cluster_ids=analytic_cluster,
    )
    event_study = _cs._aggregate_event_study(
        detail,
        inf_matrix,
        cohort_sizes,
        n_units,
        alpha,
        boot_cfg=agg_boot_cfg,
        unit_cohorts=unit_cohorts,
        unit_weights=unit_w,
        cluster_ids=analytic_cluster,
    )
    pretrend = _cs._pretrend_test(
        detail,
        inf_matrix,
        n_units,
        pretest=pretest,
        pretest_periods=pretest_periods,
        cluster_ids=cluster_ids,
        pvalue_method=pretest_pvalue,
    )

    model_info: Dict[str, Any] = {
        "estimator": _estimator_label(estimator, x, unbalanced),
        "control_group": control_group,
        "notyet_cutoff": notyet_cutoff,
        "base_period": base_period,
        "anticipation": anticipation,
        "panel": unbalanced,
        "allow_unbalanced_panel": unbalanced,
    }
    if unbalanced:
        model_info["unbalanced_assumption"] = "stable group composition over time"
    model_info.update(
        {
            "n_units": n_units,
            "n_obs": n_obs,
            "weights": weights_col,
            "weighted": weights_col is not None,
            "covariates": list(x) if x else None,
            "clustervars": clustervar,
            "bstrap": bstrap,
            "biters": biters if bstrap else 0,
            "cband": cband,
            "boot_weight_type": boot_weight_type if bstrap else None,
            "n_periods": len(time_periods),
            "n_cohorts": len(cohorts),
            "cohorts": cohorts,
            "comparison_only_cohorts": [float(c) for c in exclude_cohorts],
            "dropped_gt_cells": dropped_cells,
            "event_study": event_study,
            "pretrend_test": pretrend,
            "cohort_sizes": cohort_sizes,
            "se_method": "multiplier" if bstrap else "analytic",
            "inference_note": _cs._inference_note(bstrap, cband),
            "crit_val_uniform": crit_val_uniform,
            "_cluster_ids": cluster_ids,
            "_row_cohorts": None if unbalanced else g_arr,
            "_row_times": None if unbalanced else t_arr,
            "_unit_ids": unit_ids,
            "_unit_cohorts": unit_cohorts,
            "_unit_weights": unit_w,
        }
    )
    if not cell_mean:
        model_info["variance_convention"] = (
            variance_convention if estimator == "dr" else None
        )

    kind = "unbalanced panel" if unbalanced else "repeated cross-sections"
    return CausalResult(
        method=f"Callaway and Sant'Anna (2021) — {kind}",
        estimand="ATT",
        estimate=agg_est,
        se=agg_se,
        pvalue=agg_pval,
        ci=agg_ci,
        alpha=alpha,
        n_obs=n_obs,
        detail=detail,
        model_info=model_info,
        _influence_funcs=inf_matrix,
        _citation_key="callaway_santanna",
    )
