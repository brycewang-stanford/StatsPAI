"""
Sequential Synthetic Difference-in-Differences.

Arkhangelsky & Samkov (arXiv:2404.00164, 2024) extend the Arkhangelsky,
Athey, Hirshberg, Imbens & Wager (2021) SDID estimator to **staggered
adoption** designs in which parallel trends can fail *across* cohorts.
The idea: process cohorts in the order they are treated. For each cohort
``g``, use donors drawn from the *not-yet-treated* units (including
later-treated units in their pre-period) and the already-estimated
counterfactuals for *earlier* cohorts. This sequentially peels off the
treatment from first-adopter to last, avoiding the negative-weights
pathology of TWFE and the overlap failures that break SDID in
staggered panels.

This module delivers :func:`sequential_sdid`, the main public entry
point, and :class:`SequentialSDIDResult` with per-cohort ATT(g) and an
aggregated ATT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.results import CausalResult
from .sdid import sdid as _sdid_base


__all__ = ["sequential_sdid", "SequentialSDIDResult"]


@dataclass
class SequentialSDIDResult:
    """Per-cohort and aggregated output of :func:`sequential_sdid`."""

    aggregate_att: float
    aggregate_se: float
    aggregate_ci: tuple
    per_cohort: pd.DataFrame
    model_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lo, hi = self.aggregate_ci
        lines = [
            "Sequential Synthetic DID (Arkhangelsky & Samkov 2024)",
            "=" * 60,
            f"  Aggregate ATT    : {self.aggregate_att:.6f}",
            f"  Aggregate SE     : {self.aggregate_se:.6f}",
            f"  95% CI           : [{lo:.6f}, {hi:.6f}]",
            f"  # cohorts        : {len(self.per_cohort)}",
            "",
            "Per-cohort ATT(g):",
            self.per_cohort.to_string(index=False, float_format="%.4f"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<SequentialSDIDResult: {len(self.per_cohort)} cohorts, "
            f"aggregate ATT = {self.aggregate_att:.4f}>"
        )


def sequential_sdid(
    data: pd.DataFrame,
    *,
    outcome: str,
    unit: str,
    time: str,
    cohort: str,
    never_treated_value: Any = 0,
    se_method: str = "placebo",
    n_reps: int = 200,
    cohort_weights: str = "size",
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """Sequential Synthetic DID for staggered-adoption panels.

    Parameters
    ----------
    data : DataFrame
        Balanced long-format panel.
    outcome : str
        Outcome column.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    cohort : str
        Column giving each unit's first-treated period. Never-treated
        units take the value ``never_treated_value`` (default ``0``).
    never_treated_value : scalar, default 0
        Sentinel value in ``cohort`` indicating a never-treated unit.
    se_method : {'placebo', 'bootstrap', 'jackknife'}, default 'placebo'
        Forwarded to the inner SDID call.
    n_reps : int, default 200
    cohort_weights : {'size', 'equal'}, default 'size'
        Aggregation weights across cohorts. ``'size'`` weights by the
        number of treated units × post-periods in each cohort (CS-style).
    alpha : float, default 0.05
    seed : int, optional

    Returns
    -------
    CausalResult
        With ``estimand='ATT'``, ``method='sequential_sdid'``, and a
        ``detail`` DataFrame of per-cohort ATT(g), SE(g), n_treated(g),
        treatment_period(g).

    Notes
    -----
    The sequential algorithm (Arkhangelsky & Samkov 2024, Section 3):

    1. Sort cohorts by treatment time ``g_1 < g_2 < ... < g_K``.
    2. For each cohort ``g_k``:

       a. Subset to units in cohort ``g_k`` ∪ never-treated ∪ units with
          ``cohort[i] > g_k`` (not yet treated at time ``g_k``).
       b. Restrict to times ``t ≤ current_end`` where ``current_end`` is
          the last period before the *next* cohort's treatment.
       c. Run single-cohort SDID on this subpanel.

    3. Aggregate ATT(g) into an overall ATT using ``cohort_weights``.

    When there is only one cohort this function reduces exactly to the
    classical SDID.

    References
    ----------
    Arkhangelsky, D. & Samkov, A. (arXiv:2404.00164, 2024).
    Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021). AER 111(12). [@arkhangelsky2024sequential]
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    for col in (outcome, unit, time, cohort):
        if col not in data.columns:
            raise ValueError(f"Column {col!r} not found in `data`.")

    df = data.copy()
    # Treated cohorts (sorted in adoption order).
    treated_cohorts = sorted(
        c for c in df[cohort].unique()
        if not pd.isna(c) and c != never_treated_value
    )
    if not treated_cohorts:
        raise ValueError(
            f"No treated cohorts found in column {cohort!r}. All values "
            f"equal the never-treated sentinel {never_treated_value!r}."
        )

    if cohort_weights not in ("size", "equal"):
        raise ValueError(
            f"`cohort_weights` must be 'size' or 'equal'; got {cohort_weights!r}."
        )

    per_cohort_rows: List[Dict[str, Any]] = []
    max_time = df[time].max()

    for idx, g in enumerate(treated_cohorts):
        # Next-cohort boundary: the subpanel runs up to just before the
        # next cohort's treatment (so later-treated units contribute as
        # pure donors, per Arkhangelsky-Samkov §3).
        if idx + 1 < len(treated_cohorts):
            # Inclusive upper limit: last period before next cohort enters.
            next_g = treated_cohorts[idx + 1]
            t_max_g = next_g - 1 if np.issubdtype(type(next_g), np.integer) else (
                df.loc[df[time] < next_g, time].max()
            )
        else:
            t_max_g = max_time

        # Donor pool: never-treated + not-yet-treated (cohort > g) restricted
        # to times ≤ t_max_g.
        donor_mask = (
            (df[cohort] == never_treated_value) | (df[cohort] > g)
        ) & (df[time] <= t_max_g)
        treated_mask = (df[cohort] == g) & (df[time] <= t_max_g)

        sub = pd.concat(
            [df.loc[donor_mask], df.loc[treated_mask]], axis=0
        ).drop_duplicates(subset=[unit, time]).reset_index(drop=True)

        treated_units = sub.loc[sub[cohort] == g, unit].unique().tolist()
        if not treated_units:
            continue
        # Need at least 2 pre-periods for SDID time weights.
        pre_times = sub.loc[sub[time] < g, time].unique()
        post_times = sub.loc[(sub[time] >= g) & (sub[time] <= t_max_g), time].unique()
        if pre_times.size < 2 or post_times.size < 1:
            per_cohort_rows.append({
                "cohort": g, "treatment_period": g,
                "att": np.nan, "se": np.nan,
                "n_treated": len(treated_units),
                "n_donors": int((sub[cohort] != g).sum() / max(len(sub[time].unique()), 1)),
                "note": "insufficient pre/post periods",
            })
            continue

        try:
            res_g = _sdid_base(
                sub, outcome=outcome, unit=unit, time=time,
                treated_unit=treated_units, treatment_time=g,
                method="sdid", se_method=se_method, n_reps=n_reps,
                seed=seed, alpha=alpha,
            )
            per_cohort_rows.append({
                "cohort": g,
                "treatment_period": g,
                "att": float(res_g.estimate),
                "se": float(res_g.se),
                "n_treated": len(treated_units),
                "n_donors": int(
                    len(sub.loc[sub[cohort] != g, unit].unique())
                ),
                "note": "",
            })
        except Exception as exc:  # noqa: BLE001 — surface but keep going
            per_cohort_rows.append({
                "cohort": g, "treatment_period": g,
                "att": np.nan, "se": np.nan,
                "n_treated": len(treated_units),
                "n_donors": int(
                    len(sub.loc[sub[cohort] != g, unit].unique())
                ),
                "note": f"SDID failed: {type(exc).__name__}: {exc}",
            })

    per_cohort = pd.DataFrame(per_cohort_rows)
    valid = per_cohort.dropna(subset=["att", "se"]).copy()
    if valid.empty:
        raise RuntimeError(
            "Sequential SDID failed for every cohort. See per-cohort notes."
        )

    # Aggregate
    if cohort_weights == "equal":
        w = np.ones(len(valid))
    else:
        w = valid["n_treated"].to_numpy(dtype=float)
    w = w / w.sum()
    agg_att = float(np.sum(w * valid["att"].to_numpy()))
    # Independent-cohort SE (conservative): sqrt(sum w^2 * se^2).
    agg_var = float(np.sum((w ** 2) * (valid["se"].to_numpy() ** 2)))
    agg_se = float(np.sqrt(agg_var))
    # Normal-approx CI
    from scipy import stats as _stats
    z = _stats.norm.ppf(1 - alpha / 2)
    agg_ci = (agg_att - z * agg_se, agg_att + z * agg_se)
    pval = (
        2 * (1 - _stats.norm.cdf(abs(agg_att) / agg_se))
        if agg_se > 0 else float("nan")
    )

    return CausalResult(
        method="sequential_sdid",
        estimand="ATT",
        estimate=agg_att,
        se=agg_se,
        pvalue=pval,
        ci=agg_ci,
        alpha=alpha,
        n_obs=int(len(data)),
        detail=per_cohort,
        model_info={
            "n_cohorts": int(len(treated_cohorts)),
            "n_valid_cohorts": int(len(valid)),
            "cohort_weights": cohort_weights,
            "se_method": se_method,
            "n_reps": int(n_reps),
            "reference": "Arkhangelsky & Samkov (arXiv:2404.00164, 2024)",
        },
    )
