"""
Harvesting Differences-in-Differences and Event-Study Designs.

Implements the unified-estimation framework from MIT/NBER Working Paper
34550 (2025), which "harvests" every valid 2×2 DID comparison implied by
staggered-adoption panel data and combines them into a single
precision-weighted estimator.

The core idea is that under parallel-trends, any 2×2 comparison

    ATT_hat(g, g', t₁, t₂) =
        [ Ȳ(g, t₂) - Ȳ(g, t₁) ] - [ Ȳ(g', t₂) - Ȳ(g', t₁) ]

where ``g`` is a treated cohort, ``g'`` a control cohort that is not yet
treated at ``t₂``, and ``t₁ < t₂`` straddle ``g``'s treatment time, is an
unbiased estimator of the treatment effect on cohort ``g`` between
``t₁`` and ``t₂``.  Harvesting collects all such valid comparisons and
weights them by inverse variance.

The result is numerically equivalent to the Callaway–Sant'Anna (2021)
ATT(g, t) building blocks when restricted to a single horizon ``t₂``,
but generalises naturally to (a) pre-period placebo tests, (b)
event-study aggregation, and (c) long-difference contrasts.

References
----------
Abadie, Angrist, Frandsen & Pischke (NBER WP 34550, 2025).
"Harvesting Differences-in-Differences and Event-Study Evidence."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.results import CausalResult


__all__ = [
    "harvest_did",
    "HarvestDIDResult",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class HarvestDIDResult:
    """Full diagnostic output of :func:`harvest_did`."""

    estimate: float
    se: float
    ci: tuple
    alpha: float
    n_comparisons: int
    comparisons: pd.DataFrame
    event_study: pd.DataFrame
    pretrend_test: Dict[str, float]
    method: str = "harvest_did"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lo, hi = self.ci
        lines = [
            "Harvesting DID / Event-Study (Abadie et al. NBER WP 34550, 2025)",
            "-" * 62,
            f"  Aggregate ATT          : {self.estimate:+.6f}",
            f"  Standard error         : {self.se:.6f}",
            f"  {100 * (1 - self.alpha):.0f}% CI                 : "
            f"[{lo:+.6f}, {hi:+.6f}]",
            f"  # 2x2 comparisons      : {self.n_comparisons}",
            f"  Pre-trend joint p-value: {self.pretrend_test.get('pvalue', float('nan')):.4f}",
        ]
        if not self.event_study.empty:
            lines.append("")
            lines.append("  Event study (relative_time -> ATT):")
            lines.append(self.event_study.head(10).to_string(index=False))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _cell_means(
    df: pd.DataFrame, *, unit: str, time: str, outcome: str,
) -> pd.DataFrame:
    """Compute Ȳ(i, t), Ȳ variance, and n for each (unit, time) cell."""
    grp = df.groupby([unit, time])[outcome]
    out = grp.agg(["mean", "var", "count"]).reset_index()
    out.columns = [unit, time, "ybar", "yvar", "n"]
    out["yvar"] = out["yvar"].fillna(0.0)
    return out


def _build_cohort(
    df: pd.DataFrame, *, unit: str, time: str, treat: str,
    cohort: Optional[str] = None, never_value: Any = 0,
) -> pd.DataFrame:
    """Return (unit -> treatment time) mapping.  ``never_value`` signals
    'never treated'.
    """
    if cohort is not None:
        return df[[unit, cohort]].drop_duplicates().rename(columns={cohort: "__g__"})
    # Derive first-treatment time from the binary `treat` column.
    treated = df[df[treat].astype(bool)]
    if treated.empty:
        raise ValueError("No treated observations found — check `treat` column.")
    first = treated.groupby(unit)[time].min().rename("__g__").reset_index()
    all_units = df[unit].drop_duplicates().to_frame()
    out = all_units.merge(first, on=unit, how="left")
    out["__g__"] = out["__g__"].fillna(never_value)
    return out


def _harvest_comparisons(
    means: pd.DataFrame,
    cohort_map: pd.DataFrame,
    *,
    unit: str,
    time: str,
    never_value: Any,
    horizons: Sequence[int],
    reference: int,
) -> List[Dict[str, Any]]:
    """Enumerate all valid 2x2 DID comparisons indexed by (cohort, horizon).

    For each treated cohort ``g`` and each target horizon ``e``:

      - Let t₂ = g + e  (outcome period)
      - Let t₁ = g + reference  (pre-treatment reference period)
      - Let the control group be the union of all cohorts not yet
        treated at t₂ (including never-treated).

    The 2×2 estimate contrasts (ybar_{g, t₂} − ybar_{g, t₁}) against
    the weighted mean of the same contrast over the control group.
    """
    means = means.merge(cohort_map, on=unit, how="left")
    records: List[Dict[str, Any]] = []
    # Index by (unit, time) for fast lookup via pivot tables.
    pivot_y = means.pivot(index=unit, columns=time, values="ybar")
    all_times = sorted(pivot_y.columns.tolist())
    cohorts = cohort_map.set_index(unit)["__g__"]
    unique_gs = sorted(
        [g for g in cohorts.unique() if g != never_value and pd.notna(g)]
    )

    for g in unique_gs:
        treated_units = cohorts[cohorts == g].index
        for e in horizons:
            t2 = g + e
            t1 = g + reference
            if t1 not in all_times or t2 not in all_times or t1 == t2:
                continue
            # Clean-control cohorts: not yet treated in either reference
            # period (max of t1, t2 guards pre-period placebos too).
            t_last = max(t1, t2)
            control_mask = (cohorts == never_value) | (cohorts > t_last)
            control_units = cohorts[control_mask].index
            if len(treated_units) == 0 or len(control_units) == 0:
                continue
            try:
                yT_t2 = pivot_y.loc[treated_units, t2].to_numpy(dtype=float)
                yT_t1 = pivot_y.loc[treated_units, t1].to_numpy(dtype=float)
                yC_t2 = pivot_y.loc[control_units, t2].to_numpy(dtype=float)
                yC_t1 = pivot_y.loc[control_units, t1].to_numpy(dtype=float)
            except KeyError:
                continue
            # Drop units missing any of the four cells.
            ok_T = np.isfinite(yT_t2) & np.isfinite(yT_t1)
            ok_C = np.isfinite(yC_t2) & np.isfinite(yC_t1)
            if ok_T.sum() < 2 or ok_C.sum() < 2:
                continue
            # Unit-level long differences.
            dT = yT_t2[ok_T] - yT_t1[ok_T]
            dC = yC_t2[ok_C] - yC_t1[ok_C]
            delta_T = float(np.mean(dT))
            delta_C = float(np.mean(dC))
            att = float(delta_T - delta_C)
            # Cluster-robust variance at the unit level: difference of two
            # independent sample means.
            v_T = float(np.var(dT, ddof=1) / len(dT)) if len(dT) > 1 else 0.0
            v_C = float(np.var(dC, ddof=1) / len(dC)) if len(dC) > 1 else 0.0
            se = float(np.sqrt(max(v_T + v_C, 0.0)))
            records.append(
                dict(
                    cohort=g,
                    horizon=e,
                    t1=t1,
                    t2=t2,
                    att=att,
                    se=se,
                    weight=float(ok_T.sum()),
                    n_treated=int(ok_T.sum()),
                    n_control=int(ok_C.sum()),
                )
            )
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def harvest_did(
    data: pd.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
    treat: Optional[str] = None,
    cohort: Optional[str] = None,
    never_value: Any = 0,
    horizons: Optional[Sequence[int]] = None,
    reference: int = -1,
    alpha: float = 0.05,
    weighting: str = "precision",
) -> CausalResult:
    """Harvest every valid 2×2 DID comparison and aggregate them.

    Parameters
    ----------
    data : DataFrame
        Long-format panel.
    unit, time, outcome : str
        Column names.
    treat : str, optional
        Binary treatment indicator.  If provided, the cohort (first
        treatment time) is inferred per unit.
    cohort : str, optional
        Alternative to ``treat``: a column containing the already-computed
        cohort (first treatment time) per unit.
    never_value : any, default 0
        Value that marks "never treated" in the ``cohort`` column.  If
        you use ``treat``, units without any treated observation are
        mapped to this value automatically.
    horizons : sequence of int, optional
        Event-time horizons to evaluate.  Defaults to ``[-3, -2, -1, 0,
        1, 2, 3, 4]``.  Positive values are post-treatment, ``0`` is the
        first treated period, negative values are placebo/pre-trend.
    reference : int, default -1
        Pre-treatment reference horizon relative to each cohort's
        treatment time.  ``-1`` = period immediately before treatment
        (standard event-study convention).
    alpha : float, default 0.05
    weighting : {'precision', 'equal', 'n_treated'}, default 'precision'
        How to aggregate the harvested 2×2 estimates.
        ``precision``  uses inverse-variance weights (minimum-variance
        aggregate under independence).
        ``equal`` averages without weights.
        ``n_treated`` weights each comparison by its treated-unit count.

    Returns
    -------
    CausalResult
        ``estimand`` is the aggregated post-treatment ATT (average over
        non-negative horizons).  ``detail`` exposes the full 2×2 table;
        ``model_info['event_study']`` is the per-horizon aggregation.

    Notes
    -----
    Inference assumes independence across **units within each cohort**
    (unit-level cluster-robust SEs), but the cross-horizon covariance
    induced by shared units is ignored when aggregating the event study
    into a single ATT.  For strict inference, wrap this call in
    :func:`sp.inference.bootstrap` at the unit level, or use the
    per-comparison table to feed :func:`sp.inference.multiway_cluster_vcov`.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.utils.dgp_did(n_units=80, n_periods=12, seed=0)
    >>> out = sp.harvest_did(
    ...     df, unit='unit', time='time', outcome='y',
    ...     treat='treated', horizons=range(-3, 5),
    ... )
    >>> out.estimate  # doctest: +SKIP
    """
    for col in (unit, time, outcome):
        if col not in data.columns:
            raise ValueError(f"column {col!r} not in data")
    if treat is None and cohort is None:
        raise ValueError("either `treat` or `cohort` must be supplied")
    if horizons is None:
        horizons = list(range(-3, 5))
    horizons = list(horizons)
    if weighting not in ("precision", "equal", "n_treated"):
        raise ValueError(
            f"weighting must be 'precision', 'equal', or 'n_treated'; "
            f"got {weighting!r}"
        )

    cohort_map = _build_cohort(
        data, unit=unit, time=time,
        treat=treat or "__unused__",
        cohort=cohort, never_value=never_value,
    )
    means = _cell_means(data, unit=unit, time=time, outcome=outcome)

    records = _harvest_comparisons(
        means, cohort_map,
        unit=unit, time=time,
        never_value=never_value,
        horizons=horizons,
        reference=reference,
    )
    if not records:
        raise RuntimeError(
            "Harvest produced 0 valid 2x2 comparisons — check cohort / "
            "horizons alignment against the panel's time range."
        )
    tbl = pd.DataFrame(records)

    # --- Per-horizon event-study aggregation ------------------------------
    from scipy.stats import norm as _norm
    event_rows = []
    for e in sorted(tbl["horizon"].unique()):
        sub = tbl[tbl["horizon"] == e]
        w = _weights(sub, weighting)
        att_e = float(np.average(sub["att"], weights=w))
        # Variance of weighted mean with weights summing to 1
        w_n = w / w.sum() if w.sum() > 0 else w
        var_e = float(np.sum((w_n ** 2) * (sub["se"].to_numpy() ** 2)))
        se_e = float(np.sqrt(max(var_e, 0.0)))
        pv_e = (
            float(2 * (1 - _norm.cdf(abs(att_e) / se_e)))
            if se_e > 0 else float("nan")
        )
        event_rows.append(
            dict(relative_time=int(e),
                 att=att_e,
                 se=se_e,
                 pvalue=pv_e,
                 n_comparisons=int(len(sub)))
        )
    event_study = pd.DataFrame(event_rows)

    # --- Aggregate ATT over non-negative horizons -------------------------
    post = event_study[event_study["relative_time"] >= 0]
    if len(post) == 0:
        raise RuntimeError(
            "No post-treatment horizons in the harvest — extend `horizons`."
        )
    w_post = 1.0 / np.maximum(post["se"].to_numpy() ** 2, 1e-12)
    agg = float(np.average(post["att"], weights=w_post))
    w_post_n = w_post / w_post.sum()
    agg_var = float(np.sum((w_post_n ** 2) * (post["se"].to_numpy() ** 2)))
    agg_se = float(np.sqrt(max(agg_var, 0.0)))

    # --- Pre-trend joint test (Wald of horizon<0 ATTs) --------------------
    pre = event_study[event_study["relative_time"] < 0]
    if len(pre) > 0 and (pre["se"] > 0).all():
        chi2 = float(np.sum((pre["att"] / pre["se"]) ** 2))
        from scipy.stats import chi2 as _chi2
        pv = float(1 - _chi2.cdf(chi2, df=len(pre)))
        pretrend = {"chi2": chi2, "df": int(len(pre)), "pvalue": pv}
    else:
        pretrend = {"chi2": float("nan"), "df": 0, "pvalue": float("nan")}

    # --- CI / p-value -----------------------------------------------------
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    ci = (agg - z * agg_se, agg + z * agg_se)
    pval = 2 * (1 - norm.cdf(abs(agg) / agg_se)) if agg_se > 0 else float("nan")

    _result = CausalResult(
        method="harvest_did",
        estimand="ATT (post-treatment average)",
        estimate=agg,
        se=agg_se,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=int(len(data)),
        detail=tbl,
        model_info={
            "n_comparisons": int(len(tbl)),
            "event_study": event_study,
            "pretrend_test": pretrend,
            "weighting": weighting,
            "horizons": list(horizons),
            "reference": int(reference),
        },
        _citation_key="harvest_did",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.did.harvest_did",
            params={
                "unit": unit, "time": time, "outcome": outcome,
                "treat": treat, "cohort": cohort,
                "never_value": never_value,
                "horizons": list(horizons) if horizons else None,
                "reference": int(reference),
                "alpha": alpha,
                "weighting": weighting,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _weights(sub: pd.DataFrame, scheme: str) -> np.ndarray:
    if scheme == "precision":
        return 1.0 / np.maximum(sub["se"].to_numpy() ** 2, 1e-12)
    if scheme == "equal":
        return np.ones(len(sub))
    if scheme == "n_treated":
        return sub["n_treated"].to_numpy(dtype=float)
    raise ValueError(scheme)
