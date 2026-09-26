"""
Harvesting every valid 2x2 DID comparison in a staggered panel.

For each treated cohort ``g`` and event horizon ``e`` the building block is
the 2x2 comparison

    ATT_hat(g, g + e) =
        [ Ybar(g, g+e) - Ybar(g, g-1) ] - [ Ybar(C, g+e) - Ybar(C, g-1) ]

with a *universal* base period ``g - 1`` (``reference=-1``) and a clean
control group ``C`` = never-treated units plus cohorts not yet treated at
``max(g - 1, g + e)`` -- cohort ``g`` itself excluded. That is exactly the
Callaway--Sant'Anna ATT(g, t) of ``did::att_gt(control_group =
"notyettreated", base_period = "universal")`` without covariates
[@callaway2021difference]. The cells are then combined into an event study
(per horizon, over cohorts) and one post-treatment aggregate, under a
choice of weights.

Scope of the name. "Harvesting" follows the title of Abadie, Angrist,
Frandsen & Pischke (NBER WP 34550, 2025) [@abadie2025harvesting], a survey
chapter on DiD and event-study evidence. That chapter does not define this
estimator or its inverse-variance aggregation; earlier versions of this
docstring said it did, which was wrong. The building blocks are
Callaway--Sant'Anna's; the ``precision`` aggregation is StatsPAI's own.

Inference. Every cell carries its unit-level influence function (the
``did`` package's analytic form: population variances, divisor ``n``), so
event-study and aggregate standard errors include the covariance created by
units shared across cells -- the same never-treated controls serve every
cohort, and the same treated units appear at every horizon. With
``weighting="n_treated"`` the event-study weights are estimated cohort
shares and their own influence function is added, which reproduces
``did::aggte(type = "dynamic")`` exactly. ``precision`` weights are treated
as fixed (their estimation error is ignored, as in feasible GLS).

References
----------
callaway2021difference
abadie2025harvesting
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin
from ..core.results import CausalResult

__all__ = [
    "harvest_did",
    "HarvestDIDResult",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class HarvestDIDResult(ResultProtocolMixin):
    """Full diagnostic output of :func:`harvest_did`.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> es = pd.DataFrame({
    ...     "relative_time": [0, 1], "att": [0.4, 0.5],
    ...     "se": [0.1, 0.1], "pvalue": [0.01, 0.02], "n_comparisons": [3, 3],
    ... })
    >>> res = sp.HarvestDIDResult(
    ...     estimate=0.45, se=0.07, ci=(0.31, 0.59), alpha=0.05,
    ...     n_comparisons=6,
    ...     comparisons=pd.DataFrame({"att": [0.4]}),
    ...     event_study=es,
    ...     pretrend_test={"pvalue": 0.6},
    ... )
    >>> float(res.estimate)
    0.45
    """

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
            "Harvesting DID / Event-Study (2x2 cells harvested and aggregated)",
            "-" * 62,
            f"  Aggregate ATT          : {self.estimate:+.6f}",
            f"  Standard error         : {self.se:.6f}",
            f"  {100 * (1 - self.alpha):.0f}% CI                 : "
            f"[{lo:+.6f}, {hi:+.6f}]",
            f"  # 2x2 comparisons      : {self.n_comparisons}",
            "  Pre-trend joint p-value: "
            f"{self.pretrend_test.get('pvalue', float('nan')):.4f}",
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
    df: pd.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
) -> pd.DataFrame:
    """Compute Ȳ(i, t), Ȳ variance, and n for each (unit, time) cell."""
    grp = df.groupby([unit, time])[outcome]
    out = grp.agg(["mean", "var", "count"]).reset_index()
    out.columns = [unit, time, "ybar", "yvar", "n"]
    out["yvar"] = out["yvar"].fillna(0.0)
    return out


def _build_cohort(
    df: pd.DataFrame,
    *,
    unit: str,
    time: str,
    treat: str,
    cohort: Optional[str] = None,
    never_value: Any = 0,
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
) -> tuple:
    """Enumerate all valid 2x2 DID comparisons indexed by (cohort, horizon).

    For each treated cohort ``g`` and each target horizon ``e``:

      - t2 = g + e (outcome period), t1 = g + reference (base period);
      - the control group is every never-treated unit plus every cohort
        not yet treated at ``max(t1, t2)``, excluding cohort ``g`` itself
        (for a pre-period placebo, ``max(t1, t2) = t1 < g`` and cohort
        ``g`` would otherwise count as its own control).

    Returns ``(records, psi, n_units, unit_cohort)``: ``psi`` is the
    ``(n_units, n_cells)`` matrix of unit-level influence functions,
    scaled so that ``se = sqrt(sum(psi**2)) / n_units``.
    """
    means = means.merge(cohort_map, on=unit, how="left")
    records: List[Dict[str, Any]] = []
    psis: List[np.ndarray] = []
    pivot_y = means.pivot(index=unit, columns=time, values="ybar")
    all_times = sorted(pivot_y.columns.tolist())
    cohorts = cohort_map.drop_duplicates(subset=[unit]).set_index(unit)["__g__"]
    cohorts = cohorts.reindex(pivot_y.index)
    n_units = len(pivot_y.index)
    is_never = (cohorts == never_value) | cohorts.isna()
    unique_gs = sorted(
        [g for g in cohorts.unique() if g != never_value and pd.notna(g)]
    )

    for g in unique_gs:
        is_g = (cohorts == g).to_numpy()
        for e in horizons:
            t2 = g + e
            t1 = g + reference
            if t1 not in all_times or t2 not in all_times or t1 == t2:
                continue
            t_last = max(t1, t2)
            ctrl = (is_never | ((cohorts > t_last) & (cohorts != g))).to_numpy()
            if not is_g.any() or not ctrl.any():
                continue
            d = (pivot_y[t2] - pivot_y[t1]).to_numpy(dtype=float)
            ok = np.isfinite(d)
            sel_T = is_g & ok
            sel_C = ctrl & ok
            if sel_T.sum() < 2 or sel_C.sum() < 2:
                continue
            dT = d[sel_T]
            dC = d[sel_C]
            att = float(dT.mean() - dC.mean())
            psi = np.zeros(n_units, dtype=float)
            psi[sel_T] = (dT - dT.mean()) * (n_units / sel_T.sum())
            psi[sel_C] = -(dC - dC.mean()) * (n_units / sel_C.sum())
            se = float(np.sqrt(np.sum(psi**2)) / n_units)
            records.append(
                dict(
                    cohort=g,
                    horizon=e,
                    t1=t1,
                    t2=t2,
                    att=att,
                    se=se,
                    weight=float(sel_T.sum()),
                    n_treated=int(sel_T.sum()),
                    n_control=int(sel_C.sum()),
                )
            )
            psis.append(psi)
    psi_mat = np.column_stack(psis) if psis else np.zeros((n_units, 0), dtype=float)
    return records, psi_mat, n_units, cohorts.to_numpy()


def _share_wif(
    cohorts_of_cells: np.ndarray,
    unit_cohort: np.ndarray,
    n_units: int,
) -> np.ndarray:
    """Influence function of cohort-share weights ``w_k = p_{g_k} / sum p``.

    ``did``'s ``wif()``: column ``k`` is
    ``(1{G=g_k} - p_k) / sum(p) - sum_j (1{G=g_j} - p_j) * p_k / sum(p)^2``
    with ``p_g`` the share of units in cohort ``g``.
    """
    ind = np.column_stack([(unit_cohort == g).astype(float) for g in cohorts_of_cells])
    p = ind.mean(axis=0)
    sp_ = p.sum()
    if1 = (ind - p) / sp_
    if2 = np.sum(ind - p, axis=1, keepdims=True) * (p / sp_**2)[None, :]
    return if1 - if2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@accepts_aliases(id="unit", y="outcome")
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
    Units are the sampling unit (independent across units, arbitrary
    dependence over time within a unit). Each 2x2 cell's standard error is
    the influence-function form of ``did::att_gt`` (population variances,
    divisor ``n``); event-study, aggregate and pre-trend inference use the
    joint covariance of the cells through their unit-level influence
    functions. Up to StatsPAI 1.28.0 cell SEs used ``ddof=1`` sample
    variances and every aggregation assumed independent cells, which
    understated the event-study and aggregate SEs; pre-period placebo
    cells also counted the treated cohort among its own controls.

    With ``weighting="n_treated"`` the cells and the event study reproduce
    ``did::att_gt(control_group="notyettreated", base_period="universal")``
    and ``did::aggte(type="dynamic")`` (``tests/reference_parity/
    test_did_synth_misc_parity.py``). The post-treatment aggregate is an
    inverse-variance average over horizons, which has no counterpart in
    ``did`` (``aggte`` averages horizons equally).

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
        data,
        unit=unit,
        time=time,
        treat=treat or "__unused__",
        cohort=cohort,
        never_value=never_value,
    )
    means = _cell_means(data, unit=unit, time=time, outcome=outcome)

    records, psi, n_units, unit_cohort = _harvest_comparisons(
        means,
        cohort_map,
        unit=unit,
        time=time,
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
    # Each horizon's influence function is the weighted sum of its cells'
    # influence functions, so the covariance created by shared control
    # units enters the standard error. For ``n_treated`` the weights are
    # estimated cohort shares and their influence function is added
    # (did::aggte(type = "dynamic")).
    from scipy.stats import norm as _norm

    event_rows = []
    horizon_if = []
    for e in sorted(tbl["horizon"].unique()):
        idx = np.flatnonzero(tbl["horizon"].to_numpy() == e)
        sub = tbl.iloc[idx]
        w = _weights(sub, weighting)
        w_n = w / w.sum()
        att_c = sub["att"].to_numpy(dtype=float)
        att_e = float(np.sum(w_n * att_c))
        if_e = psi[:, idx] @ w_n
        if weighting == "n_treated":
            if_e = (
                if_e
                + _share_wif(sub["cohort"].to_numpy(), unit_cohort, n_units) @ att_c
            )
        se_e = float(np.sqrt(np.sum(if_e**2)) / n_units)
        pv_e = float(2 * _norm.sf(abs(att_e) / se_e)) if se_e > 0 else float("nan")
        event_rows.append(
            dict(
                relative_time=int(e),
                att=att_e,
                se=se_e,
                pvalue=pv_e,
                n_comparisons=int(len(sub)),
            )
        )
        horizon_if.append(if_e)
    event_study = pd.DataFrame(event_rows)
    horizon_if = np.column_stack(horizon_if)

    # --- Aggregate ATT over non-negative horizons -------------------------
    # Inverse-variance weights across horizons (treated as fixed); the
    # variance uses the full cross-horizon covariance, because the same
    # treated and control units appear at every horizon.
    post_idx = np.flatnonzero(event_study["relative_time"].to_numpy() >= 0)
    if len(post_idx) == 0:
        raise RuntimeError(
            "No post-treatment horizons in the harvest — extend `horizons`."
        )
    post = event_study.iloc[post_idx]
    w_post = 1.0 / np.maximum(post["se"].to_numpy() ** 2, 1e-12)
    w_post_n = w_post / w_post.sum()
    agg = float(np.sum(w_post_n * post["att"].to_numpy()))
    agg_if = horizon_if[:, post_idx] @ w_post_n
    agg_se = float(np.sqrt(np.sum(agg_if**2)) / n_units)

    # --- Pre-trend joint test (Wald of horizon<0 ATTs) --------------------
    # Uses the joint covariance of the pre-period horizons: they share the
    # base period g-1 and the control units, so they are not independent.
    pre_idx = np.flatnonzero(event_study["relative_time"].to_numpy() < 0)
    if len(pre_idx) > 0 and (event_study["se"].iloc[pre_idx] > 0).all():
        from scipy.stats import chi2 as _chi2

        b_pre = event_study["att"].to_numpy()[pre_idx]
        if_pre = horizon_if[:, pre_idx]
        V_pre = (if_pre.T @ if_pre) / n_units**2
        rank = int(np.linalg.matrix_rank(V_pre))
        if rank < len(pre_idx):
            warnings.warn(
                "harvest_did: the pre-period covariance matrix is singular "
                f"(rank {rank} of {len(pre_idx)}); the pre-trend Wald test "
                "uses its pseudo-inverse with df = rank.",
                UserWarning,
                stacklevel=2,
            )
        chi2 = float(b_pre @ np.linalg.pinv(V_pre) @ b_pre)
        pv = float(_chi2.sf(chi2, df=rank))
        # ``statistic`` is the DiD family's canonical key for this payload
        # (_core / callaway_santanna / bjs_inference / did_multiplegt /
        # did_imputation all use it, and CausalResult.summary() reads it).
        # harvest_did emitted only ``chi2``, so .summary() died with
        # ``KeyError: 'statistic'``. Emit both: the canonical key for the
        # shared machinery, ``chi2`` kept as an alias so anything already
        # reading it keeps working.
        pretrend = {
            "statistic": chi2,
            "chi2": chi2,
            "df": rank,
            "pvalue": pv,
        }
    else:
        pretrend = {
            "statistic": float("nan"),
            "chi2": float("nan"),
            "df": 0,
            "pvalue": float("nan"),
        }

    # --- CI / p-value -----------------------------------------------------
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    ci = (agg - z * agg_se, agg + z * agg_se)
    pval = 2 * norm.sf(abs(agg) / agg_se) if agg_se > 0 else float("nan")

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
            "aggregate_horizon_weights": dict(
                zip(post["relative_time"].astype(int).tolist(), w_post_n.tolist())
            ),
            "se_convention": (
                "influence function, divisor n (did::att_gt analytic SE); "
                "event-study and aggregate SEs include cross-cell covariance"
            ),
        },
        _citation_key="harvest_did",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.harvest_did",
            params={
                "unit": unit,
                "time": time,
                "outcome": outcome,
                "treat": treat,
                "cohort": cohort,
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
        return np.asarray(
            1.0 / np.maximum(sub["se"].to_numpy() ** 2, 1e-12),
            dtype=float,
        )
    if scheme == "equal":
        return np.ones(len(sub), dtype=float)
    if scheme == "n_treated":
        return np.asarray(sub["n_treated"].to_numpy(dtype=float), dtype=float)
    raise ValueError(scheme)
