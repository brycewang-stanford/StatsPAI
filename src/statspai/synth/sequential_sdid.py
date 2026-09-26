"""
Cohort-by-cohort synthetic difference-in-differences for staggered adoption.

.. warning::
   Despite its name, :func:`sequential_sdid` is **not** the Sequential SDiD
   estimator of Arkhangelsky & Samkov (arXiv:2404.00164). Their Algorithm 1
   works on cohort-aggregated outcomes, loops over event horizons ``k`` and
   cohorts, uses later cohorts as donors, and *imputes* each estimated
   effect back into the treated cohort's outcome (``Y_{a,a+k} -= tau_{a,k}``)
   so that it serves as pre-period history for later horizons; it uses a
   penalty ``eta^2 sum omega_j^2 / pi_j`` and a Bayesian bootstrap. None of
   that is implemented here, and no public code of that estimator was found
   to compare against.

What this module computes: for each adoption cohort ``g`` (in adoption
order) it runs the classic single-block SDID of Arkhangelsky, Athey,
Hirshberg, Imbens & Wager (``sp.sdid``, which matches R ``synthdid``) on
the sub-panel of cohort-``g`` units plus never-treated and not-yet-treated
units, truncated at the period before the next cohort adopts, and then
averages the cohort ATTs. Each cohort-level ATT(g) therefore only covers
the window before the next cohort's adoption (the last cohort's window runs
to the end of the panel).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin
from ..core.results import CausalResult
from .sdid import sdid as _sdid_base

__all__ = ["sequential_sdid", "SequentialSDIDResult"]


@dataclass
class SequentialSDIDResult(ResultProtocolMixin):
    """Per-cohort and aggregated output of :func:`sequential_sdid`.

    Dataclass container bundling the aggregate ATT / SE / CI with a
    ``per_cohort`` table of cohort-specific ATT(g). Exposes a formatted
    :meth:`summary`.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> per_cohort = pd.DataFrame(
    ...     {"cohort": [5, 8], "att": [2.10, 1.90], "se": [0.30, 0.40]}
    ... )
    >>> res = sp.SequentialSDIDResult(
    ...     aggregate_att=2.0, aggregate_se=0.25,
    ...     aggregate_ci=(1.51, 2.49), per_cohort=per_cohort,
    ... )
    >>> len(res.per_cohort)
    2
    >>> summary_text = res.summary()
    """

    _citation_keys = ("arkhangelsky2024sequential",)
    aggregate_att: float
    aggregate_se: float
    aggregate_ci: tuple
    per_cohort: pd.DataFrame
    model_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lo, hi = self.aggregate_ci
        lines = [
            "Cohort-by-cohort SDID (sequential_sdid; not Arkhangelsky-Samkov)",
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


@accepts_aliases(id="unit", y="outcome")
def sequential_sdid(
    data: pd.DataFrame,
    *,
    outcome: str,
    unit: str,
    time: str,
    cohort: str,
    never_treated_value: Any = 0,
    se_method: Literal["placebo", "bootstrap", "jackknife"] = "placebo",
    n_reps: int = 200,
    cohort_weights: str = "size",
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """Cohort-by-cohort SDID for staggered adoption (see module warning).

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
        Aggregation weights across cohorts. ``'size'`` weights each cohort
        by its number of treated units (not by units x post-periods).
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
    Algorithm (see the module docstring -- this is not Arkhangelsky &
    Samkov's Algorithm 1):

    1. Sort cohorts by treatment time ``g_1 < g_2 < ... < g_K``.
    2. For each cohort ``g_k``: keep units in cohort ``g_k``, never-treated
       units and units with ``cohort > g_k``, and periods
       ``t <= g_{k+1} - 1`` (all periods for the last cohort); run
       single-block SDID on that sub-panel. ATT(g_k) equals
       ``synthdid::synthdid_estimate`` on the same sub-panel.
    3. Aggregate ATT(g) with ``cohort_weights``; the aggregate SE treats the
       cohort estimates as independent (``sqrt(sum w_g^2 se_g^2)``).

    With a single cohort this is exactly ``sp.sdid``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> periods = np.arange(1, 11)
    >>> rows = []
    >>> for u in range(16):
    ...     g = 5 if u < 2 else (8 if u < 4 else 0)
    ...     for t in periods:
    ...         treated = g != 0 and t >= g
    ...         val = (u * 0.3 + t * 0.2 + 2.0 * treated
    ...                + rng.normal(0, 0.2))
    ...         rows.append({"y": val, "unit": u, "time": t,
    ...                      "cohort": g})
    >>> panel = pd.DataFrame(rows)
    >>> res = sp.sequential_sdid(
    ...     panel, outcome="y", unit="unit", time="time",
    ...     cohort="cohort", seed=42,
    ... )
    >>> res.model_info["n_cohorts"]  # two adoption waves
    2
    >>> round(res.estimate, 1)  # true effect = 2.0
    1.7
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    for col in (outcome, unit, time, cohort):
        if col not in data.columns:
            raise ValueError(f"Column {col!r} not found in `data`.")

    df = data.copy()
    # Treated cohorts (sorted in adoption order).
    treated_cohorts = sorted(
        c for c in df[cohort].unique() if not pd.isna(c) and c != never_treated_value
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
            t_max_g = (
                next_g - 1
                if np.issubdtype(type(next_g), np.integer)
                else (df.loc[df[time] < next_g, time].max())
            )
        else:
            t_max_g = max_time

        # Donor pool: never-treated + not-yet-treated (cohort > g) restricted
        # to times ≤ t_max_g.
        donor_mask = ((df[cohort] == never_treated_value) | (df[cohort] > g)) & (
            df[time] <= t_max_g
        )
        treated_mask = (df[cohort] == g) & (df[time] <= t_max_g)

        sub = (
            pd.concat([df.loc[donor_mask], df.loc[treated_mask]], axis=0)
            .drop_duplicates(subset=[unit, time])
            .reset_index(drop=True)
        )

        treated_units = sub.loc[sub[cohort] == g, unit].unique().tolist()
        if not treated_units:
            continue  # pragma: no cover
        # Need at least 2 pre-periods for SDID time weights.
        pre_times = sub.loc[sub[time] < g, time].unique()
        post_times = sub.loc[(sub[time] >= g) & (sub[time] <= t_max_g), time].unique()
        if pre_times.size < 2 or post_times.size < 1:
            per_cohort_rows.append(
                {
                    "cohort": g,
                    "treatment_period": g,
                    "att": np.nan,
                    "se": np.nan,
                    "n_treated": len(treated_units),
                    "n_donors": int(
                        (sub[cohort] != g).sum() / max(len(sub[time].unique()), 1)
                    ),
                    "note": "insufficient pre/post periods",
                }
            )
            continue  # pragma: no cover

        try:
            res_g = _sdid_base(
                sub,
                outcome=outcome,
                unit=unit,
                time=time,
                treated_unit=treated_units,
                treatment_time=g,
                method="sdid",
                se_method=se_method,
                n_reps=n_reps,
                seed=seed,
                alpha=alpha,
            )
            per_cohort_rows.append(
                {
                    "cohort": g,
                    "treatment_period": g,
                    "att": float(res_g.estimate),
                    "se": float(res_g.se),
                    "n_treated": len(treated_units),
                    "n_donors": int(len(sub.loc[sub[cohort] != g, unit].unique())),
                    "note": "",
                }
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            per_cohort_rows.append(
                {
                    "cohort": g,
                    "treatment_period": g,
                    "att": np.nan,
                    "se": np.nan,
                    "n_treated": len(treated_units),
                    "n_donors": int(len(sub.loc[sub[cohort] != g, unit].unique())),
                    "note": f"SDID failed: {type(exc).__name__}: {exc}",
                }
            )

    per_cohort = pd.DataFrame(per_cohort_rows)
    failed_notes = [
        f"cohort {r['cohort']}: {r['note']}" for r in per_cohort_rows if r.get("note")
    ]
    if failed_notes:
        import warnings

        warnings.warn(
            "sequential_sdid dropped cohort(s) from the aggregate: "
            + "; ".join(failed_notes),
            RuntimeWarning,
            stacklevel=2,
        )
    valid = per_cohort.dropna(subset=["att", "se"]).copy()
    if valid.empty:
        raise RuntimeError(  # pragma: no cover
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
    agg_var = float(np.sum((w**2) * (valid["se"].to_numpy() ** 2)))
    agg_se = float(np.sqrt(agg_var))
    # Normal-approx CI
    from scipy import stats as _stats

    z = _stats.norm.ppf(1 - alpha / 2)
    agg_ci = (agg_att - z * agg_se, agg_att + z * agg_se)
    pval = 2 * _stats.norm.sf(abs(agg_att) / agg_se) if agg_se > 0 else float("nan")

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
            "estimator": "cohort_by_cohort_sdid",
            "reference": (
                "cohort-wise SDID (Arkhangelsky et al. 2021 per cohort); not "
                "the Arkhangelsky & Samkov (arXiv:2404.00164) sequential "
                "imputation algorithm"
            ),
        },
    )
