"""Time-varying covariate DiD with baseline-frozen covariates.

Motivation
----------
The canonical "controlled DiD" regression with contemporaneous
time-varying covariates X_{i,t} suffers a bad-controls problem when
treatment affects the covariates. The fix implemented here is to freeze
covariates at their pre-treatment value X_{i, g-1} and fit an
outcome-regression ATT(g, t) estimator with the frozen covariate.

.. note::
   The attribution is now confirmed: Caetano, Callaway, Payne &
   Rodrigues (2022), "Difference in Differences with Time-Varying
   Covariates", arXiv:2202.02903 (bib key ``caetano2022difference``). An
   earlier pass could not find it and downgraded this to
   "(citation needed)" -- the search had gone through Crossref, which
   indexes arXiv preprints poorly. Verified against the arXiv record and
   DataCite: all four authors, title, year and DOI.

   The paper is a preprint; there is no published version as of this
   writing. Its ``X_{g-1}`` estimator is implemented in Callaway's R
   package ``ptetools`` (and, for the outcome-regression cells, in
   ``did``), which is what this module is pinned against.

This implementation computes, for every treated cohort ``g`` and post
period ``t >= g``, the outcome-regression ATT(g, t) of the long difference
``Y_t - Y_{g-1}`` with covariates frozen at ``g + baseline_offset``
(default ``g - 1``) for treated **and** comparison units alike, the
regression being fitted on the never-treated comparison units only. That is
the ``X_{g-1}`` estimator of ``ptetools::pte_default(d_outcome = TRUE,
est_method = "reg")`` and of ``did::att_gt(est_method = "reg")`` (whose panel
2x2 cells take covariates from the base period); both are pinned in
``tests/reference_parity/test_did_synth_didvar_parity.py``.

Scope & caveats
---------------
- Outcome regression only; no IPW / DR variant yet.
- Never-treated comparison units only.
- The standard error is a unit (cluster) bootstrap, so it agrees with the
  reference's multiplier-bootstrap SE only up to Monte-Carlo error.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core._bootstrap import bootstrap_se as _bootstrap_se
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
    aggregation: str = "group",
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """DiD with time-varying covariates frozen at baseline.

    For each cohort ``g`` and post period ``t >= g``::

        dY_i      = Y_{i,t} - Y_{i,g-1}
        beta_gt   = OLS of dY on (1, X_{i,g+offset}) among never-treated i
        ATT(g, t) = mean over cohort-g units of (dY_i - (1, X_{i,g+offset}) beta_gt)

    Covariates are read at the same period ``g + baseline_offset`` for
    treated and comparison units, so post-treatment movement in ``X`` cannot
    leak into the adjustment.

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
        frozen and used as the controls for every post period of cohort g,
        for both the cohort and its comparison units.
    never_value : any, default 0
        Value in ``cohort`` that marks never-treated units.
    baseline_offset : int, default -1
        Offset relative to first-treatment period for freezing covariates.
        -1 = last pre-treatment period.
    aggregation : {"group", "simple"}, default "group"
        ``"group"``: average ATT(g, t) over each cohort's post periods, then
        across cohorts weighted by cohort size -- the overall ATT reported by
        ``ptetools`` and ``did::aggte(type = "group")``. ``"simple"``: weight
        every post cell by its number of treated units, as
        ``did::aggte(type = "simple")`` does on a balanced panel.
    n_boot : int, default 500
        Cluster-bootstrap replications for SE.
    alpha : float, default 0.05
    seed : int, optional

    Returns
    -------
    CausalResult
        Aggregate ATT; ``detail`` carries per-(g, t) ATTs;
        ``model_info`` carries both aggregates.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(40):
    ...     g = int(rng.choice([3, 5, 0]))  # cohort; never_value=0 = never treated
    ...     for year in range(1, 8):
    ...         on = 1 if (g != 0 and year >= g) else 0
    ...         age = 25 + year + rng.normal(0, 1)
    ...         wage_prev = 10 + 0.3 * age + rng.normal(0, 1)
    ...         rows.append({'i': i, 'year': year, 'g': g, 'age': age,
    ...                      'wage_prev': wage_prev,
    ...                      'earnings': 5 + 0.2 * age + 2.0 * on
    ...                                  + rng.normal(0, 0.5)})
    >>> df = pd.DataFrame(rows)
    >>> r = sp.did_timevarying_covariates(
    ...     df, y='earnings', unit='i', time='year', cohort='g',
    ...     covariates=['age', 'wage_prev'], n_boot=50, seed=0,
    ... )
    >>> bool(np.isfinite(r.estimate))
    True

    References
    ----------
    caetano2022difference
    """
    df = data.copy()
    for col in [y, unit, time, cohort] + list(covariates):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in data")
    if aggregation not in ("group", "simple"):
        raise ValueError(
            f"aggregation must be 'group' or 'simple'; got {aggregation!r}"
        )

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

    kw = dict(
        y=y,
        unit=unit,
        time=time,
        cohort=cohort,
        covariates=list(covariates),
        treated_cohorts=treated_cohorts,
        never_value=never_value,
        baseline_offset=baseline_offset,
    )
    main = _compute_att_gt(df, **kw)
    key = f"att_{aggregation}"

    # Cluster bootstrap
    boot_overall = np.full(n_boot, np.nan)
    for b in range(n_boot):
        bdf = _dc.cluster_bootstrap_draw(
            df,
            cluster_col=unit,
            rng=rng,
            relabel_cols=[unit],
        )
        try:
            best = _compute_att_gt(bdf, **kw)
        except np.linalg.LinAlgError:
            continue  # replicate stays NaN; bootstrap_se tracks the failure
        boot_overall[b] = best[key]

    se = _bootstrap_se(boot_overall, label="did.timevarying_covariates")
    est = float(main[key])
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    if se > 0 and np.isfinite(se):
        z = est / se
        p = float(2 * stats.norm.sf(abs(z)))
        ci = (est - z_crit * se, est + z_crit * se)
    else:
        p = np.nan
        ci = (np.nan, np.nan)

    detail_df = pd.DataFrame(main["cell_estimates"])

    return CausalResult(
        method="DiD with time-varying covariates (baseline-frozen X)",
        estimand=f"ATT ({aggregation} aggregation of cohort x time cells)",
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
            "aggregation": aggregation,
            "att_group": float(main["att_group"]),
            "att_simple": float(main["att_simple"]),
            "estimator": "outcome regression on never-treated units",
            "n_cells": len(main["cell_estimates"]),
            "n_boot": n_boot,
            "cluster_var": unit,
        },
    )


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
    baseline_offset: int = -1,
) -> Dict[str, Any]:
    """ATT(g, t) by outcome regression on covariates frozen at g + offset."""
    cells: List[Dict[str, Any]] = []
    times = set(df[time].unique())

    for g in treated_cohorts:
        pre_t = g - 1
        cov_t = g + baseline_offset
        if pre_t not in times or cov_t not in times:
            continue
        post_times = sorted(t for t in times if t >= g)

        pop = df[(df[cohort] == g) | (df[cohort] == never_value)]
        base = pop[pop[time] == pre_t][[unit, y]].rename(columns={y: "_y_pre"})
        xs = pop[pop[time] == cov_t][[unit] + covariates]
        xs = xs.rename(columns={c: f"_x_{j}" for j, c in enumerate(covariates)})
        base = base.merge(xs, on=unit, how="inner")
        x_cols = [f"_x_{j}" for j in range(len(covariates))]
        for t in post_times:
            post_df = pop[pop[time] == t][[unit, y, cohort]]
            merged = post_df.merge(base, on=unit, how="inner")
            if merged.empty:
                continue
            dy = (merged[y] - merged["_y_pre"]).to_numpy(dtype=float)
            X = np.column_stack(
                [np.ones(len(merged)), merged[x_cols].to_numpy(dtype=float)]
            )
            treated = (merged[cohort] == g).to_numpy()
            valid = np.isfinite(dy) & np.all(np.isfinite(X), axis=1)
            tr = treated & valid
            co = ~treated & valid
            n_treated, n_control = int(tr.sum()), int(co.sum())
            if n_treated < 1 or n_control <= X.shape[1]:
                continue
            beta = np.linalg.lstsq(X[co], dy[co], rcond=None)[0]
            att_gt = float(np.mean(dy[tr] - X[tr] @ beta))
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
        return {"att_group": np.nan, "att_simple": np.nan, "cell_estimates": []}

    cell_df = pd.DataFrame(cells)
    w = cell_df["n_treated"].to_numpy(dtype=float)
    att_simple = float(np.sum(w * cell_df["att_gt"].to_numpy()) / w.sum())
    by_g = cell_df.groupby("cohort").agg(att=("att_gt", "mean"), n=("n_treated", "max"))
    att_group = float(np.sum(by_g["att"] * by_g["n"]) / by_g["n"].sum())
    return {"att_group": att_group, "att_simple": att_simple, "cell_estimates": cells}
