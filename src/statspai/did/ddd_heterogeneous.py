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

References
----------
- Olden & Møen (2022). *The Econometrics Journal*, DOI 10.1093/ectj/utac010.
  Verified via paper.bib `olden2022triple`.
- Callaway & Sant'Anna (2021) `callaway2021difference` for the
  group-time aggregation template.

.. note::
   **(citation needed.)** A CS-style decomposition for DDD is also associated
   with work by Strezhnev, but the exact title / venue / DOI could not be
   confirmed against Crossref, so no citation string is asserted here and
   nothing was added to ``paper.bib``. Per CLAUDE.md section 10 a missing
   citation is preferable to an unverified one.

Cross-language reference
------------------------
The per-(g, t) cells are pinned against ``triplediff::ddd`` (Ortiz-Villavicencio
& Sant'Anna 2025, `ortiz2025better`) with ``xformla = ~1``, ``est_method="dr"``,
``control_group="nevertreated"``: with no covariates the doubly-robust DDD
reduces to the unconditional cell means computed here, and the two agree to
machine precision. See ``tests/reference_parity/test_ddd_triplediff_parity.py``
and Track A module ``77_ddd``.

The **aggregation weights differ by convention** and are exposed as
``weight_by``; see that parameter's documentation. The cell estimates --
the substantive object -- are identical either way.

Scope & caveats
---------------
- First cut supports **never-treated controls only**. Not-yet-treated
  controls per-(g, t) is a straightforward extension left for a
  follow-up once parity tests exist.
- Covariate adjustment is not implemented. ``triplediff::ddd`` supports
  regression adjustment, IPW and doubly-robust estimation with an
  ``xformla``; the parity above only covers the no-covariate case, which
  is the only one this function can express.
- Inference is cluster bootstrap at the unit level (n_boot draws),
  matching the pattern used in `sp.did_multiplegt`. An analytical
  influence-function variance is not implemented, so standard errors are
  NOT pinned against ``triplediff`` (which reports analytical ones).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core._bootstrap import bootstrap_se as _bootstrap_se
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, NumericalInstability
from . import _core as _dc
from ._ddd_dr import ddd_dr_cell


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
    weight_by: str = "eligible",
    x: Optional[List[str]] = None,
    est_method: str = "dr",
    se: Optional[str] = None,
    control_group: str = "nevertreated",
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
    weight_by : {"eligible", "cohort"}, default "eligible"
        How the per-``(g, t)`` cells are weighted into the overall ATT.

        ``"eligible"`` weights cell ``(g, t)`` by the number of treated
        units in cohort ``g`` that are in the affected subgroup -- the
        units the reported effect is actually about.

        ``"cohort"`` weights by the size of cohort ``g`` counting *both*
        subgroups, which is what ``triplediff::ddd`` + ``agg_ddd(type=
        "simple")`` does (its ``pg`` is ``mean(first_treat == g)`` over
        all units). Pass this to reproduce that package's overall number.

        The two coincide when the affected share is the same in every
        cohort, and drift apart as it varies. Which one is wanted depends
        on the estimand: neither is a bug, and the per-cell estimates are
        identical either way. The default is left on ``"eligible"``
        because changing it would move the number existing callers get
        back; ``model_info["weight_by"]`` records which was used.
    x : list of str, optional
        Base-period covariates. Identification then rests on *conditional*
        DDD parallel trends: the three two-by-two comparisons behind the
        estimand each get a propensity score and an outcome regression,
        combined doubly robustly.
    est_method : {"dr", "ipw", "reg"}, default "dr"
        Doubly robust (consistent if either nuisance is right), inverse
        probability weighting (needs the propensity score), or outcome
        regression (needs the regression). With no covariates all three
        collapse to the same unconditional difference of means.
    control_group : {"nevertreated", "notyettreated"}, default "nevertreated"
        Which units serve as controls. ``"notyettreated"`` adds cohorts
        treated after the comparison period, run against the treated cohort
        one control cohort at a time and combined by minimum distance
        (inverse-covariance weights), which is the structure the reference
        implementation uses.

        .. warning::
           The not-yet-treated path does NOT reproduce ``triplediff`` 0.2.4's
           numbers, on purpose. Its per-control-cohort estimates agree with
           ours exactly, but it writes each cohort's influence function into
           the panel-length vector using a boolean index of the wrong length
           -- R prints "number of items to replace is not a multiple of
           replacement length" on every such call -- so the combined
           influence function picks up units that appear in no comparison,
           and that feeds the weights, the estimate and the standard error.
           Cells where the comparison happens to span the whole panel are
           unaffected and do agree. See
           ``tests/reference_parity/test_ddd_triplediff_parity.py``.
    se : {"analytic", "bootstrap"}, optional
        ``"analytic"`` uses the influence functions -- exact, fast, and what
        ``triplediff`` reports. ``"bootstrap"`` clusters on ``unit`` and is
        the only path that fills in ``model_info['placebo_joint_test']``,
        because that test needs the joint covariance of the placebo arms
        rather than of the DDD.

        Defaults to ``"bootstrap"`` without covariates (unchanged from
        earlier releases, and it keeps the placebo test) and to
        ``"analytic"`` with them, where there is no prior behaviour to
        preserve and the bootstrap would refit both nuisances inside every
        draw. ``model_info['se_method']`` records the resolved choice.

    Returns
    -------
    CausalResult
        ``estimate`` is the aggregate DDD ATT; ``detail`` carries per
        (g, t) decomposition; ``model_info['placebo_joint_test']`` is a
        joint Wald on the unaffected-subgroup DIDs.

    Examples
    --------
    Staggered-adoption panel: two treated cohorts (2012, 2014) plus a
    never-treated control arm. Only the affected subgroup (B=1) carries
    the planted DDD effect of 4.0.

    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> years = list(range(2010, 2016))
    >>> rows = []
    >>> for u in range(120):
    ...     g = rng.choice([0, 2012, 2014])
    ...     unit_fe = rng.normal(0, 1)
    ...     for b in (0, 1):
    ...         for yr in years:
    ...             treated = (g != 0) and (yr >= g)
    ...             effect = 4.0 if (treated and b == 1) else 0.0
    ...             earn = (10 + unit_fe + 0.2 * (yr - 2010) + effect
    ...                     + rng.normal(0, 0.5))
    ...             rows.append({'i': f'{u}_{b}', 'year': yr,
    ...                          'first_treat': g, 'affected': b,
    ...                          'earnings': earn})
    >>> df = pd.DataFrame(rows)
    >>> r = sp.ddd_heterogeneous(df, y='earnings', unit='i', time='year',
    ...                          cohort='first_treat', subgroup='affected',
    ...                          n_boot=100, seed=0)
    >>> bool(np.isfinite(r.estimate))
    True

    References
    ----------
    olden2022triple, callaway2021difference
    """
    df = data.copy()
    for col in (y, unit, time, cohort, subgroup):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in data")

    if weight_by not in {"eligible", "cohort"}:
        raise ValueError(f"weight_by must be 'eligible' or 'cohort', got {weight_by!r}")
    if control_group not in {"nevertreated", "notyettreated"}:
        raise ValueError(
            "control_group must be 'nevertreated' or 'notyettreated', got "
            f"{control_group!r}"
        )
    if est_method not in {"dr", "ipw", "reg"}:
        raise ValueError(f"est_method must be 'dr', 'ipw' or 'reg', got {est_method!r}")
    covars = list(x or [])
    if se is None:
        # Without covariates the cluster bootstrap is what this function has
        # always reported, and it is what carries the placebo joint test, so
        # it stays the default. With covariates there is no prior behaviour
        # to preserve and the bootstrap would refit the propensity score and
        # outcome regression inside every draw, so the analytic
        # influence-function variance -- which is also the reference's --
        # is the sensible default.
        se = "analytic" if covars else "bootstrap"
    if se not in {"analytic", "bootstrap"}:
        raise ValueError(f"se must be 'analytic' or 'bootstrap', got {se!r}")
    for _c in covars:
        if _c not in df.columns:
            raise ValueError(f"Covariate {_c!r} not in data")
    if covars and se == "bootstrap":
        warnings.warn(
            "ddd_heterogeneous: covariates are supplied with se='bootstrap'. "
            "The cluster bootstrap refits the propensity score and outcome "
            "regression inside every draw -- valid, but far slower than the "
            "analytic influence-function variance and no more accurate.",
            UserWarning,
            stacklevel=2,
        )

    if not set(df[subgroup].dropna().unique()) <= {0, 1}:
        raise ValueError(f"subgroup column {subgroup!r} must be binary 0/1")

    rng = np.random.default_rng(seed)

    # Identify cohorts and control set.
    cohort_vals = sorted(df[cohort].dropna().unique())
    treated_cohorts = [g for g in cohort_vals if g != never_value]
    if not treated_cohorts:
        raise ValueError("No treated cohorts found (all units are never-treated).")
    if control_group == "nevertreated" and never_value not in df[cohort].values:
        raise ValueError(
            f"No never-treated units (cohort == {never_value!r}) found. "
            "Pass control_group='notyettreated' to use later-treated cohorts "
            "as controls instead."
        )

    # Compute per-(g, t) DDD decomposition
    def _estimate(work_df: pd.DataFrame) -> Dict[str, Any]:
        units = pd.Index(sorted(work_df[unit].unique()))
        positions = pd.Series(np.arange(len(units)), index=units)
        return _compute_ddd_gt(
            unit_positions=positions,
            df=work_df,
            y=y,
            unit=unit,
            time=time,
            cohort=cohort,
            subgroup=subgroup,
            treated_cohorts=treated_cohorts,
            never_value=never_value,
            weight_by=weight_by,
            covars=covars,
            est_method=est_method,
            control_group=control_group,
        )

    main = _estimate(df)

    if se == "analytic":
        return _finish_analytic(
            main=main,
            df=df,
            unit=unit,
            cohort=cohort,
            subgroup=subgroup,
            never_value=never_value,
            alpha=alpha,
            weight_by=weight_by,
            est_method=est_method,
            control_group=control_group,
            covars=covars,
            treated_cohorts=treated_cohorts,
        )

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
            continue  # replicate stays NaN; bootstrap_se tracks the failure
        boot_overall[b] = best["ddd_overall"]
        plac_vals = [r["did_placebo"] for r in best["cell_estimates"]]
        # Align by (g, t) — assume same order if #cells matches.
        if len(plac_vals) == boot_placebo_gt.shape[1]:
            boot_placebo_gt[b, :] = plac_vals

    se_overall = _bootstrap_se(boot_overall, label="did.ddd_heterogeneous")
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
        method="DDD — heterogeneity-robust group-time decomposition",
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
            "weight_by": weight_by,
            "est_method": est_method,
            "control_group": control_group,
            "se_method": "bootstrap",
            "covariates": list(covars),
        },
    )


def _finish_analytic(
    *,
    main: Dict[str, Any],
    df: pd.DataFrame,
    unit: str,
    cohort: str,
    subgroup: str,
    never_value: Any,
    alpha: float,
    weight_by: str,
    est_method: str,
    control_group: str,
    covars: List[str],
    treated_cohorts: List[Any],
) -> CausalResult:
    """Aggregate the cells through their influence functions.

    Three things this has to get right, each of which would silently
    understate the standard error if skipped:

    1. **Rescaling.** Each cell's influence function is normalised by the
       units in its own two-period subsample. Before the cells can be
       combined they are put on the full-panel scale, ``psi * n / n_cell``.
    2. **Covariance.** The cells share control units, so they are combined
       as functions and squared once at the end, not squared individually
       and summed.
    3. **Estimated weights.** The aggregation weights are cohort shares
       estimated from the same data, and that estimation has its own
       influence. The correction is the Callaway-Sant'Anna ``wif`` term.
    """
    cells = main["cell_estimates"]
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    all_units = pd.Index(sorted(df[unit].unique()))
    n_units = len(all_units)

    # Unit-level cohort and eligibility, taken once per unit.
    first = df.drop_duplicates(subset=[unit]).set_index(unit).reindex(all_units)
    unit_cohort = first[cohort].to_numpy()
    unit_eligible = first[subgroup].to_numpy().astype(float)

    def _pg_indicator(g: Any) -> np.ndarray:
        in_cohort = (unit_cohort == g).astype(float)
        if weight_by == "cohort":
            return in_cohort
        return in_cohort * unit_eligible

    ind = np.column_stack([_pg_indicator(c["cohort"]) for c in cells])
    pg = ind.mean(axis=0)
    total = float(pg.sum())
    if total <= 0:
        w = np.full(len(cells), 1.0 / len(cells))
        wif = None
    else:
        w = pg / total
        # Callaway-Sant'Anna weight influence: d/d(pg) of the normalised
        # weights, applied to the cell estimates.
        if1 = (ind - pg[None, :]) / total
        if2 = np.outer((ind - pg[None, :]).sum(axis=1), pg / total**2)
        wif = if1 - if2

    psi = np.zeros(n_units, dtype=float)
    for weight, cell in zip(w, cells):
        psi += weight * cell["_influence"]
    if wif is not None:
        psi = psi + wif @ np.array([float(c["ddd"]) for c in cells])

    est = float(main["ddd_overall"])
    se_overall = float(np.sqrt(np.mean(psi**2) / n_units))
    if se_overall > 0 and np.isfinite(se_overall):
        z = est / se_overall
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
        ci = (est - z_crit * se_overall, est + z_crit * se_overall)
    else:
        p = np.nan
        ci = (np.nan, np.nan)

    detail_df = pd.DataFrame(
        [{k: v for k, v in c.items() if not k.startswith("_")} for c in cells]
    )
    return CausalResult(
        method="DDD — heterogeneity-robust group-time decomposition",
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
            "n_cells": len(cells),
            # The placebo joint test needs the joint covariance of the
            # unaffected-subgroup arms, which the analytic path does not
            # build: its influence functions are for the DDD, not for that
            # component. se='bootstrap' still reports it.
            "placebo_joint_test": None,
            "n_boot": 0,
            "cluster_var": unit,
            "weight_by": weight_by,
            "est_method": est_method,
            "control_group": control_group,
            "se_method": "analytic",
            "covariates": list(covars),
            "n_units": n_units,
            "influence_function": psi,
        },
    )


def _cell_arrays(
    frame: pd.DataFrame,
    *,
    y: str,
    unit: str,
    time: str,
    cohort: str,
    subgroup: str,
    g: Any,
    t_pre: Any,
    t_post: Any,
    covars: List[str],
):
    """Reduce a two-period frame to one row per unit.

    Returns ``(ids, cell_codes, dy, X)`` or ``None`` when the cell cannot
    support a triple difference. Only units observed in BOTH periods count:
    a unit missing either end has no outcome change.
    """
    pre = frame[frame[time] == t_pre].set_index(unit)
    post = frame[frame[time] == t_post].set_index(unit)
    ids = pre.index.intersection(post.index)
    if len(ids) == 0:
        return None
    pre = pre.loc[ids]
    post = post.loc[ids]

    dy = post[y].to_numpy(dtype=float) - pre[y].to_numpy(dtype=float)
    treated = (pre[cohort].to_numpy() == g).astype(int)
    eligible = pre[subgroup].to_numpy().astype(int)
    # 4 treated+eligible, 3 treated+ineligible, 2 untreated+eligible,
    # 1 untreated+ineligible -- the reference implementation's coding.
    cell = np.where(
        (treated == 1) & (eligible == 1),
        4,
        np.where(
            (treated == 1) & (eligible == 0),
            3,
            np.where(eligible == 1, 2, 1),
        ),
    )
    if len(np.unique(cell)) < 4:
        # A missing corner makes the triple difference unidentified.
        return None

    if covars:
        X = np.column_stack(
            [np.ones(len(ids))] + [pre[c].to_numpy(dtype=float) for c in covars]
        )
        if not np.all(np.isfinite(X)):
            raise DataInsufficient(
                "ddd_heterogeneous: covariates contain non-finite values in "
                f"the base period of cohort {g!r}.",
                diagnostics={"cohort": g, "t_pre": t_pre},
            )
    else:
        X = None
    return np.asarray(ids), cell, dy, X


def _ddd_cell_fit(
    *,
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    cohort: str,
    subgroup: str,
    g: Any,
    t_pre: Any,
    t_post: Any,
    never_value: Any,
    covars: List[str],
    est_method: str,
    control_group: str = "nevertreated",
    unit_positions: Optional[pd.Series] = None,
) -> Optional[Dict[str, Any]]:
    """One ``ATT_DDD(g, t)`` cell, with its influence function on the panel.

    The influence function comes back on the FULL unit vector already scaled
    to the panel (``psi * n / n_cell``), so callers can weight and add the
    cells without knowing how each was assembled. That matters most for the
    not-yet-treated control group, where a cell is not one comparison but a
    minimum-distance combination over several.
    """
    keep = df[df[time].isin([t_pre, t_post])]
    n_panel = (
        len(unit_positions) if unit_positions is not None else keep[unit].nunique()
    )

    if control_group != "notyettreated":
        frame = keep[(keep[cohort] == g) | (keep[cohort] == never_value)]
        arrays = _cell_arrays(
            frame,
            y=y,
            unit=unit,
            time=time,
            cohort=cohort,
            subgroup=subgroup,
            g=g,
            t_pre=t_pre,
            t_post=t_post,
            covars=covars,
        )
        if arrays is None:
            return None
        ids, cell, dy, X = arrays
        fit = ddd_dr_cell(cell=cell, dy=dy, X=X, est_method=est_method)
        psi = np.zeros(n_panel, dtype=float)
        if unit_positions is not None:
            psi[unit_positions.reindex(ids).to_numpy()] = (
                n_panel / len(ids)
            ) * fit.influence
        return {
            "att": float(fit.att),
            "influence": psi,
            "n_cell": int(len(ids)),
            "n_controls": 1,
        }

    # Not-yet-treated: the later-treated cohorts are NOT pooled into one
    # control group. The DDD is run against each control cohort separately
    # and the results combined by minimum distance, weighting by the inverse
    # covariance of their influence functions -- the structure the reference
    # implementation uses. Pooling instead would be a different estimator.
    #
    # ⚠️ The NUMBERS here do not match triplediff 0.2.4 on this path, and
    # deliberately so. Its per-control-cohort estimates agree with ours
    # exactly, but it writes each cohort's influence function into the
    # panel-length vector with a boolean index of the wrong length -- R
    # warns "number of items to replace is not a multiple of replacement
    # length" on every not-yet-treated call. The combined influence function
    # then carries nonzero entries for units in no comparison at all (on the
    # parity fixture, all 150 units of a cohort that is neither treated nor
    # a control for that cell), and that feeds the minimum-distance weights,
    # the combined estimate and the standard error. Reproducing it would
    # mean encoding an upstream indexing defect. The never-treated path is
    # unaffected and is what carries the parity claim.
    max_period = max(t_pre, t_post)
    is_control = (keep[cohort] == never_value) | (
        (keep[cohort] > max_period) & (keep[cohort] != g)
    )
    scope = keep[(keep[cohort] == g) | is_control]
    if scope.empty:
        return None
    size_gt = scope[unit].nunique()
    controls = sorted(
        c
        for c in scope.loc[
            is_control.reindex(scope.index, fill_value=False), cohort
        ].unique()
        if c != g
    )
    if not controls:
        return None

    atts: List[float] = []
    cols: List[np.ndarray] = []
    for ctrl in controls:
        frame = keep[(keep[cohort] == g) | (keep[cohort] == ctrl)]
        arrays = _cell_arrays(
            frame,
            y=y,
            unit=unit,
            time=time,
            cohort=cohort,
            subgroup=subgroup,
            g=g,
            t_pre=t_pre,
            t_post=t_post,
            covars=covars,
        )
        if arrays is None:
            continue
        ids, cell, dy, X = arrays
        fit = ddd_dr_cell(cell=cell, dy=dy, X=X, est_method=est_method)
        col = np.zeros(n_panel, dtype=float)
        if unit_positions is not None:
            col[unit_positions.reindex(ids).to_numpy()] = (
                n_panel / len(ids)
            ) * fit.influence
        atts.append(float(fit.att))
        cols.append(col)

    if not atts:
        return None
    if len(atts) == 1:
        return {
            "att": atts[0],
            "influence": cols[0],
            "n_cell": int(size_gt),
            "n_controls": 1,
        }

    inf_mat = np.column_stack(cols)
    omega = np.cov(inf_mat, rowvar=False)
    try:
        inv_omega = np.linalg.inv(omega)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstability(
            "ddd_heterogeneous(control_group='notyettreated'): the control "
            f"cohorts for (g={g!r}, t={t_post!r}) have a singular influence "
            "covariance, so the minimum-distance weights are not identified. "
            "Use control_group='nevertreated', or drop a control cohort.",
            diagnostics={"cohort": g, "time": t_post, "n_controls": len(atts)},
        ) from exc
    total = float(inv_omega.sum())
    w = inv_omega.sum(axis=0) / total
    return {
        "att": float(np.sum(w * np.asarray(atts)) / float(w.sum())),
        "influence": inf_mat @ w,
        "n_cell": int(size_gt),
        "n_controls": len(atts),
        # Minimum-distance variance -- the reference's formula, and the right
        # one for a GLS-weighted combination. It is NOT the same number as
        # sqrt(mean(psi^2)/n) off the combined influence function, because
        # the covariance is mean-centred and uses n-1.
        "cell_se": float(np.sqrt(1.0 / (n_panel * total))),
    }


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
    weight_by: str = "eligible",
    covars: Optional[List[str]] = None,
    est_method: str = "dr",
    control_group: str = "nevertreated",
    unit_positions: Optional[pd.Series] = None,
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
            n_cohort_units = int(cohort_df[unit].nunique())

            # The doubly-robust cell. With no covariates this reproduces the
            # 2x2x2 difference of means above exactly (the propensity score
            # and the outcome regression are both constants, so they cancel),
            # which is why the two paths agree bit for bit when x is None.
            fit = _ddd_cell_fit(
                unit_positions=unit_positions,
                df=df,
                y=y,
                unit=unit,
                time=time,
                cohort=cohort,
                subgroup=subgroup,
                g=g,
                t_pre=pre_period,
                t_post=t,
                never_value=never_value,
                covars=covars or [],
                est_method=est_method,
                control_group=control_group,
            )
            if fit is None:
                continue

            cells.append(
                {
                    "cohort": g,
                    "time": t,
                    "did_affected": float(did_b1),
                    "did_placebo": float(did_b0),
                    "ddd": float(fit["att"]),
                    # NOT fit.se. The reference reports two different
                    # standard errors for the same influence function: the
                    # standalone two-period call uses sd(psi)/sqrt(n) (the
                    # n-1 denominator), while the multi-period path
                    # recomputes it from the rescaled full-panel vector as
                    # sqrt(mean(psi^2)/n). They differ in the fourth digit.
                    # This is the multi-period path, so use its convention.
                    "se": float(
                        fit.get(
                            "cell_se",
                            np.sqrt(
                                np.mean(fit["influence"] ** 2) / len(fit["influence"])
                            ),
                        )
                    ),
                    "n_treated_affected": int(n_treated_affected),
                    "n_cohort_units": n_cohort_units,
                    "_influence": fit["influence"],
                    "_n_controls": fit["n_controls"],
                }
            )

    if not cells:
        return {
            "ddd_overall": np.nan,
            "cell_estimates": [],
        }

    # Simple CS-style aggregation. "eligible" weights each DDD(g, t) by the
    # treated-affected units contributing to it; "cohort" weights by the whole
    # cohort, which is triplediff::ddd's pg. See the `weight_by` docs.
    weight_key = "n_cohort_units" if weight_by == "cohort" else "n_treated_affected"
    weights = np.array([c[weight_key] for c in cells], dtype=float)
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
