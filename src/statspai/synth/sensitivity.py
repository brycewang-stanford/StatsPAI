"""
Sensitivity and Robustness Analysis for Synthetic Control Methods.

Provides a comprehensive diagnostic toolkit for assessing the reliability
of SCM estimates, going beyond any single existing package:

* **Leave-One-Out** — refit dropping each donor; identify influential units
* **Time Placebos** — "backdating" test with fake treatment times
* **Donor Pool Sensitivity** — bootstrap-style random donor subsets
* **Pre-RMSPE Filtering** — re-rank p-values at multiple quality thresholds
* **Unified Summary** — run all diagnostics in one call

References
----------
Abadie, A., Diamond, A. and Hainmueller, J. (2010).
"Synthetic Control Methods for Comparative Case Studies: Estimating
the Effect of California's Tobacco Control Program."
*Journal of the American Statistical Association*, 105(490), 493-505. [@abadie2010synthetic]

Abadie, A., Diamond, A. and Hainmueller, J. (2015).
"Comparative Politics and the Synthetic Control Method."
*American Journal of Political Science*, 59(2), 495-510. [@abadie2015comparative]

Ferman, B. and Pinto, C. (2021).
"Synthetic Controls with Imperfect Pre-Treatment Fit."
*Quantitative Economics*, 12(4), 1197-1221. [@ferman2021synthetic]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..exceptions import MethodIncompatibility

# ======================================================================
# Internal helpers
# ======================================================================


def _fit_scm_core(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    donor_subset: Optional[List[Any]] = None,
    penalization: float = 0.0,
) -> Dict[str, Any]:
    """
    Lightweight SCM fit returning raw diagnostics (no placebo).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    outcome, unit, time : str
        Column names.
    treated_unit : any
        Treated unit identifier.
    treatment_time : any
        First treatment period.
    donor_subset : list, optional
        Restrict donor pool to these units.
    penalization : float, default 0.0
        Ridge penalty.

    Returns
    -------
    dict
        Keys: att, pre_rmse, weights, gap, Y_synth, Y_treated, times.
    """
    from .scm import SyntheticControl

    if donor_subset is not None:
        keep_units = list(donor_subset) + [treated_unit]
        data = data[data[unit].isin(keep_units)].copy()

    model = SyntheticControl(
        data=data,
        outcome=outcome,
        unit=unit,
        time=time,
        treated_unit=treated_unit,
        treatment_time=treatment_time,
        penalization=penalization,
    )

    Y_pre_treated = model.Y_treated[model.pre_mask]
    Y_pre_donors = model.Y_donors[model.pre_mask]

    # Mirror SyntheticControl.fit's call pattern: forward the predictor
    # matrices the model built and let it pick equal-V vs nested-V.  The
    # solver returns a dict; we want the weight vector.
    solver_out = model._solve_weights(
        Y_pre_treated,
        Y_pre_donors,
        model.X_treated,
        model.X_donors,
        run_nested=model._should_run_nested(),
    )
    weights = solver_out["w"]
    Y_synth = model.Y_donors @ weights
    gap = model.Y_treated - Y_synth

    gap_pre = gap[model.pre_mask]
    gap_post = gap[model.post_mask]

    pre_rmse = float(np.sqrt(np.mean(gap_pre**2)))
    att = float(np.mean(gap_post))
    # Naive standard error of the mean post-period gap, treating the T1
    # post-period gaps as i.i.d. (population SD, ddof=0).  Undefined with a
    # single post-period: np.std of one number is 0, which used to turn
    # into se = 0, z = inf and p = 0 -- a "significant" placebo by
    # construction.
    n_post = len(gap_post)
    se = float(np.std(gap_post)) / np.sqrt(n_post) if n_post >= 2 else float("nan")

    return {
        "att": att,
        "se": se,
        "pre_rmse": pre_rmse,
        "weights": weights,
        "gap": gap,
        "Y_synth": Y_synth,
        "Y_treated": model.Y_treated,
        "times": model.times,
        "donor_units": model.donor_units,
        "pre_mask": model.pre_mask,
        "post_mask": model.post_mask,
    }


def _naive_z_pvalue(att: float, se: float) -> float:
    """Two-sided normal p-value of ``att / se``; NaN when ``se`` is not > 0.

    ``se`` is the i.i.d. post-gap standard error of :func:`_fit_scm_core`.
    A zero or undefined SE carries no information, so the p-value is NaN
    rather than 0 (the old code returned 0, even for ``att == 0``).
    """
    if not np.isfinite(se) or se <= 0.0:
        return float("nan")
    return float(2 * stats.norm.sf(abs(att) / se))


def _warn_dropped(kind: str, failed: List[Dict[str, Any]], n_total: int) -> None:
    if failed:
        warnings.warn(
            f"{len(failed)} of {n_total} {kind} fits failed and were dropped: "
            + "; ".join(f"{f['what']}: {f['error']}" for f in failed[:5]),
            RuntimeWarning,
            stacklevel=3,
        )


# ======================================================================
# 1.  Leave-One-Out Donors
# ======================================================================


@accepts_aliases(id="unit", y="outcome")
def synth_loo(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    penalization: float = 0.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Leave-one-out donor sensitivity for Synthetic Control.

    Re-fits SCM dropping each donor in turn.  Identifies influential
    donors whose removal shifts the ATT substantially.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    outcome : str
        Outcome variable.
    unit : str
        Unit identifier column.
    time : str
        Time column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period.
    penalization : float, default 0.0
        Ridge penalty forwarded to SCM.
    alpha : float, default 0.05
        Accepted for API symmetry with :func:`synth_sensitivity`; it does
        not affect any returned column.

    Returns
    -------
    pd.DataFrame
        Columns: ``dropped_unit``, ``att`` (mean post-period gap),
        ``se``, ``pvalue``, ``pre_rmse`` (pre-period RMSPE). ``se`` is the
        naive i.i.d. standard error of the mean post-period gap
        (``np.std(gap_post) / sqrt(T1)``, ddof 0) and ``pvalue`` its
        two-sided normal p-value. They ignore serial correlation and the
        estimation error of the weights, so they are descriptive, not a
        valid test; both are NaN when ``T1 < 2`` or the SE is zero.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> loo = sp.synth_loo(df, outcome='packspercapita', unit='state',
    ...                    time='year', treated_unit='California',
    ...                    treatment_time=1989)
    >>> loo.sort_values('att')  # doctest: +SKIP
    """
    all_units = data[unit].unique()
    donors = [u for u in all_units if u != treated_unit]

    records: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for drop in donors:
        subset = [d for d in donors if d != drop]
        if len(subset) < 1:
            continue  # pragma: no cover
        try:
            res = _fit_scm_core(
                data,
                outcome,
                unit,
                time,
                treated_unit,
                treatment_time,
                donor_subset=subset,
                penalization=penalization,
            )
            pval = _naive_z_pvalue(res["att"], res["se"])
            records.append(
                {
                    "dropped_unit": drop,
                    "att": res["att"],
                    "se": res["se"],
                    "pvalue": pval,
                    "pre_rmse": res["pre_rmse"],
                }
            )
        except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover
            failed.append({"what": f"drop {drop!r}", "error": repr(exc)})

    _warn_dropped("leave-one-out", failed, len(donors))
    return pd.DataFrame(records)


# ======================================================================
# 2.  Time Placebos
# ======================================================================


@accepts_aliases(id="unit", y="outcome")
def synth_time_placebo(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    penalization: float = 0.0,
    n_placebo_times: Optional[int] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Time-placebo ("backdating") test for Synthetic Control.

    Re-fits SCM using fake treatment times drawn from the pre-treatment
    period.  If the method finds large "effects" where none should exist,
    the original estimate is suspect.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    outcome : str
        Outcome variable.
    unit : str
        Unit identifier column.
    time : str
        Time column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        Real first treatment period.
    penalization : float, default 0.0
        Ridge penalty forwarded to SCM.
    n_placebo_times : int, optional
        Max number of placebo treatment times to try (a subset drawn with
        a fixed seed). Default is all feasible pre-treatment times: every
        pre-period from the third on, so each placebo fit keeps >= 2
        pre-periods. Real post-treatment periods are discarded before the
        placebo fits.
    alpha : float, default 0.05
        Accepted for API symmetry with :func:`synth_sensitivity`; it does
        not affect any returned column.

    Returns
    -------
    pd.DataFrame
        Columns: ``placebo_time``, ``att`` (mean placebo post-period gap
        up to the real treatment time), ``se``, ``pvalue``. ``se`` /
        ``pvalue`` are the naive i.i.d.-gap quantities described in
        :func:`synth_loo` (NaN for the last placebo time, which has a
        single placebo post-period).

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> tp = sp.synth_time_placebo(df, outcome='packspercapita', unit='state',
    ...     time='year', treated_unit='California', treatment_time=1989)
    >>> tp.columns.tolist()
    ['placebo_time', 'att', 'se', 'pvalue']
    """
    # Only use pre-treatment data for the placebo exercise
    pre_data = data[data[time] < treatment_time].copy()
    all_times = np.sort(pre_data[time].unique())

    # Need >= 2 pre-periods and >= 1 "post"-period for each placebo
    candidate_times = all_times[2:]

    if n_placebo_times is not None and n_placebo_times < len(candidate_times):
        rng = np.random.default_rng(42)
        candidate_times = rng.choice(
            candidate_times,
            size=n_placebo_times,
            replace=False,
        )
        candidate_times = np.sort(candidate_times)

    records: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for pt in candidate_times:
        try:
            res = _fit_scm_core(
                pre_data,
                outcome,
                unit,
                time,
                treated_unit,
                pt,
                penalization=penalization,
            )
            pval = _naive_z_pvalue(res["att"], res["se"])
            records.append(
                {
                    "placebo_time": pt,
                    "att": res["att"],
                    "se": res["se"],
                    "pvalue": pval,
                }
            )
        except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover
            failed.append({"what": f"placebo time {pt!r}", "error": repr(exc)})

    _warn_dropped("time-placebo", failed, len(candidate_times))
    return pd.DataFrame(records)


# ======================================================================
# 3.  Donor Pool Sensitivity
# ======================================================================


@accepts_aliases(id="unit", y="outcome")
def synth_donor_sensitivity(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    k: Optional[int] = None,
    n_samples: int = 100,
    penalization: float = 0.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Donor-pool bootstrap sensitivity for Synthetic Control.

    Draws ``n_samples`` random subsets of size ``k`` from the donor
    pool and re-fits SCM for each, producing a distribution of ATT
    estimates.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    outcome : str
        Outcome variable.
    unit : str
        Unit identifier column.
    time : str
        Time column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period.
    k : int, optional
        Donor subset size.  Default is ``floor(J * 0.75)`` where *J*
        is the total number of donors.
    n_samples : int, default 100
        Number of random donor subsets to draw.
    penalization : float, default 0.0
        Ridge penalty forwarded to SCM.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``iteration``, ``donors_used``, ``att``, ``pre_rmse``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> ds = sp.synth_donor_sensitivity(df, outcome='packspercapita',
    ...     unit='state', time='year', treated_unit='California',
    ...     treatment_time=1989, n_samples=50, seed=42)
    >>> ds['att'].describe()  # doctest: +SKIP
    """
    rng = np.random.default_rng(seed)

    all_units = data[unit].unique()
    donors = [u for u in all_units if u != treated_unit]
    J = len(donors)

    if k is None:
        k = max(2, int(np.floor(J * 0.75)))
    k = min(k, J)

    records: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for i in range(n_samples):
        subset = list(rng.choice(donors, size=k, replace=False))
        try:
            res = _fit_scm_core(
                data,
                outcome,
                unit,
                time,
                treated_unit,
                treatment_time,
                donor_subset=subset,
                penalization=penalization,
            )
            records.append(
                {
                    "iteration": i,
                    "donors_used": ",".join(str(d) for d in sorted(subset)),
                    "att": res["att"],
                    "pre_rmse": res["pre_rmse"],
                }
            )
        except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover
            failed.append({"what": f"iteration {i}", "error": repr(exc)})

    _warn_dropped("donor-subset", failed, n_samples)
    return pd.DataFrame(records)


# ======================================================================
# 4.  Pre-RMSPE Robustness
# ======================================================================


@accepts_aliases(id="unit", y="outcome")
def synth_rmspe_filter(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    thresholds: Optional[List[float]] = None,
    penalization: float = 0.0,
    *,
    metric: str = "rmspe",
    placebo_pool: str = "include_treated",
) -> pd.DataFrame:
    """
    Pre-fit-filtered placebo p-values (Abadie et al. 2010).

    Runs the in-space placebo SCM on every donor unit, then recomputes the
    rank p-value of the post/pre RMSPE ratio after discarding placebos
    whose pre-treatment fit is worse than a multiple of the treated unit's.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    outcome : str
        Outcome variable.
    unit : str
        Unit identifier column.
    time : str
        Time column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period.
    thresholds : list of float, optional
        Cut-off multiples of the treated unit's pre-treatment fit; a placebo
        is kept when its fit is ``<= threshold x`` the treated unit's.
        Default ``[1, 2, 5, 10, 20, np.inf]``.
    penalization : float, default 0.0
        Ridge penalty.
    metric : {'rmspe', 'mspe'}, default 'rmspe'
        Scale on which ``thresholds`` act. ``'rmspe'`` (default) compares
        pre-period RMSPEs, the convention of Stata ``synth_runner``'s
        ``pre_limit_mult()``. ``'mspe'`` compares pre-period MSPEs, the
        convention of Abadie et al. (2010), who drop placebos with a pre-period
        MSPE above 20 / 5 / 2 times California's, and of R
        ``SCtools::mspe.test(discard.extreme = TRUE, mspe.limit = ...)``.
        A threshold ``c`` on the MSPE scale equals ``sqrt(c)`` on the RMSPE
        scale.
    placebo_pool : {'include_treated', 'exclude_treated'}, default 'include_treated'
        Donor pool of each placebo fit. ``'include_treated'`` (default)
        uses every other unit, including the actually-treated one, which is
        the exact permutation design under the sharp null of no effect and
        matches ``sp.synth``'s own placebo loop. ``'exclude_treated'`` drops
        the treated unit from every placebo's donor pool, as R
        ``SCtools::generate.placebos`` and Stata ``synth_runner`` do.

    Returns
    -------
    pd.DataFrame
        Columns: ``threshold``, ``n_placebos``, ``pvalue``,
        ``treated_pre_rmspe``. The p-value is
        ``(1 + #{kept placebos with ratio >= treated ratio}) / (n_kept + 1)``
        (treated unit ranked with its placebos, as ``SCtools::mspe.test``;
        Stata ``synth_runner`` instead reports ``#{...} / n_kept``).
        ``df.attrs['placebos']`` holds each placebo's ``unit``,
        ``pre_rmspe`` and post/pre RMSPE ``ratio``; ``df.attrs['treated_ratio']``
        the treated unit's ratio.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> rp = sp.synth_rmspe_filter(df, outcome='packspercapita', unit='state',
    ...     time='year', treated_unit='California', treatment_time=1989)
    >>> rp.columns.tolist()
    ['threshold', 'n_placebos', 'pvalue', 'treated_pre_rmspe']
    """
    if thresholds is None:
        thresholds = [1.0, 2.0, 5.0, 10.0, 20.0, np.inf]
    if metric not in ("rmspe", "mspe"):
        raise MethodIncompatibility(f"metric must be 'rmspe' or 'mspe', got {metric!r}")
    if placebo_pool not in ("include_treated", "exclude_treated"):
        raise MethodIncompatibility(
            "placebo_pool must be 'include_treated' or 'exclude_treated', "
            f"got {placebo_pool!r}"
        )

    all_units = data[unit].unique()
    donors = [u for u in all_units if u != treated_unit]

    # --- Treated unit ---
    treated_res = _fit_scm_core(
        data,
        outcome,
        unit,
        time,
        treated_unit,
        treatment_time,
        penalization=penalization,
    )
    treated_pre_rmspe = treated_res["pre_rmse"]
    gap_post_treated = treated_res["gap"][treated_res["post_mask"]]
    post_mspe_treated = float(np.mean(gap_post_treated**2))
    ratio_treated = (
        np.sqrt(post_mspe_treated) / treated_pre_rmspe
        if treated_pre_rmspe > 1e-10
        else np.inf
    )

    # --- Placebo units ---
    placebo_info: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for d in donors:
        other_donors = [u for u in donors if u != d]
        pool = (
            other_donors + [treated_unit]
            if placebo_pool == "include_treated"
            else other_donors
        )
        if len(pool) < 1:
            continue  # pragma: no cover
        try:
            pres = _fit_scm_core(
                data[data[unit].isin(pool + [d])],
                outcome,
                unit,
                time,
                d,
                treatment_time,
                penalization=penalization,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover
            failed.append({"what": f"placebo {d!r}", "error": repr(exc)})
            continue
        gap_post_p = pres["gap"][pres["post_mask"]]
        post_mspe_p = float(np.mean(gap_post_p**2))
        pre_rmspe_p = pres["pre_rmse"]
        ratio_p = np.sqrt(post_mspe_p) / pre_rmspe_p if pre_rmspe_p > 1e-10 else np.inf
        placebo_info.append({"unit": d, "pre_rmspe": pre_rmspe_p, "ratio": ratio_p})
    _warn_dropped("in-space placebo", failed, len(donors))

    # --- Filter at each threshold ---
    records: List[Dict[str, Any]] = []
    for thr in thresholds:
        if not np.isfinite(thr):
            kept = list(placebo_info)
        elif metric == "mspe":
            cutoff = thr * treated_pre_rmspe**2
            kept = [p for p in placebo_info if p["pre_rmspe"] ** 2 <= cutoff]
        else:
            cutoff = thr * treated_pre_rmspe
            kept = [p for p in placebo_info if p["pre_rmspe"] <= cutoff]
        n_kept = len(kept)
        if n_kept == 0:
            pval = np.nan
        else:
            n_extreme = sum(1 for p in kept if p["ratio"] >= ratio_treated)
            # Include treated unit in the ranking
            pval = (n_extreme + 1) / (n_kept + 1)
        records.append(
            {
                "threshold": thr,
                "n_placebos": n_kept,
                "pvalue": pval,
                "treated_pre_rmspe": treated_pre_rmspe,
            }
        )

    out = pd.DataFrame(records)
    out.attrs["placebos"] = pd.DataFrame(placebo_info)
    out.attrs["treated_ratio"] = float(ratio_treated)
    out.attrs["metric"] = metric
    out.attrs["placebo_pool"] = placebo_pool
    return out


# ======================================================================
# 5.  Comprehensive Summary
# ======================================================================


@accepts_aliases(id="unit", y="outcome")
def synth_sensitivity(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    penalization: float = 0.0,
    n_donor_samples: int = 100,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Run all SCM sensitivity diagnostics in a single call.

    Combines leave-one-out, time placebos, donor pool bootstrap,
    and pre-RMSPE filtering into one bundled report.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    outcome : str
        Outcome variable.
    unit : str
        Unit identifier column.
    time : str
        Time column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period.
    penalization : float, default 0.0
        Ridge penalty.
    n_donor_samples : int, default 100
        Number of random donor subsets for donor sensitivity.
    seed : int, optional
        Random seed.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys:

        * ``'loo'`` — leave-one-out DataFrame
        * ``'time_placebo'`` — time placebo DataFrame
        * ``'donor_sensitivity'`` — donor bootstrap DataFrame
        * ``'rmspe_filter'`` — RMSPE-filtered p-values DataFrame
        * ``'summary'`` — formatted string summary

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> sens = sp.synth_sensitivity(df, outcome='packspercapita',
    ...     unit='state', time='year', treated_unit='California',
    ...     treatment_time=1989, n_donor_samples=50, seed=42)
    >>> sorted(sens.keys())
    ['donor_sensitivity', 'loo', 'rmspe_filter', 'summary', 'time_placebo']
    >>> print(sens['summary'])  # doctest: +SKIP
    """
    # --- Leave-one-out ---
    loo_df = synth_loo(
        data,
        outcome,
        unit,
        time,
        treated_unit,
        treatment_time,
        penalization=penalization,
        alpha=alpha,
    )

    # --- Time placebos ---
    tp_df = synth_time_placebo(
        data,
        outcome,
        unit,
        time,
        treated_unit,
        treatment_time,
        penalization=penalization,
        alpha=alpha,
    )

    # --- Donor sensitivity ---
    ds_df = synth_donor_sensitivity(
        data,
        outcome,
        unit,
        time,
        treated_unit,
        treatment_time,
        n_samples=n_donor_samples,
        penalization=penalization,
        seed=seed,
    )

    # --- RMSPE filter ---
    rp_df = synth_rmspe_filter(
        data,
        outcome,
        unit,
        time,
        treated_unit,
        treatment_time,
        penalization=penalization,
    )

    # --- Build summary ---
    lines: List[str] = []
    lines.append("=" * 62)
    lines.append("  SCM Sensitivity & Robustness Summary")
    lines.append("=" * 62)
    lines.append(f"  Treated unit:  {treated_unit}")
    lines.append(f"  Treatment time: {treatment_time}")

    # Baseline ATT
    base = _fit_scm_core(
        data,
        outcome,
        unit,
        time,
        treated_unit,
        treatment_time,
        penalization=penalization,
    )
    lines.append(f"  Baseline ATT:  {base['att']:.4f}")
    lines.append(f"  Pre-RMSE:      {base['pre_rmse']:.6f}")
    lines.append("")

    # LOO
    if len(loo_df) > 0:
        lines.append("--- Leave-One-Out ---")
        lines.append(
            f"  ATT range: [{loo_df['att'].min():.4f}, " f"{loo_df['att'].max():.4f}]"
        )
        lines.append(f"  ATT mean:  {loo_df['att'].mean():.4f}")
        most_influential = loo_df.loc[(loo_df["att"] - base["att"]).abs().idxmax()]
        lines.append(
            f"  Most influential donor: {most_influential['dropped_unit']} "
            f"(ATT = {most_influential['att']:.4f})"
        )
        lines.append("")

    # Time placebo
    if len(tp_df) > 0:
        n_sig = (tp_df["pvalue"] < alpha).sum()
        n_def = int(tp_df["pvalue"].notna().sum())
        lines.append("--- Time Placebos ---")
        lines.append(f"  Placebo times tested: {len(tp_df)}")
        lines.append(
            f"  Naive iid-gap z p < {alpha:.0%}: {n_sig} / {n_def} "
            "(descriptive; ignores serial correlation)"
        )
        lines.append(f"  Max |placebo ATT|: {tp_df['att'].abs().max():.4f}")
        lines.append("")

    # Donor sensitivity
    if len(ds_df) > 0:
        lines.append("--- Donor Pool Sensitivity ---")
        lines.append(f"  Iterations: {len(ds_df)}")
        lines.append(f"  ATT mean:   {ds_df['att'].mean():.4f}")
        lines.append(f"  ATT std:    {ds_df['att'].std():.4f}")
        lines.append(
            f"  ATT 95% CI: [{ds_df['att'].quantile(0.025):.4f}, "
            f"{ds_df['att'].quantile(0.975):.4f}]"
        )
        lines.append("")

    # RMSPE
    if len(rp_df) > 0:
        lines.append("--- Pre-RMSPE Filtered P-values ---")
        for _, row in rp_df.iterrows():
            thr_label = (
                f"{row['threshold']:.0f}x" if np.isfinite(row["threshold"]) else "all"
            )
            lines.append(
                f"  {thr_label:>5s}: p = {row['pvalue']:.3f}  "
                f"(n = {int(row['n_placebos'])})"
            )
        lines.append("")

    lines.append("=" * 62)
    summary_str = "\n".join(lines)

    return {
        "loo": loo_df,
        "time_placebo": tp_df,
        "donor_sensitivity": ds_df,
        "rmspe_filter": rp_df,
        "summary": summary_str,
    }


# ======================================================================
# 6.  Sensitivity Plot
# ======================================================================


def synth_sensitivity_plot(
    sensitivity_result: Dict[str, Any],
    figsize: Tuple[float, float] = (14, 10),
    title: Optional[str] = None,
) -> Any:
    """
    Multi-panel sensitivity diagnostic plot.

    Parameters
    ----------
    sensitivity_result : dict
        Output from :func:`synth_sensitivity`.
    figsize : tuple, default (14, 10)
        Figure size in inches.
    title : str, optional
        Super-title for the figure.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> sens = sp.synth_sensitivity(df, outcome='packspercapita',
    ...     unit='state', time='year', treated_unit='California',
    ...     treatment_time=1989, n_donor_samples=50, seed=42)
    >>> fig = sp.synth_sensitivity_plot(sens)
    >>> fig.savefig('synth_sensitivity.png', dpi=150)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    loo_df = sensitivity_result["loo"]
    tp_df = sensitivity_result["time_placebo"]
    ds_df = sensitivity_result["donor_sensitivity"]
    rp_df = sensitivity_result["rmspe_filter"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ------------------------------------------------------------------
    # Panel 1: LOO ATT distribution
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    if len(loo_df) > 0:
        sorted_loo = loo_df.sort_values("att")
        y_pos = np.arange(len(sorted_loo))
        ax.barh(y_pos, sorted_loo["att"].values, color="#4C72B0", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_loo["dropped_unit"].values, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("ATT (donor dropped)")
    ax.set_title("Leave-One-Out")

    # ------------------------------------------------------------------
    # Panel 2: Time placebo ATTs
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    if len(tp_df) > 0:
        ax.bar(
            tp_df["placebo_time"].astype(str),
            tp_df["att"].values,
            color="#DD8452",
            alpha=0.8,
        )
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.set_xlabel("Placebo treatment time")
    ax.set_ylabel("Placebo ATT")
    ax.set_title("Time Placebos")

    # ------------------------------------------------------------------
    # Panel 3: Donor pool ATT distribution
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    if len(ds_df) > 0:
        ax.hist(
            ds_df["att"].values, bins=30, color="#55A868", alpha=0.8, edgecolor="white"
        )
        ax.axvline(
            ds_df["att"].mean(),
            color="red",
            linewidth=1.5,
            linestyle="--",
            label="Mean",
        )
        ax.axvline(
            ds_df["att"].quantile(0.025),
            color="orange",
            linewidth=1,
            linestyle=":",
            label="2.5%",
        )
        ax.axvline(
            ds_df["att"].quantile(0.975),
            color="orange",
            linewidth=1,
            linestyle=":",
            label="97.5%",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("ATT")
    ax.set_ylabel("Frequency")
    ax.set_title("Donor Pool Sensitivity")

    # ------------------------------------------------------------------
    # Panel 4: RMSPE-filtered p-values
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    if len(rp_df) > 0:
        labels = []
        for _, row in rp_df.iterrows():
            labels.append(
                f"{row['threshold']:.0f}x" if np.isfinite(row["threshold"]) else "All"
            )
        ax.plot(
            labels,
            rp_df["pvalue"].values,
            "o-",
            color="#C44E52",
            linewidth=2,
            markersize=8,
        )
        ax.axhline(0.05, color="gray", linewidth=0.8, linestyle="--", label="p = 0.05")
        ax.axhline(0.10, color="gray", linewidth=0.8, linestyle=":", label="p = 0.10")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, max(1.05, rp_df["pvalue"].max() + 0.05))
    ax.set_xlabel("Pre-RMSPE threshold (x treated)")
    ax.set_ylabel("P-value")
    ax.set_title("RMSPE-Filtered P-values")

    # ------------------------------------------------------------------
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    fig.tight_layout()
    return fig
