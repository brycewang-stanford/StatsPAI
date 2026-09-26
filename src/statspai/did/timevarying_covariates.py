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

Estimators, comparison groups and standard errors
-------------------------------------------------
``est_method`` picks the cell estimator: outcome regression (the default,
and what the reference implementations use), stabilised IPW, or the doubly
robust combination (Sant'Anna & Zhao 2020).  All three are the functions
``sp.callaway_santanna`` dispatches to, applied to the long difference with
the frozen covariates -- not a second implementation -- so a ``'dr'`` cell
here is consistent if *either* the outcome model or the propensity score is
right.  ``control_group='notyettreated'`` widens the comparison set from
never-treated units to units not yet treated at either period of the cell.

``vce`` picks the standard error.  ``'bootstrap'`` (the default) is the
unit-level cluster bootstrap this module has always used.  ``'analytic'``
is the influence-function plug-in: the cell influence functions of the
chosen estimator, aggregated with the same weights as the point estimate.
``'multiplier'`` is the multiplier bootstrap on those influence functions,
which is the family ``ptetools`` reports.

``covariate_pretest=True`` adds the two placebo statistics of the 2026
paper's section 5.2, computed on the covariate itself: over the pre-treatment
periods (where a non-zero value undermines the design) and over the post
periods (where a non-zero value is the evidence that the covariate is a
genuine bad control).

Scope & caveats
---------------
- This is approach 1 of the bad-controls paper: condition on the
  *pre-treatment* value of the bad control.  The nested-expectation
  estimator under covariate unconfoundedness with additional confounders
  (their Theorem 2) is not implemented.
- The analytic variance conditions on the realised cell counts; the two
  bootstrap paths do not.

Every cell estimator and both comparison groups are pinned against
``did::att_gt(xformla = ..., base_period = "universal")``, whose panel 2x2
takes its covariates from the base period ``g - 1`` -- the same frozen
covariate -- in
``tests/reference_parity/test_did_synth_didvar_parity.py``: ATTs and
per-cell standard errors agree to 3.6e-15 (``reg`` with not-yet-treated
comparisons), 1.0e-11 (``dr``) and 3.8e-11 (``ipw``), the last two floored
by the logit solve's convergence tolerance.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core._bootstrap import bootstrap_se as _bootstrap_se
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _core as _dc


@accepts_aliases(id="unit")
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
    est_method: str = "reg",
    control_group: str = "nevertreated",
    vce: str = "bootstrap",
    covariate_pretest: bool = False,
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
    est_method : {"reg", "ipw", "dr"}, default "reg"
        Cell estimator. ``"reg"`` is outcome regression on the comparison
        units (the reference implementations' choice, and the default so the
        shipped numbers do not move); ``"ipw"`` is the stabilised
        (Hajek-normalised) inverse-probability weighting estimator;
        ``"dr"`` is the doubly robust combination, consistent if either the
        outcome model or the propensity score is correctly specified.

        .. versionadded:: 1.31.0
    control_group : {"nevertreated", "notyettreated"}, default "nevertreated"
        Comparison units. ``"notyettreated"`` adds cohorts that have not
        switched by either period of the cell (``G > max(t, g - 1)``), the
        rule R ``did`` uses.

        .. versionadded:: 1.31.0
    vce : {"bootstrap", "analytic", "multiplier"}, default "bootstrap"
        Standard error. ``"analytic"`` is the influence-function plug-in and
        needs no resampling; ``"multiplier"`` is the multiplier bootstrap on
        the same influence functions.

        .. versionadded:: 1.31.0
    covariate_pretest : bool, default False
        Also run the comparison with each covariate as the outcome, before
        and after treatment, and report it in
        ``model_info['covariate_pretest']``.

        .. versionadded:: 1.31.0
    n_boot : int, default 500
        Bootstrap replications; ignored when ``vce='analytic'``.
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
    caetano2026difference (the current version of the paper; supersedes
    caetano2022difference, whose fourth author is Rodrigues rather than
    Sant'Anna)
    """
    df = data.copy()
    for col in [y, unit, time, cohort] + list(covariates):
        if col not in df.columns:
            raise MethodIncompatibility(
                f"Column {col!r} not in data.",
                diagnostics={"columns": list(df.columns)[:40]},
            )
    if aggregation not in ("group", "simple"):
        raise MethodIncompatibility(
            f"aggregation must be 'group' or 'simple'; got {aggregation!r}"
        )
    if est_method not in _EST_METHODS:
        raise MethodIncompatibility(
            f"est_method must be one of {_EST_METHODS}; got {est_method!r}",
            diagnostics={"est_method": est_method},
        )
    if control_group not in _CONTROL_GROUPS:
        raise MethodIncompatibility(
            f"control_group must be one of {_CONTROL_GROUPS}; got "
            f"{control_group!r}",
            diagnostics={"control_group": control_group},
        )
    if vce not in _VCE:
        raise MethodIncompatibility(
            f"vce must be one of {_VCE}; got {vce!r}",
            diagnostics={"vce": vce},
        )

    rng = np.random.default_rng(seed)
    cohort_vals = sorted(df[cohort].dropna().unique())
    treated_cohorts = [g for g in cohort_vals if g != never_value]
    if not treated_cohorts:
        raise DataInsufficient(
            "No treated cohorts found.",
            diagnostics={"cohorts": [str(c) for c in cohort_vals[:10]]},
        )
    if control_group == "nevertreated" and never_value not in df[cohort].values:
        raise DataInsufficient(
            f"No never-treated units (cohort == {never_value!r}). Pass "
            "control_group='notyettreated' to compare against units that "
            "have not switched yet.",
            diagnostics={"never_value": never_value},
        )

    unit_index = pd.Index(sorted(df[unit].unique()), name=unit)
    kw = dict(
        y=y,
        unit=unit,
        time=time,
        cohort=cohort,
        covariates=list(covariates),
        treated_cohorts=treated_cohorts,
        never_value=never_value,
        baseline_offset=baseline_offset,
        est_method=est_method,
        control_group=control_group,
    )
    main = _compute_att_gt(df, unit_index=unit_index, **kw)
    key = f"att_{aggregation}"
    est = float(main[key])
    if not np.isfinite(est):
        raise DataInsufficient(
            "No (cohort, period) cell had both a usable baseline period and "
            "enough comparison units to estimate an ATT.",
            diagnostics={"n_cells": len(main["cell_estimates"])},
        )

    n_units = len(unit_index)
    inf = main["influence"]
    inf_agg: Optional[np.ndarray] = (
        None if inf is None else np.asarray(inf @ main["weights"][aggregation])
    )

    if vce == "bootstrap":
        boot_overall = np.full(n_boot, np.nan)
        for b in range(n_boot):
            bdf = _dc.cluster_bootstrap_draw(
                df, cluster_col=unit, rng=rng, relabel_cols=[unit]
            )
            try:
                best = _compute_att_gt(bdf, unit_index=None, **kw)
            except np.linalg.LinAlgError:
                continue  # replicate stays NaN; bootstrap_se tracks the failure
            boot_overall[b] = best[key]
        se = _bootstrap_se(boot_overall, label="did.timevarying_covariates")
    else:
        if inf_agg is None:
            raise DataInsufficient(
                f"vce={vce!r} needs the cells' influence functions and no "
                "cell produced one. Use vce='bootstrap', or check that the "
                "cohorts have usable baseline periods.",
                diagnostics={"n_cells": len(main["cell_estimates"])},
            )
        if vce == "analytic":
            se = float(_dc.influence_function_se(inf_agg))
        else:  # multiplier
            se_arr, _ = _dc.multiplier_bootstrap(
                inf_agg[:, None],
                n_units,
                alpha,
                n_boot,
                random_state=seed,
            )
            se = float(se_arr[0])

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    if se > 0 and np.isfinite(se):
        z = est / se
        p = float(2 * stats.norm.sf(abs(z)))
        ci = (est - z_crit * se, est + z_crit * se)
    else:
        p = np.nan
        ci = (np.nan, np.nan)

    detail_df = pd.DataFrame(main["cell_estimates"])
    pretest = (
        _covariate_pretest(df, alpha=alpha, unit_index=unit_index, **kw)
        if covariate_pretest
        else None
    )

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
            "est_method": est_method,
            "control_group": control_group,
            "vce": vce,
            "estimator": f"{est_method} on {control_group} comparison units",
            "n_cells": len(main["cell_estimates"]),
            "n_boot": n_boot if vce != "analytic" else 0,
            "cluster_var": unit,
            "covariate_pretest": pretest,
        },
    )


_EST_METHODS = ("reg", "ipw", "dr")
_CONTROL_GROUPS = ("nevertreated", "notyettreated")
_VCE = ("bootstrap", "analytic", "multiplier")


def _cell_estimator(
    est_method: str,
) -> Callable[
    [np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray],
    Tuple[float, float, np.ndarray],
]:
    """The (att, se, influence) estimator for one 2x2 cell.

    These are the same three functions ``sp.callaway_santanna`` dispatches
    to, so a cell here is that estimator applied to the long difference
    with the frozen covariates -- not a second implementation.
    """
    from .callaway_santanna import _DEFAULT_PSCORE_TRIM, _dr_att, _ipw_att, _reg_att

    if est_method == "dr":
        return lambda dy, d, x, w: _dr_att(dy, d, x, _DEFAULT_PSCORE_TRIM, w)
    if est_method == "ipw":
        return lambda dy, d, x, w: _ipw_att(dy, d, x, _DEFAULT_PSCORE_TRIM, w)
    return lambda dy, d, x, w: _reg_att(dy, d, x, w)


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
    est_method: str = "reg",
    control_group: str = "nevertreated",
    unit_index: Optional[pd.Index] = None,
    horizons: Optional[int] = None,
) -> Dict[str, Any]:
    """ATT(g, t) on covariates frozen at ``g + baseline_offset``.

    ``horizons`` caps how many post periods each cohort contributes, which
    is what the covariate pre-test uses to keep its placebo to a single
    period.

    When ``unit_index`` is given, each cell's influence function is embedded
    in that unit universe (zero for units the cell does not use, rescaled by
    ``n_total / n_cell`` exactly as the Callaway--Sant'Anna path does) so the
    columns can be aggregated across cells.
    """
    estimate_cell = _cell_estimator(est_method)
    cells: List[Dict[str, Any]] = []
    inf_cols: List[np.ndarray] = []
    times = set(df[time].unique())
    want_inf = unit_index is not None
    if unit_index is not None:
        pos = pd.Series(np.arange(len(unit_index)), index=unit_index)
        n_total = len(unit_index)

    for g in treated_cohorts:
        pre_t = g - 1
        cov_t = g + baseline_offset
        if pre_t not in times or cov_t not in times:
            continue
        window = sorted(t for t in times if t >= g)
        if horizons is not None:
            window = window[:horizons]

        for t in window:
            if control_group == "nevertreated":
                comparison = df[cohort] == never_value
            else:
                # Not yet treated at either period of the 2x2 comparison,
                # the rule R ``did`` uses: G > max(t, base).
                comparison = (df[cohort] == never_value) | (df[cohort] > max(t, pre_t))
            pop = df[(df[cohort] == g) | comparison]
            base = pop[pop[time] == pre_t][[unit, y]].rename(columns={y: "_y_pre"})
            xs = pop[pop[time] == cov_t][[unit] + covariates]
            xs = xs.rename(columns={c: f"_x_{j}" for j, c in enumerate(covariates)})
            base = base.merge(xs, on=unit, how="inner")
            x_cols = [f"_x_{j}" for j in range(len(covariates))]

            post_df = pop[pop[time] == t][[unit, y, cohort]]
            merged = post_df.merge(base, on=unit, how="inner")
            if merged.empty:
                continue
            dy = (merged[y] - merged["_y_pre"]).to_numpy(dtype=float)
            xmat = merged[x_cols].to_numpy(dtype=float)
            treated = (merged[cohort] == g).to_numpy()
            valid = np.isfinite(dy) & np.all(np.isfinite(xmat), axis=1)
            tr = treated & valid
            co = ~treated & valid
            n_treated, n_control = int(tr.sum()), int(co.sum())
            if n_treated < 1 or n_control <= xmat.shape[1] + 1:
                continue

            keep = valid
            dy_k = dy[keep]
            d_k = treated[keep].astype(float)
            x_k = xmat[keep]
            var = np.var(x_k, axis=0)
            x_k = x_k[:, var >= 1e-12] if np.any(var >= 1e-12) else None
            w_k = np.ones(len(dy_k), dtype=float)
            att_gt, se_gt, inf_local = estimate_cell(dy_k, d_k, x_k, w_k)

            cells.append(
                {
                    "cohort": g,
                    "time": t,
                    "att_gt": float(att_gt),
                    "se_gt": float(se_gt),
                    "n_treated": n_treated,
                    "n_control": n_control,
                }
            )
            if want_inf:
                col = np.zeros(n_total)
                idx = pos.reindex(merged.loc[keep, unit]).to_numpy()
                col[idx.astype(int)] = inf_local * (n_total / len(dy_k))
                inf_cols.append(col)

    if not cells:
        return {
            "att_group": np.nan,
            "att_simple": np.nan,
            "cell_estimates": [],
            "influence": None,
            "weights": {},
        }

    cell_df = pd.DataFrame(cells)
    att = cell_df["att_gt"].to_numpy(dtype=float)

    # Simple: every post cell weighted by its treated count.
    w_simple = cell_df["n_treated"].to_numpy(dtype=float)
    w_simple = w_simple / w_simple.sum()

    # Group: equal weight within a cohort's post cells, cohort size across.
    by_g = cell_df.groupby("cohort")["n_treated"].max()
    counts = cell_df.groupby("cohort")["att_gt"].size()
    w_group = np.array([by_g[c] / counts[c] for c in cell_df["cohort"]], dtype=float)
    w_group = w_group / w_group.sum()

    out = {
        "att_group": float(att @ w_group),
        "att_simple": float(att @ w_simple),
        "cell_estimates": cells,
        "influence": np.column_stack(inf_cols) if inf_cols else None,
        "weights": {"group": w_group, "simple": w_simple},
    }
    return out


def _covariate_pretest(
    df: pd.DataFrame,
    *,
    alpha: float,
    unit_index: pd.Index,
    y: str,
    unit: str,
    time: str,
    cohort: str,
    covariates: List[str],
    treated_cohorts: List[Any],
    never_value: Any,
    baseline_offset: int,
    est_method: str,
    control_group: str,
) -> Dict[str, Dict[str, float]]:
    """Two statistics per covariate, computed on the covariate itself.

    The identifying assumptions are not testable, but two of their
    implications are visible in the data (Caetano, Callaway, Payne and
    Sant'Anna, section 5.2), and both are the module's own comparison run
    with the covariate in place of the outcome:

    ``post`` runs it over the post periods.  A non-zero value is *not* a
    problem -- it is the evidence that the covariate is a genuine bad
    control, i.e. that the treatment moves it, and so that conditioning on
    its contemporaneous value would have been the wrong adjustment.

    ``pre`` is the placebo: every cohort is moved one period earlier, so the
    comparison runs from ``g - 2`` to ``g - 1`` with the covariates frozen at
    ``g - 2``, entirely before treatment.  Under simple covariate
    unconfoundedness this is zero; a non-zero value says the cohort's
    covariate was already moving differently conditional on its own baseline,
    which is what the frozen-baseline adjustment assumes away.  It is a
    forward comparison shifted earlier, not a backward difference from the
    base period -- selection on the level of the covariate makes the latter
    non-zero whether or not the assumption holds.
    """
    shifted = df.assign(
        **{
            cohort: np.where(
                df[cohort] != never_value,
                pd.to_numeric(df[cohort], errors="coerce") - 1,
                never_value,
            )
        }
    )
    shifted_cohorts = [g - 1 for g in treated_cohorts]

    out: Dict[str, Dict[str, float]] = {}
    for cov in covariates:
        entry: Dict[str, float] = {}
        # The conditioning set is the same frozen one the ATT uses, the
        # covariate's own baseline included: Assumption 4 is X_t(0)
        # independent of D given X_{t-1} and Z, so the statistic has to hold
        # the baseline fixed or mean reversion in X reads as movement.
        common = dict(
            y=cov,
            unit=unit,
            time=time,
            cohort=cohort,
            covariates=list(covariates),
            never_value=never_value,
            baseline_offset=baseline_offset,
            est_method=est_method,
            control_group=control_group,
            unit_index=unit_index,
        )
        for window, frame, cohorts, cap in (
            ("post", df, treated_cohorts, None),
            ("pre", shifted, shifted_cohorts, 1),
        ):
            res = _compute_att_gt(
                frame, treated_cohorts=cohorts, horizons=cap, **common
            )
            att = res["att_group"]
            if not np.isfinite(att) or res["influence"] is None:
                entry[window] = float("nan")
                entry[f"{window}_se"] = float("nan")
                entry[f"{window}_pvalue"] = float("nan")
                continue
            psi = res["influence"] @ res["weights"]["group"]
            se = float(_dc.influence_function_se(psi))
            entry[window] = float(att)
            entry[f"{window}_se"] = se
            entry[f"{window}_pvalue"] = (
                float(2 * stats.norm.sf(abs(att / se))) if se > 0 else float("nan")
            )
        out[cov] = entry
    return out
