"""Shared input checks for the staggered-adoption DiD estimators.

Every staggered estimator reads a long panel keyed by (unit, period) with a
cohort column holding the first treated period. Before 2026-10 each one
checked a different subset of what can go wrong with that layout, so the
same data gave a refusal from one estimator and a silent number from the
next (2026-10-19 audit):

* a cohort value that changes within a unit was used row by row by
  ``sun_abraham``, from the first row by ``callaway_santanna`` and not at
  all by ``did_imputation``, with three different estimates and no
  warning. R ``did`` stops on it.
* repeated (unit, period) rows were refused by ``callaway_santanna`` and
  silently used by the regression estimators.
* a cohort treated in or before the first observed period (no untreated
  row of its own) gave ``sun_abraham`` zero standard errors.
* one treated cluster produced a warning from ``callaway_santanna`` only.

The helpers here state the rule once. What each estimator then *does* with
an always-treated cohort differs on purpose (CS drops its cells, BJS
refuses unless ``autosample=True`` as Stata does, the regression
estimators drop the units), so :func:`always_treated_units` only finds
them and the caller applies its policy.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
    StatsPAIWarning,
)

__all__ = [
    "FEW_TREATED_CLUSTERS",
    "normalized_cohort",
    "require_constant_cohort",
    "duplicate_unit_period_rows",
    "always_treated_units",
    "drop_always_treated",
    "warn_few_treated_clusters",
    "inference_note",
]

#: Below this many treated clusters a cluster-robust variance estimates the
#: treated side's contribution from that many draws and over-rejects however
#: many control clusters there are (``sp.did_few_treated``). The same cutoff
#: as ``callaway_santanna._FEW_TREATED_UNITS`` and ``sp.audit``.
FEW_TREATED_CLUSTERS = 10


class RepeatedUnitPeriodWarning(StatsPAIWarning):
    """Several rows share a (unit, period) pair in a staggered DiD input."""

    code = "repeated_unit_period"


def normalized_cohort(series: pd.Series) -> pd.Series:
    """Cohort column with the never-treated codes (NaN, +/-inf, 0) set to 0.

    Non-numeric values are left to the estimator's own parsing; this helper
    only exists so that "NaN in one row and 0 in another" is not read as a
    cohort that varies within the unit.
    """
    num = pd.to_numeric(series, errors="coerce")
    return num.fillna(0).replace([np.inf, -np.inf], 0)


def require_constant_cohort(
    data: pd.DataFrame,
    *,
    unit: str,
    cohort: str,
    function: str,
) -> None:
    """Refuse a cohort (first-treatment period) that changes within a unit.

    A unit has one first treatment date. When the column disagrees across a
    unit's rows there is no defined answer: estimators that read one row per
    unit pick an arbitrary one, and estimators that read every row put the
    unit in several cohorts at once. R ``did`` stops with "The value of
    gname must be the same across all periods for each particular unit".
    """
    g = normalized_cohort(data[cohort])
    nuniq = g.groupby(data[unit].to_numpy()).nunique(dropna=False)
    bad = nuniq[nuniq > 1]
    if bad.empty:
        return
    examples = bad.index.tolist()[:5]
    raise MethodIncompatibility(
        f"{function}: the cohort column {cohort!r} takes more than one value "
        f"within {len(bad)} unit(s) (e.g. {examples}). A unit has a single "
        "first treatment period; with several, the unit's cohort is "
        "undefined and each estimator would resolve it differently.",
        recovery_hint=(
            f"Recompute {cohort!r} as the first period in which the unit is "
            "treated (e.g. data.groupby(unit)['period'].transform(lambda s: "
            "s[treat==1].min())), with 0 or NaN for never-treated units. If "
            "treatment switches on and off, use sp.did_multiplegt_dyn, which "
            "takes the treatment path instead of a cohort."
        ),
        diagnostics={
            "function": function,
            "cohort": cohort,
            "n_units_with_varying_cohort": int(len(bad)),
            "example_units": [str(u) for u in examples],
        },
        alternative_functions=["sp.did_multiplegt_dyn"],
    )


def duplicate_unit_period_rows(
    data: pd.DataFrame,
    *,
    unit: str,
    time: str,
    function: str,
    stacklevel: int = 3,
) -> int:
    """Count rows that share a (unit, period) pair and warn when there are any.

    The regression estimators (Sun-Abraham, BJS imputation, Gardner, stacked,
    ETWFE) accept several rows per (unit, period) on purpose: with ``unit``
    set to a group level (a state) and individuals sampled in each period the
    unit effect is a group effect, as in Stata ``did_imputation`` and
    ``fixest``. The same layout is also what an accidental many-to-one merge
    produces, and then every duplicated row counts twice and the standard
    errors shrink. The two cannot be told apart from the data, so the
    estimators keep the rows and say so; ``callaway_santanna`` (one row per
    unit and period by construction) refuses instead.

    Returns the number of rows in a duplicated pair (0 when none).
    """
    dup = data.duplicated(subset=[unit, time], keep=False)
    n_dup = int(dup.sum())
    if n_dup:
        n_cells = int(data.loc[dup, [unit, time]].drop_duplicates().shape[0])
        message = (
            f"{function}: {n_dup} rows share a ({unit!r}, {time!r}) pair "
            f"({n_cells} pairs have more than one row). They are kept and "
            f"each counts as an observation, with the {unit!r} fixed effect "
            "and clusters at that level. That is right when "
            f"{unit!r} is a group (a state) with several individuals per "
            "period; if the rows are accidental copies (a many-to-one "
            "merge), drop them first, since duplicated rows understate the "
            "standard errors."
        )
        warnings.warn(
            RepeatedUnitPeriodWarning(
                message,
                recovery_hint=(
                    "data.drop_duplicates([unit, time]) if the copies are "
                    "accidental; otherwise nothing to do."
                ),
                diagnostics={
                    "function": function,
                    "n_rows_in_repeated_pairs": n_dup,
                    "n_repeated_pairs": n_cells,
                },
            ),
            stacklevel=stacklevel,
        )
    return n_dup


def always_treated_units(
    data: pd.DataFrame,
    *,
    unit: str,
    time: str,
    cohort: str,
    anticipation: int = 0,
    first_period: Optional[Any] = None,
) -> Dict[str, Any]:
    """Units whose cohort leaves them no untreated period in the sample.

    A unit first treated at ``g`` (net of ``anticipation``) at or before the
    first period of the panel is treated in every row it has. A regression
    with unit fixed effects cannot separate its effect from its level, and a
    comparison-based estimator has no pre-period to difference from.

    Returns a dict with ``mask`` (boolean array over rows), ``units`` (list),
    ``cohorts`` (sorted list) and ``first_period``.
    """
    g = normalized_cohort(data[cohort]).to_numpy(dtype=float)
    t_vals = pd.to_numeric(data[time], errors="coerce").to_numpy(dtype=float)
    t0 = float(np.nanmin(t_vals)) if first_period is None else float(first_period)
    t_max = float(np.nanmax(t_vals))
    treated = (g > 0) & (g <= t_max)
    mask = treated & (g - anticipation <= t0)
    units: List[Any] = pd.unique(data.loc[mask, unit]).tolist() if mask.any() else []
    cohorts = (
        [int(v) if float(v).is_integer() else float(v) for v in np.unique(g[mask])]
        if mask.any()
        else []
    )
    return {"mask": mask, "units": units, "cohorts": cohorts, "first_period": t0}


def drop_always_treated(
    data: pd.DataFrame,
    *,
    unit: Optional[str],
    time: str,
    cohort: str,
    function: str,
    stacklevel: int = 3,
) -> "tuple[pd.DataFrame, Optional[Dict[str, Any]]]":
    """Drop the rows of cohorts treated in or before the first period.

    For the regression estimators (ETWFE, stacked) such a cohort has no
    untreated row: its cohort x period cells sum to its own fixed effect, so
    the design is rank deficient and whichever cell the solver drops turns a
    treated period into a comparison for every other cohort. R ``did`` and
    ``callaway_santanna`` leave these cohorts out; so do these estimators,
    with a warning. Returns the filtered frame (index preserved) and a record
    for ``model_info['dropped_always_treated']`` (``None`` when nothing was
    dropped).
    """
    info = always_treated_units(data, unit=unit or time, time=time, cohort=cohort)
    mask = info["mask"]
    if not mask.any():
        return data, None
    n_rows = int(mask.sum())
    record: Dict[str, Any] = {
        "cohorts": info["cohorts"],
        "first_period": info["first_period"],
        "n_rows": n_rows,
        "n_units": len(info["units"]) if unit is not None else None,
    }
    if bool(mask.all()):
        raise DataInsufficient(
            f"{function}: every row belongs to a cohort treated in or before "
            f"the first period ({info['first_period']:g}); nothing is "
            "untreated to compare against.",
            recovery_hint="Add earlier periods or untreated units.",
            diagnostics={"function": function, **record},
        )
    who = (
        f"{record['n_units']} unit(s), {n_rows} rows"
        if unit is not None
        else f"{n_rows} rows"
    )
    warnings.warn(
        f"{function}: dropped cohort(s) {info['cohorts']} ({who}) treated in "
        f"or before the first period ({info['first_period']:g}). They have no "
        "untreated period, so their effects are not identified against "
        "their own fixed effects and would contaminate the period effects "
        "the other cohorts are compared with. R `did` and "
        "sp.callaway_santanna leave such cohorts out too.",
        UserWarning,
        stacklevel=stacklevel,
    )
    return data.loc[~mask], record


def warn_few_treated_clusters(
    data: pd.DataFrame,
    *,
    treated_rows: np.ndarray,
    cluster: str,
    function: str,
    stacklevel: int = 3,
) -> int:
    """Warn when the treated rows fall in fewer than ``FEW_TREATED_CLUSTERS``.

    Returns the number of treated clusters.
    """
    n_treated = int(pd.Series(data.loc[treated_rows, cluster]).nunique())
    if 0 < n_treated < FEW_TREATED_CLUSTERS:
        warnings.warn(
            f"{function}: the treated observations fall in only {n_treated} "
            f"cluster(s) of {cluster!r}. A cluster-robust standard error "
            "estimates the treated side's variance from that many draws and "
            "over-rejects however many control clusters there are. Report "
            "sp.did_few_treated (Conley-Taber / Ferman-Pinto) alongside.",
            AssumptionWarning,
            stacklevel=stacklevel,
        )
    return n_treated


def inference_note(bstrap: bool, cband: bool) -> Optional[str]:
    """How the reported intervals differ from R ``did``'s defaults, if they do.

    R ``did::att_gt`` defaults to ``bstrap = TRUE, cband = TRUE``: multiplier
    bootstrap standard errors and simultaneous (uniform) bands over the
    ATT(g, t). StatsPAI keeps ``bstrap=False, cband=False`` (analytic,
    pointwise), so an interval copied side by side with R's is narrower by
    construction; the note says so on the result.
    """
    if bstrap and cband:
        return None
    if bstrap:
        return (
            "multiplier-bootstrap SEs with pointwise intervals; R "
            "did::att_gt's default also sets cband=TRUE (uniform bands "
            "over the ATT(g,t)); pass cband=True for those"
        )
    return (
        "analytic SEs with pointwise intervals; R did::att_gt defaults to "
        "bstrap=TRUE, cband=TRUE (multiplier bootstrap; uniform bands over "
        "the ATT(g,t), wider than pointwise intervals); pass bstrap=True, "
        "cband=True for those"
    )
