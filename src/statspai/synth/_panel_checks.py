"""Outcome-panel completeness checks shared by the synthetic-control estimators.

The single-treated-unit estimators each pivot the long panel themselves, and
until 2026-11 none of them looked at what the pivot left behind: one NaN in
the treated unit's pre-period outcome made the classic solver's loss NaN,
left the donor weights at their equal starting values with
``converged=True`` and moved the estimate; a NaN in a donor's outcome dropped
the donor without a word, crashed the factor and clustering methods with a
``LinAlgError`` / sklearn error, or gave a NaN estimate (``method='sdid'``).
The rules, for every method:

* ``inf`` in the outcome is refused (the package-wide policy of
  :func:`statspai._input_validation.require_finite`);
* a treated unit with a missing outcome in any period is refused;
* a donor with a missing outcome in any period is dropped with an
  :class:`~statspai.exceptions.AssumptionWarning`, and the drop is recorded
  in ``model_info['dropped_donors']``; methods whose reference refuses an
  unbalanced panel (``sdid``, ``gsynth``, ``rcm``) refuse instead, and
  ``mc`` keeps them (matrix completion is built for missing cells).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..exceptions import DataInsufficient


def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


__all__ = ["outcome_panel_gaps", "warn_dropped_donors", "screen_synth_panel"]


def outcome_panel_gaps(
    data: Any,
    *,
    outcome: str,
    unit: str,
    time: str,
    treated_units: Any,
    function: str,
) -> Dict[Any, int]:
    """Refuse an unusable outcome panel; return the donors with missing cells.

    The shared pre-processing guard of the synthetic-control estimators.
    Raises :class:`~statspai.exceptions.DataInsufficient` (with a
    ``recovery_hint``) when

    * the outcome column holds ``inf`` (the package-wide policy of
      :func:`statspai._input_validation.require_finite`), or
    * a treated unit's outcome is missing (NaN, or no row) in any period of
      the panel, pre- or post-treatment. The weights are fitted to the
      treated unit's own pre-period path and the effect is read off its
      post-period path, so neither can be filled in by the method. A single
      NaN used to turn the solver loss into NaN, leave the donor weights at
      their equal starting values with ``converged=True``, and move the
      estimate.

    Returns ``{donor: number of periods with a missing outcome}`` for the
    donors that have any (empty when the panel is complete); what to do with
    them is the caller's rule (:func:`warn_dropped_donors` when it drops
    them). The data are only read, never modified.
    """
    import pandas as pd

    from .._input_validation import require_finite

    require_finite(data, [outcome], function=function)
    if not isinstance(treated_units, (list, tuple, set, np.ndarray, pd.Index)):
        treated_units = [treated_units]
    treated = list(treated_units)
    periods = pd.unique(data[time].dropna())
    n_periods = len(periods)
    observed = data.loc[data[outcome].notna(), [unit, time]]
    n_observed = observed.drop_duplicates().groupby(unit, sort=False)[time].nunique()
    units = pd.unique(data[unit].dropna())
    gaps = {
        u: int(n_periods - int(n_observed.get(u, 0)))
        for u in units
        if int(n_observed.get(u, 0)) < n_periods
    }
    for u in treated:
        if u not in gaps:
            continue
        seen = set(observed.loc[observed[unit] == u, time])
        missing = [
            _plain(t)
            for t in sorted(
                (t for t in periods if t not in seen), key=lambda t: (str(type(t)), t)
            )
        ]
        raise DataInsufficient(
            f"{function}: the treated unit {u!r} has no observed outcome "
            f"{outcome!r} in {len(missing)} period(s): {list(missing)[:10]}"
            + (" ..." if len(missing) > 10 else "")
            + ". Synthetic control matches the treated unit's own pre-period "
            "path and reads the effect off its post-period path, so a "
            "missing treated outcome cannot be filled in by the method.",
            recovery_hint=(
                "Restrict the panel to periods where the treated unit is "
                "observed (drop those periods for every unit), or impute the "
                "missing values explicitly before fitting."
            ),
            diagnostics={
                "treated_unit": _plain(u),
                "missing_periods": missing,
                "n_periods": int(n_periods),
            },
        )
    return {u: n for u, n in gaps.items() if u not in set(treated)}


def warn_dropped_donors(
    gaps: Dict[Any, int], *, function: str, n_donors: int, stacklevel: int = 3
) -> Dict[str, Any]:
    """Warn that the donors in ``gaps`` were dropped; return the record.

    The record (``{"dropped_donors": [...], "dropped_donor_missing_periods":
    {...}, "n_donors_before_drop": n}``) goes into the result's
    ``model_info`` so the reduced donor pool is visible after the fact.
    """
    if not gaps:
        return {}
    from ..exceptions import AssumptionWarning

    names = [_plain(u) for u in gaps]
    record = {
        "dropped_donors": names,
        "dropped_donor_missing_periods": {str(k): int(v) for k, v in gaps.items()},
        "n_donors_before_drop": int(n_donors),
    }
    warnings.warn(
        AssumptionWarning(
            f"{function}: {len(names)} of {n_donors} donor unit(s) dropped "
            f"because their outcome is missing in some period: "
            f"{names[:10]}" + (" ..." if len(names) > 10 else "") + ".",
            recovery_hint=(
                "The synthetic control is built from the remaining donors. "
                "Impute the missing donor outcomes, or drop those units "
                "yourself to silence this note."
            ),
            diagnostics=record,
        ),
        stacklevel=stacklevel,
    )
    return record


def refuse_donor_gaps(gaps: Dict[Any, int], *, function: str) -> None:
    """Refuse donors with missing outcome cells (balanced-panel methods)."""
    if not gaps:
        return
    names = [_plain(u) for u in gaps]
    raise DataInsufficient(
        f"{function}: {len(names)} control unit(s) have a missing outcome in "
        f"some period: {names[:10]}"
        + (" ..." if len(names) > 10 else "")
        + ". This estimator needs a balanced outcome panel.",
        recovery_hint=(
            "Drop those units (data[~data[unit].isin([...])]) or impute their "
            "missing outcomes; sp.synth(method='classic') drops such donors "
            "with a warning, and method='mc' completes the missing cells."
        ),
        diagnostics={
            "units_with_missing_outcome": {str(k): int(v) for k, v in gaps.items()},
        },
    )


#: How ``sp.synth`` treats an incomplete outcome panel, by method. Methods
#: not listed check their own panel (the classic family, sdid, gsynth) or
#: take a different data layout (staggered, multi_outcome).
_DONOR_RULE = {
    **dict.fromkeys(
        (
            "demeaned",
            "detrended",
            "unconstrained",
            "elastic_net",
            "augmented",
            "ascm",
            "scpi",
            "prediction_interval",
            "bayesian",
            "penscm",
            "abadie_lhour",
            "pairwise",
            "fdid",
            "forward_did",
            "cluster",
            "sparse",
            "lasso",
            "kernel",
            "kernel_ridge",
        ),
        "drop",
    ),
    # matrix completion fills missing donor cells by construction
    **dict.fromkeys(("mc", "matrix_completion"), "keep"),
    # the regression control method refuses missing cells with its own message
    **dict.fromkeys(("rcm", "hcw", "regression_control", "pda"), "keep"),
    # state-space (bsts) and distributional (discos) fits handle a missing
    # observation themselves; only inf is refused
    **dict.fromkeys(("bsts", "causal_impact", "discos", "distributional"), "inf"),
}


def screen_synth_panel(
    data: Any,
    *,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    method: str,
    inference: Optional[str],
) -> Tuple[Any, Dict[str, Any]]:
    """Apply the outcome-panel rules before ``sp.synth`` dispatches.

    Returns the data to fit (the same object when nothing is dropped) and
    the record to merge into the result's ``model_info``.
    """
    import pandas as pd

    from .._input_validation import require_finite

    if inference in ("ttest", "conformal"):
        rule: Optional[str] = "drop"  # these build their own matrices
    else:
        rule = _DONOR_RULE.get(method)
    if rule is None or treated_unit is None or not isinstance(data, pd.DataFrame):
        return data, {}
    if any(c not in data.columns for c in (outcome, unit, time)):
        return data, {}  # the method's own column check names the typo
    function = f"sp.synth(method={method!r})"
    if rule == "inf":
        require_finite(data, [outcome], function=function)
        return data, {}
    gaps = outcome_panel_gaps(
        data,
        outcome=outcome,
        unit=unit,
        time=time,
        treated_units=[treated_unit],
        function=function,
    )
    if rule == "keep" or not gaps:
        return data, {}
    n_donors = int(pd.Series(data[unit].unique()).ne(treated_unit).sum())
    if len(gaps) >= n_donors:
        raise DataInsufficient(
            f"{function}: every donor unit has a missing outcome in some "
            "period, so no donor is left.",
            recovery_hint="Add donor units with fully observed outcomes.",
            diagnostics={"n_donors": n_donors},
        )
    record = warn_dropped_donors(
        gaps, function=function, n_donors=n_donors, stacklevel=4
    )
    return data[~data[unit].isin(list(gaps))], record
