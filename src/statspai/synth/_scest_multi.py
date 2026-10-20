"""Synthetic control weights for several treated units with staggered adoption.

The design follows the multiple-treated-unit setting of the scpi framework
[@cattaneo2025scpi]: every treated unit ``i`` that adopts the treatment at
period ``T_i`` gets its own synthetic control, fitted on its pre-treatment
periods (those before ``T_i - anticipation``) with the never-treated units
as donors. The constraint sets are separate across treated units, so the
joint least-squares problem over the stacked weight vector splits into one
problem per unit, each the single-unit problem of :func:`statspai.scest`.

Three summaries of the post-treatment gaps ``tau_{it} = Y_{it} - Y^sc_{it}``
are reported, as R ``scpi::scdataMulti(effect=)`` names them:

* ``'unit-time'``: every ``tau_{it}``;
* ``'unit'``: each unit's average over its post-treatment periods;
* ``'time'``: the average over treated units at each horizon ``k = 1, ...,
  min_i T1_i`` (horizon 1 is the first period after the pre-treatment
  window), the horizons every treated unit reaches.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from ..exceptions import DataInsufficient, MethodIncompatibility

FN = "sp.scest"
EFFECTS = ("unit-time", "unit", "time")


def _refuse(message: str, hint: str, **diag: Any) -> MethodIncompatibility:
    return MethodIncompatibility(
        f"{FN}: {message}", recovery_hint=hint, diagnostics=diag
    )


def adoption_times(
    data: pd.DataFrame, unit: str, time: str, treat: str
) -> Dict[Any, Any]:
    """First treated period of every ever-treated unit.

    Refuses a treatment column that is not 0/1 or that switches off after
    switching on (the staggered design assumes adoption is absorbing).
    """
    d = data[[unit, time, treat]]
    if d[treat].isna().any():
        raise _refuse(
            f"the treatment column {treat!r} has missing values.",
            "Code every unit-period as 0 (untreated) or 1 (treated).",
        )
    values = set(pd.unique(d[treat]))
    if not values <= {0, 1, True, False}:
        raise _refuse(
            f"the treatment column {treat!r} must be 0/1, found "
            f"{sorted(values, key=str)[:5]}.",
            "Code the treatment as 1 from the adoption period on and 0 before.",
        )
    adopt: Dict[Any, Any] = {}
    for u, g in d.sort_values(time).groupby(unit, sort=True):
        flags = g[treat].astype(int).to_numpy()
        if flags.any():
            first = int(np.argmax(flags))
            if not flags[first:].all():
                raise _refuse(
                    f"unit {u!r} is treated and later untreated; staggered "
                    "synthetic control needs adoption to be absorbing.",
                    "Keep the treatment switched on after adoption, or drop "
                    "the unit.",
                    unit=u,
                )
            adopt[u] = g[time].iloc[first]
    return adopt


def scest_multi(
    data: pd.DataFrame,
    *,
    outcome: str,
    unit: str,
    time: str,
    treat: str,
    treated_unit: Any,
    effect: str,
    anticipation: int,
    single: Any,
    fit_kw: Dict[str, Any],
) -> Dict[str, Any]:
    """Fit one synthetic control per treated unit; see the module docstring.

    ``single`` is the single-unit estimator (:func:`statspai.scest`) and
    ``fit_kw`` its constraint arguments.
    """
    if effect not in EFFECTS:
        raise _refuse(
            f"effect={effect!r} is not one of {list(EFFECTS)}.",
            "'unit-time' (every gap), 'unit' (each unit's average) or "
            "'time' (average across units by horizon).",
        )
    if (
        isinstance(anticipation, (bool, np.bool_))
        or not isinstance(anticipation, (int, np.integer))
        or anticipation < 0
    ):
        raise _refuse(
            f"anticipation={anticipation!r} is not a non-negative integer.",
            "Pass the number of periods before adoption that may already "
            "respond, e.g. anticipation=1; 0 for none.",
        )
    check_columns(
        data,
        {"outcome": outcome, "unit": unit, "time": time, "treat": treat},
        function=FN,
    )
    adopt = adoption_times(data, unit, time, treat)
    if not adopt:
        raise DataInsufficient(
            f"{FN}: no unit is ever treated in {treat!r}.",
            recovery_hint="Check the treatment column.",
        )
    donors = sorted(set(data[unit]) - set(adopt), key=lambda v: (str(type(v)), v))
    if not donors:
        raise DataInsufficient(
            f"{FN}: every unit is treated at some point, so there are no "
            "never-treated donors.",
            recovery_hint="Synthetic control needs untreated donor units; "
            "for not-yet-treated comparisons see sp.staggered_synth.",
            alternative_functions=["staggered_synth"],
        )
    if treated_unit is None:
        targets: List[Any] = sorted(adopt, key=lambda v: (str(type(v)), v))
    else:
        targets = (
            list(treated_unit)
            if isinstance(treated_unit, (list, tuple, set, np.ndarray, pd.Index))
            else [treated_unit]
        )
        unknown = [u for u in targets if u not in adopt]
        if unknown:
            raise _refuse(
                f"treated_unit={unknown!r} never receive the treatment in "
                f"{treat!r}.",
                f"Choose among the treated units: {sorted(map(str, adopt))[:10]}.",
            )

    periods = np.sort(pd.unique(data[time]))
    fits: Dict[Any, Dict[str, Any]] = {}
    start: Dict[Any, Any] = {}
    for u in targets:
        pos = int(np.searchsorted(periods, adopt[u])) - int(anticipation)
        if pos < 1:
            raise DataInsufficient(
                f"{FN}: unit {u!r} has no pre-treatment period before "
                f"{adopt[u]!r} (anticipation={anticipation}).",
                recovery_hint="Drop units treated from the first period, or "
                "reduce anticipation=.",
                diagnostics={"unit": u, "adoption": adopt[u]},
            )
        start[u] = periods[pos]
        sub = data[data[unit].isin([u, *donors])]
        fits[u] = single(
            sub,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=u,
            treatment_time=start[u],
            **fit_kw,
        )

    weights = pd.DataFrame(
        {
            u: pd.Series(fits[u]["weights"], index=fits[u]["donor_names"])
            for u in targets
        }
    )
    rows_post, rows_pre = [], []
    for u in targets:
        sc = fits[u]["sc_data"]
        for k, (t, y, s) in enumerate(
            zip(sc["post_times"], sc["Y_post"], fits[u]["Y_synth_post"]), start=1
        ):
            rows_post.append((u, t, k, y, s, y - s))
        for t, y, s in zip(sc["pre_times"], sc["Y_pre"], fits[u]["Y_synth_pre"]):
            rows_pre.append((u, t, y, s, y - s))
    gaps = pd.DataFrame(
        rows_post,
        columns=["unit", "time", "horizon", "outcome", "synthetic", "effect"],
    )
    pre_fit = pd.DataFrame(
        rows_pre, columns=["unit", "time", "outcome", "synthetic", "residual"]
    )
    if effect == "unit-time":
        table = gaps
    elif effect == "unit":
        table = (
            gaps.groupby("unit", sort=False)
            .agg(
                n_post=("effect", "size"),
                outcome=("outcome", "mean"),
                synthetic=("synthetic", "mean"),
                effect=("effect", "mean"),
            )
            .reset_index()
        )
    else:
        reach = int(gaps.groupby("unit")["horizon"].max().min())
        kept = gaps[gaps["horizon"] <= reach]
        table = (
            kept.groupby("horizon")
            .agg(
                n_units=("effect", "size"),
                outcome=("outcome", "mean"),
                synthetic=("synthetic", "mean"),
                effect=("effect", "mean"),
            )
            .reset_index()
        )
    return {
        "weights": weights,
        "w_constr": fits[targets[0]]["w_constr"],
        "w_constr_spec": {u: fits[u]["w_constr_spec"] for u in targets},
        "effect": effect,
        "effects": table,
        "Y_synth_post": table["synthetic"].to_numpy(dtype=float),
        "unit_time_effects": gaps,
        "pre_fit": pre_fit,
        "pre_rmspe": {u: fits[u]["pre_rmspe"] for u in targets},
        "treated_units": list(targets),
        "adoption_times": {u: adopt[u] for u in targets},
        "treatment_times": start,
        "anticipation": int(anticipation),
        "donor_names": list(donors),
        "unit_fits": fits,
    }
