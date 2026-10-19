"""Estimation-sample rules of :func:`statspai.sun_abraham`.

Kept apart from ``sun_abraham.py`` so the estimator file stays about the
regression and its aggregation.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility


def resolve_reference_periods(anticipation: Any, ref_p: Any) -> List[int]:
    """The omitted relative period(s): ``ref_p``, else ``-1 - anticipation``.

    Mirrors ``fixest::sunab(ref.p = )``. A reference period inside the
    anticipation window would put rows that already carry an effect into
    the comparison, so it is refused.
    """
    if isinstance(anticipation, bool) or not isinstance(
        anticipation, (int, np.integer)
    ):
        raise MethodIncompatibility(
            f"anticipation must be a non-negative int, got {anticipation!r}.",
            recovery_hint="Pass anticipation=0 (no anticipation) or a positive int.",
            diagnostics={"anticipation": repr(anticipation)},
        )
    k = int(anticipation)
    if k < 0:
        raise MethodIncompatibility(
            f"anticipation must be >= 0, got {k}.",
            recovery_hint="Pass anticipation=0 (no anticipation) or a positive int.",
            diagnostics={"anticipation": k},
        )
    if ref_p is None:
        return [-1 - k]
    raw: Sequence[Any] = (
        list(ref_p)
        if isinstance(ref_p, (list, tuple, set, np.ndarray, pd.Index))
        else [ref_p]
    )
    out: List[int] = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise MethodIncompatibility(
                f"ref_p must hold integer relative periods, got {v!r}.",
                recovery_hint="Pass e.g. ref_p=-1 or ref_p=[-1, -5].",
                diagnostics={"ref_p": repr(ref_p)},
            )
        out.append(int(v))
    if not out:
        raise MethodIncompatibility(
            "ref_p is empty; at least one reference period is needed.",
            recovery_hint="Pass e.g. ref_p=-1.",
            diagnostics={"ref_p": repr(ref_p)},
        )
    late = [v for v in out if v >= -k]
    if late:
        raise MethodIncompatibility(
            f"ref_p={out} includes relative period(s) {late} at or after "
            f"-anticipation={-k}; those rows may already respond to "
            "treatment and cannot serve as the reference.",
            recovery_hint=(
                f"Use reference periods below {-k}, e.g. ref_p={-1 - k}, or "
                "lower anticipation."
            ),
            diagnostics={"ref_p": out, "anticipation": k},
        )
    return sorted(set(out))


def drop_unidentified_cohorts(
    df: pd.DataFrame,
    *,
    g: str,
    t: str,
    unit: Optional[str],
    cohorts: List[Any],
    ref_set: set,
    outside_window_is_reference: bool,
    event_window: Optional[Tuple[int, int]],
) -> Tuple[pd.DataFrame, List[Any], Dict[str, Any]]:
    """Drop estimated cohorts with no row in a reference cell.

    Every row of such a cohort carries one of the cohort's interaction
    dummies, so the dummies sum to the cohort's own fixed effects (unit or
    cohort) and the regression is rank deficient. The usual case is a cohort
    treated in or before the first period of the panel ("always treated").
    Before 2026-10-19 the collinearity was resolved by the solver: on a
    balanced panel with such a cohort every post-period estimate came back
    with a standard error of 0 and p = 1, without a warning.

    Returns the filtered frame (index preserved), the remaining cohorts, and
    ``{"cohorts", "n_units", "n_rows"}`` describing what was dropped.
    """
    rel = df[t].to_numpy(dtype=float) - df[g].to_numpy(dtype=float)
    is_ref = np.isin(rel, list(ref_set))
    if outside_window_is_reference and event_window is not None:
        is_ref |= (rel < event_window[0]) | (rel > event_window[1])
    g_vals = df[g].to_numpy()
    bad = [c for c in cohorts if not is_ref[g_vals == c].any()]
    info: Dict[str, Any] = {"cohorts": [], "n_units": 0, "n_rows": 0}
    if not bad:
        return df, cohorts, info
    drop = np.isin(g_vals, bad)
    n_rows = int(drop.sum())
    n_units = int(df.loc[drop, unit].nunique()) if unit is not None else 0
    remaining = [c for c in cohorts if c not in bad]
    info = {"cohorts": [int(c) for c in bad], "n_units": n_units, "n_rows": n_rows}
    if not remaining:
        raise DataInsufficient(
            f"sun_abraham: no estimated cohort has a row in the reference "
            f"period(s) {sorted(ref_set)}; cohorts {info['cohorts']} are "
            "treated in every row the panel has for them, so no effect is "
            "identified against their own fixed effects.",
            recovery_hint=(
                "Add earlier periods to the panel, lower anticipation, or "
                "choose a reference period the cohorts are observed in "
                "(ref_p=)."
            ),
            diagnostics={"function": "sun_abraham", **info},
        )
    who = f"{n_units} unit(s), {n_rows} rows" if unit is not None else f"{n_rows} rows"
    warnings.warn(
        f"sun_abraham: dropped cohort(s) {info['cohorts']} ({who}): they "
        f"have no row in the reference period(s) {sorted(ref_set)}, i.e. "
        "they are treated in every period the panel has for them, so their "
        "effects are not identified against their own fixed effects. R "
        "`did` and sp.callaway_santanna leave such cohorts out too; "
        "fixest::sunab keeps them and drops a collinear cell, which moves "
        "the other cohorts' estimates through the period effects.",
        UserWarning,
        stacklevel=4,
    )
    return df.loc[~drop], remaining, info
