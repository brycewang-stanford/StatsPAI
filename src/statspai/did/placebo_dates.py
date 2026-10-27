"""Placebo event dates for never-treated units.

An event study of a one-off event (a move abroad, a job loss, a
hospitalisation) on individual data has a natural comparison group, the
units that never experience it, but they have no event date, so they
cannot be lined up in event time. The standard device (used, e.g., for
out-migration events in the wealth-tax literature) is to give each
never-treated unit a *placebo* date drawn from the distribution of the
treated units' dates, then treat both groups symmetrically: relative time
is ``time - date`` for everyone, and the event-time profile of the treated
is compared with that of the placebo-dated controls. Because the placebo
dates follow the treated dates' distribution, each event time mixes the
same calendar years in both groups and calendar-time shocks cancel.

``assign_placebo_dates`` does the draw: within strata if asked, only among
dates at which the unit is observed if asked, reproducibly (the result
depends on the seed and the set of units, not on the row order).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility, StatsPAIWarning

__all__ = ["assign_placebo_dates"]


class PlaceboDateWarning(StatsPAIWarning):
    """Never-treated units that could not be given a placebo date."""


def _unit_constant(data: pd.DataFrame, id: str, col: str) -> pd.Series:
    nun = data.groupby(id, sort=False, dropna=False)[col].nunique(dropna=False)
    bad = nun[nun > 1]
    if len(bad):
        raise MethodIncompatibility(
            f"assign_placebo_dates: {col!r} must be constant within each unit; "
            f"{len(bad)} unit(s) have several values (e.g. {bad.index[0]!r}).",
            recovery_hint=f"Give each unit a single {col} (the event date, or "
            "missing for never-treated units).",
            diagnostics={"column": col, "n_units": int(len(bad))},
        )
    return data.groupby(id, sort=False, dropna=False)[col].first()


def _sorted_units(index: pd.Index) -> List[Any]:
    try:
        return sorted(index.tolist())
    except TypeError:
        # mixed types: order by type name then text, still row-order free
        return sorted(index.tolist(), key=lambda v: (type(v).__name__, str(v)))


def _proportional(counts: np.ndarray, n: int) -> np.ndarray:
    """Largest-remainder allocation of ``n`` draws to frequencies ``counts``."""
    share = counts / counts.sum() * n
    base = np.floor(share).astype(int)
    rest = n - base.sum()
    if rest:
        order = np.lexsort((np.arange(len(share)), -(share - base)))
        base[order[:rest]] += 1
    return base


def assign_placebo_dates(
    data: pd.DataFrame,
    id: str,
    event_time: str,
    *,
    time: Optional[str] = None,
    strata: Optional[str] = None,
    method: Literal["empirical", "proportional"] = "empirical",
    require_observed: bool = False,
    seed: Optional[int] = None,
    column: str = "placebo_event_time",
    flag: str = "placebo",
) -> pd.DataFrame:
    """Draw placebo event dates for never-treated units.

    Every unit whose ``event_time`` is missing receives a date drawn from
    the empirical distribution of the treated units' dates (one date per
    treated unit), so that an event study can compare treated and
    never-treated units in event time.

    Parameters
    ----------
    data : pd.DataFrame
        Panel (one row per unit and period) or one row per unit.
    id : str
        Unit identifier.
    event_time : str
        Event date of each unit, constant within the unit; missing for
        never-treated units.
    time : str, optional
        Period column. When given, a relative-time column
        ``<column>_rel = time - date`` is added, and ``require_observed``
        becomes available.
    strata : str, optional
        Column constant within units (e.g. birth cohort x wealth bin):
        dates are drawn from the treated units of the same stratum.
    method : {'empirical', 'proportional'}, default 'empirical'
        ``'empirical'``: independent draws from the treated dates'
        frequencies. ``'proportional'``: the never-treated units of a
        stratum receive the treated dates in proportion to their
        frequencies (largest-remainder rounding), in random order; when the
        number of never-treated units is a multiple of the number of treated
        units the frequencies are reproduced exactly.
    require_observed : bool, default False
        Draw only among dates at which the unit has a row (needs ``time``),
        so that the placebo event falls inside the unit's observation
        window. Only with ``method='empirical'``. A unit observed at none of
        the treated dates gets no placebo date (warning).
    seed : int, optional
        Seed. The draw depends on the seed and on the set of units and their
        attributes, not on the order of the rows.
    column : str, default 'placebo_event_time'
        Name of the new date column: the event date for treated units, the
        placebo date for never-treated ones.
    flag : str, default 'placebo'
        Name of the new boolean column, True on the rows of units whose date
        is a placebo.

    Returns
    -------
    pd.DataFrame
        A copy of ``data`` with ``column``, ``flag`` and, with ``time``,
        ``column + '_rel'``. ``attrs['placebo_dates']`` records the number
        of treated, placebo-dated and undated units and the method.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({'id': [1, 1, 2, 2, 3, 3, 4, 4],
    ...                    'year': [2000, 2001] * 4,
    ...                    'move': [2000, 2000, 2001, 2001] + [None] * 4})
    >>> out = sp.assign_placebo_dates(df, 'id', 'move', time='year', seed=1)
    >>> out.groupby('id')['placebo'].first().tolist()
    [False, False, True, True]
    >>> bool(out.loc[out['placebo'], 'placebo_event_time'].isin([2000, 2001]).all())
    True
    """
    if method not in ("empirical", "proportional"):
        raise MethodIncompatibility(
            f"assign_placebo_dates: method must be 'empirical' or "
            f"'proportional'; got {method!r}.",
            recovery_hint="Use method='empirical' (independent draws).",
            diagnostics={"method": method},
        )
    if require_observed and time is None:
        raise MethodIncompatibility(
            "assign_placebo_dates: require_observed needs time=.",
            recovery_hint="Pass the period column as time=.",
        )
    if require_observed and method == "proportional":
        raise MethodIncompatibility(
            "assign_placebo_dates: require_observed restricts each unit to its "
            "own set of dates, which a proportional allocation cannot respect.",
            recovery_hint="Use method='empirical' with require_observed=True.",
        )
    needed = [id, event_time] + [c for c in (time, strata) if c is not None]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"assign_placebo_dates: columns not found: {missing}.",
            recovery_hint="Check the column names.",
            diagnostics={"missing": missing},
        )
    for c in (column, flag):
        if c in data.columns:
            raise MethodIncompatibility(
                f"assign_placebo_dates: output column {c!r} already exists.",
                recovery_hint="Pass another name with column= / flag=.",
                diagnostics={"column": c},
            )
    for c in [id] + ([strata] if strata is not None else []):
        if data[c].isna().any():
            raise MethodIncompatibility(
                f"assign_placebo_dates: {c!r} has missing values.",
                recovery_hint=f"Drop or fill the rows without {c}.",
                diagnostics={"column": c},
            )

    ev = _unit_constant(data, id, event_time)
    st = (
        _unit_constant(data, id, strata)
        if strata is not None
        else pd.Series(0, index=ev.index)
    )
    units = _sorted_units(ev.index)
    ev = ev.reindex(units)
    st = st.reindex(units)
    treated = ev.notna()
    if not treated.any():
        raise DataInsufficient(
            "assign_placebo_dates: no unit has an event date, so there is no "
            "distribution to draw from.",
            recovery_hint="Check the event_time column.",
        )
    observed: Dict[Any, np.ndarray] = {}
    if require_observed:
        obs = data.dropna(subset=[time]).groupby(id, sort=False)[time]
        observed = {k: np.unique(v.to_numpy()) for k, v in obs}

    rng = np.random.default_rng(seed)
    picked: Dict[Any, Any] = {}
    undated: List[Any] = []
    no_donor: List[Any] = []
    for s in _sorted_units(pd.Index(st.unique())):
        in_s = (st == s).to_numpy()
        donors = ev[in_s & treated.to_numpy()]
        never = [u for u, m in zip(units, in_s & ~treated.to_numpy()) if m]
        if not never:
            continue
        if donors.empty:
            no_donor.append(s)
            undated.extend(never)
            continue
        support, counts = np.unique(donors.to_numpy(), return_counts=True)
        if method == "proportional":
            alloc = _proportional(counts.astype(float), len(never))
            dates = np.repeat(support, alloc)
            dates = dates[rng.permutation(len(dates))]
            for u, dt in zip(never, dates):
                picked[u] = dt
            continue
        if not require_observed:
            pick = rng.choice(len(support), size=len(never), p=counts / counts.sum())
            for u, j in zip(never, pick):
                picked[u] = support[j]
            continue
        for u in never:
            probs = counts.astype(float)
            if require_observed:
                ok = np.isin(support, observed.get(u, np.array([])))
                probs = np.where(ok, probs, 0.0)
                if probs.sum() == 0:
                    undated.append(u)
                    continue
            picked[u] = support[rng.choice(len(support), p=probs / probs.sum())]

    if undated:
        warnings.warn(
            f"assign_placebo_dates: {len(undated)} never-treated unit(s) got no "
            "placebo date"
            + (
                f" ({len(no_donor)} stratum/strata without treated units)"
                if no_donor
                else ""
            )
            + (
                "; some are observed at none of the treated dates"
                if require_observed
                else ""
            )
            + ". They keep a missing date and placebo=False.",
            PlaceboDateWarning,
            stacklevel=2,
        )

    drawn = pd.Series(picked, dtype=object).reindex(ev.index)
    unit_date = ev.astype(object).where(treated, drawn)
    if pd.api.types.is_numeric_dtype(data[event_time]):
        unit_date = unit_date.astype(float)
    elif pd.api.types.is_datetime64_any_dtype(data[event_time]):
        unit_date = pd.to_datetime(unit_date)
    unit_flag = (~treated) & drawn.notna()
    out = data.copy()
    out[column] = out[id].map(unit_date)
    out[flag] = out[id].map(unit_flag).astype(bool)
    if time is not None:
        out[column + "_rel"] = out[time] - out[column]
    out.attrs = dict(data.attrs)
    out.attrs["placebo_dates"] = {
        "n_treated_units": int(treated.sum()),
        "n_placebo_units": int(unit_flag.sum()),
        "n_undated_units": len(undated),
        "strata_without_treated": no_donor,
        "method": method,
        "require_observed": bool(require_observed),
        "seed": seed,
    }
    return out
