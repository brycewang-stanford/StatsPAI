"""Panel layout for dynamic-panel GMM.

Every variable is pivoted onto a dense ``(n_units, n_periods)`` array with
``NaN`` marking "not observed".  Two things follow from that choice:

1. **Availability is per variable, not per row.**  The previous
   implementation applied a listwise ``dropna`` across ``[id, time, y] + x``
   *before* building anything, so a missing covariate at period *t* deleted
   ``y_{i,t}`` from the **instrument** pool as well as from the estimation
   sample.  Reproducing Arellano & Bond (1991) Table 4 — which needs
   ``L2.k``, hence leading ``NaN``s — silently lost 13 of 32 instruments and
   280 of 611 observations, moving ρ̂ from 0.849 to 0.660.  Per-variable
   availability is the fix.
2. Time is treated as an **ordinal calendar**: the sorted distinct values of
   the time column define consecutive positions, so an interior gap is a
   real gap (its column is all-``NaN`` for that unit) while irregular
   labels are collapsed to rank order.

References
----------
Arellano, M. and Bond, S. (1991). *Review of Economic Studies* 58(2),
277-297. [@arellano1991some]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

__all__ = ["PanelArrays", "build_panel_arrays"]


@dataclass
class PanelArrays:
    """Dense ``(n_units, n_periods)`` view of a long panel.

    Attributes
    ----------
    values : dict[str, np.ndarray]
        One ``(N, T)`` float array per variable; ``NaN`` where unobserved.
    units : np.ndarray
        Unit labels in first-appearance-sorted order, indexing axis 0.
    times : np.ndarray
        Sorted distinct time labels, indexing axis 1.
    """

    values: Dict[str, np.ndarray]
    units: np.ndarray
    times: np.ndarray

    @property
    def n_units(self) -> int:
        return int(self.units.size)

    @property
    def n_periods(self) -> int:
        return int(self.times.size)

    def get(self, var: str) -> np.ndarray:
        try:
            return self.values[var]
        except KeyError:  # pragma: no cover - guarded upstream
            raise KeyError(
                f"variable {var!r} was not materialised; available: "
                f"{sorted(self.values)}"
            ) from None

    def lagged(self, var: str, lag: int) -> np.ndarray:
        """``var`` shifted ``lag`` periods forward in the time index.

        ``lagged(v, 1)[:, p]`` is ``v_{i, p-1}``; positions with no source
        period are ``NaN``.  Vectorised over units, so no Python loop over
        the panel is ever needed to form a lag.
        """
        arr = self.get(var)
        if lag == 0:
            return arr
        out = np.full_like(arr, np.nan)
        if lag < arr.shape[1]:
            out[:, lag:] = arr[:, : arr.shape[1] - lag]
        return out

    def observed(self, var: str) -> np.ndarray:
        """Boolean ``(N, T)`` availability mask for ``var``."""
        return np.isfinite(self.get(var))

    def unit_period_mask(self) -> np.ndarray:
        """``(N, T)`` mask of periods where the unit appears at all."""
        stacked = np.stack([np.isfinite(v) for v in self.values.values()])
        return stacked.any(axis=0)


def build_panel_arrays(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    variables: Sequence[str],
) -> PanelArrays:
    """Pivot ``data`` into :class:`PanelArrays`.

    Rows with a missing ``id`` or ``time`` are dropped (they cannot be
    placed on the panel grid); missing *values* are preserved as ``NaN`` so
    that availability stays per variable.

    Raises
    ------
    ValueError
        If a ``(id, time)`` pair is duplicated — silently keeping the last
        row would make the estimate depend on input ordering — or if a
        requested variable is absent from ``data``.
    """
    missing = [v for v in dict.fromkeys(variables) if v not in data.columns]
    if missing:
        raise ValueError(
            f"variable(s) {missing} not found in the data. Available columns: "
            f"{list(data.columns)}"
        )
    for col, what in ((id_col, "id"), (time_col, "time")):
        if col not in data.columns:
            raise ValueError(f"{what} column {col!r} not found in the data.")

    cols = [id_col, time_col] + list(dict.fromkeys(variables))
    df = data.loc[:, cols].copy()
    df = df[df[id_col].notna() & df[time_col].notna()]
    if df.empty:
        raise ValueError("no rows left after dropping missing id / time values.")

    dup = df.duplicated(subset=[id_col, time_col]).sum()
    if dup:
        raise ValueError(
            f"{dup} duplicated (id, time) pair(s) in the panel. Dynamic-panel "
            "GMM needs one row per unit-period; aggregate or de-duplicate first."
        )

    units = np.asarray(pd.unique(df[id_col].sort_values()))
    times = np.asarray(np.sort(pd.unique(df[time_col])))
    unit_pos = pd.Series(np.arange(units.size), index=units)
    time_pos = pd.Series(np.arange(times.size), index=times)

    ui = unit_pos.reindex(df[id_col]).to_numpy()
    ti = time_pos.reindex(df[time_col]).to_numpy()

    values: Dict[str, np.ndarray] = {}
    for var in dict.fromkeys(variables):
        arr = np.full((units.size, times.size), np.nan, dtype=float)
        arr[ui, ti] = pd.to_numeric(df[var], errors="coerce").to_numpy(dtype=float)
        values[var] = arr

    return PanelArrays(values=values, units=units, times=times)


def add_time_dummies(
    panel: PanelArrays, drop_first: int = 1, prefix: str = "_T"
) -> List[str]:
    """Materialise period dummies as panel variables, in place.

    Roodman (2009) recommends always including time dummies in dynamic
    panel GMM: they absorb common shocks and make the
    no-cross-sectional-correlation assumption behind the moment conditions
    considerably more plausible.

    ``drop_first`` periods are omitted to avoid collinearity with the
    (differenced-away) constant.  The dummies are defined on every
    unit-period the panel covers — including periods where the outcome is
    missing — so they behave like the deterministic regressors they are.

    Returns the names of the created variables, in period order.
    """
    if drop_first < 0:
        raise ValueError("drop_first must be >= 0")
    present = panel.unit_period_mask()
    names: List[str] = []
    for p in range(drop_first, panel.n_periods):
        name = f"{prefix}{panel.times[p]}"
        arr = np.where(present, 0.0, np.nan)
        arr[:, p] = np.where(present[:, p], 1.0, np.nan)
        panel.values[name] = arr
        names.append(name)
    return names
