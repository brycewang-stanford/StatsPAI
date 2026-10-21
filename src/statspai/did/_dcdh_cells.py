"""(group, period) cells and stayer slopes for ``sp.did_multiplegt``.

Rows below the group level are averaged to (group, period) cells weighted by
their row count (Stata ``did_multiplegt_old``), and ``controls=`` of the
main DID_M estimate are netted out with slopes fitted on the stayer cells.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility
from ._dcdh_first_switch import period_fe_slopes

_CELL_N = "_dcdh_cell_n"


def _collapse_to_cells(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    cluster_var: str,
) -> Tuple[pd.DataFrame, str]:
    """Average rows below the group level to (group, period) cells.

    Every analysis variable becomes its cell mean and the cell keeps its
    row count in ``_CELL_N``, the weight Stata ``did_multiplegt_old``
    gives it. A treatment that varies within a cell is averaged too, as
    Stata does. A clustering variable must be constant within a cell.
    """
    keys = [group, time]
    if cluster_var not in keys:
        varies = df.groupby(keys, sort=False)[cluster_var].nunique(dropna=False)
        if (varies > 1).any():
            raise MethodIncompatibility(
                f"sp.did_multiplegt: cluster variable {cluster_var!r} varies "
                f"within ({group}, {time}) cells. The estimator works on "
                "(group, period) cell means, so the bootstrap clusters must "
                "contain whole cells.",
                recovery_hint=(
                    f"Cluster on {group!r} or on a variable coarser than the "
                    "(group, period) cell, e.g. a region that contains whole "
                    "groups."
                ),
                diagnostics={"cluster": cluster_var},
                alternative_functions=["sp.did_multiplegt_dyn"],
            )
    cols = list(dict.fromkeys([y, treatment, *(controls or [])]))
    grouped = df.groupby(keys, sort=False)
    out = grouped[cols].mean()
    out[_CELL_N] = grouped.size().astype(float)
    if cluster_var not in keys:
        out[cluster_var] = grouped[cluster_var].first()
    return out.reset_index(), _CELL_N


def _first_differences(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: List[str],
    w: Optional[str],
) -> pd.DataFrame:
    """(group, period) first differences between consecutive periods."""
    periods = sorted(df[time].unique())
    frames = []
    cols = [group, y, treatment, *controls]
    for t_prev, t_curr in zip(periods[:-1], periods[1:]):
        cur = df.loc[df[time] == t_curr, cols + ([w] if w else [])]
        prev = df.loc[df[time] == t_prev, cols]
        m = cur.merge(prev, on=group, how="inner", suffixes=("", "_prev"))
        if m.empty:
            continue
        out = pd.DataFrame(
            {
                "_t": t_curr,
                "_d": m[treatment].to_numpy(dtype=float),
                "_d_prev": m[f"{treatment}_prev"].to_numpy(dtype=float),
                "_dy": m[y].to_numpy(dtype=float)
                - m[f"{y}_prev"].to_numpy(dtype=float),
                "_w": m[w].to_numpy(dtype=float) if w else 1.0,
            }
        )
        for c in controls:
            out[f"_dx_{c}"] = m[c].to_numpy(dtype=float) - m[f"{c}_prev"].to_numpy(
                dtype=float
            )
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _stayer_slopes(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: List[str],
    w: Optional[str],
) -> Dict[float, np.ndarray]:
    """Slopes of dY on dX with period effects, over stayers, per baseline.

    The sample for baseline ``d`` is every (group, period) cell whose
    treatment is ``d`` in both periods. A baseline with no stayer gets no
    slope (its switchers have no comparison cell anyway).
    """
    fd = _first_differences(df, y, group, time, treatment, controls, w)
    if fd.empty:
        return {}
    dx_cols = [f"_dx_{c}" for c in controls]
    stay = fd[fd["_d"] == fd["_d_prev"]]
    slopes: Dict[float, np.ndarray] = {}
    for base, sub in stay.groupby("_d_prev", sort=True):
        slopes[float(base)] = period_fe_slopes(
            sub[dx_cols].to_numpy(dtype=float),
            sub["_t"],
            sub["_dy"].to_numpy(dtype=float),
            sub["_w"].to_numpy(dtype=float),
        )
    return slopes


def _wmean(x: np.ndarray, w: Optional[np.ndarray]) -> float:
    """Mean of ``x``, weighted by ``w`` when given."""
    if w is None:
        return float(np.mean(x))
    return float(np.sum(w * x) / np.sum(w))
