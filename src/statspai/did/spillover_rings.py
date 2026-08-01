"""Heterogeneity-robust DiD with spatial spillovers (Butts).

The usual fix for spillovers is to add a spatial lag of treatment to a
two-way fixed-effects regression. That inherits every problem TWFE has
under staggered adoption -- already-treated units serve as controls, the
implied weights can go negative -- and adds one of its own: the "control"
units nearest the treated are precisely the ones the spillover reaches, so
the direct effect is measured against a contaminated baseline.

Butts's answer is to stop pretending the control group is homogeneous.
Units are sorted by distance to the nearest treated unit into

* the **treated** units themselves,
* one or more **spillover rings** -- untreated units within a given
  distance band of a treated unit, and
* the **clean controls** -- untreated units beyond every ring.

Each ring then gets its own group-time effect, estimated against the clean
controls only. The direct effect is likewise measured against clean
controls, so it is no longer diluted by units the treatment reached
indirectly. Reporting the rings is the point: a spillover that decays with
distance is visible, and one that does not tells you the rings are too
narrow.

Validation
----------
There is no reference implementation to pin against -- no CRAN or GitHub
package implements this estimator -- so correctness rests on recovering a
known design. ``tests/reference_parity/test_spillover_rings.py`` plants a
direct effect and two ring effects in a spatial DGP and asserts all three
are recovered, and asserts that ``sp.spatial_did`` (the TWFE + spatial-lag
design) is biased on the same data where this is not.

References
----------
butts2021difference
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["SpilloverRingResult", "spillover_did"]


@dataclass
class SpilloverRingResult(ResultProtocolMixin):
    """Direct and ring-by-ring spillover effects."""

    direct: float
    direct_se: float
    rings: pd.DataFrame
    detail: pd.DataFrame
    n_units: int
    n_clean_controls: int
    ring_edges: np.ndarray
    alpha: float = 0.05
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    method: str = "Spillover-ring DiD (Butts)"

    @property
    def ci(self) -> tuple:
        z = float(stats.norm.ppf(1 - self.alpha / 2))
        return (self.direct - z * self.direct_se, self.direct + z * self.direct_se)

    def summary(self) -> str:
        lo, hi = self.ci
        lines = [
            self.method,
            "=" * len(self.method),
            f"  units             : {self.n_units}",
            f"  clean controls    : {self.n_clean_controls}",
            f"  ring edges        : {np.round(self.ring_edges, 4).tolist()}",
            "",
            f"  direct effect     : {self.direct:.6f} (se {self.direct_se:.6f}, "
            f"{100 * (1 - self.alpha):.0f}% CI [{lo:.6f}, {hi:.6f}])",
            "",
            "Spillover by ring (untreated units, by distance to the nearest "
            "treated unit):",
            self.rings.to_string(index=False),
            "",
        ]
        if self.n_clean_controls < 30:
            lines.append(
                f"WARNING: only {self.n_clean_controls} clean controls. Every "
                "effect here is measured against them, so with this few the "
                "standard errors are optimistic and the rings may simply be "
                "too wide for the geography."
            )
        else:
            lines.append(
                "All effects are measured against the clean controls, so the "
                "direct effect is not diluted by units the treatment reached "
                "indirectly. If the ring effects do not decay with distance, "
                "the outermost ring is probably not clean either."
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "direct": self.direct,
            "direct_se": self.direct_se,
            "rings": self.rings.to_dict(orient="records"),
            "n_units": self.n_units,
            "n_clean_controls": self.n_clean_controls,
            "ring_edges": np.asarray(self.ring_edges).tolist(),
            "alpha": self.alpha,
            "diagnostics": {
                k: v for k, v in self.diagnostics.items() if not k.startswith("_")
            },
        }


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def spillover_did(
    data: pd.DataFrame,
    y: str,
    *,
    unit: str,
    time: str,
    cohort: str,
    coords: Optional[Sequence[str]] = None,
    distances: Optional[np.ndarray] = None,
    ring_edges: Sequence[float] = (0.0, 1.0),
    never_value: Any = 0,
    alpha: float = 0.05,
) -> SpilloverRingResult:
    """Direct and spillover effects with distance-banded control groups.

    Parameters
    ----------
    data : DataFrame
        Long-format panel.
    y : str
        Outcome column.
    unit, time, cohort : str
        Unit identifier, period, and first-treatment period
        (``never_value`` marks never-treated units).
    coords : sequence of str, optional
        Two columns giving each unit's position. Distances are Euclidean in
        whatever units these are. Ignored when ``distances`` is given.
    distances : ndarray, optional
        Pre-computed ``(n_units, n_units)`` distance matrix, ordered by
        sorted unit id. Use this for great-circle or network distances.
    ring_edges : sequence of float, default (0.0, 1.0)
        Ring boundaries. ``(0, 1, 2)`` makes two rings, ``(0, 1]`` and
        ``(1, 2]``; untreated units beyond the last edge are the clean
        controls. Rings are not estimated where no untreated unit falls in
        them.
    never_value : any, default 0
        Value in ``cohort`` marking never-treated units.
    alpha : float, default 0.05

    Returns
    -------
    SpilloverRingResult

    Notes
    -----
    Standard errors are the influence-function form for a difference of
    group means, aggregated across ``(cohort, period)`` cells so the shared
    control units are accounted for. There is no reference implementation
    to pin them against; see the module docstring on how correctness is
    established instead.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x = rng.uniform(0, 10, n); yc = rng.uniform(0, 10, n)
    >>> treated = (x < 3) & (yc < 3)
    >>> rows = []
    >>> for i in range(n):
    ...     for t in (1, 2):
    ...         rows.append((i, t, 2 if treated[i] else 0, x[i], yc[i],
    ...                      rng.normal() + (1.0 if (treated[i] and t == 2) else 0)))
    >>> df = pd.DataFrame(rows, columns=["i", "t", "g", "x", "y2", "y"])
    >>> res = sp.spillover_did(df, y="y", unit="i", time="t", cohort="g",
    ...                        coords=["x", "y2"], ring_edges=(0.0, 2.0))
    >>> bool(np.isfinite(res.direct))
    True

    References
    ----------
    butts2021difference
    """
    context = "spillover_did"
    if (coords is None) == (distances is None):
        raise MethodIncompatibility(
            f"{context}: pass exactly one of `coords` or `distances`.",
            diagnostics={"context": context},
        )
    if not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility(
            f"{context}: alpha must be in (0, 1).",
            diagnostics={"context": context, "alpha": alpha},
        )
    edges = np.asarray(ring_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise MethodIncompatibility(
            f"{context}: `ring_edges` must be increasing with at least two " "entries.",
            diagnostics={"context": context, "ring_edges": edges.tolist()},
        )
    for col in (y, unit, time, cohort):
        if col not in data.columns:
            raise MethodIncompatibility(
                f"{context}: column {col!r} not in data.",
                diagnostics={"context": context, "columns": list(data.columns)},
            )

    df = data.copy()
    units = pd.Index(sorted(df[unit].unique()))
    n_units = len(units)
    first = df.drop_duplicates(subset=[unit]).set_index(unit).reindex(units)
    unit_cohort = first[cohort].to_numpy()

    if distances is not None:
        dist = np.asarray(distances, dtype=float)
        if dist.shape != (n_units, n_units):
            raise MethodIncompatibility(
                f"{context}: `distances` must be ({n_units}, {n_units}), "
                f"got {dist.shape}.",
                diagnostics={"context": context, "shape": list(dist.shape)},
            )
    else:
        coords = list(coords)
        if len(coords) != 2:
            raise MethodIncompatibility(
                f"{context}: `coords` must name exactly two columns.",
                diagnostics={"context": context, "coords": coords},
            )
        for col in coords:
            if col not in df.columns:
                raise MethodIncompatibility(
                    f"{context}: coordinate column {col!r} not in data.",
                    diagnostics={"context": context, "columns": list(df.columns)},
                )
        dist = _pairwise_distances(first[coords].to_numpy(dtype=float))

    treated_mask = unit_cohort != never_value
    if not treated_mask.any():
        raise DataInsufficient(
            f"{context}: no treated units.",
            diagnostics={"context": context},
        )
    # Distance from each unit to the NEAREST treated unit. Treated units get
    # zero by construction and are handled separately.
    nearest = dist[:, treated_mask].min(axis=1)

    ring_of = np.full(n_units, -1, dtype=int)  # -1 = clean control
    untreated = ~treated_mask
    for r in range(len(edges) - 1):
        lo, hi = edges[r], edges[r + 1]
        in_ring = untreated & (nearest > lo) & (nearest <= hi)
        ring_of[in_ring] = r
    # Untreated units at exactly the innermost edge (usually distance 0 is
    # impossible for an untreated unit, but a zero-distance duplicate is)
    # belong to the first ring, not to the clean controls.
    ring_of[untreated & (nearest <= edges[0])] = 0

    clean = untreated & (ring_of == -1)
    n_clean = int(clean.sum())
    if n_clean == 0:
        raise DataInsufficient(
            f"{context}: no clean controls -- every untreated unit falls "
            "inside a spillover ring, so there is nothing to measure against. "
            "Narrow `ring_edges` or widen the study area.",
            diagnostics={
                "context": context,
                "n_units": n_units,
                "outer_edge": float(edges[-1]),
                "max_distance": (
                    float(nearest[untreated].max()) if untreated.any() else 0.0
                ),
            },
        )
    if n_clean < 30:
        warnings.warn(
            f"{context}: only {n_clean} clean control units. Every effect is "
            "measured against them, so the standard errors below are "
            "optimistic and the outermost ring may not be clean either.",
            UserWarning,
            stacklevel=2,
        )

    periods = sorted(df[time].unique())
    cohorts = [g for g in sorted(pd.unique(unit_cohort)) if g != never_value]
    y_wide = df.pivot_table(index=unit, columns=time, values=y).reindex(units)

    groups = {"direct": treated_mask}
    for r in range(len(edges) - 1):
        if (ring_of == r).any():
            groups[f"ring_{r + 1}"] = ring_of == r

    rows: List[Dict[str, Any]] = []
    psis: Dict[str, List[np.ndarray]] = {k: [] for k in groups}
    wts: Dict[str, List[float]] = {k: [] for k in groups}
    for g in cohorts:
        base = g - 1
        if base not in periods:
            continue
        for t in [p for p in periods if p >= g]:
            if t not in y_wide.columns or base not in y_wide.columns:
                continue
            dy = (y_wide[t] - y_wide[base]).to_numpy(dtype=float)
            ok = np.isfinite(dy)
            ctrl = clean & ok
            if ctrl.sum() < 2:
                continue
            c_mean = float(dy[ctrl].mean())
            for name, member in groups.items():
                # The direct group is cohort-specific; the rings are defined
                # by geography and contribute to every cell.
                sel = (
                    (member & (unit_cohort == g) & ok)
                    if name == "direct"
                    else (member & ok)
                )
                if sel.sum() < 2:
                    continue
                est = float(dy[sel].mean()) - c_mean
                psi = np.zeros(n_units, dtype=float)
                psi[sel] = (dy[sel] - dy[sel].mean()) * (n_units / sel.sum())
                psi[ctrl] -= (dy[ctrl] - c_mean) * (n_units / ctrl.sum())
                rows.append(
                    {
                        "group": name,
                        "cohort": g,
                        "time": t,
                        "estimate": est,
                        "n": int(sel.sum()),
                        "n_clean": int(ctrl.sum()),
                    }
                )
                psis[name].append(psi)
                wts[name].append(float(sel.sum()))

    if not rows:
        raise DataInsufficient(
            f"{context}: no estimable (cohort, period) cells. Each treated "
            "cohort needs a period before it and some clean controls.",
            diagnostics={"context": context, "cohorts": cohorts},
        )

    detail = pd.DataFrame(rows)

    def _combine(name: str) -> tuple:
        if not psis[name]:
            return np.nan, np.nan
        w = np.asarray(wts[name], dtype=float)
        w = w / w.sum()
        sub = detail[detail["group"] == name]
        est = float(np.sum(w * sub["estimate"].to_numpy()))
        psi = np.sum([wi * p for wi, p in zip(w, psis[name])], axis=0)
        return est, float(np.sqrt(np.mean(psi**2) / n_units))

    direct, direct_se = _combine("direct")
    ring_rows = []
    z = float(stats.norm.ppf(1 - alpha / 2))
    for r in range(len(edges) - 1):
        name = f"ring_{r + 1}"
        if name not in groups:
            continue
        est, se = _combine(name)
        ring_rows.append(
            {
                "ring": r + 1,
                "lower": float(edges[r]),
                "upper": float(edges[r + 1]),
                "n_units": int((ring_of == r).sum()),
                "estimate": est,
                "se": se,
                "ci_lower": est - z * se,
                "ci_upper": est + z * se,
            }
        )

    return SpilloverRingResult(
        direct=direct,
        direct_se=direct_se,
        rings=pd.DataFrame(ring_rows),
        detail=detail,
        n_units=n_units,
        n_clean_controls=n_clean,
        ring_edges=edges,
        alpha=float(alpha),
        diagnostics={
            "cohorts": cohorts,
            "n_treated": int(treated_mask.sum()),
            "distance_to_nearest_treated": {
                "min_untreated": (
                    float(nearest[untreated].min()) if untreated.any() else np.nan
                ),
                "max_untreated": (
                    float(nearest[untreated].max()) if untreated.any() else np.nan
                ),
            },
        },
    )
