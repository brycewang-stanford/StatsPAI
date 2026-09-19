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

Staggered adoption. A ring unit is exposed from the first period in
which a treated unit lies within the outer ring edge, so each ring is
split by exposure onset and each onset cohort is compared with the clean
controls from its own base period -- the timing rule of Butts's own
staggered applications, where the spillover indicator switches on at the
first year a treated unit is within the distance band. Up to StatsPAI
1.28.0 every ring unit entered every treated cohort's cell regardless of
when it was exposed, which biased the ring effects under staggered
adoption (single-cohort designs were unaffected).

Validation
----------
No package implements this estimator, but every piece has a reference:

* Single treatment cohort: the direct and ring effects are the
  coefficients of the first-difference regression
  ``dY ~ treat + ring_1 + ... + ring_R`` that Butts's replication code
  runs with ``fixest::feols``, and the standard errors are its
  heteroskedasticity-robust HC0 errors.
* Any design: each group (direct, ring r) equals
  ``did::att_gt(control_group = "nevertreated")`` +
  ``did::aggte(type = "simple")`` on that group plus the clean controls,
  with ring units' cohort set to their exposure onset -- point estimates
  and analytic standard errors, including the cohort-share weight term.

The ring construction itself (distance to the nearest treated unit, bands,
onsets) is recomputed independently in the R generator. See
``tests/reference_parity/test_did_synth_misc_parity.py``; the known-truth
recovery tests are in ``tests/reference_parity/test_spillover_rings.py``.

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
    group means (``did``'s analytic SE), aggregated across
    ``(cohort, period)`` cells so the shared control units are accounted
    for, plus the influence function of the cohort-share weights (zero
    with a single cohort). Ring units are grouped by exposure onset; see
    the module docstring for the staggered-adoption rule and for the
    ``did`` / ``fixest`` references the numbers are pinned to.

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

    # Exposure onset of each ring unit: the earliest cohort among treated
    # units within the outer ring edge. A ring unit is exposed only from
    # then on, so under staggered adoption it is its own "cohort" in the
    # ring comparison (Butts's staggered design turns the spillover
    # indicator on at the first period a treated unit lies within the
    # distance band).
    outer = edges[-1]
    treated_cohorts = unit_cohort[treated_mask]
    within = dist[:, treated_mask] <= outer
    onset = np.full(n_units, never_value, dtype=object)
    for i in np.flatnonzero(untreated & (ring_of >= 0)):
        onset[i] = min(treated_cohorts[within[i]])
    # Units whose ring deepens later (a closer unit is treated after the
    # first exposure) are assigned their final ring from the onset on.
    first_ring = np.full(n_units, -1, dtype=int)
    for i in np.flatnonzero(untreated & (ring_of >= 0)):
        d_first = dist[i, treated_mask][treated_cohorts == onset[i]].min()
        # First band whose outer edge covers the first-exposure distance.
        first_ring[i] = int(np.searchsorted(edges[1:], d_first, side="left"))
    n_ring_changes = int(np.sum((ring_of >= 0) & (first_ring != ring_of)))
    if n_ring_changes:
        warnings.warn(
            f"{context}: {n_ring_changes} ring unit(s) move to a closer ring "
            "after their first exposure (a nearer unit is treated later). "
            "They are counted in their final ring from their first exposure "
            "on, which mixes two exposure levels in the early periods.",
            UserWarning,
            stacklevel=2,
        )

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
    cell_members: Dict[str, List[np.ndarray]] = {k: [] for k in groups}
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
                # The direct group is the treated cohort g; a ring group is
                # the ring's units whose exposure began at g. Both are
                # measured against clean controls from base period g - 1.
                sel = (
                    (member & (unit_cohort == g) & ok)
                    if name == "direct"
                    else (member & (onset == g) & ok)
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
                cell_members[name].append(
                    (member & (unit_cohort == g))
                    if name == "direct"
                    else (member & (onset == g))
                )

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
        att = sub["estimate"].to_numpy()
        est = float(np.sum(w * att))
        psi = np.sum([wi * p for wi, p in zip(w, psis[name])], axis=0)
        # The weights are estimated cohort shares (cells weighted by their
        # group size, as in did::aggte(type = "simple")); add their
        # influence function. It is identically zero with one cohort.
        ind = np.column_stack(cell_members[name]).astype(float)
        p = ind.mean(axis=0)
        wif = (ind - p) / p.sum() - np.sum(ind - p, axis=1, keepdims=True) * (
            p / p.sum() ** 2
        )[None, :]
        psi = psi + wif @ att
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
            "n_ring_changes": n_ring_changes,
            "ring_onsets": {
                f"ring_{r + 1}": {
                    str(h): int(np.sum((ring_of == r) & (onset == h)))
                    for h in cohorts
                    if np.any((ring_of == r) & (onset == h))
                }
                for r in range(len(edges) - 1)
            },
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
