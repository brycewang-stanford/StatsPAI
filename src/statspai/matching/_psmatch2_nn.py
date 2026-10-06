"""Stata ``psmatch2`` one-nearest-neighbour matching with ``ties`` / ``ate``.

``sp.match`` picks exactly one control per treated unit.  Two ``psmatch2``
options change which rows enter the matched sample, and PSM-DID papers
then regress on ``_weight != .``:

* ``ties`` -- every control at the minimal propensity-score distance is a
  match (each receives ``1/m`` of the treated unit), not just one of them.
  With a coarse propensity model (few covariates, repeated firm-years)
  duplicate scores are common and the matched sample roughly doubles.
* ``ate`` -- controls are matched to treated units as well.  Common
  support becomes two-sided, treated units *start* at weight 0 (instead of
  1) and only gain weight by serving as a control's match, so
  ``_weight != .`` marks the units used as a match in either direction.

Semantics follow the ``psmatch2`` 4.0.12 documentation and its observable
output (verified against Stata 18): the distance is ``|p_i - p_j|``; a
match requires distance ``< caliper``; off-support rows are neither
matched nor used; ``_weight`` is missing where it is zero or off support;
ATT, ATU and ATE are the ``_treated``-arm means of ``y - _y`` / ``_y - y``
and their ``N1 / (N0 + N1)`` mix; the default ATT standard error is
:func:`statspai.matching._matched_frame.psmatch2_se`.

Without ``ties`` a unit with several equidistant candidates is matched to
the first in the pool's (stable) propensity-score order; ``psmatch2``
takes the first in its own sort order, which it warns can be arbitrary.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import _matched_frame as _mf


def _nearest_sets(
    target: np.ndarray, pool: np.ndarray, ties: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Matches of each ``target`` score among ``pool`` scores.

    Returns ``(starts, stops, dmin, first)`` over the *sorted* pool order:
    target ``i`` is matched to sorted-pool positions ``starts[i]:stops[i]``
    (a run of duplicates on one side, or two runs joined when the nearest
    values below and above are exactly equidistant -- then the run spans
    both and the caller filters by distance), ``dmin[i]`` is the distance
    and ``first[i]`` the position used for ``_n1``.
    """
    n_pool = pool.size
    pos = np.searchsorted(pool, target, side="left")
    lo = np.clip(pos - 1, 0, n_pool - 1)
    hi = np.clip(pos, 0, n_pool - 1)
    dlo = np.where(pos > 0, np.abs(target - pool[lo]), np.inf)
    dhi = np.where(pos < n_pool, np.abs(pool[hi] - target), np.inf)
    dmin = np.minimum(dlo, dhi)
    use_lo = dlo <= dhi  # psmatch2's forward sweep settles on the lower one
    first = np.where(use_lo, lo, hi)
    if not ties:
        # First of the duplicate run in pool order.
        first = np.searchsorted(pool, pool[first], side="left")
        return first, first + 1, dmin, first
    lo_start = np.searchsorted(pool, pool[lo], side="left")
    lo_stop = np.searchsorted(pool, pool[lo], side="right")
    hi_start = np.searchsorted(pool, pool[hi], side="left")
    hi_stop = np.searchsorted(pool, pool[hi], side="right")
    take_lo = dlo == dmin
    take_hi = dhi == dmin
    starts = np.where(take_lo, lo_start, hi_start)
    stops = np.where(take_hi, hi_stop, lo_stop)
    first = np.where(take_lo, lo_start, hi_start)
    return starts, stops, dmin, first


def _match_arm(
    pscore: np.ndarray,
    targets: np.ndarray,
    pool: np.ndarray,
    *,
    ties: bool,
    caliper: float,
    weight: np.ndarray,
    y: Optional[np.ndarray],
    matched_y: np.ndarray,
    nn: np.ndarray,
    n1: np.ndarray,
    support: np.ndarray,
) -> None:
    """Match every ``targets`` row to ``pool`` rows, updating arrays in place."""
    if targets.size == 0 or pool.size == 0:
        support[targets] = False
        return
    order = np.argsort(pscore[pool], kind="stable")
    pool_sorted = pool[order]
    ps_sorted = pscore[pool_sorted]
    starts, stops, dmin, first = _nearest_sets(pscore[targets], ps_sorted, ties)
    ok = dmin < caliper
    support[targets[~ok]] = False
    for i in np.flatnonzero(ok):
        cand = pool_sorted[starts[i] : stops[i]]
        if ties and cand.size > 1:
            cand = cand[np.abs(pscore[targets[i]] - pscore[cand]) <= dmin[i]]
        m = cand.size
        weight[cand] += 1.0 / m
        nn[targets[i]] = m
        n1[targets[i]] = pool_sorted[first[i]]
        if y is not None:
            matched_y[targets[i]] = float(np.mean(y[cand]))


def psmatch2_nn(
    pscore: np.ndarray,
    treated: np.ndarray,
    y: Optional[np.ndarray],
    *,
    ties: bool,
    ate: bool,
    common_support: bool,
    caliper: Optional[float] = None,
    treated_range: bool = False,
) -> Dict[str, object]:
    """psmatch2 ``neighbor(1)`` matching with the ``ties`` / ``ate`` options.

    ``treated_range`` is the common support of Becker and Ichino's
    ``pscore`` / ``attnd``: every unit whose score lies outside the range
    of the treated scores is off support, so a control outside it cannot
    be a match.

    All arrays are positional over the estimation sample.  Returns the
    matched-frame columns (``_support``, ``_weight``, ``_n1``, ``_nn``,
    ``_pdif``, ``_y``) and, when ``y`` is given, ``att`` / ``se_att`` and
    (with ``ate``) ``atu`` / ``ate``.
    """
    p = np.asarray(pscore, dtype=float)
    t = np.asarray(treated).astype(int)
    n = p.size
    cal = np.inf if caliper is None else float(caliper)
    support = np.ones(n, dtype=bool)
    if common_support:
        support &= _mf.common_support_mask(p, t, rule="minmax")
        if ate:
            # psmatch2 ... ate common: controls outside the treated range too.
            lo, hi = p[t == 1].min(), p[t == 1].max()
            support[(t == 0) & ((p < lo) | (p > hi))] = False

    if treated_range:
        lo, hi = p[t == 1].min(), p[t == 1].max()
        support[(p < lo) | (p > hi)] = False

    weight = np.where(t == 1, 0.0 if ate else 1.0, 0.0)
    matched_y = np.full(n, np.nan)
    nn = np.zeros(n)
    n1 = np.full(n, -1, dtype=np.int64)
    yv = None if y is None else np.asarray(y, dtype=float)

    trt = np.flatnonzero((t == 1) & support)
    ctl = np.flatnonzero((t == 0) & support)
    if ate:
        # Controls matched to treated first (psmatch2's order); a control
        # without a match inside the caliper goes off support.
        _match_arm(
            p,
            ctl,
            trt,
            ties=ties,
            caliper=cal,
            weight=weight,
            y=yv,
            matched_y=matched_y,
            nn=nn,
            n1=n1,
            support=support,
        )
    _match_arm(
        p,
        trt,
        ctl,
        ties=ties,
        caliper=cal,
        weight=weight,
        y=yv,
        matched_y=matched_y,
        nn=nn,
        n1=n1,
        support=support,
    )
    weight = np.where((weight == 0) | ~support, np.nan, weight)

    obs_id = np.arange(1, n + 1, dtype=float)
    has_n1 = n1 >= 0
    n1_id = np.where(has_n1, obs_id[np.clip(n1, 0, None)], np.nan)
    pdif = np.where(has_n1, np.abs(p - p[np.clip(n1, 0, None)]), np.nan)
    arm_rows = (t == 1) | ate
    out: Dict[str, object] = {
        "support": support,
        "weight": weight,
        "n1": np.where(arm_rows, n1_id, np.nan),
        "nn": np.where(arm_rows, nn, 0.0),
        "pdif": np.where(arm_rows, pdif, np.nan),
        "matched_y": np.where(support & arm_rows, matched_y, np.nan),
    }
    if yv is not None:
        tr_on = (t == 1) & support
        att = float(np.mean(yv[tr_on] - matched_y[tr_on]))
        out["att"] = att
        out["se_att"] = _mf.psmatch2_se(yv, t, support, weight)
        out["n_treated_on_support"] = int(tr_on.sum())
        if ate:
            c_on = (t == 0) & support
            atu = float(np.mean(matched_y[c_on] - yv[c_on]))
            n1_, n0_ = int(tr_on.sum()), int(c_on.sum())
            out["atu"] = atu
            out["ate"] = (att * n1_ + atu * n0_) / (n1_ + n0_)
            out["n_controls_on_support"] = n0_
    return out
