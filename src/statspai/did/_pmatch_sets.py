"""Matched sets on treatment history for ``sp.panel_match``.

For every "treated" observation (i, t) the matched set holds the other
units observed in the control state at t whose treatment history over
t-L .. t-1 is identical to unit i's (Imai, Kim and Wang 2023, eq. 11).

* ``att``: i switches 0 -> 1 at t; controls are 0 at t.
* ``art``: i switches 1 -> 0 at t; controls are 1 at t.
* ``atc``: i is 0 at t-1 and t; controls are 1 at t (the effect of
  treatment on the observations that stay out).

Clean-room module: written from the paper and the PanelMatch
documentation; no PanelMatch source was consulted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ._pmatch_prep import PMPanel

# (state of the treated observation at t-1, at t, state of controls at t)
_STATES = {"att": (0.0, 1.0, 0.0), "art": (1.0, 0.0, 1.0), "atc": (0.0, 0.0, 1.0)}


@dataclass
class MatchedSet:
    unit: int  # row index of the treated unit
    t: int  # period index of the treated observation
    controls: np.ndarray
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def empty(self) -> bool:
        return self.controls.size == 0


def _same_history(H: np.ndarray, h: np.ndarray, match_missing: bool) -> np.ndarray:
    """Rows of H equal to h (missing equals missing when match_missing)."""
    hn = np.isnan(h)
    if hn.any() and not match_missing:
        return np.zeros(H.shape[0], dtype=bool)
    Hn = np.isnan(H)
    eq = (H == h) | (Hn & hn)
    if not match_missing:
        eq &= ~Hn
    return eq.all(axis=1)


def build_sets(
    panel: PMPanel,
    *,
    kind: str,
    lag: int,
    leads: Sequence[int],
    match_missing: bool,
    exact: Optional[List[np.ndarray]] = None,
    restrict_control_period: Optional[int] = None,
    forbid_treatment_reversal: bool = False,
    outcome_history: bool = False,
) -> List[MatchedSet]:
    """Matched sets (unweighted) in (period, unit) order.

    Order of the screens (each reproduces PanelMatch 3.1.5 on our
    fixtures): treated observations failing ``restrict_control_period``
    are dropped; a set that is empty on treatment history alone is kept
    as an unmatched treated observation; otherwise the set is dropped when
    the treated unit lacks the outcome at t-1 or at a lead (or reverses
    treatment under ``forbid_treatment_reversal``), controls lacking those
    outcomes (or reversing) are removed and the set is dropped if none is
    left; exact matching comes last and a set it empties is kept as
    unmatched.

    ``outcome_history`` additionally requires the outcome to be observed
    over t-L .. t-1 (the placebo-test sets).
    """
    D, Y = panel.D, panel.Y
    N, T = D.shape
    prev_state, now_state, ctrl_state = _STATES[kind]
    max_lead = max(leads)
    lead_cols = sorted(set(int(f) for f in leads))
    out: List[MatchedSet] = []
    all_units = np.arange(N)
    for t in range(lag, T):
        treated = np.flatnonzero((D[:, t] == now_state) & (D[:, t - 1] == prev_state))
        if restrict_control_period:
            r = int(restrict_control_period)
            lo = max(t - r, 0)
            ok = np.all(D[treated, lo:t] == prev_state, axis=1) & (t - r >= 0)
            treated = treated[ok]
        if treated.size == 0:
            continue
        H = D[:, t - lag : t]
        is_ctrl = D[:, t] == ctrl_state
        for i in treated:
            cand = is_ctrl & _same_history(H, H[i], match_missing)
            cand[i] = False
            ctrl = all_units[cand]
            out.append(MatchedSet(int(i), t, ctrl))
    sets: List[MatchedSet] = []
    for s in out:
        if s.empty:
            sets.append(s)
            continue
        i, t = s.unit, s.t
        cols = [t + f for f in lead_cols]
        if cols[-1] >= T:
            continue
        need = [t - 1] + cols
        if outcome_history:
            need = list(range(t - lag, t)) + cols
        if np.isnan(Y[i, need]).any():
            continue
        if forbid_treatment_reversal:
            if not np.all(D[i, t : t + max_lead + 1] == now_state):
                continue
        ctrl = s.controls
        keep = ~np.isnan(Y[np.ix_(ctrl, need)]).any(axis=1)
        if forbid_treatment_reversal:
            keep &= np.all(
                D[np.ix_(ctrl, range(t, t + max_lead + 1))] == ctrl_state, axis=1
            )
        ctrl = ctrl[keep]
        if ctrl.size == 0:
            continue
        if exact:
            # after the outcome / reversal screens: a set emptied here is
            # kept as an unmatched treated observation
            for v in exact:
                win = v[:, t - lag : t + 1]
                ctrl = ctrl[np.all(win[ctrl] == win[i], axis=1)]
        sets.append(MatchedSet(i, t, ctrl))
    sets.sort(key=lambda s: (s.t, s.unit))
    return sets
