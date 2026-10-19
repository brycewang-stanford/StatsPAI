"""Matched sets for ``sp.panel_match`` (Imai, Kim and Wang 2023).

A treated observation is a unit-period ``(i, t)`` whose treatment switches
(0 at ``t - 1``, 1 at ``t`` for the ATT). Its matched set is every other
unit with the same treatment history over ``t - L, ..., t - 1`` that is
untreated at ``t``. Units and periods are held as positions in a balanced
``(units x periods)`` grid; a unit-period absent from the input is a row of
missing values, as in R ``PanelMatch`` (``PanelData`` balances the panel).

This module builds the sets and applies the screening rules that remove
treated observations or controls (outcome data over the lead window,
treatment reversal, exact matching, the placebo window). Refinement weights
are in :mod:`._panel_match_refine`.

Written from the paper and from the behaviour of R ``PanelMatch`` 3.1.5
observed through its outputs (the reference fixtures); the R package is
GPL-3 and no code is taken from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..exceptions import DataInsufficient

__all__ = ["Grid", "MatchedSet", "build_matched_sets"]


@dataclass
class Grid:
    """A balanced panel: ``units x periods`` arrays, NaN where missing."""

    unit_ids: np.ndarray  # sorted unit labels, one per row
    periods: np.ndarray  # period labels, one per column, consecutive
    treat: np.ndarray  # (N, P) float, NaN missing
    outcome: np.ndarray  # (N, P) float, NaN missing
    columns: Dict[str, np.ndarray] = field(default_factory=dict)  # (N, P)


@dataclass
class MatchedSet:
    """One treated observation and its controls (row positions)."""

    unit: int  # row of the treated unit
    period: int  # column of the treatment period t
    controls: np.ndarray  # rows of the control units, ascending
    weights: Optional[np.ndarray] = None  # refinement weights, sum to 1

    @property
    def empty(self) -> bool:
        return self.controls.size == 0


def _switch_events(D: np.ndarray, lag: int, qoi: str) -> List[tuple]:
    """Treated observations: switches into treatment (ATT / ART on the
    flipped indicator) or stays out of it (ATC), with ``lag`` observed
    periods before them inside the panel."""
    N, P = D.shape
    events = []
    for i in range(N):
        for t in range(max(lag, 1), P):
            prev, now = D[i, t - 1], D[i, t]
            if np.isnan(prev) or np.isnan(now) or prev != 0:
                continue
            if (qoi == "atc" and now == 0) or (qoi != "atc" and now == 1):
                events.append((i, t))
    return events


def _history_matches(
    D: np.ndarray,
    i: int,
    t: int,
    lag: int,
    target: float,
    match_missing: bool,
) -> np.ndarray:
    """Rows other than ``i`` whose treatment over ``t-lag..t-1`` equals
    unit ``i``'s and whose treatment at ``t`` equals ``target``."""
    hist = D[:, t - lag : t + 1].copy()
    if match_missing:
        hist = np.where(np.isnan(hist), -1.0, hist)
    want = hist[i].copy()
    want[-1] = target
    with np.errstate(invalid="ignore"):
        same = np.all(hist == want[None, :], axis=1)
    same[i] = False
    return np.flatnonzero(same)


def _complete(values: np.ndarray, row: int, lo: int, hi: int) -> bool:
    """Whether ``values[row, lo..hi]`` lies inside the panel and is observed."""
    P = values.shape[1]
    if lo < 0 or hi >= P:
        return False
    return not bool(np.isnan(values[row, lo : hi + 1]).any())


def _require_outcome(
    sets: List[MatchedSet], Y: np.ndarray, lo_off: int, hi_off: int
) -> List[MatchedSet]:
    """Keep treated observations and controls observed on ``t+lo..t+hi``.

    A treated observation without the data is dropped with its set; a set
    whose controls all lack it is dropped too (it would contribute nothing
    but a denominator count).
    """
    kept = []
    for s in sets:
        if not _complete(Y, s.unit, s.period + lo_off, s.period + hi_off):
            continue
        ok = [
            c
            for c in s.controls
            if _complete(Y, c, s.period + lo_off, s.period + hi_off)
        ]
        if ok:
            kept.append(MatchedSet(s.unit, s.period, np.asarray(ok, dtype=int)))
    return kept


def _forbid_reversal(
    sets: List[MatchedSet], D: np.ndarray, max_lead: int
) -> List[MatchedSet]:
    """Treated units stay treated through ``t + max_lead`` and controls stay
    untreated over ``t - 1 .. t + max_lead``."""
    P = D.shape[1]
    kept = []
    for s in sets:
        t, hi = s.period, s.period + max_lead
        if hi >= P:
            continue
        row = D[s.unit]
        if not (row[t - 1] == 0 and np.all(row[t : hi + 1] == 1)):
            continue
        block = D[s.controls, t - 1 : hi + 1]
        with np.errstate(invalid="ignore"):
            good = np.all(block == 0, axis=1)
        if good.any():
            kept.append(MatchedSet(s.unit, s.period, s.controls[good]))
    return kept


def _exact_match(
    sets: List[MatchedSet], columns: Dict[str, np.ndarray], lag: int
) -> List[MatchedSet]:
    """Controls whose exact-match variables equal the treated unit's in
    every period ``t - lag .. t``; missing values never match."""
    out = []
    for s in sets:
        lo, hi = s.period - lag, s.period
        keep = np.ones(s.controls.size, dtype=bool)
        for values in columns.values():
            ref = values[s.unit, lo : hi + 1]
            block = values[s.controls, lo : hi + 1]
            with np.errstate(invalid="ignore"):
                keep &= np.all(block == ref[None, :], axis=1) & ~np.isnan(ref).any()
        out.append(MatchedSet(s.unit, s.period, s.controls[keep]))
    return out


def build_matched_sets(
    grid: Grid,
    *,
    lag: int,
    max_lead: int,
    qoi: str,
    match_missing: bool,
    forbid_treatment_reversal: bool,
    restrict_control_period: Optional[int],
    exact_columns: Dict[str, np.ndarray],
    placebo: bool,
) -> tuple[List[MatchedSet], List[MatchedSet]]:
    """Matched sets before refinement.

    Returns ``(sets, empty)``: the sets with at least one control, and the
    treated observations whose history no control shares (reported, not
    used in estimation). ``qoi='art'`` must be passed with the treatment
    already flipped.
    """
    D, Y = grid.treat, grid.outcome
    events = _switch_events(D, lag, qoi)
    if restrict_control_period is not None:
        r = restrict_control_period
        D_hist = np.where(np.isnan(D), -1.0, D) if match_missing else D
        events = [(i, t) for (i, t) in events if np.all(D_hist[i, t - r : t] == 0)]
    if not events:
        raise DataInsufficient(
            f"panel_match: no treated observations for qoi={qoi!r} with "
            f"{lag} observed periods before them.",
            recovery_hint=(
                "Lower lag=, or check that the treatment switches inside the "
                "panel (from 0 to 1 for the ATT, from 1 to 0 for the ART)."
            ),
            diagnostics={"lag": lag, "qoi": qoi},
        )
    target = 1.0 if qoi == "atc" else 0.0
    all_sets = [
        MatchedSet(i, t, _history_matches(D, i, t, lag, target, match_missing))
        for (i, t) in events
    ]
    empty = [s for s in all_sets if s.empty]
    sets = [s for s in all_sets if not s.empty]
    if not sets:
        return [], empty

    sets = _require_outcome(sets, Y, -1, max_lead)
    if not sets:
        raise DataInsufficient(
            "panel_match: no treated observation has outcome data at t-1 "
            "and through the last lead for itself and at least one control.",
            recovery_hint="Use fewer leads, or a panel with fewer gaps in the outcome.",
        )
    if placebo:
        sets = _require_outcome(sets, Y, -lag, -1)
        if not sets:
            raise DataInsufficient(
                "panel_match: the placebo test needs the outcome over the "
                "whole lag window; no matched set has it.",
                recovery_hint="Lower lag= or drop placebo_test=True.",
            )
    if forbid_treatment_reversal:
        sets = _forbid_reversal(sets, D, max_lead)
        if not sets:
            raise DataInsufficient(
                "panel_match: with forbid_treatment_reversal=True no treated "
                "observation stays treated (and keeps a control that stays "
                "untreated) through the last lead.",
                recovery_hint="Use fewer leads or forbid_treatment_reversal=False.",
            )
    if exact_columns:
        matched = _exact_match(sets, exact_columns, lag)
        empty = empty + [s for s in matched if s.empty]
        sets = [s for s in matched if not s.empty]
    return sets, empty
