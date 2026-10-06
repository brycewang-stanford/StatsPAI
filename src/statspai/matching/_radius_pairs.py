"""Radius matching with every treated-control pair counted once.

Becker and Ichino's ``attr``. The usual radius estimator (``psmatch2,
radius caliper(r)``) first averages the controls within the radius of each
treated unit and then averages those differences, so every matched treated
unit counts once. ``attr`` pools the pairs instead: a control is weighted
by the number of treated units it lies within the radius of, and the
weighted mean of the controls is taken from the mean of the matched
treated. A treated unit with many controls nearby therefore weighs more
than one with few. The two agree when every treated unit has the same
number of controls within the radius.

A pair is within the radius when the scores differ by strictly less than
it. The standard error is the one ``attr`` reports: the variance of the
treated outcomes over their number, plus the variance of the outcomes of
the controls that are used times the sum of their squared weights (scaled
to add up to the number of matched treated) over that number squared.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

__all__ = ["radius_pairs"]


def _within(sorted_scores: np.ndarray, at: np.ndarray, radius: float) -> np.ndarray:
    """How many of ``sorted_scores`` lie strictly within ``radius`` of each
    entry of ``at``."""
    lo = np.searchsorted(sorted_scores, at - radius, side="right")
    hi = np.searchsorted(sorted_scores, at + radius, side="left")
    return np.maximum(hi - lo, 0)


def radius_pairs(
    pscore: np.ndarray, treated: np.ndarray, y: np.ndarray, radius: float
) -> Dict[str, Any]:
    """The pooled-pairs radius estimate of the ATT.

    Returns ``att``, ``se``, ``weight`` (per row: 1 for a matched treated
    unit, the scaled pair count for a control that is used, ``nan``
    otherwise), ``n_treated`` and ``n_control``.
    """
    p = np.asarray(pscore, dtype=float)
    t = np.asarray(treated).astype(int)
    out = np.asarray(y, dtype=float)
    is_t, is_c = t == 1, t == 0
    pairs_of_control = _within(np.sort(p[is_t]), p[is_c], radius).astype(float)
    controls_of_treated = _within(np.sort(p[is_c]), p[is_t], radius)
    matched = controls_of_treated > 0
    n_t = int(matched.sum())
    weight = np.full(p.size, np.nan)
    if n_t == 0 or pairs_of_control.sum() == 0:
        return {"att": np.nan, "se": np.nan, "weight": weight, "n_treated": 0,
                "n_control": 0}  # fmt: skip
    y_t = out[is_t][matched]
    y_c = out[is_c]
    scaled = n_t * pairs_of_control / pairs_of_control.sum()
    used = pairs_of_control > 0
    att = float(y_t.mean() - np.sum(scaled * y_c) / np.sum(scaled))
    var_t = float(np.var(y_t, ddof=1)) if n_t > 1 else np.nan
    var_c = float(np.var(y_c[used], ddof=1)) if used.sum() > 1 else np.nan
    se = float(np.sqrt(var_t / n_t + np.sum(scaled**2) / n_t**2 * var_c))
    treated_rows = np.flatnonzero(is_t)
    weight[treated_rows[matched]] = 1.0
    control_rows = np.flatnonzero(is_c)
    weight[control_rows[used]] = scaled[used]
    return {"att": att, "se": se, "weight": weight, "n_treated": n_t,
            "n_control": int(used.sum())}  # fmt: skip
