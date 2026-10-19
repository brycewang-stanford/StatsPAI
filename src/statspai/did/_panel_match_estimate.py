"""Estimates, standard errors and diagnostics for ``sp.panel_match``.

Every estimate here is a weighted difference-in-differences averaged over
matched sets. For lead ``F`` and a set with treated unit ``i`` at ``t``,
controls ``c`` and weights ``w_c``,

    (Y[i, t+F] - Y[i, t-1]) - sum_c w_c (Y[c, t+F] - Y[c, t-1]),

averaged over the sets. Writing it as ``sum_{i,s} W[i, s] Y[i, s] / n_sets``
with a weight grid ``W`` gives per-unit sums ``A_i = sum_s W[i, s] Y[i, s]``
and per-unit set counts ``B_i``; the estimate is ``sum A / sum B``, and the
standard errors treat units as the sampling unit:

* ``'conditional'`` (on the number of sets): ``N_W var(A) / (sum B)^2``,
  ``N_W`` the number of units with a non-zero weight;
* ``'unconditional'``: the delta-method variance of the ratio of the two
  unit means;
* ``'bootstrap'``: units resampled with replacement (draws without any
  treated unit are redrawn), the estimate recomputed as a ratio of the
  resampled sums.

The placebo test reuses the same ratio with the outcome change taken over
the pre-treatment window, ``t-k`` to ``t-1``.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from ._panel_match_sets import Grid, MatchedSet

__all__ = [
    "unit_sums",
    "ratio_estimate",
    "analytic_se",
    "bootstrap_draws",
    "covariate_balance",
]


def unit_sums(
    sets: List[MatchedSet],
    Y: np.ndarray,
    start_offset: int,
    end_offsets: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-unit sums for the change from ``t+start_offset`` to ``t+end``.

    Returns ``A`` (N x len(end_offsets)), ``B`` (N,) and ``nonzero``
    (N x len(end_offsets)): whether the unit carries any weight.
    """
    N, P = Y.shape
    Y0 = np.where(np.isnan(Y), 0.0, Y)
    A = np.zeros((N, len(end_offsets)))
    nonzero = np.zeros((N, len(end_offsets)), dtype=bool)
    B = np.zeros(N)
    for s in sets:
        B[s.unit] += 1
    for k, off in enumerate(end_offsets):
        W = np.zeros((N, P))
        for s in sets:
            a, b = s.period + start_offset, s.period + off
            W[s.unit, a] -= 1.0
            W[s.unit, b] += 1.0
            w = s.weights
            nz = w != 0
            np.add.at(W, (s.controls[nz], a), w[nz])
            np.add.at(W, (s.controls[nz], b), -w[nz])
        A[:, k] = np.sum(W * Y0, axis=1)
        nonzero[:, k] = np.any(W != 0, axis=1)
    return A, B, nonzero


def ratio_estimate(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A.sum(axis=0) / B.sum()


def analytic_se(
    A: np.ndarray, B: np.ndarray, nonzero: np.ndarray, method: str
) -> np.ndarray:
    """Conditional or unconditional standard error of ``sum A / sum B``."""
    N = A.shape[0]
    varA = A.var(axis=0, ddof=1)
    if method == "conditional":
        return np.sqrt(nonzero.sum(axis=0) * varA / B.sum() ** 2)
    EB = B.mean() * N
    VB = B.var(ddof=1) * N
    VA = N * varA
    EA = N * A.mean(axis=0)
    covAB = np.array([np.cov(A[:, k], B, ddof=1)[0, 1] for k in range(A.shape[1])])
    var = (VA - 2 * (EA / EB) * covAB + (EA / EB) ** 2 * VB) / EB**2
    return np.sqrt(var)


def bootstrap_draws(
    parts: List[tuple],
    treated_units: List[np.ndarray],
    n_units: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit-bootstrap draws of the (possibly combined) ratio estimate.

    ``parts`` holds ``(A, B, sign)`` triples; one part is the ATT / ATC /
    ART, two parts are the ATE (combined by their set counts). A draw is
    redrawn until each part has at least one of its treated units.
    """
    out = np.empty((n_boot, parts[0][0].shape[1]))
    p = np.full(n_units, 1.0 / n_units)
    masks = []
    for units in treated_units:
        m = np.zeros(n_units, dtype=bool)
        m[units] = True
        masks.append(m)
    filled = 0
    while filled < n_boot:
        counts = rng.multinomial(n_units, p, size=n_boot - filled).astype(float)
        ok = np.ones(counts.shape[0], dtype=bool)
        for m in masks:
            ok &= counts[:, m].sum(axis=1) > 0
        counts = counts[ok]
        num = 0.0
        den = 0.0
        for A, B, sign in parts:
            nb = counts @ B
            num = num + sign * (counts @ A)
            den = den + nb
        out[filled : filled + counts.shape[0]] = num / den[:, None]
        filled += counts.shape[0]
    return out


def covariate_balance(
    sets: List[MatchedSet],
    grid: Grid,
    variables: Sequence[str],
    lag: int,
    refined: bool,
) -> pd.DataFrame:
    """Standardised mean difference, treated minus weighted controls, at
    each period ``t-lag .. t``; scaled by the SD of the treated units'
    values in that period across sets. Rows ``t-lag .. t-0``."""
    rows = {}
    for k in range(lag + 1):
        off = k - lag
        vals = {}
        for v in variables:
            X = grid.columns[v]
            treated = np.array([X[s.unit, s.period + off] for s in sets])
            sd = (
                np.nanstd(treated, ddof=1) if np.sum(~np.isnan(treated)) > 1 else np.nan
            )
            if not np.isfinite(sd) or sd <= 1.5e-8 * max(
                1.0, np.nanmax(np.abs(treated))
            ):
                vals[v] = np.nan
                continue
            diffs = []
            for s, x_t in zip(sets, treated):
                xc = X[s.controls, s.period + off]
                w = (
                    s.weights
                    if refined
                    else np.full(s.controls.size, 1.0 / s.controls.size)
                )
                diffs.append(x_t - np.nansum(w * xc))
            vals[v] = float(np.nanmean(np.asarray(diffs) / sd))
        rows[f"t-{lag - k}"] = vals
    return pd.DataFrame.from_dict(rows, orient="index")[list(variables)]
