"""Estimates, standard errors, balance and placebo for ``sp.panel_match``.

The DiD matching estimator (Imai, Kim and Wang 2023, eq. 18) is linear
in the outcome: with observation weights ``W*`` (eq. 23) it is
``sum_i A_i / sum_i B_i`` where ``A_i = sum_t W*_it Y_it`` and ``B_i`` is
the number of non-empty matched sets of unit ``i``. The standard errors
treat units as the independent draws:

* conditional: ``N* var(A_i) / B^2``;
* unconditional: the first-order Taylor (delta-method) variance of the
  ratio ``A / B``;
* bootstrap: units resampled with replacement, weights held fixed.

The moment conventions that reproduce PanelMatch 3.1.5 (black box, see
``docs/dev/modules/did.md``): ``var`` uses all ``N`` units of the
balanced panel with divisor ``N - 1``; ``N*`` counts the units that are
treated in a non-empty set or carry a positive control weight; in the
unconditional variance ``V(A) = N var(A_i)`` and ``V(B) = N var(B_i)``
but the covariance term is ``cov(A_i, B_i)`` without the factor ``N``.

Clean-room module: no PanelMatch source was consulted.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ._pmatch_sets import MatchedSet


def observation_weights(
    sets: List[MatchedSet], shape: Tuple[int, int], post: int, pre: int, sign: float
) -> np.ndarray:
    """``W*`` for the contrast ``Y[t+post] - Y[t+pre]`` (offsets relative
    to the treated period ``t``)."""
    W = np.zeros(shape)
    for s in sets:
        if s.empty:
            continue
        a, b = s.t + post, s.t + pre
        W[s.unit, a] += sign
        W[s.unit, b] -= sign
        np.add.at(W, (s.controls, a), -sign * s.weights)
        np.add.at(W, (s.controls, b), sign * s.weights)
    return W


def unit_sums(W: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.where(W != 0, W * np.nan_to_num(Y), 0.0).sum(axis=1)


def set_counts(sets: List[MatchedSet], n_units: int) -> np.ndarray:
    B = np.zeros(n_units)
    for s in sets:
        if not s.empty:
            B[s.unit] += 1
    return B


def contributing_units(sets: List[MatchedSet], n_units: int) -> np.ndarray:
    used = np.zeros(n_units, dtype=bool)
    for s in sets:
        if s.empty:
            continue
        used[s.unit] = True
        used[s.controls[s.weights > 0]] = True
    return used


def analytic_se(A: np.ndarray, B: np.ndarray, n_star: int, method: str) -> float:
    N = A.size
    if method == "conditional":
        return float(np.sqrt(n_star * np.var(A, ddof=1)) / B.sum())
    EA, EB = A.sum(), B.sum()
    VA = N * np.var(A, ddof=1)
    VB = N * np.var(B, ddof=1)
    C = np.cov(A, B, ddof=1)[0, 1]
    r = EA / EB
    v = (VA - 2.0 * r * C + r * r * VB) / EB**2
    return float(np.sqrt(max(v, 0.0)))


def bootstrap_draws(
    A: np.ndarray, B: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot, n_contrasts) ratios over unit resamples; ``A`` is
    (N, n_contrasts)."""
    N = B.size
    counts = rng.multinomial(N, np.full(N, 1.0 / N), size=n_boot).astype(float)
    den = counts @ B
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (counts @ A) / den[:, None]
    draws[den == 0] = np.nan
    return draws


class Family:
    """Matched sets of one direction with the sign of their contrast."""

    def __init__(self, name: str, sets: List[MatchedSet]):
        self.name = name
        self.sets = sets
        self.sign = -1.0 if name == "atc" else 1.0


def contrasts(
    families: Sequence[Family],
    Y: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Unit sums ``A`` (N, n_pairs), set counts ``B`` (N,) and ``N*``."""
    N = Y.shape[0]
    A = np.zeros((N, len(pairs)))
    B = np.zeros(N)
    used = np.zeros(N, dtype=bool)
    for j, (post, pre) in enumerate(pairs):
        W = np.zeros(Y.shape)
        for fam in families:
            W += observation_weights(fam.sets, Y.shape, post, pre, fam.sign)
        A[:, j] = unit_sums(W, Y)
    for fam in families:
        B += set_counts(fam.sets, N)
        used |= contributing_units(fam.sets, N)
    return A, B, int(used.sum())


def inference_table(
    labels: Sequence,
    A: np.ndarray,
    B: np.ndarray,
    n_star: int,
    se_method: str,
    alpha: float,
    draws: Optional[np.ndarray],
) -> pd.DataFrame:
    est = A.sum(axis=0) / B.sum()
    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for j, lab in enumerate(labels):
        if se_method == "bootstrap":
            d = draws[:, j]
            d = d[np.isfinite(d)]
            se = float(np.std(d, ddof=1))
            lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
        else:
            se = analytic_se(A[:, j], B, n_star, se_method)
            lo, hi = est[j] - z * se, est[j] + z * se
        p = float(2 * stats.norm.sf(abs(est[j]) / se)) if se > 0 else np.nan
        rows.append((lab, float(est[j]), se, float(lo), float(hi), p))
    return pd.DataFrame(
        rows, columns=["term", "estimate", "se", "ci_lower", "ci_upper", "pvalue"]
    )


def pooled_summary(
    A: np.ndarray,
    B: np.ndarray,
    n_star: int,
    se_method: str,
    alpha: float,
    draws: Optional[np.ndarray],
) -> Tuple[float, float, Tuple[float, float], float]:
    """Average over the contrasts: the same linear estimator with the
    averaged observation weights."""
    a = A.mean(axis=1)
    est = float(a.sum() / B.sum())
    if se_method == "bootstrap":
        d = np.nanmean(draws, axis=1) if draws.shape[1] > 1 else draws[:, 0]
        d = d[np.isfinite(d)]
        se = float(np.std(d, ddof=1))
        ci = tuple(float(v) for v in np.quantile(d, [alpha / 2, 1 - alpha / 2]))
    else:
        se = analytic_se(a, B, n_star, se_method)
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (est - z * se, est + z * se)
    p = float(2 * stats.norm.sf(abs(est) / se)) if se > 0 else np.nan
    return est, se, ci, p


def run_bootstrap(
    A: np.ndarray, B: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    draws = bootstrap_draws(A, B, n_boot, rng)
    bad = int(np.isnan(draws[:, 0]).sum())
    if bad:
        warnings.warn(
            f"panel_match: {bad} of {n_boot} bootstrap resamples contain no "
            "treated observation with a matched set and are left out of the "
            "standard error.",
            UserWarning,
            stacklevel=4,
        )
    return draws


# ----------------------------------------------------------------------
# covariate balance (Imai, Kim and Wang 2023, eqs. 19-20)
# ----------------------------------------------------------------------


def balance_table(
    sets: List[MatchedSet],
    variables: Dict[str, np.ndarray],
    lag: int,
    refined: bool,
) -> pd.DataFrame:
    """Mean standardised difference between each treated observation and
    its (weighted) controls at t-L .. t. The scale is the standard
    deviation of the variable over the treated observations in that
    period offset. A missing control value adds nothing to the weighted
    control sum and a missing treated value drops that set (PanelMatch
    3.1.5 conventions, black box)."""
    live = [s for s in sets if not s.empty]
    rows = {}
    for name, V in variables.items():
        col = []
        for ell in range(lag, -1, -1):
            if not live:
                col.append(np.nan)
                continue
            tv = np.array([V[s.unit, s.t - ell] for s in live])
            sd = np.nanstd(tv, ddof=1) if np.isfinite(tv).sum() > 1 else np.nan
            diffs = np.empty(len(live))
            for k, s in enumerate(live):
                vc = V[s.controls, s.t - ell]
                w = s.weights if refined else np.full(vc.size, 1.0 / vc.size)
                ctrl = np.where(np.isnan(vc), 0.0, vc) @ w
                diffs[k] = (tv[k] - ctrl) / sd if sd and np.isfinite(sd) else np.nan
            col.append(np.nanmean(diffs) if np.isfinite(diffs).any() else np.nan)
        rows[name] = col
    index = [f"t-{ell}" for ell in range(lag, -1, -1)]
    return pd.DataFrame(rows, index=index)
