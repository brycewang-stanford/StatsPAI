"""Shared arithmetic for Rosenbaum-type sensitivity analyses of sum statistics.

A stratum holds ``N`` units, ``n`` of them treated, each with a fixed score
``q``. The statistic is the sum of the treated scores. Under the sensitivity
model with parameter ``Gamma`` the chance of a treatment assignment ``z`` in
the stratum is proportional to ``exp(log(Gamma) * z'u)`` for an unobserved
``u`` in ``[0, 1]^N``. Rosenbaum and Krieger [@rosenbaum1990sensitivity]
showed that the expectation of the statistic is largest at a ``u`` that is
``1`` for the ``m`` largest scores and ``0`` for the rest, for one of
``m = 1, ..., N - 1``.

For such a ``u`` the number ``A`` of treated units among the top ``m`` has
Fisher's noncentral hypergeometric distribution, and given ``A = a`` the
treated scores are two simple random samples without replacement, ``a`` from
the top ``m`` and ``n - a`` from the bottom ``N - m``. The moments follow from
the two-stage decomposition and are computed here exactly, with no
large-sample approximation to the hypergeometric part.

With several strata the worst ``u`` is found stratum by stratum
[@gastwirth2000asymptotic]: the *separable* approximation takes the largest
expectation in each stratum, and among equal expectations the largest
variance. Rosenbaum [@rosenbaum2018sensitivity] added a bound that does not
rely on that approximation: because the square root is concave,
``sqrt(V) <= sqrt(V0) + (V - V0) / (2 sqrt(V0))`` for any ``V0``, so the
critical value ``E + kappa sqrt(V)`` is bounded by a quantity that is a sum
over strata and can again be maximised one stratum at a time.

Nothing in this module is public; :mod:`statspai.diagnostics.weighted_rank`
and :mod:`statspai.diagnostics.rosenbaum_strata` build on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import special, stats

#: Relative tolerance below which two expectations count as tied. Exact
#: equality would let rounding noise pick the variance when ``Gamma = 1``,
#: where every ``m`` has the same expectation.
_TIE_RTOL = 1e-12


def fnch_moments(
    n_top: np.ndarray, n_total: int, n_drawn: int, gamma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """First two moments of Fisher's noncentral hypergeometric distribution.

    ``A`` counts the treated units that fall among ``n_top`` marked units when
    ``n_drawn`` of ``n_total`` are treated and the odds of treatment are
    ``gamma`` times higher for a marked unit. Returns ``E[A]`` and ``E[A^2]``
    for every entry of ``n_top``; the probabilities are normalised in logs,
    so the result is exact to rounding for any ``gamma``.
    """
    m = np.asarray(n_top, dtype=float).reshape(-1, 1)
    a = np.arange(n_drawn + 1, dtype=float).reshape(1, -1)
    valid = (a <= m) & (n_drawn - a <= n_total - m)
    with np.errstate(invalid="ignore", divide="ignore"):
        logw = (
            special.gammaln(m + 1)
            - special.gammaln(a + 1)
            - special.gammaln(np.where(valid, m - a, 0.0) + 1)
            + special.gammaln(n_total - m + 1)
            - special.gammaln(n_drawn - a + 1)
            - special.gammaln(np.where(valid, n_total - m - n_drawn + a, 0.0) + 1)
            + a * np.log(gamma)
        )
    logw = np.where(valid, logw, -np.inf)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(axis=1, keepdims=True)
    return (w * a).sum(axis=1), (w * a * a).sum(axis=1)


def moment_tables(
    sorted_scores: np.ndarray, n_treated: int, gamma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Expectation and variance of the treated score sum for every ``m``.

    Parameters
    ----------
    sorted_scores : array, shape (S, N)
        Scores of ``S`` strata that all hold ``N`` units, each row sorted in
        increasing order.
    n_treated : int
        Treated units per stratum, ``1 <= n_treated <= N - 1``.
    gamma : float
        Sensitivity parameter, ``>= 1``.

    Returns
    -------
    mu, nu : arrays, shape (S, N - 1)
        Column ``m - 1`` is the expectation (variance) when the ``m`` largest
        scores have ``u = 1``.
    """
    q = np.asarray(sorted_scores, dtype=float)
    n_strata, size = q.shape
    m = np.arange(1, size)
    n_bottom = size - m

    # Means and sample variances of the bottom N - m and top m scores, from
    # cumulative sums of scores centred per stratum (centring keeps the
    # sums of squares from cancelling when the scores are large).
    centre = q.mean(axis=1, keepdims=True)
    qc = q - centre
    csum = np.cumsum(qc, axis=1)
    csum2 = np.cumsum(qc * qc, axis=1)
    sum0 = csum[:, n_bottom - 1]
    sum1 = csum[:, -1:] - sum0
    ss0 = csum2[:, n_bottom - 1]
    ss1 = csum2[:, -1:] - ss0
    mean0 = sum0 / n_bottom
    mean1 = sum1 / m
    with np.errstate(invalid="ignore", divide="ignore"):
        var0 = np.where(n_bottom > 1, (ss0 - n_bottom * mean0**2) / (n_bottom - 1), 0.0)
        var1 = np.where(m > 1, (ss1 - m * mean1**2) / (m - 1), 0.0)
    var0 = np.maximum(var0, 0.0)
    var1 = np.maximum(var1, 0.0)

    ea, ea2 = fnch_moments(m, size, n_treated, gamma)
    va = np.maximum(ea2 - ea**2, 0.0)
    eb = n_treated - ea
    eb2 = n_treated**2 - 2.0 * n_treated * ea + ea2

    mu = n_treated * centre + ea * mean1 + eb * mean0
    nu = (
        va * (mean1 - mean0) ** 2 + var1 * (ea - ea2 / m) + var0 * (eb - eb2 / n_bottom)
    )
    return mu, np.maximum(nu, 0.0)


def separable_choice(mu: np.ndarray, nu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Largest expectation per stratum and, among ties, the largest variance."""
    top = mu.max(axis=1, keepdims=True)
    scale = np.maximum(np.abs(top), 1.0)
    tied = mu >= top - _TIE_RTOL * scale
    return top[:, 0], np.where(tied, nu, -np.inf).max(axis=1)


@dataclass
class _Group:
    """Strata that share a size and a treated count."""

    weights: np.ndarray  # (S,) non-negative multiplier of the scores
    mu: np.ndarray  # (S, N - 1) moments of the unweighted scores
    nu: np.ndarray


@dataclass
class BoundPieces:
    """Bounds on the null distribution of a sum statistic at one ``Gamma``."""

    expectation: float
    variance: float
    expectation_taylor: float
    variance_taylor: float

    def deviate(self, statistic: float, taylor: bool = False) -> float:
        e = self.expectation_taylor if taylor else self.expectation
        v = self.variance_taylor if taylor else self.variance
        if v <= 0:
            if statistic == e:
                return 0.0
            return float(np.sign(statistic - e) * np.inf)
        return float((statistic - e) / np.sqrt(v))


def combine(groups: Sequence[_Group], kappa: float) -> BoundPieces:
    """Sum the per-stratum choices into the two bounds.

    ``weights`` multiply the scores of a stratum, so its expectation scales
    with the weight and its variance with the square. The separable choice
    does not depend on a positive weight; the Taylor choice does.
    """
    e_sep = v_sep = 0.0
    chosen: List[Tuple[np.ndarray, np.ndarray]] = []
    for g in groups:
        e_i, v_i = separable_choice(g.mu, g.nu)
        chosen.append((e_i, v_i))
        e_sep += float(np.sum(g.weights * e_i))
        v_sep += float(np.sum(g.weights**2 * v_i))
    if v_sep <= 0:
        return BoundPieces(e_sep, v_sep, e_sep, v_sep)
    half = kappa / (2.0 * np.sqrt(v_sep))
    e_lin = v_lin = 0.0
    for g, (e_i, v_i) in zip(groups, chosen):
        w = g.weights[:, None]
        gain = w * (g.mu - e_i[:, None]) + half * w**2 * (g.nu - v_i[:, None])
        pick = np.argmax(gain, axis=1)
        rows = np.arange(len(pick))
        e_lin += float(np.sum(g.weights * g.mu[rows, pick]))
        v_lin += float(np.sum(g.weights**2 * g.nu[rows, pick]))
    return BoundPieces(e_sep, v_sep, e_lin, v_lin)


def group_strata(
    scores: np.ndarray, treated: np.ndarray, strata: np.ndarray
) -> Tuple[Dict[Tuple[int, int], np.ndarray], float, int, int, int]:
    """Sort scores within strata and stack strata of equal shape.

    Strata with no treated unit or no control carry no information about
    the assignment and are dropped. Returns the stacked sorted scores keyed
    by ``(size, n_treated)``, the observed statistic, and the numbers of
    strata, treated units and controls that were kept.
    """
    codes, inverse = np.unique(strata, return_inverse=True)
    order = np.lexsort((scores, inverse))
    inv_sorted = inverse[order]
    sc_sorted = scores[order]
    z_sorted = treated[order]
    sizes = np.bincount(inv_sorted, minlength=len(codes))
    n_treat = np.bincount(inv_sorted, weights=z_sorted, minlength=len(codes))
    n_treat = np.rint(n_treat).astype(int)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    keep = (n_treat > 0) & (n_treat < sizes)
    stacks: Dict[Tuple[int, int], List[np.ndarray]] = {}
    statistic = 0.0
    for s in np.flatnonzero(keep):
        block = slice(starts[s], starts[s] + sizes[s])
        stacks.setdefault((int(sizes[s]), int(n_treat[s])), []).append(sc_sorted[block])
        statistic += float(np.sum(sc_sorted[block] * z_sorted[block]))
    stacked = {key: np.vstack(rows) for key, rows in stacks.items()}
    return (
        stacked,
        statistic,
        int(keep.sum()),
        int(n_treat[keep].sum()),
        int((sizes[keep] - n_treat[keep]).sum()),
    )


def upper_pvalue(deviate: float) -> float:
    """Upper-tail normal probability of a standardised deviate."""
    return float(stats.norm.sf(deviate))


def solve_gamma(
    pvalue_at: Callable[[float], float],
    alpha: float,
    gamma_max: float = 1e4,
    tol: float = 1e-8,
) -> float:
    """The ``Gamma`` at which an increasing p-value bound crosses ``alpha``.

    Returns ``1.0`` when the bound already exceeds ``alpha`` without hidden
    bias and ``inf`` when it stays below ``alpha`` up to ``gamma_max``.
    """
    if pvalue_at(1.0) > alpha:
        return 1.0
    lo, hi = 1.0, 2.0
    while pvalue_at(hi) <= alpha:
        lo, hi = hi, hi * 2.0
        if hi > gamma_max:
            return float("inf")
    while hi - lo > tol * lo:
        mid = 0.5 * (lo + hi)
        if pvalue_at(mid) <= alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


__all__: List[str] = []
