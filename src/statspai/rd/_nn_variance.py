"""Nearest-neighbour residuals for the RD variance estimator.

Clean-room implementation from the published definition (CCT 2014 §5;
CCFT 2017 §2.2); reference (R rdrobust 4.0.0, GPL-3) used only as a
black box.

Calonico, Cattaneo and Titiunik (bib key ``calonico2014robust``, Section 5)
estimate the conditional (co)variance of the outcome at each observation
by comparing it with the mean of its ``J`` nearest neighbours on the same
side of the cutoff, scaled by ``J / (J + 1)``.  When the running variable
has mass points the ``J``-th neighbour is not unique; following the
tie rule stated in Calonico, Cattaneo, Farrell and Titiunik (2017, Stata
Journal, §2.2), every neighbour at the boundary distance is used, so the
effective number of neighbours ``J_i`` can exceed ``J``.

For unit ``i`` with ``d_i`` the ``J``-th smallest distance ``|x_j - x_i|``
over ``j != i`` (``J = min(matches, n - 1)``), the neighbour set is
``N_i = {j != i : |x_j - x_i| <= d_i}`` and the residual is

    r_i = sqrt(J_i / (J_i + 1)) * (y_i - mean_{j in N_i} y_j),
    J_i = |N_i|,

so that ``r_i r_i'`` is the nearest-neighbour estimate of
``Var(y_i | x_i)`` (or the covariance matrix when ``y`` has columns).
"""

from __future__ import annotations

import numpy as np

__all__ = ["nn_residuals"]


def nn_residuals(
    x: np.ndarray,
    y: np.ndarray,
    dups: np.ndarray,
    dupsid: np.ndarray,
    matches: int,
) -> np.ndarray:
    """Nearest-neighbour residuals on one side of the cutoff.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Running variable on one side of the cutoff, sorted ascending.
    y : ndarray, shape (n,) or (n, k)
        Outcome(s) aligned with ``x``.  Several columns give the residual
        vector used for covariances (e.g. outcome and covariates).
    dups : ndarray, shape (n,)
        Size of the tie run that ``x[i]`` belongs to.  Accepted for
        interface compatibility; the residuals are computed from ``x``
        alone, so these values are not used.
    dupsid : ndarray, shape (n,)
        1-based position of unit ``i`` inside its tie run.  Accepted for
        interface compatibility and not used.
    matches : int
        Minimum number of neighbours ``J`` (``nnmatch``).  Capped at
        ``n - 1``.  All neighbours equidistant with the ``J``-th one are
        included.

    Returns
    -------
    ndarray
        Residuals with the same shape as ``y``.

    Raises
    ------
    ValueError
        If ``x`` is not sorted, lengths disagree, ``matches < 1``, or
        ``n < 2`` (a single observation has no neighbour).

    Notes
    -----
    The neighbour set of a unit depends only on its distinct ``x`` value,
    so the work is done on the ``m`` distinct values: starting from the
    unit's own value, the nearer of the next distinct value to the left or
    right is absorbed (both when they are equally far, compared in
    floating point exactly as ``|x_j - x_i|``) until at least ``J`` other
    units are covered.  Each step absorbs at least one unit, so at most
    ``J`` vectorised steps over the ``m`` values are needed: time
    ``O(n + m J)``, memory ``O(n)``.  Window sums are accumulated value by
    value rather than through cumulative sums, which keeps them free of
    cancellation for large ``n``.

    References
    ----------
    calonico2014robust
    """
    x = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x.ndim != 1:
        raise ValueError("nn_residuals: x must be one-dimensional")
    n = x.shape[0]
    if y_arr.ndim not in (1, 2) or y_arr.shape[0] != n:
        raise ValueError(
            "nn_residuals: y must have shape (n,) or (n, k) with n = len(x)"
        )
    if len(dups) != n or len(dupsid) != n:
        raise ValueError("nn_residuals: dups and dupsid must have length n")
    matches = int(matches)
    if matches < 1:
        raise ValueError("nn_residuals: matches must be at least 1")
    if n < 2:
        raise ValueError(
            "nn_residuals: at least two observations are needed on a side "
            "of the cutoff to form a nearest-neighbour residual"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("nn_residuals: x contains non-finite values")
    if np.any(x[1:] < x[:-1]):
        raise ValueError("nn_residuals: x must be sorted ascending")

    J = min(matches, n - 1)
    y2 = y_arr.reshape(n, -1)

    # Distinct values, their run starts and counts.
    starts = np.concatenate(([0], np.flatnonzero(x[1:] != x[:-1]) + 1))
    m = starts.shape[0]
    vals = x[starts]
    counts = np.diff(np.append(starts, n))
    gsum = np.add.reduceat(y2, starts, axis=0)  # (m, k)
    group_of = np.repeat(np.arange(m), counts)

    # Expand a window [lo, hi] of distinct values around each value.
    lo = np.arange(m) - 1  # next candidate on the left
    hi = np.arange(m) + 1  # next candidate on the right
    covered = counts - 1  # other units already in the window
    wsum = gsum.copy()  # y-sum over the window (own unit included)
    active = covered < J
    while np.any(active):
        idx = np.flatnonzero(active)
        L = lo[idx]
        R = hi[idx]
        v = vals[idx]
        has_l = L >= 0
        has_r = R < m
        d_l = np.full(idx.shape[0], np.inf)
        d_r = np.full(idx.shape[0], np.inf)
        d_l[has_l] = v[has_l] - vals[L[has_l]]
        d_r[has_r] = vals[R[has_r]] - v[has_r]
        take_l = has_l & (d_l <= d_r)
        take_r = has_r & (d_r <= d_l)

        il = idx[take_l]
        Ll = L[take_l]
        covered[il] += counts[Ll]
        wsum[il] += gsum[Ll]
        lo[il] -= 1

        ir = idx[take_r]
        Rr = R[take_r]
        covered[ir] += counts[Rr]
        wsum[ir] += gsum[Rr]
        hi[ir] += 1

        active = covered < J

    n_nb = covered[group_of].astype(float)  # J_i
    nb_mean = (wsum[group_of] - y2) / n_nb[:, None]
    res = np.sqrt(n_nb / (n_nb + 1.0))[:, None] * (y2 - nb_mean)
    return res.reshape(y_arr.shape)
