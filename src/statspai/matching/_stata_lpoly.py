"""Local polynomial smoother compatible with Stata's ``lpoly``.

This module backs ``sp.match(method='llr', llr_stata_compat=True)``. When
``psmatch2`` is asked for local linear regression matching with its
Epanechnikov kernel, what it reports is not local linear matching: it
smooths the control outcome on the propensity score with ``lpoly`` and then
matches each treated unit to its nearest control, taking that control's
smoothed value as the counterfactual (see section 5 of the specification).
The functions here supply the smoothing step.

Provenance
----------
Clean-room implementation written from the Stata Base Reference Manual
entries [R] kdensity (kernel table) and [R] lpoly (Methods and formulas)
and from Fan & Gijbels (1996). Stata's ``psmatch2`` and ``lpoly`` were used
only as black boxes, by comparing their displayed and stored outputs. The
specification is ``docs/dev/cleanroom/psmatch2.md`` (sections 1, 5 and 6).

The Epanechnikov kernel used by ``lpoly`` is the unit-variance form,
``K(z) = 3/4 (1 - z^2/5) / sqrt(5)`` on ``|z| <= sqrt(5)``, so a bandwidth
``h`` gives a window of half-width ``sqrt(5) h``. This differs from the
compact ``1 - z^2`` kernel on ``[-1, 1]`` used for kernel matching, where a
bandwidth ``h`` gives a window of half-width ``h``.

References
----------
Fan, J. and Gijbels, I. (1996). *Local Polynomial Modelling and Its
Applications*. Chapman & Hall.  [``fan1996local``]
"""

from __future__ import annotations

import numpy as np

#: Half-width of the unit-variance Epanechnikov kernel, in bandwidth units.
_EPAN_SUPPORT = np.sqrt(5.0)


def stata_epanechnikov(u: np.ndarray) -> np.ndarray:
    """Unit-variance Epanechnikov kernel of [R] kdensity.

    Evaluated on the closed interval ``|u| <= sqrt(5)`` and zero outside.
    """
    u = np.asarray(u, dtype=float)
    return np.where(
        np.abs(u) <= _EPAN_SUPPORT, 0.75 * (1 - u**2 / 5) / _EPAN_SUPPORT, 0.0
    )


def lpoly_predict(
    x: np.ndarray,
    y: np.ndarray,
    at: np.ndarray,
    bandwidth: float,
    degree: int = 1,
) -> np.ndarray:
    """Kernel-weighted local polynomial fit of ``y`` on ``x``, read at ``at``.

    At each evaluation point ``x0`` a polynomial of order ``degree`` in
    ``x - x0`` is fitted by weighted least squares with weights
    ``stata_epanechnikov((x - x0) / bandwidth)``; the intercept is the
    smoothed value.

    Parameters
    ----------
    x, y : np.ndarray
        Scatterplot data. Pairs where either value is not finite are dropped.
    at : np.ndarray
        One-dimensional array of evaluation points.
    bandwidth : float
        Kernel bandwidth; must be positive. There is no rule-of-thumb
        default, so ``None`` raises ``TypeError``.
    degree : int, default 1
        Order of the local polynomial.

    Returns
    -------
    np.ndarray
        Smoothed values, same shape as ``at``. An entry is ``nan`` where
        fewer than ``degree + 1`` points carry positive kernel weight, or
        where the local normal equations are singular. Stata's ``lpoly``
        instead reports a lower-degree fit (such as the local mean) at such
        points; this function does not reproduce that behaviour.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    at = np.asarray(at, dtype=float)
    h = float(bandwidth)
    if not h > 0:
        raise ValueError(f"lpoly bandwidth must be positive, got {h}")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    out = np.full(at.shape, np.nan)
    for i, x0 in enumerate(at):
        gap = x - x0
        k = stata_epanechnikov(gap / h)
        keep = k > 0
        if np.sum(keep) < degree + 1:
            continue
        X = np.vander(gap[keep], degree + 1, increasing=True)
        A = X.T @ (k[keep][:, None] * X)
        b = X.T @ (k[keep] * y[keep])
        try:
            coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        out[i] = coef[0]
    return out


def psmatch2_llr_smoothed_outcome(
    outcome: np.ndarray,
    pscore: np.ndarray,
    treated: np.ndarray,
    support: np.ndarray,
    bandwidth: float,
    degree: int = 1,
) -> np.ndarray:
    """Smoothed control outcome curve, evaluated at every unit's score.

    The outcome of the untreated units on common support is smoothed on the
    propensity score with :func:`lpoly_predict`, and the curve is read at the
    score of every unit, treated or not. This is the series that psmatch2's
    local linear option with the Epanechnikov kernel matches on (spec
    ``docs/dev/cleanroom/psmatch2.md``, sections 5 and 6).

    Parameters
    ----------
    outcome, pscore : np.ndarray
        Outcome and propensity score of all ``n`` units.
    treated, support : np.ndarray
        Treatment and common-support indicators (bool or 0/1).
    bandwidth, degree
        Passed to :func:`lpoly_predict`.

    Returns
    -------
    np.ndarray
        Length-``n`` smoothed values (all ``nan`` when no control is on
        support).
    """
    y = np.asarray(outcome, dtype=float)
    p = np.asarray(pscore, dtype=float)
    on_support = np.asarray(support).astype(bool)
    controls = (np.asarray(treated, dtype=float) == 0) & on_support
    return lpoly_predict(p[controls], y[controls], p, bandwidth, degree)
