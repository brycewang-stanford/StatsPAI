"""Callaway-Goodman-Bacon-Sant'Anna continuous-treatment DiD primitives.

With a continuous treatment there is no single ATT. A unit that got a dose
of 0.2 and a unit that got 0.8 are not the same comparison, and the
two-way fixed-effects coefficient mixes them with weights nobody chose.
CGS replace it with two dose-indexed curves:

``ATT(d)``
    the effect of receiving dose ``d`` versus none, for units that got ``d``.

``ACRT(d)``
    the *causal response* at ``d`` -- the derivative of ``ATT`` -- which is
    what a marginal-dose policy question actually asks about.

Estimation, per ``(g, t)`` cell
-------------------------------
Take the two-period subset (the treated cohort at ``g``, its base period,
and the controls), difference the outcome within unit, and regress that
change on a B-spline basis in the dose among the treated. The control
group's mean change is the level the whole curve is measured against::

    ATT(d)  = b(d)' beta - mean(dy | dose = 0)
    ACRT(d) = b'(d)' beta

The spline degree and knot count are the only smoothing choices, and both
are explicit rather than implied by a bandwidth.

Ported formula-by-formula from ``contdid`` 0.1.1 (``cont_did_acrt``,
``cont_two_by_two_subset``), including its ``splines2::bSpline`` basis with
``intercept = FALSE`` and its influence function for the overall ACRT.

References
----------
callaway2024difference
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["ContDoseFit", "bspline_basis", "cont_did_cell"]


@dataclass
class ContDoseFit:
    """One ``(g, t)`` cell of the continuous-treatment estimator."""

    dose_grid: np.ndarray
    att_d: np.ndarray
    acrt_d: np.ndarray
    att_overall: float
    acrt_overall: float
    influence: np.ndarray
    coefficients: np.ndarray
    knots: np.ndarray
    n_treated: int
    n_control: int


def bspline_basis(
    x: Sequence[float],
    *,
    degree: int,
    knots: Sequence[float],
    lower: float,
    upper: float,
    deriv: int = 0,
) -> np.ndarray:
    """``splines2::bSpline(x, degree, knots, intercept = FALSE)``.

    SciPy's ``BSpline`` is right-open, so the basis it reports *at* the upper
    boundary is zero rather than the left limit. Evaluating with
    ``extrapolate=True`` continues the final polynomial piece, which is that
    limit, and makes the boundary agree with R instead of silently dropping
    the largest dose out of the basis.
    """
    from scipy.interpolate import BSpline

    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if degree < 1:
        raise MethodIncompatibility(
            "continuous DiD: spline `degree` must be at least 1.",
            diagnostics={"degree": degree},
        )
    if knots.size and (knots.min() <= lower or knots.max() >= upper):
        raise MethodIncompatibility(
            "continuous DiD: interior knots must lie strictly inside the "
            "observed dose range.",
            diagnostics={
                "knots": knots.tolist(),
                "lower": float(lower),
                "upper": float(upper),
            },
        )
    t = np.concatenate(
        [np.full(degree + 1, lower), np.sort(knots), np.full(degree + 1, upper)]
    )
    n_basis = len(t) - degree - 1
    out = np.zeros((len(x), n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spline = BSpline(t, c, degree, extrapolate=True)
        f = spline.derivative(deriv) if deriv else spline
        out[:, j] = f(x)
    # intercept = FALSE drops the first basis function; the regression
    # supplies its own intercept.
    return out[:, 1:]


def _ols(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OLS with an intercept; returns (coef, bread) with coef[0] the intercept."""
    Xi = np.column_stack([np.ones(len(X)), X])
    xtx = Xi.T @ Xi
    if np.linalg.cond(xtx) > 1.0 / np.finfo(float).eps:
        raise DataInsufficient(
            "continuous DiD: the dose spline basis is collinear on this cell "
            "-- usually too many knots for the doses actually observed.",
            diagnostics={"n_rows": int(Xi.shape[0]), "n_cols": int(Xi.shape[1])},
        )
    coef = np.linalg.solve(xtx, Xi.T @ y)
    # sandwich::bread for lm is n * (X'X)^-1.
    bread = np.linalg.inv(xtx) * len(Xi)
    return coef, bread


def cont_did_cell(
    *,
    dose: np.ndarray,
    dy: np.ndarray,
    dose_grid: np.ndarray,
    degree: int = 1,
    knots: Optional[Sequence[float]] = None,
) -> ContDoseFit:
    """``ATT(d)`` and ``ACRT(d)`` for one two-period cell.

    Parameters
    ----------
    dose : ndarray
        Per-unit dose, zero for control units.
    dy : ndarray
        Per-unit change in the outcome across the two periods.
    dose_grid : ndarray
        Doses at which the curves are reported.
    degree : int, default 1
        B-spline degree. 1 is piecewise linear, which with no interior knots
        makes ``ACRT`` a single constant -- the closest thing to the familiar
        "effect per unit of dose".
    knots : sequence of float, optional
        Interior knots. More knots buy flexibility at the cost of variance,
        and there is no automatic selector here: the choice is the analyst's
        and is recorded on the result.
    """
    dose = np.asarray(dose, dtype=float)
    dy = np.asarray(dy, dtype=float)
    dose_grid = np.asarray(dose_grid, dtype=float)
    treated = dose > 0
    control = dose == 0
    if not treated.any():
        raise DataInsufficient(
            "continuous DiD: no units with a positive dose in this cell.",
            diagnostics={"n": int(len(dose))},
        )
    if not control.any():
        raise DataInsufficient(
            "continuous DiD: no zero-dose units to level the curve against, "
            "so ATT(d) is not identified (ACRT(d) would still be).",
            diagnostics={"n": int(len(dose))},
        )

    d_treated = dose[treated]
    lower, upper = float(d_treated.min()), float(d_treated.max())
    knots_arr = np.asarray([] if knots is None else knots, dtype=float)

    B = bspline_basis(
        d_treated, degree=degree, knots=knots_arr, lower=lower, upper=upper
    )
    coef, bread = _ols(B, dy[treated])
    slope = coef[1:]

    B_grid = bspline_basis(
        dose_grid, degree=degree, knots=knots_arr, lower=lower, upper=upper
    )
    dB_grid = bspline_basis(
        dose_grid, degree=degree, knots=knots_arr, lower=lower, upper=upper, deriv=1
    )
    control_mean = float(np.mean(dy[control]))
    att_d = coef[0] + B_grid @ slope - control_mean
    acrt_d = dB_grid @ slope

    dB_treated = bspline_basis(
        d_treated, degree=degree, knots=knots_arr, lower=lower, upper=upper, deriv=1
    )
    att_overall = float(np.mean(coef[0] + B @ slope) - control_mean)
    acrt_overall = float(np.mean(dB_treated @ slope))

    # Influence function of the overall ACRT: the pointwise deviation plus
    # the estimation effect of the spline coefficients, exactly as the
    # reference assembles it from sandwich::estfun and sandwich::bread.
    Xi = np.column_stack([np.ones(len(B)), B])
    resid = dy[treated] - Xi @ coef
    estfun = Xi * resid[:, None]
    grad = np.concatenate([[0.0], dB_treated.mean(axis=0)])
    inf_1 = dB_treated @ slope - acrt_overall
    inf_2 = estfun @ bread @ grad
    influence = np.zeros(len(dose))
    influence[treated] = inf_1 + inf_2

    return ContDoseFit(
        dose_grid=dose_grid,
        att_d=att_d,
        acrt_d=acrt_d,
        att_overall=att_overall,
        acrt_overall=acrt_overall,
        influence=influence,
        coefficients=coef,
        knots=knots_arr,
        n_treated=int(treated.sum()),
        n_control=int(control.sum()),
    )


def default_dose_grid(
    dose: np.ndarray, *, lower_q: float = 0.10, upper_q: float = 0.99
) -> np.ndarray:
    """Percentile grid of positive doses, matching the reference's default.

    The curves are only reported where there are doses to support them; the
    reference trims to the 10th-99th percentile of the positive doses for
    the same reason.
    """
    positive = np.asarray(dose, dtype=float)
    positive = positive[positive > 0]
    if positive.size == 0:
        raise DataInsufficient(
            "continuous DiD: no positive doses to build a grid from.",
            diagnostics={},
        )
    probs = np.arange(round(lower_q * 100), round(upper_q * 100) + 1) / 100.0
    return np.quantile(positive, probs)


def knots_by_quantile(dose: np.ndarray, num_knots: int) -> List[float]:
    """Interior knots at equally spaced quantiles of the positive doses."""
    if num_knots <= 0:
        return []
    positive = np.asarray(dose, dtype=float)
    positive = positive[positive > 0]
    probs = np.linspace(0.0, 1.0, num_knots + 2)[1:-1]
    return [float(q) for q in np.quantile(positive, probs)]
