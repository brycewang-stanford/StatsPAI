"""Einav, Finkelstein and Cullen (2010), Online Appendix Table A2, rows 1-5.

Panel B of the table publishes the demand and cost coefficients of each
specification; Panel A the welfare numbers they imply. Feeding the
published coefficients to the welfare solver behind
``sp.efc_welfare(demand=...)`` must give the published welfare numbers up to
the rounding of the published coefficients: every published number has to
lie inside the range spanned by the corners of the rounding box (each
coefficient +/- half a unit of its last printed digit), widened by half a
unit of the number's own last printed digit.

Source: QJE2010_Appendix.pdf on Liran Einav's site
(web.stanford.edu/~leinav/pubs/), Table A2. Rows: 1 baseline linear demand;
2 probit demand; 3 linear demand through (Q, P) = (1, $0); 4 linear demand
through (0, $800); 5 quadratic demand through both. The cost curve is the
baseline one (gamma 391.7, delta 0.155) in all five rows. Coefficients
marked "(imposed)" in the table follow from the constraint: alpha = 1 in
rows 3 and 5, alpha = -800 beta in row 4, and in row 5 the curvature is
-(1 + 800 beta) / 800^2 (the table prints -1.1e-06).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from statspai.adverse_selection._efc_nonlinear import efc_quantities, rows_for

KEYS = ("coverage_eq", "price_eq", "coverage_eff", "price_eff", "dwl")
# Panel A, columns (1)-(5).
PUBLISHED = {
    1: (0.617, 463.51, 0.756, 263.94, 9.55),
    2: (0.619, 463.59, 0.790, 187.85, 11.32),
    3: (0.612, 463.56, 0.750, 299.04, 7.81),
    4: (0.562, 463.59, 0.688, 387.90, 3.30),
    5: (0.587, 463.58, 0.738, 343.51, 5.00),
}
# Half a unit of the last printed digit of each published number.
HALF = (0.0005, 0.005, 0.0005, 0.005, 0.005)


def _spec(row, u):
    gamma = 391.7 + 0.05 * u[2]
    delta = 0.155 + 0.0005 * u[3]
    if row in (1, 2):
        a0, b0 = (0.940, -0.00070) if row == 1 else (1.149, -0.00183)
        kind = "linear" if row == 1 else "probit"
        return kind, [a0 + 0.0005 * u[0], b0 + 0.000005 * u[1]], gamma, delta
    if row == 3:
        return "linear", [1.0, -0.00084 + 0.000005 * u[1]], gamma, delta
    if row == 4:
        b = -0.00167 + 0.000005 * u[1]
        return "linear", [-800 * b, b], gamma, delta
    b1 = -0.00039 + 0.000005 * u[1]
    return "quadratic", [1.0, b1, -(1 + 800 * b1) / 800**2], gamma, delta


@pytest.mark.parametrize("row", sorted(PUBLISHED))
def test_published_welfare_numbers_follow_from_published_coefficients(row):
    vals = []
    for corner in itertools.product((-1.0, 1.0), repeat=4):
        kind, coef, gamma, delta = _spec(row, corner)
        out = dict(zip(rows_for(kind), efc_quantities(kind, coef, gamma, delta)))
        vals.append([out[k] for k in KEYS])
    vals = np.asarray(vals)
    lo, hi = vals.min(axis=0), vals.max(axis=0)
    for k, pub, h, a, b in zip(KEYS, PUBLISHED[row], HALF, lo, hi):
        assert a - h <= pub <= b + h, (row, k, pub, a, b)
