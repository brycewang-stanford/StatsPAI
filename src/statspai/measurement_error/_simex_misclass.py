"""Misclassification SIMEX (MC-SIMEX) for a binary variable.

A 0/1 variable is observed through a misclassification matrix ``P`` with
``P[j, i] = Pr(observed = j | true = i)`` (columns sum to one, as in R
``simex::mcsimex``). MC-SIMEX misclassifies the observed variable again
with ``P ** lambda``, so that the total misclassification is
``P ** (1 + lambda)``, and extrapolates to ``lambda = -1`` like SIMEX.
``P ** lambda`` is ``V diag(e ** lambda) V^{-1}`` from the eigen
decomposition, rounded to seven decimals like R's ``zapsmall``; it is a
misclassification matrix for every ``lambda >= 0`` when ``P[0, 0] +
P[1, 1] > 1``.

References
----------
[@kuchenhoff2006general]
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def mc_matrix(name: str, value: Any) -> np.ndarray:
    """Validate one 2 x 2 misclassification matrix (rows, columns = 0, 1)."""
    if isinstance(value, pd.DataFrame):
        try:
            value = value.loc[[0, 1], [0, 1]]
        except KeyError:
            value = value.to_numpy()
    try:
        P = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        P = np.full((1, 1), np.nan)
    hint = (
        "Pass misclassification={'x': [[p00, p01], [p10, p11]]} with "
        "P[j, i] = Pr(observed j | true i); each column sums to one."
    )
    if P.shape != (2, 2) or not np.all(np.isfinite(P)):
        raise MethodIncompatibility(
            f"sp.simex: the misclassification matrix of {name!r} must be a "
            "finite 2 x 2 matrix.",
            recovery_hint=hint,
        )
    if np.any(P < 0) or np.any(P > 1) or not np.allclose(P.sum(axis=0), 1.0, atol=1e-8):
        raise MethodIncompatibility(
            f"sp.simex: the misclassification matrix of {name!r} must have "
            "entries in [0, 1] and columns that sum to one.",
            recovery_hint=hint,
            diagnostics={"column_sums": P.sum(axis=0).tolist()},
        )
    if not P[0, 0] + P[1, 1] > 1.0:
        raise MethodIncompatibility(
            f"sp.simex: the misclassification matrix of {name!r} has "
            "P[0, 0] + P[1, 1] <= 1; its fractional powers are not "
            "misclassification matrices and the error cannot be undone.",
            recovery_hint="A classifier no better than chance carries no "
            "information; check the orientation of the matrix (columns are "
            "the true values).",
        )
    return P


def mc_power(P: np.ndarray, lam: float) -> np.ndarray:
    """``P ** lam`` by eigen decomposition, rounded like R ``zapsmall``."""
    evalue, evec = np.linalg.eig(P)
    out = np.real(evec @ np.diag(evalue.astype(complex) ** lam) @ np.linalg.inv(evec))
    mx = float(np.max(np.abs(out)))
    digits = max(0, 7 - int(np.ceil(np.log10(mx)))) if mx > 0 else 7
    rounded: np.ndarray = np.round(out, digits)
    return rounded


def misclassify(values: np.ndarray, P_lam: np.ndarray, rng: Any) -> np.ndarray:
    """Draw a new 0/1 vector: ``Pr(new = 1 | old = i) = P_lam[1, i]``."""
    prob_one = P_lam[1, values.astype(int)]
    out: np.ndarray = (rng.random(values.size) < prob_one).astype(float)
    return out
