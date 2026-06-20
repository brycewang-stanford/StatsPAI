"""Private validation helpers shared by native fast estimators."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility


def positive_int(value: Any, *, name: str, context: str) -> int:
    """Return ``value`` as an int after requiring ``value >= 1``."""
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be a positive integer"
        ) from exc
    if isinstance(value, bool) or parsed < 1:
        raise MethodIncompatibility(f"{context}: {name} must be a positive integer")
    return int(parsed)


def nonnegative_finite_float(value: Any, *, name: str, context: str) -> float:
    """Return ``value`` as float after requiring finite ``value >= 0``."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and non-negative"
        ) from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and non-negative"
        )
    return parsed


def open_unit_float(value: Any, *, name: str, context: str) -> float:
    """Return ``value`` as float after requiring finite ``0 < value < 1``."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and in the open interval (0, 1)"
        ) from exc
    if not np.isfinite(parsed) or not (0.0 < parsed < 1.0):
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and in the open interval (0, 1)"
        )
    return parsed


def nonempty_sample(n_obs: int, *, context: str) -> None:
    """Reject empty estimator inputs before linear algebra is attempted."""
    if int(n_obs) < 1:
        raise DataInsufficient(f"{context}: data must contain at least one row")


def positive_weight_mass(weights: np.ndarray, *, context: str) -> None:
    """Reject samples whose validated weights have no positive mass."""
    if not (np.asarray(weights, dtype=np.float64) > 0.0).any():
        raise DataInsufficient(f"{context}: weights contain no positive mass")
