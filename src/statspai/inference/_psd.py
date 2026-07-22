"""Standard errors from a covariance matrix that may not be positive definite.

Kernel-weighted HAC estimators — Conley spatial HAC above all — are *not*
guaranteed positive semi-definite in finite samples. With a uniform (indicator)
spatial kernel the weight matrix is generally not a PSD kernel, so
``S' W S`` can land with negative entries on its diagonal. This is a
substantive property of the estimator, not floating-point noise: Stata's
``acreg`` reproduces the same negative variances and reports the affected
standard errors as missing (``.``).

The repo-wide idiom ``np.sqrt(np.maximum(np.diag(V), 0))`` is fine for
sandwiches that are PSD by construction, where a ``-1e-18`` is rounding. Used
on a Conley vcov it silently converts "this estimator failed here" into
``se = 0`` — which downstream reads as an infinitely precise estimate: ``t =
inf``, ``p = 0``. That is the most dangerous possible way to fail.

So: clamp what is genuinely rounding, and return ``nan`` plus a loud warning
for anything materially negative.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np

from ..exceptions import NumericalInstability

__all__ = ["se_from_vcov", "psd_diagnostics"]

#: A diagonal entry counts as rounding noise if it is negative but smaller in
#: magnitude than ``_NOISE_TOL`` times the largest positive variance. Anything
#: bigger is reported, not hidden.
_NOISE_TOL = 1e-10


def psd_diagnostics(vcov: np.ndarray, tol: float = _NOISE_TOL) -> dict:
    """Classify the diagonal of ``vcov`` into ok / rounding-noise / negative."""
    diag = np.asarray(np.diag(np.asarray(vcov, dtype=float)), dtype=float)
    scale = float(np.max(diag)) if diag.size and np.max(diag) > 0 else 1.0
    noise = (diag < 0) & (np.abs(diag) <= tol * scale)
    negative = (diag < 0) & ~noise
    return {
        "diag": diag,
        "noise_mask": noise,
        "negative_mask": negative,
        "n_negative": int(negative.sum()),
        "scale": scale,
    }


def se_from_vcov(
    vcov: np.ndarray,
    names: Optional[Sequence[str]] = None,
    *,
    estimator: str = "HAC",
    on_negative: str = "warn",
    tol: float = _NOISE_TOL,
) -> np.ndarray:
    """Standard errors from ``vcov``, honest about non-PSD diagonals.

    Parameters
    ----------
    vcov : ndarray
        Covariance matrix.
    names : sequence of str, optional
        Coefficient names, used to say *which* terms failed.
    estimator : str
        Estimator label for the message (e.g. ``"Conley spatial HAC"``).
    on_negative : {"warn", "raise"}
        ``"warn"`` returns ``nan`` for the affected terms and warns (this is
        Stata ``acreg``'s behaviour). ``"raise"`` raises
        :class:`~statspai.exceptions.NumericalInstability`.
    tol : float
        Relative tolerance below which a negative variance is treated as
        rounding noise and clamped to zero silently.

    Returns
    -------
    ndarray
        Standard errors; ``nan`` wherever the variance was materially negative.
    """
    if on_negative not in ("warn", "raise"):
        raise ValueError(f"on_negative must be 'warn' or 'raise', got {on_negative!r}")

    diagnostics = psd_diagnostics(vcov, tol=tol)
    diag = diagnostics["diag"].copy()
    negative = diagnostics["negative_mask"]

    # Rounding-level negatives are genuinely zero; clamp them without noise.
    diag[diagnostics["noise_mask"]] = 0.0

    if not negative.any():
        return np.sqrt(diag)

    if names is not None and len(names) == diag.size:
        offenders = [str(names[i]) for i in np.flatnonzero(negative)]
    else:
        offenders = [f"index {i}" for i in np.flatnonzero(negative)]
    worst = float(diag[negative].min())

    message = (
        f"{estimator} produced a non-positive-definite covariance matrix: "
        f"{len(offenders)} of {diag.size} variance(s) are negative "
        f"(most negative: {worst:.3g}) for term(s) {offenders}. "
        f"Standard errors for those terms are reported as nan, not 0 — a "
        f"negative variance means the estimator has failed here, and clamping "
        f"it to 0 would imply infinite precision (t=inf, p=0).\n"
        f"This is expected behaviour for kernel-weighted HAC estimators and "
        f"Stata's acreg reports the same terms as missing. Common remedies: "
        f"widen or narrow the distance cutoff, switch to kernel='bartlett' "
        f"(tapered kernels are far better behaved than the uniform "
        f"indicator), or check whether the coordinates are collinear with the "
        f"absorbed fixed effects — if every unit in an FE group shares one "
        f"location, the spatial weights carry no within-group information."
    )

    if on_negative == "raise":
        raise NumericalInstability(message)

    warnings.warn(message, RuntimeWarning, stacklevel=3)
    se = np.full(diag.size, np.nan, dtype=float)
    ok = ~negative
    se[ok] = np.sqrt(diag[ok])
    return se
