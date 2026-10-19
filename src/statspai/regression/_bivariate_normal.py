"""Bivariate standard-normal CDF shared by the bivariate-probit family.

``binormal_cdf(h, k, rho) = Pr(X <= h, Y <= k)`` for standard normals with
correlation ``rho``, vectorised over observations and complex-step safe (no
dtype casts, no clipping), so the ML estimators that use it get exact
per-observation scores from :func:`._optim_helpers.complex_step_scores`.

It evaluates Sheppard's identity

    Phi2(h, k; rho) = Phi(h) Phi(k)
        + 1 / (2 pi) * int_0^{arcsin rho}
          exp(-(h^2 - 2 h k sin t + k^2) / (2 cos^2 t)) dt

by Gauss-Legendre quadrature. Integrating over ``t = arcsin r`` instead of
``r`` removes the ``1 / sqrt(1 - r^2)`` singularity of the density at
``|r| -> 1``: with 32 nodes the error is below 1e-14 up to ``|rho| = 0.99``
and below 1e-9 at ``|rho| = 0.999``, where the ``r``-form with 24 nodes was
off by 9e-7 and 4e-4.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import special

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(32)
_S = 0.5 * (_NODES + 1.0)  # [-1, 1] -> [0, 1]
_W = 0.5 * _WEIGHTS


def binormal_cdf(h: Any, k: Any, rho: Any) -> np.ndarray:
    """``Pr(X <= h, Y <= k)`` for a standard bivariate normal, corr ``rho``."""
    h = np.asarray(h)
    k = np.asarray(k)
    rho = np.asarray(rho)
    if rho.ndim == 0:
        rho = rho * np.ones(h.shape)
    a = np.arcsin(rho)
    hh_kk = h * h + k * k
    hk2 = 2.0 * h * k
    integral = np.zeros(h.shape, dtype=np.result_type(h, k, rho, float))
    for s_j, w_j in zip(_S, _W):
        t = a * s_j
        integral = integral + w_j * np.exp(
            -(hh_kk - hk2 * np.sin(t)) / (2.0 * np.cos(t) ** 2)
        )
    return np.asarray(special.ndtr(h) * special.ndtr(k) + a * integral / (2.0 * np.pi))


__all__ = ["binormal_cdf"]
