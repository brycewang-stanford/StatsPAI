"""Shared least-squares and variance pieces of the boundary-RD estimators.

Clean-room implementation (2026-10-21). The estimators are those of
Cattaneo, Titiunik and Yu (location-based: arXiv 2505.05670; distance-based:
arXiv 2510.26051; software article: arXiv 2505.07989, bib key
``cattaneo2025boundary`` for the package). R ``rd2d`` 1.0.0 (GPL-3) was used
only as a black box; every convention below that the papers leave open was
fixed by comparing outputs and is recorded in
``docs/dev/cleanroom/rd2d.md``.

The pieces:

* kernels on ``[-1, 1]`` (Gaussian: the standard normal density) and the
  second moment / roughness of their two-dimensional radial versions, which
  enter the rule-of-thumb pilot bandwidth;
* weighted least squares on a window with per-observation influence
  functions of a linear functional of the coefficients;
* the variance of a difference of two side estimates under HC0-HC3 or
  cluster-robust sums, for side-by-side ("separate") or treatment-interacted
  ("joint") fits, and the same rules for covariances across evaluation
  points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import DataInsufficient, NumericalInstability

__all__ = [
    "KERNEL_NAMES",
    "canonical_kernel",
    "kernel_1d",
    "RADIAL_MOMENTS",
    "monomials",
    "poly_basis",
    "WindowFit",
    "fit_window",
    "side_variances",
    "point_covariance",
    "cer_rate",
]

KERNEL_NAMES = {
    "tri": "triangular",
    "triangular": "triangular",
    "epa": "epanechnikov",
    "epanechnikov": "epanechnikov",
    "uni": "uniform",
    "uniform": "uniform",
    "gau": "gaussian",
    "gaussian": "gaussian",
}

# Second moment  mu2 = int u_1^2 K(|u|) du  and roughness  R = int K(|u|)^2 du
# of the radial kernel K(|u|) on the unit disk, normalised to integrate to
# one (Gaussian: the bivariate standard normal). With K(r) = c (1 - r)
# (triangular, c = 3/pi), c (1 - r^2) (Epanechnikov, c = 2/pi) and c
# (uniform, c = 1/pi), polar integration gives the values below.
RADIAL_MOMENTS = {
    "triangular": (3.0 / 20.0, 3.0 / (2.0 * math.pi)),
    "epanechnikov": (1.0 / 6.0, 4.0 / (3.0 * math.pi)),
    "uniform": (1.0 / 4.0, 1.0 / math.pi),
    "gaussian": (1.0, 1.0 / (4.0 * math.pi)),
}


def canonical_kernel(kernel: str) -> str:
    """Full kernel name for any accepted spelling (KeyError otherwise)."""
    return KERNEL_NAMES[str(kernel).lower()]


def kernel_1d(u: np.ndarray, kernel: str) -> np.ndarray:
    """Univariate kernel profile at ``u`` (zero outside ``|u| <= 1``)."""
    if kernel == "gaussian":
        return np.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)
    a = np.abs(u)
    inside = a <= 1.0
    if kernel == "triangular":
        return np.where(inside, 1.0 - a, 0.0)
    if kernel == "epanechnikov":
        return np.where(inside, 0.75 * (1.0 - u * u), 0.0)
    return np.where(inside, 0.5, 0.0)


def monomials(order: int) -> List[Tuple[int, int]]:
    """Exponent pairs of all bivariate monomials of degree <= order.

    Ordered by total degree, then by decreasing power of the first
    coordinate: 1, x1, x2, x1^2, x1 x2, x2^2, ...
    """
    return [(s - j, j) for s in range(order + 1) for j in range(s + 1)]


def poly_basis(d1: np.ndarray, d2: Optional[np.ndarray], order: int) -> np.ndarray:
    """Design of the monomials (bivariate) or of 1, d, ..., d^order."""
    if d2 is None:
        return np.column_stack([d1**j for j in range(order + 1)])
    return np.column_stack([d1**i * d2**j for i, j in monomials(order)])


@dataclass
class WindowFit:
    """Weighted least squares on the observations with positive weight."""

    beta: np.ndarray
    a_inv: np.ndarray
    scores: np.ndarray  # rows w_i x_i e_i
    leverage: np.ndarray
    mask: np.ndarray  # positive-weight observations of the side
    n: int
    k: int
    cluster: Optional[np.ndarray] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def influence(self, vec: np.ndarray, vce: str) -> np.ndarray:
        """Per-observation influence of ``vec' beta`` (window rows only).

        HC2/HC3 rescale each residual by its leverage; HC1 and cluster
        factors are applied when the variance is assembled.
        """
        s = self.scores
        if vce == "hc2":
            s = s / np.sqrt(1.0 - self.leverage)[:, None]
        elif vce == "hc3":
            s = s / (1.0 - self.leverage)[:, None]
        return s @ (self.a_inv @ vec)


def fit_window(
    design: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    cluster: Optional[np.ndarray] = None,
    what: str = "local polynomial",
) -> WindowFit:
    """WLS of ``y`` on ``design`` over the rows with positive weight."""
    mask = weights > 0
    X = design[mask]
    w = weights[mask]
    n, k = X.shape
    if n < k:
        raise DataInsufficient(
            f"rd2d: the {what} fit has {n} observations in its window for "
            f"{k} coefficients.",
            recovery_hint="Use a larger bandwidth (or bwcheck=) or a lower "
            "polynomial order.",
            diagnostics={"n_window": n, "n_coef": k},
        )
    XtW = X.T * w
    A = XtW @ X
    try:
        a_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstability(
            f"rd2d: the {what} design is singular inside the bandwidth window.",
            recovery_hint="Use a larger bandwidth or a lower polynomial order.",
            diagnostics={"n_window": n, "n_coef": k},
        ) from exc
    beta = a_inv @ (XtW @ y[mask])
    resid = y[mask] - X @ beta
    lev = np.einsum("ij,jk,ik->i", X, a_inv, X) * w
    return WindowFit(
        beta=beta,
        a_inv=a_inv,
        scores=X * (w * resid)[:, None],
        leverage=lev,
        mask=mask,
        n=n,
        k=k,
        cluster=None if cluster is None else np.asarray(cluster)[mask],
    )


def _cluster_factor(G: int, N: int, K: int, vce: str) -> float:
    # CR1 small-sample factor; with vce='hc1' the residuals also carry the
    # HC1 factor N/(N-K) (black box: both factors appear).
    f = G / (G - 1.0) * (N - 1.0) / (N - K)
    if vce == "hc1":
        f *= N / (N - K)
    return f


def side_variances(
    fits: Sequence[WindowFit],
    infl: Sequence[np.ndarray],
    vce: str,
    joint: bool,
) -> Tuple[float, float, float]:
    """``(V_0, V_1, C_01)`` of two side estimates with influences ``infl``.

    The variance of the jump ``theta_1 - theta_0`` is ``V_0 + V_1 - 2 C_01``.
    Without clusters the sides are independent (``C_01 = 0``); HC1 uses
    ``N/(N-K)`` per side ("separate") or with ``N``, ``K`` pooled over both
    sides ("joint", the treatment-interacted regression). With clusters,
    "joint" sums each cluster's contributions from both sides before
    squaring, which gives the cross term.
    """
    f0, f1 = fits
    if f0.cluster is None:
        if vce == "hc1":
            if joint:
                N, K = f0.n + f1.n, f0.k + f1.k
                c0 = c1 = N / (N - K)
            else:
                c0 = f0.n / (f0.n - f0.k)
                c1 = f1.n / (f1.n - f1.k)
        else:
            c0 = c1 = 1.0
        return c0 * float(infl[0] @ infl[0]), c1 * float(infl[1] @ infl[1]), 0.0
    if not joint:
        out = []
        for f, r in zip(fits, infl):
            codes, inv = np.unique(f.cluster, return_inverse=True)
            s = np.bincount(inv, weights=r)
            out.append(_cluster_factor(len(codes), f.n, f.k, vce) * float(s @ s))
        return out[0], out[1], 0.0
    codes, inv = np.unique(
        np.concatenate([f0.cluster, f1.cluster]), return_inverse=True
    )
    G = len(codes)
    s0 = np.bincount(inv[: f0.n], weights=infl[0], minlength=G)
    s1 = np.bincount(inv[f0.n :], weights=infl[1], minlength=G)
    fac = _cluster_factor(G, f0.n + f1.n, f0.k + f1.k, vce)
    return fac * float(s0 @ s0), fac * float(s1 @ s1), fac * float(s0 @ s1)


def point_covariance(
    points: Sequence[Dict[str, object]], vce: str, joint: bool
) -> np.ndarray:
    """Covariance of the jumps at several evaluation points.

    Each entry of ``points`` holds, for side 0 and 1, the window fit, the
    influence vector on that window and the side's full-sample positions
    (``pos0``, ``pos1``). Each point's influence is scaled by the square
    root of its own small-sample factor (black box: the cross term carries
    the geometric mean of the two points' factors).
    """
    J = len(points)
    if J == 0:
        return np.zeros((0, 0))
    n = int(points[0]["n_total"])
    cluster = points[0]["cluster"]
    vecs = []
    for pt in points:
        f0, f1 = pt["fits"]
        r0, r1 = pt["infl"]
        if cluster is None:
            if vce == "hc1":
                if joint:
                    N, K = f0.n + f1.n, f0.k + f1.k
                    c0 = c1 = N / (N - K)
                else:
                    c0, c1 = f0.n / (f0.n - f0.k), f1.n / (f1.n - f1.k)
            else:
                c0 = c1 = 1.0
            v = np.zeros(n)
            v[pt["pos0"][f0.mask]] = -math.sqrt(c0) * r0
            v[pt["pos1"][f1.mask]] = math.sqrt(c1) * r1
            vecs.append(("obs", v))
            continue
        codes_all = pt["codes"]
        G_total = int(pt["n_codes"])
        if joint:
            cl_win = np.concatenate([f0.cluster, f1.cluster])
            G = len(np.unique(cl_win))
            fac = _cluster_factor(G, f0.n + f1.n, f0.k + f1.k, vce)
            c0_ = codes_all[pt["pos0"][f0.mask]]
            c1_ = codes_all[pt["pos1"][f1.mask]]
            s = np.bincount(c0_, weights=-r0, minlength=G_total) + np.bincount(
                c1_, weights=r1, minlength=G_total
            )
            vecs.append(("cl", math.sqrt(fac) * s))
        else:
            parts = []
            for f, r, pos, sign in (
                (f0, r0, pt["pos0"], -1.0),
                (f1, r1, pt["pos1"], 1.0),
            ):
                G = len(np.unique(f.cluster))
                fac = _cluster_factor(G, f.n, f.k, vce)
                parts.append(
                    sign
                    * math.sqrt(fac)
                    * np.bincount(codes_all[pos[f.mask]], weights=r, minlength=G_total)
                )
            vecs.append(("cl2", np.concatenate(parts)))
    M = np.vstack([v for _, v in vecs])
    cov = M @ M.T
    return 0.5 * (cov + cov.T)


def cer_rate(n: float, p: int) -> float:
    """Factor turning an MSE-rate bandwidth into a coverage-error-rate one.

    ``n^(1/(2p+4) - 1/(p+4))`` (two-dimensional rates; black box).
    """
    return float(n) ** (1.0 / (2 * p + 4) - 1.0 / (p + 4))
