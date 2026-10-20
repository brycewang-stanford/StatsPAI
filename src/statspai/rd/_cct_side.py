"""One side of an RD cutoff: kernel windows, local fits and their variances.

The building blocks shared by the CCT bandwidth selector
(:mod:`statspai.rd._cct_bandwidth`) and the robust bias-corrected estimator
(:mod:`statspai.rd._cct_inference`). Clean-room implementation written from
Calonico, Cattaneo and Titiunik (2014, bib key ``calonico2014robust``) and
Calonico, Cattaneo, Farrell and Titiunik (2019, Review of Economics and
Statistics 101(3), the covariate-adjusted estimator and its HC / cluster
variances, Supplemental Appendix Part IV); R ``rdrobust`` 4.0.0 (GPL-3) was
used only as a black box to pin the conventions listed in
``docs/dev/cleanroom/rd_cct.md``.

Notation. On one side of the cutoff the data are the running variable
``x`` (already centred at the cutoff) and a matrix of *columns* ``C``: the
outcome first, then the treatment indicator for a fuzzy design, then the
covariates. Every estimator in this module is linear in ``C``; the
statistic of interest is a fixed linear combination ``s`` of the columns
(``s = (1, -gamma)`` for covariate adjustment, the delta-method weights for
a fuzzy ratio), so a variance is the sandwich of the scalar
``a_i * (s' rho_i)`` where ``a_i`` is the observation's weight in the
estimator and ``rho_i`` its residual vector.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ..exceptions import NumericalInstability
from ._core import _kernel_fn, _spd_inverse
from ._nn_variance import nn_residuals

__all__ = ["Side", "LocalFit", "kernel_name", "complete_rows", "cluster_kind"]

#: With clusters every ``vce`` is one of R rdrobust 4.0.0's three cluster
#: variances: ``nn`` / ``hc0`` / ``hc1`` -> CR1, ``hc2`` -> CR2, ``hc3`` -> CR3
#: (black box; the help page of ``rdrobust`` states the mapping).
_CLUSTER_KIND = {
    "nn": "cr1",
    "hc0": "cr1",
    "hc1": "cr1",
    "cr1": "cr1",
    "hc2": "cr2",
    "cr2": "cr2",
    "hc3": "cr3",
    "cr3": "cr3",
}

_KERNEL_ALIASES = {
    "tri": "triangular",
    "triangular": "triangular",
    "uni": "uniform",
    "uniform": "uniform",
    "epa": "epanechnikov",
    "epanechnikov": "epanechnikov",
}


def kernel_name(kernel: str) -> str:
    """Canonical kernel name; raises ``ValueError`` for an unknown kernel."""
    key = str(kernel).lower()
    if key not in _KERNEL_ALIASES:
        raise ValueError(
            f"kernel must be 'triangular', 'uniform' or 'epanechnikov', got {kernel!r}"
        )
    return _KERNEL_ALIASES[key]


def complete_rows(*arrays: Optional[np.ndarray]) -> np.ndarray:
    """Rows where every given array (1-D or 2-D, ``None`` skipped) is finite."""
    keep: Optional[np.ndarray] = None
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.dtype.kind not in "fc":
            a = a.astype(float) if a.dtype.kind in "iub" else None
        if a is None:
            continue
        ok = np.isfinite(a) if a.ndim == 1 else np.isfinite(a).all(axis=1)
        keep = ok if keep is None else keep & ok
    return keep


class LocalFit:
    """Weighted least-squares polynomial fit of every column on one side.

    Attributes
    ----------
    mask : boolean array over the side's observations inside the window
        (positive kernel-times-user weight).
    x : running variable inside the window.
    design : ``[1, x, ..., x^order]`` inside the window.
    w : kernel weights ``K(x/h)/h`` (times user weights) inside the window.
    ginv : inverse of ``design' W design``.
    coef : ``(order + 1, n_columns)`` coefficients.
    singular : True when the Gram matrix was not positive definite and a
        pseudo-inverse was used.
    """

    __slots__ = ("mask", "x", "design", "w", "ginv", "coef", "order", "h", "singular")

    def __init__(self, side: "Side", h: float, order: int, kernel: str) -> None:
        k = side.kernel_weights(h, kernel)
        mask = k > 0
        x = side.x[mask]
        design = np.vander(x, order + 1, increasing=True)
        w = k[mask]
        gram = (design * w[:, None]).T @ design
        ginv, singular = _spd_inverse(gram)
        # A polynomial of order k needs k + 1 distinct support points with
        # positive weight; Cholesky can succeed on such a matrix in floating
        # point, so count them directly.
        singular = singular or np.unique(x).size < order + 1
        self.mask = mask
        self.x = x
        self.design = design
        self.w = w
        self.ginv = ginv
        self.coef = ginv @ ((design * w[:, None]).T @ side.cols[mask])
        self.order = order
        self.h = h
        self.singular = singular

    @property
    def n(self) -> int:
        """Number of observations in the window."""
        return int(self.x.size)

    def weights_for(self, ell: np.ndarray) -> np.ndarray:
        """Observation weights ``a_i`` with ``ell' coef = sum_i a_i C_i``."""
        return (self.design @ (self.ginv @ ell)) * self.w

    def bias_factor(self, ell: np.ndarray) -> float:
        """``ell' G^-1 sum_i w_i r_i x_i^(order+1)``.

        Multiplied by the coefficient of ``x^(order+1)`` in the true
        regression function this is the leading finite-sample bias of
        ``ell' coef`` (CCT 2014, Lemma 1).
        """
        target = (self.design * self.w[:, None]).T @ self.x ** (self.order + 1)
        return float(ell @ self.ginv @ target)

    def leverage(self) -> np.ndarray:
        """Diagonal of the weighted hat matrix inside the window."""
        return np.einsum("ij,jk,ik->i", self.design, self.ginv, self.design) * self.w


class Side:
    """Observations on one side of the cutoff, sorted by the running variable."""

    def __init__(
        self,
        x: np.ndarray,
        cols: np.ndarray,
        weights: Optional[np.ndarray] = None,
        cluster: Optional[np.ndarray] = None,
    ) -> None:
        order = np.argsort(x, kind="mergesort")
        self.x = np.asarray(x, dtype=float)[order]
        cols = np.asarray(cols, dtype=float)
        self.cols = (cols[:, None] if cols.ndim == 1 else cols)[order]
        self.weights = None if weights is None else np.asarray(weights, float)[order]
        self.cluster: Optional[np.ndarray] = None
        if cluster is not None:
            _, codes = np.unique(np.asarray(cluster)[order], return_inverse=True)
            self.cluster = codes

    @property
    def n(self) -> int:
        return int(self.x.size)

    def kernel_weights(self, h: float, kernel: str) -> np.ndarray:
        """``K(x/h)/h`` times the user weights, zero outside the window."""
        k = _kernel_fn(self.x / h, kernel) / h
        if self.weights is not None:
            k = k * self.weights
        return k

    def fit(self, h: float, order: int, kernel: str) -> LocalFit:
        return LocalFit(self, h, order, kernel)

    def subset_columns(self, ncols: int) -> "Side":
        """The same observations restricted to the first ``ncols`` columns."""
        out = Side.__new__(Side)
        out.x = self.x
        out.cols = self.cols[:, :ncols]
        out.weights = self.weights
        out.cluster = self.cluster
        return out


def residual_rows(
    side: Side,
    fit: LocalFit,
    vce: str,
    nnmatch: int,
    n_dof: Optional[int] = None,
) -> np.ndarray:
    """Residual vectors ``rho_i`` (one row per observation in the window).

    ``vce='nn'``: nearest-neighbour residuals among the observations of the
    window (:func:`statspai.rd._nn_variance.nn_residuals`). Otherwise the
    plug-in residuals of ``fit``, scaled by the HC weight: 1 (hc0),
    ``sqrt(n/(n-k))`` (hc1, ``n = n_dof`` or the window size, ``k`` the
    number of polynomial coefficients), ``1/sqrt(1-lev)`` (hc2) or
    ``1/(1-lev)`` (hc3). With clusters (:func:`cluster_kind`): CR1 uses the
    raw plug-in residuals and puts its small-sample factor on the clustered
    sum (:func:`variance_of`); CR2 / CR3 use
    :func:`cluster_adjusted_residuals`.
    """
    cols = side.cols[fit.mask]
    if side.cluster is not None:
        kind = cluster_kind(vce)
        resid = cols - fit.design @ fit.coef
        if kind == "cr1":
            return resid
        return cluster_adjusted_residuals(
            fit.design, fit.w, fit.ginv, side.cluster[fit.mask], resid, kind
        )
    if vce == "nn":
        ties = np.zeros(fit.n, dtype=int)
        return nn_residuals(fit.x, cols, ties, ties, nnmatch)
    resid = cols - fit.design @ fit.coef
    if vce == "hc0":
        return resid
    if vce == "hc1":
        n = fit.n if n_dof is None else n_dof
        k = fit.order + 1
        return resid * math.sqrt(n / (n - k))
    lev = fit.leverage()
    if vce == "hc2":
        return resid / np.sqrt(1.0 - lev)[:, None]
    if vce == "hc3":
        return resid / (1.0 - lev)[:, None]
    raise ValueError(f"unknown vce {vce!r}")


def cluster_kind(vce: str) -> str:
    """The cluster variance (``'cr1'``, ``'cr2'``, ``'cr3'``) a ``vce`` means
    once clusters are given."""
    try:
        return _CLUSTER_KIND[str(vce).lower()]
    except KeyError:
        raise ValueError(f"unknown vce {vce!r}") from None


def cluster_adjusted_residuals(
    design: np.ndarray,
    w: np.ndarray,
    ginv: np.ndarray,
    codes: np.ndarray,
    resid: np.ndarray,
    kind: str,
) -> np.ndarray:
    """CR2 / CR3 residual rows of one weighted polynomial fit.

    ``design`` / ``w`` / ``codes`` / ``resid`` are row-aligned; ``ginv`` is
    the inverse Gram matrix of the fit. Rows with ``w = 0`` (outside the
    fit's window but inside the union window of the two fits) may be
    included. Within each cluster ``g`` the residuals are multiplied by
    ``(I - H_gg)^(-1/2)`` (CR2, Bell and McCaffrey) or ``(I - H_gg)^(-1)``
    (CR3, the leave-one-cluster-out jackknife form), where
    ``H_gg = P_g G^-1 P_g' W_g`` is the cluster's block of the weighted hat
    matrix of this fit; the clustered sum of ``a_i s' rho_i`` then carries
    no small-sample factor.

    The matrix power is taken through the symmetric form
    ``S = W^(1/2) P G^-1 P' W^(1/2)`` on the positive-weight rows, ``f(I -
    H) = W^(-1/2) f(I - S) W^(1/2)``. Zero-weight rows have zero columns
    in ``H_gg`` but not zero rows, so ``f(I - H_gg)`` adds to their residual
    ``P_o G^-1 Z' g(S) W^(1/2) e`` with ``g(s) = (f(1 - s) - 1) / s``.

    Black box: equals R rdrobust 4.0.0's ``cr2`` / ``cr3`` to about 1e-13
    (the zero-weight rows matter for the robust variance when ``h > b``),
    the leverage of each variance coming from its own fit (order ``p`` at
    ``h`` for the conventional variance, order ``q`` at ``b`` for the
    robust one). A singleton cluster reduces to ``hc2`` / ``hc3``.
    """
    power = -0.5 if kind == "cr2" else -1.0
    out = np.array(resid, dtype=float, copy=True)
    pos = w > 0
    sw = np.sqrt(np.where(pos, w, 0.0))
    z = design * sw[:, None]
    order = np.argsort(codes, kind="mergesort")
    starts = np.flatnonzero(np.r_[True, np.diff(codes[order]) != 0])
    for lo, hi in zip(starts, np.r_[starts[1:], order.size]):
        idx = order[lo:hi]
        inside = idx[pos[idx]]
        if inside.size == 0:
            continue
        outside = idx[~pos[idx]]
        zi = z[inside]
        lam, vecs = np.linalg.eigh(zi @ ginv @ zi.T)
        if (1.0 - lam).min() <= 1e-10:
            _singular_cluster(design.shape[1] - 1)
        t = sw[inside, None] * resid[inside]
        coef = vecs.T @ t
        out[inside] = (vecs @ (((1.0 - lam) ** power)[:, None] * coef)) / sw[
            inside, None
        ]
        if outside.size:
            small = np.abs(lam) < 1e-8
            safe = np.where(small, 1.0, lam)
            g = np.where(small, -power, ((1.0 - lam) ** power - 1.0) / safe)
            out[outside] += design[outside] @ ginv @ zi.T @ (vecs @ (g[:, None] * coef))
    return out


def _singular_cluster(order: int) -> None:
    raise NumericalInstability(
        "rdrobust: a cluster determines a local polynomial fit of order "
        f"{order} by itself: its block of the hat matrix has an eigenvalue "
        "of one, so the CR2 / CR3 leverage correction is not defined.",
        recovery_hint="Use vce='cr1' (or 'hc1' with cluster=), a wider "
        "bandwidth, or more clusters.",
        diagnostics={"order": order},
    )


def variance_of(
    side: Side,
    mask: np.ndarray,
    a: np.ndarray,
    rho: np.ndarray,
    s: np.ndarray,
    n_coef: int,
    vce: str = "hc1",
) -> float:
    """Sandwich variance of ``sum_i a_i s' C_i`` given residual rows ``rho``.

    Heteroskedasticity-robust: ``sum_i (a_i s' rho_i)^2``. Clustered: the
    squared cluster sums; for CR1 (``vce`` ``nn`` / ``hc0`` / ``hc1`` /
    ``cr1``) times ``G/(G-1) * (n-1)/(n-n_coef)``, with ``n`` and ``G`` the
    observations and clusters among the rows of ``mask``. CR2 / CR3 carry
    their correction in ``rho`` (:func:`cluster_adjusted_residuals`).
    """
    u = a * (rho @ s)
    if side.cluster is None:
        return float(np.sum(u * u))
    codes = side.cluster[mask]
    _, inv = np.unique(codes, return_inverse=True)
    sums = np.bincount(inv, weights=u)
    if cluster_kind(vce) != "cr1":
        return float(np.sum(sums * sums))
    g = sums.size
    n = codes.size
    factor = (g / (g - 1.0)) * ((n - 1.0) / (n - n_coef)) if g > 1 else 1.0
    return float(factor * np.sum(sums * sums))


def n_clusters(side: Side, mask: Optional[np.ndarray] = None) -> int:
    """Distinct clusters among the side's observations (or ``mask`` rows)."""
    if side.cluster is None:
        return 0
    codes = side.cluster if mask is None else side.cluster[mask]
    return int(np.unique(codes).size)


def partial_out_covariates(
    side: Side, fit: LocalFit, n_core: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Covariate slopes of one side's fit, and the adjusted coefficients.

    Returns ``(gamma, adjusted)``: ``gamma`` is ``(d, n_core)``, the WLS
    slope of each core column (outcome, treatment) on the covariates after
    both are residualised on the side's polynomial; ``adjusted`` is the
    ``(order + 1, n_core)`` polynomial coefficients of the core columns net
    of ``coef_Z gamma``.
    """
    d = side.cols.shape[1] - n_core
    core = fit.coef[:, :n_core]
    if d == 0:
        return np.zeros((0, n_core)), core
    resid = side.cols[fit.mask] - fit.design @ fit.coef
    z, y = resid[:, n_core:], resid[:, :n_core]
    zw = z * fit.w[:, None]
    gamma = np.linalg.solve(zw.T @ z, zw.T @ y)
    return gamma, core - fit.coef[:, n_core:] @ gamma
