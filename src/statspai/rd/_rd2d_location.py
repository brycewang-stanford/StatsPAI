"""Location-based boundary discontinuity estimation (port of R ``rd2d``).

Numerical core behind :func:`statspai.rd2d` with ``approach="location"``
and :func:`statspai.rd2d_bw` with ``approach="location"``.  Each function
below mirrors one routine of the R package ``rd2d`` 1.0.0 (Cattaneo,
Titiunik and Yu) -- the R name is given in every docstring -- and keeps
its order of operations, so the two agree to machine precision on the
same data bytes (``tests/reference_parity/test_rd2d_R_parity.py``).

Scope of the port: sharp and fuzzy designs, product and radial kernels,
``vce`` in {hc0, hc1, hc2, hc3}, cluster-robust variances, joint and
separate fits, every ``bwselect`` (MSE / CER, common / two-sided,
pointwise / integrated), ``method`` in {dpi, rot}, mass-point handling,
``bwcheck``, derivatives and tangential derivatives.  Covariate-adjusted
fits (R's ``covs.eff``) are not ported.

All arrays here are already NA-free; the public wrapper does the
cleaning.  ``x1``/``x2`` are the raw running variables, ``d`` is the 0/1
assignment and ``Y`` is an ``(n, m)`` matrix of outcomes (``m = 2`` for
a fuzzy design: outcome, then treatment take-up).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import DataInsufficient

__all__ = [
    "kernel_weight",
    "n_basis",
    "rd2d_location_fit",
    "rd2d_location_bw",
]

_KERNEL_CANON = {
    "tri": "triangular",
    "triangular": "triangular",
    "epa": "epanechnikov",
    "epanechnikov": "epanechnikov",
    "uni": "uniform",
    "uniform": "uniform",
    "gau": "gaussian",
    "gaussian": "gaussian",
}


# ----------------------------------------------------------------------
# Small primitives
# ----------------------------------------------------------------------


def kernel_weight(u: np.ndarray, kernel: str) -> np.ndarray:
    """R ``kernel_weight``: K(u) with the rd2d normalisations."""
    k = _KERNEL_CANON[kernel]
    au = np.abs(u)
    if k == "triangular":
        return (1.0 - au) * (au <= 1)
    if k == "uniform":
        return 0.5 * (au <= 1)
    if k == "gaussian":
        return np.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)
    return 0.75 * (1.0 - u * u) * (au <= 1)


def n_basis(p: int) -> int:
    """Number of bivariate monomials of total degree <= p."""
    return (p + 1) * (p + 2) // 2


def basis_xy(u1: np.ndarray, u2: np.ndarray, p: int) -> np.ndarray:
    """R ``get_basis_xy``: 1, u1, u2, u1^2, u1 u2, u2^2, ..."""
    n = len(u1)
    out = np.empty((n, n_basis(p)))
    out[:, 0] = 1.0
    c = 1
    for j in range(1, p + 1):
        for k in range(j + 1):
            out[:, c] = u1 ** (j - k) * u2**k
            c += 1
    return out


def h_diag(hx: float, hy: float, p: int, inverse: bool = False) -> np.ndarray:
    """Diagonal of R ``get_H`` (or ``get_invH``)."""
    out = np.empty(n_basis(p))
    out[0] = 1.0
    c = 1
    for j in range(1, p + 1):
        for k in range(j + 1):
            v = hx ** (j - k) * hy**k
            out[c] = 1.0 / v if inverse else v
            c += 1
    return out


def xx_inv(x: np.ndarray) -> np.ndarray:
    """R ``qrXXinv``: ``chol2inv(chol(crossprod(x)))`` with a g-inverse fallback."""
    mat = x.T @ x
    try:
        L = np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Calculating inverse of (t(X)%*%X), matrix is not positive-definite. "
            "Using generalized inverse.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.linalg.pinv(mat)
    Linv = np.linalg.inv(L)
    return Linv.T @ Linv


def cluster_sums(
    values: np.ndarray, eC: np.ndarray, clusters: np.ndarray
) -> np.ndarray:
    """R ``rd2d_cluster_sums``: rows of ``values`` summed within ``clusters``."""
    values = np.atleast_2d(values.T).T if values.ndim == 1 else values
    out = np.zeros((len(clusters), values.shape[1]))
    if len(eC) == 0:
        return out
    pos = {c: i for i, c in enumerate(clusters.tolist())}
    idx = np.fromiter((pos[c] for c in eC.tolist()), dtype=int, count=len(eC))
    np.add.at(out, idx, values)
    return out


def unique_in_order(x: np.ndarray) -> np.ndarray:
    """R ``unique``: first-appearance order."""
    _, first = np.unique(x, return_index=True)
    return x[np.sort(first)]


def vce_multiplier(
    vce: str, eN: int, k: int, sqrtw_R: np.ndarray, invG: np.ndarray, where: str
) -> np.ndarray | float:
    """Residual multiplier for HC0-HC3 (R's ``w.vce``)."""
    if vce == "hc0":
        return 1.0
    if eN <= k:
        warnings.warn(
            f"vce='{vce}' may be undefined {where} because the effective sample "
            f"size eN={eN} is not larger than the number of basis terms k={k}. "
            "Increase the bandwidth or bwcheck, lower p/q, or use vce='hc0'.",
            RuntimeWarning,
            stacklevel=3,
        )
    if vce == "hc1":
        return math.sqrt(eN / (eN - k))
    hii = np.sum((sqrtw_R @ invG) * sqrtw_R, axis=1)
    if vce == "hc2":
        return np.sqrt(1.0 / (1.0 - hii))
    return 1.0 / (1.0 - hii)


def joint_scale(
    vce: str, eN: int, k: int, eC: Optional[np.ndarray], clustered: bool
) -> float:
    """R ``rd2d_joint_scale``."""
    scale = 1.0
    if vce == "hc1":
        scale *= math.sqrt(eN / (eN - k))
    if clustered:
        g = len(np.unique(eC))
        scale *= math.sqrt(((eN - 1) / (eN - k)) * (g / (g - 1)))
    return scale


def kth_smallest(x: np.ndarray, k: int) -> float:
    """``sort.int(x, partial = k)[k]`` (1-indexed ``k``)."""
    return float(np.partition(x, k - 1)[k - 1])


def unique_locations(x1: np.ndarray, x2: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Row indices of R ``rd2d_unique``: last row of each run of equal (x1, x2)."""
    order = np.lexsort((x2, x1))
    a1, a2 = x1[order], x2[order]
    n = len(order)
    if n == 0:
        return order
    keep = np.ones(n, dtype=bool)
    keep[:-1] = (a1[1:] != a1[:-1]) | (a2[1:] != a2[:-1])
    return order[keep]


# ----------------------------------------------------------------------
# Local design and fits
# ----------------------------------------------------------------------


class _Side:
    """Centred data of one side of the boundary for one evaluation point."""

    __slots__ = ("x1", "x2", "dist", "Y", "cl")

    def __init__(self, x1: Any, x2: Any, dist: Any, Y: Any, cl: Any) -> None:
        self.x1, self.x2, self.dist, self.Y, self.cl = x1, x2, dist, Y, cl


def _h_normalize(h: Any, kernel_type: str) -> Tuple[np.ndarray, Tuple[float, float]]:
    h = np.atleast_1d(np.asarray(h, dtype=float))
    if kernel_type == "prod":
        if h.size == 1:
            h = np.array([h[0], h[0]])
        return h, (float(h[0]), float(h[1]))
    if h.size == 2:
        h = np.array([math.sqrt(h[0] ** 2 + h[1] ** 2)])
    return h, (float(h[0]), float(h[0]))


def _weights(s: _Side, h: Any, kernel: str, kernel_type: str) -> np.ndarray:
    h, _ = _h_normalize(h, kernel_type)
    if kernel_type == "prod":
        return (
            kernel_weight(s.x1 / h[0], kernel)
            * kernel_weight(s.x2 / h[1], kernel)
            / (h[0] * h[1])
        )
    return kernel_weight(s.dist / h[0], kernel) / h[0] ** 2


def _local_design(s: _Side, h: Any, p: int, kernel: str, kernel_type: str) -> dict:
    """R ``rd2d_local_design``."""
    _, hxy = _h_normalize(h, kernel_type)
    w = _weights(s, h, kernel, kernel_type)
    ind = w > 0
    ew = w[ind]
    sqrt_ew = np.sqrt(ew)
    eR = basis_xy(s.x1[ind] / hxy[0], s.x2[ind] / hxy[1], p)
    sqrtw_R = sqrt_ew[:, None] * eR
    return {
        "hxy": hxy,
        "ind": ind,
        "eN": int(ind.sum()),
        "ew": ew,
        "sqrt_ew": sqrt_ew,
        "eY": s.Y[ind],
        "eC": None if s.cl is None else s.cl[ind],
        "eR": eR,
        "sqrtw_R": sqrtw_R,
        "w_R": ew[:, None] * eR,
        "k": eR.shape[1],
        "invG": xx_inv(sqrtw_R),
        "H": h_diag(hxy[0], hxy[1], p),
        "invH": h_diag(hxy[0], hxy[1], p, inverse=True),
    }


def _lm(
    s: _Side, h: Any, p: int, vce: str, kernel: str, kernel_type: str, varr: bool
) -> Any:
    """R ``rd2d_lm`` / ``rd2d_lm_multi``: beta (k x m) and per-outcome cov.const."""
    loc = _local_design(s, h, p, kernel, kernel_type)
    sqrtw_Y = loc["sqrt_ew"][:, None] * loc["eY"]
    beta = loc["invH"][:, None] * (loc["invG"] @ (loc["sqrtw_R"].T @ sqrtw_Y))
    covs: List[np.ndarray] = []
    if varr:
        resd = loc["eY"] - (loc["eR"] * loc["H"][None, :]) @ beta
        wv = vce_multiplier(
            vce, loc["eN"], loc["k"], loc["sqrtw_R"], loc["invG"], "in rd2d_lm()"
        )
        resd = resd * (wv[:, None] if isinstance(wv, np.ndarray) else wv)
        hx, hy = loc["hxy"]
        for j in range(resd.shape[1]):
            if loc["eC"] is None:
                sig = resd[:, j][:, None] * loc["w_R"]
                sigma = (sig.T @ sig) * hx * hy
            else:
                n, k = len(loc["eC"]), loc["w_R"].shape[1]
                cl = unique_in_order(loc["eC"])
                g = len(cl)
                ww = ((n - 1) / (n - k)) * (g / (g - 1))
                sc = cluster_sums(loc["w_R"] * resd[:, j][:, None], loc["eC"], cl)
                sigma = (sc.T @ sc) * hx * hy * ww
            covs.append(loc["invG"].T @ (sigma @ loc["invG"]))
    return beta, covs, loc["eN"]


def _cov_half(
    s: _Side,
    h: Any,
    p: int,
    vce: str,
    kernel: str,
    kernel_type: str,
    clusters: Optional[np.ndarray],
    cluster_df: bool,
) -> Any:
    """R ``get_cov_half_v2`` / ``get_cov_half_multi_v2``: (ind, half[:, k*m])."""
    loc = _local_design(s, h, p, kernel, kernel_type)
    sqrtw_Y = loc["sqrt_ew"][:, None] * loc["eY"]
    beta = loc["invH"][:, None] * (loc["invG"] @ (loc["sqrtw_R"].T @ sqrtw_Y))
    resd = loc["eY"] - loc["eR"] @ (loc["H"][:, None] * beta)
    wv = vce_multiplier(
        vce, loc["eN"], loc["k"], loc["sqrtw_R"], loc["invG"], "in get_cov_half_v2()"
    )
    resd = resd * (wv[:, None] if isinstance(wv, np.ndarray) else wv)
    hx, hy = loc["hxy"]
    k = loc["k"]
    m = resd.shape[1]
    blocks = []
    if s.cl is None:
        for j in range(m):
            sig = (
                (loc["sqrt_ew"] * resd[:, j])[:, None]
                * loc["sqrtw_R"]
                * math.sqrt(hx * hy)
            )
            blocks.append(sig @ loc["invG"])
    else:
        n = len(loc["eC"])
        g_eff = len(np.unique(loc["eC"]))
        ww = ((n - 1) / (n - k)) * (g_eff / (g_eff - 1)) if cluster_df else 1.0
        for j in range(m):
            sc = cluster_sums(loc["w_R"] * resd[:, j][:, None], loc["eC"], clusters)
            blocks.append((sc * math.sqrt(hx * hy * ww)) @ loc["invG"])
    return loc["ind"], np.hstack(blocks)


# ----------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------


class Rd2dData:
    """NA-free design: coordinates, assignment, outcome matrix, clusters."""

    def __init__(
        self, x1: Any, x2: Any, d: Any, Y: Any, cluster: Optional[Any] = None
    ) -> None:
        self.x1 = np.asarray(x1, dtype=float)
        self.x2 = np.asarray(x2, dtype=float)
        self.d = np.asarray(d, dtype=float)
        Y = np.asarray(Y, dtype=float)
        self.Y = Y[:, None] if Y.ndim == 1 else Y
        self.cluster = None if cluster is None else np.asarray(cluster)
        self.s0 = self.d == 0
        self.s1 = self.d == 1

    @property
    def n(self) -> int:
        return len(self.x1)

    def side(
        self,
        side: int,
        ev: Tuple[float, float],
        metric: str,
        sd: Any = (1.0, 1.0),
        Y: Optional[Any] = None,
    ) -> _Side:
        mask = self.s0 if side == 0 else self.s1
        dx1 = self.x1[mask] - ev[0]
        dx2 = self.x2[mask] - ev[1]
        if metric == "prod":
            dist = np.maximum(np.abs(dx1 / sd[0]), np.abs(dx2 / sd[1]))
        else:
            dist = np.sqrt(dx1**2 + dx2**2)
        Yall = self.Y if Y is None else Y
        cl = None if self.cluster is None else self.cluster[mask]
        return _Side(dx1, dx2, dist, Yall[mask], cl)


def bwcheck_bounds(
    x1: Any,
    x2: Any,
    d: Any,
    eval_pts: Any,
    bwcheck: int,
    masspoints: str,
    metric: str,
    scale: Any = (1.0, 1.0),
) -> np.ndarray:
    """R ``rd2d_bwcheck_bounds``: columns min.0, min.1, max.0, max.1."""
    if masspoints == "adjust":
        u = unique_locations(x1, x2, d)
        x1, x2, d = x1[u], x2[u], d[u]
    out = np.empty((len(eval_pts), 4))
    s0, s1 = d == 0, d == 1
    for i, (b1, b2) in enumerate(eval_pts):
        dx1, dx2 = x1 - b1, x2 - b2
        if metric == "prod":
            dist = np.maximum(np.abs(dx1 / scale[0]), np.abs(dx2 / scale[1]))
        else:
            dist = np.sqrt(dx1**2 + dx2**2)
        d0, d1 = dist[s0], dist[s1]
        if len(d0) < bwcheck or len(d1) < bwcheck:
            unit = "unique mass points" if masspoints == "adjust" else "observations"
            raise DataInsufficient(
                f"bwcheck={bwcheck} is larger than the available {unit} at evaluation "
                f"point {i + 1}: {len(d0)} on the control side and {len(d1)} on the "
                "treatment side. Decrease bwcheck or set bwcheck=None.",
                recovery_hint="Decrease bwcheck or set bwcheck=None.",
            )
        out[i] = (
            kth_smallest(d0, bwcheck),
            kth_smallest(d1, bwcheck),
            float(d0.max()),
            float(d1.max()),
        )
    return out


def sd(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1))


# ----------------------------------------------------------------------
# Point estimation (R rd2d_fit / rd2d_fit_multi)
# ----------------------------------------------------------------------


def fit(
    data: Rd2dData,
    eval_pts: np.ndarray,
    e_deriv: np.ndarray,
    p: int,
    hgrid0: np.ndarray,
    hgrid1: np.ndarray,
    kernel: str,
    kernel_type: str,
    vce: str,
    bw_bounds: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """R ``rd2d_fit`` (m = 1) / ``rd2d_fit_multi`` (m = 2).

    Returns ``mu0``/``mu1``/``se0``/``se1`` as ``(neval, m)`` arrays and the
    bandwidths actually used (after ``bwcheck`` clamping) and effective
    sample sizes as ``(neval,)`` arrays.
    """
    sd1, sd2 = sd(data.x1), sd(data.x2)
    neval = len(eval_pts)
    m = data.Y.shape[1]
    out = {k: np.empty((neval, m)) for k in ("mu0", "mu1", "se0", "se1")}
    for k in ("h0x", "h0y", "h1x", "h1y", "eN0", "eN1"):
        out[k] = np.empty(neval)
    mult = np.array([sd1, sd2]) if kernel_type == "prod" else np.array([1.0, 1.0])
    for i in range(neval):
        ev = (float(eval_pts[i, 0]), float(eval_pts[i, 1]))
        vec = e_deriv[i]
        h0 = np.asarray(hgrid0[i], dtype=float)
        h1 = np.asarray(hgrid1[i], dtype=float)
        if bw_bounds is not None:
            mn0, mn1, mx0, mx1 = bw_bounds[i]
            h0 = np.minimum(np.maximum(h0, mn0 * mult), mx0 * mult)
            h1 = np.minimum(np.maximum(h1, mn1 * mult), mx1 * mult)
        side0 = data.side(0, ev, kernel_type, (sd1, sd2))
        side1 = data.side(1, ev, kernel_type, (sd1, sd2))
        b0, c0, n0 = _lm(side0, h0, p, vce, kernel, kernel_type, True)
        b1, c1, n1 = _lm(side1, h1, p, vce, kernel, kernel_type, True)
        h0x, h0y = (float(h0[0]), float(h0[-1]))
        h1x, h1y = (float(h1[0]), float(h1[-1]))
        # Scale the variance with the bandwidths the fit actually used.  For
        # the radial kernel that is the radius sqrt(hx^2 + hy^2) (R's
        # rd2d_h_normalize); R rescales with hx * hy instead, which inflates
        # its radial-kernel variances by (hx^2 + hy^2) / (hx hy).  See
        # docs/dev/campaign_phase3/r2_rd_open.md.
        s0x, s0y = _h_normalize(h0, kernel_type)[1]
        s1x, s1y = _h_normalize(h1, kernel_type)[1]
        iH0 = h_diag(s0x, s0y, p, inverse=True)
        iH1 = h_diag(s1x, s1y, p, inverse=True)
        v0, v1 = iH0 * vec, iH1 * vec
        for j in range(m):
            out["mu0"][i, j] = float(vec @ b0[:, j])
            out["mu1"][i, j] = float(vec @ b1[:, j])
            out["se0"][i, j] = math.sqrt(float(v0 @ c0[j] @ v0) / (s0x * s0y))
            out["se1"][i, j] = math.sqrt(float(v1 @ c1[j] @ v1) / (s1x * s1y))
        out["h0x"][i], out["h0y"][i], out["h1x"][i], out["h1y"][i] = h0x, h0y, h1x, h1y
        out["eN0"][i], out["eN1"][i] = n0, n1
    return out


# ----------------------------------------------------------------------
# Covariance across evaluation points (R rd2d_cov_project_sides & co.)
# ----------------------------------------------------------------------


def project_sides(
    data: Rd2dData,
    eval_pts: np.ndarray,
    e_deriv: np.ndarray,
    p: int,
    hgrid0: np.ndarray,
    hgrid1: np.ndarray,
    kernel: str,
    kernel_type: str,
    vce: str,
    fitmethod: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], bool]:
    """R ``rd2d_cov_project_sides`` + ``rd2d_project_cov_halves``.

    Returns, per outcome column, the projected influence matrices of the
    control and treated sides (rows = observations of that side, or
    clusters; columns = evaluation points).
    """
    clustered = data.cluster is not None
    joint = fitmethod == "joint"
    vce_half = "hc0" if (joint and vce == "hc1") else vce
    cluster_df = not (joint and clustered)
    clusters = unique_in_order(data.cluster) if clustered else None
    m = data.Y.shape[1]
    kd = e_deriv.shape[1]
    neval = len(eval_pts)
    nrow0 = len(clusters) if clustered else int(data.s0.sum())
    nrow1 = len(clusters) if clustered else int(data.s1.sum())
    P0 = [np.zeros((nrow0, neval)) for _ in range(m)]
    P1 = [np.zeros((nrow1, neval)) for _ in range(m)]
    for i in range(neval):
        ev = (float(eval_pts[i, 0]), float(eval_pts[i, 1]))
        h0 = np.asarray(hgrid0[i], dtype=float)
        h1 = np.asarray(hgrid1[i], dtype=float)
        side0 = data.side(0, ev, "rad")
        side1 = data.side(1, ev, "rad")
        ind0, half0 = _cov_half(
            side0, h0, p, vce_half, kernel, kernel_type, clusters, cluster_df
        )
        ind1, half1 = _cov_half(
            side1, h1, p, vce_half, kernel, kernel_type, clusters, cluster_df
        )
        if joint and (vce == "hc1" or clustered):
            eC = None
            if clustered:
                eC = np.concatenate([side0.cl[ind0], side1.cl[ind1]])
            sc = joint_scale(vce, int(ind0.sum() + ind1.sum()), 2 * kd, eC, clustered)
            half0 = half0 * sc
            half1 = half1 * sc
        for side, half, ind, h, P in (
            (0, half0, ind0, h0, P0),
            (1, half1, ind1, h1, P1),
        ):
            hxy = _h_normalize(h, kernel_type)[1]  # see the note in fit()
            v = h_diag(hxy[0], hxy[1], p, inverse=True) * e_deriv[i]
            for j in range(m):
                score = (half[:, j * kd : (j + 1) * kd] @ v) / math.sqrt(
                    hxy[0] * hxy[1]
                )
                if clustered:
                    P[j][:, i] = score
                else:
                    P[j][ind, i] = score
    return P0, P1, clustered


def cov_from_projects(
    A0: Any, A1: Any, B0: Any, B1: Any, clustered_joint: bool
) -> np.ndarray:
    """Cross-covariance of two projected estimators (sides combined)."""
    if clustered_joint:
        return (A1 - A0).T @ (B1 - B0)
    return A0.T @ B0 + A1.T @ B1


def _sym(a: np.ndarray) -> np.ndarray:
    return (a + a.T) / 2.0


# ----------------------------------------------------------------------
# Bandwidth selection (R rdbw2d and helpers)
# ----------------------------------------------------------------------

_ROT_CONSTS = {
    "epanechnikov": (1 / 6, 4 / (3 * math.pi)),
    "triangular": (3 / 20, 3 / (2 * math.pi)),
    "uniform": (1 / 4, 1 / math.pi),
    "gaussian": (1.0, 1 / (4 * math.pi)),
}


def _rot(x1: np.ndarray, x2: np.ndarray, kernel: str, M: int) -> float:
    """R ``rdbw2d_rot``: bivariate rule-of-thumb pilot bandwidth."""
    mu2K, l2K = _ROT_CONSTS[_KERNEL_CANON[kernel]]
    S = np.cov(np.vstack([x1, x2]))
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S, rcond=1e-20)
    det = float(np.linalg.det(S))
    if not (np.isfinite(det) and det > 0):
        from scipy.linalg import sqrtm

        sqrt_det = float(np.real(np.linalg.det(sqrtm(S))))
    else:
        sqrt_det = math.sqrt(det)
    trace_const = (
        1.0
        / (2**4 * math.pi**1 * sqrt_det)
        * (2 * float(np.trace(Sinv @ Sinv)) + float(np.trace(Sinv)) ** 2)
    )
    return ((2 * l2K) / (M * mu2K * trace_const)) ** (1 / 6)


def _get_coeff(
    s: _Side, vec: np.ndarray, p: int, dn: float, kernel: str, kernel_type: str
) -> np.ndarray:
    """R ``get_coeff``: projection of the (p+1)-th order terms (padded)."""
    if kernel_type == "prod":
        w = (
            kernel_weight(s.x1 / dn, kernel)
            * kernel_weight(s.x2 / dn, kernel)
            / (dn * dn)
        )
    else:
        w = kernel_weight(s.dist / dn, kernel) / dn**2
    ind = w > 0
    ew = w[ind]
    aug = basis_xy(s.x1[ind] / dn, s.x2[ind] / dn, p + 1)
    kp, kp1 = n_basis(p), n_basis(p + 1)
    sw = np.sqrt(ew)[:, None]
    sR = sw * aug[:, :kp]
    sS = sw * aug[:, kp:kp1]
    invG = xx_inv(sR)
    vq = vec @ invG @ sR.T @ sS
    return np.concatenate([np.zeros(kp), vq])


def _joint_info(
    s0: _Side, s1: _Side, h0: Any, h1: Any, kernel: Any, kernel_type: Any
) -> Any:
    """R ``rdbw2d_joint_effective_info``."""
    i0 = _weights(s0, h0, kernel, kernel_type) > 0
    i1 = _weights(s1, h1, kernel, kernel_type) > 0
    eC = None
    if s0.cl is not None or s1.cl is not None:
        eC = np.concatenate([s0.cl[i0], s1.cl[i1]])
    return int(i0.sum() + i1.sum()), eC


def _bw_consts(
    s: _Side,
    p: int,
    vec: np.ndarray,
    dn: float,
    bn1: float,
    bn2: Optional[float],
    vce: str,
    kernel: str,
    kernel_type: str,
    fitmethod: str,
    clusters: Optional[np.ndarray],
    jinfo_v: Any,
    jinfo_b: Any,
) -> Dict[str, object]:
    """R ``rdbw2d_bw_v2`` (no covariates)."""
    joint = fitmethod == "joint"
    clustered = s.cl is not None
    vce_local = "hc0" if (joint and vce == "hc1") else vce
    cluster_df = not (joint and clustered)
    y = s.Y[:, 0]
    if kernel_type == "prod":
        w = kernel_weight(s.x1 / dn, kernel) * kernel_weight(s.x2 / dn, kernel) / dn**2
    else:
        w = kernel_weight(s.dist / dn, kernel) / dn**2
    ind = w > 0
    eN = int(ind.sum())
    ew = w[ind]
    eY = y[ind]
    eC = None if s.cl is None else s.cl[ind]
    u1, u2 = s.x1[ind] / dn, s.x2[ind] / dn
    kp, kp1, kp2 = n_basis(p), n_basis(p + 1), n_basis(p + 2)
    aug = basis_xy(u1, u2, p + 1 if bn2 is None else p + 2)
    eR = aug[:, :kp]
    sw = np.sqrt(ew)[:, None]
    sR = sw * eR
    sS = sw * aug[:, kp:kp1]
    sY = np.sqrt(ew) * eY
    wR = ew[:, None] * eR
    k = eR.shape[1]
    invG = xx_inv(sR)
    vq = np.concatenate([np.zeros(kp), vec @ invG @ sR.T @ sS])
    vt = None
    if bn2 is not None:
        sT = sw * aug[:, kp1:kp2]
        vt = np.concatenate([np.zeros(kp1), vec @ invG @ sR.T @ sT])
    iH = h_diag(dn, dn, p, inverse=True)
    H = h_diag(dn, dn, p)
    beta = iH * (invG @ (sR.T @ sY))
    resd = eY - eR @ (H * beta)
    wv = vce_multiplier(vce_local, eN, k, sR, invG, "in the bandwidth selector")
    resd = resd * wv
    if clustered:
        g = len(np.unique(eC))
        csc = ((eN - 1) / (eN - k)) * (g / (g - 1)) if cluster_df else 1.0
        half = cluster_sums(wR * resd[:, None], eC, clusters) * math.sqrt(dn**2 * csc)
        half = half @ invG
    else:
        half = ((np.sqrt(ew) * resd)[:, None] * sR * dn) @ invG
    if joint and (vce == "hc1" or clustered):
        half = half * joint_scale(vce, jinfo_v[0], 2 * k, jinfo_v[1], clustered)
    V_half = half @ vec
    V = float(V_half @ V_half)

    sb = _Side(s.x1, s.x2, s.dist, s.Y[:, :1], s.cl)
    bb, _, _ = _lm(sb, bn1, p + 1, vce_local, kernel, kernel_type, False)
    B = float(vq @ bb[:, 0])
    ind_b, bias_half = _cov_half(
        sb, bn1, p + 1, vce_local, kernel, kernel_type, clusters, cluster_df
    )
    if joint and (vce == "hc1" or clustered):
        bias_half = bias_half * joint_scale(
            vce, jinfo_b[0], 2 * bias_half.shape[1], jinfo_b[1], clustered
        )
    Reg1_half = (bias_half @ vq) / bn1 ** (p + 2)
    Reg1 = float(Reg1_half @ Reg1_half)
    Reg2 = float("nan")
    if bn2 is not None:
        bt, _, _ = _lm(sb, bn2, p + 2, vce_local, kernel, kernel_type, False)
        Reg2 = float(dn * (vt @ bt[:, 0]))
    return {
        "B": B,
        "V": V,
        "Reg2": Reg2,
        "Reg1": Reg1,
        "V_half": V_half,
        "Reg1_half": Reg1_half,
    }


def _bwselect_base(bwselect: str) -> str:
    return {
        "cerrd": "mserd",
        "certwo": "msetwo",
        "icerrd": "imserd",
        "icertwo": "imsetwo",
    }.get(bwselect, bwselect)


def _is_cer(bwselect: str) -> bool:
    return bwselect in ("cerrd", "certwo", "icerrd", "icertwo")


def _is_common(bwselect: str) -> bool:
    return _bwselect_base(bwselect) in ("mserd", "imserd")


def cer_factor(n: float, p: int) -> float:
    """R ``rd2d_cer_factor``."""
    return n ** (1 / (2 * p + 4) - 1 / (p + 4))


def e_deriv_matrix(
    neval: int, p: int, deriv: Sequence[int], tangvec: Any
) -> np.ndarray:
    """R's ``e_deriv``: selection vector of the target derivative."""
    e = np.zeros((neval, n_basis(p)))
    ds = int(deriv[0]) + int(deriv[1])
    if tangvec is not None:
        e[:, 1] = tangvec[:, 0]
        e[:, 2] = tangvec[:, 1]
        return e
    if ds >= 1:
        idx = (math.factorial(ds + 1) // (math.factorial(ds - 1) * 2)) + int(deriv[1])
        e[:, idx] = math.factorial(int(deriv[0])) * math.factorial(int(deriv[1]))
    else:
        e[:, 0] = 1.0
    return e


def rd2d_location_bw(
    data: Rd2dData,
    eval_pts: np.ndarray,
    p: int,
    deriv: Sequence[int],
    tangvec: Any,
    kernel: str,
    kernel_type: str,
    bwselect: str,
    method: str,
    vce: str,
    bwcheck: Optional[int],
    masspoints: str,
    fitmethod: str,
    scaleregul: float,
    scalebiascrct: float,
    stdvars: bool,
    bwparam: str = "main",
) -> Dict[str, object]:
    """R ``rdbw2d``: MSE/CER-optimal bandwidths at every evaluation point.

    Returns ``bws`` (neval x 4: h01, h02, h11, h12 in the units of x1/x2)
    and the MSE constants table.
    """
    if tangvec is not None:
        deriv = (1, 0)
    ds = int(deriv[0]) + int(deriv[1])
    e_deriv = e_deriv_matrix(len(eval_pts), p, deriv, tangvec)
    deriv_denom = 2 * p + 2 - 2 * ds
    fuzzy = data.Y.shape[1] == 2
    if stdvars:
        sd1, sd2 = sd(data.x1), sd(data.x2)
    else:
        sd1 = sd2 = 1.0
    x1, x2 = data.x1 / sd1, data.x2 / sd2
    ev_std = np.column_stack([eval_pts[:, 0] / sd1, eval_pts[:, 1] / sd2])
    sdat = Rd2dData(x1, x2, data.d, data.Y, data.cluster)
    N = sdat.n
    M, M0, M1 = N, int(sdat.s0.sum()), int(sdat.s1.sum())
    if masspoints in ("check", "adjust"):
        u = unique_locations(x1, x2, data.d)
        M0 = int((data.d[u] == 0).sum())
        M1 = int((data.d[u] == 1).sum())
        M = M0 + M1
        if 1 - M / N >= 0.2:
            warnings.warn(
                "Mass points detected in the running variables.",
                RuntimeWarning,
                stacklevel=3,
            )
            if masspoints == "check":
                warnings.warn(
                    "Try using option masspoints=adjust.", RuntimeWarning, stacklevel=3
                )
            if bwcheck is None:
                bwcheck = 50 + p + 1
    dn = _rot(x1, x2, kernel, M)
    bounds = None
    if bwcheck is not None:
        bounds = bwcheck_bounds(x1, x2, data.d, ev_std, bwcheck, masspoints, "rad")
    clusters = unique_in_order(data.cluster) if data.cluster is not None else None
    base = _bwselect_base(bwselect)
    cols = [
        "h01",
        "h02",
        "h11",
        "h12",
        "N.Co",
        "N.Tr",
        "bias.0",
        "bias.1",
        "var.0",
        "var.1",
        "var.01",
        "reg.bias.0",
        "reg.bias.1",
        "reg.var.0",
        "reg.var.1",
        "reg.var.01",
    ]
    res = np.full((len(eval_pts), len(cols)), np.nan)
    for i in range(len(eval_pts)):
        ev = (float(ev_std[i, 0]), float(ev_std[i, 1]))
        vec = e_deriv[i]
        s0 = sdat.side(0, ev, "rad")
        s1 = sdat.side(1, ev, "rad")
        dn0 = dn1 = dn
        if bounds is not None:
            mn0, mn1, mx0, mx1 = bounds[i]
            dn0 = min(max(dn, mn0), mx0)
            dn1 = min(max(dn, mn1), mx1)
        b0 = _Side(s0.x1, s0.x2, s0.dist, s0.Y[:, :1].copy(), s0.cl)
        b1 = _Side(s1.x1, s1.x2, s1.dist, s1.Y[:, :1].copy(), s1.cl)
        if fuzzy and bwparam == "main":
            g0, _, _ = _lm(s0, dn0, p, vce, kernel, kernel_type, False)
            g1, _, _ = _lm(s1, dn1, p, vce, kernel, kernel_type, False)
            t_itt = float(vec @ g1[:, 0] - vec @ g0[:, 0])
            t_fs = float(vec @ g1[:, 1] - vec @ g0[:, 1])
            if np.isfinite(t_fs) and abs(t_fs) > math.sqrt(np.finfo(float).eps):
                gi, gf = 1.0 / t_fs, -t_itt / t_fs**2
                b0.Y = (gi * s0.Y[:, 0] + gf * s0.Y[:, 1])[:, None]
                b1.Y = (gi * s1.Y[:, 0] + gf * s1.Y[:, 1])[:, None]
            else:
                warnings.warn(
                    "Weak or zero first-stage fuzzy RD estimate detected in bandwidth "
                    "selection; using reduced-form outcome bandwidth.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        eN0 = int((_weights(s0, dn0, kernel, kernel_type) > 0).sum())
        eN1 = int((_weights(s1, dn1, kernel, kernel_type) > 0).sum())
        ji_dn = _joint_info(s0, s1, dn0, dn1, kernel, kernel_type)
        vq0 = _get_coeff(s0, vec, p, dn0, kernel, kernel_type)
        vq1 = _get_coeff(s1, vec, p, dn1, kernel, kernel_type)
        th0 = float(np.median(s0.dist))
        th1 = float(np.median(s1.dist))
        ji_th = _joint_info(s0, s1, th0, th1, kernel, kernel_type)
        bn0, bn1 = th0, th1
        if method == "dpi":
            c0 = _bw_consts(
                b0,
                p + 1,
                vq0,
                dn0,
                th0,
                None,
                vce,
                kernel,
                kernel_type,
                fitmethod,
                clusters,
                ji_dn,
                ji_th,
            )
            c1 = _bw_consts(
                b1,
                p + 1,
                vq1,
                dn1,
                th1,
                None,
                vce,
                kernel,
                kernel_type,
                fitmethod,
                clusters,
                ji_dn,
                ji_th,
            )
            num = 2 + 2 * (p + 1)
            den = 2 * (p + 1) + 2 - 2 * (p + 1)
            bn0 = (
                num * c0["V"] / (den * (c0["B"] ** 2 + scaleregul * c0["Reg1"]))
            ) ** (1 / (2 * p + 6))
            bn1 = (
                num * c1["V"] / (den * (c1["B"] ** 2 + scaleregul * c1["Reg1"]))
            ) ** (1 / (2 * p + 6))
            if bounds is not None:
                bn0 = min(max(bn0, mn0), mx0)
                bn1 = min(max(bn1, mn1), mx1)
        ji_bn = _joint_info(s0, s1, bn0, bn1, kernel, kernel_type)
        h0c = _bw_consts(
            b0,
            p,
            vec,
            dn0,
            bn0,
            th0,
            vce,
            kernel,
            kernel_type,
            fitmethod,
            clusters,
            ji_dn,
            ji_bn,
        )
        h1c = _bw_consts(
            b1,
            p,
            vec,
            dn1,
            bn1,
            th1,
            vce,
            kernel,
            kernel_type,
            fitmethod,
            clusters,
            ji_dn,
            ji_bn,
        )
        var01 = reg01 = 0.0
        if base in ("mserd", "imserd"):
            if fitmethod == "joint" and data.cluster is not None:
                var01 = float(h0c["V_half"] @ h1c["V_half"])
                reg01 = float(h0c["Reg1_half"] @ h1c["Reg1_half"])
            Vd = h0c["V"] + h1c["V"] - 2 * var01
            Rd = h0c["Reg1"] + h1c["Reg1"] - 2 * reg01
            bias = (
                h0c["B"]
                + scalebiascrct * h0c["Reg2"]
                - h1c["B"]
                - scalebiascrct * h1c["Reg2"]
            )
            hn = (
                (2 + 2 * ds) * Vd / ((2 * p + 2 - 2 * ds) * (bias**2 + scaleregul * Rd))
            ) ** (1 / (2 * p + 4))
            if bounds is not None:
                hn = max(hn, mn0, mn1)
                hn = min(hn, max(mx0, mx1))
            hn0 = hn1 = hn
        else:  # msetwo / imsetwo
            hn0 = (
                (2 + 2 * ds)
                * h0c["V"]
                / (
                    deriv_denom
                    * (
                        (h0c["B"] + scalebiascrct * h0c["Reg2"]) ** 2
                        + scaleregul * h0c["Reg1"]
                    )
                )
            ) ** (1 / (2 * p + 4))
            hn1 = (
                (2 + 2 * ds)
                * h1c["V"]
                / (
                    deriv_denom
                    * (
                        (h1c["B"] + scalebiascrct * h1c["Reg2"]) ** 2
                        + scaleregul * h1c["Reg1"]
                    )
                )
            ) ** (1 / (2 * p + 4))
            if bounds is not None:
                hn0 = min(max(hn0, mn0), mx0)
                hn1 = min(max(hn1, mn1), mx1)
        res[i] = (
            hn0,
            hn0,
            hn1,
            hn1,
            eN0,
            eN1,
            h0c["B"],
            h1c["B"],
            h0c["V"],
            h1c["V"],
            var01,
            h0c["Reg2"],
            h1c["Reg2"],
            h0c["Reg1"],
            h1c["Reg1"],
            reg01,
        )
    c = {name: j for j, name in enumerate(cols)}
    if base == "imserd":
        VV = np.mean(res[:, c["var.0"]] + res[:, c["var.1"]] - 2 * res[:, c["var.01"]])
        BB = np.mean(
            (
                res[:, c["bias.0"]]
                + scalebiascrct * res[:, c["reg.bias.0"]]
                - res[:, c["bias.1"]]
                - scalebiascrct * res[:, c["reg.bias.1"]]
            )
            ** 2
            + scaleregul
            * (
                res[:, c["reg.var.0"]]
                + res[:, c["reg.var.1"]]
                - 2 * res[:, c["reg.var.01"]]
            )
        )
        hI = ((2 + 2 * ds) * VV / (deriv_denom * BB)) ** (1 / (2 * p + 4))
        res[:, 0:4] = hI
    if base == "imsetwo":
        V0, V1 = np.mean(res[:, c["var.0"]]), np.mean(res[:, c["var.1"]])
        B0 = np.mean(
            (res[:, c["bias.0"]] + scalebiascrct * res[:, c["reg.bias.0"]]) ** 2
            + scaleregul * res[:, c["reg.var.0"]]
        )
        B1 = np.mean(
            (res[:, c["bias.1"]] + scalebiascrct * res[:, c["reg.bias.1"]]) ** 2
            + scaleregul * res[:, c["reg.var.1"]]
        )
        res[:, 0:2] = ((2 + 2 * ds) * V0 / (deriv_denom * B0)) ** (1 / (2 * p + 4))
        res[:, 2:4] = ((2 + 2 * ds) * V1 / (deriv_denom * B1)) ** (1 / (2 * p + 4))
    if _is_cer(bwselect):
        if _is_common(bwselect):
            res[:, 0:4] *= cer_factor(M, p)
        else:
            res[:, 0:2] *= cer_factor(M0, p)
            res[:, 2:4] *= cer_factor(M1, p)
    res[:, 0] *= sd1
    res[:, 1] *= sd2
    res[:, 2] *= sd1
    res[:, 3] *= sd2
    return {
        "bws": res[:, 0:4].copy(),
        "mseconsts": res,
        "mseconst_names": cols,
        "M": M,
        "M0": M0,
        "M1": M1,
        "bwcheck": bwcheck,
    }


# ----------------------------------------------------------------------
# Full estimator (R rd2d)
# ----------------------------------------------------------------------


def rd2d_location_fit(
    data: Rd2dData,
    eval_pts: np.ndarray,
    h: Any,
    p: int,
    q: int,
    deriv: Sequence[int],
    tangvec: Any,
    kernel: str,
    kernel_type: str,
    vce: str,
    masspoints: str,
    bwcheck: Optional[int],
    fitmethod: str,
    bwselect: str,
    method: str,
    scaleregul: float,
    scalebiascrct: float,
    stdvars: bool,
    bwparam: str = "main",
) -> Dict[str, object]:
    """R ``rd2d`` (no covariates).  ``h`` is None, a scalar, or (neval, 4)."""
    neval = len(eval_pts)
    fuzzy = data.Y.shape[1] == 2
    if tangvec is not None:
        warnings.warn(
            "Tangvec provided. Ignore option deriv.", RuntimeWarning, stacklevel=3
        )
    e_deriv = e_deriv_matrix(neval, p, deriv, tangvec)
    if tangvec is not None:
        deriv = (1, 0)
    N = data.n
    N0, N1 = int(data.s0.sum()), int(data.s1.sum())
    M = M0 = M1 = None
    if masspoints in ("check", "adjust"):
        u = unique_locations(data.x1, data.x2, data.d)
        M0 = int((data.d[u] == 0).sum())
        M1 = int((data.d[u] == 1).sum())
        M = M0 + M1
        if 1 - M / N >= 0.2:
            warnings.warn(
                "Mass points detected in the running variables.",
                RuntimeWarning,
                stacklevel=3,
            )
            if masspoints == "check":
                warnings.warn(
                    "Try using option masspoints=adjust.", RuntimeWarning, stacklevel=3
                )
            if bwcheck is None:
                bwcheck = 50 + p + 1
    if bwcheck is not None:
        a0, a1 = (M0, M1) if masspoints == "adjust" else (N0, N1)
        unit = "unique mass points" if masspoints == "adjust" else "observations"
        if a0 < bwcheck or a1 < bwcheck:
            raise DataInsufficient(
                f"bwcheck={bwcheck} requires at least {bwcheck} {unit} on each side of "
                f"the cutoff; found {a0} on the control side and {a1} on the treatment "
                "side. Decrease bwcheck or set bwcheck=None.",
                recovery_hint="Decrease bwcheck or set bwcheck=None.",
            )
    bw_info = None
    if h is None:
        bw_info = rd2d_location_bw(
            data,
            eval_pts,
            p,
            deriv,
            tangvec,
            kernel,
            kernel_type,
            bwselect,
            method,
            vce,
            bwcheck,
            masspoints,
            fitmethod,
            scaleregul,
            scalebiascrct,
            stdvars,
            bwparam,
        )
        bws = bw_info["bws"]
        hgrid0, hgrid1 = bws[:, 0:2], bws[:, 2:4]
        bwselect_used = bwselect
    else:
        bwselect_used = "user provided"
        harr = np.asarray(h, dtype=float)
        if harr.ndim == 0 or harr.size == 1:
            hgrid0 = np.full((neval, 2), float(harr.reshape(-1)[0]))
            hgrid1 = hgrid0.copy()
        else:
            hgrid0, hgrid1 = harr[:, 0:2], harr[:, 2:4]
    sd1, sd2 = sd(data.x1), sd(data.x2)
    bounds = None
    if bwcheck is not None:
        bounds = bwcheck_bounds(
            data.x1,
            data.x2,
            data.d,
            eval_pts,
            bwcheck,
            masspoints,
            kernel_type,
            (sd1, sd2),
        )
    kq = n_basis(q)
    e_deriv_q = np.zeros((neval, kq))
    e_deriv_q[:, : e_deriv.shape[1]] = e_deriv
    fp = fit(
        data, eval_pts, e_deriv, p, hgrid0, hgrid1, kernel, kernel_type, vce, bounds
    )
    fq = (
        fp
        if q == p
        else fit(
            data,
            eval_pts,
            e_deriv_q,
            q,
            hgrid0,
            hgrid1,
            kernel,
            kernel_type,
            vce,
            bounds,
        )
    )

    def _hp(f: Any) -> Any:
        return np.column_stack([f["h0x"], f["h0y"]]), np.column_stack(
            [f["h1x"], f["h1y"]]
        )

    out: Dict[str, object] = {
        "hgrid0": hgrid0,
        "hgrid1": hgrid1,
        "fit_p": fp,
        "fit_q": fq,
        "N": N,
        "N0": N0,
        "N1": N1,
        "M": M,
        "M0": M0,
        "M1": M1,
        "bwselect": bwselect_used,
        "bw_info": bw_info,
        "bwcheck": bwcheck,
        "e_deriv": e_deriv,
        "e_deriv_q": e_deriv_q,
        "deriv": tuple(deriv),
    }
    joint = fitmethod == "joint"
    clustered_joint = joint and data.cluster is not None

    def _cov_tables(f: Any, ed: Any, order: Any) -> Any:
        g0, g1 = _hp(f)
        P0, P1, _ = project_sides(
            data, eval_pts, ed, order, g0, g1, kernel, kernel_type, vce, fitmethod
        )
        return P0, P1

    # Projections at the bandwidths each fit used (joint fits and the
    # cross-point covariance both need them).
    Pp = _cov_tables(fp, e_deriv, p)
    Pq = Pp if q == p else _cov_tables(fq, e_deriv_q, q)
    denom_tol = math.sqrt(np.finfo(float).eps)
    for tag, f, P in (("p", fp, Pp), ("q", fq, Pq)):
        P0, P1 = P
        tau = f["mu1"] - f["mu0"]  # (neval, m)
        if not fuzzy:
            cov = _sym(cov_from_projects(P0[0], P1[0], P0[0], P1[0], clustered_joint))
            if joint:
                se = np.sqrt(np.maximum(np.diag(cov), 0))
                se0 = np.sqrt(np.maximum(np.diag(_sym(P0[0].T @ P0[0])), 0))
                se1 = np.sqrt(np.maximum(np.diag(_sym(P1[0].T @ P1[0])), 0))
            else:
                se = np.sqrt(f["se0"][:, 0] ** 2 + f["se1"][:, 0] ** 2)
                se0, se1 = f["se0"][:, 0], f["se1"][:, 0]
            out[f"tau_{tag}"] = tau[:, 0]
            out[f"se_{tag}"] = se
            out[f"cov_{tag}"] = cov
            out[f"mu0_{tag}"], out[f"mu1_{tag}"] = f["mu0"][:, 0], f["mu1"][:, 0]
            out[f"se0_{tag}"], out[f"se1_{tag}"] = se0, se1
        else:
            t_itt, t_fs = tau[:, 0], tau[:, 1]
            valid = np.isfinite(t_fs) & (np.abs(t_fs) > denom_tol)
            est = np.full(neval, np.nan)
            est[valid] = t_itt[valid] / t_fs[valid]
            yy = cov_from_projects(P0[0], P1[0], P0[0], P1[0], clustered_joint)
            dd = cov_from_projects(P0[1], P1[1], P0[1], P1[1], clustered_joint)
            yd = cov_from_projects(P0[0], P1[0], P0[1], P1[1], clustered_joint)
            dy = cov_from_projects(P0[1], P1[1], P0[0], P1[0], clustered_joint)
            a = 1.0 / t_fs
            b = -t_itt / t_fs**2
            cov = (
                np.outer(a, a) * yy
                + np.outer(a, b) * yd
                + np.outer(b, a) * dy
                + np.outer(b, b) * dd
            )
            cov[~np.outer(valid, valid)] = np.nan
            cov = _sym(cov)
            if joint:
                se = np.sqrt(np.maximum(np.diag(cov), 0))
                se_itt = np.sqrt(np.maximum(np.diag(_sym(yy)), 0))
                se_fs = np.sqrt(np.maximum(np.diag(_sym(dd)), 0))
            else:
                se_itt = np.sqrt(f["se0"][:, 0] ** 2 + f["se1"][:, 0] ** 2)
                se_fs = np.sqrt(f["se0"][:, 1] ** 2 + f["se1"][:, 1] ** 2)
                # R rdbw2d_se_fuzzy, separate fit: delta method with the
                # within-side ITT/FS cross terms.  R evaluates the cross
                # term at the selected (not bwcheck-clamped) bandwidths;
                # kept so the two agree when the clamp binds.
                ed = e_deriv if tag == "p" else e_deriv_q
                order = p if tag == "p" else q
                U0, U1, _ = project_sides(
                    data,
                    eval_pts,
                    ed,
                    order,
                    hgrid0,
                    hgrid1,
                    kernel,
                    kernel_type,
                    vce,
                    fitmethod,
                )
                cross = np.zeros(neval)
                for side_P in (U0, U1):
                    cross += np.einsum("ij,ij->j", side_P[0], side_P[1])
                var = (
                    se_itt**2 / t_fs**2
                    + t_itt**2 * se_fs**2 / t_fs**4
                    - 2 * t_itt * cross / t_fs**3
                )
                se = np.full(neval, np.nan)
                ok = valid & np.isfinite(se_itt) & np.isfinite(se_fs)
                se[ok] = np.sqrt(np.maximum(var[ok], 0))
            out[f"tau_{tag}"] = est
            out[f"se_{tag}"] = se
            out[f"cov_{tag}"] = cov
            out[f"itt_{tag}"], out[f"fs_{tag}"] = t_itt, t_fs
            out[f"se_itt_{tag}"], out[f"se_fs_{tag}"] = se_itt, se_fs
    if fuzzy and not (
        np.all(np.isfinite(out["tau_p"])) and np.all(np.isfinite(out["tau_q"]))
    ):
        warnings.warn(
            "Weak or zero first-stage fuzzy RD estimates detected; returning NaN for "
            "affected fuzzy estimates.",
            RuntimeWarning,
            stacklevel=3,
        )
    return out
