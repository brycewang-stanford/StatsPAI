"""Location-based boundary RD: bandwidth selection and estimation.

Clean-room implementation (2026-10-21) of the location-based estimator of
Cattaneo, Titiunik and Yu (arXiv 2505.05670, sections 3.1-3.3; software
article arXiv 2505.07989, sections 3.1-3.7). R ``rd2d`` 1.0.0 (GPL-3) was
used only as a black box; ``docs/dev/cleanroom/rd2d.md`` records every
convention fixed that way.

Estimation at a boundary point ``b`` (for given bandwidths): on each side
``t`` a bivariate local polynomial of order ``p`` (and ``q``) in
``x - b`` with kernel weights; the effect is the jump of the selected
coefficient combination (level, partial derivative or directional
derivative).

Bandwidth selection (MSE / CER, pointwise or integrated, common or per
side) is a three-step plug-in:

1. a pilot bandwidth ``a`` from a normal-reference rule on the score
   distribution (no outcome involved);
2. the variance constant ``a^2 Var(theta_t(a))`` and the bias factor of
   the order-``p`` fit at ``a``;
3. the order-``p+1`` coefficients that drive the bias, from a local fit
   at a bandwidth chosen either by a second plug-in step ("dpi") or as the
   median distance of the side's observations to ``b`` ("rot"); the
   order-``p+2`` terms of a fit at the median distance give a higher-order
   correction and the regularisation term.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import DataInsufficient
from ._rd2d_common import (
    RADIAL_MOMENTS,
    WindowFit,
    cer_rate,
    fit_window,
    kernel_1d,
    monomials,
    point_covariance,
    poly_basis,
    side_variances,
)

# A first-stage jump below this (treatment receipt is on a 0-1 scale) is
# treated as zero: the fuzzy ratio is undefined.
FIRST_STAGE_TOL = 1e-8

__all__ = [
    "BoundaryData",
    "selection_vector",
    "location_bandwidths",
    "location_estimates",
]


class BoundaryData:
    """Complete-case boundary-RD sample: scores, side, outcomes, clusters.

    ``Y`` has one column (sharp) or two (outcome and treatment receipt,
    fuzzy).
    """

    def __init__(
        self, x1: Any, x2: Any, side: Any, Y: Any, cluster: Any = None
    ) -> None:
        self.x1 = np.asarray(x1, dtype=float)
        self.x2 = np.asarray(x2, dtype=float)
        self.side = np.asarray(side).astype(int)
        Y = np.asarray(Y, dtype=float)
        self.Y = Y[:, None] if Y.ndim == 1 else Y
        self.cluster = None if cluster is None else np.asarray(cluster)
        self.pos = [np.flatnonzero(self.side == t) for t in (0, 1)]
        if self.cluster is not None:
            codes, inv = np.unique(self.cluster, return_inverse=True)
            self.codes = inv
            self.n_codes = len(codes)

    @property
    def n(self) -> int:
        return int(self.x1.size)

    @property
    def fuzzy(self) -> bool:
        return bool(self.Y.shape[1] == 2)

    def n_unique(self) -> int:
        return int(np.unique(np.column_stack([self.x1, self.x2]), axis=0).shape[0])

    def side_cluster(self, t: int) -> Optional[np.ndarray]:
        return None if self.cluster is None else self.cluster[self.pos[t]]


def selection_vector(
    order: int, deriv: Sequence[int], tangvec: Any = None
) -> np.ndarray:
    """Coefficient combination of the target on the order-``order`` basis.

    Level: the intercept. Partial derivative ``d^(n1+n2)``: ``n1! n2!`` times
    the coefficient of ``x1^n1 x2^n2``. Direction ``v``: ``v1 * d/dx1 +
    v2 * d/dx2``.
    """
    mono = monomials(order)
    s = np.zeros(len(mono))
    if tangvec is not None:
        s[mono.index((1, 0))] = float(tangvec[0])
        s[mono.index((0, 1))] = float(tangvec[1])
        return s
    i, j = int(deriv[0]), int(deriv[1])
    s[mono.index((i, j))] = math.factorial(i) * math.factorial(j)
    return s


def _target_order(deriv: Sequence[int], tangvec: Any) -> int:
    return 1 if tangvec is not None else int(deriv[0]) + int(deriv[1])


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------


def _snap(u: np.ndarray) -> np.ndarray:
    # Observations on the window edge up to rounding are on the edge: the
    # enlarged (bwcheck) window is built from an observed distance, so its
    # edge passes through observations exactly (black box: lattice data).
    a = np.abs(u)
    return np.where(np.abs(a - 1.0) < 1e-12, np.sign(u), u)


def _weights(
    d1: np.ndarray, d2: np.ndarray, h1: float, h2: float, kernel: str, kernel_type: str
) -> np.ndarray:
    if kernel_type == "prod":
        wp: np.ndarray = (
            kernel_1d(_snap(d1 / h1), kernel)
            * kernel_1d(_snap(d2 / h2), kernel)
            / (h1 * h2)
        )
        return wp
    # Radial kernel at radius sqrt(h1^2 + h2^2) (the reference's fit; see
    # the clean-room record for its variance scaling, not reproduced).
    r = math.hypot(h1, h2)
    return kernel_1d(_snap(np.hypot(d1, d2) / r), kernel) / (r * r)


def _scalar_weights(
    d1: np.ndarray, d2: np.ndarray, h: float, kernel: str, kernel_type: str
) -> np.ndarray:
    if kernel_type == "prod":
        wp: np.ndarray = kernel_1d(d1 / h, kernel) * kernel_1d(d2 / h, kernel)
        return wp
    return kernel_1d(np.hypot(d1, d2) / h, kernel)


# --------------------------------------------------------------------------
# pilot bandwidth
# --------------------------------------------------------------------------


def _normal_roughness(S: np.ndarray) -> float:
    """``int (tr Hess f)^2`` for the bivariate normal density with covariance S."""
    Si = np.linalg.inv(S)
    num = np.trace(Si @ Si) / 2.0 + np.trace(Si) ** 2 / 4.0
    return float(num / (4.0 * math.pi * math.sqrt(np.linalg.det(S))))


def pilot_bandwidth(X: np.ndarray, kernel: str, m: int) -> float:
    """Normal-reference pilot ``(2 R / (mu2 m I(S)))^(1/6)``.

    ``S`` the sample covariance of the scores, ``I(S)`` the integrated
    squared Laplacian of the normal density with covariance ``S``,
    ``(mu2, R)`` the radial kernel moments, ``m`` the number of distinct
    score locations (or observations with ``masspoints='off'``).
    """
    mu2, R = RADIAL_MOMENTS[kernel]
    S = np.cov(X.T)
    return float((2.0 * R / (mu2 * m * _normal_roughness(S))) ** (1.0 / 6.0))


# --------------------------------------------------------------------------
# bandwidth selection
# --------------------------------------------------------------------------


def _bias_factors(
    d1: np.ndarray,
    d2: np.ndarray,
    h: float,
    kernel: str,
    kernel_type: str,
    order: int,
    degree: int,
    sel: np.ndarray,
) -> np.ndarray:
    """``sel' Gamma^-1 E[r(U) U^k K]`` for every monomial k of ``degree``.

    Computed with the scaled design ``U = (x - b) / h``; the raw-scale bias
    of the target is then ``h^(degree - |target|) * sum_k factor_k * c_k``
    for the raw polynomial coefficients ``c_k``.
    """
    w = _scalar_weights(d1, d2, h, kernel, kernel_type)
    m = w > 0
    U1, U2, ww = d1[m] / h, d2[m] / h, w[m]
    X = poly_basis(U1, U2, order)
    G = (X.T * ww) @ X
    cols = [(i, j) for i, j in monomials(degree) if i + j == degree]
    M = np.column_stack([(X.T * ww) @ (U1**i * U2**j) for i, j in cols])
    return sel @ np.linalg.solve(G, M)


def _degree_index(order: int, degree: int) -> List[int]:
    mono = monomials(order)
    return [mono.index((i, j)) for i, j in mono if i + j == degree]


def _side_fit(
    data_side: Tuple[np.ndarray, np.ndarray],
    Ycol: np.ndarray,
    h: float,
    kernel: str,
    kernel_type: str,
    order: int,
    cluster: Optional[np.ndarray],
    what: str,
) -> WindowFit:
    d1, d2 = data_side
    w = _scalar_weights(d1, d2, h, kernel, kernel_type)
    return fit_window(poly_basis(d1, d2, order), Ycol, w, cluster, what)


def _point_constants(
    sides: Sequence[Tuple[np.ndarray, np.ndarray]],
    ycols: Any,
    clusters: Any,
    A: Any,
    kernel: str,
    kernel_type: str,
    p: int,
    vce: str,
    method: str,
    scaleregul: float,
    joint: bool,
    sel: np.ndarray,
) -> Dict[str, Any]:
    """Variance, bias, higher-order bias and regularisation constants at one
    evaluation point, per side (and the cross-side covariances)."""
    q = p + 1
    k_q = len(monomials(q))
    k_r = len(monomials(p + 2))
    idx_q = _degree_index(q, q)
    idx_r = _degree_index(p + 2, p + 2)

    def pad(v: np.ndarray, idx: List[int], K: int) -> np.ndarray:
        z = np.zeros(K)
        z[idx] = v
        return z

    dmed = [float(np.median(np.hypot(*sides[t]))) for t in (0, 1)]
    fp = [
        _side_fit(
            sides[t], ycols[t], A[t], kernel, kernel_type, p, clusters[t], "pilot"
        )
        for t in (0, 1)
    ]
    vsel = [sel * np.array([A[t] ** (i + j) for i, j in monomials(p)]) for t in (0, 1)]
    v0, v1, v01 = side_variances(
        fp, [fp[t].influence(vsel[t], vce) for t in (0, 1)], vce, joint
    )
    fr = [
        _side_fit(
            sides[t],
            ycols[t],
            dmed[t],
            kernel,
            kernel_type,
            p + 2,
            clusters[t],
            "higher-order bias",
        )
        for t in (0, 1)
    ]
    F = [_bias_factors(*sides[t], A[t], kernel, kernel_type, p, q, sel) for t in (0, 1)]
    F2 = [
        _bias_factors(*sides[t], A[t], kernel, kernel_type, p, p + 2, sel)
        for t in (0, 1)
    ]
    if method == "rot":
        g = dmed
    else:
        fa = [
            _side_fit(
                sides[t],
                ycols[t],
                A[t],
                kernel,
                kernel_type,
                q,
                clusters[t],
                "bias pilot",
            )
            for t in (0, 1)
        ]
        vth = side_variances(
            fa,
            [fa[t].influence(pad(F[t], idx_q, k_q), vce) for t in (0, 1)],
            vce,
            joint,
        )
        H = [
            F[t]
            @ _bias_factors(
                *sides[t], A[t], kernel, kernel_type, q, p + 2, np.eye(k_q)
            )[idx_q, :]
            for t in (0, 1)
        ]
        rt = side_variances(
            fr,
            [fr[t].influence(pad(H[t], idx_r, k_r), vce) for t in (0, 1)],
            vce,
            joint,
        )
        g = []
        for t in (0, 1):
            bt = float(H[t] @ fr[t].beta[idx_r])
            denom = bt * bt + scaleregul * rt[t]
            g.append(
                ((p + 2) * vth[t] * A[t] ** (2 * p + 4) / denom) ** (1.0 / (2 * p + 6))
            )
    fq = [
        _side_fit(sides[t], ycols[t], g[t], kernel, kernel_type, q, clusters[t], "bias")
        for t in (0, 1)
    ]
    r0, r1, r01 = side_variances(
        fq, [fq[t].influence(pad(F[t], idx_q, k_q), vce) for t in (0, 1)], vce, joint
    )
    return {
        "var": (A[0] ** 2 * v0, A[1] ** 2 * v1),
        "var01": A[0] * A[1] * v01,
        "bias": tuple(float(F[t] @ fq[t].beta[idx_q]) for t in (0, 1)),
        "bias_hi": tuple(float(A[t] * F2[t] @ fr[t].beta[idx_r]) for t in (0, 1)),
        "reg": (r0, r1),
        "reg01": r01,
        "n_pilot": (fp[0].n, fp[1].n),
    }


def _kth_distance(d: np.ndarray, k: int) -> float:
    d = np.sort(d)
    return float(d[min(k, d.size) - 1])


def location_bandwidths(
    data: BoundaryData,
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
) -> Dict[str, Any]:
    """MSE- or CER-optimal bandwidths at each evaluation point.

    Returns ``h`` (J x 4: control x1, control x2, treated x1, treated x2) and
    the per-point constants.
    """
    X = np.column_stack([data.x1, data.x2])
    scale = np.std(X, axis=0, ddof=1) if stdvars else np.ones(2)
    Xs = X / scale
    bs = np.asarray(eval_pts, dtype=float) / scale
    m = data.n if masspoints == "off" else data.n_unique()
    a0 = pilot_bandwidth(Xs, kernel, m)
    nu = _target_order(deriv, tangvec)
    joint = fitmethod == "joint"
    base = bwselect.replace("cer", "mse").replace("imse", "ise")
    two = base.endswith("two")
    integrated = bwselect.startswith("i")
    J = bs.shape[0]
    unique_floor = masspoints == "adjust"
    consts = []
    floors = []
    for j in range(J):
        sides: List[Tuple[np.ndarray, np.ndarray]] = []
        for t in (0, 1):
            pos = data.pos[t]
            sides.append((Xs[pos, 0] - bs[j, 0], Xs[pos, 1] - bs[j, 1]))
        clusters = [data.side_cluster(t) for t in (0, 1)]
        tv = None
        if tangvec is not None:
            tv = np.asarray(tangvec[j], dtype=float)
        sel = selection_vector(p, deriv, tv)
        A = [a0, a0]
        fl = [0.0, 0.0]
        if bwcheck is not None:
            for t in (0, 1):
                d1, d2 = sides[t]
                if unique_floor:
                    u = np.unique(np.column_stack([d1, d2]), axis=0)
                    dist = np.hypot(u[:, 0], u[:, 1])
                else:
                    dist = np.hypot(d1, d2)
                fl[t] = _kth_distance(dist, bwcheck)
                A[t] = max(a0, fl[t])
        floors.append(fl)
        ycols = _selector_outcomes(data, sides, A, kernel, kernel_type, p, bwparam)
        consts.append(
            _point_constants(
                sides,
                ycols,
                clusters,
                A,
                kernel,
                kernel_type,
                p,
                vce,
                method,
                scaleregul,
                joint,
                sel,
            )
        )

    def mse(var: Any, bias2: Any) -> Any:
        return ((2.0 + 2.0 * nu) * var / (2.0 * (p + 1 - nu) * bias2)) ** (
            1.0 / (2 * p + 4)
        )

    def pieces(c: Dict[str, Any], t: Optional[int] = None) -> Tuple[Any, Any]:
        bt = [c["bias"][s] + scalebiascrct * c["bias_hi"][s] for s in (0, 1)]
        if t is None:
            V = c["var"][0] + c["var"][1] - 2.0 * c["var01"]
            B = bt[1] - bt[0]
            R = c["reg"][0] + c["reg"][1] - 2.0 * c["reg01"]
        else:
            V, B, R = c["var"][t], bt[t], c["reg"][t]
        return V, B * B + scaleregul * R

    H = np.empty((J, 2))
    if integrated:
        side_set: Tuple[Optional[int], ...] = (0, 1) if two else (None,)
        for side in side_set:
            Vs = sum(pieces(c, side)[0] for c in consts)
            Bs = sum(pieces(c, side)[1] for c in consts)
            h = mse(Vs, Bs)
            if side is None:
                H[:, :] = h
            else:
                H[:, side] = h
    else:
        for j, c in enumerate(consts):
            if two:
                H[j] = [mse(*pieces(c, 0)), mse(*pieces(c, 1))]
            else:
                H[j] = mse(*pieces(c, None))
    # The bwcheck floor applies to the MSE bandwidth, before the CER rate;
    # integrated selectors are not floored (black box)
    # (black box: the Gaussian CER cell of the fixture).
    if bwcheck is not None and not integrated:
        for j in range(J):
            if two:
                H[j] = np.maximum(H[j], floors[j])
            else:
                H[j] = max(H[j, 0], max(floors[j]))
    if bwselect.startswith(("cer", "icer")):
        if two:
            H[:, 0] *= cer_rate(data.pos[0].size, p)
            H[:, 1] *= cer_rate(data.pos[1].size, p)
        else:
            H *= cer_rate(data.n, p)
    out = np.column_stack(
        [H[:, 0] * scale[0], H[:, 0] * scale[1], H[:, 1] * scale[0], H[:, 1] * scale[1]]
    )
    return {"h": out, "constants": consts, "scale": scale, "pilot": a0}


def _selector_outcomes(
    data: BoundaryData,
    sides: Any,
    A: Any,
    kernel: str,
    kernel_type: str,
    p: int,
    bwparam: str,
) -> List[np.ndarray]:
    """Outcome used by the selector on each side.

    Sharp: ``Y``. Fuzzy with ``bwparam='main'``: the linearised Wald ratio
    ``Y / tau_T - tau_Y / tau_T^2 * T`` with both jumps from the order-``p``
    fits at the pilot; ``'itt'``: ``Y``.
    """
    Ys = [data.Y[data.pos[t]] for t in (0, 1)]
    if not data.fuzzy or bwparam == "itt":
        return [Ys[t][:, 0] for t in (0, 1)]
    jumps = []
    for col in (0, 1):
        lv = []
        for t in (0, 1):
            f = _side_fit(
                sides[t], Ys[t][:, col], A[t], kernel, kernel_type, p, None, "pilot"
            )
            lv.append(f.beta[0])
        jumps.append(lv[1] - lv[0])
    ty, tt = jumps
    if not np.isfinite(tt) or abs(tt) < FIRST_STAGE_TOL:
        raise DataInsufficient(
            "rd2d: zero first-stage jump at the pilot bandwidth; the fuzzy "
            "bandwidth is undefined.",
            recovery_hint="Use bwparam='itt' or supply h=.",
        )
    return [Ys[t][:, 0] / tt - ty / tt**2 * Ys[t][:, 1] for t in (0, 1)]


# --------------------------------------------------------------------------
# estimation for given bandwidths
# --------------------------------------------------------------------------


def _effective_bandwidth(
    d1: np.ndarray,
    d2: np.ndarray,
    h1: float,
    h2: float,
    kernel_type: str,
    bwcheck: Optional[int],
    unique: bool,
    scale: Tuple[float, float],
) -> Tuple[float, float]:
    """Enlarge ``(h1, h2)`` so that at least ``bwcheck`` locations are used.

    Product kernels: ``lam`` is the ``bwcheck``-th smallest
    ``max(|d1|/s1, |d2|/s2)`` in the selector's coordinates ``x / scale``
    and each bandwidth becomes ``max(h_j, lam * s_j)``. Radial kernels:
    ``lam`` is the ``bwcheck``-th smallest raw Euclidean distance and each
    bandwidth becomes ``max(h_j, lam)``. The reported bandwidth stays the
    requested one (all black box).
    """
    if bwcheck is None:
        return h1, h2
    if unique:
        u = np.unique(np.column_stack([d1, d2]), axis=0)
        d1, d2 = u[:, 0], u[:, 1]
    if kernel_type == "prod":
        s1, s2 = scale
        lam = _kth_distance(np.maximum(np.abs(d1) / s1, np.abs(d2) / s2), bwcheck)
        return max(h1, lam * s1), max(h2, lam * s2)
    lam = _kth_distance(np.hypot(d1, d2), bwcheck)
    return max(h1, lam), max(h2, lam)


def location_estimates(
    data: BoundaryData,
    eval_pts: np.ndarray,
    H: np.ndarray,
    p: int,
    q: int,
    deriv: Sequence[int],
    tangvec: Any,
    kernel: str,
    kernel_type: str,
    vce: str,
    fitmethod: str,
    bwcheck: Optional[int],
    masspoints: str,
    scale: Tuple[float, float] = (1.0, 1.0),
) -> Dict[str, Any]:
    """Side estimates and influences at every point (orders p and q)."""
    J = eval_pts.shape[0]
    out: Dict[str, Any] = {}
    used = np.empty((J, 4))
    for order, tag in ((p, "p"), (q, "q")):
        cols = data.Y.shape[1]
        est = np.empty((J, cols, 2))
        pts: List[List[Dict[str, Any]]] = [[] for _ in range(cols)]
        n_side = np.zeros((J, 2), dtype=int)
        for j in range(J):
            sel = selection_vector(
                order, deriv, None if tangvec is None else tangvec[j]
            )
            fits_by_col: List[List[Any]] = [[None, None] for _ in range(cols)]
            infl_by_col: List[List[Any]] = [[None, None] for _ in range(cols)]
            for t in (0, 1):
                pos = data.pos[t]
                d1 = data.x1[pos] - eval_pts[j, 0]
                d2 = data.x2[pos] - eval_pts[j, 1]
                h1, h2 = H[j, 2 * t], H[j, 2 * t + 1]
                h1, h2 = _effective_bandwidth(
                    d1, d2, h1, h2, kernel_type, bwcheck, masspoints == "adjust", scale
                )
                used[j, 2 * t : 2 * t + 2] = (h1, h2)
                w = _weights(d1, d2, h1, h2, kernel, kernel_type)
                Xd = poly_basis(d1, d2, order)
                for c in range(cols):
                    f = fit_window(
                        Xd, data.Y[pos, c], w, data.side_cluster(t), "boundary"
                    )
                    fits_by_col[c][t] = f
                    infl_by_col[c][t] = f.influence(sel, vce)
                    est[j, c, t] = float(sel @ f.beta)
                n_side[j, t] = fits_by_col[0][t].n
            for c in range(cols):
                pts[c].append({"fits": fits_by_col[c], "infl": infl_by_col[c]})
        out[f"est_{tag}"] = est
        out[f"points_{tag}"] = pts
        out[f"n_{tag}"] = n_side
    out["h_used"] = used
    return out


def assemble(
    data: BoundaryData,
    raw: Dict[str, Any],
    vce: str,
    fitmethod: str,
) -> Dict[str, Any]:
    """Jumps, side and jump standard errors, covariances (sharp or fuzzy)."""
    joint = fitmethod == "joint"
    res: Dict[str, Any] = {}
    for tag in ("p", "q"):
        est = raw[f"est_{tag}"]
        pts = raw[f"points_{tag}"]
        J = est.shape[0]
        fuzzy = est.shape[1] == 2
        cols = [0, 1] if fuzzy else [0]
        side_se = {}
        jump = {}
        jump_se = {}
        for c in cols:
            jump[c] = est[:, c, 1] - est[:, c, 0]
            se = np.empty((J, 2))
            jse = np.empty(J)
            for j in range(J):
                v0, v1, v01 = side_variances(
                    pts[c][j]["fits"], pts[c][j]["infl"], vce, joint
                )
                se[j] = (math.sqrt(v0), math.sqrt(v1))
                jse[j] = math.sqrt(max(v0 + v1 - 2.0 * v01, 0.0))
            side_se[c] = se
            jump_se[c] = jse
        main_pts: List[Optional[Dict[str, Any]]] = []
        if fuzzy:
            ty, tt = jump[0], jump[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                tau = np.where(np.abs(tt) < FIRST_STAGE_TOL, np.nan, ty / tt)
            main_se = np.empty(J)
            for j in range(J):
                fy, fw = pts[0][j], pts[1][j]
                if not np.isfinite(tau[j]):
                    main_se[j] = np.nan
                    main_pts.append(None)
                    continue
                infl = [
                    fy["infl"][t] / tt[j] - ty[j] / tt[j] ** 2 * fw["infl"][t]
                    for t in (0, 1)
                ]
                v0, v1, v01 = side_variances(fy["fits"], infl, vce, joint)
                main_se[j] = math.sqrt(max(v0 + v1 - 2.0 * v01, 0.0))
                main_pts.append({"fits": fy["fits"], "infl": infl})
            res[f"tau_{tag}"] = tau
            res[f"se_{tag}"] = main_se
            res[f"itt_{tag}"], res[f"fs_{tag}"] = ty, tt
            res[f"se_itt_{tag}"], res[f"se_fs_{tag}"] = jump_se[0], jump_se[1]
        else:
            res[f"tau_{tag}"] = jump[0]
            res[f"se_{tag}"] = jump_se[0]
            main_pts = list(pts[0])
            res[f"mu0_{tag}"], res[f"mu1_{tag}"] = est[:, 0, 0], est[:, 0, 1]
            res[f"se0_{tag}"], res[f"se1_{tag}"] = side_se[0][:, 0], side_se[0][:, 1]
        res[f"cov_{tag}"] = _covariance(data, main_pts, vce, joint)
    res["n_co"] = raw["n_p"][:, 0]
    res["n_tr"] = raw["n_p"][:, 1]
    res["h_used"] = raw["h_used"]
    return res


def _covariance(
    data: BoundaryData, pts: List[Optional[Dict[str, Any]]], vce: str, joint: bool
) -> np.ndarray:
    J = len(pts)
    present = [pt for pt in pts if pt is not None]
    if len(present) < J:
        return np.full((J, J), np.nan)
    payload = []
    for pt in present:
        d = {
            "fits": pt["fits"],
            "infl": pt["infl"],
            "pos0": data.pos[0],
            "pos1": data.pos[1],
            "n_total": data.n,
            "cluster": data.cluster,
        }
        if data.cluster is not None:
            d["codes"] = data.codes
            d["n_codes"] = data.n_codes
        payload.append(d)
    return point_covariance(payload, vce, joint)
