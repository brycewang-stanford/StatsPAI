"""Distance-based boundary RD: bandwidth selection and estimation.

Clean-room implementation (2026-10-21) of the distance-based estimator of
Cattaneo, Titiunik and Yu (arXiv 2510.26051, sections 3-5.3; software
article arXiv 2505.07989, section 4). R ``rd2d`` 1.0.0 (GPL-3) was used only
as a black box; ``docs/dev/cleanroom/rd2d.md`` records the conventions fixed
that way.

At each boundary point the score is the signed distance ``D`` (negative on
the control side). On each side a univariate local polynomial in ``|D|``
is fitted; the effect is the jump of the intercepts.

Bandwidth selection, per point: a normal-reference pilot ``c`` from the
second moment of ``D``; the variance constant from the order-``p`` fit at
``c`` with residuals of a global polynomial of order ``p+1`` fitted on the
observations whose ``|D|`` is below the ``cqt`` quantile; the bias from that
polynomial's leading coefficient. Unknown kinks shrink the bandwidth to the
``n^(-1/4)`` rate (and the inference bandwidth to ``n^(-1/3)``); a known kink
bounds the bandwidth near it by the distance to the kink.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import DataInsufficient
from ._rd2d_common import (
    RADIAL_MOMENTS,
    canonical_kernel,
    cer_rate,
    fit_window,
    kernel_1d,
    point_covariance,
    poly_basis,
    side_variances,
)

__all__ = ["distance_bandwidths", "distance_estimates", "distance_assemble"]

# A first-stage jump below this (treatment receipt is on a 0-1 scale) is
# treated as zero: the fuzzy ratio is undefined.
FIRST_STAGE_TOL = 1e-8


def _pilot(Dj: np.ndarray, kernel: str, n: int) -> float:
    """Normal-reference pilot for a radially symmetric score with
    ``E|X - b|^2 = mean(D^2)``: ``sigma (4 pi R / (mu2 n))^(1/6)``,
    ``sigma^2 = mean(D^2) / 2``."""
    mu2, R = RADIAL_MOMENTS[kernel]
    sigma = math.sqrt(float(np.mean(Dj * Dj)) / 2.0)
    return float(sigma * (4.0 * math.pi * R / (mu2 * n)) ** (1.0 / 6.0))


def _bwcheck_limits(dist: np.ndarray, bwcheck: int) -> Tuple[float, float]:
    """``(k-th smallest, largest)`` of ``dist`` (NaN, NaN when empty)."""
    d = np.sort(np.asarray(dist, dtype=float))
    if d.size == 0:
        return float("nan"), float("nan")
    return float(d[min(int(bwcheck), d.size) - 1]), float(d[-1])


def _quantile_poly_fit(
    y: np.ndarray, x: np.ndarray, degree: int
) -> Tuple[np.ndarray, np.ndarray]:
    """OLS of ``y`` on ``1, x, ..., x^degree`` with its homoskedastic
    covariance; refuses a sample too small for the polynomial."""
    n = y.size
    if n <= degree + 1:
        raise DataInsufficient(
            f"rd2d: {n} observations below the cqt quantile are too few for "
            f"the order-{degree} bias polynomial.",
            recovery_hint="Increase cqt= or use more data near the boundary.",
            diagnostics={"n": int(n), "degree": int(degree)},
        )
    X = poly_basis(x, None, degree)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    sigma2 = float(resid @ resid) / (n - degree - 1)
    return beta, XtX_inv * sigma2


def _side_jumps(
    y: np.ndarray,
    f: Optional[np.ndarray],
    dist: np.ndarray,
    h: Any,
    p: int,
    kernel: str,
) -> Tuple[float, float]:
    """Jumps of the order-``p`` local intercepts of ``y`` and of ``f`` at
    bandwidth ``h`` (a scalar or one per side; ``f`` may be None); NaN when a
    side lacks data."""
    kernel = canonical_kernel(kernel)
    hs = (h, h) if np.ndim(h) == 0 else tuple(h)
    out = []
    for col in (y, f):
        if col is None:
            out.append(float("nan"))
            continue
        lv = []
        for side_mask, h in zip((dist < 0, dist >= 0), hs):
            u = np.abs(dist[side_mask])
            w = kernel_1d(u / h, kernel)
            m = w > 0
            if m.sum() <= p:
                lv.append(float("nan"))
                continue
            X = poly_basis(u[m], None, p)
            lv.append(
                float(
                    np.linalg.solve((X.T * w[m]) @ X, (X.T * w[m]) @ col[side_mask][m])[
                        0
                    ]
                )
            )
        out.append(lv[1] - lv[0])
    return out[0], out[1]


def _kink_distance(eval_pts: np.ndarray, kink: np.ndarray) -> np.ndarray:
    """Distance from each evaluation point to the nearest kink point."""
    eval_pts = np.asarray(eval_pts, dtype=float)
    kink = np.asarray(kink, dtype=bool)
    if not kink.any():
        return np.full(eval_pts.shape[0], np.inf)
    K = eval_pts[kink]
    dmin: np.ndarray = np.min(
        np.hypot(
            eval_pts[:, None, 0] - K[None, :, 0], eval_pts[:, None, 1] - K[None, :, 1]
        ),
        axis=1,
    )
    return dmin


def _point_constants(
    Dj: np.ndarray,
    y: np.ndarray,
    kernel: str,
    p: int,
    cqt: float,
    vce: str,
    joint: bool,
    n_pilot: int,
    cluster: Optional[np.ndarray],
    floors: Any,
) -> Dict[str, Any]:
    c = _pilot(Dj, kernel, n_pilot)
    A = [max(c, floors[0]), max(c, floors[1])]
    fits, infl, bias, reg = [], [], [], []
    for t, mask in enumerate((Dj < 0, Dj >= 0)):
        u = np.abs(Dj[mask])
        yy = y[mask]
        q = np.quantile(u, cqt)
        sub = u <= q
        beta, V = _quantile_poly_fit(yy[sub], u[sub], p + 1)
        resid = yy - poly_basis(u, None, p + 1) @ beta
        w = kernel_1d(u / A[t], kernel) / A[t]
        cl = None if cluster is None else cluster[mask]
        Xd = poly_basis(u, None, p)
        f = fit_window(Xd, resid, w, cl, "distance pilot")
        # residuals of the bias polynomial replace the fit's own residuals
        m = f.mask
        f.scores = Xd[m] * (w[m] * resid[m])[:, None]
        e0 = np.zeros(p + 1)
        e0[0] = 1.0
        fits.append(f)
        infl.append(f.influence(e0, vce))
        # bias factor of the scaled order-p fit for u^(p+1)
        U = u[m] / A[t]
        ww = w[m]
        XU = poly_basis(U, None, p)
        F = float(np.linalg.solve((XU.T * ww) @ XU, (XU.T * ww) @ U ** (p + 1))[0])
        bias.append(F * beta[p + 1])
        reg.append(F * F * V[p + 1, p + 1])
    v0, v1, v01 = side_variances(fits, infl, vce, joint)
    return {
        "var": (A[0] ** 2 * v0, A[1] ** 2 * v1),
        "var01": A[0] * A[1] * v01,
        "bias": tuple(bias),
        "reg": tuple(reg),
        "n_pilot": (fits[0].n, fits[1].n),
        "pilot": c,
    }


def distance_bandwidths(
    y: np.ndarray,
    D: np.ndarray,
    eval_pts: Optional[np.ndarray],
    p: int,
    kink_unknown: Sequence[bool],
    kink_position: np.ndarray,
    kernel: str,
    bwselect: str,
    vce: str,
    bwcheck: Optional[int],
    masspoints: str,
    cluster: Optional[np.ndarray],
    scaleregul: float,
    cqt: float,
    fitmethod: str,
    fuzzy: Optional[np.ndarray] = None,
    bwparam: str = "main",
) -> Dict[str, Any]:
    """Bandwidths ``h0``, ``h1`` (and inference bandwidths) per point."""
    n, J = D.shape
    joint = fitmethod == "joint"
    two = bwselect.endswith("two")
    integrated = bwselect.startswith("i")
    consts, floors = [], []
    for j in range(J):
        Dj = D[:, j]
        fl = [0.0, 0.0]
        if bwcheck is not None:
            for t, mask in enumerate((Dj < 0, Dj >= 0)):
                # all observations count, also with masspoints='adjust'
                # (black box: R's bw.min is the k-th smallest |D|)
                fl[t] = _bwcheck_limits(np.abs(Dj[mask]), bwcheck)[0]
        floors.append(fl)
        yy = y
        if fuzzy is not None and bwparam == "main":
            c = _pilot(Dj, kernel, n)
            # the jumps use the bwcheck-floored pilot of each side
            ty, tt = _side_jumps(
                y, fuzzy, Dj, (max(c, fl[0]), max(c, fl[1])), p, kernel
            )
            if not np.isfinite(tt) or abs(tt) < FIRST_STAGE_TOL:
                raise DataInsufficient(
                    "rd2d: zero first-stage jump at the pilot bandwidth; the "
                    "fuzzy bandwidth is undefined.",
                    recovery_hint="Use bwparam='itt' or supply h=.",
                )
            yy = y / tt - ty / tt**2 * fuzzy
        consts.append(
            _point_constants(Dj, yy, kernel, p, cqt, vce, joint, n, cluster, fl)
        )

    def mse(var: Any, b2: Any) -> Any:
        return (2.0 * var / ((2 * p + 2) * b2)) ** (1.0 / (2 * p + 4))

    def pieces(c: Dict[str, Any], t: Optional[int]) -> Tuple[Any, Any]:
        if t is None:
            V = c["var"][0] + c["var"][1] - 2.0 * c["var01"]
            B = c["bias"][1] - c["bias"][0]
            R = c["reg"][0] + c["reg"][1]
        else:
            V, B, R = c["var"][t], c["bias"][t], c["reg"][t]
        return V, B * B + scaleregul * R

    H = np.empty((J, 2))
    if integrated:
        side_set: Tuple[Optional[int], ...] = (0, 1) if two else (None,)
        for side in side_set:
            h = mse(
                sum(pieces(c, side)[0] for c in consts),
                sum(pieces(c, side)[1] for c in consts),
            )
            if side is None:
                H[:, :] = h
            else:
                H[:, side] = h
    else:
        for j, cst in enumerate(consts):
            H[j] = (
                [mse(*pieces(cst, 0)), mse(*pieces(cst, 1))]
                if two
                else mse(*pieces(cst, None))
            )
    # bwcheck floor on the MSE bandwidth, then the CER rate, then the kink
    # rates (floor first: black box for the CER case of the location
    # selector; the kink rates are then exact).
    if bwcheck is not None:
        for j in range(J):
            H[j] = np.maximum(H[j], floors[j]) if two else max(H[j, 0], max(floors[j]))
    n_side = [int((D[:, 0] < 0).sum()), int((D[:, 0] >= 0).sum())]
    if bwselect.startswith(("cer", "icer")):
        if two:
            H[:, 0] *= cer_rate(n_side[0], p)
            H[:, 1] *= cer_rate(n_side[1], p)
        else:
            H *= cer_rate(n, p)
    smooth = 1.0 / (2 * p + 4)
    rate = (
        np.array([n_side[0] ** (smooth - 0.25), n_side[1] ** (smooth - 0.25)])
        if two
        else np.full(2, n ** (smooth - 0.25))
    )
    kink_unknown = tuple(bool(v) for v in kink_unknown)
    if kink_unknown[0]:
        H = H * rate
    elif np.asarray(kink_position, dtype=bool).any():
        dk = _kink_distance(
            np.asarray(eval_pts, dtype=float), np.asarray(kink_position, dtype=bool)
        )
        H = np.minimum(H, np.maximum(H * rate, dk[:, None]))
    H_rbc = H.copy()
    if kink_unknown[1]:
        rbc = (
            np.array([n_side[0] ** (0.25 - 1.0 / 3.0), n_side[1] ** (0.25 - 1.0 / 3.0)])
            if two
            else np.full(2, n ** (0.25 - 1.0 / 3.0))
        )
        H_rbc = H * rbc
    return {"h": H, "h_rbc": H_rbc, "constants": consts}


def distance_estimates(
    Y: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    H_rbc: np.ndarray,
    p: int,
    q: int,
    kernel: str,
    bwcheck: Optional[int],
    masspoints: str,
    cluster: Optional[np.ndarray],
    vce: str,
) -> Dict[str, Any]:
    """Side intercepts and their influences for orders ``p`` and ``q``."""
    Y = Y[:, None] if Y.ndim == 1 else Y
    n, J = D.shape
    out: Dict[str, Any] = {}
    for order, tag, HH in ((p, "p", H), (q, "q", H_rbc)):
        cols = Y.shape[1]
        est = np.empty((J, cols, 2))
        pts: List[List[Dict[str, Any]]] = [[] for _ in range(cols)]
        n_side = np.zeros((J, 2), dtype=int)
        pos_all = []
        for j in range(J):
            Dj = D[:, j]
            fits_c: List[List[Any]] = [[None, None] for _ in range(cols)]
            infl_c: List[List[Any]] = [[None, None] for _ in range(cols)]
            pos_j = []
            for t, mask in enumerate((Dj < 0, Dj >= 0)):
                pos = np.flatnonzero(mask)
                pos_j.append(pos)
                u = np.abs(Dj[mask])
                h = HH[j, t]
                if bwcheck is not None:
                    h = max(h, _bwcheck_limits(u, bwcheck)[0])
                w = kernel_1d(u / h, kernel) / h
                Xd = poly_basis(u, None, order)
                e0 = np.zeros(order + 1)
                e0[0] = 1.0
                cl = None if cluster is None else cluster[mask]
                for c in range(cols):
                    f = fit_window(Xd, Y[mask, c], w, cl, "distance")
                    fits_c[c][t] = f
                    infl_c[c][t] = f.influence(e0, vce)
                    est[j, c, t] = float(f.beta[0])
                n_side[j, t] = fits_c[0][t].n
            pos_all.append(pos_j)
            for c in range(cols):
                pts[c].append({"fits": fits_c[c], "infl": infl_c[c], "pos": pos_j})
        out[f"est_{tag}"] = est
        out[f"points_{tag}"] = pts
        out[f"n_{tag}"] = n_side
    return out


def distance_assemble(
    raw: Dict[str, Any],
    n: int,
    cluster: Optional[np.ndarray],
    vce: str,
    fitmethod: str,
) -> Dict[str, Any]:
    """Jumps, standard errors and covariances (sharp or fuzzy)."""
    joint = fitmethod == "joint"
    if cluster is not None:
        codes, inv = np.unique(cluster, return_inverse=True)
    res: Dict[str, Any] = {}
    for tag in ("p", "q"):
        est = raw[f"est_{tag}"]
        pts = raw[f"points_{tag}"]
        J = est.shape[0]
        fuzzy = est.shape[1] == 2
        jump, jump_se, side_se = {}, {}, {}
        for c in ([0, 1] if fuzzy else [0]):
            jump[c] = est[:, c, 1] - est[:, c, 0]
            se = np.empty((J, 2))
            jse = np.empty(J)
            for j in range(J):
                v0, v1, v01 = side_variances(
                    pts[c][j]["fits"], pts[c][j]["infl"], vce, joint
                )
                se[j] = (math.sqrt(v0), math.sqrt(v1))
                jse[j] = math.sqrt(max(v0 + v1 - 2.0 * v01, 0.0))
            side_se[c], jump_se[c] = se, jse
        if fuzzy:
            ty, tt = jump[0], jump[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                tau = np.where(np.abs(tt) < FIRST_STAGE_TOL, np.nan, ty / tt)
            main_se = np.empty(J)
            main_pts: List[Optional[Dict[str, Any]]] = []
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
                main_pts.append({"fits": fy["fits"], "infl": infl, "pos": fy["pos"]})
            res[f"tau_{tag}"], res[f"se_{tag}"] = tau, main_se
            res[f"itt_{tag}"], res[f"fs_{tag}"] = ty, tt
            res[f"se_itt_{tag}"], res[f"se_fs_{tag}"] = jump_se[0], jump_se[1]
        else:
            res[f"tau_{tag}"], res[f"se_{tag}"] = jump[0], jump_se[0]
            main_pts = pts[0]
            res[f"mu0_{tag}"], res[f"mu1_{tag}"] = est[:, 0, 0], est[:, 0, 1]
            res[f"se0_{tag}"], res[f"se1_{tag}"] = side_se[0][:, 0], side_se[0][:, 1]
        present = [pt for pt in main_pts if pt is not None]
        if len(present) < len(main_pts):
            res[f"cov_{tag}"] = np.full((J, J), np.nan)
        else:
            payload = []
            for pt in present:
                d = {
                    "fits": pt["fits"],
                    "infl": pt["infl"],
                    "pos0": pt["pos"][0],
                    "pos1": pt["pos"][1],
                    "n_total": n,
                    "cluster": cluster,
                }
                if cluster is not None:
                    d["codes"], d["n_codes"] = inv, len(codes)
                payload.append(d)
            res[f"cov_{tag}"] = point_covariance(payload, vce, joint)
    res["n_co"] = raw["n_p"][:, 0]
    res["n_tr"] = raw["n_p"][:, 1]
    return res
